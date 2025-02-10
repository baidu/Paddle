// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/determinant_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/impl/determinant_kernel_impl.h"
#include "paddle/phi/kernels/slogdeterminant_kernel.h"

namespace phi {

// T is not complex
template <typename T>
T sign(T val) {
  return static_cast<T>(T(0) < val) - (val < T(0));
}

// T is complex
template <typename T>
T sign(T det, T modulus) {
  return det / modulus;
}

template <typename T, typename Context>
struct SlogDeterminantFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  int64_t rank,
                  int64_t batch_count,
                  DenseTensor* output) {
    std::vector<T> input_vec;
    std::vector<T> sign_vec;
    std::vector<T> log_vec;
    std::vector<T> output_vec;
    phi::TensorToVector(input, dev_ctx, &input_vec);
    for (int64_t i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_iter,
                             end_iter);  // get every square matrix data
      typename detail::EigenMatrix<T>::MatrixType matrix(rank, rank);
      for (int64_t i = 0; i < rank; ++i) {
        for (int64_t j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      auto det_val = matrix.determinant();
      sign_vec.push_back(sign(det_val));
      det_val >= 0
          ? log_vec.push_back(std::log(det_val))
          : log_vec.push_back(std::log(std::abs(
                det_val)));  // for computing log value of a negative value.
    }
    // merge sign_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sign_vec.begin(), sign_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    phi::TensorFromVector(output_vec, dev_ctx, output);
  }
};

template <typename T>
__global__ void GetSlogDetFromLUComplex(const T* lu_data,
                                        const int* ipiv,
                                        int64_t n,
                                        int64_t batch_size,
                                        T* out_data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    int offset_lu = idx * n * n;
    int offset_ipiv = idx * n;
    T det_val = T(1.0, 0.0);
    T negative = T(-1.0, 0.0);
    for (int i = 0; i < n; ++i) {
      det_val *= lu_data[offset_lu + i * n + i];
      if (ipiv[offset_ipiv + i] != i + 1) {
        det_val *= negative;
      }
    }
    T abs_det = static_cast<T>(abs(det_val));
    T sign = det_val / abs_det;
    T log_abs_det = log(abs_det);
    out_data[idx] = sign;
    out_data[idx + batch_size] = log_abs_det;
  }
}

template <typename T, typename Context>
struct SlogDeterminantFunctor<phi::dtype::complex<T>, Context> {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  int64_t rank,
                  int64_t batch_count,
                  DenseTensor* output) {
#ifndef PADDLE_WITH_HIP
    phi::Allocator::AllocationPtr tmp_gpu_mat_data;
    const phi::dtype::complex<T>* gpu_mat =
        input.data<phi::dtype::complex<T>>();
    // Copy all elements of input matrix A to a temporary memory space to
    // avoid being overridden by getrf.
    tmp_gpu_mat_data = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        input.numel() * sizeof(phi::dtype::complex<T>),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    memory_utils::Copy(dev_ctx.GetPlace(),
                       tmp_gpu_mat_data->ptr(),
                       dev_ctx.GetPlace(),
                       input.data(),
                       input.numel() * sizeof(phi::dtype::complex<T>),
                       dev_ctx.stream());
    gpu_mat = reinterpret_cast<const phi::dtype::complex<T>*>(
        tmp_gpu_mat_data->ptr());

    std::vector<const phi::dtype::complex<T>*> cpu_ptrs(batch_count);
    for (int i = 0; i < batch_count; ++i) {
      cpu_ptrs[i] = gpu_mat + i * rank * rank;
    }

    // num_ints is for pivot (rank * batch_count) and info (batch_count)
    int num_ints = batch_count * (rank + 1);
    size_t total_bytes =
        batch_count * sizeof(phi::dtype::complex<T>*) + num_ints * sizeof(int);
    phi::Allocator::AllocationPtr tmp_gpu_ptrs_data = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        total_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    memory_utils::Copy(dev_ctx.GetPlace(),
                       tmp_gpu_ptrs_data->ptr(),
                       phi::CPUPlace(),
                       static_cast<void*>(cpu_ptrs.data()),
                       cpu_ptrs.size() * sizeof(phi::dtype::complex<T>*),
                       dev_ctx.stream());

    phi::dtype::complex<T>** gpu_mat_ptr =
        reinterpret_cast<phi::dtype::complex<T>**>(tmp_gpu_ptrs_data->ptr());
    int* gpu_info_ptr = reinterpret_cast<int*>(gpu_mat_ptr + cpu_ptrs.size());
    int* pivot_data = gpu_info_ptr + batch_count;

    auto blas = phi::funcs::GetBlas<Context, phi::dtype::complex<T>>(dev_ctx);
    // This function performs the LU factorization of each matrix A by the
    // equation P * A = L * U. L and U are written back to original matrix A,
    // and diagonal elements of L are discarded.
    blas.BatchedGETRF(rank, gpu_mat_ptr, pivot_data, gpu_info_ptr, batch_count);
    phi::dtype::complex<T>* out_data =
        dev_ctx.template Alloc<phi::dtype::complex<T>>(output);
    int block_size = std::min(256, dev_ctx.GetMaxThreadsPerBlock());
    dim3 dim_block(block_size);
    dim3 num_blocks((batch_count + block_size - 1) / block_size);
    GetSlogDetFromLUComplex<phi::dtype::complex<T>><<<num_blocks, dim_block>>>(
        gpu_mat, pivot_data, rank, batch_count, out_data);
#else
    using MatrixType =
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>;
    std::vector<phi::dtype::complex<T>> input_vec;
    std::vector<phi::dtype::complex<T>> sign_vec;
    std::vector<phi::dtype::complex<T>> log_vec;
    std::vector<phi::dtype::complex<T>> output_vec;
    phi::TensorToVector(input, dev_ctx, &input_vec);
    for (int64_t i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<phi::dtype::complex<T>> sub_vec(
          begin_iter,
          end_iter);  // get every square matrix data
      MatrixType matrix(rank, rank);
      for (int64_t i = 0; i < rank; ++i) {
        for (int64_t j = 0; j < rank; ++j) {
          matrix(i, j) = static_cast<std::complex<T>>(sub_vec[rank * i + j]);
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      std::complex<T> det_val = matrix.determinant();
      T abs_det_val = std::abs(det_val);
      sign_vec.push_back(static_cast<phi::dtype::complex<T>>(
          sign(det_val, static_cast<std::complex<T>>(abs_det_val))));
      log_vec.push_back(
          static_cast<phi::dtype::complex<T>>(std::log(abs_det_val)));
    }
    // merge sign_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sign_vec.begin(), sign_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    phi::TensorFromVector(output_vec, dev_ctx, output);
#endif
  }
};

template <typename T, typename Context>
void SlogDeterminantKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           DenseTensor* out) {
  auto input_dim = common::vectorize(x.dims());
  auto input_dim_size = input_dim.size();

  auto batch_count = detail::GetBatchCount(x.dims());
  VLOG(2) << "input dim:" << x.dims();
  PADDLE_ENFORCE_GE(
      input_dim_size,
      2,
      errors::InvalidArgument(
          "the input matrix dimension size should greater than 2."));
  PADDLE_ENFORCE_EQ(
      input_dim[input_dim_size - 1],
      input_dim[input_dim_size - 2],
      errors::InvalidArgument("the input matrix should be square matrix."));
  auto rank = input_dim[input_dim_size - 1];  // square matrix length
  SlogDeterminantFunctor<T, Context>()(dev_ctx, x, rank, batch_count, out);
  std::vector<int> output_dim_vec(input_dim.begin(), input_dim.end() - 2);
  if (input_dim.size() == static_cast<size_t>(2)) {
    // when input is a two-dimension matrix, The det value is a number.
    output_dim_vec = {};
  }
  output_dim_vec.insert(output_dim_vec.begin(),
                        2);  // make the output dims as same as numpy
  auto output_dims = common::make_ddim(output_dim_vec);
  out->Resize(output_dims);
  VLOG(2) << "output dim:" << out->dims();
}

}  // namespace phi

PD_REGISTER_KERNEL(slogdet,
                   GPU,
                   ALL_LAYOUT,
                   phi::SlogDeterminantKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
