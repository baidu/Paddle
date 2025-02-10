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

#include "paddle/phi/kernels/determinant_kernel.h"

#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <vector>
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"

#include "glog/logging.h"
#include "paddle/phi/common/amp_type_traits.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace detail {
template <typename T>
class EigenMatrix {};

template <>
class EigenMatrix<phi::dtype::float16> {
 public:
  using MatrixType =
      Eigen::Matrix<phi::dtype::float16, Eigen::Dynamic, Eigen::Dynamic>;
};

template <>
class EigenMatrix<float> {
 public:
  using MatrixType = Eigen::MatrixXf;
};

template <>
class EigenMatrix<double> {
 public:
  using MatrixType = Eigen::MatrixXd;
};

inline int64_t GetBatchCount(const DDim dims) {
  int64_t batch_count = 1;
  auto dim_size = dims.size();
  PADDLE_ENFORCE_GE(
      dim_size,
      2,
      common::errors::InvalidArgument(
          "the input matrix dimension size should greater than 2."));

  // Cumulative multiplying each dimension until the last 2 to get the batch
  // count,
  // for example a tensor with shape [3,3,3,3], the batch count of matrices is
  // 9.
  for (int64_t i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }

  return batch_count;
}
}  // namespace detail

template <typename T, typename Context>
struct DeterminantCudaFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  int64_t rank,
                  int64_t batch_count,
                  DenseTensor* output) {
    std::vector<T> input_vec;
    std::vector<T> output_vec;
    phi::TensorToVector(input, dev_ctx, &input_vec);
    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
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
      output_vec.push_back(
          static_cast<T>(matrix.template cast<MPType>().determinant()));
    }
    phi::TensorFromVector(output_vec, dev_ctx, output);
  }
};

template <typename T>
__global__ void GetDetFromLUComplex(const T* lu_data,
                                    const int* ipiv,
                                    int64_t n,
                                    int64_t batch_size,
                                    T* out_data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    int offset_lu = idx * n * n;
    int offset_ipiv = idx * n;
    T out_idx = T(1.0, 0.0);
    T negative = T(-1.0, 0.0);
    for (int i = 0; i < n; ++i) {
      out_idx *= lu_data[offset_lu + i * n + i];
      if (ipiv[offset_ipiv + i] != i + 1) {
        out_idx *= negative;
      }
    }
    out_data[idx] = out_idx;
  }
}

template <typename T, typename Context>
struct DeterminantCudaFunctor<phi::dtype::complex<T>, Context> {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& a,
                  int64_t n,
                  int64_t batch_size,
                  DenseTensor* output) {
#ifndef PADDLE_WITH_HIP
    phi::Allocator::AllocationPtr tmp_gpu_mat_data;
    const phi::dtype::complex<T>* gpu_mat = a.data<phi::dtype::complex<T>>();
    // Copy all elements of input matrix A to a temporary memory space to
    // avoid being overridden by getrf.
    tmp_gpu_mat_data = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        a.numel() * sizeof(phi::dtype::complex<T>),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    memory_utils::Copy(dev_ctx.GetPlace(),
                       tmp_gpu_mat_data->ptr(),
                       dev_ctx.GetPlace(),
                       a.data(),
                       a.numel() * sizeof(phi::dtype::complex<T>),
                       dev_ctx.stream());
    gpu_mat = reinterpret_cast<const phi::dtype::complex<T>*>(
        tmp_gpu_mat_data->ptr());

    std::vector<const phi::dtype::complex<T>*> cpu_ptrs(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      cpu_ptrs[i] = gpu_mat + i * n * n;
    }

    int num_ints = batch_size * (n + 1);
    // num_ints is for pivot (n * batch_size) and info (batch_size)
    size_t total_bytes =
        batch_size * sizeof(phi::dtype::complex<T>*) + num_ints * sizeof(int);
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
    int* pivot_data = gpu_info_ptr + batch_size;

    auto blas = phi::funcs::GetBlas<Context, phi::dtype::complex<T>>(dev_ctx);
    // This function performs the LU factorization of each matrix A by the
    // equation P * A = L * U. L and U are written back to original matrix A,
    // and diagonal elements of L are discarded.
    blas.BatchedGETRF(n, gpu_mat_ptr, pivot_data, gpu_info_ptr, batch_size);
    phi::dtype::complex<T>* out_data =
        dev_ctx.template Alloc<phi::dtype::complex<T>>(output);
    int block_size = std::min(256, dev_ctx.GetMaxThreadsPerBlock());
    dim3 dim_block(block_size);
    dim3 num_blocks((batch_size + block_size - 1) / block_size);
    GetDetFromLUComplex<phi::dtype::complex<T>><<<num_blocks, dim_block>>>(
        gpu_mat, pivot_data, n, batch_size, out_data);
#else
    using MatrixType =
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>;
    std::vector<phi::dtype::complex<T>> input_vec;
    std::vector<phi::dtype::complex<T>> output_vec;
    phi::TensorToVector(a, dev_ctx, &input_vec);
    for (int64_t i = 0; i < batch_size; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * n * n;
      auto end_iter = input_vec.begin() + (i + 1) * n * n;
      std::vector<phi::dtype::complex<T>> sub_vec(
          begin_iter,
          end_iter);  // get every square matrix data
      MatrixType matrix(n, n);
      for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          matrix(i, j) = static_cast<std::complex<T>>(sub_vec[n * i + j]);
        }
      }
      output_vec.push_back(
          static_cast<phi::dtype::complex<T>>(matrix.determinant()));
    }
    phi::TensorFromVector(output_vec, dev_ctx, output);
#endif
  }
};

template <typename T, typename Context>
void DeterminantKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  auto input_dim = common::vectorize(x.dims());
  auto input_dim_size = input_dim.size();

  auto batch_count = detail::GetBatchCount(x.dims());
  VLOG(10) << "input dim:" << x.dims();
  PADDLE_ENFORCE_GE(
      input_dim_size,
      2,
      common::errors::InvalidArgument("the input matrix dimension size should "
                                      "greater than or equal to 2."));
  PADDLE_ENFORCE_EQ(input_dim[input_dim_size - 1],
                    input_dim[input_dim_size - 2],
                    common::errors::InvalidArgument(
                        "the input matrix should be square matrix."));
  auto rank = input_dim[input_dim_size - 1];  // square matrix length
  DeterminantCudaFunctor<T, Context>()(dev_ctx, x, rank, batch_count, out);
  auto output_dims = common::slice_ddim(x.dims(), 0, input_dim_size - 2);
  if (input_dim_size > 2) {
    out->Resize(output_dims);
  } else {
    // when input is a two-dimension matrix, The det value is a number.
    out->Resize(common::make_ddim({}));
  }
  VLOG(10) << "output dim:" << out->dims();
}

}  // namespace phi

PD_REGISTER_KERNEL(determinant,
                   GPU,
                   ALL_LAYOUT,
                   phi::DeterminantKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
