/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/matrix_inverse.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void MatrixInverseFunctor<Context, T>::operator()(const Context& dev_ctx,
                                                  const DenseTensor& a,
                                                  DenseTensor* a_inv) {
#ifndef PADDLE_WITH_HIP
  const auto& mat_dims = a.dims();
  const int rank = mat_dims.size();
  int n = mat_dims[rank - 1];
  int batch_size = rank > 2 ? a.numel() / (n * n) : 1;

  phi::Allocator::AllocationPtr tmp_gpu_mat_data;
  const T* gpu_mat = a.data<T>();
  if (n >= 32) {
    // Copy all elements of input matrix A to a temporary memory space to
    // avoid being overridden by getrf.
    tmp_gpu_mat_data = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        a.numel() * sizeof(T),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    memory_utils::Copy(dev_ctx.GetPlace(),
                       tmp_gpu_mat_data->ptr(),
                       dev_ctx.GetPlace(),
                       a.data(),
                       a.numel() * sizeof(T),
                       dev_ctx.stream());
    gpu_mat = reinterpret_cast<const T*>(tmp_gpu_mat_data->ptr());
  }

  std::vector<const T*> cpu_ptrs(batch_size * 2);
  for (int i = 0; i < batch_size; ++i) {
    cpu_ptrs[i] = gpu_mat + i * n * n;
    cpu_ptrs[i + batch_size] = a_inv->data<T>() + i * n * n;
  }

  // Copy the addresses of A and A_inv from host to device,
  // and allocate device memory for info and pivots.
  int num_ints = n < 32 ? batch_size : batch_size * (n + 1);
  size_t total_bytes = cpu_ptrs.size() * sizeof(T*) + num_ints * sizeof(int);
  phi::Allocator::AllocationPtr tmp_gpu_ptrs_data = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      total_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  memory_utils::Copy(dev_ctx.GetPlace(),
                     tmp_gpu_ptrs_data->ptr(),
                     phi::CPUPlace(),
                     static_cast<void*>(cpu_ptrs.data()),
                     cpu_ptrs.size() * sizeof(T*),
                     dev_ctx.stream());
  T** gpu_inv_pivot_info = reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr());
  T** gpu_inv_ptrs = gpu_inv_pivot_info + batch_size;
  int* gpu_info_ptr =
      reinterpret_cast<int*>(gpu_inv_pivot_info + cpu_ptrs.size());

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  std::vector<int> info;  // only for singular checking
  info.resize(batch_size);
  // This functions in cuBLAS is intended to be used for matrices of small
  // sizes where the launch overhead is a significant factor.
  // TODO(Xreki): call function in cusolver for large matrices.
  if (n < 32) {
    // cublas<S/D>matinvBatched is a short cut of cublas<S/D>getrfBatched
    // plus cublas<S/D>getriBatched.
    // However it only works if N is less than 32. If not, we need to
    // go through cublas<S/D>getrfBatched and cublas<S/D>getriBatched.
    blas.BatchedMatInv(n,
                       reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                       gpu_inv_ptrs,
                       gpu_info_ptr,
                       batch_size);
  } else {
    // This function performs the LU factorization of each matrix A by the
    // equation P * A = L * U. L and U are written back to original matrix A,
    // and diagonal elements of L are discarded.
    int* gpu_pivot_ptr = gpu_info_ptr + batch_size;
    blas.BatchedGETRF(n,
                      reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()),
                      gpu_pivot_ptr,
                      gpu_info_ptr,
                      batch_size);

    blas.BatchedGETRI(n,
                      reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                      gpu_pivot_ptr,
                      gpu_inv_ptrs,
                      gpu_info_ptr,
                      batch_size);
  }
  memory_utils::Copy(phi::CPUPlace(),
                     info.data(),
                     dev_ctx.GetPlace(),
                     gpu_info_ptr,
                     sizeof(int) * batch_size,
                     dev_ctx.stream());
  for (int i = 0; i < batch_size; ++i) {
    PADDLE_ENFORCE_EQ(info[i],
                      0,
                      common::errors::PreconditionNotMet(
                          "For batch [%d]: U(%d, %d) is zero, singular U. "
                          "Please check the matrix value and change it to a "
                          "non-singular matrix",
                          i,
                          info[i],
                          info[i]));
  }
#else
  ComputeInverseEigen<Context, T>(dev_ctx, a, a_inv);
#endif
}

template class MatrixInverseFunctor<GPUContext, float>;
template class MatrixInverseFunctor<GPUContext, double>;
template class MatrixInverseFunctor<GPUContext, phi::dtype::complex<float>>;
template class MatrixInverseFunctor<GPUContext, phi::dtype::complex<double>>;

}  // namespace funcs
}  // namespace phi
