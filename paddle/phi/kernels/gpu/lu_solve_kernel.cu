// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"

namespace phi {

template <typename T>
void cusolver_getrs(const cusolverDnHandle_t& cusolverH,
                    cublasOperation_t trans,
                    int n,
                    int nrhs,
                    T *a,
                    int lda,
                    int *ipiv,
                    T *b,
                    int ldb,
                    int *info);

template <>
void cusolver_getrs<float>(const cusolverDnHandle_t& cusolverH,
                           cublasOperation_t trans,
                           int n,
                           int nrhs,
                           float *a,
                           int lda,
                           int *ipiv,
                           float *b,
                           int ldb,
                           int *info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgetrs(
      cusolverH, trans, n, nrhs, a, lda, ipiv, b, ldb, info));
}

template <>
void cusolver_getrs<double>(const cusolverDnHandle_t& cusolverH,
                            cublasOperation_t trans,
                            int n,
                            int nrhs,
                            double *a,
                            int lda,
                            int *ipiv,
                            double *b,
                            int ldb,
                            int *info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgetrs(
      cusolverH, trans, n, nrhs, a, lda, ipiv, b, ldb, info));
}

template <typename T, typename Context>
void lu_solve_kernel(const Context& dev_ctx,
                     cublasOperation_t trans,
                     int n,
                     int nrhs,
                     T *a,
                     int lda,
                     int *ipiv,
                     T *b,
                     int ldb,
                     int *info) {
  /* step 1: get cusolver handle*/
  auto cusolverH = dev_ctx.cusolver_dn_handle();
  /* step 2: LU_SOLVE factorization */
  cusolver_getrs(cusolverH, trans, n, nrhs, a, lda, ipiv, b, ldb, info);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
}


template <typename T, typename Context>
void LuSolveKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const std::string& trans,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  // Copy x to out since cusolverDn*getrs overwrites the input
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  // Validate input dimensions
  auto x_dims = x.dims();
  auto lu_dims = lu.dims();

  cublasOperation_t trans_op;
  if (trans == "N") {
    trans_op = CUBLAS_OP_N;
  } else if (trans == "T") {
    trans_op = CUBLAS_OP_T;
  } else if (trans == "C") {
    trans_op = CUBLAS_OP_C;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "trans must be one of ['N', 'T', 'C'], but got %s", trans));
  }
  int n = static_cast<int>(lu_dims[lu_dims.size() - 1]);
  int nrhs = static_cast<int>(x_dims[x_dims.size() - 1]);
  int lda = std::max(1, n);
  int ldb = std::max(1, n);

  DenseTensor info_tensor;
  info_tensor.Resize({1});
  dev_ctx.template Alloc<int>(&info_tensor);
  int* d_info = info_tensor.data<int>();

  auto outdims = out->dims();
  auto outrank = outdims.size();
  int batchsize = product(common::slice_ddim(outdims, 0, outrank - 2));
  auto out_data = out->data<T>();
  auto lu_data = reinterpret_cast<T*>(const_cast<T*>(lu.data<T>()));
  auto pivots_data = reinterpret_cast<int*>(const_cast<int*>(pivots.data<int>()));
  for (int i = 0; i < batchsize; i++) {
    auto* out_data_item = &out_data[i * n * n];
    auto* lu_data_item = &lu_data[i * n * n];
    auto* pivots_data_item = &pivots_data[i * n];
    lu_solve_kernel(dev_ctx,
                      trans_op,
                      n,
                      nrhs,
                      lu_data_item,
                      lda,
                      pivots_data_item,
                      out_data_item,
                      ldb,
                      d_info);
  }
  // Synchronize to ensure the solve is complete
  dev_ctx.Wait();
}

}  // namespace phi

PD_REGISTER_KERNEL(
    lu_solve, GPU, ALL_LAYOUT, phi::LuSolveKernel, float, double) {}

#endif  // not PADDLE_WITH_HIP