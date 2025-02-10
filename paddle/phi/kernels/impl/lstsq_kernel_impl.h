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

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/optional.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/impl/activation_impl.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cusolver.h"
#endif

#if defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/dynload/rocsolver.h"
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

namespace phi {

inline int GetBatchCount(const DDim& dims) {
  int count = 1;
  int num_dims = dims.size();
  for (int i = 0; i < num_dims - 2; ++i) {
    count *= dims[i];
  }
  return count;
}

inline int GetMatrixStride(const DDim& dims) {
  int num_dims = dims.size();
  return dims[num_dims - 1] * dims[num_dims - 2];
}

inline bool IsComplexDtype(const DataType& type) {
  return (type == DataType::COMPLEX64 || type == DataType::COMPLEX128);
}

template <typename DeviceContext, typename T>
inline void GetResidualsTensor(const DeviceContext& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               DenseTensor* solution,
                               DenseTensor* residuals) {
  auto x_dims = x.dims();
  int dim_size = x_dims.size();
  int m = x_dims[dim_size - 2];
  int n = x_dims[dim_size - 1];

  if (m > n) {
    DenseTensor matmul_tensor =
        phi::Matmul<T>(dev_ctx, x, *solution, false, false);
    DenseTensor sub_tensor = phi::Subtract<T>(dev_ctx, matmul_tensor, y);
    DenseTensor* pow_tensor = new DenseTensor();
    pow_tensor->Resize(sub_tensor.dims());
    dev_ctx.template Alloc<T>(pow_tensor);
    phi::PowKernel<T>(dev_ctx, sub_tensor, Scalar(2), pow_tensor);

    auto sum_tensor = phi::Sum<T>(
        dev_ctx, *pow_tensor, phi::IntArray({-2}), pow_tensor->dtype(), false);
    phi::Copy<DeviceContext>(
        dev_ctx, sum_tensor, dev_ctx.GetPlace(), true, residuals);
  } else {
    IntArray empty_shape({0});
    DenseTensor empty_tensor =
        phi::Empty<T, DeviceContext>(dev_ctx, empty_shape);
    phi::Copy<DeviceContext>(
        dev_ctx, empty_tensor, dev_ctx.GetPlace(), true, residuals);
  }
}

#ifdef PADDLE_WITH_HIP
template <typename DeviceContext, typename T>
inline void BatchedOrmqr(const DeviceContext& dev_ctx,
                         bool left,
                         bool transpose,
                         int batch_size,
                         int m,
                         int n,
                         int k,
                         T* a,
                         int a_stride,
                         T* tau,
                         int tau_stride,
                         T* other,
                         int other_stride);

#define FUNC_WITH_TYPES(m) m(float, s) m(double, d)
#define ORMQR_BATCH_INSTANCE(T, C)                                        \
  template <>                                                             \
  inline void BatchedOrmqr<GPUContext, T>(const GPUContext& dev_ctx,      \
                                          bool left,                      \
                                          bool transpose,                 \
                                          int batch_size,                 \
                                          int m,                          \
                                          int n,                          \
                                          int k,                          \
                                          T* a,                           \
                                          int a_stride,                   \
                                          T* tau,                         \
                                          int tau_stride,                 \
                                          T* other,                       \
                                          int other_stride) {             \
    auto side = left ? rocblas_side_left : rocblas_side_right;            \
    auto trans =                                                          \
        transpose ? rocblas_operation_transpose : rocblas_operation_none; \
    int lda = std::max<int>(1, left ? m : n);                             \
    int ldc = std::max<int>(1, m);                                        \
    auto handle = dev_ctx.cusolver_dn_handle();                           \
    for (int i = 0; i < batch_size; ++i) {                                \
      T* a_working_ptr = &a[i * a_stride];                                \
      T* tau_working_ptr = &tau[i * tau_stride];                          \
      T* other_working_ptr = &other[i * other_stride];                    \
      PADDLE_ENFORCE_GPU_SUCCESS(                                         \
          phi::dynload::rocsolver_##C##ormqr(handle,                      \
                                             side,                        \
                                             trans,                       \
                                             m,                           \
                                             n,                           \
                                             k,                           \
                                             a_working_ptr,               \
                                             lda,                         \
                                             tau_working_ptr,             \
                                             other_working_ptr,           \
                                             ldc));                       \
    }                                                                     \
  }
FUNC_WITH_TYPES(ORMQR_BATCH_INSTANCE);
#endif
#if defined(PADDLE_WITH_CUDA)
template <typename DeviceContext, typename T>
inline void BatchedOrmqr(const DeviceContext& dev_ctx,
                         bool left,
                         bool transpose,
                         int batch_size,
                         int m,
                         int n,
                         int k,
                         T* a,
                         int a_stride,
                         T* tau,
                         int tau_stride,
                         T* other,
                         int other_stride);

template <>
inline void BatchedOrmqr<GPUContext, float>(const GPUContext& dev_ctx,
                                            bool left,
                                            bool transpose,
                                            int batch_size,
                                            int m,
                                            int n,
                                            int k,
                                            float* a,
                                            int a_stride,
                                            float* tau,
                                            int tau_stride,
                                            float* other,
                                            int other_stride) {
  int lwork = 0;
  auto side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  auto trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = std::max<int>(1, left ? m : n);
  int ldc = std::max<int>(1, m);

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSormqr_bufferSize(
      handle, side, trans, m, n, k, a, lda, tau, other, ldc, &lwork));
  DenseTensor* info = new DenseTensor();
  info->Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    float* other_working_ptr = &other[i * other_stride];

    handle = dev_ctx.cusolver_dn_handle();
    DenseTensor* workspace = new DenseTensor();
    workspace->Resize(common::make_ddim({lwork}));
    float* workspace_ptr = dev_ctx.template Alloc<float>(workspace);

    // compute ormgr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSormqr(handle,
                                                              side,
                                                              trans,
                                                              m,
                                                              n,
                                                              k,
                                                              a_working_ptr,
                                                              lda,
                                                              tau_working_ptr,
                                                              other_working_ptr,
                                                              ldc,
                                                              workspace_ptr,
                                                              lwork,
                                                              info_d));

    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver info is not zero but [%d]", i, info_h));
  }
}

template <>
inline void BatchedOrmqr<GPUContext, double>(const GPUContext& dev_ctx,
                                             bool left,
                                             bool transpose,
                                             int batch_size,
                                             int m,
                                             int n,
                                             int k,
                                             double* a,
                                             int a_stride,
                                             double* tau,
                                             int tau_stride,
                                             double* other,
                                             int other_stride) {
  int lwork = 0;
  auto side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  auto trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = std::max<int>(1, left ? m : n);
  int ldc = std::max<int>(1, m);

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDormqr_bufferSize(
      handle, side, trans, m, n, k, a, lda, tau, other, ldc, &lwork));
  DenseTensor* info = new DenseTensor();
  info->Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    double* other_working_ptr = &other[i * other_stride];

    handle = dev_ctx.cusolver_dn_handle();
    DenseTensor* workspace = new DenseTensor();
    workspace->Resize(common::make_ddim({lwork}));
    double* workspace_ptr = dev_ctx.template Alloc<double>(workspace);

    // compute ormgr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDormqr(handle,
                                                              side,
                                                              trans,
                                                              m,
                                                              n,
                                                              k,
                                                              a_working_ptr,
                                                              lda,
                                                              tau_working_ptr,
                                                              other_working_ptr,
                                                              ldc,
                                                              workspace_ptr,
                                                              lwork,
                                                              info_d));

    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver info is not zero but [%d]", i, info_h));
  }
}
#endif

}  // namespace phi
