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
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cusolver.h"
#endif  // PADDLE_WITH_CUDA
#ifdef PADDLE_WITH_HIP
#include <thrust/device_vector.h>
#include "paddle/phi/backends/dynload/rocsolver.h"
#endif  // PADDLE_WITH_HIP
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/common/errors.h"
#endif
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace funcs {

inline int64_t GetBatchSize(const phi::DDim &dims) {
  int64_t batch_size = 1;
  auto dim_size = dims.size();
  for (int i = 0; i < dim_size - 2; ++i) {
    batch_size *= dims[i];
  }
  return batch_size;
}

static void CheckEighResult(const int batch, const int info) {
  PADDLE_ENFORCE_LE(
      info,
      0,
      common::errors::PreconditionNotMet(
          "For batch [%d]: the [%d] off-diagonal elements of an intermediate"
          "tridiagonal form did not converge to zero",
          batch,
          info));
  PADDLE_ENFORCE_GE(
      info,
      0,
      common::errors::PreconditionNotMet(
          "For batch [%d]: the [%d] argument had an illegal value",
          batch,
          info));
}

#ifdef PADDLE_WITH_CUDA

#if CUDA_VERSION >= 11031
static bool use_cusolver_syevj_batched = true;
#else
static bool use_cusolver_syevj_batched = false;
#endif

#define CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(scalar_t, value_t)     \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, const scalar_t *A, int lda, const value_t *W, int *lwork,      \
      syevjInfo_t params, int batchsize

template <class scalar_t, class value_t = scalar_t>
void syevjBatched_bufferSize(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(scalar_t, value_t)) {
  PADDLE_THROW(common::errors::InvalidArgument(
      "syevjBatched_bufferSize: not implemented for %s",
      typeid(scalar_t).name()));
}

template <>
inline void syevjBatched_bufferSize<float>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(float, float)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevjBatched_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
}

template <>
inline void syevjBatched_bufferSize<double>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(double, double)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDsyevjBatched_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
}

template <>
inline void syevjBatched_bufferSize<phi::dtype::complex<float>, float>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(phi::dtype::complex<float>,
                                                 float)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCheevjBatched_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuComplex *>(A),
      lda,
      W,
      lwork,
      params,
      batchsize));
}

template <>
inline void syevjBatched_bufferSize<phi::dtype::complex<double>, double>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(phi::dtype::complex<double>,
                                                 double)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZheevjBatched_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuDoubleComplex *>(A),
      lda,
      W,
      lwork,
      params,
      batchsize));
}

#define CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(scalar_t, value_t)                \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, scalar_t *A, int lda, value_t *W, scalar_t *work, int lwork,   \
      int *info, syevjInfo_t params, int batchsize

template <class scalar_t, class value_t = scalar_t>
void syevjBatched(CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(scalar_t, value_t)) {
  PADDLE_THROW(common::errors::InvalidArgument(
      "syevjBatched: not implemented for %s", typeid(scalar_t).name()));
}

template <>
inline void syevjBatched<float>(CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(float,
                                                                  float)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevjBatched(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
}

template <>
inline void syevjBatched<double>(CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(double,
                                                                   double)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDsyevjBatched(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
}

template <>
inline void syevjBatched<phi::dtype::complex<float>, float>(
    CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(phi::dtype::complex<float>, float)) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCheevjBatched(handle,
                                       jobz,
                                       uplo,
                                       n,
                                       reinterpret_cast<cuComplex *>(A),
                                       lda,
                                       W,
                                       reinterpret_cast<cuComplex *>(work),
                                       lwork,
                                       info,
                                       params,
                                       batchsize));
}

template <>
inline void syevjBatched<phi::dtype::complex<double>, double>(
    CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(phi::dtype::complex<double>, double)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZheevjBatched(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex *>(A),
      lda,
      W,
      reinterpret_cast<cuDoubleComplex *>(work),
      lwork,
      info,
      params,
      batchsize));
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
static void CheckEighResult(const GPUContext &dev_ctx,
                            const int64_t batch_size,
                            int *info) {
  std::vector<int> error_info(batch_size);
  memory_utils::Copy(phi::CPUPlace(),
                     error_info.data(),
                     dev_ctx.GetPlace(),
                     info,
                     sizeof(int) * batch_size,
                     dev_ctx.stream());
  dev_ctx.Wait();
  for (auto i = 0; i < batch_size; ++i) {
    CheckEighResult(i, error_info[i]);
  }
}
#endif

template <typename DeviceContext, typename T>
struct MatrixEighFunctor {
  void operator()(const DeviceContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors);
};

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices, and uses the variable has_vectors to
// control whether to return the eigenvectors.
template <typename T>
struct MatrixEighFunctor<CPUContext, T> {
 public:
  void operator()(const CPUContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;
    ValueType *out_value = dev_ctx.template Alloc<ValueType>(eigen_values);

    DenseTensor input_trans;
    // lapack is a column-major storage, transpose make the input to
    // have a continuous memory layout
    input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input);
    T *input_vector = input_trans.data<T>();

    auto dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);

    int vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    int values_stride = dims[dim_size - 1];
    char uplo = is_lower ? 'L' : 'U';
    char jobz = has_vectors ? 'V' : 'N';
    int n = dims[dim_size - 1];
    int64_t lda = std::max<int64_t>(1, n);
    // if work = -1, it means that you need to use the lapack function to
    // query
    // the optimal value
    int lwork = -1;      // The length of the array work
    int lrwork = -1;     // The dimension of the array rwork,rwork is REAL array
    int liwork = -1;     // The dimension of the array iwork
    int iwork_opt = -1;  // The optimal length of the array liwork
    T lwork_opt = static_cast<T>(-1);  // The optimal length of the array work
    ValueType rwork_opt =
        static_cast<ValueType>(-1);  // The optimal length of the array rwork

    int info = 0;
    // Call lapackEigh to get the optimal size of work data
    phi::funcs::lapackEigh<T, ValueType>(jobz,
                                         uplo,
                                         n,
                                         input_vector,
                                         lda,
                                         out_value,
                                         &lwork_opt,
                                         lwork,
                                         &rwork_opt,
                                         lrwork,
                                         &iwork_opt,
                                         liwork,
                                         &info);
    lwork = std::max<int>(1, static_cast<int>(lwork_opt));
    liwork = std::max<int>(1, iwork_opt);

    DenseTensor rwork_tensor;
    ValueType *rwork_data = nullptr;

    // complex type
    if (input.type() == phi::DataType::COMPLEX64 ||
        input.type() == phi::DataType::COMPLEX128) {
      lrwork = std::max<int>(1, static_cast<int>(rwork_opt));

      rwork_tensor.Resize(common::make_ddim({lrwork}));
      rwork_data = dev_ctx.template Alloc<ValueType>(&rwork_tensor);
    }

    DenseTensor iwork_tensor, work_tensor;

    iwork_tensor.Resize(common::make_ddim({liwork}));
    int *iwork_data = dev_ctx.template Alloc<int>(&iwork_tensor);

    work_tensor.Resize(common::make_ddim({lwork}));
    T *work_data = dev_ctx.template Alloc<T>(&work_tensor);

    for (auto i = 0; i < batch_size; i++) {
      auto *value_data = out_value + i * values_stride;
      auto *input_data = input_vector + i * vector_stride;
      phi::funcs::lapackEigh<T, ValueType>(jobz,
                                           uplo,
                                           n,
                                           input_data,
                                           lda,
                                           value_data,
                                           work_data,
                                           lwork,
                                           rwork_data,
                                           lrwork,
                                           iwork_data,
                                           liwork,
                                           &info);
      CheckEighResult(i, info);
    }
    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              common::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated, "
                                  "so the eigenvectors must be provided."));
      input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }
};

#ifdef PADDLE_WITH_HIP
#define ROCSOLVER_SYEVJ_BATCHED_ARGTYPES(scalar_t, value_t)            \
  solverHandle_t handle, rocblas_esort esort, rocblas_evect evect,     \
      rocblas_fill uplo, int n, scalar_t *const A[], int lda,          \
      const scalar_t abstol, scalar_t *residual, const int max_sweeps, \
      int *n_sweeps, value_t *W, const int strideW, int *info,         \
      const int batch_count

template <class scalar_t, class value_t = scalar_t>
void syevjBatched(ROCSOLVER_SYEVJ_BATCHED_ARGTYPES(scalar_t, value_t)) {
  PADDLE_THROW(common::errors::InvalidArgument(
      "syevjBatched: not implemented for %s", typeid(scalar_t).name()));
}

template <>
inline void syevjBatched<float>(ROCSOLVER_SYEVJ_BATCHED_ARGTYPES(float,
                                                                 float)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::rocsolver_ssyevj_batched(handle,
                                                               esort,
                                                               evect,
                                                               uplo,
                                                               n,
                                                               A,
                                                               lda,
                                                               abstol,
                                                               residual,
                                                               max_sweeps,
                                                               n_sweeps,
                                                               W,
                                                               strideW,
                                                               info,
                                                               batch_count));
}

template <>
inline void syevjBatched<double>(ROCSOLVER_SYEVJ_BATCHED_ARGTYPES(double,
                                                                  double)) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::rocsolver_dsyevj_batched(handle,
                                                               esort,
                                                               evect,
                                                               uplo,
                                                               n,
                                                               A,
                                                               lda,
                                                               abstol,
                                                               residual,
                                                               max_sweeps,
                                                               n_sweeps,
                                                               W,
                                                               strideW,
                                                               info,
                                                               batch_count));
}

template <typename T>
struct MatrixEighFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;

    auto &dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);
    int last_dim = dims[dim_size - 1];
    int lda = std::max<int>(1, last_dim);
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    rocblas_fill uplo = is_lower ? rocblas_fill_lower : rocblas_fill_upper;
    rocblas_evect evect =
        has_vectors ? rocblas_evect_original : rocblas_evect_none;

    ValueType *out_value = dev_ctx.template Alloc<ValueType>(eigen_values);
    DenseTensor input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input);
    T *input_vector = input_trans.data<T>();

    auto handle = dev_ctx.cusolver_dn_handle();

    size_t total_bytes = sizeof(T) * batch_size + sizeof(int) * batch_size * 2;
    auto info = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        total_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    auto *residual_ptr = reinterpret_cast<T *>(info->ptr());
    auto *info_ptr = reinterpret_cast<int *>(residual_ptr + batch_size);
    auto *n_sweeps_ptr = reinterpret_cast<int *>(info_ptr + batch_size);

    std::vector<T *> output_ptrs;
    for (int i = 0; i < batch_size; i++) {
      output_ptrs.emplace_back(input_vector + i * vector_stride);
    }
    thrust::device_vector<T *> dev_output_ptrs(output_ptrs.begin(),
                                               output_ptrs.end());

    syevjBatched<T>(handle,
                    rocblas_esort_ascending,
                    evect,
                    uplo,
                    last_dim,
                    thrust::raw_pointer_cast(dev_output_ptrs.data()),
                    lda,
                    0,
                    residual_ptr,
                    100,  // 100 max_sweeps default
                    n_sweeps_ptr,
                    out_value,
                    values_stride,
                    info_ptr,
                    batch_size);

    CheckEighResult(dev_ctx, batch_size, info_ptr);

    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              common::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated,"
                                  "so the eigenvectors must be provided."));
      input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }
};
#endif

#ifdef PADDLE_WITH_CUDA

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices on GPU, and uses the variable has_vectors
// to control whether to return the eigenvectors.
template <typename T>
struct MatrixEighFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;

    int workspace_size = 0;
    auto &dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);
    int last_dim = dims[dim_size - 1];
    int lda = std::max<int>(1, last_dim);
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    cublasFillMode_t uplo =
        is_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz =
        has_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    ValueType *out_value = dev_ctx.template Alloc<ValueType>(eigen_values);
    DenseTensor input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input);
    T *input_vector = input_trans.data<T>();

    // Precision loss will occur in some cases while using
    // cusolverDnZheevjBatched to calculate in Paddle(cuda11.7) but it works
    // well in Paddle(cuda10.2)
    use_cusolver_syevj_batched = (use_cusolver_syevj_batched) &&
                                 (batch_size > 1) &&
                                 (input.dtype() != phi::DataType::COMPLEX128);
    bool use_cusolver_syevj = (input.dtype() == phi::DataType::FLOAT32 &&
                               last_dim >= 32 && last_dim <= 512);
    auto handle = dev_ctx.cusolver_dn_handle();

    syevjInfo_t syevj_params;
    if (use_cusolver_syevj_batched) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cusolverDnCreateSyevjInfo(&syevj_params));
      syevjBatched_bufferSize<T>(handle,
                                 jobz,
                                 uplo,
                                 last_dim,
                                 input_vector,
                                 lda,
                                 out_value,
                                 &workspace_size,
                                 syevj_params,
                                 batch_size);
    } else if (use_cusolver_syevj) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cusolverDnCreateSyevjInfo(&syevj_params));
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevj_bufferSize(
          dev_ctx.cusolver_dn_handle(),
          jobz,
          uplo,
          last_dim,
          reinterpret_cast<const float *>(input_vector),
          lda,
          reinterpret_cast<const float *>(out_value),
          &workspace_size,
          syevj_params));
    } else {
      EvdBuffer(dev_ctx.cusolver_dn_handle(),
                jobz,
                uplo,
                last_dim,
                input_vector,
                lda,
                out_value,
                &workspace_size);
    }
    size_t total_bytes = sizeof(T) * workspace_size + sizeof(int) * batch_size;
    auto work = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        total_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    auto *work_ptr = reinterpret_cast<T *>(work->ptr());
    auto *info_ptr = reinterpret_cast<int *>(work_ptr + workspace_size);

    for (auto i = 0; i < batch_size; ++i) {
      auto *input_data = input_vector + i * vector_stride;
      auto *value_data = out_value + i * values_stride;
      if (use_cusolver_syevj_batched) {
        syevjBatched<T>(handle,
                        jobz,
                        uplo,
                        last_dim,
                        input_data,
                        lda,
                        value_data,
                        work_ptr,
                        workspace_size,
                        &info_ptr[i],
                        syevj_params,
                        batch_size);
        break;
      } else if (use_cusolver_syevj) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cusolverDnSsyevj(handle,
                                      jobz,
                                      uplo,
                                      last_dim,
                                      reinterpret_cast<float *>(input_data),
                                      lda,
                                      reinterpret_cast<float *>(value_data),
                                      reinterpret_cast<float *>(work_ptr),
                                      workspace_size,
                                      &info_ptr[i],
                                      syevj_params));
      } else {
        Evd(handle,
            jobz,
            uplo,
            last_dim,
            input_data,
            lda,
            value_data,
            work_ptr,
            workspace_size,
            &info_ptr[i]);
      }
    }
    CheckEighResult(dev_ctx, batch_size, info_ptr);

    if (use_cusolver_syevj_batched || use_cusolver_syevj) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cusolverDnDestroySyevjInfo(syevj_params));
    }
    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              common::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated,"
                                  "so the eigenvectors must be provided."));
      input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }

  using ValueType = phi::dtype::Real<T>;
  inline void EvdBuffer(cusolverDnHandle_t handle,
                        cusolverEigMode_t jobz,
                        cublasFillMode_t uplo,
                        int n,
                        const T *A,
                        int lda,
                        const ValueType *W,
                        int *lwork) const;

  inline void Evd(cusolverDnHandle_t handle,
                  cusolverEigMode_t jobz,
                  cublasFillMode_t uplo,
                  int n,
                  T *A,
                  int lda,
                  ValueType *W,
                  T *work,
                  int lwork,
                  int *devInfo) const;
};

using phi::dtype::complex;

#define FUNC_WITH_TYPES(m)                       \
  m(float, Ssy, float) m(double, Dsy, double) m( \
      complex<float>, Che, cuComplex) m(complex<double>, Zhe, cuDoubleComplex)

#define EVDBUFFER_INSTANCE(T, C, CastType)                             \
  template <>                                                          \
  inline void MatrixEighFunctor<GPUContext, T>::EvdBuffer(             \
      cusolverDnHandle_t handle,                                       \
      cusolverEigMode_t jobz,                                          \
      cublasFillMode_t uplo,                                           \
      int n,                                                           \
      const T *A,                                                      \
      int lda,                                                         \
      const ValueType *W,                                              \
      int *lwork) const {                                              \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##evd_bufferSize( \
        handle,                                                        \
        jobz,                                                          \
        uplo,                                                          \
        n,                                                             \
        reinterpret_cast<const CastType *>(A),                         \
        lda,                                                           \
        W,                                                             \
        lwork));                                                       \
  }

FUNC_WITH_TYPES(EVDBUFFER_INSTANCE);

#define EVD_INSTANCE(T, C, CastType)                                           \
  template <>                                                                  \
  inline void MatrixEighFunctor<GPUContext, T>::Evd(cusolverDnHandle_t handle, \
                                                    cusolverEigMode_t jobz,    \
                                                    cublasFillMode_t uplo,     \
                                                    int n,                     \
                                                    T *A,                      \
                                                    int lda,                   \
                                                    ValueType *W,              \
                                                    T *work,                   \
                                                    int lwork,                 \
                                                    int *devInfo) const {      \
    PADDLE_ENFORCE_GPU_SUCCESS(                                                \
        dynload::cusolverDn##C##evd(handle,                                    \
                                    jobz,                                      \
                                    uplo,                                      \
                                    n,                                         \
                                    reinterpret_cast<CastType *>(A),           \
                                    lda,                                       \
                                    W,                                         \
                                    reinterpret_cast<CastType *>(work),        \
                                    lwork,                                     \
                                    devInfo));                                 \
  }

FUNC_WITH_TYPES(EVD_INSTANCE);

#undef FUNC_WITH_TYPES
#undef EVDBUFFER_INSTANCE
#undef EVD_INSTANCE

#endif  // PADDLE_WITH_CUDA

}  // namespace funcs
}  // namespace phi
