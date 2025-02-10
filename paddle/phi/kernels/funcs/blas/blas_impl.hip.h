//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/flags.h"

#include "paddle/phi/backends/dynload/rocblas.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"

COMMON_DECLARE_bool(enable_cublas_tensor_op_math);
COMMON_DECLARE_bool(gemm_use_half_precision_compute_type);

namespace phi {
namespace funcs {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_sgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_saxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_sscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_scopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_sgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_sgemm_strided_batched(args...));
  }

  // HIP not supported, refer to the doc here:
  // https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.5.x/docs/markdown/CUBLAS_API_supported_by_HIP.md
  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasSgemmEx is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_strsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasSgetrfBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasSgetriBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasSmatinvBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
#if HIP_VERSION >= 30000000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_strsm_batched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasStrsmBatched is not supported on HIP platform."));
#endif
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_dgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_daxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_dscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_dcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_dgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_dgemm_strided_batched(args...));
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently there are not cublasDgemmEx."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_dtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasDgetrfBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasDgetriBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasDmatinvBatched is not supported on HIP platform."));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
#if HIP_VERSION >= 30000000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_dtrsm_batched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasDtrsmBatched is not supported on HIP platform."));
#endif
  }
};

template <>
struct CUBlas<phi::dtype::float16> {
  using float16 = phi::dtype::float16;

  static void GEMM(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float16 *alpha,
                   const float16 *A,
                   int lda,
                   const float16 *B,
                   int ldb,
                   const float16 *beta,
                   float16 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_hgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const rocblas_half *>(alpha),
        reinterpret_cast<const rocblas_half *>(A),
        lda,
        reinterpret_cast<const rocblas_half *>(B),
        ldb,
        reinterpret_cast<const rocblas_half *>(beta),
        reinterpret_cast<rocblas_half *>(C),
        ldc));
  }

  static void GEMM_STRIDED_BATCH(rocblas_handle handle,
                                 rocblas_operation transa,
                                 rocblas_operation transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float16 *alpha,
                                 const float16 *A,
                                 int lda,
                                 long long int strideA,  // NOLINT
                                 const float16 *B,       // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const float16 *beta,
                                 float16 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_hgemm_strided_batched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const rocblas_half *>(alpha),
        reinterpret_cast<const rocblas_half *>(A),
        lda,
        strideA,
        reinterpret_cast<const rocblas_half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const rocblas_half *>(beta),
        reinterpret_cast<rocblas_half *>(C),
        ldc,
        strideC,
        batchCount));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      rocblas_operation transa,
                      rocblas_operation transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      rocblas_datatype Atype,
                      int lda,
                      const void *B,
                      rocblas_datatype Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      rocblas_datatype Ctype,
                      int ldc,
                      rocblas_datatype computeType) {
    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_gemm_ex(handle,
                                                               transa,
                                                               transb,
                                                               m,
                                                               n,
                                                               k,
                                                               alpha,
                                                               A,
                                                               Atype,
                                                               lda,
                                                               B,
                                                               Btype,
                                                               ldb,
                                                               beta,
                                                               C,
                                                               Ctype,
                                                               ldc,
                                                               C,
                                                               Ctype,
                                                               ldc,
                                                               computeType,
                                                               algo,
                                                               0,
                                                               0));
    });
  }
};

template <>
struct CUBlas<phi::dtype::complex<float>> {
  static void GEMV(rocblas_handle handle,
                   rocblas_operation transa,
                   int m,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   const phi::dtype::complex<float> *B,
                   int ldb,
                   const phi::dtype::complex<float> *beta,
                   phi::dtype::complex<float> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_cgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const rocblas_float_complex *>(alpha),
        reinterpret_cast<const rocblas_float_complex *>(A),
        lda,
        reinterpret_cast<const rocblas_float_complex *>(B),
        ldb,
        reinterpret_cast<const rocblas_float_complex *>(beta),
        reinterpret_cast<rocblas_float_complex *>(C),
        ldc));
  }

  static void AXPY(rocblas_handle handle,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *X,
                   const int incX,
                   phi::dtype::complex<float> *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_caxpy(
        handle,
        n,
        reinterpret_cast<const rocblas_float_complex *>(alpha),
        reinterpret_cast<const rocblas_float_complex *>(X),
        incX,
        reinterpret_cast<rocblas_float_complex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(rocblas_handle handle,
                                 rocblas_operation transa,
                                 rocblas_operation transb,
                                 int m,
                                 int n,
                                 int k,
                                 const phi::dtype::complex<float> *alpha,
                                 const phi::dtype::complex<float> *A,
                                 int lda,
                                 long long int strideA,                // NOLINT
                                 const phi::dtype::complex<float> *B,  // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const phi::dtype::complex<float> *beta,
                                 phi::dtype::complex<float> *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_cgemm_strided_batched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const rocblas_float_complex *>(alpha),
        reinterpret_cast<const rocblas_float_complex *>(A),
        lda,
        strideA,
        reinterpret_cast<const rocblas_float_complex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const rocblas_float_complex *>(beta),
        reinterpret_cast<rocblas_float_complex *>(C),
        ldc,
        strideC,
        batchCount));
  }

  static void GEMM(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   const phi::dtype::complex<float> *B,
                   int ldb,
                   const phi::dtype::complex<float> *beta,
                   phi::dtype::complex<float> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_cgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const rocblas_float_complex *>(alpha),
        reinterpret_cast<const rocblas_float_complex *>(A),
        lda,
        reinterpret_cast<const rocblas_float_complex *>(B),
        ldb,
        reinterpret_cast<const rocblas_float_complex *>(beta),
        reinterpret_cast<rocblas_float_complex *>(C),
        ldc));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      rocblas_operation transa,
                      rocblas_operation transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      rocblas_datatype Atype,
                      int lda,
                      const void *B,
                      rocblas_datatype Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      rocblas_datatype Ctype,
                      int ldc,
                      rocblas_datatype computeType) {
    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_gemm_ex(handle,
                                                               transa,
                                                               transb,
                                                               m,
                                                               n,
                                                               k,
                                                               alpha,
                                                               A,
                                                               Atype,
                                                               lda,
                                                               B,
                                                               Btype,
                                                               ldb,
                                                               beta,
                                                               C,
                                                               Ctype,
                                                               ldc,
                                                               C,
                                                               Ctype,
                                                               ldc,
                                                               computeType,
                                                               algo,
                                                               0,
                                                               0));
    });
  }
};

template <>
struct CUBlas<phi::dtype::complex<double>> {
  static void GEMV(rocblas_handle handle,
                   rocblas_operation transa,
                   int m,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   const phi::dtype::complex<double> *B,
                   int ldb,
                   const phi::dtype::complex<double> *beta,
                   phi::dtype::complex<double> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_zgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const rocblas_double_complex *>(alpha),
        reinterpret_cast<const rocblas_double_complex *>(A),
        lda,
        reinterpret_cast<const rocblas_double_complex *>(B),
        ldb,
        reinterpret_cast<const rocblas_double_complex *>(beta),
        reinterpret_cast<rocblas_double_complex *>(C),
        ldc));
  }

  static void AXPY(rocblas_handle handle,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *X,
                   const int incX,
                   phi::dtype::complex<double> *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_zaxpy(
        handle,
        n,
        reinterpret_cast<const rocblas_double_complex *>(alpha),
        reinterpret_cast<const rocblas_double_complex *>(X),
        incX,
        reinterpret_cast<rocblas_double_complex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(
      rocblas_handle handle,
      rocblas_operation transa,
      rocblas_operation transb,
      int m,
      int n,
      int k,
      const phi::dtype::complex<double> *alpha,
      const phi::dtype::complex<double> *A,
      int lda,
      long long int strideA,                 // NOLINT
      const phi::dtype::complex<double> *B,  // NOLINT
      int ldb,
      long long int strideB,  // NOLINT
      const phi::dtype::complex<double> *beta,
      phi::dtype::complex<double> *C,
      int ldc,
      long long int strideC,  // NOLINT
      int batchCount) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_zgemm_strided_batched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const rocblas_double_complex *>(alpha),
        reinterpret_cast<const rocblas_double_complex *>(A),
        lda,
        strideA,
        reinterpret_cast<const rocblas_double_complex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const rocblas_double_complex *>(beta),
        reinterpret_cast<rocblas_double_complex *>(C),
        ldc,
        strideC,
        batchCount));
  }

  static void GEMM(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   const phi::dtype::complex<double> *B,
                   int ldb,
                   const phi::dtype::complex<double> *beta,
                   phi::dtype::complex<double> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_zgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const rocblas_double_complex *>(alpha),
        reinterpret_cast<const rocblas_double_complex *>(A),
        lda,
        reinterpret_cast<const rocblas_double_complex *>(B),
        ldb,
        reinterpret_cast<const rocblas_double_complex *>(beta),
        reinterpret_cast<rocblas_double_complex *>(C),
        ldc));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      rocblas_operation transa,
                      rocblas_operation transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      rocblas_datatype Atype,
                      int lda,
                      const void *B,
                      rocblas_datatype Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      rocblas_datatype Ctype,
                      int ldc,
                      rocblas_datatype computeType) {
    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_gemm_ex(handle,
                                                               transa,
                                                               transb,
                                                               m,
                                                               n,
                                                               k,
                                                               alpha,
                                                               A,
                                                               Atype,
                                                               lda,
                                                               B,
                                                               Btype,
                                                               ldb,
                                                               beta,
                                                               C,
                                                               Ctype,
                                                               ldc,
                                                               C,
                                                               Ctype,
                                                               ldc,
                                                               computeType,
                                                               algo,
                                                               0,
                                                               0));
    });
  }
};

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMM(handle,
                    cuTransB,
                    cuTransA,
                    N,
                    M,
                    K,
                    &alpha,
                    B,
                    ldb,
                    A,
                    lda,
                    &beta,
                    C,
                    N);
  });
}

template <>
template <typename T, typename U>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 U alpha,
                                 const T *A,
                                 const T *B,
                                 U beta,
                                 T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  T t_alpha = static_cast<T>(alpha);
  T t_beta = static_cast<T>(beta);

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMM(handle,
                    cuTransB,
                    cuTransA,
                    N,
                    M,
                    K,
                    &t_alpha,
                    B,
                    ldb,
                    A,
                    lda,
                    &t_beta,
                    C,
                    N);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  rocblas_datatype compute_type = rocblas_datatype_f32_r;
  if (FLAGS_gemm_use_half_precision_compute_type == true) {
    compute_type = rocblas_datatype_f16_r;
  }
  VLOG(4) << "gemm_use_half_precision_compute_type: "
          << FLAGS_gemm_use_half_precision_compute_type;

  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::float16>::GEMM_EX(&cuda_ctx,
                                       cuTransB,
                                       cuTransA,
                                       N,
                                       M,
                                       K,
                                       &h_alpha,
                                       B,
                                       rocblas_datatype_f16_r,
                                       ldb,
                                       A,
                                       rocblas_datatype_f16_r,
                                       lda,
                                       &h_beta,
                                       C,
                                       rocblas_datatype_f16_r,
                                       N,
                                       compute_type);
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        float alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        float beta,
                                        phi::dtype::float16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = alpha;
  float h_beta = beta;

  rocblas_datatype compute_type = rocblas_datatype_f32_r;
  if (FLAGS_gemm_use_half_precision_compute_type == true) {
    compute_type = rocblas_datatype_f16_r;
  }
  VLOG(4) << "gemm_use_half_precision_compute_type: "
          << FLAGS_gemm_use_half_precision_compute_type;

  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::float16>::GEMM_EX(&cuda_ctx,
                                       cuTransB,
                                       cuTransA,
                                       N,
                                       M,
                                       K,
                                       &h_alpha,
                                       B,
                                       rocblas_datatype_f16_r,
                                       ldb,
                                       A,
                                       rocblas_datatype_f16_r,
                                       lda,
                                       &h_beta,
                                       C,
                                       rocblas_datatype_f16_r,
                                       N,
                                       compute_type);
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  // TODO(zhiqiu): 80 has the same meaning for rocm and cuda?
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "rocblas bf16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  context_.TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_gemm_ex(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &h_alpha,
                                      B,
                                      rocblas_datatype_bf16_r,
                                      ldb,
                                      A,
                                      rocblas_datatype_bf16_r,
                                      lda,
                                      &h_beta,
                                      C,
                                      rocblas_datatype_bf16_r,
                                      N,
                                      C,
                                      rocblas_datatype_bf16_r,
                                      N,
                                      rocblas_datatype_f32_r,
                                      algo,
                                      0,
                                      0));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        float alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        float beta,
                                        phi::dtype::bfloat16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  // TODO(zhiqiu): 80 has the same meaning for rocm and cuda?
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "rocblas bf16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = alpha;
  float h_beta = beta;
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  context_.TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_gemm_ex(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &h_alpha,
                                      B,
                                      rocblas_datatype_bf16_r,
                                      ldb,
                                      A,
                                      rocblas_datatype_bf16_r,
                                      lda,
                                      &h_beta,
                                      C,
                                      rocblas_datatype_bf16_r,
                                      N,
                                      C,
                                      rocblas_datatype_bf16_r,
                                      N,
                                      rocblas_datatype_f32_r,
                                      algo,
                                      0,
                                      0));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<float> alpha,
                                        const phi::dtype::complex<float> *A,
                                        const phi::dtype::complex<float> *B,
                                        phi::dtype::complex<float> beta,
                                        phi::dtype::complex<float> *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas complex64 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<float> c_alpha =
      thrust::complex<float>(alpha.real, alpha.imag);
  thrust::complex<float> c_beta = thrust::complex<float>(beta.real, beta.imag);

  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<float>>::GEMM_EX(&cuda_ctx,
                                              cuTransB,
                                              cuTransA,
                                              N,
                                              M,
                                              K,
                                              &c_alpha,
                                              B,
                                              rocblas_datatype_f32_c,
                                              ldb,
                                              A,
                                              rocblas_datatype_f32_c,
                                              lda,
                                              &c_beta,
                                              C,
                                              rocblas_datatype_f32_c,
                                              N,
                                              rocblas_datatype_f32_c);
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<double> alpha,
                                        const phi::dtype::complex<double> *A,
                                        const phi::dtype::complex<double> *B,
                                        phi::dtype::complex<double> beta,
                                        phi::dtype::complex<double> *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas complex128 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<double> c_alpha =
      thrust::complex<double>(alpha.real, alpha.imag);
  thrust::complex<double> c_beta =
      thrust::complex<double>(beta.real, beta.imag);

  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<double>>::GEMM_EX(&cuda_ctx,
                                               cuTransB,
                                               cuTransA,
                                               N,
                                               M,
                                               K,
                                               &c_alpha,
                                               B,
                                               rocblas_datatype_f64_c,
                                               ldb,
                                               A,
                                               rocblas_datatype_f64_c,
                                               lda,
                                               &c_beta,
                                               C,
                                               rocblas_datatype_f64_c,
                                               N,
                                               rocblas_datatype_f64_c);
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(bool transA,
                                 bool transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 const T *B,
                                 int ldb,
                                 T beta,
                                 T *C,
                                 int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  rocblas_operation cuTransA =
      transA ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation cuTransB =
      transB ? rocblas_operation_transpose : rocblas_operation_none;
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMM(handle,
                    cuTransB,
                    cuTransA,
                    N,
                    M,
                    K,
                    &alpha,
                    B,
                    ldb,
                    A,
                    lda,
                    &beta,
                    C,
                    ldc);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        int lda,
                                        const phi::dtype::float16 *B,
                                        int ldb,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C,
                                        int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  rocblas_operation cuTransA =
      transA ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation cuTransB =
      transB ? rocblas_operation_transpose : rocblas_operation_none;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<phi::dtype::float16>::GEMM(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &alpha,
                                      B,
                                      ldb,
                                      A,
                                      lda,
                                      &beta,
                                      C,
                                      ldc);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        int lda,
                                        const phi::dtype::bfloat16 *B,
                                        int ldb,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C,
                                        int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  rocblas_operation cuTransA =
      transA ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation cuTransB =
      transB ? rocblas_operation_none : rocblas_operation_transpose;
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "rocblas bf16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  context_.TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_gemm_ex(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &h_alpha,
                                      B,
                                      rocblas_datatype_bf16_r,
                                      ldb,
                                      A,
                                      rocblas_datatype_bf16_r,
                                      lda,
                                      &h_beta,
                                      C,
                                      rocblas_datatype_bf16_r,
                                      ldc,
                                      C,
                                      rocblas_datatype_bf16_r,
                                      ldc,
                                      rocblas_datatype_f32_r,
                                      algo,
                                      0,
                                      0));
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::AXPY(int n, T alpha, const T *x, T *y) const {
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::SCAL(int n, const T alpha, T *x) const {
  context_.CublasCall(
      [&](rocblas_handle handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::VCOPY(int n, const T *x, T *y) const {
  context_.CublasCall(
      [&](rocblas_handle handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  rocblas_operation cuTransA =
      !trans_a ? rocblas_operation_transpose : rocblas_operation_none;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMV(handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {
  // Because cublas doesn't support half gemv, we use cublasHgemm to achieve it.
  if (trans_a) {
    this->template GEMM<phi::dtype::float16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::dtype::float16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {
  // Because rocblas doesn't support bfloat16 gemv, we use gemmex to achieve it.
  if (trans_a) {
    this->template GEMM<phi::dtype::bfloat16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::dtype::bfloat16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T *A,
                                        const T *B,
                                        T beta,
                                        T *C,
                                        int batchCount,
                                        int64_t strideA,
                                        int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  const int64_t strideC = M * N;
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                  cuTransB,
                                  cuTransA,
                                  N,
                                  M,
                                  K,
                                  &alpha,
                                  B,
                                  ldb,
                                  strideB,
                                  A,
                                  lda,
                                  strideA,
                                  &beta,
                                  C,
                                  ldc,
                                  strideC,
                                  batchCount);
  });
}

template <>
template <typename T, typename U>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        U alpha,
                                        const T *A,
                                        const T *B,
                                        U beta,
                                        T *C,
                                        int batchCount,
                                        int64_t strideA,
                                        int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  const int64_t strideC = M * N;

  T h_alpha = static_cast<T>(alpha);
  T h_beta = static_cast<T>(beta);

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                  cuTransB,
                                  cuTransA,
                                  N,
                                  M,
                                  K,
                                  &h_alpha,
                                  B,
                                  ldb,
                                  strideB,
                                  A,
                                  lda,
                                  strideA,
                                  &h_beta,
                                  C,
                                  ldc,
                                  strideC,
                                  batchCount);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float16 alpha,
                                               const float16 *A,
                                               const float16 *B,
                                               float16 beta,
                                               float16 *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  const int64_t strideC = M * N;
  context_.CublasCall([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_hgemm_strided_batched(
        handle,
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        reinterpret_cast<const rocblas_half *>(&alpha),
        reinterpret_cast<const rocblas_half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const rocblas_half *>(A),
        lda,
        strideA,
        reinterpret_cast<const rocblas_half *>(&beta),
        reinterpret_cast<rocblas_half *>(C),
        ldc,
        strideC,
        batchCount));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               const float16 *A,
                                               const float16 *B,
                                               float beta,
                                               float16 *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  const int64_t strideC = M * N;

  float16 h_alpha = static_cast<float16>(alpha);
  float16 h_beta = static_cast<float16>(beta);

  context_.CublasCall([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_hgemm_strided_batched(
        handle,
        cuTransB,
        cuTransA,
        N,
        M,
        K,
        reinterpret_cast<const rocblas_half *>(&h_alpha),
        reinterpret_cast<const rocblas_half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const rocblas_half *>(A),
        lda,
        strideA,
        reinterpret_cast<const rocblas_half *>(&h_beta),
        reinterpret_cast<rocblas_half *>(C),
        ldc,
        strideC,
        batchCount));
  });
}

// note(wangran16): unknown bug. parameters dislocation when calling
// GEMM_STRIDED_BATCH<float> and GEMM_STRIDED_BATCH<double>
template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               const float *A,
                                               const float *B,
                                               float beta,
                                               float *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  const int64_t strideC = M * N;
  context_.CublasCall([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_sgemm_strided_batched(handle,
                                                    cuTransB,
                                                    cuTransA,
                                                    N,
                                                    M,
                                                    K,
                                                    &alpha,
                                                    B,
                                                    ldb,
                                                    strideB,
                                                    A,
                                                    lda,
                                                    strideA,
                                                    &beta,
                                                    C,
                                                    ldc,
                                                    strideC,
                                                    batchCount));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               double alpha,
                                               const double *A,
                                               const double *B,
                                               double beta,
                                               double *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  const int64_t strideC = M * N;
  context_.CublasCall([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_dgemm_strided_batched(handle,
                                                    cuTransB,
                                                    cuTransA,
                                                    N,
                                                    M,
                                                    K,
                                                    &alpha,
                                                    B,
                                                    ldb,
                                                    strideB,
                                                    A,
                                                    lda,
                                                    strideA,
                                                    &beta,
                                                    C,
                                                    ldc,
                                                    strideC,
                                                    batchCount));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::bfloat16 alpha,
                                               const phi::dtype::bfloat16 *A,
                                               const phi::dtype::bfloat16 *B,
                                               phi::dtype::bfloat16 beta,
                                               phi::dtype::bfloat16 *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  const int64_t strideC = M * N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  context_.TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_gemm_strided_batched_ex(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &h_alpha,
                                                      B,
                                                      rocblas_datatype_bf16_r,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      rocblas_datatype_bf16_r,
                                                      lda,
                                                      strideA,
                                                      &h_beta,
                                                      C,
                                                      rocblas_datatype_bf16_r,
                                                      ldc,
                                                      strideC,
                                                      C,
                                                      rocblas_datatype_bf16_r,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      rocblas_datatype_f32_r,
                                                      algo,
                                                      0,
                                                      0));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               const phi::dtype::bfloat16 *A,
                                               const phi::dtype::bfloat16 *B,
                                               float beta,
                                               phi::dtype::bfloat16 *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  const int64_t strideC = M * N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  float h_alpha = alpha;
  float h_beta = beta;
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  context_.TensorCoreCublasCallIfAvailable([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_gemm_strided_batched_ex(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &h_alpha,
                                                      B,
                                                      rocblas_datatype_bf16_r,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      rocblas_datatype_bf16_r,
                                                      lda,
                                                      strideA,
                                                      &h_beta,
                                                      C,
                                                      rocblas_datatype_bf16_r,
                                                      ldc,
                                                      strideC,
                                                      C,
                                                      rocblas_datatype_bf16_r,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      rocblas_datatype_f32_r,
                                                      algo,
                                                      0,
                                                      0));
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T **A,
                                        const T **B,
                                        T beta,
                                        T **C,
                                        int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::float16 alpha,
                                               const phi::dtype::float16 **A,
                                               const phi::dtype::float16 **B,
                                               phi::dtype::float16 beta,
                                               phi::dtype::float16 **C,
                                               int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<phi::dtype::float16>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::bfloat16 alpha,
                                               const phi::dtype::bfloat16 **A,
                                               const phi::dtype::bfloat16 **B,
                                               phi::dtype::bfloat16 beta,
                                               phi::dtype::bfloat16 **C,
                                               int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<phi::dtype::bfloat16>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::TRSM(CBLAS_SIDE side,
                                 CBLAS_UPLO uplo,
                                 CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 T *B,
                                 int ldb) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  rocblas_side cuSide =
      (side == CblasLeft) ? rocblas_side_right : rocblas_side_left;
  rocblas_fill cuUplo =
      (uplo == CblasLower) ? rocblas_fill_upper : rocblas_fill_lower;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_diagonal cuDiag =
      (diag == CblasUnit) ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::TRSM(
        handle, cuSide, cuUplo, cuTransA, cuDiag, N, M, &alpha, A, lda, B, ldb);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRF(
    int n, T **a, int *ipiv, int *info, int batch_size) const {
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GETRF_BATCH(handle, n, a, n, ipiv, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRI(int n,
                                         const T **a,
                                         const int *ipiv,
                                         T **a_inv,
                                         int *info,
                                         int batch_size) const {
  PADDLE_ENFORCE_NE(
      a_inv,
      a,
      common::errors::InvalidArgument(
          "cuBLAS function 'cublas<S/D>getrfBatched' cannot be executed "
          "in-place. The memory space of output matrix (address: %p) cannot "
          "overlap memory space of input matrix (address: %p).",
          a_inv,
          a));
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GETRI_BATCH(handle, n, a, n, ipiv, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedMatInv(
    int n, const T **a, T **a_inv, int *info, int batch_size) const {
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::MATINV_BATCH(handle, n, a, n, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRS(CBLAS_TRANSPOSE trans,
                                         int n,
                                         int nrhs,
                                         const T **a,
                                         int lda,
                                         int *ipiv,
                                         T **b,
                                         int ldb,
                                         int *info,
                                         int batch_size) const {
  rocblas_operation cuTrans = (trans == CblasNoTrans)
                                  ? rocblas_operation_none
                                  : rocblas_operation_transpose;
  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::GETRS_BATCH(
        handle, cuTrans, n, nrhs, a, lda, ipiv, b, ldb, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedTRSM(CBLAS_SIDE side,
                                        CBLAS_UPLO uplo,
                                        CBLAS_TRANSPOSE transA,
                                        CBLAS_DIAG diag,
                                        int M,
                                        int N,
                                        T alpha,
                                        const T **A,
                                        int lda,
                                        T **B,
                                        int ldb,
                                        int batch_size) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  rocblas_side cuSide =
      (side == CblasLeft) ? rocblas_side_right : rocblas_side_left;
  rocblas_fill cuUplo =
      (uplo == CblasLower) ? rocblas_fill_upper : rocblas_fill_lower;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_diagonal cuDiag =
      (diag == CblasUnit) ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;

  context_.CublasCall([&](rocblas_handle handle) {
    CUBlas<T>::TRSM_BATCH(handle,
                          cuSide,
                          cuUplo,
                          cuTransA,
                          cuDiag,
                          N,
                          M,
                          &alpha,
                          A,
                          lda,
                          B,
                          ldb,
                          batch_size);
  });
}

static void Int8GEMM_EX(phi::GPUContext *dev_ctx,
                        rocblas_operation transa,
                        rocblas_operation transb,
                        int m,
                        int n,
                        int k,
                        const void *alpha,
                        const void *A,
                        rocblas_datatype Atype,
                        int lda,
                        const void *B,
                        rocblas_datatype Btype,
                        int ldb,
                        const void *beta,
                        void *C,
                        rocblas_datatype Ctype,
                        int ldc,
                        rocblas_datatype computeType) {
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
  dev_ctx->CublasCall([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::rocblas_gemm_ex(handle,
                                                             transa,
                                                             transb,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             A,
                                                             Atype,
                                                             lda,
                                                             B,
                                                             Btype,
                                                             ldb,
                                                             beta,
                                                             C,
                                                             Ctype,
                                                             ldc,
                                                             C,
                                                             Ctype,
                                                             ldc,
                                                             computeType,
                                                             algo,
                                                             0,
                                                             0));
  });
}

inline void Int8GEMM(const GPUContext &context_,
                     CBLAS_TRANSPOSE transA,
                     CBLAS_TRANSPOSE transB,
                     int M,
                     int N,
                     int K,
                     int32_t alpha,
                     const int8_t *A,
                     const int8_t *B,
                     int32_t beta,
                     int32_t *C) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  Int8GEMM_EX(&cuda_ctx,
              cuTransB,
              cuTransA,
              N,
              M,
              K,
              &alpha,
              B,
              rocblas_datatype_i8_r,
              ldb,
              A,
              rocblas_datatype_i8_r,
              lda,
              &beta,
              C,
              rocblas_datatype_i32_r,
              N,
              rocblas_datatype_i32_r);
}

inline void Int8BatchedGEMM(const GPUContext &context_,
                            CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB,
                            int M,
                            int N,
                            int K,
                            int32_t alpha,
                            const int8_t *A,
                            const int8_t *B,
                            int32_t beta,
                            int32_t *C,
                            int batchCount,
                            int64_t strideA,
                            int64_t strideB) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  const int64_t strideC = M * N;
  rocblas_operation cuTransA = (transA == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;
  rocblas_operation cuTransB = (transB == CblasNoTrans)
                                   ? rocblas_operation_none
                                   : rocblas_operation_transpose;

  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  context_.CublasCall([&](rocblas_handle handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::rocblas_gemm_strided_batched_ex(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &alpha,
                                                      B,
                                                      rocblas_datatype_i8_r,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      rocblas_datatype_i8_r,
                                                      lda,
                                                      strideA,
                                                      &beta,
                                                      C,
                                                      rocblas_datatype_i32_r,
                                                      ldc,
                                                      strideC,
                                                      C,
                                                      rocblas_datatype_i32_r,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      rocblas_datatype_i32_r,
                                                      algo,
                                                      0,
                                                      0));
  });
}

inline void Int8BatchedGEMM(const GPUContext &context_,
                            CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB,
                            int M,
                            int N,
                            int K,
                            int32_t alpha,
                            const int8_t **A,
                            const int8_t **B,
                            int32_t beta,
                            int32_t **C,
                            int batchCount) {
  for (int k = 0; k < batchCount; ++k) {
    Int8GEMM(context_, transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

inline void Int8GEMV(const GPUContext &context_,
                     bool trans_a,
                     int M,
                     int N,
                     int32_t alpha,
                     const int8_t *A,
                     const int8_t *B,
                     int32_t beta,
                     int32_t *C) {
  if (trans_a) {
    Int8GEMM(
        context_, CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    Int8GEMM(
        context_, CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

}  // namespace funcs
}  // namespace phi
