//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(__NVCC__)
#include <thrust/device_vector.h>
#endif
#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/backends/dynload/cublas.h"
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasScopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemmBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "SgemmBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasSgemmStridedBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "SgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float *alpha,
                      const void *A,
                      cudaDataType_t Atype,
                      int lda,
                      const void *B,
                      cudaDataType_t Btype,
                      int ldb,
                      const float *beta,
                      void *C,
                      cudaDataType_t Ctype,
                      int ldc) {
// Because the gcc 4.8 doesn't expand template parameter pack that
// appears in a lambda-expression, I can not use template parameter pack
// here.
#if CUDA_VERSION >= 8000
    VLOG(5) << "use_tensor_op_math: "
            << (dev_ctx->tensor_core_available() ? "True" : "False");
    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemmEx(handle,
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
                                                             ldc));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasSgemmEx is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrfBatched(args...));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetriBatched(args...));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSmatinvBatched(args...));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrsBatched(args...));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsmBatched(args...));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemmBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "DgemmBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasDgemmStridedBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "DgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently there are not cublasDgemmEx."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetrfBatched(args...));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetriBatched(args...));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDmatinvBatched(args...));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetrsBatched(args...));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsmBatched(args...));
  }
};

template <>
struct CUBlas<phi::dtype::float16> {
  using float16 = phi::dtype::float16;

  static void GEMM(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
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
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasHgemm(handle,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  reinterpret_cast<const __half *>(alpha),
                                  reinterpret_cast<const __half *>(A),
                                  lda,
                                  reinterpret_cast<const __half *>(B),
                                  ldb,
                                  reinterpret_cast<const __half *>(beta),
                                  reinterpret_cast<__half *>(C),
                                  ldc));
  }

#if defined(__NVCC__)
  static void GEMM_BATCH(phi::GPUContext *dev_ctx,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float *alpha,
                         const float16 **A,
                         cudaDataType_t Atype,
                         int lda,
                         const float16 **B,
                         cudaDataType_t Btype,
                         int ldb,
                         const float *beta,
                         float16 **C,
                         cudaDataType_t Ctype,
                         int ldc,
                         int batchCount,
                         cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000
    thrust::device_vector<const void *> A_ptr(A, A + batchCount);
    thrust::device_vector<const void *> B_ptr(B, B + batchCount);
    thrust::device_vector<void *> C_ptr(C, C + batchCount);
    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmBatchedEx(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A_ptr.data().get(),
                                            Atype,
                                            lda,
                                            B_ptr.data().get(),
                                            Btype,
                                            ldb,
                                            beta,
                                            C_ptr.data().get(),
                                            Ctype,
                                            ldc,
                                            batchCount,
                                            computeType,
                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmBatchedEx is not supported on cuda <= 7.5"));
#endif
  }
#endif

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
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
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasHgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const __half *>(alpha),
        reinterpret_cast<const __half *>(A),
        lda,
        strideA,
        reinterpret_cast<const __half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const __half *>(beta),
        reinterpret_cast<__half *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "HgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      cudaDataType_t Atype,
                      int lda,
                      const void *B,
                      cudaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      cudaDataType_t Ctype,
                      int ldc,
                      cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
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
                                                            computeType,
                                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
  }
};

template <>
struct CUBlas<phi::dtype::complex<float>> {
  static void GEMV(cublasHandle_t handle,
                   cublasOperation_t transa,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc));
  }

  static void AXPY(cublasHandle_t handle,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *X,
                   const int incX,
                   phi::dtype::complex<float> *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCaxpy(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(X),
        incX,
        reinterpret_cast<cuFloatComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
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
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "CgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  static void GEMM(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc));
  }

  static void TRSM(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t transa,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   phi::dtype::complex<float> *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<cuFloatComplex *>(B),
        ldb));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      cudaDataType_t Atype,
                      int lda,
                      const void *B,
                      cudaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      cudaDataType_t Ctype,
                      int ldc,
                      cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
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
                                                            computeType,
                                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
  }

  static void TRSM_BATCH(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t transa,
                         cublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::dtype::complex<float> *alpha,
                         const phi::dtype::complex<float> **A,
                         int lda,
                         phi::dtype::complex<float> **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex **>(A),
        lda,
        reinterpret_cast<cuFloatComplex **>(B),
        ldb,
        batch_size));
  }

  static void GETRF_BATCH(cublasHandle_t handle,
                          int n,
                          phi::dtype::complex<float> **A,
                          int lda,
                          int *ipiv,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgetrfBatched(
        handle,
        n,
        reinterpret_cast<cuFloatComplex **>(A),
        lda,
        ipiv,
        info,
        batch_size));
  }

  static void GETRI_BATCH(cublasHandle_t handle,
                          int n,
                          const phi::dtype::complex<float> **A,
                          int lda,
                          const int *ipiv,
                          phi::dtype::complex<float> **Ainv,
                          int ldc,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgetriBatched(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex **>(A),
        lda,
        ipiv,
        reinterpret_cast<cuFloatComplex **>(Ainv),
        ldc,
        info,
        batch_size));
  }

  static void MATINV_BATCH(cublasHandle_t handle,
                           int n,
                           const phi::dtype::complex<float> **A,
                           int lda,
                           phi::dtype::complex<float> **Ainv,
                           int lda_inv,
                           int *info,
                           int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCmatinvBatched(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex **>(A),
        lda,
        reinterpret_cast<cuFloatComplex **>(Ainv),
        lda_inv,
        info,
        batch_size));
  }
};

template <>
struct CUBlas<phi::dtype::complex<double>> {
  static void GEMV(cublasHandle_t handle,
                   cublasOperation_t transa,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc));
  }

  static void AXPY(cublasHandle_t handle,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *X,
                   const int incX,
                   phi::dtype::complex<double> *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZaxpy(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(X),
        incX,
        reinterpret_cast<cuDoubleComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(
      cublasHandle_t handle,
      cublasOperation_t transa,
      cublasOperation_t transb,
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
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "CgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  static void GEMM(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc));
  }

  static void TRSM(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t transa,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   phi::dtype::complex<double> *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<cuDoubleComplex *>(B),
        ldb));
  }

  static void TRSM_BATCH(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t transa,
                         cublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::dtype::complex<double> *alpha,
                         const phi::dtype::complex<double> **A,
                         int lda,
                         phi::dtype::complex<double> **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex **>(A),
        lda,
        reinterpret_cast<cuDoubleComplex **>(B),
        ldb,
        batch_size));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      cudaDataType_t Atype,
                      int lda,
                      const void *B,
                      cudaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      cudaDataType_t Ctype,
                      int ldc,
                      cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
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
                                                            computeType,
                                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
  }

  static void GETRF_BATCH(cublasHandle_t handle,
                          int n,
                          phi::dtype::complex<double> **A,
                          int lda,
                          int *ipiv,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgetrfBatched(
        handle,
        n,
        reinterpret_cast<cuDoubleComplex **>(A),
        lda,
        ipiv,
        info,
        batch_size));
  }

  static void GETRI_BATCH(cublasHandle_t handle,
                          int n,
                          const phi::dtype::complex<double> **A,
                          int lda,
                          const int *ipiv,
                          phi::dtype::complex<double> **Ainv,
                          int ldc,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgetriBatched(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex **>(A),
        lda,
        ipiv,
        reinterpret_cast<cuDoubleComplex **>(Ainv),
        ldc,
        info,
        batch_size));
  }

  static void MATINV_BATCH(cublasHandle_t handle,
                           int n,
                           const phi::dtype::complex<double> **A,
                           int lda,
                           phi::dtype::complex<double> **Ainv,
                           int lda_inv,
                           int *info,
                           int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZmatinvBatched(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex **>(A),
        lda,
        reinterpret_cast<cuDoubleComplex **>(Ainv),
        lda_inv,
        info,
        batch_size));
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
    CUBlas<T>::GEMM_EX(&cuda_ctx,
                       cuTransB,
                       cuTransA,
                       N,
                       M,
                       K,
                       &alpha,
                       B,
                       CUDA_R_32F,
                       ldb,
                       A,
                       CUDA_R_32F,
                       lda,
                       &beta,
                       C,
                       CUDA_R_32F,
                       N);
  } else {
#endif  // CUDA_VERSION >= 8000
    context_.CublasCall([&](cublasHandle_t handle) {
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

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

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

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pseudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::float16>::GEMM_EX(&cuda_ctx,
                                       cuTransB,
                                       cuTransA,
                                       N,
                                       M,
                                       K,
                                       &h_alpha,
                                       B,
                                       CUDA_R_16F,
                                       ldb,
                                       A,
                                       CUDA_R_16F,
                                       lda,
                                       &h_beta,
                                       C,
                                       CUDA_R_16F,
                                       N,
                                       CUDA_R_32F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::float16>::GEMM(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &h_alpha,
                                      h_B,
                                      ldb,
                                      h_A,
                                      lda,
                                      &h_beta,
                                      h_C,
                                      N);
  });
#endif  // CUDA_VERSION >= 8000
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
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = context_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
                                                          cuTransB,
                                                          cuTransA,
                                                          N,
                                                          M,
                                                          K,
                                                          &h_alpha,
                                                          B,
                                                          CUDA_R_16BF,
                                                          ldb,
                                                          A,
                                                          CUDA_R_16BF,
                                                          lda,
                                                          &h_beta,
                                                          C,
                                                          CUDA_R_16BF,
                                                          N,
                                                          CUDA_R_32F,
                                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

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

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pseudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<float>>::GEMM_EX(&cuda_ctx,
                                              cuTransB,
                                              cuTransA,
                                              N,
                                              M,
                                              K,
                                              &c_alpha,
                                              B,
                                              CUDA_C_32F,
                                              ldb,
                                              A,
                                              CUDA_C_32F,
                                              lda,
                                              &c_beta,
                                              C,
                                              CUDA_C_32F,
                                              N,
                                              CUDA_C_32F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::complex<float>>::GEMM(handle,
                                             cuTransB,
                                             cuTransA,
                                             N,
                                             M,
                                             K,
                                             &c_alpha,
                                             h_B,
                                             ldb,
                                             h_A,
                                             lda,
                                             &c_beta,
                                             h_C,
                                             N);
  });
#endif  // CUDA_VERSION >= 8000
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

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

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pseudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<double>>::GEMM_EX(&cuda_ctx,
                                               cuTransB,
                                               cuTransA,
                                               N,
                                               M,
                                               K,
                                               &c_alpha,
                                               B,
                                               CUDA_C_64F,
                                               ldb,
                                               A,
                                               CUDA_C_64F,
                                               lda,
                                               &c_beta,
                                               C,
                                               CUDA_C_64F,
                                               N,
                                               CUDA_C_64F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::complex<double>>::GEMM(handle,
                                              cuTransB,
                                              cuTransA,
                                              N,
                                              M,
                                              K,
                                              &c_alpha,
                                              h_B,
                                              ldb,
                                              h_A,
                                              lda,
                                              &c_beta,
                                              h_C,
                                              N);
  });
#endif  // CUDA_VERSION >= 8000
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
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
    CUBlas<T>::GEMM_EX(&cuda_ctx,
                       cuTransB,
                       cuTransA,
                       N,
                       M,
                       K,
                       &alpha,
                       B,
                       CUDA_R_32F,
                       ldb,
                       A,
                       CUDA_R_32F,
                       lda,
                       &beta,
                       C,
                       CUDA_R_32F,
                       ldc);
  } else {
#endif  // CUDA_VERSION >= 8000

    context_.CublasCall([&](cublasHandle_t handle) {
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

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
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
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  context_.CublasCall([&](cublasHandle_t handle) {
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
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = context_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
                                                          cuTransB,
                                                          cuTransA,
                                                          N,
                                                          M,
                                                          K,
                                                          &h_alpha,
                                                          B,
                                                          CUDA_R_16BF,
                                                          ldb,
                                                          A,
                                                          CUDA_R_16BF,
                                                          lda,
                                                          &h_beta,
                                                          C,
                                                          CUDA_R_16BF,
                                                          ldc,
                                                          CUDA_R_32F,
                                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <typename T>
void Blas<phi::GPUContext>::AXPY(int n, T alpha, const T *x, T *y) const {
  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::SCAL(int n, const T alpha, T *x) const {
  context_.CublasCall(
      [&](cublasHandle_t handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::VCOPY(int n, const T *x, T *y) const {
  context_.CublasCall(
      [&](cublasHandle_t handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
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
  cublasOperation_t cuTransA = !trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

  context_.CublasCall([&](cublasHandle_t handle) {
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
  // Because cublas doesn't support bfloat gemv, we use cublasHgemm to achieve
  // it.
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

#if CUDA_VERSION >= 9010
  if ((FLAGS_enable_cublas_tensor_op_math && (std::is_same<T, float>::value)) ||
      std::is_same<T, phi::dtype::float16>::value) {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
    bool use_tensor_op_math = context_.tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
    VLOG(4) << "use_half_precision_compute_type: "
            << FLAGS_gemm_use_half_precision_compute_type;

    auto fp = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_16F;
#if CUDA_VERSION >= 11000
    auto compute_type = CUBLAS_COMPUTE_32F;
#else
    auto compute_type = CUDA_R_32F;
#endif

    float h_alpha = static_cast<float>(alpha);
    float h_beta = static_cast<float>(beta);
    void *a = static_cast<void *>(&h_alpha);
    void *b = static_cast<void *>(&h_beta);
    // set ComputeType as CUDA_R_32F for fp16, for better accuracy
    if (FLAGS_gemm_use_half_precision_compute_type == true &&
        std::is_same<T, phi::dtype::float16>::value) {
      a = static_cast<void *>(&alpha);
      b = static_cast<void *>(&beta);
#if CUDA_VERSION >= 11000
      compute_type = CUBLAS_COMPUTE_16F;
#else
      compute_type = CUDA_R_16F;
#endif
    }

    context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   N,
                                                   M,
                                                   K,
                                                   a,
                                                   B,
                                                   fp,
                                                   ldb,
                                                   strideB,
                                                   A,
                                                   fp,
                                                   lda,
                                                   strideA,
                                                   b,
                                                   C,
                                                   fp,
                                                   ldc,
                                                   strideC,
                                                   batchCount,
                                                   compute_type,
                                                   algo));
    });
  } else {
#endif  // CUDA_VERSION >= 9010

    context_.CublasCall([&](cublasHandle_t handle) {
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

#if CUDA_VERSION >= 9010
  }
#endif  // CUDA_VERSION >= 9010
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
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
  bool use_tensor_op_math = context_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                 cuTransB,
                                                 cuTransA,
                                                 N,
                                                 M,
                                                 K,
                                                 &h_alpha,
                                                 B,
                                                 CUDA_R_16BF,
                                                 ldb,
                                                 strideB,
                                                 A,
                                                 CUDA_R_16BF,
                                                 lda,
                                                 strideA,
                                                 &h_beta,
                                                 C,
                                                 CUDA_R_16BF,
                                                 ldc,
                                                 strideC,
                                                 batchCount,
                                                 CUBLAS_COMPUTE_32F,
                                                 algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmStridedBatchedEx with bfloat16 is not supported on cuda <= "
      "11"));
#endif  // CUDA_VERSION >= 11000
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

#if defined(__NVCC__)
template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               double alpha,
                                               const double **A,
                                               const double **B,
                                               double beta,
                                               double **C,
                                               int batchCount) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  thrust::device_vector<const double *> A_ptr(A, A + batchCount);
  thrust::device_vector<const double *> B_ptr(B, B + batchCount);
  thrust::device_vector<double *> C_ptr(C, C + batchCount);

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<double>::GEMM_BATCH(handle,
                               cuTransB,
                               cuTransA,
                               N,
                               M,
                               K,
                               &alpha,
                               B_ptr.data().get(),
                               ldb,
                               A_ptr.data().get(),
                               lda,
                               &beta,
                               C_ptr.data().get(),
                               ldc,
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
                                               float alpha,
                                               const float **A,
                                               const float **B,
                                               float beta,
                                               float **C,
                                               int batchCount) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  thrust::device_vector<const float *> A_ptr(A, A + batchCount);
  thrust::device_vector<const float *> B_ptr(B, B + batchCount);
  thrust::device_vector<float *> C_ptr(C, C + batchCount);

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<float>::GEMM_BATCH(handle,
                              cuTransB,
                              cuTransA,
                              N,
                              M,
                              K,
                              &alpha,
                              B_ptr.data().get(),
                              ldb,
                              A_ptr.data().get(),
                              lda,
                              &beta,
                              C_ptr.data().get(),
                              ldc,
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
                                               phi::dtype::float16 alpha,
                                               const phi::dtype::float16 **A,
                                               const phi::dtype::float16 **B,
                                               phi::dtype::float16 beta,
                                               phi::dtype::float16 **C,
                                               int batchCount) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));
  float f_alpha = static_cast<float>(alpha);
  float f_beta = static_cast<float>(beta);
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::float16>::GEMM_BATCH(&cuda_ctx,
                                          cuTransB,
                                          cuTransA,
                                          N,
                                          M,
                                          K,
                                          &f_alpha,
                                          B,
                                          CUDA_R_16F,
                                          ldb,
                                          A,
                                          CUDA_R_16F,
                                          lda,
                                          &f_beta,
                                          C,
                                          CUDA_R_16F,
                                          ldc,
                                          batchCount,
                                          CUDA_R_32F);
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
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          context_.GetComputeCapability()));

  float f_alpha = static_cast<float>(alpha);
  float f_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = context_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  thrust::device_vector<const void *> A_ptr(A, A + batchCount);
  thrust::device_vector<const void *> B_ptr(B, B + batchCount);
  thrust::device_vector<void *> C_ptr(C, C + batchCount);
  context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasGemmBatchedEx(handle,
                                          cuTransB,
                                          cuTransA,
                                          N,
                                          M,
                                          K,
                                          &f_alpha,
                                          B_ptr.data().get(),
                                          CUDA_R_16BF,
                                          ldb,
                                          A_ptr.data().get(),
                                          CUDA_R_16BF,
                                          lda,
                                          &f_beta,
                                          C_ptr.data().get(),
                                          CUDA_R_16BF,
                                          ldc,
                                          batchCount,
                                          CUDA_R_32F,
                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmBatchedEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}
#endif

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
  cublasSideMode_t cuSide =
      (side == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasFillMode_t cuUplo =
      (uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuDiag =
      (diag == CblasUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::TRSM(
        handle, cuSide, cuUplo, cuTransA, cuDiag, N, M, &alpha, A, lda, B, ldb);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRF(
    int n, T **a, int *ipiv, int *info, int batch_size) const {
  context_.CublasCall([&](cublasHandle_t handle) {
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
  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GETRI_BATCH(handle, n, a, n, ipiv, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedMatInv(
    int n, const T **a, T **a_inv, int *info, int batch_size) const {
  context_.CublasCall([&](cublasHandle_t handle) {
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
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTrans =
      (trans == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  context_.CublasCall([&](cublasHandle_t handle) {
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
  cublasSideMode_t cuSide =
      (side == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasFillMode_t cuUplo =
      (uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuDiag =
      (diag == CblasUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  context_.CublasCall([&](cublasHandle_t handle) {
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

}  // namespace funcs
}  // namespace phi
