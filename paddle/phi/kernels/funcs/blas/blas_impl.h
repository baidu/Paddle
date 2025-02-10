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
#include "paddle/phi/backends/cpu/cpu_context.h"
#ifdef PADDLE_WITH_MKLML
#include <mkl.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

namespace detail {
template <typename T>
static void axpy(
    int n, const T alpha, const T *x, const int incx, T *y, const int incy) {
  // Y = Y + alpha * X
  while (n-- > 0) {
    *y += alpha * *x;
    y = y + incy;
    x = x + incx;
  }
}
}  // namespace detail

template <typename T>
struct CBlas;

template <>
struct CBlas<int8_t> {
  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Blas VCOPY do not supported on CPU, please check your code"));
  }
};

template <>
struct CBlas<int16_t> {
  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Blas VCOPY do not supported on CPU, please check your code"));
  }
};

template <>
struct CBlas<phi::dtype::bfloat16> {
  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Blas VCOPY do not supported on CPU with bfloat16,"
        " please check your code"));
  }

  template <typename... ARGS>
  static void VADD(int n,
                   const phi::dtype::bfloat16 *x,
                   const phi::dtype::bfloat16 *y,
                   phi::dtype::bfloat16 *z) {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] + y[i];
    }
  }

  template <typename... ARGS>
  static void VMUL(int n,
                   const phi::dtype::bfloat16 *x,
                   const phi::dtype::bfloat16 *y,
                   phi::dtype::bfloat16 *z) {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] * y[i];
    }
  }

  template <typename... ARGS>
  static void VSUB(int n,
                   const phi::dtype::bfloat16 *x,
                   const phi::dtype::bfloat16 *y,
                   phi::dtype::bfloat16 *z) {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] - y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    phi::dynload::cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static float *GEMM_ALLOC(ARGS... args) {
    return phi::dynload::cblas_sgemm_alloc(args...);
  }

  template <typename... ARGS>
  static void GEMM_PACK(ARGS... args) {
    phi::dynload::cblas_sgemm_pack(args...);
  }

  template <typename... ARGS>
  static void GEMM_COMPUTE(ARGS... args) {
    phi::dynload::cblas_sgemm_compute(args...);
  }

  template <typename... ARGS>
  static void GEMM_FREE(ARGS... args) {
    phi::dynload::cblas_sgemm_free(args...);
  }

#ifdef PADDLE_WITH_LIBXSMM
  template <typename... ARGS>
  static void SMM_GEMM(ARGS... args) {
    libxsmm_sgemm(args...);
  }
#endif

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    phi::dynload::cblas_saxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    phi::dynload::cblas_scopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    phi::dynload::cblas_sgemv(args...);
  }

  template <typename... ARGS>
  static float DOT(ARGS... args) {
    return phi::dynload::cblas_sdot(args...);
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    phi::dynload::cblas_sscal(args...);
  }

  template <typename... ARGS>
  static float ASUM(ARGS... args) {
    return phi::dynload::cblas_sasum(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    phi::dynload::cblas_sgemm_batch(args...);
  }

  template <typename... ARGS>
  static void VADD(ARGS... args) {
    phi::dynload::vsAdd(args...);
  }

  template <typename... ARGS>
  static void VSUB(ARGS... args) {
    phi::dynload::vsSub(args...);
  }

  template <typename... ARGS>
  static void VMUL(ARGS... args) {
    phi::dynload::vsMul(args...);
  }

  template <typename... ARGS>
  static void VDIV(ARGS... args) {
    phi::dynload::vsDiv(args...);
  }

  template <typename... ARGS>
  static void VEXP(ARGS... args) {
    phi::dynload::vsExp(args...);
  }

  template <typename... ARGS>
  static void VSQUARE(ARGS... args) {
    phi::dynload::vsSqr(args...);
  }

  template <typename... ARGS>
  static void VPOW(ARGS... args) {
    phi::dynload::vsPowx(args...);
  }

  template <typename... ARGS>
  static void VINV(ARGS... args) {
    phi::dynload::vsInv(args...);
  }

  template <typename... ARGS>
  static void VMERF(ARGS... args) {
    phi::dynload::vmsErf(args...);
  }
#if !defined(_WIN32)
  template <typename... ARGS>
  static void CSRMM(ARGS... args) {
    phi::dynload::mkl_scsrmm(args...);
  }
#endif

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    phi::dynload::cblas_strsm(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    phi::dynload::cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static double *GEMM_ALLOC(ARGS... args) {
    return phi::dynload::cblas_dgemm_alloc(args...);
  }

  template <typename... ARGS>
  static void GEMM_PACK(ARGS... args) {
    phi::dynload::cblas_dgemm_pack(args...);
  }

  template <typename... ARGS>
  static void GEMM_COMPUTE(ARGS... args) {
    phi::dynload::cblas_dgemm_compute(args...);
  }

  template <typename... ARGS>
  static void GEMM_FREE(ARGS... args) {
    phi::dynload::cblas_dgemm_free(args...);
  }

#ifdef PADDLE_WITH_LIBXSMM
  template <typename... ARGS>
  static void SMM_GEMM(ARGS... args) {
    libxsmm_dgemm(args...);
  }
#endif

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    phi::dynload::cblas_daxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    phi::dynload::cblas_dcopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    phi::dynload::cblas_dgemv(args...);
  }

  template <typename... ARGS>
  static double DOT(ARGS... args) {
    return phi::dynload::cblas_ddot(args...);
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    phi::dynload::cblas_dscal(args...);
  }

  template <typename... ARGS>
  static double ASUM(ARGS... args) {
    return phi::dynload::cblas_dasum(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    phi::dynload::cblas_dgemm_batch(args...);
  }

  template <typename... ARGS>
  static void VADD(ARGS... args) {
    phi::dynload::vdAdd(args...);
  }

  template <typename... ARGS>
  static void VSUB(ARGS... args) {
    phi::dynload::vdSub(args...);
  }

  template <typename... ARGS>
  static void VMUL(ARGS... args) {
    phi::dynload::vdMul(args...);
  }

  template <typename... ARGS>
  static void VDIV(ARGS... args) {
    phi::dynload::vdDiv(args...);
  }

  template <typename... ARGS>
  static void VEXP(ARGS... args) {
    phi::dynload::vdExp(args...);
  }

  template <typename... ARGS>
  static void VSQUARE(ARGS... args) {
    phi::dynload::vdSqr(args...);
  }

  template <typename... ARGS>
  static void VPOW(ARGS... args) {
    phi::dynload::vdPowx(args...);
  }

  template <typename... ARGS>
  static void VINV(ARGS... args) {
    phi::dynload::vdInv(args...);
  }

  template <typename... ARGS>
  static void VMERF(ARGS... args) {
    phi::dynload::vmdErf(args...);
  }
#if !defined(_WIN32)
  template <typename... ARGS>
  static void CSRMM(ARGS... args) {
    phi::dynload::mkl_dcsrmm(args...);
  }
#endif

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    phi::dynload::cblas_dtrsm(args...);
  }
};

template <>
struct CBlas<phi::dtype::complex<float>> {
  template <typename... ARGS>
  static void AXPY(int n,
                   const phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *X,
                   const int incX,
                   phi::dtype::complex<float> *Y,
                   const int incY) {
    phi::dynload::cblas_caxpy(n, &alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    phi::dynload::cblas_ccopy(args...);
  }

  // the libmklml_intel.so paddle used has no vcAdd, vcSub,
  // vcMul, vcDiv apis before rebuild from source
  // so replace with the raw operator methods
  /*
  template <typename... ARGS>
  static void VADD(ARGS... args) {
    phi::dynload::vcAdd(args...);
  }

  template <typename... ARGS>
  static void VSUB(ARGS... args) {
    phi::dynload::vcSub(args...);
  }

  template <typename... ARGS>
  static void VMUL(ARGS... args) {
    phi::dynload::vcMul(args...);
  }

  template <typename... ARGS>
  static void VDIV(ARGS... args) {
    phi::dynload::vcDiv(args...);
  }
  */

  template <typename... ARGS>
  static void VADD(int n,
                   const phi::dtype::complex<float> *a,
                   const phi::dtype::complex<float> *b,
                   phi::dtype::complex<float> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] + b[i];
    }
  }

  template <typename... ARGS>
  static void VSUB(int n,
                   const phi::dtype::complex<float> *a,
                   const phi::dtype::complex<float> *b,
                   phi::dtype::complex<float> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] - b[i];
    }
  }

  template <typename... ARGS>
  static void VMUL(int n,
                   const phi::dtype::complex<float> *a,
                   const phi::dtype::complex<float> *b,
                   phi::dtype::complex<float> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] * b[i];
    }
  }
  template <typename... ARGS>
  static void VDIV(int n,
                   const phi::dtype::complex<float> *a,
                   const phi::dtype::complex<float> *b,
                   phi::dtype::complex<float> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] / b[i];
    }
  }

  template <typename... ARGS>
  static void GEMV(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans,
                   int M,
                   int N,
                   phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   const phi::dtype::complex<float> *X,
                   int incx,
                   phi::dtype::complex<float> beta,
                   phi::dtype::complex<float> *Y,
                   int incy) {
    const void *a_ = (const void *)(A);
    const void *x_ = (const void *)(X);
    void *y_ = static_cast<void *>(Y);
    phi::dynload::cblas_cgemv(
        layout, trans, M, N, &alpha, a_, lda, x_, incx, &beta, y_, incy);
  }

  template <typename... ARGS>
  static void GEMM(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   int M,
                   int N,
                   int K,
                   phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   const phi::dtype::complex<float> *B,
                   int ldb,
                   phi::dtype::complex<float> beta,
                   phi::dtype::complex<float> *C,
                   int ldc) {
    const void *a_ = (const void *)(A);
    const void *b_ = (const void *)(B);
    void *c_ = static_cast<void *>(C);
    phi::dynload::cblas_cgemm(layout,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              &alpha,
                              a_,
                              lda,
                              b_,
                              ldb,
                              &beta,
                              c_,
                              ldc);
  }

  static void TRSM(CBLAS_LAYOUT layout,
                   CBLAS_SIDE side,
                   CBLAS_UPLO uplo,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_DIAG diag,
                   int M,
                   int N,
                   phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   phi::dtype::complex<float> *B,
                   int ldb) {
    const void *a_ = (const void *)(A);
    void *b_ = static_cast<void *>(B);
    phi::dynload::cblas_ctrsm(
        layout, side, uplo, trans_a, diag, M, N, &alpha, a_, lda, b_, ldb);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE *trans_a,
                         CBLAS_TRANSPOSE *trans_b,
                         int *M,
                         int *N,
                         int *K,
                         phi::dtype::complex<float> *alpha,
                         const phi::dtype::complex<float> **A,
                         const int *lda,
                         const phi::dtype::complex<float> **B,
                         const int *ldb,
                         phi::dtype::complex<float> *beta,
                         phi::dtype::complex<float> **C,
                         const int *ldc,
                         int group_count,
                         int *group_size) {
    const void **A_void = (const void **)(&(*A));
    const void **B_void = (const void **)(&(*B));
    void **C_void = reinterpret_cast<void **>(C);

    phi::dynload::cblas_cgemm_batch(layout,
                                    trans_a,
                                    trans_b,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    A_void,
                                    lda,
                                    B_void,
                                    ldb,
                                    beta,
                                    C_void,
                                    ldc,
                                    group_count,
                                    group_size);
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    phi::dynload::cblas_cgemm_batch(args...);
  }
};

template <>
struct CBlas<phi::dtype::complex<double>> {
  template <typename... ARGS>
  static void AXPY(int n,
                   const phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *X,
                   const int incX,
                   phi::dtype::complex<double> *Y,
                   const int incY) {
    phi::dynload::cblas_zaxpy(n, &alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    phi::dynload::cblas_zcopy(args...);
  }

  // the libmklml_intel.so paddle used has no vzAdd, vzSub,
  // vzMul, vzDiv apis before rebuild from source
  // so replace with the raw operator methods
  /*
  template <typename... ARGS>
  static void VADD(ARGS... args) {
    phi::dynload::vzAdd(args...);
  }

  template <typename... ARGS>
  static void VSUB(ARGS... args) {
    phi::dynload::vzSub(args...);
  }

  template <typename... ARGS>
  static void VMUL(ARGS... args) {
    phi::dynload::vzMul(args...);
  }

  template <typename... ARGS>
  static void VDIV(ARGS... args) {
    phi::dynload::vzDiv(args...);
  }
  */

  template <typename... ARGS>
  static void VADD(int n,
                   const phi::dtype::complex<double> *a,
                   const phi::dtype::complex<double> *b,
                   phi::dtype::complex<double> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] + b[i];
    }
  }

  template <typename... ARGS>
  static void VSUB(int n,
                   const phi::dtype::complex<double> *a,
                   const phi::dtype::complex<double> *b,
                   phi::dtype::complex<double> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] - b[i];
    }
  }

  template <typename... ARGS>
  static void VMUL(int n,
                   const phi::dtype::complex<double> *a,
                   const phi::dtype::complex<double> *b,
                   phi::dtype::complex<double> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] * b[i];
    }
  }
  template <typename... ARGS>
  static void VDIV(int n,
                   const phi::dtype::complex<double> *a,
                   const phi::dtype::complex<double> *b,
                   phi::dtype::complex<double> *y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] / b[i];
    }
  }

  template <typename... ARGS>
  static void GEMV(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans,
                   int M,
                   int N,
                   phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   const phi::dtype::complex<double> *X,
                   int incx,
                   phi::dtype::complex<double> beta,
                   phi::dtype::complex<double> *Y,
                   int incy) {
    const void *a_ = (const void *)(A);
    const void *x_ = (const void *)(X);
    void *y_ = static_cast<void *>(Y);
    phi::dynload::cblas_zgemv(
        layout, trans, M, N, &alpha, a_, lda, x_, incx, &beta, y_, incy);
  }

  template <typename... ARGS>
  static void GEMM(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   int M,
                   int N,
                   int K,
                   phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   const phi::dtype::complex<double> *B,
                   int ldb,
                   phi::dtype::complex<double> beta,
                   phi::dtype::complex<double> *C,
                   int ldc) {
    const void *a_ = (const void *)(A);
    const void *b_ = (const void *)(B);
    void *c_ = static_cast<void *>(C);
    phi::dynload::cblas_zgemm(layout,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              &alpha,
                              a_,
                              lda,
                              b_,
                              ldb,
                              &beta,
                              c_,
                              ldc);
  }

  static void TRSM(CBLAS_LAYOUT layout,
                   CBLAS_SIDE side,
                   CBLAS_UPLO uplo,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_DIAG diag,
                   int M,
                   int N,
                   phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   phi::dtype::complex<double> *B,
                   int ldb) {
    const void *a_ = (const void *)(A);
    void *b_ = static_cast<void *>(B);
    phi::dynload::cblas_ztrsm(
        layout, side, uplo, trans_a, diag, M, N, &alpha, a_, lda, b_, ldb);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE *trans_a,
                         CBLAS_TRANSPOSE *trans_b,
                         int *M,
                         int *N,
                         int *K,
                         phi::dtype::complex<double> *alpha,
                         const phi::dtype::complex<double> **A,
                         const int *lda,
                         const phi::dtype::complex<double> **B,
                         const int *ldb,
                         phi::dtype::complex<double> *beta,
                         phi::dtype::complex<double> **C,
                         const int *ldc,
                         int group_count,
                         int *group_size) {
    const void **A_void = (const void **)(&(*A));
    const void **B_void = (const void **)(&(*B));
    void **C_void = reinterpret_cast<void **>(C);

    phi::dynload::cblas_zgemm_batch(layout,
                                    trans_a,
                                    trans_b,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    A_void,
                                    lda,
                                    B_void,
                                    ldb,
                                    beta,
                                    C_void,
                                    ldc,
                                    group_count,
                                    group_size);
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    phi::dynload::cblas_zgemm_batch(args...);
  }
};

#else

template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    cblas_saxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_scopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_sgemv(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    cblas_strsm(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    cblas_daxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_dcopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_dgemv(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    cblas_dtrsm(args...);
  }
};

template <>
struct CBlas<phi::dtype::complex<float>> {
  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_ccopy(args...);
  }

  template <typename... ARGS>
  static void AXPY(int n,
                   const phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *X,
                   const int incX,
                   phi::dtype::complex<float> *Y,
                   const int incY) {
    cblas_caxpy(n, &alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const int M,
                   const int N,
                   const phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *A,
                   const int lda,
                   const phi::dtype::complex<float> *X,
                   const int incX,
                   const phi::dtype::complex<float> beta,
                   phi::dtype::complex<float> *Y,
                   const int incY) {
    cblas_cgemv(layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
  }

  template <typename... ARGS>
  static void GEMM(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB,
                   const int M,
                   const int N,
                   const int K,
                   const phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *A,
                   const int lda,
                   const phi::dtype::complex<float> *B,
                   const int ldb,
                   const phi::dtype::complex<float> beta,
                   phi::dtype::complex<float> *C,
                   const int ldc) {
    cblas_cgemm(
        layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  static void TRSM(const CBLAS_LAYOUT layout,
                   const CBLAS_SIDE side,
                   const CBLAS_UPLO uplo,
                   const CBLAS_TRANSPOSE transA,
                   const CBLAS_DIAG diag,
                   const int M,
                   const int N,
                   const phi::dtype::complex<float> alpha,
                   const phi::dtype::complex<float> *A,
                   const int lda,
                   phi::dtype::complex<float> *B,
                   const int ldb) {
    cblas_ctrsm(layout, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb);
  }
};

template <>
struct CBlas<phi::dtype::complex<double>> {
  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_zcopy(args...);
  }

  template <typename... ARGS>
  static void AXPY(int n,
                   const phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *X,
                   const int incX,
                   phi::dtype::complex<double> *Y,
                   const int incY) {
    cblas_zaxpy(n, &alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const int M,
                   const int N,
                   const phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *A,
                   const int lda,
                   const phi::dtype::complex<double> *X,
                   const int incX,
                   const phi::dtype::complex<double> beta,
                   phi::dtype::complex<double> *Y,
                   const int incY) {
    cblas_zgemv(layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
  }

  template <typename... ARGS>
  static void GEMM(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB,
                   const int M,
                   const int N,
                   const int K,
                   const phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *A,
                   const int lda,
                   const phi::dtype::complex<double> *B,
                   const int ldb,
                   const phi::dtype::complex<double> beta,
                   phi::dtype::complex<double> *C,
                   const int ldc) {
    cblas_zgemm(
        layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  static void TRSM(const CBLAS_LAYOUT layout,
                   const CBLAS_SIDE side,
                   const CBLAS_UPLO uplo,
                   const CBLAS_TRANSPOSE transA,
                   const CBLAS_DIAG diag,
                   const int M,
                   const int N,
                   const phi::dtype::complex<double> alpha,
                   const phi::dtype::complex<double> *A,
                   const int lda,
                   phi::dtype::complex<double> *B,
                   const int ldb) {
    cblas_ztrsm(layout, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb);
  }
};

#endif

template <>
struct CBlas<phi::dtype::float16> {
  static void GEMM(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 GEMM not supported on CPU, please check your code"));
  }

  static void SMM_GEMM(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 SMM_GEMM not supported on CPU, please check your code"));
  }
  static void VMUL(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 VMUL not supported on CPU, please check your code"));
  }
  static void VEXP(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 VEXP not supported on CPU, please check your code"));
  }
  static void VSQUARE(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 VSQUARE not supported on CPU, please check your code"));
  }
  static void VPOW(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 VPOW not supported on CPU, please check your code"));
  }
  static void DOT(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 DOT not supported on CPU, please check your code"));
  };
  static void SCAL(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 SCAL not supported on CPU, please check your code"));
  };
  static void ASUM(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 ASUM not supported on CPU, please check your code"));
  };
#ifdef PADDLE_WITH_MKLML
  static void GEMM_BATCH(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 GEMM_BATCH not supported on CPU, please check your code"));
  }
#endif
};

#ifdef PADDLE_WITH_MKLML
template <>
template <typename T>
T *Blas<phi::CPUContext>::GEMM_ALLOC(const CBLAS_IDENTIFIER id,
                                     const int M,
                                     const int N,
                                     const int K) const {
  return CBlas<T>::GEMM_ALLOC(id, M, N, K);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMM_PACK(const CBLAS_IDENTIFIER id,
                                      const CBLAS_TRANSPOSE trans,
                                      int M,
                                      int N,
                                      int K,
                                      const T alpha,
                                      const T *src,
                                      const int ld,
                                      T *dst) const {
  CBlas<T>::GEMM_PACK(CblasRowMajor, id, trans, M, N, K, alpha, src, ld, dst);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMM_COMPUTE(int transA,
                                         int transB,
                                         int M,
                                         int N,
                                         int K,
                                         const T *A,
                                         const int lda,
                                         const T *B,
                                         const int ldb,
                                         T beta,
                                         T *C,
                                         const int ldc) const {
  CBlas<T>::GEMM_COMPUTE(
      CblasRowMajor, transA, transB, M, N, K, A, lda, B, ldb, beta, C, ldc);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMM_FREE(T *data) const {
  CBlas<T>::GEMM_FREE(data);
}
#endif

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T, typename U>
void Blas<phi::CPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 U alpha,
                                 const T *A,
                                 const T *B,
                                 U beta,
                                 T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMM(bool transA,
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
  CBlas<T>::GEMM(CblasRowMajor,
                 transA == false ? CblasNoTrans : CblasTrans,
                 transB == false ? CblasNoTrans : CblasTrans,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
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
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const phi::DenseTensor &mat_a,
                                 bool trans_a,
                                 const phi::DenseTensor &mat_b,
                                 bool trans_b,
                                 T alpha,
                                 phi::DenseTensor *mat_out,
                                 T beta) const {
  const auto &dim_a = mat_a.dims();
  const auto &dim_b = mat_b.dims();
  const auto &dim_out = mat_out->dims();
  PADDLE_ENFORCE_EQ(
      dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
      true,
      common::errors::InvalidArgument(
          "The input and output of matmul should be matrix, the dim size must "
          "be 2,"
          "but received dim size input_a:%d, input_b:%d, output:%d",
          dim_a.size(),
          dim_b.size(),
          dim_out.size()));
  PADDLE_ENFORCE_EQ(
      mat_a.place() == mat_b.place() && mat_a.place() == mat_out->place(),
      true,
      common::errors::InvalidArgument("The places of matrices in the matmul "
                                      "should be same, please check your "
                                      "code."));

  int M = dim_out[0];
  int N = dim_out[1];
  int K = !trans_a ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !trans_b ? CblasNoTrans : CblasTrans;

  this->GEMM(transA,
             transB,
             M,
             N,
             K,
             alpha,
             mat_a.data<T>(),
             mat_b.data<T>(),
             beta,
             mat_out->data<T>());
}

template <>
template <typename T>
void Blas<phi::CPUContext>::AXPY(int n, T alpha, const T *x, T *y) const {
  CBlas<T>::AXPY(n, alpha, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VCOPY(int n, const T *x, T *y) const {
  CBlas<T>::VCOPY(n, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VADD(int n, const T *x, const T *y, T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VADD(n, x, y, z);
#else
  if (x == z) {
    this->template AXPY<T>(n, (T)(1.), y, z);
  } else {
    this->template VCOPY<T>(n, y, z);
    this->template AXPY<T>(n, (T)(1.), x, z);
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VSUB(int n, const T *x, const T *y, T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VSUB(n, x, y, z);
#else
  // try to find if openblas support vsub
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VMUL(int n, const T *x, const T *y, T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VMUL(n, x, y, z);
#else
  // try to find if openblas support vmul
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VDIV(int n, const T *x, const T *y, T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VDIV(n, x, y, z);
#else
  // try to find if openblas support vdiv
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] / y[i];
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VEXP(int n, const T *x, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VEXP(n, x, y);
#else
  // try to find if openblas support vexp
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VSQUARE(int n, const T *x, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VSQUARE(n, x, y);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] * x[i];
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VPOW(int n, const T *x, T a, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VPOW(n, x, a, y);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = std::pow(x[i], a);
  }
#endif
}

template <>
template <typename T>
T Blas<phi::CPUContext>::DOT(int n, const T *x, const T *y) const {
#ifdef PADDLE_WITH_MKLML
  return CBlas<T>::DOT(n, x, 1, y, 1);
#else
  // try to find if openblas support cblas_dot
  T sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::SCAL(int n, const T a, T *x) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::SCAL(n, a, x, 1);
#else
  // try to find if openblas support cblas_scal
  for (int i = 0; i < n; ++i) {
    x[i] = a * x[i];
  }
#endif
}

template <>
template <typename T>
T Blas<phi::CPUContext>::ASUM(int n, T *x, int inc) const {
  auto sum = static_cast<T>(0.0);
#ifdef PADDLE_WITH_MKLML
  sum = CBlas<T>::ASUM(n, x, inc);
#else
  // TODO(jczaja): check if openblas does provide cblas_sasum/cblas_dasum
  for (int c = 0; c < n; ++c) {
    sum += x[c];
  }
#endif
  return sum;
}

template <>
template <typename T>
void Blas<phi::CPUContext>::GEMV(bool trans_a,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBlas<T>::GEMV(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  PADDLE_ENFORCE_NOT_NULL(
      A, common::errors::InvalidArgument("Pointer A should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      B, common::errors::InvalidArgument("Pointer B should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      C, common::errors::InvalidArgument("Pointer C should not be null."));
#ifdef PADDLE_WITH_MKLML
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }

  CBlas<T>::GEMM_BATCH(CblasRowMajor,
                       &transA,
                       &transB,
                       &M,
                       &N,
                       &K,
                       &alpha,
                       a_array.data(),
                       &lda,
                       b_array.data(),
                       &ldb,
                       &beta,
                       c_array.data(),
                       &ldc,
                       1 /* group_count */,
                       &batchCount);
#else
  for (int k = 0; k < batchCount; ++k) {
    auto *Ak = &A[k * strideA];
    auto *Bk = &B[k * strideB];
    auto *Ck = &C[k * M * N];
    this->template GEMM<T>(transA, transB, M, N, K, alpha, Ak, Bk, beta, Ck);
  }
#endif
}

template <>
template <typename T, typename U>
void Blas<phi::CPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  PADDLE_ENFORCE_NOT_NULL(
      A, common::errors::InvalidArgument("Pointer A should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      B, common::errors::InvalidArgument("Pointer B should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      C, common::errors::InvalidArgument("Pointer C should not be null."));
#ifdef PADDLE_WITH_MKLML
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }

  CBlas<T>::GEMM_BATCH(CblasRowMajor,
                       &transA,
                       &transB,
                       &M,
                       &N,
                       &K,
                       &alpha,
                       a_array.data(),
                       &lda,
                       b_array.data(),
                       &ldb,
                       &beta,
                       c_array.data(),
                       &ldc,
                       1 /* group_count */,
                       &batchCount);
#else
  for (int k = 0; k < batchCount; ++k) {
    auto *Ak = &A[k * strideA];
    auto *Bk = &B[k * strideB];
    auto *Ck = &C[k * M * N];
    this->template GEMM<T>(transA, transB, M, N, K, alpha, Ak, Bk, beta, Ck);
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
#ifdef PADDLE_WITH_MKLML
  const int lda = (std::max)((transA == CblasNoTrans) ? K : M, 1);
  const int ldb = (std::max)((transB == CblasNoTrans) ? N : K, 1);
  const int ldc = (std::max)(N, 1);
  CBlas<T>::GEMM_BATCH(CblasRowMajor,
                       &transA,
                       &transB,
                       &M,
                       &N,
                       &K,
                       &alpha,
                       A,
                       &lda,
                       B,
                       &ldb,
                       &beta,
                       C,
                       &ldc,
                       1 /* group_count */,
                       &batchCount);
#else
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
#endif
}

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)  // @{ Group Blas MKLML: BatchedGEMMWithHead
template <>
template <typename T>
void Blas<phi::CPUContext>::BatchedGEMMWithHead(CBLAS_TRANSPOSE transA,
                                                CBLAS_TRANSPOSE transB,
                                                int W1,
                                                int H1,
                                                int W2,
                                                int H2,
                                                T alpha,
                                                const T *A,
                                                const T *B,
                                                T beta,
                                                T *C,
                                                int batchCount,
                                                int64_t strideA,
                                                int64_t strideB,
                                                int64_t head_number,
                                                bool split_b_vertical) const {
  int lda = (transA == CblasNoTrans) ? W1 : H1;
  int ldb = (transB == CblasNoTrans) ? W2 : H2;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);

  if (split_b_vertical) {
    int ldc = W2;
    int sub_width = W2 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W2 / head_number)
                                : i * (W2 / head_number) * H2;
      int sub_matC_offset = i * W2 / head_number;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor,
                           &transA,
                           &transB,
                           &H1,
                           &sub_width,
                           &H2,
                           &alpha,
                           a_array.data(),
                           &lda,
                           b_array.data(),
                           &ldb,
                           &beta,
                           c_array.data(),
                           &ldc,
                           1 /* group_count */,
                           &batchCount);
    }

  } else {
    PADDLE_ENFORCE_EQ(
        W1,
        H2,
        common::errors::InvalidArgument(
            "The first matrix width should be same as second matrix height,"
            "but received first matrix width %d"
            ", second matrix height %d",
            W1,
            H2));
    int ldc = W2 * head_number;
    int sub_width = W1 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W1 / head_number) * W2
                                : i * (W1 / head_number);
      int sub_matC_offset = i * W2;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * head_number * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor,
                           &transA,
                           &transB,
                           &H1,
                           &W2,
                           &sub_width,
                           &alpha,
                           a_array.data(),
                           &lda,
                           b_array.data(),
                           &ldb,
                           &beta,
                           c_array.data(),
                           &ldc,
                           1 /* group_count */,
                           &batchCount);
    }
  }
}
#endif  // @} End Group Blas MKLML: BatchedGEMMWithHead

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(
    const int M, const int N, const int K, const T *A, const T *B, T *C) const {
  this->template GEMM<T>(CblasRowMajor,
                         CblasNoTrans,
                         CblasNoTrans,
                         M,
                         N,
                         K,
                         static_cast<T>(1),
                         A,
                         K,
                         B,
                         N,
                         static_cast<T>(0),
                         C,
                         N);
}

template <>
template <typename T>
void Blas<phi::CPUContext>::MatMul(
    const int M, const int N, const int K, const T *A, const T *B, T *C) const {
#ifdef PADDLE_WITH_LIBXSMM
  // Refer to https://github.com/hfp/libxsmm/blob/master/README.md
  // But the threshold is custom constexpr int LIBXSMM_THRESHOLD = 20 * 20 * 20;

  // Since the matrix is very small,
  // so the unit of calculation is already very fast,
  // and the if( M*N*K < LIBXSMM_THRESHOLD) would be overhead,
  // use xsmm directly.
  // Note: SMM use ColMajor
  const char transa = 'N';
  const char transb = 'N';
  const T alpha = static_cast<T>(1);
  const T beta = static_cast<T>(0);
  CBlas<T>::SMM_GEMM(
      &transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
  return;
#endif

  CBlas<T>::GEMM(CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 M,
                 N,
                 K,
                 static_cast<T>(1),
                 A,
                 K,
                 B,
                 N,
                 static_cast<T>(0),
                 C,
                 N);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const phi::DenseTensor &mat_a,
                                 const MatDescriptor &dim_a,
                                 const phi::DenseTensor &mat_b,
                                 const MatDescriptor &dim_b,
                                 T alpha,
                                 phi::DenseTensor *mat_out,
                                 T beta) const {
  MatMul(mat_a.data<T>(),
         dim_a,
         mat_b.data<T>(),
         dim_b,
         alpha,
         mat_out->data<T>(),
         beta);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const T *mat_a,
                                 const MatDescriptor &dim_a,
                                 const T *mat_b,
                                 const MatDescriptor &dim_b,
                                 T alpha,
                                 T *mat_out,
                                 T beta) const {
  PADDLE_ENFORCE_EQ(
      dim_a.width_,
      dim_b.height_,
      common::errors::InvalidArgument(
          "The first matrix width should be same as second matrix height,"
          "but received first matrix width %d"
          ", second matrix height %d",
          dim_a.width_,
          dim_b.height_));

  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;
  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    this->template GEMM<T>(transA,
                           transB,
                           dim_a.height_,
                           dim_b.width_,
                           dim_a.width_,
                           alpha,
                           mat_a,
                           mat_b,
                           beta,
                           mat_out);
  } else {
    PADDLE_ENFORCE_EQ(
        dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
            dim_b.batch_size_ == 0,
        true,
        common::errors::InvalidArgument(
            "dim_a.batch_size should be equal to dim_b.batch_size, or "
            "one of dim_a.batch_size and dim_b.batch_size should be 0. "
            "But got dim_a.batch_size = %d, dim_b.batch_size = %d.",
            dim_a.batch_size_,
            dim_b.batch_size_));
    this->template BatchedGEMM<T>(
        transA,
        transB,
        dim_a.height_,
        dim_b.width_,
        dim_a.width_,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_out,
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_);
  }
}

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
// @{ Group Blas MKLML: MatMulWithHead
/*
 * Multiple two matrixes with multiple heads
 *
 * A new parameter, i.e head_number is added compared to normal MatMul.
 * The head_number describes the number of heads a matrix is vertically
 * split.
 *
 * When user calls this API, the multiplication of two big matrixes is split
 * into multiplication of several (head_number_) small matrixes. e.g. if Mat A
 * is [3, 24] and Mat B is [24, 4], when multiple A and B with head_number as
 * 4, Mat A will be split as 4 matrix of [3, 6] and Mat B will be
 * (horizontally) split as 4 matrix of [6, 4]. The result of final matrix
 * will be 4 matrix of [3, 4], i.e. [3, 16].
 * Another example is A is [3, 8], B is [2, 16], head_number is 4. In this
 * case, A will be split as [3, 2], B will be (vertically) split as
 * [2, 4]. The final result will be 4 matrix of 4 matrix of [3,4], i.e. [3, 16]
 */
template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMulWithHead(const phi::DenseTensor &mat_a,
                                         const MatDescriptor &dim_a,
                                         const phi::DenseTensor &mat_b,
                                         const MatDescriptor &dim_b,
                                         T alpha,
                                         int head_number,
                                         phi::DenseTensor *mat_out,
                                         T beta,
                                         bool mat_b_split_vertical) const {
  PADDLE_ENFORCE_EQ(
      dim_a.width_ % head_number,
      0,
      common::errors::InvalidArgument(
          "The first input width must be some times the head number"
          "but received first input width %d"
          ",  head_number %d",
          dim_a.width_,
          head_number));
  PADDLE_ENFORCE_GE(head_number,
                    1,
                    common::errors::InvalidArgument(
                        "The head number should be greater equal 1,"
                        "but received head number %d",
                        head_number));
  PADDLE_ENFORCE_LE(
      head_number,
      dim_a.width_,
      common::errors::InvalidArgument(
          "The head number should be less equal first input width,"
          "but received first input width %d"
          ",  head_number %d",
          dim_a.width_,
          head_number));
  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;

  if (mat_b_split_vertical) {
    PADDLE_ENFORCE_EQ(
        dim_b.height_,
        dim_a.width_ / head_number,
        common::errors::InvalidArgument(
            "The second input height should be equal than first input width,"
            "but received second input height %d, first input width %d",
            dim_b.height_,
            dim_a.width_ / head_number));
    PADDLE_ENFORCE_EQ(
        dim_a.width_ % head_number,
        0,
        common::errors::InvalidArgument(
            "The second input width should be some times the head number"
            "but received second input width %d"
            ",  head_number %d",
            dim_b.width_,
            head_number));
  }

  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    int lda = !dim_a.trans_ ? dim_a.width_ : dim_a.height_;
    int ldb = !dim_b.trans_ ? dim_b.width_ : dim_b.height_;
    int sub_matA_offset;
    int sub_matB_offset;
    int sub_matC_offset;
    int sub_mat_M = dim_a.height_;
    int sub_mat_N;
    int sub_mat_K;
    int ldc;

    for (int i = 0; i < head_number; i++) {
      sub_matA_offset = dim_a.trans_
                            ? i * (dim_a.width_ / head_number) * dim_a.height_
                            : i * (dim_a.width_ / head_number);
      if (mat_b_split_vertical) {
        sub_matB_offset = dim_b.trans_
                              ? i * (dim_b.width_ / head_number) * dim_b.height_
                              : i * (dim_b.width_ / head_number);
        sub_matC_offset = i * dim_b.width_ / head_number;

        sub_mat_N = dim_b.width_ / head_number;
        sub_mat_K = dim_b.height_;

        ldc = dim_b.width_;
      } else {
        sub_matB_offset =
            dim_b.trans_ ? i * (dim_b.height_ / head_number)
                         : i * (dim_b.height_ / head_number) * dim_b.width_;
        sub_matC_offset = i * dim_b.width_;

        sub_mat_N = dim_b.width_;
        sub_mat_K = dim_a.width_ / head_number;

        ldc = head_number * dim_b.width_;
      }

      this->template GEMM<T>(transA,
                             transB,
                             sub_mat_M,
                             sub_mat_N,
                             sub_mat_K,
                             alpha,
                             mat_a.data<T>() + sub_matA_offset,
                             lda,
                             mat_b.data<T>() + sub_matB_offset,
                             ldb,
                             beta,
                             mat_out->data<T>() + sub_matC_offset,
                             ldc);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        (dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
         dim_b.batch_size_ == 0),
        true,
        common::errors::InvalidArgument(
            "The first input batch size should be equal than second input,"
            "either two input batch size is 0, but received first input batch "
            "size"
            " %d, second input batch size %d",
            dim_a.batch_size_,
            dim_b.batch_size_));

    this->template BatchedGEMMWithHead<T>(
        transA,
        transB,
        dim_a.width_,
        dim_a.height_,
        dim_b.width_,
        dim_b.height_,
        alpha,
        mat_a.data<T>(),
        mat_b.data<T>(),
        beta,
        mat_out->data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_,
        head_number,
        mat_b_split_vertical);
  }
}
#endif  // @} End Group Blas MKLML: MatMulWithHead

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::VINV(int n, const T *a, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VINV(n, a, y);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = 1.0 / a[i];
  }
#endif
}

template <>
template <typename T>
void Blas<phi::CPUContext>::VMERF(int n, const T *a, T *y, int64_t mode) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VMERF(n, a, y, mode);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = std::erf(a[i]);
  }
#endif
}

#ifdef PADDLE_WITH_MKLML
template <>
template <typename T>
void Blas<phi::CPUContext>::CSRMM(const char *transa,
                                  const int *m,
                                  const int *n,
                                  const int *k,
                                  const T *alpha,
                                  const char *matdescra,
                                  const T *val,
                                  const int *indx,
                                  const int *pntrb,
                                  const int *pntre,
                                  const T *b,
                                  const int *ldb,
                                  const T *beta,
                                  T *c,
                                  const int *ldc) const {
  CBlas<T>::CSRMM(transa,
                  m,
                  n,
                  k,
                  alpha,
                  matdescra,
                  val,
                  indx,
                  pntrb,
                  pntre,
                  b,
                  ldb,
                  beta,
                  c,
                  ldc);
}
#endif

template <>
template <typename T>
void Blas<phi::CPUContext>::TRSM(CBLAS_SIDE side,
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
  CBlas<T>::TRSM(
      CblasRowMajor, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb);
}

}  // namespace funcs
}  // namespace phi
