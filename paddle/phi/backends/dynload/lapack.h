/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <complex>
#include <mutex>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

// Because lapack doesn't provide appropriate header file,
// we should expose API statement yourself.

// getrf_(For example)
extern "C" void dgetrf_(
    int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(
    int *m, int *n, float *a, int *lda, int *ipiv, int *info);

// getrs_
extern "C" void sgetrs_(char *trans,
                        int *n,
                        int *nrhs,
                        float *a,
                        int *lda,
                        int *ipiv,
                        float *b,
                        int *ldb,
                        int *info);
extern "C" void dgetrs_(char *trans,
                        int *n,
                        int *nrhs,
                        double *a,
                        int *lda,
                        int *ipiv,
                        double *b,
                        int *ldb,
                        int *info);

// evd
extern "C" void zheevd_(char *jobz,
                        char *uplo,
                        int *n,
                        std::complex<double> *a,
                        int *lda,
                        double *w,
                        std::complex<double> *work,
                        int *lwork,
                        double *rwork,
                        int *lrwork,
                        int *iwork,
                        int *liwork,
                        int *info);
extern "C" void cheevd_(char *jobz,
                        char *uplo,
                        int *n,
                        std::complex<float> *a,
                        int *lda,
                        float *w,
                        std::complex<float> *work,
                        int *lwork,
                        float *rwork,
                        int *lrwork,
                        int *iwork,
                        int *liwork,
                        int *info);
extern "C" void dsyevd_(char *jobz,
                        char *uplo,
                        int *n,
                        double *a,
                        int *lda,
                        double *w,
                        double *work,
                        int *lwork,
                        int *iwork,
                        int *liwork,
                        int *info);
extern "C" void ssyevd_(char *jobz,
                        char *uplo,
                        int *n,
                        float *a,
                        int *lda,
                        float *w,
                        float *work,
                        int *lwork,
                        int *iwork,
                        int *liwork,
                        int *info);

// geev
extern "C" void dgeev_(char *jobvl,
                       char *jobvr,
                       int *n,
                       double *a,
                       int *lda,
                       double *wr,
                       double *wi,
                       double *vl,
                       int *ldvl,
                       double *vr,
                       int *ldvr,
                       double *work,
                       int *lwork,
                       int *info);
extern "C" void sgeev_(char *jobvl,
                       char *jobvr,
                       int *n,
                       float *a,
                       int *lda,
                       float *wr,
                       float *wi,
                       float *vl,
                       int *ldvl,
                       float *vr,
                       int *ldvr,
                       float *work,
                       int *lwork,
                       int *info);
extern "C" void zgeev_(char *jobvl,
                       char *jobvr,
                       int *n,
                       std::complex<double> *a,
                       int *lda,
                       std::complex<double> *w,
                       std::complex<double> *vl,
                       int *ldvl,
                       std::complex<double> *vr,
                       int *ldvr,
                       std::complex<double> *work,
                       int *lwork,
                       double *rwork,
                       int *info);
extern "C" void cgeev_(char *jobvl,
                       char *jobvr,
                       int *n,
                       std::complex<float> *a,
                       int *lda,
                       std::complex<float> *w,
                       std::complex<float> *vl,
                       int *ldvl,
                       std::complex<float> *vr,
                       int *ldvr,
                       std::complex<float> *work,
                       int *lwork,
                       float *rwork,
                       int *info);

// gels
extern "C" void dgels_(char *trans,
                       int *m,
                       int *n,
                       int *nrhs,
                       double *a,
                       int *lda,
                       double *b,
                       int *ldb,
                       double *work,
                       int *lwork,
                       int *info);
extern "C" void sgels_(char *trans,
                       int *m,
                       int *n,
                       int *nrhs,
                       float *a,
                       int *lda,
                       float *b,
                       int *ldb,
                       float *work,
                       int *lwork,
                       int *info);

// gelsd
extern "C" void dgelsd_(int *m,
                        int *n,
                        int *nrhs,
                        double *a,
                        int *lda,
                        double *b,
                        int *ldb,
                        double *s,
                        double *rcond,
                        int *rank,
                        double *work,
                        int *lwork,
                        int *iwork,
                        int *info);
extern "C" void sgelsd_(int *m,
                        int *n,
                        int *nrhs,
                        float *a,
                        int *lda,
                        float *b,
                        int *ldb,
                        float *s,
                        float *rcond,
                        int *rank,
                        float *work,
                        int *lwork,
                        int *iwork,
                        int *info);

// gelsy
extern "C" void dgelsy_(int *m,
                        int *n,
                        int *nrhs,
                        double *a,
                        int *lda,
                        double *b,
                        int *ldb,
                        int *jpvt,
                        double *rcond,
                        int *rank,
                        double *work,
                        int *lwork,
                        int *info);
extern "C" void sgelsy_(int *m,
                        int *n,
                        int *nrhs,
                        float *a,
                        int *lda,
                        float *b,
                        int *ldb,
                        int *jpvt,
                        float *rcond,
                        int *rank,
                        float *work,
                        int *lwork,
                        int *info);

// gelss
extern "C" void dgelss_(int *m,
                        int *n,
                        int *nrhs,
                        double *a,
                        int *lda,
                        double *b,
                        int *ldb,
                        double *s,
                        double *rcond,
                        int *rank,
                        double *work,
                        int *lwork,
                        int *info);
extern "C" void sgelss_(int *m,
                        int *n,
                        int *nrhs,
                        float *a,
                        int *lda,
                        float *b,
                        int *ldb,
                        float *s,
                        float *rcond,
                        int *rank,
                        float *work,
                        int *lwork,
                        int *info);

extern "C" void zpotrs_(char *uplo,
                        int *n,
                        int *nrhs,
                        std::complex<double> *a,
                        int *lda,
                        std::complex<double> *b,
                        int *ldb,
                        int *info);
extern "C" void cpotrs_(char *uplo,
                        int *n,
                        int *nrhs,
                        std::complex<float> *a,
                        int *lda,
                        std::complex<float> *b,
                        int *ldb,
                        int *info);
extern "C" void dpotrs_(char *uplo,
                        int *n,
                        int *nrhs,
                        double *a,
                        int *lda,
                        double *b,
                        int *ldb,
                        int *info);
extern "C" void spotrs_(char *uplo,
                        int *n,
                        int *nrhs,
                        float *a,
                        int *lda,
                        float *b,
                        int *ldb,
                        int *info);
extern "C" void dgesdd_(char *,
                        int *,
                        int *,
                        double *,
                        int *,
                        double *,
                        double *,
                        int *,
                        double *,
                        int *,
                        double *,
                        int *,
                        int *,
                        int *);
extern "C" void sgesdd_(char *,
                        int *,
                        int *,
                        float *,
                        int *,
                        float *,
                        float *,
                        int *,
                        float *,
                        int *,
                        float *,
                        int *,
                        int *,
                        int *);

namespace phi {
namespace dynload {

extern std::once_flag lapack_dso_flag;
extern void *lapack_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load lapack routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_LAPACK_WRAP(__name)                             \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using lapackFunc = decltype(&::__name);                        \
      std::call_once(lapack_dso_flag, []() {                         \
        lapack_dso_handle = phi::dynload::GetLAPACKDsoHandle();      \
      });                                                            \
      static void *p_##_name = dlsym(lapack_dso_handle, #__name);    \
      return reinterpret_cast<lapackFunc>(p_##_name)(args...);       \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_LAPACK_WRAP(__name) \
  DYNAMIC_LOAD_LAPACK_WRAP(__name)

#define LAPACK_ROUTINE_EACH(__macro) \
  __macro(dgetrf_);                  \
  __macro(sgetrf_);                  \
  __macro(sgetrs_);                  \
  __macro(dgetrs_);                  \
  __macro(zheevd_);                  \
  __macro(cheevd_);                  \
  __macro(dsyevd_);                  \
  __macro(ssyevd_);                  \
  __macro(dgeev_);                   \
  __macro(sgeev_);                   \
  __macro(zgeev_);                   \
  __macro(cgeev_);                   \
  __macro(dgels_);                   \
  __macro(sgels_);                   \
  __macro(dgelsd_);                  \
  __macro(sgelsd_);                  \
  __macro(dgelsy_);                  \
  __macro(sgelsy_);                  \
  __macro(dgelss_);                  \
  __macro(sgelss_);                  \
  __macro(sgesdd_);                  \
  __macro(dgesdd_);                  \
  __macro(zpotrs_);                  \
  __macro(cpotrs_);                  \
  __macro(dpotrs_);                  \
  __macro(spotrs_);

LAPACK_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_LAPACK_WRAP);

#undef DYNAMIC_LOAD_LAPACK_WRAP

}  // namespace dynload
}  // namespace phi
