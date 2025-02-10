//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace funcs {

// LU (for example)
template <typename T>
void lapackLu(int m, int n, T *a, int lda, int *ipiv, int *info);

// Lu_solve
template <typename T>
void lapackLuSolve(char trans,
                   int n,
                   int nrhs,
                   T *a,
                   int lda,
                   int *ipiv,
                   T *b,
                   int ldb,
                   int *info);

// Eigh
template <typename T, typename ValueType = T>
void lapackEigh(char jobz,
                char uplo,
                int n,
                T *a,
                int lda,
                ValueType *w,
                T *work,
                int lwork,
                ValueType *rwork,
                int lrwork,
                int *iwork,
                int liwork,
                int *info);

// Eig
template <typename T1, typename T2 = T1>
void lapackEig(char jobvl,
               char jobvr,
               int n,
               T1 *a,
               int lda,
               T1 *w,
               T1 *vl,
               int ldvl,
               T1 *vr,
               int ldvr,
               T1 *work,
               int lwork,
               T2 *rwork,
               int *info);

// Gels
template <typename T>
void lapackGels(char trans,
                int m,
                int n,
                int nrhs,
                T *a,
                int lda,
                T *b,
                int ldb,
                T *work,
                int lwork,
                int *info);

// Gelsd
template <typename T1, typename T2>
void lapackGelsd(int m,
                 int n,
                 int nrhs,
                 T1 *a,
                 int lda,
                 T1 *b,
                 int ldb,
                 T2 *s,
                 T2 rcond,
                 int *rank,
                 T1 *work,
                 int lwork,
                 T2 *rwork,
                 int *iwork,
                 int *info);

// Gelsy
template <typename T1, typename T2>
void lapackGelsy(int m,
                 int n,
                 int nrhs,
                 T1 *a,
                 int lda,
                 T1 *b,
                 int ldb,
                 int *jpvt,
                 T2 rcond,
                 int *rank,
                 T1 *work,
                 int lwork,
                 T2 *rwork,
                 int *info);

// Gelss
template <typename T1, typename T2>
void lapackGelss(int m,
                 int n,
                 int nrhs,
                 T1 *a,
                 int lda,
                 T1 *b,
                 int ldb,
                 T2 *s,
                 T2 rcond,
                 int *rank,
                 T1 *work,
                 int lwork,
                 T2 *rwork,
                 int *info);

template <typename T>
void lapackSvd(char jobz,
               int m,
               int n,
               T *a,
               int lda,
               T *s,
               T *u,
               int ldu,
               T *vt,
               int ldvt,
               T *work,
               int lwork,
               int *iwork,
               int *info);

template <typename T>
void lapackCholeskySolve(
    char uplo, int n, int nrhs, T *a, int lda, T *b, int ldb, int *info);

}  // namespace funcs
}  // namespace phi
