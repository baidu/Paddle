// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/impl/lu_solve_kernel_impl.h"

namespace phi {

template <typename T>
class LUSolveFunctor<T, CPUContext> {
 public:
  void operator()(const CPUContext& dev_ctx,
                  bool trans,
                  int M,
                  int N,
                  T* Adata,
                  int lda,
                  const int* ipiv,
                  T* Bdata,
                  int* devInfo) {
    char trans_opt = trans ? 'T' : 'N';
    funcs::lapackGetrs<T>(
        trans_opt, M, N, Adata, lda, ipiv, Bdata, lda, devInfo);
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(
    lu_solve, CPU, ALL_LAYOUT, phi::LUSolveKernel, float, double) {}
