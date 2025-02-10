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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/lu_solve_kernel_impl.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cusolver.h"
#endif

namespace phi {

#if defined(PADDLE_WITH_CUDA)
template <typename T>
class LUSolveFunctor<T, GPUContext> {
 public:
  void operator()(const GPUContext& dev_ctx,
                  bool trans,
                  int M,
                  int N,
                  T* Adata,
                  int lda,
                  const int* ipiv,
                  T* Bdata,
                  int* devInfo) {
    auto handle = dev_ctx.cusolver_dn_handle();

    // Workspace sizes differ for different data types
    int workspace_size;
    if (std::is_same<T, float>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cusolverDnSgetrs(handle,
                                         trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                         M,
                                         N,
                                         reinterpret_cast<const float*>(Adata),
                                         lda,
                                         ipiv,
                                         reinterpret_cast<float*>(Bdata),
                                         lda,
                                         devInfo));
    } else if (std::is_same<T, double>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cusolverDnDgetrs(handle,
                                         trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                         M,
                                         N,
                                         reinterpret_cast<const double*>(Adata),
                                         lda,
                                         ipiv,
                                         reinterpret_cast<double*>(Bdata),
                                         lda,
                                         devInfo));
    }
  }
};
#endif

}  // namespace phi

#if defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(
    lu_solve, GPU, ALL_LAYOUT, phi::LUSolveKernel, float, double) {}
#endif
