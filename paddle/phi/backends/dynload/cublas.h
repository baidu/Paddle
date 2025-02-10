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

#include <cublasXt.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag cublas_dso_flag;
extern void *cublas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)                            \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using cublas_func =                                                   \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);         \
      std::call_once(cublas_dso_flag, []() {                                \
        cublas_dso_handle = phi::dynload::GetCublasDsoHandle();             \
      });                                                                   \
      static void *p_##__name = dlsym(cublas_dso_handle, #__name);          \
      return reinterpret_cast<cublas_func>(p_##__name)(args...);            \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSaxpy_v2);                \
  __macro(cublasDaxpy_v2);                \
  __macro(cublasCaxpy_v2);                \
  __macro(cublasZaxpy_v2);                \
  __macro(cublasSscal_v2);                \
  __macro(cublasDscal_v2);                \
  __macro(cublasScopy_v2);                \
  __macro(cublasDcopy_v2);                \
  __macro(cublasSgemv_v2);                \
  __macro(cublasDgemv_v2);                \
  __macro(cublasCgemv_v2);                \
  __macro(cublasZgemv_v2);                \
  __macro(cublasSgemm_v2);                \
  __macro(cublasDgemm_v2);                \
  __macro(cublasCgemm_v2);                \
  __macro(cublasZgemm_v2);                \
  __macro(cublasHgemm);                   \
  __macro(cublasSgemmEx);                 \
  __macro(cublasSgeam);                   \
  __macro(cublasDgeam);                   \
  __macro(cublasStrsm_v2);                \
  __macro(cublasDtrsm_v2);                \
  __macro(cublasCtrsm_v2);                \
  __macro(cublasZtrsm_v2);                \
  __macro(cublasCreate_v2);               \
  __macro(cublasDestroy_v2);              \
  __macro(cublasSetStream_v2);            \
  __macro(cublasSetPointerMode_v2);       \
  __macro(cublasGetPointerMode_v2);       \
  __macro(cublasSgemmBatched);            \
  __macro(cublasDgemmBatched);            \
  __macro(cublasCgemmBatched);            \
  __macro(cublasZgemmBatched);            \
  __macro(cublasStrsmBatched);            \
  __macro(cublasDtrsmBatched);            \
  __macro(cublasCtrsmBatched);            \
  __macro(cublasZtrsmBatched);            \
  __macro(cublasSgetrfBatched);           \
  __macro(cublasSgetriBatched);           \
  __macro(cublasDgetrfBatched);           \
  __macro(cublasDgetriBatched);           \
  __macro(cublasCgetrfBatched);           \
  __macro(cublasCgetriBatched);           \
  __macro(cublasZgetrfBatched);           \
  __macro(cublasZgetriBatched);           \
  __macro(cublasSmatinvBatched);          \
  __macro(cublasDmatinvBatched);          \
  __macro(cublasCmatinvBatched);          \
  __macro(cublasZmatinvBatched);          \
  __macro(cublasSgetrsBatched);           \
  __macro(cublasSetSmCountTarget);           \
  __macro(cublasDgetrsBatched);

CUBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

// APIs available after CUDA 8.0
#if CUDA_VERSION >= 8000
#define CUBLAS_BLAS_ROUTINE_EACH_R2(__macro) \
  __macro(cublasGemmEx);                     \
  __macro(cublasSgemmStridedBatched);        \
  __macro(cublasDgemmStridedBatched);        \
  __macro(cublasCgemmStridedBatched);        \
  __macro(cublasZgemmStridedBatched);        \
  __macro(cublasHgemmStridedBatched);

CUBLAS_BLAS_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)
#endif

// APIs available after CUDA 9.0
#if CUDA_VERSION >= 9000
#define CUBLAS_BLAS_ROUTINE_EACH_R3(__macro) \
  __macro(cublasSetMathMode);                \
  __macro(cublasGetMathMode);

CUBLAS_BLAS_ROUTINE_EACH_R3(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)
#endif

// APIs available after CUDA 9.1
#if CUDA_VERSION >= 9010
#define CUBLAS_BLAS_ROUTINE_EACH_R4(__macro) \
  __macro(cublasGemmBatchedEx);              \
  __macro(cublasGemmStridedBatchedEx);

CUBLAS_BLAS_ROUTINE_EACH_R4(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)
#endif

#undef DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP
}  // namespace dynload
}  // namespace phi
