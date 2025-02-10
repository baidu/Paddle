/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag hipblasLt_dso_flag;
extern void *hipblasLt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load hipblasLt routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_HIPBLASLT_WRAP(__name)                         \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using hipblasLt_func =                                                \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);         \
      std::call_once(hipblasLt_dso_flag, []() {                             \
        hipblasLt_dso_handle = phi::dynload::GetCublasLtDsoHandle();        \
      });                                                                   \
      static void *p_##__name = dlsym(hipblasLt_dso_handle, #__name);       \
      return reinterpret_cast<hipblasLt_func>(p_##__name)(args...);         \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

#define HIPBLASLT_BLAS_ROUTINE_EACH(__macro)      \
  __macro(hipblasLtCreate);                       \
  __macro(hipblasLtDestroy);                      \
  __macro(hipblasLtMatmul);                       \
  __macro(hipblasLtMatmulDescCreate);             \
  __macro(hipblasLtMatmulDescDestroy);            \
  __macro(hipblasLtMatmulDescSetAttribute);       \
  __macro(hipblasLtMatmulDescGetAttribute);       \
  __macro(hipblasLtMatrixLayoutCreate);           \
  __macro(hipblasLtMatrixLayoutDestroy);          \
  __macro(hipblasLtMatrixLayoutSetAttribute);     \
  __macro(hipblasLtMatrixLayoutGetAttribute);     \
  __macro(hipblasLtMatmulPreferenceCreate);       \
  __macro(hipblasLtMatmulPreferenceDestroy);      \
  __macro(hipblasLtMatmulPreferenceSetAttribute); \
  __macro(hipblasLtMatmulPreferenceGetAttribute); \
  __macro(hipblasLtMatmulAlgoGetHeuristic);       \
  __macro(hipblasLtMatrixTransform);              \
  __macro(hipblasLtMatrixTransformDescCreate);    \
  __macro(hipblasLtMatrixTransformDescDestroy);   \
  __macro(hipblasLtMatrixTransformDescSetAttribute);

HIPBLASLT_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPBLASLT_WRAP)

#undef DECLARE_DYNAMIC_LOAD_HIPBLASLT_WRAP
}  // namespace dynload
}  // namespace phi
