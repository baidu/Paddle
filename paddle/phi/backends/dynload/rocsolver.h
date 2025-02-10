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

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {
extern std::once_flag rocsolver_dso_flag;
extern void *rocsolver_dso_handle;

#define DECLARE_DYNAMIC_LOAD_ROCSOLVER_WRAP(__name)                   \
  struct DynLoad__##__name {                                          \
    template <typename... Args>                                       \
    rocblas_status operator()(Args... args) {                         \
      using rocsolverFunc = decltype(&::__name);                      \
      std::call_once(rocsolver_dso_flag, []() {                       \
        rocsolver_dso_handle = phi::dynload::GetCusolverDsoHandle();  \
      });                                                             \
      static void *p_##__name = dlsym(rocsolver_dso_handle, #__name); \
      return reinterpret_cast<rocsolverFunc>(p_##__name)(args...);    \
    }                                                                 \
  };                                                                  \
  extern DynLoad__##__name __name

#define ROCSOLVER_ROUTINE_EACH(__macro) \
  __macro(rocsolver_spotrs);            \
  __macro(rocsolver_dpotrs);            \
  __macro(rocsolver_cpotrs);            \
  __macro(rocsolver_zpotrs);            \
  __macro(rocsolver_sgetrf);            \
  __macro(rocsolver_dgetrf);            \
  __macro(rocsolver_cgetrf);            \
  __macro(rocsolver_zgetrf);            \
  __macro(rocsolver_spotrf);            \
  __macro(rocsolver_dpotrf);            \
  __macro(rocsolver_cpotrf);            \
  __macro(rocsolver_zpotrf);            \
  __macro(rocsolver_spotrf_batched);    \
  __macro(rocsolver_dpotrf_batched);    \
  __macro(rocsolver_cpotrf_batched);    \
  __macro(rocsolver_zpotrf_batched);    \
  __macro(rocsolver_sgeqrf);            \
  __macro(rocsolver_dgeqrf);            \
  __macro(rocsolver_sorgqr);            \
  __macro(rocsolver_dorgqr);            \
  __macro(rocsolver_dormqr);            \
  __macro(rocsolver_sormqr);

ROCSOLVER_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ROCSOLVER_WRAP);

#if HIP_VERSION >= 50300000
#define ROCSOLVER_ROUTINE_EACH1(__macro) \
  __macro(rocsolver_ssyevj_batched);     \
  __macro(rocsolver_dsyevj_batched);

ROCSOLVER_ROUTINE_EACH1(DECLARE_DYNAMIC_LOAD_ROCSOLVER_WRAP);
#endif

#undef DECLARE_DYNAMIC_LOAD_ROCSOLVER_WRAP
}  // namespace dynload
}  // namespace phi
