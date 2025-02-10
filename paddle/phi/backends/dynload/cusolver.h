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

#include <cuda.h>
#include <cusolverDn.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {
extern std::once_flag cusolver_dso_flag;
extern void *cusolver_dso_handle;

#define DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP(__name)                   \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    cusolverStatus_t operator()(Args... args) {                      \
      using cusolverFunc = decltype(&::__name);                      \
      std::call_once(cusolver_dso_flag, []() {                       \
        cusolver_dso_handle = phi::dynload::GetCusolverDsoHandle();  \
      });                                                            \
      static void *p_##__name = dlsym(cusolver_dso_handle, #__name); \
      return reinterpret_cast<cusolverFunc>(p_##__name)(args...);    \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define CUSOLVER_ROUTINE_EACH(__macro)  \
  __macro(cusolverDnCreate);            \
  __macro(cusolverDnDestroy);           \
  __macro(cusolverDnSetStream);         \
  __macro(cusolverDnSpotrf_bufferSize); \
  __macro(cusolverDnDpotrf_bufferSize); \
  __macro(cusolverDnSpotrf);            \
  __macro(cusolverDnDpotrf);            \
  __macro(cusolverDnSpotrs);            \
  __macro(cusolverDnDpotrs);            \
  __macro(cusolverDnCpotrs);            \
  __macro(cusolverDnZpotrs);            \
  __macro(cusolverDnSsyevd_bufferSize); \
  __macro(cusolverDnDsyevd_bufferSize); \
  __macro(cusolverDnCheevd_bufferSize); \
  __macro(cusolverDnZheevd_bufferSize); \
  __macro(cusolverDnSsyevd);            \
  __macro(cusolverDnDsyevd);            \
  __macro(cusolverDnCheevd);            \
  __macro(cusolverDnZheevd);

CUSOLVER_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP);

#if CUDA_VERSION >= 9020
#define CUSOLVER_ROUTINE_EACH_R1(__macro) \
  __macro(cusolverDnSpotrfBatched);       \
  __macro(cusolverDnDpotrfBatched);       \
  __macro(cusolverDnSpotrsBatched);       \
  __macro(cusolverDnDpotrsBatched);       \
  __macro(cusolverDnSgesvdj_bufferSize);  \
  __macro(cusolverDnSgetrf_bufferSize);   \
  __macro(cusolverDnDgetrf_bufferSize);   \
  __macro(cusolverDnCgetrf_bufferSize);   \
  __macro(cusolverDnZgetrf_bufferSize);   \
  __macro(cusolverDnSgeqrf_bufferSize);   \
  __macro(cusolverDnDgeqrf_bufferSize);   \
  __macro(cusolverDnCgeqrf_bufferSize);   \
  __macro(cusolverDnZgeqrf_bufferSize);   \
  __macro(cusolverDnSorgqr_bufferSize);   \
  __macro(cusolverDnDorgqr_bufferSize);   \
  __macro(cusolverDnSormqr_bufferSize);   \
  __macro(cusolverDnDormqr_bufferSize);   \
  __macro(cusolverDnCungqr_bufferSize);   \
  __macro(cusolverDnZungqr_bufferSize);   \
  __macro(cusolverDnDestroyGesvdjInfo);   \
  __macro(cusolverDnCreateGesvdjInfo);    \
  __macro(cusolverDnDgesvdj_bufferSize);  \
  __macro(cusolverDnSgesvdj);             \
  __macro(cusolverDnDgesvdj);             \
  __macro(cusolverDnSgetrf);              \
  __macro(cusolverDnSgetrs);              \
  __macro(cusolverDnDgetrs);              \
  __macro(cusolverDnDgetrf);              \
  __macro(cusolverDnCgetrf);              \
  __macro(cusolverDnZgetrf);              \
  __macro(cusolverDnSgeqrf);              \
  __macro(cusolverDnDgeqrf);              \
  __macro(cusolverDnCgeqrf);              \
  __macro(cusolverDnZgeqrf);              \
  __macro(cusolverDnSorgqr);              \
  __macro(cusolverDnDorgqr);              \
  __macro(cusolverDnSormqr);              \
  __macro(cusolverDnDormqr);              \
  __macro(cusolverDnCungqr);              \
  __macro(cusolverDnZungqr);

CUSOLVER_ROUTINE_EACH_R1(DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP)
#endif

#if CUDA_VERSION >= 9020
#define CUSOLVER_ROUTINE_EACH_R2(__macro)      \
  __macro(cusolverDnCreateSyevjInfo);          \
  __macro(cusolverDnSsyevj_bufferSize);        \
  __macro(cusolverDnDsyevj_bufferSize);        \
  __macro(cusolverDnSsyevj);                   \
  __macro(cusolverDnDsyevj);                   \
  __macro(cusolverDnDestroySyevjInfo);         \
  __macro(cusolverDnXsyevjSetSortEig);         \
  __macro(cusolverDnSsyevjBatched_bufferSize); \
  __macro(cusolverDnDsyevjBatched_bufferSize); \
  __macro(cusolverDnCheevjBatched_bufferSize); \
  __macro(cusolverDnZheevjBatched_bufferSize); \
  __macro(cusolverDnSsyevjBatched);            \
  __macro(cusolverDnDsyevjBatched);            \
  __macro(cusolverDnCheevjBatched);            \
  __macro(cusolverDnZheevjBatched);

CUSOLVER_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP)
#endif

#undef DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP
}  // namespace dynload
}  // namespace phi
