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

#include <hiprand.h>
#include <dlfcn.h>

#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {
extern std::once_flag hiprand_dso_flag;
extern void *hiprand_dso_handle;
#ifdef PADDLE_USE_DSO
#define DECLARE_DYNAMIC_LOAD_HIPRAND_WRAP(__name)                              \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    hiprandStatus_t operator()(Args... args) {                                 \
      typedef hiprandStatus_t (*hiprandFunc)(Args...);                         \
      std::call_once(hiprand_dso_flag, []() {                                  \
        hiprand_dso_handle = paddle::platform::dynload::GetHiprandDsoHandle(); \
      });                                                                      \
      void *p_##__name = dlsym(hiprand_dso_handle, #__name);                   \
      return reinterpret_cast<hiprandFunc>(p_##__name)(args...);               \
    }                                                                          \
  };                                                                           \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_HIPRAND_WRAP(__name) \
  struct DynLoad__##__name {                      \
    template <typename... Args>                   \
    hiprandStatus_t operator()(Args... args) {    \
      return __name(args...);                     \
    }                                             \
  };                                              \
  extern DynLoad__##__name __name
#endif

#define HIPRAND_RAND_ROUTINE_EACH(__macro)      \
  __macro(hiprandCreateGenerator);              \
  __macro(hiprandSetStream);                    \
  __macro(hiprandSetPseudoRandomGeneratorSeed); \
  __macro(hiprandGenerateUniform);              \
  __macro(hiprandGenerateUniformDouble);        \
  __macro(hiprandGenerateNormal);               \
  __macro(hiprandDestroyGenerator);

HIPRAND_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPRAND_WRAP);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
