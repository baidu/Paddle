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

#include <NvInfer.h>
#include <NvInferPlugin.h>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace dynload {

void* GetTensorRtHandle();

extern std::once_flag tensorrt_dso_flag;
extern void* tensorrt_dso_handle;

void* GetTensorRtPluginHandle();
extern std::once_flag tensorrt_plugin_dso_flag;
extern void* tensorrt_plugin_dso_handle;

#define DECLARE_DYNAMIC_LOAD_TENSORRT_POINTER_WRAP(__name)             \
  struct DynLoad__##__name {                                           \
    template <typename... Args>                                        \
    void* operator()(Args... args) {                                   \
      std::call_once(tensorrt_dso_flag, []() {                         \
        tensorrt_dso_handle = phi::dynload::GetTensorRtHandle();       \
      });                                                              \
      static void* p_##__name = dlsym(tensorrt_dso_handle, #__name);   \
      if (p_##__name == nullptr) {                                     \
        return nullptr;                                                \
      }                                                                \
      using tensorrt_func = decltype(&::__name);                       \
      auto ret = reinterpret_cast<tensorrt_func>(p_##__name)(args...); \
      return static_cast<void*>(ret);                                  \
    }                                                                  \
  };                                                                   \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_TENSORRT_NON_POINTER_WRAP(__name)              \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {        \
      std::call_once(tensorrt_dso_flag, []() {                              \
        tensorrt_dso_handle = phi::dynload::GetTensorRtHandle();            \
      });                                                                   \
      static void* p_##__name = dlsym(tensorrt_dso_handle, #__name);        \
      PADDLE_ENFORCE_NOT_NULL(p_##__name,                                   \
                              common::errors::Unavailable(                  \
                                  "Load tensorrt api %s failed", #__name)); \
      using tensorrt_func = decltype(&::__name);                            \
      return reinterpret_cast<tensorrt_func>(p_##__name)(args...);          \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_TENSORRT_PLUGIN_WRAP(__name)                      \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {           \
      std::call_once(tensorrt_plugin_dso_flag, []() {                          \
        tensorrt_plugin_dso_handle = phi::dynload::GetTensorRtPluginHandle();  \
      });                                                                      \
      static void* p_##__name = dlsym(tensorrt_plugin_dso_handle, #__name);    \
      PADDLE_ENFORCE_NOT_NULL(p_##__name,                                      \
                              common::errors::Unavailable(                     \
                                  "Load tensorrt plugin %s failed", #__name)); \
      using tensorrt_plugin_func = decltype(&::__name);                        \
      return reinterpret_cast<tensorrt_plugin_func>(p_##__name)(args...);      \
    }                                                                          \
  };                                                                           \
  extern DynLoad__##__name __name

#ifdef NV_TENSORRT_MAJOR

#if (NV_TENSORRT_MAJOR >= 6)
#define TENSORRT_RAND_ROUTINE_EACH_POINTER(__macro) \
  __macro(createInferBuilder_INTERNAL);             \
  __macro(createInferRuntime_INTERNAL);             \
  __macro(createInferRefitter_INTERNAL);            \
  __macro(getPluginRegistry);
#else
#define TENSORRT_RAND_ROUTINE_EACH_POINTER(__macro) \
  __macro(createInferBuilder_INTERNAL);             \
  __macro(createInferRuntime_INTERNAL);
#endif

#define TENSORRT_RAND_ROUTINE_EACH_NON_POINTER(__macro) \
  __macro(getInferLibVersion);

#define TENSORRT_PLUGIN_RAND_ROUTINE_EACH(__macro) \
  __macro(initLibNvInferPlugins);

TENSORRT_RAND_ROUTINE_EACH_POINTER(DECLARE_DYNAMIC_LOAD_TENSORRT_POINTER_WRAP)
TENSORRT_RAND_ROUTINE_EACH_NON_POINTER(
    DECLARE_DYNAMIC_LOAD_TENSORRT_NON_POINTER_WRAP)
TENSORRT_PLUGIN_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_TENSORRT_PLUGIN_WRAP)

#endif  // end of NV_TENSORRT_MAJOR

}  // namespace dynload
}  // namespace phi
