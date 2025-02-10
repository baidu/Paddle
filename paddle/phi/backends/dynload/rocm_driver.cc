/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/dynload/rocm_driver.h"

namespace phi {
namespace dynload {

std::once_flag rocm_dso_flag;
void* rocm_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

ROCM_ROUTINE_EACH(DEFINE_WRAP);

ROCM_ROUTINE_EACH_VVM(DEFINE_WRAP);

bool HasCUDADriver() {
  std::call_once(rocm_dso_flag, []() { rocm_dso_handle = GetCUDADsoHandle(); });
  return rocm_dso_handle != nullptr;
}

}  // namespace dynload
}  // namespace phi
