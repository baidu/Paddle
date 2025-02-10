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

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"

namespace phi {
namespace dynload {

void* GetFP8FP8GemmFusedHandle();

}  // namespace dynload
}  // namespace phi
