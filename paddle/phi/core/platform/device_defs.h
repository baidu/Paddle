/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace platform {

enum DeviceType {
  CPU = 0,
  CUDA = 1,
  XPU = 3,
  IPU = 4,
  CUSTOM_DEVICE = 6,

  MAX_DEVICE_TYPES = 7,
};

DeviceType Place2DeviceType(const phi::Place& place);

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kCUSTOM_DEVICE = DeviceType::CUSTOM_DEVICE;

using DeviceContext = phi::DeviceContext;
using DeviceContextPool = phi::DeviceContextPool;

}  // namespace platform
}  // namespace paddle
