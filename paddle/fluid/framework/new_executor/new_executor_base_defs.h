// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <unordered_set>
#include "paddle/phi/core/platform/device_event_defs.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle {
namespace framework {

using SchedulingPriority = int64_t;

constexpr const char* kCoalesceTensor = "coalesce_tensor";

// stream types
constexpr const char* kCustomStream = "CustomStream";
constexpr const char* kDefaultStream = "DefaultStream";
constexpr const char* kD2HStream = "D2HStream";
constexpr const char* kH2DStream = "H2DStream";

constexpr int kEmptyVarIndex = 0;

struct EventInter {
  explicit EventInter(size_t instr_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type)
      : instr_id_(instr_id), event_(event), waiter_type_(waiter_type) {}
  size_t instr_id_;
  std::shared_ptr<platform::DeviceEvent> event_;
  platform::DeviceType waiter_type_;
};

enum class OpFuncType {
  kCpuSync,  // CPU kernel, block host
  kGpuSync,  // GPU or other device kernel without asynchronous operation
  kGpuAsync  // GPU or other device kernel with asynchronous operation
};

}  // namespace framework
}  // namespace paddle
