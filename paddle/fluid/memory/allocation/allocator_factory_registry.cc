// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/allocator_factory_registry.h"
#include "paddle/fluid/memory/allocation/allocator_factory.h"

namespace paddle {
namespace memory {
namespace allocation {
AllocatorFactory& AllocatorFactoryRegistry::Get() {
  for (auto& factory : factories_) {
    if (factory->CanBuild()) {
      return *factory;
    }
  }
  PADDLE_THROW(
      "Cannot find suitable allocator factory. Please whether check "
      "FLAGS_allocator_strategy is set correctly.");
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
