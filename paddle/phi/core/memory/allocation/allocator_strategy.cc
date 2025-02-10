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

#include "paddle/phi/core/memory/allocation/allocator_strategy.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"

COMMON_DECLARE_string(allocator_strategy);

namespace paddle::memory::allocation {

static AllocatorStrategy GetStrategyFromFlag() {
  if (FLAGS_allocator_strategy == "naive_best_fit") {
    return AllocatorStrategy::kNaiveBestFit;
  }

  if (FLAGS_allocator_strategy == "auto_growth") {
    return AllocatorStrategy::kAutoGrowth;
  }

  if (FLAGS_allocator_strategy == "thread_local") {
    return AllocatorStrategy::kThreadLocal;
  }

  PADDLE_THROW(common::errors::InvalidArgument(
      "Unsupported allocator strategy: %s, candidates are naive_best_fit, "
      "auto_growth or thread_local.",
      FLAGS_allocator_strategy));
}

AllocatorStrategy GetAllocatorStrategy() {
  static AllocatorStrategy strategy = GetStrategyFromFlag();
  return strategy;
}

void UseAllocatorStrategyGFlag() {}
}  // namespace paddle::memory::allocation
