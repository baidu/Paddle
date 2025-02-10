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

#include <queue>

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/platform/device_event.h"

COMMON_DECLARE_bool(fast_eager_deletion_mode);
COMMON_DECLARE_bool(new_executor_use_cuda_graph);

namespace paddle {
namespace framework {

using Garbage = std::shared_ptr<memory::Allocation>;
using GarbageQueue = std::deque<Garbage>;

class InterpreterCoreGarbageCollector {
 public:
  InterpreterCoreGarbageCollector();
  virtual ~InterpreterCoreGarbageCollector() {}

  virtual void Add(Variable* var, const Instruction& instruction) = 0;

  virtual void Add(Variable* var, const InstructionBase* instruction) = 0;

  DISABLE_COPY_AND_ASSIGN(InterpreterCoreGarbageCollector);

 protected:
  std::unique_ptr<GarbageQueue> garbages_;
  int64_t max_memory_size_;
  int64_t cur_memory_size_;
  memory::SpinLock spinlock_;
};

inline bool IsInterpretercoreFastGCEnabled() {
  // When using cuda graph, fast GC must be used. Because
  // `EventQuery` method in event GC cannot be used in
  // cuda graph.
  PADDLE_ENFORCE_EQ(memory::allocation::AllocatorFacade::Instance()
                                .IsStreamSafeCUDAAllocatorUsed() == true &&
                        memory::allocation::AllocatorFacade::Instance()
                                .IsCUDAMallocAsyncAllocatorUsed() == true,
                    false,
                    common::errors::InvalidArgument(
                        "StreamSafeAllocator and AsyncAllocator shouldn't be "
                        "True together."));
  PADDLE_ENFORCE_EQ(memory::allocation::AllocatorFacade::Instance()
                                .IsStreamSafeCUDAAllocatorUsed() == false &&
                        memory::allocation::AllocatorFacade::Instance()
                                .IsCUDAMallocAsyncAllocatorUsed() == false &&
                        FLAGS_new_executor_use_cuda_graph,
                    false,
                    common::errors::InvalidArgument(
                        "When FLAGS_new_executor_use_cuda_graph is true, "
                        "Either IsStreamSafeCUDAAllocatorUsed or "
                        "IsCUDAMallocAsyncAllocatorUsed must be true, but "
                        "got false."));
  return (memory::allocation::AllocatorFacade::Instance()
              .IsStreamSafeCUDAAllocatorUsed() &&
          FLAGS_fast_eager_deletion_mode) ||
         FLAGS_new_executor_use_cuda_graph;
}

std::unique_ptr<InterpreterCoreGarbageCollector>
CreateInterpreterCoreGarbageCollector(
    const phi::Place& place, const std::vector<Instruction>& vec_instruction);

std::unique_ptr<InterpreterCoreGarbageCollector>
CreateInterpreterCoreGarbageCollector(
    const phi::Place& place,
    const std::vector<std::unique_ptr<InstructionBase>>& vec_instruction);

}  // namespace framework
}  // namespace paddle
