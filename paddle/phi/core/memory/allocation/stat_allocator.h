// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/stats.h"
#include "paddle/phi/core/platform/profiler/mem_tracing.h"

namespace paddle {
namespace memory {
namespace allocation {

class StatAllocator : public Allocator {
 public:
  explicit StatAllocator(std::shared_ptr<Allocator> underlying_allocator)
      : underlying_allocator_(std::move(underlying_allocator)) {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  void FreeImpl(phi::Allocation* allocation) override {
    platform::RecordMemEvent(allocation->ptr(),
                             allocation->place(),
                             allocation->size(),
                             phi::TracerMemEventType::Free);
    underlying_allocator_->Free(allocation);
  }

  phi::Allocation* AllocateImpl(size_t size) override {
    phi::Allocator::AllocationPtr allocation =
        underlying_allocator_->Allocate(size);

    const phi::Place& place = allocation->place();
    if (phi::is_cpu_place(place) || phi::is_cuda_pinned_place(place)) {
      HOST_MEMORY_STAT_UPDATE(
          Allocated, place.GetDeviceId(), allocation->size());
    } else {
      DEVICE_MEMORY_STAT_UPDATE(
          Allocated, place.GetDeviceId(), allocation->size());
    }
    platform::RecordMemEvent(allocation->ptr(),
                             allocation->place(),
                             allocation->size(),
                             phi::TracerMemEventType::Allocate);
    return allocation.release();
  }

  uint64_t ReleaseImpl(const phi::Place& place) override {
    return underlying_allocator_->Release(place);
  }

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
