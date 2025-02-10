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

#include "paddle/phi/core/memory/allocation/cuda_managed_allocator.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include <string>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle::memory::allocation {
bool CUDAManagedAllocator::IsAllocThreadSafe() const { return true; }

void CUDAManagedAllocator::FreeImpl(phi::Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      allocation->place(),
      place_,
      common::errors::PermissionDenied(
          "GPU memory is freed in incorrect device. This may be a bug"));
  platform::RecordedGpuFree(
      allocation->ptr(), allocation->size(), place_.device);
  delete allocation;
}

phi::Allocation* CUDAManagedAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });

  int dev_id = place_.device;  // NOLINT
  void* ptr;
  auto result = platform::RecordedGpuMalloc(&ptr,
                                            size,
                                            dev_id,
                                            /* malloc_managed_memory = */ true);
  if (LIKELY(result == gpuSuccess)) {
    return new Allocation(ptr, size, phi::Place(place_));
  }

  uint64_t limit_size = platform::RecordedGpuLimitSize(dev_id);
  uint64_t malloc_size = platform::RecordedGpuMallocSize(dev_id);
  bool is_limited =
      platform::IsGpuMallocRecorded(dev_id) && malloc_size + size > limit_size;

  std::string err_msg;
  if (UNLIKELY(is_limited)) {
    int64_t limit_size_mb = limit_size >> 20;  // NOLINT
    err_msg = string::Sprintf(
        "Or set environment variable `FLAGS_gpu_memory_limit_mb` to a larger "
        "value. Currently `FLAGS_gpu_memory_limit_mb` is %d, so the maximum "
        "GPU memory usage is limited to %d MB.\n"
        "   The command is `export FLAGS_gpu_memory_limit_mb=xxx`.",
        limit_size_mb,
        limit_size_mb);
  }

  PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
      "\n\nOut of memory error on GPU %d. "
      "Cannot allocate %s CUDA managed memory on GPU %d, %s memory has been "
      "allocated.\n\n"
      "Please check whether there is any other process using GPU %d.\n"
      "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
      "2. If no, please decrease the batch size of your model. %s\n\n",
      dev_id,
      string::HumanReadableSize(size),
      dev_id,
      string::HumanReadableSize(malloc_size),
      dev_id,
      err_msg));
}

}  // namespace paddle::memory::allocation
