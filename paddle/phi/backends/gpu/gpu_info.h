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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <stddef.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_types.h"

namespace phi {
namespace backends {
namespace gpu {

//! Get the version of dnn
int DnnVersion();

//! Get the total number of GPU devices in system.
int GetGPUDeviceCount();

//! Get the compute capability of the ith GPU (format: major * 10 + minor)
int GetGPUComputeCapability(int id);

//! Get the runtime version of the ith GPU
int GetGPURuntimeVersion(int id);

//! Get the driver version of the ith GPU
int GetGPUDriverVersion(int id);

//! Whether the current device support TensorCore
bool TensorCoreAvailable();

//! Get the MultiProcessors of the ith GPU.
int GetGPUMultiProcessors(int id);

//! Get the MaxThreads of each MultiProcessor of the ith GPU.
int GetGPUMaxThreadsPerMultiProcessor(int id);

//! Get the MaxThreads of each block of the ith GPU.
int GetGPUMaxThreadsPerBlock(int id);

//! Get the current GPU device id in system.
int GetCurrentDeviceId();

//! Get the maximum GridDim size for GPU buddy allocator.
std::array<unsigned int, 3> GetGpuMaxGridDimSize(int);

std::pair<int, int> GetGpuStreamPriorityRange();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices();

//! Get the properties of the ith GPU device.
const gpuDeviceProp &GetDeviceProperties(int id);

//! Set the GPU device id for next execution.
void SetDeviceId(int device_id);

//! Get the available memory to allocate, which is the size of available gpu
//! minus reserving.
size_t GpuAvailableMemToAlloc();

//! Get the minimum chunk size for GPU buddy allocator.
size_t GpuMinChunkSize();

//! Copy memory from address src to dst asynchronously.
void GpuMemcpyAsync(void *dst,
                    const void *src,
                    size_t count,
                    gpuMemcpyKind kind,
                    gpuStream_t stream);

//! Copy memory from address src to dst synchronously.
void GpuMemcpySync(void *dst,
                   const void *src,
                   size_t count,
                   gpuMemcpyKind kind);

//! Copy memory from one device to another device asynchronously.
void GpuMemcpyPeerAsync(void *dst,
                        int dst_device,
                        const void *src,
                        int src_device,
                        size_t count,
                        gpuStream_t stream);

//! Copy memory from one device to another device synchronously.
void GpuMemcpyPeerSync(
    void *dst, int dst_device, const void *src, int src_device, size_t count);

//! Set memory dst with value count size asynchronously
void GpuMemsetAsync(void *dst, int value, size_t count, gpuStream_t stream);

//! Blocks until stream has completed all operations.
void GpuStreamSync(gpuStream_t stream);

void GpuDestroyStream(gpuStream_t stream);

// ! Blocks until device has completed all operations.
void GpuDeviceSync();

gpuError_t GpuGetLastError();

bool IsGPUManagedMemorySupported(int dev_id);

bool IsGPUManagedMemoryOversubscriptionSupported(int dev_id);

class GPUDeviceGuard {
 public:
  explicit inline GPUDeviceGuard(int dev_id) {
    int prev_id = GetCurrentDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      SetDeviceId(dev_id);
    }
  }
  inline ~GPUDeviceGuard() {
    if (prev_id_ != -1) {
      SetDeviceId(prev_id_);
    }
  }
  GPUDeviceGuard(const GPUDeviceGuard &o) = delete;
  GPUDeviceGuard &operator=(const GPUDeviceGuard &o) = delete;

 private:
  int prev_id_{-1};
};

}  // namespace gpu
}  // namespace backends
}  // namespace phi

#endif
