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

#include "paddle/phi/backends/gpu/gpu_info.h"

#include <unordered_set>
#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"

static std::once_flag g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<phi::gpuDeviceProp> g_device_props;

namespace phi::backends::gpu {

int DnnVersion() {
  if (!dynload::HasCUDNN()) return -1;
  return dynload::cudnnGetVersion();  // NOLINT
}

static int GetGPUDeviceCountImpl() {
  int driverVersion = 0;
  cudaError_t status = cudaDriverGetVersion(&driverVersion);

  if (!(status == gpuSuccess && driverVersion != 0)) {
    // No GPU driver
    VLOG(2) << "GPU Driver Version can't be detected. No GPU driver!";
    return 0;
  }

  const auto *cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");

  if (cuda_visible_devices != nullptr) {
    std::string cuda_visible_devices_str(cuda_visible_devices);
    if (!cuda_visible_devices_str.empty()) {
      cuda_visible_devices_str.erase(
          0, cuda_visible_devices_str.find_first_not_of('\''));
      cuda_visible_devices_str.erase(
          cuda_visible_devices_str.find_last_not_of('\'') + 1);
      cuda_visible_devices_str.erase(
          0, cuda_visible_devices_str.find_first_not_of('\"'));
      cuda_visible_devices_str.erase(
          cuda_visible_devices_str.find_last_not_of('\"') + 1);
    }
    if (std::all_of(cuda_visible_devices_str.begin(),
                    cuda_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "CUDA_VISIBLE_DEVICES is set to be "
                 "empty. No GPU detected.";
      return 0;
    }
  }
  int count;
  status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    VLOG(2) << "You have gpu driver and cuda installed, but the machine not "
               "has any gpu card.";
    count = 0;
  }
  return count;
}

int GetGPUDeviceCount() {
  // cache the count
  static auto dev_cnt = GetGPUDeviceCountImpl();
  return dev_cnt;
}

int GetGPUComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int major, minor;
  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);

  PADDLE_ENFORCE_GPU_SUCCESS(major_error_code);
  PADDLE_ENFORCE_GPU_SUCCESS(minor_error_code);
  return major * 10 + minor;
}

int GetGPURuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int runtime_version = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

int GetGPUDriverVersion(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int driver_version = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDriverGetVersion(&driver_version));
  return driver_version;
}

bool TensorCoreAvailable() {
  int device = GetCurrentDeviceId();
  int driver_version = GetGPUComputeCapability(device);
  return driver_version >= 70;
}

int GetGPUMultiProcessors(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id));
  return count;
}

int GetGPUMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id));

  return count;
}

int GetGPUMaxThreadsPerBlock(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id));
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&device_id));
  return device_id;
}

std::array<unsigned int, 3> GetGpuMaxGridDimSize(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  std::array<unsigned int, 3> ret = {};
  int size;
  auto error_code_x = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id);
  PADDLE_ENFORCE_GPU_SUCCESS(error_code_x);
  ret[0] = size;

  auto error_code_y = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id);
  PADDLE_ENFORCE_GPU_SUCCESS(error_code_y);
  ret[1] = size;

  auto error_code_z = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id);
  PADDLE_ENFORCE_GPU_SUCCESS(error_code_z);
  ret[2] = size;
  return ret;
}

std::pair<int, int> GetGpuStreamPriorityRange() {
  int least_priority, greatest_priority;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  return std::make_pair(least_priority, greatest_priority);
}

const gpuDeviceProp &GetDeviceProperties(int id) {
  std::call_once(g_device_props_size_init_flag, [&] {
    int gpu_num = 0;
    gpu_num = GetGPUDeviceCount();
    g_device_props_init_flags.resize(gpu_num);
    g_device_props.resize(gpu_num);
    for (int i = 0; i < gpu_num; ++i) {
      g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
    }
  });

  if (id == -1) {
    id = GetCurrentDeviceId();
  }

  if (id < 0 || id >= static_cast<int>(g_device_props.size())) {
    PADDLE_THROW(common::errors::OutOfRange(
        "The device id %d is out of range [0, %d), where %d is the number of "
        "devices on this machine. Because the device id should be greater than "
        "or equal to zero and smaller than the number of gpus. Please input "
        "appropriate device again!",
        id,
        static_cast<int>(g_device_props.size()),
        static_cast<int>(g_device_props.size())));
  }

  std::call_once(*(g_device_props_init_flags[id]), [&] {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaGetDeviceProperties(&g_device_props[id], id));
  });

  return g_device_props[id];
}

void SetDeviceId(int id) {
  static thread_local bool first_call = true;
  if (first_call) {
    PADDLE_ENFORCE_LT(id,
                      GetGPUDeviceCount(),
                      common::errors::InvalidArgument(
                          "Device id must be less than GPU count, "
                          "but received id is: %d. GPU count is: %d.",
                          id,
                          GetGPUDeviceCount()));

    PADDLE_RETRY_CUDA_SUCCESS(cudaSetDevice(id));
    VLOG(4) << "SetDeviceId " << id;
    first_call = false;
    return;
  }

  int prev_id;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&prev_id));
  if (prev_id != id) {
    PADDLE_ENFORCE_LT(id,
                      GetGPUDeviceCount(),
                      common::errors::InvalidArgument(
                          "Device id must be less than GPU count, "
                          "but received id is: %d. GPU count is: %d.",
                          id,
                          GetGPUDeviceCount()));

    PADDLE_RETRY_CUDA_SUCCESS(cudaSetDevice(id));
    VLOG(4) << "SetDeviceId " << id;
  }
}

void GpuMemcpyAsync(void *dst,
                    const void *src,
                    size_t count,
                    gpuMemcpyKind kind,
                    gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(dst, src, count, kind, stream));
}

void GpuMemcpySync(void *dst,
                   const void *src,
                   size_t count,
                   gpuMemcpyKind kind) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(dst, src, count, kind));
}

void GpuMemcpyPeerAsync(void *dst,
                        int dst_device,
                        const void *src,
                        int src_device,
                        size_t count,
                        gpuStream_t stream) {
  static std::unordered_set<int> p2p_enabled_devices;
  if (p2p_enabled_devices.find(src_device) == p2p_enabled_devices.end()) {
    int p2p_flag;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaDeviceCanAccessPeer(&p2p_flag, dst_device, src_device));
    if (p2p_flag == 1) {
      cudaError_t ret = cudaDeviceEnablePeerAccess(src_device, 0);
      if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
        LOG(INFO) << " Cuda error(" << ret << "), " << cudaGetErrorString(ret)
                  << ".";
      } else {
        VLOG(10) << "Peer memcpy is enable for device " << src_device;
        cudaGetLastError();
      }
    }
    p2p_enabled_devices.insert(src_device);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream));
}

void GpuMemcpyPeerSync(
    void *dst, int dst_device, const void *src, int src_device, size_t count) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyPeer(dst, dst_device, src, src_device, count));
}

void GpuMemsetAsync(void *dst, int value, size_t count, gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(dst, value, count, stream));
}

void GpuStreamSync(gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

void GpuDestroyStream(gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream));
}

void GpuDeviceSync() { PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize()); }

gpuError_t GpuGetLastError() { return cudaGetLastError(); }

// See
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
// for more detail about managed memory requirements
bool IsGPUManagedMemorySupported(int dev_id) {
  PADDLE_ENFORCE_LT(dev_id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        dev_id,
                        GetGPUDeviceCount()));
#if defined(__linux__) || defined(_WIN32)
  int ManagedMemoryAttr;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceGetAttribute(
      &ManagedMemoryAttr, cudaDevAttrManagedMemory, dev_id));
  return ManagedMemoryAttr != 0;
#else
  return false;
#endif
}

bool IsGPUManagedMemoryOversubscriptionSupported(int dev_id) {
  PADDLE_ENFORCE_LT(dev_id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        dev_id,
                        GetGPUDeviceCount()));
#ifdef __linux__
  return IsGPUManagedMemorySupported(dev_id) &&
         GetGPUComputeCapability(dev_id) >= 60;
#else
  return false;
#endif
}

}  // namespace phi::backends::gpu
