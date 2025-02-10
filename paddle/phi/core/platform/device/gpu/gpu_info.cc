// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#include <array>
#include <cstdlib>
#include <mutex>
#include <set>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/memory.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/lock_guard_ptr.h"
#include "paddle/phi/core/platform/profiler/mem_tracing.h"
#include "paddle/utils/string/split.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/gpu/rocm/hip_graph.h"
#else
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#endif

#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10020
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif
#else  // PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/rocm_driver.h"
#endif

COMMON_DECLARE_double(fraction_of_gpu_memory_to_use);
COMMON_DECLARE_uint64(initial_gpu_memory_in_mb);
COMMON_DECLARE_uint64(reallocate_gpu_memory_in_mb);
COMMON_DECLARE_bool(enable_cublas_tensor_op_math);
COMMON_DECLARE_uint64(gpu_memory_limit_mb);

PHI_DEFINE_EXPORTED_bool(enable_gpu_memory_usage_log,
                         false,
                         "Whether to print the message of gpu memory usage "
                         "at exit, mainly used for UT and CI.");
PHI_DEFINE_EXPORTED_bool(enable_gpu_memory_usage_log_mb,
                         true,
                         "Whether to print the message of gpu memory usage "
                         "MB as a unit of measurement.");
PHI_DEFINE_EXPORTED_uint64(cuda_memory_async_pool_release_threshold,
                           ULLONG_MAX,
                           "Amount of reserved memory in bytes to hold onto "
                           "before trying to release memory back to the OS");

namespace paddle::platform {

void GpuMemoryUsage(size_t *available, size_t *total) {
  size_t actual_available, actual_total;
  RecordedGpuMemGetInfo(available,
                        total,
                        &actual_available,
                        &actual_total,
                        platform::GetCurrentDeviceId());
}

size_t GpuAvailableMemToAlloc() {
  return phi::backends::gpu::GpuAvailableMemToAlloc();
}

size_t GpuMaxAllocSize() {
  return std::max(GpuInitAllocSize(), GpuReallocSize());
}

static size_t GpuAllocSize(bool realloc) {
  size_t available_to_alloc = GpuAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc,
      0,
      common::errors::ResourceExhausted("Not enough available GPU memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20
                     : available_to_alloc *
                           FLAGS_fraction_of_gpu_memory_to_use);  // NOLINT
  PADDLE_ENFORCE_GE(
      available_to_alloc,
      alloc_bytes,
      common::errors::ResourceExhausted("Not enough available GPU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

size_t GpuInitAllocSize() { return GpuAllocSize(/* realloc = */ false); }

size_t GpuReallocSize() { return GpuAllocSize(/* realloc = */ true); }

size_t GpuMinChunkSize() { return phi::backends::gpu::GpuMinChunkSize(); }

size_t GpuMaxChunkSize() {
  size_t max_chunk_size = GpuMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

static void RaiseNonOutOfMemoryError(gpuError_t *status) {
  if (*status == gpuErrorOutOfMemory) {
    *status = gpuSuccess;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(*status);

  *status = platform::GpuGetLastError();
  if (*status == gpuErrorOutOfMemory) {
    *status = gpuSuccess;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(*status);
}

class RecordedGpuMallocHelper {
 private:
  explicit RecordedGpuMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_ = std::make_unique<std::mutex>();
    }

    if (FLAGS_enable_gpu_memory_usage_log) {
      // A fake UPDATE to trigger the construction of memory stat instances,
      // make sure that they are destructed after RecordedGpuMallocHelper.
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id, 0);
      DEVICE_MEMORY_STAT_UPDATE(Allocated, dev_id, 0);
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedGpuMallocHelper);

 public:
  ~RecordedGpuMallocHelper() {
    if (FLAGS_enable_gpu_memory_usage_log) {
      if (FLAGS_enable_gpu_memory_usage_log_mb) {
        std::cout << "[Memory Usage (MB)] gpu " << dev_id_ << " : Reserved = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, dev_id_) /
                         1048576.0
                  << ", Allocated = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, dev_id_) /
                         1048576.0
                  << std::endl;
      } else {
        std::cout << "[Memory Usage (Byte)] gpu " << dev_id_ << " : Reserved = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, dev_id_)
                  << ", Allocated = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, dev_id_)
                  << std::endl;
      }
    }
  }

  static RecordedGpuMallocHelper *Instance(int dev_id) {
    static std::vector<std::unique_ptr<RecordedGpuMallocHelper>> instances_;

    std::call_once(once_flag_, [] {
      int dev_cnt = GetGPUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        instances_.emplace_back(
            new RecordedGpuMallocHelper(i, FLAGS_gpu_memory_limit_mb << 20));
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id,
        0,
        common::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id,
        instances_.size(),
        common::errors::OutOfRange("Device id %d exceeds gpu card number %d.",
                                   dev_id,
                                   instances_.size()));
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` gpu memory. Only cudaErrorMemoryAllocation
   * or cudaSuccess would be returned, and the cudaGetLastError() flag
   * would be clear.
   */
  gpuError_t Malloc(void **ptr,
                    size_t size,
                    bool malloc_managed_memory = false) {
    // LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_.load() + size > limit_size_)) {
      return gpuErrorOutOfMemory;
    }

    CUDADeviceGuard guard(dev_id_);
    gpuError_t result;
#ifdef PADDLE_WITH_HIP
    phi::backends::gpu::CUDAGraphCaptureModeGuard capture_mode_guard;
    if (UNLIKELY(malloc_managed_memory)) {
      result = hipMallocManaged(ptr, size);
    } else {
      result = hipMalloc(ptr, size);
    }
#else
    phi::backends::gpu::CUDAGraphCaptureModeGuard capture_mode_guard;
    if (UNLIKELY(malloc_managed_memory)) {
      result = cudaMallocManaged(ptr, size);
    } else {
      result = cudaMalloc(ptr, size);
      VLOG(10) << "[cudaMalloc] size=" << static_cast<double>(size) / (1 << 20)
               << " MB, result=" << result;
    }
#endif
    if (result == gpuSuccess) {
      cur_size_.fetch_add(size);
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, size);
      platform::RecordMemEvent(ptr,
                               GPUPlace(dev_id_),
                               size,
                               phi::TracerMemEventType::ReservedAllocate);
#ifdef PADDLE_WITH_TESTING
      std::lock_guard<std::mutex> lock_guard(gpu_ptrs_mutex);
      gpu_ptrs.insert(*ptr);
#endif

      return gpuSuccess;
    } else {
      RaiseNonOutOfMemoryError(&result);
      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError. Therefore, we can
      // return cudaErrorMemoryAllocation directly here.
      return gpuErrorOutOfMemory;
    }
  }

  /**
   * Try to allocate `size` gpu memory. Only cudaErrorMemoryAllocation
   * or cudaSuccess would be returned, and the cudaGetLastError() flag
   * would be clear.
   */
  gpuError_t MallocAsync(void **ptr, size_t size, gpuStream_t stream) {
#if defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUDA) && (CUDA_VERSION >= 11020)
    LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_.load() + size > limit_size_)) {
      return gpuErrorOutOfMemory;
    }
    CUDADeviceGuard guard(dev_id_);

    std::call_once(set_cudamempoolattr_once_flag_, [&]() {
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaDeviceGetDefaultMemPool(&memPool_, dev_id_));
#else  // PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipDeviceGetDefaultMemPool(&memPool_, dev_id_));
#endif
      uint64_t thresholdVal = FLAGS_cuda_memory_async_pool_release_threshold;
      VLOG(10) << "[cudaMallocAsync] set cudaMemPoolAttrReleaseThreshold to "
               << thresholdVal;
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemPoolSetAttribute(memPool_,
                                  cudaMemPoolAttrReleaseThreshold,
                                  reinterpret_cast<void *>(&thresholdVal)));
#else  // PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemPoolSetAttribute(memPool_,
                                  hipMemPoolAttrReleaseThreshold,
                                  reinterpret_cast<void *>(&thresholdVal)));
#endif
    });

    gpuError_t result;
#ifdef PADDLE_WITH_CUDA
    result = cudaMallocAsync(ptr, size, stream);
#else  // PADDLE_WITH_HIP
    result = hipMallocAsync(ptr, size, stream);
#endif
    VLOG(10) << "[cudaMallocAsync] ptr = " << (*ptr)
             << " size = " << static_cast<double>(size) / (1 << 20)
             << " MB result = " << result << " stream = " << stream;
    if (result == gpuSuccess) {
      cur_size_.fetch_add(size);
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, size);
      platform::RecordMemEvent(ptr,
                               GPUPlace(dev_id_),
                               size,
                               phi::TracerMemEventType::ReservedAllocate);
#ifdef PADDLE_WITH_TESTING
      std::lock_guard<std::mutex> lock_guard(gpu_ptrs_mutex);
      gpu_ptrs.insert(*ptr);
#endif

      return gpuSuccess;
    } else {
      RaiseNonOutOfMemoryError(&result);
      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError. Therefore, we can
      // return cudaErrorMemoryAllocation directly here.
      return gpuErrorOutOfMemory;
    }
#else
    PADDLE_THROW(common::errors::Unavailable(
        "MallocAsync is not supported in this version of CUDA."));
#endif
  }

  /**
   * Free gpu memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void *ptr, size_t size) {
    // Purposefully allow cudaErrorCudartUnloading, because
    // that is returned if you ever call cudaFree after the
    // driver has already shutdown. This happens only if the
    // process is terminating, in which case we don't care if
    // cudaFree succeeds.
    CUDADeviceGuard guard(dev_id_);
#ifdef PADDLE_WITH_HIP
    auto err = hipFree(ptr);
    if (err != hipErrorDeinitialized) {
#else
    auto err = cudaFree(ptr);
    VLOG(10) << "[cudaFree] size=" << static_cast<double>(size) / (1 << 20)
             << " MB";
    if (err != cudaErrorCudartUnloading) {
#endif
      PADDLE_ENFORCE_GPU_SUCCESS(err);
      cur_size_.fetch_sub(size);
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, -size);
      platform::RecordMemEvent(
          ptr, GPUPlace(dev_id_), size, phi::TracerMemEventType::ReservedFree);
    } else {
      platform::GpuGetLastError();  // clear the error flag when
                                    // cudaErrorCudartUnloading /
                                    // hipErrorDeinitialized
    }
#ifdef PADDLE_WITH_TESTING
    std::lock_guard<std::mutex> lock_guard(gpu_ptrs_mutex);
    gpu_ptrs.erase(ptr);
#endif
  }

  void FreeAsync(void *ptr, size_t size, gpuStream_t stream) {
#if defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUDA) && (CUDA_VERSION >= 11020)
    // Purposefully allow cudaErrorCudartUnloading, because
    // that is returned if you ever call cudaFree after the
    // driver has already shutdown. This happens only if the
    // process is terminating, in which case we don't care if
    // cudaFree succeeds.
    CUDADeviceGuard guard(dev_id_);
#ifdef PADDLE_WITH_CUDA
    auto err = cudaFreeAsync(ptr, stream);
#else  // PADDLE_WITH_HIP
    auto err = hipFreeAsync(ptr, stream);
#endif
    VLOG(10) << "[cudaFreeAsync] ptr = " << ptr
             << " size =" << static_cast<double>(size) / (1 << 20)
             << " MB result = " << err << " stream = " << stream;
    if (err != gpuErrorCudartUnloading) {
      PADDLE_ENFORCE_GPU_SUCCESS(err);
      cur_size_.fetch_sub(size);
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, -size);
      platform::RecordMemEvent(
          ptr, GPUPlace(dev_id_), size, phi::TracerMemEventType::ReservedFree);
    } else {
      platform::GpuGetLastError();  // clear the error flag when
                                    // cudaErrorCudartUnloading /
                                    // hipErrorDeinitialized
    }
#ifdef PADDLE_WITH_TESTING
    std::lock_guard<std::mutex> lock_guard(gpu_ptrs_mutex);
    gpu_ptrs.erase(ptr);
#endif

#else
    PADDLE_THROW(common::errors::Unavailable(
        "FreeAsync is not supported in this version of CUDA."));
#endif
  }
  void *GetBasePtr(void *ptr) {
#ifdef PADDLE_WITH_TESTING
    std::lock_guard<std::mutex> lock_guard(gpu_ptrs_mutex);
    auto it = gpu_ptrs.upper_bound(ptr);
    if (it == gpu_ptrs.begin()) {
      return nullptr;
    }
    return *(--it);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "The RecordedGpuMallocHelper::GetBasePtr is only implemented with "
        "testing, should not use for release."));
    return nullptr;
#endif
  }

  bool GetMemInfo(size_t *avail,
                  size_t *total,
                  size_t *actual_avail,
                  size_t *actual_total) {
    {
      CUDADeviceGuard guard(dev_id_);
#ifdef PADDLE_WITH_HIP
      auto result = hipMemGetInfo(actual_avail, actual_total);
#else
      auto result = cudaMemGetInfo(actual_avail, actual_total);
#endif
      if (result != gpuSuccess) {
        *actual_avail = 0;
      }
      RaiseNonOutOfMemoryError(&result);
    }

    if (NeedRecord()) {
      std::lock_guard<std::mutex> lock_guard(*mtx_);
      *avail = std::min(*actual_avail, limit_size_ - cur_size_.load());
      *total = std::min(*actual_total, limit_size_);
      return *total < *actual_total;
    } else {
      *avail = *actual_avail;
      *total = *actual_total;
      return false;
    }
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

  uint64_t RecordedSize() const { return cur_size_.load(); }

  uint64_t LimitSize() const { return limit_size_; }

#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10020
  CUresult MemCreate(CUmemGenericAllocationHandle *handle,
                     size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {  // NOLINT
    auto result = phi::dynload::cuMemCreate(handle, size, prop, flags);
    if (result == CUDA_SUCCESS) {
      cur_size_.fetch_add(size);
    }
    return result;
  }

  CUresult MemRelease(CUmemGenericAllocationHandle handle, size_t size) {
    auto result = phi::dynload::cuMemRelease(handle);
    if (result == CUDA_SUCCESS) {
      cur_size_.fetch_sub(size);
    }
    return result;
  }

#endif
#else  // PADDLE_WITH_HIP
  hipError_t MemCreate(hipMemGenericAllocationHandle_t *handle,
                       size_t size,
                       const hipMemAllocationProp *prop,
                       unsigned long long flags) {  // NOLINT
    auto result = phi::dynload::hipMemCreate(handle, size, prop, flags);
    if (result == hipSuccess) {
      cur_size_.fetch_add(size);
    }
    return result;
  }

  hipError_t MemRelease(hipMemGenericAllocationHandle_t handle, size_t size) {
    auto result = phi::dynload::hipMemRelease(handle);
    if (result == hipSuccess) {
      cur_size_.fetch_sub(size);
    }
    return result;
  }

#endif

 private:
  const int dev_id_;
  const uint64_t limit_size_;
  std::atomic<uint64_t> cur_size_{0};

#if defined(PADDLE_WITH_CUDA) && (CUDA_VERSION >= 11020)
  cudaMemPool_t memPool_ = nullptr;
  static std::once_flag set_cudamempoolattr_once_flag_;
#endif
#if defined(PADDLE_WITH_HIP)
  hipMemPool_t memPool_ = nullptr;
  static std::once_flag set_cudamempoolattr_once_flag_;
#endif

  mutable std::unique_ptr<std::mutex> mtx_;
  static std::once_flag once_flag_;

  // just for testing
  std::set<void *> gpu_ptrs;
  std::mutex gpu_ptrs_mutex;
};  // NOLINT

std::once_flag RecordedGpuMallocHelper::once_flag_;

#if defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUDA) && (CUDA_VERSION >= 11020)
std::once_flag RecordedGpuMallocHelper::set_cudamempoolattr_once_flag_;
#endif

gpuError_t RecordedGpuMalloc(void **ptr,
                             size_t size,
                             int dev_id,
                             bool malloc_managed_memory) {
  return RecordedGpuMallocHelper::Instance(dev_id)->Malloc(
      ptr, size, malloc_managed_memory);
}

void RecordedGpuFree(void *p, size_t size, int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->Free(p, size);
}

gpuError_t RecordedGpuMallocAsync(void **ptr,
                                  size_t size,
                                  int dev_id,
                                  gpuStream_t stream) {
  return RecordedGpuMallocHelper::Instance(dev_id)->MallocAsync(
      ptr, size, stream);
}

void RecordedGpuFreeAsync(void *p,
                          size_t size,
                          int dev_id,
                          gpuStream_t stream) {
  return RecordedGpuMallocHelper::Instance(dev_id)->FreeAsync(p, size, stream);
}

#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10020
CUresult RecordedGpuMemCreate(CUmemGenericAllocationHandle *handle,
                              size_t size,
                              const CUmemAllocationProp *prop,
                              unsigned long long flags,  // NOLINT
                              int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->MemCreate(
      handle, size, prop, flags);
}

CUresult RecordedGpuMemRelease(CUmemGenericAllocationHandle handle,
                               size_t size,
                               int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->MemRelease(handle, size);
}
#endif
#else  // PADDLE_WITH_HIP
hipError_t RecordedGpuMemCreate(hipMemGenericAllocationHandle_t *handle,
                                size_t size,
                                const hipMemAllocationProp *prop,
                                unsigned long long flags,  // NOLINT
                                int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->MemCreate(
      handle, size, prop, flags);
}

hipError_t RecordedGpuMemRelease(hipMemGenericAllocationHandle_t handle,
                                 size_t size,
                                 int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->MemRelease(handle, size);
}
#endif

bool RecordedGpuMemGetInfo(size_t *avail,
                           size_t *total,
                           size_t *actual_avail,
                           size_t *actual_total,
                           int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->GetMemInfo(
      avail, total, actual_avail, actual_total);
}

uint64_t RecordedGpuMallocSize(int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->RecordedSize();
}

uint64_t RecordedGpuLimitSize(int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->LimitSize();
}

bool IsGpuMallocRecorded(int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->NeedRecord();
}

void EmptyCache() {
  std::vector<int> devices = GetSelectedDevices();
  for (auto device : devices) {
    memory::Release(phi::GPUPlace(device));
  }
}

bool IsGPUManagedMemorySupported(int dev_id) {
  return phi::backends::gpu::IsGPUManagedMemorySupported(dev_id);
}

bool IsGPUManagedMemoryOversubscriptionSupported(int dev_id) {
  return phi::backends::gpu::IsGPUManagedMemoryOversubscriptionSupported(
      dev_id);
}

void *GetGpuBasePtr(void *ptr, int dev_id) {
  return RecordedGpuMallocHelper::Instance(dev_id)->GetBasePtr(ptr);
}

int DnnVersion() { return phi::backends::gpu::DnnVersion(); }

int GetGPUDeviceCount() { return phi::backends::gpu::GetGPUDeviceCount(); }

int GetGPUComputeCapability(int id) {
  return phi::backends::gpu::GetGPUComputeCapability(id);
}

int GetGPURuntimeVersion(int id) {
  return phi::backends::gpu::GetGPURuntimeVersion(id);
}

int GetGPUDriverVersion(int id) {
  return phi::backends::gpu::GetGPUDriverVersion(id);
}

bool TensorCoreAvailable() { return phi::backends::gpu::TensorCoreAvailable(); }

int GetGPUMultiProcessors(int id) {
  return phi::backends::gpu::GetGPUMultiProcessors(id);
}

int GetGPUMaxThreadsPerMultiProcessor(int id) {
  return phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(id);
}

int GetGPUMaxThreadsPerBlock(int id) {
  return phi::backends::gpu::GetGPUMaxThreadsPerBlock(id);
}

int GetCurrentDeviceId() { return phi::backends::gpu::GetCurrentDeviceId(); }

std::array<unsigned int, 3> GetGpuMaxGridDimSize(int id) {
  return phi::backends::gpu::GetGpuMaxGridDimSize(id);
}

std::vector<int> GetSelectedDevices() {
  return phi::backends::gpu::GetSelectedDevices();
}

const gpuDeviceProp &GetDeviceProperties(int id) {
  return phi::backends::gpu::GetDeviceProperties(id);
}

void SetDeviceId(int device_id) { phi::backends::gpu::SetDeviceId(device_id); }

gpuError_t GpuGetLastError() { return phi::backends::gpu::GpuGetLastError(); }

void GpuStreamSync(gpuStream_t stream) {
  phi::backends::gpu::GpuStreamSync(stream);
}

void GpuDestroyStream(gpuStream_t stream) {
  phi::backends::gpu::GpuDestroyStream(stream);
}

void GpuDeviceSync() { phi::backends::gpu::GpuDeviceSync(); }

void GpuMemcpyAsync(void *dst,
                    const void *src,
                    size_t count,
                    gpuMemcpyKind kind,
                    gpuStream_t stream) {
  phi::backends::gpu::GpuMemcpyAsync(dst, src, count, kind, stream);
}

void GpuMemcpySync(void *dst,
                   const void *src,
                   size_t count,
                   gpuMemcpyKind kind) {
  phi::backends::gpu::GpuMemcpySync(dst, src, count, kind);
}

void GpuMemcpyPeerAsync(void *dst,
                        int dst_device,
                        const void *src,
                        int src_device,
                        size_t count,
                        gpuStream_t stream) {
  phi::backends::gpu::GpuMemcpyPeerAsync(
      dst, dst_device, src, src_device, count, stream);
}

void GpuMemcpyPeerSync(
    void *dst, int dst_device, const void *src, int src_device, size_t count) {
  phi::backends::gpu::GpuMemcpyPeerSync(
      dst, dst_device, src, src_device, count);
}

void GpuMemsetAsync(void *dst, int value, size_t count, gpuStream_t stream) {
  phi::backends::gpu::GpuMemsetAsync(dst, value, count, stream);
}

}  // namespace paddle::platform
