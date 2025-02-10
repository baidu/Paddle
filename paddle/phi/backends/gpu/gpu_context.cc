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

#include "paddle/phi/backends/gpu/gpu_context.h"

#include <algorithm>
#include <array>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/common/exception.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/cuda_stream.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/dynload/cusparse.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include "paddle/phi/backends/dynload/nccl.h"
#endif  // !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#endif  // !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#endif  // PADDLE_WITH_HIP

// NOTE: The paddle framework should add WITH_EIGEN option to support compile
// without eigen.
#include "unsupported/Eigen/CXX11/Tensor"

#include "paddle/phi/core/enforce.h"

namespace phi {

namespace internal {

class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice()
      : stream_(nullptr),
        allocator_(nullptr),
        device_prop_(nullptr),
        scratch_(nullptr),
        semaphore_(nullptr),
        allocations_() {
    Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override = default;

  void Reinitialize(gpuStream_t cuda_stream,
                    Allocator* allocator,
                    GPUPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    allocator_ = allocator;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const gpuStream_t& stream() const override { return stream_; }

  const gpuDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    auto buf = allocator_->Allocate(num_bytes);
    VLOG(4) << "Eigen allocated at " << buf->ptr() << " requested "
            << num_bytes;
    void* retv = buf->ptr();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.emplace(retv, std::move(buf));
    }
    return retv;
  }

  void deallocate(void* buffer) const override {
    if (LIKELY(buffer)) {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.erase(buffer);
    }
  }

  void* scratchpad() const override {
    if (scratch_ == nullptr) {
      scratch_ = allocate(Eigen::kGpuScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == nullptr) {
      char* scratch = static_cast<char*>(scratchpad()) + Eigen::kGpuScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream()));
#endif
    }
    return semaphore_;
  }

 private:
  GPUPlace place_;
  gpuStream_t stream_;                // not owned;
  Allocator* allocator_;              // not owned;
  const gpuDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void*, Allocator::AllocationPtr> allocations_;
};

#ifdef PADDLE_WITH_HIP
static void StreamCallbackFunc(gpuStream_t stream,
                               gpuError_t status,
                               void* user_data)
#endif
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
    static void CUDART_CB StreamCallbackFunc(void* user_data)
#else
    static void CUDART_CB
    StreamCallbackFunc(cudaStream_t stream, cudaError_t status, void* user_data)
#endif
#endif
{
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()>*>(user_data));
  (*func)();
}

}  // namespace internal

void DnnWorkspaceHandle::RunFuncSync(
    const std::function<void(void*)>& cudnn_func,
    size_t required_workspace_bytes,
    bool use_cached_allocation) {
  bool need_realloc = required_workspace_bytes > WorkspaceSize();
  if (need_realloc && !use_cached_allocation) {
    void* workspace_ptr = nullptr;
    size_t size = ((required_workspace_bytes + 255) >> 8) << 8;
    std::lock_guard<std::mutex> guard(*mtx_);
#ifdef PADDLE_WITH_HIP
    auto status = hipMalloc(&workspace_ptr, size);
#else
    auto status = cudaMalloc(&workspace_ptr, size);
#endif
    if (status == gpuSuccess) {
      cudnn_func(workspace_ptr);
      phi::backends::gpu::GpuStreamSync(stream_);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipFree(workspace_ptr));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
#endif
      return;
    }
  }

  RunFunc(cudnn_func, required_workspace_bytes);
  if (need_realloc) {
    // Release the workspace allocated in this running.
    ResetWorkspace();
  }
}

void DnnWorkspaceHandle::ResetWorkspace() { allocation_ = nullptr; }

void DnnWorkspaceHandle::ReallocWorkspace(size_t required_workspace_bytes) {
  if (required_workspace_bytes <= WorkspaceSize()) return;
  // reset allocation first before re-allocate to save memory
  allocation_.reset();
  allocation_ = allocator_->Allocate(required_workspace_bytes);
}

struct GPUContext::Impl {
  void Init() {
    owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    phi::InitGpuProperties(place_,
                           &compute_capability_,
                           &runtime_version_,
                           &driver_version_,
                           &multi_process_,
                           &max_threads_per_mp_,
                           &max_threads_per_block_,
                           &max_grid_dim_size_);
    stream_ = new CUDAStream(place_);
    InitEigenDevice();
    InitDnnWorkspace();
  }

  void PartialInitWithoutAllocator(int stream_priority) {
    owned_ = true;
    stream_owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    phi::InitGpuProperties(place_,
                           &compute_capability_,
                           &runtime_version_,
                           &driver_version_,
                           &multi_process_,
                           &max_threads_per_mp_,
                           &max_threads_per_block_,
                           &max_grid_dim_size_);
    stream_ = new CUDAStream(place_, stream_priority);
  }

  void PartialInitWithAllocator() {
    owned_ = true;
    stream_owned_ = true;
    backends::gpu::GPUDeviceGuard guard(place_.device);
    InitDnnWorkspace();
  }

  explicit Impl(const GPUPlace& place) : place_(place) {}

  ~Impl() {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    if (owned_) {
      DestroyInternalWorkspace();
      DestroyInternalEigenDevice();
      phi::DestroySparseHandle(sparse_handle_);
      phi::DestroySolverHandle(solver_handle_);
      phi::DestroyDnnHandle(dnn_handle_);
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      if (nccl_comm_) {
        // NOTE(liyurui): It is not recommend calling CUDA runtime API
        // in destructor. Since we can not ensure the release order of
        // static object, calling ncclCommDestroy in static object destructor
        // is a undefined behavior, CUDA driver may be already unloaded
        // from process.
        // If you really need to release the resource of nccl_comm,
        // try to get the nccl_comm out and use ncclCommDestroy outside.
      }
#endif
      phi::DestroyBlasHandle(blas_handle_);
      phi::DestroyBlasHandle(blas_tensor_core_handle_);
      phi::DestroyBlasHandle(blas_tf32_tensor_core_handle_);
      phi::DestroyBlasLtHandle(blaslt_handle_);
    }
    if (stream_owned_ && stream_) {
      delete stream_;
    }
  }

  const Place& GetPlace() const { return place_; }

  bool IsTensorCoreAvailable() const {
    return blas_tensor_core_handle_ != nullptr;
  }

  void InitDnnWorkspace() {
    PADDLE_ENFORCE_NOT_NULL(allocator_,
                            common::errors::InvalidArgument(
                                "The device allocator for GPU context is "
                                "nullptr. It must not be null."));
    workspace_ = new DnnWorkspaceHandle(allocator_, stream());
  }

  void DestroyInternalWorkspace() {
    if (owned_ && workspace_ != nullptr) {
      delete workspace_;
      workspace_ = nullptr;
    }
  }

  // TODO(wilber): The return type is a pointer, to be modified later.
  // DnnWorkspaceHandle* GetDnnWorkspace() {
  //   PD_CHECK(workspace_ != nullptr, "the gpu cudnn workspace is nullptr.");
  //   return workspace_;
  // }
  DnnWorkspaceHandle GetDnnWorkspace() {
    PADDLE_ENFORCE_NOT_NULL(allocator_,
                            common::errors::InvalidArgument(
                                "The device allocator for GPU context is "
                                "nullptr. It must not be null."));
    return DnnWorkspaceHandle(allocator_, stream());
  }

  void SetStream(gpuStream_t stream) {
    if (stream_ == nullptr) {
      auto s = Stream(reinterpret_cast<StreamId>(stream));
      stream_ = new CUDAStream(place_, s);
      stream_owned_ = true;
    }
    stream_->set_raw_stream(stream);
  }

  void SetCUDAStream(CUDAStream* stream, bool clear = true) {
    if (clear && stream_owned_ && stream_) {
      delete stream_;
    }
    stream_owned_ = false;
    stream_ = stream;
    // TODO(phi): reset related handles?
  }

  gpuStream_t stream() const {
    auto s = stream_->raw_stream();
    PADDLE_ENFORCE_NOT_NULL(
        s,
        common::errors::InvalidArgument(
            "The GPU stream is nullptr. It must not be null."));
    return s;
  }

  CUDAStream* cuda_stream() const {
    PADDLE_ENFORCE_NOT_NULL(
        stream_,
        common::errors::InvalidArgument(
            "The GPU stream is nullptr. It must not be null."));
    return stream_;
  }

  void InitEigenDevice() {
    PADDLE_ENFORCE_NOT_NULL(
        allocator_,
        common::errors::InvalidArgument(
            "The allocator for eigen device is nullptr. It must not be null."));
    eigen_stream_ = std::make_unique<internal::EigenGpuStreamDevice>();
    eigen_stream_->Reinitialize(stream(), allocator_, place_);
    eigen_device_ = new Eigen::GpuDevice(eigen_stream_.get());
  }

  void DestroyInternalEigenDevice() {
    if (owned_ && eigen_device_ != nullptr) {
      delete eigen_device_;
      eigen_device_ = nullptr;
    }
  }

  void SetEigenDevice(Eigen::GpuDevice* device) { eigen_device_ = device; }

  void SetEigenDevice(std::function<Eigen::GpuDevice*()>&& creator) {
    eigen_device_creator_ = std::move(creator);
  }

  Eigen::GpuDevice* eigen_device() {
    std::call_once(flag_eigen_device_, [&]() {
      if (!eigen_device_) {
        if (!eigen_device_creator_)
          InitEigenDevice();
        else
          eigen_device_ = eigen_device_creator_();
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        eigen_device_,
        common::errors::InvalidArgument(
            "The GPU eigen_device is nullptr. It must not be null."));
    return eigen_device_;
  }

  blasHandle_t GetBlasHandle() {
    std::call_once(flag_blas_, [&]() {
      if (!blas_handle_) {
        if (!blas_handle_creator_) {
          phi::InitBlasHandle(&blas_handle_, stream());
        } else {
          blas_handle_ = blas_handle_creator_();
        }
      }
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 9000
      if (!blas_tensor_core_handle_) {
        if (!blas_tensor_core_handle_creator_) {
          phi::InitBlasHandle(&blas_tensor_core_handle_, stream());
        } else {
          blas_tensor_core_handle_ = blas_tensor_core_handle_creator_();
        }
        PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
            blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
      }
#endif
#if CUDA_VERSION >= 11000
      if (!blas_tf32_tensor_core_handle_) {
        if (!blas_tf32_tensor_core_handle_creator_) {
          phi::InitBlasHandle(&blas_tf32_tensor_core_handle_, stream());
        } else {
          blas_tf32_tensor_core_handle_ =
              blas_tf32_tensor_core_handle_creator_();
        }
        PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
            blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
      }
#endif
#endif
    });
    PADDLE_ENFORCE_NOT_NULL(
        blas_handle_,
        common::errors::InvalidArgument(
            "The GPU blas handle is nullptr. It must not be null."));
    return blas_handle_;
  }

  void SetBlasHandle(blasHandle_t blas) { blas_handle_ = blas; }

  void SetBlasHandle(std::function<blasHandle_t()>&& handle_creator) {
    blas_handle_creator_ = std::move(handle_creator);
  }

  void SetBlasTensorCoreHandle(blasHandle_t handle) {
    blas_tensor_core_handle_ = handle;
  }

  void SetBlasTensorCoreHandle(std::function<blasHandle_t()>&& handle_creator) {
    blas_tensor_core_handle_creator_ = std::move(handle_creator);
  }

  void SetBlasTF32Handle(blasHandle_t handle) {
    blas_tf32_tensor_core_handle_ = handle;
  }

  void SetBlasTF32Handle(std::function<blasHandle_t()>&& handle_creator) {
    blas_tf32_tensor_core_handle_creator_ = std::move(handle_creator);
  }

  void SetBlasLtHandle(blasLtHandle_t blaslt) { blaslt_handle_ = blaslt; }

  void SetBlasLtHandle(std::function<blasLtHandle_t()>&& handle_creator) {
    blaslt_handle_creator_ = std::move(handle_creator);
  }

  blasLtHandle_t GetBlasLtHandle() {
    std::call_once(flag_blaslt_, [&]() {
      if (!blaslt_handle_) {
        if (!blaslt_handle_creator_)
          phi::InitBlasLtHandle(&blaslt_handle_);
        else
          blaslt_handle_ = blaslt_handle_creator_();
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        blaslt_handle_,
        common::errors::InvalidArgument(
            "The GPU blasLt handle is nullptr. It must not be null."));
    return blaslt_handle_;
  }

  dnnHandle_t GetDnnHandle() {
    std::call_once(flag_dnn_, [&]() {
      if (!dnn_handle_) {
        if (!dnn_handle_creator_) {
          phi::InitDnnHandle(&dnn_handle_, stream(), place_);
        } else {
          dnn_handle_ = dnn_handle_creator_();
        }
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        dnn_handle_,
        common::errors::InvalidArgument(
            "The GPU dnn handle is nullptr. It must not be null."));
    return dnn_handle_;
  }

  void DestroyInternalDnnHandle() {
#ifdef PADDLE_WITH_HIP
    if (owned_ && dnn_handle_ != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDestroy(dnn_handle_));
      dnn_handle_ = nullptr;
    }
#else
    if (owned_ && dnn_handle_ != nullptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDestroy(dnn_handle_));
      dnn_handle_ = nullptr;
    }
#endif  // PADDLE_WITH_HIP
  }

  void SetDnnHandle(dnnHandle_t handle) { dnn_handle_ = handle; }

  void SetDnnHandle(std::function<dnnHandle_t()>&& handle_creator) {
    dnn_handle_creator_ = std::move(handle_creator);
  }

  solverHandle_t GetSolverHandle() {
    std::call_once(flag_solver_, [&]() {
      if (!solver_handle_) {
        if (!solver_handle_creator_) {
          phi::InitSolverHandle(&solver_handle_, stream());
        } else {
          solver_handle_ = solver_handle_creator_();
        }
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        solver_handle_,
        common::errors::InvalidArgument(
            "The GPU solver handle is nullptr. It must not be null."));
    return solver_handle_;
  }

  void SetSolverHandle(solverHandle_t handle) { solver_handle_ = handle; }

  void SetSolverHandle(std::function<solverHandle_t()>&& handle_creator) {
    solver_handle_creator_ = std::move(handle_creator);
  }

  sparseHandle_t GetSparseHandle() {
    std::call_once(flag_sparse_, [&]() {
      if (!sparse_handle_) {
        if (!sparse_handle_creator_) {
          phi::InitSparseHandle(&sparse_handle_, stream());
        } else {
          sparse_handle_ = sparse_handle_creator_();
        }
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        sparse_handle_,
        common::errors::InvalidArgument(
            "The GPU sparse handle is nullptr. It must not be null."));
    return sparse_handle_;
  }

  void SetSparseHandle(sparseHandle_t handle) { sparse_handle_ = handle; }

  void SetSparseHandle(std::function<sparseHandle_t()>&& handle_creator) {
    sparse_handle_creator_ = std::move(handle_creator);
  }

  void Wait() const {
#ifdef PADDLE_WITH_HIP
    hipError_t e_sync = hipSuccess;
#if !defined(_WIN32)
    e_sync = hipStreamSynchronize(stream());
#else
    while (e_sync = hipStreamQuery(stream())) {
      if (e_sync == hipErrorNotReady) continue;
      break;
    }
#endif  // !defined(_WIN32)
#else   // PADDLE_WITH_HIP
    cudaError_t e_sync = cudaSuccess;
#if !defined(_WIN32)
    e_sync = cudaStreamSynchronize(stream());
#else
    while (e_sync = cudaStreamQuery(stream())) {
      if (e_sync == cudaErrorNotReady) continue;
      break;
    }
#endif  // !defined(_WIN32)
#endif  // PADDLE_WITH_HIP

    PADDLE_ENFORCE_GPU_SUCCESS(e_sync);
  }

  void WaitEvent(gpuEvent_t ev) const {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(stream(), ev, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(stream(), ev, 0));
#endif
  }

  ncclComm_t GetNcclComm() const {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // PD_CHECK(nccl_comm_ != nullptr, "the gpu nccl_comm is nullptr.");
    return nccl_comm_;
#endif
    return nullptr;
  }

  void SetNcclComm(ncclComm_t comm) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    nccl_comm_ = comm;
#endif
  }

  inline void CublasCall(const std::function<void(blasHandle_t)>& callback) {
    std::call_once(flag_cublas_, [&]() {
      if (!blas_handle_) {
        if (!blas_handle_creator_) {
          phi::InitBlasHandle(&blas_handle_, stream());
        } else {
          blas_handle_ = blas_handle_creator_();
        }
      }
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 9000
      if (!blas_tensor_core_handle_) {
        if (!blas_tensor_core_handle_creator_) {
          phi::InitBlasHandle(&blas_tensor_core_handle_, stream());
        } else {
          blas_tensor_core_handle_ = blas_tensor_core_handle_creator_();
        }
        PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
            blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
      }
#endif
#if CUDA_VERSION >= 11000
      if (!blas_tf32_tensor_core_handle_) {
        if (!blas_tf32_tensor_core_handle_creator_) {
          phi::InitBlasHandle(&blas_tf32_tensor_core_handle_, stream());
        } else {
          blas_tf32_tensor_core_handle_ =
              blas_tf32_tensor_core_handle_creator_();
        }
        PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
            blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
      }
#endif
#endif
    });
    if (blas_tf32_tensor_core_handle_ && phi::AllowTF32Cublas()) {
      std::lock_guard<std::mutex> guard(blas_tf32_mtx_);
      callback(blas_tf32_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  inline void TensorCoreCublasCallIfAvailable(
      const std::function<void(blasHandle_t)>& callback) {
    std::call_once(flag_tensorcore_cublas_, [&]() {
      if (!blas_handle_) {
        if (!blas_handle_creator_) {
          phi::InitBlasHandle(&blas_handle_, stream());
        } else {
          blas_handle_ = blas_handle_creator_();
        }
      }
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 9000
      if (!blas_tensor_core_handle_) {
        if (!blas_tensor_core_handle_creator_) {
          phi::InitBlasHandle(&blas_tensor_core_handle_, stream());
        } else {
          blas_tensor_core_handle_ = blas_tensor_core_handle_creator_();
        }
        PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
            blas_tensor_core_handle_, CUBLAS_TENSOR_OP_MATH));
      }
#endif
#if CUDA_VERSION >= 11000
      if (!blas_tf32_tensor_core_handle_) {
        if (!blas_tf32_tensor_core_handle_creator_) {
          phi::InitBlasHandle(&blas_tf32_tensor_core_handle_, stream());
        } else {
          blas_tf32_tensor_core_handle_ =
              blas_tf32_tensor_core_handle_creator_();
        }
        PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
            blas_tf32_tensor_core_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
      }
#endif
#endif
    });
    if (blas_tensor_core_handle_ != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tensor_core_mtx_);
      callback(blas_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  inline void CusparseCall(
      const std::function<void(sparseHandle_t)>& callback) {
    std::call_once(flag_sparse_, [&]() {
      if (!sparse_handle_) {
        if (!sparse_handle_creator_) {
          phi::InitSparseHandle(&sparse_handle_, stream());
        } else {
          sparse_handle_ = sparse_handle_creator_();
        }
      }
    });
    std::lock_guard<std::mutex> guard(sparse_mtx_);
    callback(sparse_handle_);
  }

  void RecordEvent(gpuEvent_t ev, const std::function<void()>& callback) const {
    callback();
    RecordEvent(ev);
  }

  void RecordEvent(gpuEvent_t ev) const {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(ev, stream()));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(ev, stream()));
#endif
  }

  void AddStreamCallback(const std::function<void()>& callback) const {
    // NOTE(zhiqiu): better use threadpool here, otherwise "std::async" may
    // launch too many threads and result in thread oversubscription.
    auto* callback_func = new std::function<void()>(callback);
    auto* func = new std::function<void()>([this, callback_func] {
      std::lock_guard<std::mutex> lock(stream_call_back_mtx_);
      VLOG(4) << "Stream callback";
      last_future_ = std::async(std::launch::async, [callback_func]() {
        std::unique_ptr<std::function<void()>> releaser(callback_func);
        (*callback_func)();
      });
    });

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipStreamAddCallback(stream(), internal::StreamCallbackFunc, func, 0));
#endif
#ifdef PADDLE_WITH_CUDA
#if CUDA_VERSION >= 10000
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaLaunchHostFunc(stream(), internal::StreamCallbackFunc, func));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamAddCallback(stream(), internal::StreamCallbackFunc, func, 0));
#endif
#endif
  }

  void WaitStreamCallback() const {
#if defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_CUDA)
    phi::backends::gpu::GpuStreamSync(stream());
#endif
    {
      std::lock_guard<std::mutex> lock(stream_call_back_mtx_);
      if (last_future_.valid()) {
        last_future_.wait();
      }
    }
  }

  bool HasDnnAttr(const std::string& attr_name) const {
    return dnn_attrs_.count(attr_name) != 0UL;
  }

  const Attribute& GetDnnAttr(const std::string& attr_name) const {
    auto iter = dnn_attrs_.find(attr_name);
    PADDLE_ENFORCE_NE(iter,
                      dnn_attrs_.end(),
                      common::errors::NotFound(
                          "Attribute `%s` is not found in OneDNNContext."));
    return iter->second;
  }

  void SetDnnAttr(const std::string& attr_name, Attribute attr) {
    dnn_attrs_[attr_name] = attr;
  }

  void ClearDnnAttr() { dnn_attrs_.clear(); }

  // use one flag for all handles?
  // they should be accessed consistently
  bool owned_{false};
  bool stream_owned_{false};
  Place place_;
  int compute_capability_ = 0;
  int runtime_version_ = 0;
  int driver_version_ = 0;
  int multi_process_ = 0;
  int max_threads_per_mp_ = 0;
  int max_threads_per_block_ = 0;
  std::array<unsigned int, 3> max_grid_dim_size_;

  CUDAStream* stream_{nullptr};
  Eigen::GpuDevice* eigen_device_{nullptr};
  std::function<Eigen::GpuDevice*()> eigen_device_creator_{nullptr};
  blasHandle_t blas_handle_{nullptr};
  std::function<blasHandle_t()> blas_handle_creator_{nullptr};
  blasHandle_t blas_tensor_core_handle_{nullptr};
  std::function<blasHandle_t()> blas_tensor_core_handle_creator_{nullptr};
  blasHandle_t blas_tf32_tensor_core_handle_{nullptr};
  std::function<blasHandle_t()> blas_tf32_tensor_core_handle_creator_{nullptr};
  blasLtHandle_t blaslt_handle_{nullptr};
  std::function<blasLtHandle_t()> blaslt_handle_creator_{nullptr};
  dnnHandle_t dnn_handle_{nullptr};
  std::function<dnnHandle_t()> dnn_handle_creator_{nullptr};
  solverHandle_t solver_handle_{nullptr};
  std::function<solverHandle_t()> solver_handle_creator_{nullptr};
  sparseHandle_t sparse_handle_{nullptr};
  std::function<sparseHandle_t()> sparse_handle_creator_{nullptr};
  DnnWorkspaceHandle* workspace_{nullptr};

  std::once_flag flag_sparse_;
  std::once_flag flag_blas_;
  std::once_flag flag_blaslt_;
  std::once_flag flag_dnn_;
  std::once_flag flag_solver_;
  std::once_flag flag_cublas_;
  std::once_flag flag_tensorcore_cublas_;
  std::once_flag flag_eigen_device_;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  // NCCL communicator (single process version) for NCCL collective operations.
  // NCCL collective operations provides fast collectives over multiple GPUs
  // both within and across nodes.
  // But, this collectives is used for collectives over multiple GPUs within
  // nodes.

  // NOTE: Distributed communicator, distributed framework manages its
  // resources.
  ncclComm_t nccl_comm_{nullptr};
#endif

  mutable std::mutex blas_mtx_;
  mutable std::mutex blas_tensor_core_mtx_;
  mutable std::mutex blas_tf32_mtx_;
  mutable std::mutex sparse_mtx_;
  mutable std::mutex stream_call_back_mtx_;
  mutable std::future<void> last_future_;

  Allocator* allocator_{nullptr};  // external resource.
  // A internal resource to initinalize eigen_device.
  std::unique_ptr<internal::EigenGpuStreamDevice> eigen_stream_{nullptr};

  // Holds some attributes only used by the gpudnn kernel calculation
  // Because DeviceContext is a global singleton, you need to ensure thread
  // safety, use the thread_local variable
  static thread_local AttributeMap dnn_attrs_;
};

thread_local AttributeMap GPUContext::Impl::dnn_attrs_ = {};

GPUContext::GPUContext(GPUContext&&) = default;  // NOLINT

GPUContext& GPUContext::operator=(GPUContext&&) = default;  // NOLINT

GPUContext::GPUContext(const GPUPlace& place, bool init, int stream_priority)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {
  if (init) {
    impl_->PartialInitWithoutAllocator(stream_priority);
  }
}

GPUContext::~GPUContext() = default;

const Place& GPUContext::GetPlace() const { return impl_->GetPlace(); }

gpuStream_t GPUContext::stream() const { return impl_->stream(); }

CUDAStream* GPUContext::cuda_stream() const { return impl_->cuda_stream(); }

dnnHandle_t GPUContext::cudnn_handle() const { return impl_->GetDnnHandle(); }

blasHandle_t GPUContext::cublas_handle() const {
  return impl_->GetBlasHandle();
}

blasLtHandle_t GPUContext::cublaslt_handle() const {
  return impl_->GetBlasLtHandle();
}

solverHandle_t GPUContext::cusolver_dn_handle() const {
  return impl_->GetSolverHandle();
}

sparseHandle_t GPUContext::cusparse_handle() const {
  return impl_->GetSparseHandle();
}

void GPUContext::Wait() const { impl_->Wait(); }

void GPUContext::WaitEvent(gpuEvent_t ev) const { impl_->WaitEvent(ev); }

bool GPUContext::tensor_core_available() const {
  return impl_->IsTensorCoreAvailable();
}

int GPUContext::GetComputeCapability() const {
  return impl_->compute_capability_;
}

int GPUContext::GetMaxPhysicalThreadCount() const {
  return impl_->multi_process_ * impl_->max_threads_per_mp_;
}

int GPUContext::GetSMCount() const { return impl_->multi_process_; }

int GPUContext::GetMaxThreadsPerBlock() const {
  return impl_->max_threads_per_block_;
}

std::array<unsigned int, 3> GPUContext::GetCUDAMaxGridDimSize() const {
  return impl_->max_grid_dim_size_;
}

Eigen::GpuDevice* GPUContext::eigen_device() const {
  return impl_->eigen_device();
}

DnnWorkspaceHandle GPUContext::cudnn_workspace_handle() const {
  return impl_->GetDnnWorkspace();
}

void GPUContext::CublasCall(
    const std::function<void(blasHandle_t)>& callback) const {
  impl_->CublasCall(callback);
}

void GPUContext::TensorCoreCublasCallIfAvailable(
    const std::function<void(blasHandle_t)>& callback) const {
  impl_->TensorCoreCublasCallIfAvailable(callback);
}

void GPUContext::CusparseCall(
    const std::function<void(sparseHandle_t)>& callback) const {
  impl_->CusparseCall(callback);
}

void GPUContext::RecordEvent(gpuEvent_t ev,
                             const std::function<void()>& callback) const {
  impl_->RecordEvent(ev, callback);
}

void GPUContext::RecordEvent(gpuEvent_t ev) const { impl_->RecordEvent(ev); }

void GPUContext::AddStreamCallback(
    const std::function<void()>& callback) const {
  impl_->AddStreamCallback(callback);
}

void GPUContext::WaitStreamCallback() const { impl_->WaitStreamCallback(); }

ncclComm_t GPUContext::nccl_comm() const { return impl_->GetNcclComm(); }

void GPUContext::set_nccl_comm(ncclComm_t comm) { impl_->SetNcclComm(comm); }

void GPUContext::Init() {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());  // NOLINT
  impl_->Init();
}

void GPUContext::SetStream(gpuStream_t stream) {
#if !defined(_WIN32)
  this->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                         .GetAllocator(impl_->GetPlace(), stream)
                         .get());
#endif
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());  // NOLINT
  impl_->SetStream(stream);
}

void GPUContext::SetCUDAStream(CUDAStream* stream, bool clear) {
#if !defined(_WIN32)
  this->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                         .GetAllocator(stream->place(), stream->raw_stream())
                         .get());
#endif
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());  // NOLINT
  impl_->SetCUDAStream(stream, clear);
}

void GPUContext::SetEigenDevice(Eigen::GpuDevice* device) {
  impl_->SetEigenDevice(device);
}

void GPUContext::SetEigenDevice(std::function<Eigen::GpuDevice*()>&& creator) {
  impl_->SetEigenDevice(std::move(creator));
}

void GPUContext::SetBlasHandle(blasHandle_t blas) {
  impl_->SetBlasHandle(blas);
}

void GPUContext::SetBlasHandle(std::function<blasHandle_t()>&& func) {
  impl_->SetBlasHandle(std::move(func));
}

void GPUContext::SetBlasTensorCoreHandle(blasHandle_t handle) {
  impl_->SetBlasTensorCoreHandle(handle);
}

void GPUContext::SetBlasTensorCoreHandle(std::function<blasHandle_t()>&& func) {
  impl_->SetBlasTensorCoreHandle(std::move(func));
}

void GPUContext::SetBlasTF32Handle(blasHandle_t handle) {
  impl_->SetBlasTF32Handle(handle);
}

void GPUContext::SetBlasTF32Handle(std::function<blasHandle_t()>&& func) {
  impl_->SetBlasTF32Handle(std::move(func));
}

void GPUContext::SetBlasLtHandle(blasLtHandle_t blaslt) {
  impl_->SetBlasLtHandle(blaslt);
}

void GPUContext::SetBlasLtHandle(std::function<blasLtHandle_t()>&& func) {
  impl_->SetBlasLtHandle(std::move(func));
}

void GPUContext::SetDnnHandle(dnnHandle_t handle) {
  impl_->SetDnnHandle(handle);
}

void GPUContext::SetDnnHandle(std::function<dnnHandle_t()>&& func) {
  impl_->SetDnnHandle(std::move(func));
}

void GPUContext::SetSolverHandle(solverHandle_t handle) {
  impl_->SetSolverHandle(handle);
}

void GPUContext::SetSolverHandle(std::function<solverHandle_t()>&& func) {
  impl_->SetSolverHandle(std::move(func));
}

void GPUContext::SetSparseHandle(sparseHandle_t handle) {
  impl_->SetSparseHandle(handle);
}

void GPUContext::SetSparseHandle(std::function<sparseHandle_t()>&& func) {
  impl_->SetSparseHandle(std::move(func));
}

void GPUContext::SetDnnWorkspaceHandle(DnnWorkspaceHandle* handle) {
  impl_->workspace_ = handle;
}

void GPUContext::PartialInitWithoutAllocator(int stream_priority) {
  impl_->PartialInitWithoutAllocator(stream_priority);
}

void GPUContext::PartialInitWithAllocator() {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());  // NOLINT
  impl_->PartialInitWithAllocator();
}

void GPUContext::SetComputeCapability(int val) {
  impl_->compute_capability_ = val;
}

void GPUContext::SetMaxThreadsPerMultiProcessor(int val) {
  impl_->max_threads_per_mp_ = val;
}

void GPUContext::SetMultiProcessors(int val) { impl_->multi_process_ = val; }

void GPUContext::SetMaxThreadsPerBlock(int val) {
  impl_->max_threads_per_block_ = val;
}

void GPUContext::SetMaxGridDimSize(const std::array<unsigned int, 3>& val) {
  impl_->max_grid_dim_size_ = val;
}

void GPUContext::SetDriverVersion(int val) { impl_->driver_version_ = val; }

void GPUContext::SetRuntimeVersion(int val) { impl_->runtime_version_ = val; }

bool GPUContext::HasDnnAttr(const std::string& attr_name) const {
  return impl_->HasDnnAttr(attr_name);
}

const Attribute& GPUContext::GetDnnAttr(const std::string& attr_name) const {
  return impl_->GetDnnAttr(attr_name);
}

void GPUContext::SetDnnAttr(const std::string& attr_name, Attribute attr) {
  return impl_->SetDnnAttr(attr_name, std::move(attr));
}

void GPUContext::ClearDnnAttr() { return impl_->ClearDnnAttr(); }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
GPUPinnedContext::GPUPinnedContext() {
  eigen_device_ = std::make_unique<Eigen::DefaultDevice>();
}

GPUPinnedContext::GPUPinnedContext(GPUPinnedPlace place) : place_(place) {
  eigen_device_ = std::make_unique<Eigen::DefaultDevice>();
}

Eigen::DefaultDevice* GPUPinnedContext::eigen_device() const {
  return eigen_device_.get();
}

const Place& GPUPinnedContext::GetPlace() const { return place_; }
#endif

}  // namespace phi
