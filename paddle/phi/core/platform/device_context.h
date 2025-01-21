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
#pragma once

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#include "paddle/phi/core/platform/device_defs.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/dynload/cusparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include "paddle/phi/backends/dynload/nccl.h"
#endif
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#include "paddle/phi/backends/gpu/gpu_context.h"  // NOLINT
#include "paddle/phi/backends/gpu/gpu_helper.h"   // NOLINT
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#endif
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"  // NOLINT
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "xpu/bkcl.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#endif

#include <map>

#include "glog/logging.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/stream.h"

#if !defined(PADDLE_WITH_XPU_KP) || defined(__xpu_on_host__)
#include "unsupported/Eigen/CXX11/Tensor"
#endif

namespace Eigen {
struct DefaultDevice;
struct GpuDevice;
}  // namespace Eigen

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif

namespace paddle {
namespace platform {

// Graphcore IPU
#ifdef PADDLE_WITH_IPU
class IPUDeviceContext
    : public DeviceContext,
      public phi::TypeInfoTraits<DeviceContext, IPUDeviceContext> {
 public:
  IPUDeviceContext() = delete;
  explicit IPUDeviceContext(IPUPlace place);
  virtual ~IPUDeviceContext();
  Eigen::DefaultDevice* eigen_device() const { return nullptr; }
  const Place& GetPlace() const override;
  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  static const char* name() { return "IPUDeviceContext"; }

 private:
  IPUPlace place_;
};
#endif

#ifdef PADDLE_WITH_XPU
namespace xpu = baidu::xpu::api;
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
using CUDAPinnedDeviceContext = phi::GPUPinnedContext;
#endif

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<phi::Place>& places,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority);

}  // namespace platform
}  // namespace paddle

namespace phi {

#ifdef PADDLE_WITH_IPU
template <>
struct DefaultDeviceContextType<phi::IPUPlace> {
  using TYPE = paddle::platform::IPUDeviceContext;
};
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
struct DefaultDeviceContextType<phi::GPUPinnedPlace> {
  using TYPE = paddle::platform::CUDAPinnedDeviceContext;
};
#endif

}  //  namespace phi
