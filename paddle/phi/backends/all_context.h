/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <future>  // NOLINT
#include <map>
#include <memory>

// Note: Some scenarios need to include all types of Context declarations.
// In order to avoid including the header files of each backend in turn,
// add this header file
// Note: Limit the entry of DeviceContext to backends to avoid multiple include
// path replacement after implementing phi DeviceContext

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/macros.h"

namespace phi {

template <typename Place>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<phi::CPUPlace> {
  using TYPE = phi::CPUContext;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
struct DefaultDeviceContextType<platform::CUDAPlace> {
  using TYPE = phi::GPUContext;
};
#endif

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<platform::Place>& places,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority);

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  static DeviceContextPool& Instance();

  /*! \brief  Create should only called by Init function */
  static DeviceContextPool& Init(const std::vector<phi::Place>& places);

  static bool IsInitialized();

  static void SetPool(DeviceContextPool* dev_pool);

  /*! \brief  Return handle of single device context. */
  phi::DeviceContext* Get(const phi::Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

  size_t size() const;

  const std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>&
  device_contexts() const;

  static void SetDeviceContexts(
      const std::map<Place,
                     std::shared_future<std::unique_ptr<DeviceContext>>>*);

 private:
  explicit DeviceContextPool(const std::vector<phi::Place>& places);

  std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>
      device_contexts_;
  static thread_local const std::
      map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
          external_device_contexts_;  // not owned
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace phi
