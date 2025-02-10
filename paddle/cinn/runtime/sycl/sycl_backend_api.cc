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

#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
#include <glog/logging.h>

namespace cinn {
namespace runtime {
namespace sycl {
SYCLBackendAPI* SYCLBackendAPI::Global() {
  static auto* inst = new SYCLBackendAPI();
  return inst;
}

void SYCLBackendAPI::Init(Arch arch) {
  if (initialized_) return;
  auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
  if (devices.size() == 0) {
    std::cerr << "No valid gpu device found!";
  }
  // Target::Arch -> sycl::backend
  ::sycl::backend backend;
  arch.Match(
      [&](common::UnknownArch) {
        SYCL_CALL(backend = ::sycl::device::get_devices(
                                ::sycl::info::device_type::gpu)[0]
                                .get_backend());
      },
      [&](common::X86Arch) { CINN_NOT_IMPLEMENTED },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED },
      [&](common::NVGPUArch) { backend = ::sycl::backend::ext_oneapi_cuda; },
      [&](common::HygonDCUArchHIP) { CINN_NOT_IMPLEMENTED },
      [&](common::HygonDCUArchSYCL) {
        backend = ::sycl::backend::ext_oneapi_hip;
      });
  // look for matched devices
  if (this->devices.size() < 8) {
    for (auto device : devices) {
      if (device.get_backend() == backend) {
        this->devices.push_back(device);
      }
    }
  }
  if (this->devices.size() == 0) {
    std::cerr << "No valid gpu device matched given arch \n";
  }
  this->contexts.resize(this->devices.size(), nullptr);
  this->queues.resize(this->devices.size());
  initialized_ = true;
}

void SYCLBackendAPI::set_device(int device_id) {
  if (!initialized_) Init(common::UnknownArch{});
  PADDLE_ENFORCE_GE(device_id,
                    0UL,
                    ::common::errors::InvalidArgument(
                        "please set valid device id! device id", device_id));
  PADDLE_ENFORCE_LE(
      device_id,
      this->devices.size() - 1,
      ::common::errors::InvalidArgument("set valid device id! device id: ",
                                        device_id,
                                        " > max device id:",
                                        this->devices.size() - 1));
  if (this->contexts[device_id] == nullptr) {
    auto exception_handler = [](::sycl::exception_list exceptions) {
      for (const std::exception_ptr& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const ::sycl::exception& e) {
          PADDLE_THROW(::common::errors::Fatal(
              "Caught asynchronous SYCL exception:\n %s ", e.what()));
        }
      }
    };
    ::sycl::property_list q_prop{
        ::sycl::property::queue::in_order()};  // In order queue
    // create context and queue
    this->contexts[device_id] =
        new ::sycl::context(this->devices[device_id], exception_handler);
    // one device one queue
    this->queues[device_id].push_back(new ::sycl::queue(
        *this->contexts[device_id], this->devices[device_id], q_prop));
  }
  this->now_device_id = device_id;
}

int SYCLBackendAPI::get_device() { return this->now_device_id; }

int SYCLBackendAPI::get_device_property(DeviceProperty device_property,
                                        std::optional<int> device_id) {
  int index = device_id.value_or(this->now_device_id);
  int rv = -1;

  switch (device_property) {
    case DeviceProperty::MaxBlockDimX: {
      ::sycl::_V1::id<3> max_work_item_sizes =
          this->devices[index]
              .get_info<::sycl::_V1::info::device::max_work_item_sizes<3>>();
      rv = max_work_item_sizes[0];
      break;
    }
    case DeviceProperty::MaxBlockDimY: {
      ::sycl::_V1::id<3> max_work_item_sizes =
          this->devices[index]
              .get_info<::sycl::_V1::info::device::max_work_item_sizes<3>>();
      rv = max_work_item_sizes[1];
      break;
    }
    case DeviceProperty::MaxBlockDimZ: {
      ::sycl::_V1::id<3> max_work_item_sizes =
          this->devices[index]
              .get_info<::sycl::_V1::info::device::max_work_item_sizes<3>>();
      rv = max_work_item_sizes[2];
      break;
    }
    case DeviceProperty::MaxGridDimX: {
      rv = 2097151;
      break;
    }
    case DeviceProperty::MaxGridDimY: {
      rv = 2097151;
      break;
    }
    case DeviceProperty::MaxGridDimZ: {
      rv = 2097151;
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      rv =
          this->devices[index].get_info<::sycl::info::device::local_mem_size>();
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      rv = this->devices[index]
               .get_info<::sycl::info::device::max_work_group_size>();
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      rv = this->devices[index]
               .get_info<::sycl::info::device::max_work_group_size>();
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      rv = this->devices[index]
               .get_info<::sycl::info::device::max_compute_units>();
      break;
    }
    case DeviceProperty::MaxBlocksPerSM: {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "SYCL Not supported device property : MaxBlocksPerSM !"));
      break;
    }
    case DeviceProperty::WarpSize: {
      std::vector<size_t> sub_group_sizes =
          this->devices[index]
              .get_info<::sycl::info::device::sub_group_sizes>();
      size_t max_sub_group_size =
          *max_element(std::begin(sub_group_sizes), std::end(sub_group_sizes));
      rv = static_cast<int>(max_sub_group_size);
      break;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "SYCL Not supported device property !"));
  }
  return rv;
}

void* SYCLBackendAPI::malloc(size_t numBytes) {
  VLOG(3) << "sycl malloc";
  void* dev_mem = nullptr;
  SYCL_CALL(dev_mem = ::sycl::malloc_device(numBytes,
                                            this->devices[now_device_id],
                                            *this->contexts[now_device_id]));
  PADDLE_ENFORCE_NE(dev_mem,
                    nullptr,
                    ::common::errors::InvalidArgument(
                        "allocate sycl device memory failure!"));
  return dev_mem;
}

void SYCLBackendAPI::free(void* data) {
  VLOG(3) << "sycl free";
  SYCL_CALL(::sycl::free(data, *this->contexts[now_device_id]));
}

void SYCLBackendAPI::memset(void* data, int value, size_t numBytes) {
  VLOG(3) << "sycl memset";
  SYCL_CALL(
      this->queues[now_device_id][0]->memset(data, value, numBytes).wait());
}

void SYCLBackendAPI::memcpy(void* dest,
                            const void* src,
                            size_t numBytes,
                            MemcpyType type) {
  VLOG(3) << "sycl memcpy";
  ::sycl::queue* Q;
  switch (type) {
    case MemcpyType::HostToHost:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::HostToDevice:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::DeviceToHost:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::DeviceToDevice:
      Q = this->queues[now_device_id][0];
      break;
  }
  SYCL_CALL(Q->memcpy(dest, src, numBytes).wait());
}

void SYCLBackendAPI::device_sync() {
  VLOG(3) << "sycl device sync";
  for (auto queues_in_one_device : this->queues) {
    for (auto queue : queues_in_one_device) {
      SYCL_CALL(queue->wait_and_throw());
    }
  }
}

void SYCLBackendAPI::stream_sync(void* stream) {
  VLOG(3) << "sycl stream sync";
  SYCL_CALL(static_cast<::sycl::queue*>(stream)->wait_and_throw());
}

::sycl::queue* SYCLBackendAPI::get_now_queue() {
  return this->queues[now_device_id][0];
}

std::string SYCLBackendAPI::GetGpuVersion() {
  ::sycl::device device = this->devices[now_device_id];
  ::sycl::backend backend = device.get_backend();
  switch (backend) {
    case ::sycl::backend::cuda: {
      std::string gpu_version = "sm_";
      std::string version_with_point =
          device.get_info<::sycl::info::device::driver_version>();
      size_t pos = version_with_point.find(".");
      if (pos != std::string::npos) {
        gpu_version +=
            version_with_point.substr(0, pos) +
            version_with_point.substr(pos + 1, version_with_point.size());
      }
      return gpu_version;
    }
    case ::sycl::backend::ext_oneapi_hip: {
      std::string gpu_version =
          device.get_info<::sycl::info::device::version>();
      size_t pos = gpu_version.find(":");
      if (pos != std::string::npos) gpu_version = gpu_version.substr(0, pos);
      return gpu_version;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Error! use unknown sycl backend!"));
  }
}

std::array<int, 3> SYCLBackendAPI::get_max_block_dims(
    std::optional<int> device_id) {
  std::array<int, 3> kMaxBlockDims;
  int index = device_id.value_or(this->now_device_id);
  ::sycl::_V1::id<3> max_work_item_sizes =
      this->devices[index]
          .get_info<::sycl::_V1::info::device::max_work_item_sizes<3>>();
  kMaxBlockDims = std::array<int, 3>{
      max_work_item_sizes[2], max_work_item_sizes[1], max_work_item_sizes[0]};
  return kMaxBlockDims;
}

std::array<int, 3> SYCLBackendAPI::get_max_grid_dims(
    std::optional<int> device_id) {
  std::array<int, 3> kMaxGridDims;
  kMaxGridDims = std::array<int, 3>{2097151, 2097151, 2097151};
  return kMaxGridDims;
}

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
