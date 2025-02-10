// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#ifdef CINN_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif

#include <glog/logging.h>

#include <regex>
#include <sstream>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/arch_util.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/common/enforce.h"
using cinn::runtime::BackendAPI;

namespace cinn {
namespace common {

Target::Target(OS o,
               Arch a,
               Bit b,
               const std::vector<Feature> &features,
               const std::vector<Lib> &libs)
    : os(o), arch(a), bits(b), features(features), libs(libs) {
  // check compile option
  arch.Match([&](UnknownArch) {},
             [&](X86Arch) {},
             [&](ARMArch) {},
             [&](NVGPUArch) {
#ifndef CINN_WITH_CUDA
               PADDLE_THROW(::common::errors::Unimplemented(
                   "Please recompile with flag WITH_GPU and WITH_CINN."));
#endif
             },
             [&](HygonDCUArchHIP) {
#ifndef CINN_WITH_HIP
               PADDLE_THROW(::common::errors::Unimplemented(
                   "Please recompile with flag WITH_ROCM and WITH_CINN."));
#endif
             },
             [&](HygonDCUArchSYCL) {
#ifndef CINN_WITH_SYCL
               PADDLE_THROW(::common::errors::Unimplemented(
                   "Please recompile with flag CINN_WITH_SYCL and WITH_CINN."));
#endif
             });
}

bool Target::operator==(const Target &other) const {
  return os == other.os &&      //
         arch == other.arch &&  //
         bits == other.bits &&  //
         features == other.features;
}

int GetRuntimeArchImpl(UnknownArch) { return cinn_unk_device; }

int GetRuntimeArchImpl(X86Arch) { return cinn_x86_device; }

int GetRuntimeArchImpl(ARMArch) { return cinn_arm_device; }

int GetRuntimeArchImpl(NVGPUArch) {
  PADDLE_THROW(::common::errors::InvalidArgument("Not supported arch"));
}

int GetRuntimeArchImpl(HygonDCUArchHIP) { CINN_NOT_IMPLEMENTED }

int GetRuntimeArchImpl(HygonDCUArchSYCL) { CINN_NOT_IMPLEMENTED }

int GetRuntimeArch(Arch arch) {
  return std::visit([](const auto &impl) { return GetRuntimeArchImpl(impl); },
                    arch.variant());
}

int Target::runtime_arch() const { return GetRuntimeArch(arch); }

int GetMaxNumThreadsImpl(UnknownArch arch) {
  LOG(FATAL) << "The target is not GPU! Cannot get max number of threads.";
}

int GetMaxNumThreadsImpl(X86Arch arch) {
  LOG(FATAL) << "The target is not GPU! Cannot get max number of threads.";
}

int GetMaxNumThreadsImpl(ARMArch arch) {
  LOG(FATAL) << "The target is not GPU! Cannot get max number of threads.";
}

int GetMaxNumThreadsImpl(NVGPUArch arch) { return 1024; }

int GetMaxNumThreadsImpl(HygonDCUArchHIP arch) { return 1024; }

int GetMaxNumThreadsImpl(HygonDCUArchSYCL arch) { return 1024; }

int GetMaxNumThreads(Arch arch) {
  return std::visit([](const auto &impl) { return GetMaxNumThreadsImpl(impl); },
                    arch.variant());
}

int Target::max_num_threads() const { return GetMaxNumThreads(arch); }

int GetMultiProcessCountImpl(UnknownArch arch) {
  LOG(FATAL) << "The target is not GPU! Cannot get multi processor count.";
}

int GetMultiProcessCountImpl(X86Arch arch) {
  LOG(FATAL) << "The target is not GPU! Cannot get multi processor count.";
}

int GetMultiProcessCountImpl(ARMArch arch) {
  LOG(FATAL) << "The target is not GPU! Cannot get multi processor count.";
}

int GetMultiProcessCountImpl(NVGPUArch arch) {
  int num_sm = 0;
#ifdef CINN_WITH_CUDA
  cudaDeviceGetAttribute(
      &num_sm, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, 0);
#endif
  return num_sm;
}

int GetMultiProcessCountImpl(HygonDCUArchHIP arch) {
  return BackendAPI::get_backend(arch)->get_device_property(
      BackendAPI::DeviceProperty::MultiProcessorCount);
}

int GetMultiProcessCountImpl(HygonDCUArchSYCL arch) {
  return BackendAPI::get_backend(arch)->get_device_property(
      BackendAPI::DeviceProperty::MultiProcessorCount);
}

int GetMultiProcessCount(Arch arch) {
  return std::visit(
      [](const auto &impl) { return GetMultiProcessCountImpl(impl); },
      arch.variant());
}

int Target::get_multi_processor_count() const {
  return GetMultiProcessCount(arch);
}

int GetMaxThreadsPerSmImpl(UnknownArch arch) {
  LOG(FATAL)
      << "The target is not GPU! Cannot get max threads per stream processor";
}

int GetMaxThreadsPerSmImpl(X86Arch arch) {
  LOG(FATAL)
      << "The target is not GPU! Cannot get max threads per stream processor";
}

int GetMaxThreadsPerSmImpl(ARMArch arch) {
  LOG(FATAL)
      << "The target is not GPU! Cannot get max threads per stream processor";
}

int GetMaxThreadsPerSmImpl(NVGPUArch arch) {
  int max_thread = 0;
#ifdef CINN_WITH_CUDA
  cudaDeviceGetAttribute(
      &max_thread, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, 0);
#endif
  return max_thread;
}

int GetMaxThreadsPerSmImpl(HygonDCUArchHIP arch) {
  return BackendAPI::get_backend(arch)->get_device_property(
      BackendAPI::DeviceProperty::MaxThreadsPerSM);
}

int GetMaxThreadsPerSmImpl(HygonDCUArchSYCL arch) {
  return BackendAPI::get_backend(arch)->get_device_property(
      BackendAPI::DeviceProperty::MaxThreadsPerSM);
}

int GetMaxThreadsPerSm(Arch arch) {
  return std::visit(
      [](const auto &impl) { return GetMaxThreadsPerSmImpl(impl); },
      arch.variant());
}

int Target::get_max_threads_per_sm() const { return GetMaxThreadsPerSm(arch); }

int GetMaxBlocksPerSmImpl(UnknownArch) {
  LOG(FATAL)
      << "The target is not GPU! Cannot get max blocks per stream processor";
}

int GetMaxBlocksPerSmImpl(X86Arch) {
  LOG(FATAL)
      << "The target is not GPU! Cannot get max blocks per stream processor";
}

int GetMaxBlocksPerSmImpl(ARMArch) {
  LOG(FATAL)
      << "The target is not GPU! Cannot get max blocks per stream processor";
}

int GetMaxBlocksPerSmImpl(NVGPUArch) {
  int max_blocks = 1;
#ifdef CINN_WITH_CUDA
  cudaDeviceGetAttribute(
      &max_blocks, cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor, 0);
#endif
  return max_blocks;
}

int GetMaxBlocksPerSmImpl(HygonDCUArchHIP arch) {
  return BackendAPI::get_backend(arch)->get_device_property(
      BackendAPI::DeviceProperty::MaxBlocksPerSM);
}

int GetMaxBlocksPerSmImpl(HygonDCUArchSYCL arch) {
  return BackendAPI::get_backend(arch)->get_device_property(
      BackendAPI::DeviceProperty::MaxBlocksPerSM);
}

int GetMaxBlocksPerSm(Arch arch) {
  return std::visit(
      [](const auto &impl) { return GetMaxBlocksPerSmImpl(impl); },
      arch.variant());
}

int Target::get_max_blocks_per_sm() const { return GetMaxBlocksPerSm(arch); }

std::vector<Target::Lib> Target::get_target_libs() const { return libs; }

int Target::get_target_bits() const {
  switch (bits) {
    case Bit::k32:
      return 32;
    case Bit::k64:
      return 64;
    case Bit::Unk:
      return 0;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument("Not supported Bit"));
  }
  return -1;
}

std::string Target::arch_str() const {
  std::ostringstream oss;
  oss << arch;
  return oss.str();
}

std::string Target::device_name_str() const {
#ifdef CINN_WITH_CUDA
  int device_idx = 0;
  cudaError_t result = cudaGetDevice(&device_idx);
  if (result != cudaSuccess) {
    // Call cudaGetLastError() to clear the error bit
    result = cudaGetLastError();
    PADDLE_THROW(::common::errors::Unavailable(
        " cudaGetDevice() returned error %s", cudaGetErrorString(result)));
    return 0;
  }

  cudaDeviceProp properties;
  result = cudaGetDeviceProperties(&properties, device_idx);
  if (result != cudaSuccess) {
    // Call cudaGetLastError() to clear the error bit
    result = cudaGetLastError();
    PADDLE_THROW(::common::errors::Unavailable(
        " cudaGetDeviceProperties() returned error %s",
        cudaGetErrorString(result)));
    return 0;
  }
  std::string device_name = properties.name;
  device_name = std::regex_replace(device_name, std::regex(" "), "_");
  return std::regex_replace(device_name, std::regex("-"), "_");
#else
  CINN_NOT_IMPLEMENTED
#endif
}

std::ostream &operator<<(std::ostream &os, const Target &target) {
  os << "Target<";
  switch (target.os) {
    case Target::OS::Linux:
      os << "linux";
      break;
    case Target::OS::Windows:
      os << "windows";
      break;
    case Target::OS::Unk:
      os << "unk";
      break;
  }

  os << ",";
  os << target.arch;
  os << ",";

  switch (target.bits) {
    case Target::Bit::k32:
      os << "32";
      break;
    case Target::Bit::k64:
      os << "64";
      break;
    case Target::Bit::Unk:
      os << "unk";
      break;
  }
  os << ">";

  return os;
}

const Target &UnkTarget() {
  static Target target(
      Target::OS::Unk, UnknownArch{}, Target::Bit::Unk, {}, {});
  return target;
}
const Target &DefaultHostTarget() {
  static Target target(Target::OS::Linux, X86Arch{}, Target::Bit::k64, {}, {});
  return target;
}

const Target &DefaultNVGPUTarget() {
  static Target target(
      Target::OS::Linux, NVGPUArch{}, Target::Bit::k64, {}, {});
  return target;
}

const Target &DefaultHygonDcuHipTarget() {
  static Target target(
      Target::OS::Linux, HygonDCUArchHIP{}, Target::Bit::k64, {}, {});
  return target;
}

const Target &DefaultHygonDcuSyclTarget() {
  static Target target(
      Target::OS::Linux, HygonDCUArchSYCL{}, Target::Bit::k64, {}, {});
  return target;
}

const Target &DefaultDeviceTarget() {
#ifdef CINN_WITH_CUDA
  return DefaultNVGPUTarget();
#elif defined(CINN_WITH_SYCL)
  return DefaultHygonDcuSyclTarget();
#elif defined(CINN_WITH_HIP)
  return DefaultHygonDcuHipTarget();
#endif
}

int GetMaxThreads() {
  // cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )
  int max_threads = 1;
#ifdef CINN_WITH_CUDA
  int num_sm = 1;
  cudaDeviceGetAttribute(
      &num_sm, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(
      &max_threads, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  // multiplication num_sm
  max_threads *= (num_sm * 4);
#endif
  return max_threads;
}

int GetMaxBlocks() {
  // cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )
  int max_blocks = 1;
#ifdef CINN_WITH_CUDA
  int num_sm = 1;
  cudaDeviceGetAttribute(
      &num_sm, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(
      &max_blocks, cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor, 0);

  // multiplication num_sm
  max_blocks *= num_sm;
#endif
  return max_blocks;
}

const Target &DefaultTarget() {
#ifdef CINN_WITH_CUDA
  return DefaultNVGPUTarget();
#elif defined(CINN_WITH_SYCL)
  return DefaultHygonDcuSyclTarget();
#elif defined(CINN_WITH_HIP)
  return DefaultHygonDcuHipTarget();
#else
  return DefaultHostTarget();
#endif
}

}  // namespace common
}  // namespace cinn
