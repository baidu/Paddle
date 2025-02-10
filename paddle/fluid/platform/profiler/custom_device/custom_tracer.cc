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

#include "paddle/fluid/platform/profiler/custom_device/custom_tracer.h"

#include <mutex>
#include <unordered_map>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/os_info.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

namespace paddle::platform {

CustomTracer::CustomTracer(const std::string& dev_type)
    : dev_type_(dev_type), context_(nullptr) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto selected_devices = phi::DeviceManager::GetSelectedDeviceList(dev_type_);
  if (selected_devices.size()) {
    phi::DeviceManager::SetDevice(dev_type_, selected_devices[0]);
  }
  phi::DeviceManager::ProfilerInitialize(dev_type_, &collector_, &context_);
#endif
}

CustomTracer::~CustomTracer() {  // NOLINT
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  phi::DeviceManager::ProfilerFinalize(dev_type_, &collector_, context_);
#endif
}

std::unordered_map<std::string, std::unique_ptr<CustomTracer>>&
CustomTracer::GetMap() {
  static std::unordered_map<std::string, std::unique_ptr<CustomTracer>>
      instance;
  return instance;
}

void CustomTracer::Release() {
  auto& pool = GetMap();
  for (auto& item : pool) {
    item.second.reset();
  }
  pool.clear();
}

void CustomTracer::PrepareTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::UNINITED || state_ == TracerState::STOPPED,
      true,
      common::errors::PreconditionNotMet("CustomTracer must be UNINITED"));
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  phi::DeviceManager::ProfilerPrepareTracing(dev_type_, &collector_, context_);
#endif
  state_ = TracerState::READY;
}

void CustomTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY,
      true,
      common::errors::PreconditionNotMet("Tracer must be READY or STOPPED"));
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  phi::DeviceManager::ProfilerStartTracing(dev_type_, &collector_, context_);
#endif
  tracing_start_ns_ = phi::PosixInNsec();
  state_ = TracerState::STARTED;
}

void CustomTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STARTED,
      common::errors::PreconditionNotMet("Tracer must be STARTED"));
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  phi::DeviceManager::ProfilerStopTracing(dev_type_, &collector_, context_);
#endif
  state_ = TracerState::STOPPED;
}

void CustomTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STOPPED,
      common::errors::PreconditionNotMet("Tracer must be STOPPED"));
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  phi::DeviceManager::ProfilerCollectTraceData(
      dev_type_, &collector_, tracing_start_ns_, context_);
#endif
  for (auto he : collector_.HostEvents()) {
    collector->AddHostEvent(std::move(he));
  }
  for (auto rte : collector_.RuntimeEvents()) {
    collector->AddRuntimeEvent(std::move(rte));
  }
  for (auto de : collector_.DeviceEvents()) {
    collector->AddDeviceEvent(std::move(de));
  }
  for (auto const& tn : collector_.ThreadNames()) {
    collector->AddThreadName(tn.first, tn.second);
  }
  collector_.ClearAll();
}

}  // namespace paddle::platform

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void profiler_add_runtime_trace_event(C_Profiler prof, void* event) {
  paddle::platform::RuntimeTraceEvent re =
      *reinterpret_cast<paddle::platform::RuntimeTraceEvent*>(event);
  reinterpret_cast<paddle::platform::TraceEventCollector*>(prof)
      ->AddRuntimeEvent(std::move(re));
}

void profiler_add_device_trace_event(C_Profiler prof, void* event) {
  paddle::platform::DeviceTraceEvent de =
      *reinterpret_cast<paddle::platform::DeviceTraceEvent*>(event);
  reinterpret_cast<paddle::platform::TraceEventCollector*>(prof)
      ->AddDeviceEvent(std::move(de));
}
#endif
