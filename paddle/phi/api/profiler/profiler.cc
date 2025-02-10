/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/profiler/profiler.h"

#include <mutex>  // NOLINT
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "glog/logging.h"

#include "paddle/phi/api/profiler/common_event.h"
#include "paddle/phi/api/profiler/device_tracer.h"
#include "paddle/phi/api/profiler/host_event_recorder.h"
#include "paddle/phi/api/profiler/host_tracer.h"
#include "paddle/phi/api/profiler/profiler_helper.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/os_info.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/nvtx.h"
#endif

PHI_DEFINE_bool(enable_host_event_recorder_hook,
                false,
                "enable HostEventRecorder, hook Profiler");

PHI_DEFINE_bool(enable_record_op_info,
                false,
                "enable operator supplement info recorder");

namespace phi {

ProfilerState ProfilerHelper::g_state = ProfilerState::kDisabled;
bool ProfilerHelper::g_enable_nvprof_hook = false;
thread_local uint64_t ProfilerHelper::g_thread_id;
uint32_t ProfilerHelper::g_next_thread_id = 0;
std::mutex ProfilerHelper::g_all_event_lists_mutex;
std::list<std::shared_ptr<EventList<Event>>> ProfilerHelper::g_all_event_lists;
thread_local std::shared_ptr<EventList<Event>> ProfilerHelper::g_event_list;
std::list<std::shared_ptr<EventList<MemEvent>>>
    ProfilerHelper::g_all_mem_event_lists;
thread_local std::shared_ptr<EventList<MemEvent>>
    ProfilerHelper::g_mem_event_list;
std::mutex ProfilerHelper::g_all_mem_event_lists_mutex;

Event::Event(EventType type,
             std::string name,
             uint32_t thread_id,
             EventRole role,
             std::string attr)
    : type_(type),
      name_(name),
      thread_id_(thread_id),
      role_(role),
      attr_(attr) {
  cpu_ns_ = GetTimeInNsec();
}

const EventType &Event::type() const { return type_; }

double Event::CpuElapsedMs(const Event &e) const {
  return (static_cast<double>(e.cpu_ns_ - cpu_ns_)) / (1000000.0);
}

double Event::CudaElapsedMs(const Event &e) const {
#ifdef PADDLE_WITH_CUPTI
  return static_cast<double>(gpu_ns_) / 1000000.0;
#else
  LOG_FIRST_N(WARNING, 1) << "CUDA CUPTI is not enabled";
  return 0;
#endif
}

Event *PushEvent(const std::string &name,
                 const EventRole role,
                 std::string attr) {
  return GetEventList().Record(
      EventType::kPushRange, name, ProfilerHelper::g_thread_id, role, attr);
}

void PopEvent(const std::string &name, const EventRole role, std::string attr) {
  GetEventList().Record(
      EventType::kPopRange, name, ProfilerHelper::g_thread_id, role, attr);
}

RecordEvent::RecordEvent(const char *name,
                         const TracerEventType type,
                         uint32_t level,
                         const EventRole role) {
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (ProfilerHelper::g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name);
    is_pushed_ = true;
  }
#endif
#endif
  if (FLAGS_enable_host_event_recorder_hook == false) {
    if (ProfilerHelper::g_state !=
        ProfilerState::kDisabled) {  // avoid temp string
      if (type == TracerEventType::Operator ||
          type == TracerEventType::OperatorInner ||
          type == TracerEventType::UserDefined) {
        OriginalConstruct(name, role, "none");
      }
    }
    return;
  }

  is_enabled_ = true;
  shallow_copy_name_ = name;
  role_ = role;
  type_ = type;
  start_ns_ = PosixInNsec();
}

RecordEvent::RecordEvent(const std::string &name,
                         const TracerEventType type,
                         uint32_t level,
                         const EventRole role) {
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (ProfilerHelper::g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif
  if (FLAGS_enable_host_event_recorder_hook == false) {
    if (type == TracerEventType::Operator ||
        type == TracerEventType::OperatorInner ||
        type == TracerEventType::UserDefined) {
      OriginalConstruct(name, role, "none");
    }
    return;
  }

  is_enabled_ = true;
  name_ = new std::string(name);
  role_ = role;
  type_ = type;
  start_ns_ = PosixInNsec();
}

RecordEvent::RecordEvent(const std::string &name,
                         const std::string &attr,
                         const TracerEventType type,
                         uint32_t level,
                         const EventRole role) {
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }

#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (ProfilerHelper::g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif

  if (FLAGS_enable_host_event_recorder_hook == false) {
    if (type == TracerEventType::Operator ||
        type == TracerEventType::OperatorInner ||
        type == TracerEventType::UserDefined) {
      OriginalConstruct(name, role, attr);
    }
    return;
  }

  is_enabled_ = true;
  type_ = type;
  name_ = new std::string(name);
  start_ns_ = PosixInNsec();
  attr_ = new std::string(attr);
}

void RecordEvent::OriginalConstruct(const std::string &name,
                                    const EventRole role,
                                    const std::string &attr) {
  if (ProfilerHelper::g_state == ProfilerState::kDisabled || name.empty())
    return;

  // do some initialization
  name_ = new std::string(name);
  start_ns_ = PosixInNsec();
  role_ = role;
  attr_ = new std::string(attr);
  is_enabled_ = true;
  // lock is not needed, the code below is thread-safe
  // Maybe need the same push/pop behavior.
  Event *e = PushEvent(name, role, attr);
  SetCurAnnotation(e);
  *name_ = e->name();
}

void RecordEvent::End() {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (ProfilerHelper::g_enable_nvprof_hook && is_pushed_) {
    dynload::nvtxRangePop();
    is_pushed_ = false;
  }
#endif
#endif
  if (LIKELY(FLAGS_enable_host_event_recorder_hook && is_enabled_)) {
    uint64_t end_ns = PosixInNsec();
    if (LIKELY(shallow_copy_name_ != nullptr)) {
      HostEventRecorder<CommonEvent>::GetInstance().RecordEvent(
          shallow_copy_name_, start_ns_, end_ns, role_, type_);
    } else if (name_ != nullptr) {
      if (attr_ == nullptr) {
        HostEventRecorder<CommonEvent>::GetInstance().RecordEvent(
            *name_, start_ns_, end_ns, role_, type_);
      } else {
        HostEventRecorder<CommonEvent>::GetInstance().RecordEvent(
            *name_, start_ns_, end_ns, role_, type_, *attr_);
        delete attr_;
      }
      delete name_;
    }
    // use this flag to avoid double End();
    is_enabled_ = false;
    return;
  }

  if (ProfilerHelper::g_state == ProfilerState::kDisabled || !is_enabled_)
    return;
  // lock is not needed, the code below is thread-safe
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer) {
    uint64_t end_ns = PosixInNsec();
    tracer->AddCPURecords(CurAnnotationName(),
                          start_ns_,
                          end_ns,
                          BlockDepth(),
                          ProfilerHelper::g_thread_id);
  }
  ClearCurAnnotation();
  PopEvent(*name_, role_);
  delete name_;
  delete attr_;
  // use this flag to avoid double End();
  is_enabled_ = false;
}

bool RecordEvent::IsEnabled() {
  return FLAGS_enable_host_event_recorder_hook ||
         ProfilerHelper::g_enable_nvprof_hook ||
         ProfilerHelper::g_state != ProfilerState::kDisabled;
}

RecordOpInfoSupplement::RecordOpInfoSupplement(
    const std::string &type,
    const std::vector<std::pair<const char *, std::vector<DDim>>> &input_shapes,
    const AttributeMap &attrs) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  if (IsEnabled() == false) {
    return;
  }
  uint64_t op_id = 0;
  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance().RecordEvent(
      PosixInNsec(), type, input_shapes, attrs, op_id);
}

bool RecordOpInfoSupplement::IsEnabled() { return FLAGS_enable_record_op_info; }

void EnableOpInfoRecorder() { FLAGS_enable_record_op_info = true; }

void DisableOpInfoRecorder() { FLAGS_enable_record_op_info = false; }

}  // namespace phi
