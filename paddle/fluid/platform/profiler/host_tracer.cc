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
#include "paddle/fluid/platform/profiler/host_tracer.h"

#include <sstream>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/phi/core/platform/profiler/host_event_recorder.h"

namespace paddle::platform {

namespace {

void ProcessHostEvents(const HostEventSection<CommonEvent>& host_events,
                       TraceEventCollector* collector) {
  for (const auto& thr_sec : host_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    if (thr_sec.thread_name != phi::kDefaultThreadName) {
      collector->AddThreadName(tid, thr_sec.thread_name);
    }
    for (const auto& evt : thr_sec.events) {
      HostTraceEvent event;
      event.name = evt.name;
      event.type = evt.type;
      event.start_ns = evt.start_ns;
      event.end_ns = evt.end_ns;
      event.process_id = host_events.process_id;
      event.thread_id = tid;
      collector->AddHostEvent(std::move(event));
    }
  }
}

void ProcessHostMemEvents(
    const HostEventSection<CommonMemEvent>& host_mem_events,
    TraceEventCollector* collector) {
  for (const auto& thr_sec : host_mem_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    if (thr_sec.thread_name != phi::kDefaultThreadName) {
      collector->AddThreadName(tid, thr_sec.thread_name);
    }
    for (const auto& evt : thr_sec.events) {
      MemTraceEvent event;
      event.timestamp_ns = evt.timestamp_ns;
      event.addr = evt.addr;
      event.type = evt.type;
      event.increase_bytes = evt.increase_bytes;
      event.place = evt.place.DebugString();
      event.current_allocated = evt.current_allocated;
      event.current_reserved = evt.current_reserved;
      event.peak_allocated = evt.peak_allocated;
      event.peak_reserved = evt.peak_reserved;
      event.process_id = host_mem_events.process_id;
      event.thread_id = tid;
      collector->AddMemEvent(std::move(event));
    }
  }
}

void ProcessOperatorSupplementEvents(
    const HostEventSection<OperatorSupplementOriginEvent>& op_supplement_events,
    TraceEventCollector* collector) {
  for (const auto& thr_sec : op_supplement_events.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    if (thr_sec.thread_name != phi::kDefaultThreadName) {
      collector->AddThreadName(tid, thr_sec.thread_name);
    }
    for (const auto& evt : thr_sec.events) {
      // get callstack from event
      std::vector<std::string> callstacks;
      const std::vector<std::string>* callstack_ptr = nullptr;
      auto iter = evt.attributes.find(
          framework::OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
      if (iter != evt.attributes.end()) {
        callstack_ptr =
            &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
        callstacks = *callstack_ptr;
      }
      std::ostringstream result_string;
      for (auto& stack : callstacks) {
        result_string << stack << std::endl;
      }

      OperatorSupplementEvent event;
      event.timestamp_ns = evt.timestamp_ns;
      event.op_type = evt.op_type;
      std::map<std::string, std::vector<std::vector<int64_t>>> input_shapes;
      std::map<std::string, std::vector<std::string>> dtypes;
      std::string callstack;
      for (const auto& input_shape : evt.input_shapes) {
        for (auto idx = 0lu; idx < input_shape.second.size(); idx++) {
          input_shapes[input_shape.first].push_back(std::vector<int64_t>());
          for (auto dim_idx = 0; dim_idx < input_shape.second.at(idx).size();
               dim_idx++) {
            input_shapes[input_shape.first][idx].push_back(
                input_shape.second.at(idx).at(dim_idx));
          }
        }
      }
      for (const auto& dtype : evt.dtypes) {
        for (auto idx = 0lu; idx < dtype.second.size(); idx++) {
          dtypes[dtype.first].push_back(
              framework::proto::VarType::Type_Name(dtype.second.at(idx)));
        }
      }

      event.input_shapes = input_shapes;
      event.dtypes = dtypes;
      event.callstack = result_string.str();
      event.attributes = evt.attributes;
      event.op_id = evt.op_id;
      event.process_id = op_supplement_events.process_id;
      event.thread_id = tid;
      collector->AddOperatorSupplementEvent(std::move(event));
    }
  }
}

}  // namespace

void HostTracer::PrepareTracing() {
  // warm up
  HostTraceLevel::GetInstance().SetLevel(options_.trace_level);
  state_ = TracerState::READY;
}

void HostTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY || state_ == TracerState::STOPPED,
      true,
      common::errors::PreconditionNotMet("TracerState must be READY"));
  HostEventRecorder<CommonEvent>::GetInstance().GatherEvents();
  HostEventRecorder<CommonMemEvent>::GetInstance().GatherEvents();
  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance()
      .GatherEvents();
  HostTraceLevel::GetInstance().SetLevel(options_.trace_level);
  state_ = TracerState::STARTED;
}

void HostTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STARTED,
      common::errors::PreconditionNotMet("TracerState must be STARTED"));
  HostTraceLevel::GetInstance().SetLevel(HostTraceLevel::kDisabled);
  state_ = TracerState::STOPPED;
}

void HostTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STOPPED,
      common::errors::PreconditionNotMet("TracerState must be STOPPED"));
  HostEventSection<CommonEvent> host_events =
      HostEventRecorder<CommonEvent>::GetInstance().GatherEvents();
  ProcessHostEvents(host_events, collector);
  HostEventSection<CommonMemEvent> host_mem_events =
      HostEventRecorder<CommonMemEvent>::GetInstance().GatherEvents();
  ProcessHostMemEvents(host_mem_events, collector);
  HostEventSection<OperatorSupplementOriginEvent> op_supplement_events =
      HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance()
          .GatherEvents();
  ProcessOperatorSupplementEvents(op_supplement_events, collector);
}

}  // namespace paddle::platform
