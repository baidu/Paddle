// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/instruction/instruction_defs.h"
#include "paddle/phi/api/profiler/event.h"

namespace pir {
class Value;
}  // namespace pir

namespace paddle {
namespace framework {
class ValueExecutionInfo;
class Variable;
class Scope;

class InstructionBase {
 public:
  explicit InstructionBase(size_t id, const phi::Place& place);

  virtual ~InstructionBase() = default;

  size_t Id() const { return id_; }

  bool IsArtificial() const { return is_artificial_; }
  void SetArtificial(bool is_artificial) { is_artificial_ = is_artificial; }

  bool IsSyncAfterLaunch() const { return sync_after_launch_; }
  void SetSyncAfterLaunch(bool sync) { sync_after_launch_ = sync; }

  OpFuncType KernelType() const;
  void SetKernelType(OpFuncType type) { type_ = type; }

  int GetStreamPriority() const { return stream_priority_; }
  void SetStreamPriority(int stream_priority) {
    stream_priority_ = stream_priority;
  }

  SchedulingPriority GetSchedulingPriority() const {
    return scheduling_priority_;
  }
  void SetSchedulingPriority(SchedulingPriority priority) {
    scheduling_priority_ = priority;
  }

  const std::string& GetExecutionStream() const { return execution_stream_; }
  void SetExecutionStream(const std::string& stream) {
    execution_stream_ = stream;
  }

  const phi::DeviceContext& DeviceContext() const;
  void SetDeviceContext(phi::DeviceContext* ctx) { dev_ctx_ = ctx; }

  const std::vector<size_t>& NextInstrsInDifferenceThread() const {
    return next_instrs_in_different_thread_;
  }
  void AddNextInstrInDifferentThread(size_t id) {
    next_instrs_in_different_thread_.push_back(id);
  }

  const std::vector<size_t>& NextInstrsInSameThread() const {
    return next_instrs_in_same_thread_;
  }
  void AddNextInstrInSameThread(size_t id) {
    next_instrs_in_same_thread_.push_back(id);
  }

  bool IsForceRecordEvent() const { return force_record_event_; }
  void SetForceRecordEvent(bool force_record) {
    force_record_event_ = force_record;
  }

  const std::vector<std::string>& EventsToWaitInfo() const {
    return events_to_wait_info_;
  }
  void SetEventsToWaitInfo(const std::vector<std::string>& info) {
    events_to_wait_info_ = info;
  }

  const std::string& EventToRecordInfo() const { return event_to_record_info_; }
  void SetEventToRecordInfo(const std::string& info) {
    event_to_record_info_ = info;
  }

  const std::shared_ptr<EventInter>& EventToRecord() const {
    return event_to_record_;
  }
  void AddEventToRecord(std::shared_ptr<platform::DeviceEvent> event,
                        platform::DeviceType waiter_type) {
    event_to_record_ = std::make_shared<EventInter>(id_, event, waiter_type);
  }

  const std::vector<EventInter>& EventsToWait() const {
    return events_to_wait_;
  }
  void AddEventToWait(size_t instr_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type) {
    events_to_wait_.emplace_back(instr_id, event, waiter_type);
  }

  void AddEventToWait(const EventInter* event_inter) {
    events_to_wait_.push_back(*event_inter);
  }

  void RecordEvent(const Place& place) const;
  void WaitEvent(const Place& place) const;

  const std::vector<size_t>& GCCheckVars() const;
  void AddGCCheckVar(size_t id);
  const std::vector<Variable*>& EagerGCVars() const;
  void AddEagerGCVar(Variable* var);
  void ClearEagerGCVars();

  const std::vector<std::pair<const Variable*, Variable*>>& InplaceInfo() const;
  void AddInplace(const Variable* in, Variable* out);
  void ClearInplace();

  std::map<int, int>& GetMutableInplaceBackMap() { return inplace_back_map_; }
  const std::map<int, int>& GetInplaceBackMap() { return inplace_back_map_; }

  const std::unordered_map<::pir::Value, std::vector<int>>& Inputs() const {
    return input_index_;
  }
  std::unordered_map<::pir::Value, std::vector<int>>& GetMutableInputs() {
    return input_index_;
  }
  void SetInputs(
      const std::unordered_map<::pir::Value, std::vector<int>>& inputs);

  const std::unordered_map<::pir::Value, std::vector<int>>& Outputs() const {
    return output_index_;
  }
  std::unordered_map<::pir::Value, std::vector<int>>& GetMutableOutputs() {
    return output_index_;
  }
  void SetOutputs(
      const std::unordered_map<::pir::Value, std::vector<int>>& outputs);

  const std::unordered_set<::pir::Value>& NoNeedBuffer() const {
    return no_need_buffer_values_;
  }
  void SetNoNeedBuffer(
      const std::unordered_set<::pir::Value>& no_need_buffer_values) {
    no_need_buffer_values_ = no_need_buffer_values;
  }

  virtual void Run() = 0;

  virtual const std::string& Name() const = 0;

  virtual ::pir::Operation* Operation() const = 0;

  void InitInputsOutputsIds(::pir::Operation* op,
                            const ValueExecutionInfo& value_exec_info);

  // if scope is not null, also show dimensions of arguments
  virtual std::string DebugStringEx(const paddle::framework::Scope* scope,
                                    ValueExecutionInfo* value_exe_info) const;
  bool SkipRecordStreamForGC() const { return skip_record_stream_for_gc_; }
  void SetSkipRecordStreamForGC(bool skip) {
    skip_record_stream_for_gc_ = skip;
  }

 protected:
  size_t id_;

  bool is_artificial_{
      false};  // Instruction is artificial means that it is only used
               // to assist scheduling and no need to be executed.

  bool sync_after_launch_{false};

  OpFuncType type_;

  // dist attrs：lower value, higher priority
  int stream_priority_{0};

  SchedulingPriority scheduling_priority_{0};

  std::string execution_stream_{kDefaultStream};

  phi::DeviceContext* dev_ctx_;  // not owned

  std::vector<size_t> next_instrs_in_different_thread_;

  std::vector<size_t> next_instrs_in_same_thread_;

  bool force_record_event_{false};

  std::vector<std::string> events_to_wait_info_;

  std::string event_to_record_info_{"default"};

  std::shared_ptr<EventInter> event_to_record_;

  std::vector<EventInter> events_to_wait_;

  std::vector<size_t> gc_check_vars_;

  std::vector<Variable*> eager_gc_vars_;

  std::vector<std::pair<const Variable*, Variable*>>
      vec_inplace_in_to_out_;  // If not use share data, need this ?

  std::map<int, int> inplace_back_map_;

  std::unordered_map<::pir::Value, std::vector<int>> input_index_;

  std::unordered_map<::pir::Value, std::vector<int>> output_index_;

  std::unordered_set<::pir::Value> no_need_buffer_values_;

  bool skip_record_stream_for_gc_{false};
};

}  // namespace framework
}  // namespace paddle
