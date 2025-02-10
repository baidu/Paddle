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

#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"

#include <future>
#include <unordered_set>

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/phi/core/platform/device_context.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle::framework::interpreter {

using DeviceContext = phi::DeviceContext;
using DeviceEvent = platform::DeviceEvent;

inline std::string RunTypeToString(DownstreamRunType run_type) {
  if (run_type == DownstreamRunType::kDirectRun) {
    return "DirectRun";
  } else {
    return "EventRun";
  }
}

void StreamAnalyzer::ConstructEvents(std::vector<Instruction>* instructions) {
  if (!is_event_info_build_) {
    std::vector<Instruction> cross_step_merged_instructions = *instructions;
    for (const Instruction& instr : *instructions) {
      cross_step_merged_instructions.emplace_back(instr);
    }

    std::vector<Instruction*> cross_step_merged_instructions_ptr;
    for (Instruction& instr : cross_step_merged_instructions) {
      cross_step_merged_instructions_ptr.emplace_back(&instr);
    }

    DependencyBuilder dependency_builder;
    dependency_builder.Build(cross_step_merged_instructions);

    const std::map<size_t, std::set<size_t>>& downstream_map =
        dependency_builder.OpDownstreamMap();
    const size_t instr_num = cross_step_merged_instructions.size();
    std::vector<std::vector<std::vector<size_t>>> run_type_info(
        instr_num,
        std::vector<std::vector<size_t>>(
            /*number_of_run_type = */ 2));  // NOLINT
    // instr_id -> run_type -> next_instr_id
    AnalyseAllRunType(
        cross_step_merged_instructions_ptr, downstream_map, &run_type_info);

    AnalyseAllEventInfo(
        cross_step_merged_instructions_ptr, run_type_info, event_info_.get());
    ShrinkEventInfo(dependency_builder, event_info_.get());
    is_event_info_build_ = true;
  }

  // Construct events
  std::map<size_t, std::shared_ptr<DeviceEvent>> instr2event;
  for (auto& context_item : *event_info_) {
    for (auto& waiter_item : context_item.second) {
      size_t waiter_instr_id = waiter_item.first;
      std::set<size_t>& recorder_instr_ids = waiter_item.second;

      if (waiter_instr_id >= instructions->size()) {
        waiter_instr_id -= instructions->size();
      }

      for (size_t recorder_instr_id : recorder_instr_ids) {
        // Redundant record
        if (recorder_instr_id >= instructions->size()) {
          continue;
        }

        Instruction& recorder_instr = instructions->at(recorder_instr_id);
        Instruction& waiter_instr = instructions->at(waiter_instr_id);
        platform::DeviceType waiter_type = GetWaiterType(waiter_instr);

        if (instr2event.find(recorder_instr_id) == instr2event.end()) {
          std::shared_ptr<DeviceEvent> device_event =
              std::make_shared<DeviceEvent>(
                  recorder_instr.DeviceContext().GetPlace(),
                  platform::GenerateDeviceEventFlag());
          recorder_instr.AddEventToRecord(device_event,
                                          platform::kCUDA /*unused*/);
          // It means the event will be waited for other interpreter that the
          // event name of a operator is not 'default'.
          if (recorder_instr.OpFunc()->force_record_event_ == true &&
              (*program_force_events_to_wait_)
                      .count(recorder_instr.OpFunc()->event_to_record_) == 0) {
            (*program_force_events_to_wait_)[recorder_instr.OpFunc()
                                                 ->event_to_record_] =
                recorder_instr.EventToRecord();
          }
          instr2event.emplace(recorder_instr_id, device_event);
        }

        waiter_instr.AddEventToWait(
            recorder_instr_id, instr2event.at(recorder_instr_id), waiter_type);
        VLOG(6) << "Add event: " << recorder_instr.OpBase()->Type() << "("
                << recorder_instr_id << ") -> " << waiter_instr.OpBase()->Type()
                << "(" << waiter_instr_id << "), waiter type = " << waiter_type;
      }
    }
  }
  // NOTE(lizhiyu): The manual event only support the program_interpreter to
  // analyze the streams across the sub_programs. construct manual events to
  // record
  for (auto& instruction : *instructions) {
    // create extra event to record
    auto op_func_node = instruction.OpFunc();
    if (op_func_node->force_record_event_ &&
        instruction.EventToRecord() == nullptr) {
      auto place = instruction.DeviceContext().GetPlace();
      if (phi::is_gpu_place(place)) {
        PADDLE_ENFORCE_NE(
            op_func_node->event_to_record_,
            "default",
            common::errors::InvalidArgument(
                "If the attribute 'force_record_event_' of one "
                "operator is 'true', the 'event_to_record_' of this "
                "operator can not be 'default'. But the "
                "'event_name' of the operator %s is 'default'.",
                instruction.OpBase()->Type().c_str()));
        PADDLE_ENFORCE_EQ(
            (*program_force_events_to_wait_)
                .find(op_func_node->event_to_record_),
            (*program_force_events_to_wait_).end(),
            common::errors::InvalidArgument(
                "The program_force_events_to_wait_ had the event "
                "that belongs to the operator : %s before the operator create "
                "the event, "
                "This is is weird.",
                instruction.OpBase()->Type().c_str()));
        std::shared_ptr<DeviceEvent> device_event =
            std::make_shared<DeviceEvent>(place,
                                          platform::GenerateDeviceEventFlag());
        instruction.AddEventToRecord(device_event, platform::kCUDA /*unused*/);
        (*program_force_events_to_wait_)[op_func_node->event_to_record_] =
            instruction.EventToRecord();
        VLOG(6) << "Create manual event: " << op_func_node->event_to_record_
                << " for the operator: " << instruction.OpBase()->Type();
      }
    }
    // add extra manual events
    if (!(op_func_node->events_to_wait_.empty())) {
      for (auto event_name : op_func_node->events_to_wait_) {
        PADDLE_ENFORCE_NE(
            (*program_force_events_to_wait_).find(event_name),
            (*program_force_events_to_wait_).end(),
            common::errors::InvalidArgument(
                "The program_force_events_to_wait_ don't have the event %s "
                "for the operator: %s to wait. The event should had been "
                "created by the operator "
                "whose event_to_record_ is %s.",
                event_name.c_str(),
                instruction.OpBase()->Type().c_str(),
                event_name.c_str()));

        instruction.AddEventToWait(
            (*program_force_events_to_wait_)[event_name].get());
      }
    }
  }
}

DeviceContext* StreamAnalyzer::ParseDeviceContext(
    const OpFuncNode& op_func_node) const {
  auto& op = op_func_node.operator_base_;
  if (op == nullptr) {
    return op_func_node.dev_ctx_;
  }
  auto& op_type = op->Type();
  const std::string& execution_stream = op_func_node.execution_stream_;
  const int stream_priority = op_func_node.stream_priority_;
  ContextManager& ctx_manager = ContextManager::Instance();

  DeviceContext* dev_ctx = nullptr;

  // only gpu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (phi::is_gpu_place(place_) || phi::is_custom_place(place_)) {
    VLOG(6) << "Parse DeviceContext for " << op_type
            << ", execution stream = " << execution_stream;
    if (execution_stream != kDefaultStream) {
      dev_ctx = ctx_manager
                    .Get(std::string(kCustomStream) + "-" + execution_stream,
                         place_,
                         stream_priority)
                    .get()
                    .get();
      SetDeviceCommContext(op.get(), dev_ctx);
      return dev_ctx;
    }

    if (op_type == interpreter::kMemcpyD2H) {
      dev_ctx =
          ctx_manager.Get(std::string(kD2HStream), place_, stream_priority)
              .get()
              .get();
      SetDeviceCommContext(op.get(), dev_ctx);
      return dev_ctx;
    } else if (op_type == interpreter::kMemcpyH2D) {
      dev_ctx =
          ctx_manager.Get(std::string(kH2DStream), place_, stream_priority)
              .get()
              .get();
      SetDeviceCommContext(op.get(), dev_ctx);
      return dev_ctx;
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // NOTE(Ruibiao): Here supports multi-stream overlap for c_allreduce_sum
    // with use_cal_stream==false by returning a device context getting from the
    // global NCCLCommContext instance. Because when use_calc_stream==false, in
    // OP kernel, the NCCL communication will be launched to the stream directly
    // getting from the global NCCLCommContext instance rather than the
    // DeviceContext passed from executor (see CAllReduceOpCUDAKernel in
    // c_allreduce_op.h). Now it is just a temporary solution for ONLY
    // c_allreduce_sum which is used in ResNet50 distributed training.
    if (op_type == "c_allreduce_sum" &&
        op->Attr<bool>("use_calc_stream") == false) {
      int ring_id = op->Attr<int>("ring_id");

      if (FLAGS_dynamic_static_unified_comm) {
        const auto& comm_context_manager =
            phi::distributed::CommContextManager::GetInstance();
        dev_ctx = static_cast<phi::DeviceContext*>(
            static_cast<phi::distributed::NCCLCommContext*>(
                comm_context_manager.Get(std::to_string(ring_id)))
                ->GetDevContext());
      } else {
        dev_ctx = platform::NCCLCommContext::Instance()
                      .Get(ring_id, place_)
                      ->dev_context();
      }
      return dev_ctx;
    }
#endif
  }

  if (op != nullptr) {
    SetDeviceCommContext(op.get(), op_func_node.dev_ctx_);
  }
  return op_func_node.dev_ctx_;
}

const std::unordered_set<std::string> no_need_buffer_ins(Instruction* instr) {
  auto* op = instr->OpBase();
  auto& inferer = op->Info().NoNeedBufferVarsInferer();
  if (inferer) {
    return inferer(op->Inputs(), op->Outputs(), op->Attrs());
  }
  return std::unordered_set<std::string>();
}

const std::unordered_set<pir::Value> no_need_buffer_ins(
    const paddle::framework::InstructionBase* instr) {
  return instr->NoNeedBuffer();
}

template <typename T1, typename T2>
bool has_data_dependency(T1* cur_instr, T1* next_instr) {
  // cur_instr->var->next_instr
  std::unordered_set<size_t> cur_var_ids;
  for (auto& item : cur_instr->Outputs()) {
    cur_var_ids.insert(item.second.begin(), item.second.end());
  }

  const std::unordered_set<T2> next_instr_no_need_buffer_ins =
      no_need_buffer_ins(next_instr);

  for (auto& item : next_instr->Inputs()) {
    if (next_instr_no_need_buffer_ins.find(item.first) !=
        next_instr_no_need_buffer_ins.end()) {
      continue;
    }
    for (auto next_var_id : item.second) {
      if (cur_var_ids.find(next_var_id) != cur_var_ids.end()) {
        VLOG(6) << "Found data dependency from "
                << "cur_instr(" << cur_instr->Id() << ") to "
                << "next_instr(" << next_instr->Id() << ")";
        return true;
      }
    }
  }

  // cur_instr->var && next_instr->var
  // var->cur_instr && next_instr->var
  const std::unordered_set<T2> cur_instr_no_need_buffer_ins =
      no_need_buffer_ins(cur_instr);
  for (auto& item : cur_instr->Inputs()) {
    if (cur_instr_no_need_buffer_ins.find(item.first) ==
        cur_instr_no_need_buffer_ins.end()) {
      cur_var_ids.insert(item.second.begin(), item.second.end());
    }
  }

  for (auto& item : next_instr->Outputs()) {
    for (auto next_var_id : item.second) {
      if (cur_var_ids.find(next_var_id) != cur_var_ids.end()) {
        VLOG(6) << "Found data dependency from "
                << "cur_instr(" << cur_instr->Id() << ") to "
                << "next_instr(" << next_instr->Id() << ")";
        return true;
      }
    }
  }

  return false;
}

template <typename T>
DownstreamRunType analyse_run_type_for_two_instructions(T* cur_instr,
                                                        T* next_instr,
                                                        const Place& place) {
  // xpu&ipu memcpy kernel is synchronous.
  if (phi::is_ipu_place(place) || phi::is_xpu_place(place)) {
    return DownstreamRunType::kDirectRun;
  }

  // npu d2h kernel is asynchronous.
  if (phi::is_custom_place(place)) {
    if (phi::is_cpu_place(cur_instr->DeviceContext().GetPlace()) ||
        interpreter::IsMemcpyH2D(next_instr)) {
      return DownstreamRunType::kDirectRun;
    }
  }

  if (cur_instr->KernelType() == OpFuncType::kGpuAsync &&
      (&cur_instr->DeviceContext() != &next_instr->DeviceContext())) {
    return DownstreamRunType::kEventRun;
  }

  return DownstreamRunType::kDirectRun;
}

template <typename T>
void analyse_all_run_type(
    const std::vector<T*>& instructions,
    const std::map<size_t, std::set<size_t>>& downstream_map,
    const Place& place,
    std::vector<std::vector<std::vector<size_t>>>* run_type_info) {
  for (auto& item : downstream_map) {
    size_t cur_instr_id = item.first;
    T* cur_instr = instructions[item.first];
    for (size_t next_instr_id : item.second) {
      T* next_instr = instructions[next_instr_id];
      DownstreamRunType run_type = analyse_run_type_for_two_instructions<T>(
          cur_instr, next_instr, place);

      (*run_type_info)[cur_instr_id][run_type].push_back(next_instr_id);

      VLOG(6) << RunTypeToString(run_type) << ": "
              << "cur_instr_id(" << cur_instr_id << ") -> "
              << "next_instr_id(" << next_instr_id << ")";
    }
  }
}

void StreamAnalyzer::AnalyseAllRunType(
    const std::vector<Instruction*>& instructions,
    const std::map<size_t, std::set<size_t>>& downstream_map,
    std::vector<std::vector<std::vector<size_t>>>* run_type_info) const {
  analyse_all_run_type<Instruction>(
      instructions, downstream_map, place_, run_type_info);
}

// The caller should guarantee cur_instr and next_instr is kEventRun
template <typename T>
void analyse_event_info_for_two_instructions(
    const std::vector<T*>& instructions,
    const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
    const size_t cur_instr_id,
    const size_t next_instr_id,
    std::set<size_t>* waiter_instr_ids,
    std::set<size_t>* visited_next_instr_id);

template <>
void analyse_event_info_for_two_instructions<Instruction>(
    const std::vector<Instruction*>& instructions,
    const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
    const size_t cur_instr_id,
    const size_t next_instr_id,
    std::set<size_t>* waiter_instr_ids,
    std::set<size_t>* visited_next_instr_id) {
  if (visited_next_instr_id->find(next_instr_id) !=
      visited_next_instr_id->end()) {
    return;
  }
  visited_next_instr_id->insert(next_instr_id);

  // NOTE(Ruibiao): Though depend_op as next_instr is no_need_buffer, we should
  // also wait event for it. Because depend_op is used to build dependencies for
  // fused vars in some scenarios. In those cases, we do not know which vars may
  // lead a implicit data dependency. For example,
  // ###
  // ### fused_var = fuse_op(var0, ...)
  // ### var1 = op1(fused_var)
  // ### var0 = depend_op(var0, fused_var)
  // ### var2 = op2(var0)
  // ###
  // If op1 are cross-stream with depend_op and op2, then we have:
  // ###
  // ### event_run : op1 -> depend_op
  // ### direct_run : depend_op -> op2
  // ###
  // There is actually a data dependency between op1 and op2 that var0 and
  // fused_var share the same tensor. However, as the dependency is implicit, we
  // can only add event for it with the help of depend_op.

  if (has_data_dependency<Instruction, std::string>(
          instructions[cur_instr_id], instructions[next_instr_id]) ||
      instructions[next_instr_id]->OpBase()->Type() == "depend") {
    waiter_instr_ids->insert(next_instr_id);
    return;
  }

  if (!run_type_info[next_instr_id][DownstreamRunType::kEventRun].empty()) {
    auto& next_next_instructor_ids =
        run_type_info[next_instr_id][DownstreamRunType::kEventRun];
    for (auto& id : next_next_instructor_ids) {
      if (has_data_dependency<Instruction, std::string>(
              instructions[cur_instr_id], instructions[id])) {
        waiter_instr_ids->insert(next_instr_id);
        return;
      }
    }
    return;
  }

  // NOTE(Ruibiao): If no data dependency from cur_instr to next_instr, and
  // simultaneously next_instr has no event_run downstream instr, we try to
  // recursively add events between cur_instr and next_instr's
  // direct-run-instrs. This can delay the event wait and achieve better
  // scheduling performance in some scenarios. However, when next_instr has too
  // many direct-run-instrs, it may perform worse than add event directly
  // between cur_instr and next_instr.
  for (size_t instr_id :
       run_type_info[next_instr_id][DownstreamRunType::kDirectRun]) {
    analyse_event_info_for_two_instructions<Instruction>(instructions,
                                                         run_type_info,
                                                         cur_instr_id,
                                                         instr_id,
                                                         waiter_instr_ids,
                                                         visited_next_instr_id);
  }
}

template <>
void analyse_event_info_for_two_instructions<
    paddle::framework::InstructionBase>(
    const std::vector<paddle::framework::InstructionBase*>& instructions,
    const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
    const size_t cur_instr_id,
    const size_t next_instr_id,
    std::set<size_t>* waiter_instr_ids,
    std::set<size_t>* visited_next_instr_id) {
  if (visited_next_instr_id->find(next_instr_id) !=
      visited_next_instr_id->end()) {
    return;
  }
  visited_next_instr_id->insert(next_instr_id);

  // NOTE(Ruibiao): Though depend_op as next_instr is no_need_buffer, we should
  // also wait event for it. Because depend_op is used to build dependencies for
  // fused vars in some scenarios. In those cases, we do not know which vars may
  // lead a implicit data dependency. For example,
  // ###
  // ### fused_var = fuse_op(var0, ...)
  // ### var1 = op1(fused_var)
  // ### var0 = depend_op(var0, fused_var)
  // ### var2 = op2(var0)
  // ###
  // If op1 are cross-stream with depend_op and op2, then we have:
  // ###
  // ### event_run : op1 -> depend_op
  // ### direct_run : depend_op -> op2
  // ###
  // There is actually a data dependency between op1 and op2 that var0 and
  // fused_var share the same tensor. However, as the dependency is implicit, we
  // can only add event for it with the help of depend_op.

  if (has_data_dependency<paddle::framework::InstructionBase, pir::Value>(
          instructions[cur_instr_id], instructions[next_instr_id]) ||
      instructions[next_instr_id]->Name() == "pd_op.depend") {
    waiter_instr_ids->insert(next_instr_id);
    return;
  }

  if (!run_type_info[next_instr_id][DownstreamRunType::kEventRun].empty()) {
    auto& next_next_instructor_ids =
        run_type_info[next_instr_id][DownstreamRunType::kEventRun];
    for (auto& id : next_next_instructor_ids) {
      if (has_data_dependency<paddle::framework::InstructionBase, pir::Value>(
              instructions[cur_instr_id], instructions[id])) {
        waiter_instr_ids->insert(next_instr_id);
        return;
      }
    }

    return;
  }

  // NOTE(Ruibiao): If no data dependency from cur_instr to next_instr, and
  // simultaneously next_instr has no event_run downstream instr, we try to
  // recursively add events between cur_instr and next_instr's
  // direct-run-instrs. This can delay the event wait and achieve better
  // scheduling performance in some scenarios. However, when next_instr has too
  // many direct-run-instrs, it may perform worse than add event directly
  // between cur_instr and next_instr.
  for (size_t instr_id :
       run_type_info[next_instr_id][DownstreamRunType::kDirectRun]) {
    analyse_event_info_for_two_instructions<paddle::framework::InstructionBase>(
        instructions,
        run_type_info,
        cur_instr_id,
        instr_id,
        waiter_instr_ids,
        visited_next_instr_id);
  }
}

template <typename T>
void analyse_all_event_info(
    const std::vector<T*>& instructions,
    const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info) {
  for (size_t cur_instr_id = 0; cur_instr_id < instructions.size();
       ++cur_instr_id) {
    const std::vector<size_t>& next_instr_ids =
        run_type_info[cur_instr_id][DownstreamRunType::kEventRun];
    std::set<size_t> waiter_instr_ids;
    std::set<size_t> visited_next_instr_id;

    for (size_t next_instr_id : next_instr_ids) {
      analyse_event_info_for_two_instructions(instructions,
                                              run_type_info,
                                              cur_instr_id,
                                              next_instr_id,
                                              &waiter_instr_ids,
                                              &visited_next_instr_id);
    }

    for (size_t waiter_instr_id : waiter_instr_ids) {
      (*event_info)[&(instructions[cur_instr_id]->DeviceContext())]
                   [waiter_instr_id]
                       .insert(cur_instr_id);
    }
  }
}

void StreamAnalyzer::AnalyseAllEventInfo(
    const std::vector<Instruction*>& instructions,
    const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info) const {
  analyse_all_event_info<Instruction>(instructions, run_type_info, event_info);
}

template <typename T>
void shrink_event_info(
    const T& dependency_builder,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info) {
  for (auto& item : *event_info) {
    // shrink redundant recorders, waiter instrs should only wait for the last
    // recorder instrs in each stream
    std::map<size_t, std::set<size_t>>& waiter_recorder_map = item.second;
    for (auto& waiter_recorder : waiter_recorder_map) {
      size_t waiter_instr_id = waiter_recorder.first;
      std::set<size_t>& recorder_instr_ids = waiter_recorder.second;
      std::set<size_t> unnecessary_recorder_instr_ids;
      for (size_t cur_instr_id : recorder_instr_ids) {
        for (size_t next_instr_id : recorder_instr_ids) {
          if (dependency_builder.OpHappensBefore(cur_instr_id, next_instr_id)) {
            unnecessary_recorder_instr_ids.insert(cur_instr_id);
            break;
          }
        }
      }

      for (size_t unnecessary_recorder_instr_id :
           unnecessary_recorder_instr_ids) {
        VLOG(8) << "Shrink event : " << unnecessary_recorder_instr_id << " -> "
                << waiter_instr_id;
        recorder_instr_ids.erase(unnecessary_recorder_instr_id);
      }
    }

    // shrink redundant waiters, recorder instrs should only wait by the first
    // waiter instrs in each stream
    std::map<size_t, std::set<size_t>> recorder_waiter_map;
    for (auto& waiter_recorder : waiter_recorder_map) {
      size_t waiter_instr_id = waiter_recorder.first;
      std::set<size_t>& recorder_instr_ids = waiter_recorder.second;
      for (size_t record_instr_id : recorder_instr_ids) {
        recorder_waiter_map[record_instr_id].insert(waiter_instr_id);
      }
    }

    for (auto& recorder_waiter : recorder_waiter_map) {
      size_t recorder_instr_id = recorder_waiter.first;
      std::set<size_t>& waiter_instr_ids = recorder_waiter.second;
      std::set<size_t> unnecessary_waiter_instr_ids;
      for (size_t cur_instr_id : waiter_instr_ids) {
        for (size_t next_instr_id : waiter_instr_ids) {
          if (dependency_builder.OpHappensBefore(cur_instr_id, next_instr_id) &&
              dependency_builder.IsSameDeviceContext(cur_instr_id,
                                                     next_instr_id)) {
            unnecessary_waiter_instr_ids.insert(next_instr_id);
            break;
          }
        }
      }

      for (size_t unnecessary_waiter_instr_id : unnecessary_waiter_instr_ids) {
        VLOG(8) << "Shrink event : " << recorder_instr_id << " -> "
                << unnecessary_waiter_instr_id;
        waiter_recorder_map[unnecessary_waiter_instr_id].erase(
            recorder_instr_id);
      }
    }
  }
}

void StreamAnalyzer::ShrinkEventInfo(
    const DependencyBuilder& dependency_builder,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info) const {
  shrink_event_info<DependencyBuilder>(dependency_builder, event_info);
}

platform::DeviceType StreamAnalyzer::GetWaiterType(
    const Instruction& instr) const {
  if (instr.KernelType() == OpFuncType::kCpuSync) {
    return platform::kCPU;
  } else {
    if (phi::is_xpu_place(place_)) {
      return platform::kXPU;
    } else if (phi::is_custom_place(place_)) {
      return platform::kCUSTOM_DEVICE;
    }
    return platform::kCUDA;
  }
}

std::shared_ptr<
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>
StreamAnalyzer::GetEventInfo() const {
  return event_info_;
}

void StreamAnalyzer::ShareEventInfoFrom(const StreamAnalyzer& src) {
  event_info_ = src.GetEventInfo();
  is_event_info_build_ = true;
}

/// ======================== ///
///        For new ir        ///
/// ======================== ///
void PirStreamAnalyzer::ConstructEvents(
    const std::vector<std::unique_ptr<paddle::framework::InstructionBase>>&
        instructions) {
  if (!is_event_info_build_) {
    std::vector<paddle::framework::InstructionBase*>
        cross_step_merged_instructions_ptr;
    for (auto& instr : instructions) {
      cross_step_merged_instructions_ptr.emplace_back(instr.get());
    }
    for (auto& instr : instructions) {
      cross_step_merged_instructions_ptr.emplace_back(instr.get());
    }

    PirDependencyBuilder dependency_builder;
    dependency_builder.Build(cross_step_merged_instructions_ptr);
    const std::map<size_t, std::set<size_t>>& downstream_map =
        dependency_builder.OpDownstreamMap();

    const size_t instr_num = cross_step_merged_instructions_ptr.size();
    std::vector<std::vector<std::vector<size_t>>> run_type_info(
        instr_num,
        std::vector<std::vector<size_t>>(
            /*number_of_run_type = */ 2));  // instr_id -> run_type ->
                                            // next_instr_id
    AnalyseAllRunType(
        cross_step_merged_instructions_ptr, downstream_map, &run_type_info);

    AnalyseAllEventInfo(
        cross_step_merged_instructions_ptr, run_type_info, event_info_.get());

    ShrinkEventInfo(dependency_builder, event_info_.get());

    is_event_info_build_ = true;
  }

  // Construct events
  std::map<size_t, std::shared_ptr<DeviceEvent>> instr2event;
  for (auto& context_item : *event_info_) {
    for (auto& waiter_item : context_item.second) {
      size_t waiter_instr_id = waiter_item.first;
      std::set<size_t>& recorder_instr_ids = waiter_item.second;

      if (waiter_instr_id >= instructions.size()) {
        waiter_instr_id -= instructions.size();
      }

      for (size_t recorder_instr_id : recorder_instr_ids) {
        // Redundant record
        if (recorder_instr_id >= instructions.size()) {
          continue;
        }

        paddle::framework::InstructionBase* recorder_instr =
            instructions.at(recorder_instr_id).get();
        paddle::framework::InstructionBase* waiter_instr =
            instructions.at(waiter_instr_id).get();
        platform::DeviceType waiter_type = GetWaiterType(waiter_instr);

        if (instr2event.find(recorder_instr_id) == instr2event.end()) {
          std::shared_ptr<DeviceEvent> device_event =
              std::make_shared<DeviceEvent>(
                  recorder_instr->DeviceContext().GetPlace(),
                  platform::GenerateDeviceEventFlag());
          recorder_instr->AddEventToRecord(device_event,
                                           platform::kCUDA /*unused*/);
          // It means the event will be waited for other interpreter that the
          // event name of a operator is not 'default'.
          if (recorder_instr->IsForceRecordEvent() == true &&
              (*program_force_events_to_wait_)
                      .count(recorder_instr->EventToRecordInfo()) == 0) {
            (*program_force_events_to_wait_)[recorder_instr
                                                 ->EventToRecordInfo()] =
                recorder_instr->EventToRecord();
          }
          instr2event.emplace(recorder_instr_id, device_event);
        }

        waiter_instr->AddEventToWait(
            recorder_instr_id, instr2event.at(recorder_instr_id), waiter_type);
        VLOG(6) << "Add event: " << recorder_instr->Name() << "("
                << recorder_instr_id << ") -> " << waiter_instr->Name() << "("
                << waiter_instr_id << "), waiter type = " << waiter_type;
      }
    }
  }
  // NOTE(lizhiyu): The manual event only support the program_interpreter to
  // annalyze the streams across the sub_programs. construct manual events to
  // record
  for (auto& instr : instructions) {
    // create extra event to record
    if (instr->IsForceRecordEvent() && instr->EventToRecord() == nullptr) {
      auto place = instr->DeviceContext().GetPlace();
      if (phi::is_gpu_place(place)) {
        PADDLE_ENFORCE_NE(
            instr->EventToRecordInfo(),
            "default",
            common::errors::InvalidArgument(
                "If the attribute 'force_record_event_' of one "
                "operator is 'true', the 'event_to_record_' of this "
                "operator can not be 'default'. But the "
                "'event_name' of the operator %s is 'default'.",
                instr->Name()));
        PADDLE_ENFORCE_EQ(
            (*program_force_events_to_wait_).find(instr->EventToRecordInfo()),
            (*program_force_events_to_wait_).end(),
            common::errors::InvalidArgument(
                "The program_force_events_to_wait_ had the event "
                "that belongs to the operator : %s before the operator create "
                "the event, "
                "This is is weird.",
                instr->Name()));
        std::shared_ptr<DeviceEvent> device_event =
            std::make_shared<DeviceEvent>(place,
                                          platform::GenerateDeviceEventFlag());
        instr->AddEventToRecord(device_event, platform::kCUDA /*unused*/);
        (*program_force_events_to_wait_)[instr->EventToRecordInfo()] =
            instr->EventToRecord();
        VLOG(6) << "Create manual event: " << instr->EventToRecordInfo()
                << " for the operator: " << instr->Name();
      }
    }
    // add extra manual events
    if (!(instr->EventsToWaitInfo().empty())) {
      for (auto event_name : instr->EventsToWaitInfo()) {
        PADDLE_ENFORCE_NE(
            (*program_force_events_to_wait_).find(event_name),
            (*program_force_events_to_wait_).end(),
            common::errors::InvalidArgument(
                "The program_force_events_to_wait_ don't have the event %s "
                "for the operator: %s to wait. The event should had been "
                "created by the operator "
                "whose event_to_record_ is %s.",
                event_name.c_str(),
                instr->Name(),
                event_name.c_str()));

        instr->AddEventToWait(
            (*program_force_events_to_wait_)[event_name].get());
      }
    }
  }
}

void PirStreamAnalyzer::AnalyseAllRunType(
    const std::vector<paddle::framework::InstructionBase*>& instructions,
    const std::map<size_t, std::set<size_t>>& downstream_map,
    std::vector<std::vector<std::vector<size_t>>>* run_type_info) const {
  analyse_all_run_type<paddle::framework::InstructionBase>(
      instructions, downstream_map, place_, run_type_info);
}

void PirStreamAnalyzer::AnalyseAllEventInfo(
    const std::vector<paddle::framework::InstructionBase*>& instructions,
    const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info) const {
  analyse_all_event_info<paddle::framework::InstructionBase>(
      instructions, run_type_info, event_info);
}

void PirStreamAnalyzer::ShrinkEventInfo(
    const PirDependencyBuilder& dependency_builder,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info_map) const {
  shrink_event_info<PirDependencyBuilder>(dependency_builder, event_info_map);
}

platform::DeviceType PirStreamAnalyzer::GetWaiterType(
    const paddle::framework::InstructionBase* instr) const {
  if (instr->KernelType() == OpFuncType::kCpuSync) {
    return platform::kCPU;
  } else {
    if (phi::is_xpu_place(place_)) {
      return platform::kXPU;
    } else if (phi::is_custom_place(place_)) {
      return platform::kCUSTOM_DEVICE;
    }
    return platform::kCUDA;
  }
}

void PirStreamAnalyzer::ShareEventInfoFrom(const PirStreamAnalyzer& src) {
  event_info_ = src.GetEventInfo();
  is_event_info_build_ = true;
}

std::shared_ptr<
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>
PirStreamAnalyzer::GetEventInfo() const {
  return event_info_;
}

}  // namespace paddle::framework::interpreter
