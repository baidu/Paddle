// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_defs.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/platform/device_event_base.h"
#include "paddle/phi/core/utils/rw_lock.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

#define SCOPE_VARS_READER_LOCK AutoRDLock auto_lock(&vars_lock_);
#define SCOPE_VARS_WRITER_LOCK AutoWRLock auto_lock(&vars_lock_);

namespace paddle {
namespace framework {

class InstructionBase;
class ValueExecutionInfo;
using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;

using HookFunc = std::function<void(OperatorBase*, Scope*)>;
using PirHookFunc =
    std::function<void(InstructionBase*, ValueExecutionInfo*, Scope*)>;

struct OpKernelFunc {
  OpKernelComputeFunc compute_func_;
};

struct VariableMetaInfo {
  int var_ref_count_{0};
  framework::VarDesc* var_desc_{nullptr};
  bool skip_inplace_{false};

  VariableMetaInfo() {}
  VariableMetaInfo(int var_ref_count, framework::VarDesc* var_desc)
      : var_ref_count_(var_ref_count), var_desc_(var_desc) {}
};

class VariableScope {
 public:
  explicit VariableScope(Scope* scope);

  Scope* GetMutableScope() const;

  Scope* GetMutableLocalScope() const;

  void SetScope(Scope* scope);

  void SetLocalScope(Scope* local_scope);

  ~VariableScope() = default;

  // Get variable id by name, return -1 if not found
  int GetIdByName(const std::string& name) const;

  // Get variable name by id, return "" if not found
  std::string GetNameById(int id) const;

  bool HasVar(const std::string& name) const;

  int VarId(const std::string& name) const;

  size_t VarSize() const;

  void AddVar(const std::string& name, VarDesc* var_desc);

  Variable* VarRef(int id) const;

  void SetVarDesc(const std::string& name, framework::VarDesc* var_desc);

  paddle::framework::VarDesc* VarDesc(const std::string& name) const;

  paddle::framework::VarDesc* VarDesc(int id) const;

  void CheckExist(int id) const;

  void CheckExist(const std::string& name) const;

  std::vector<VariableMetaInfo>& MutableVecMetaInfo() { return vec_meta_info_; }

  const std::vector<VariableMetaInfo>& VecMetaInfo() const {
    return vec_meta_info_;
  }

  const std::vector<std::pair<std::string, int>>& DataTransferAddedVars()
      const {
    return data_transfer_added_vars_;
  }

  std::vector<std::pair<std::string, int>>& MutableDataTransferAddedVars() {
    return data_transfer_added_vars_;
  }

  std::vector<Variable*>& MutableVarList() { return var_list_; }

  void SetVarSkipInplace(const std::string& name, bool skip);

  bool GetVarSkipInplace(int id) const;

 private:
  // not owned, better remove it since all vars should be
  // accessed by Scope instead of VariableScope
  std::vector<Variable*> var_list_;

  std::map<std::string, int> name2id_;
  std::vector<VariableMetaInfo> vec_meta_info_;

  Scope* scope_{nullptr};
  // TODO(zhiqiu): find a better way to support local scope.
  Scope* local_scope_{nullptr};
  // mutable RWLock vars_lock_;

  // var_name -> var_type
  std::vector<std::pair<std::string, int>> data_transfer_added_vars_;
};

struct OpFuncNode {
  int stream_priority_{0};  // lower value, higher priority
  // fit for phi kernel
  phi::Kernel* phi_kernel_{nullptr};  // not owned
  phi::DeviceContext* dev_ctx_;       // not owned

  std::map<int, int> inplace_back_map;

  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;

  // TODO(zhiqiu): Better make it unique_ptr
  std::shared_ptr<OperatorBase> operator_base_{nullptr};
  std::string execution_stream_{kDefaultStream};
  bool force_record_event_{false};
  std::vector<std::string> events_to_wait_;
  std::string event_to_record_{"default"};

  OpFuncType type_;
  OpKernelComputeFunc kernel_func_;

  SchedulingPriority scheduling_priority_{0};  // lower value, higher priority

  // the next only for new IR
  phi::KernelContext kernel_context_;
  phi::InferMetaContext infer_meta_context_;
  std::string phi_op_name_;
  paddle::dialect::InferMetaInterface::Concept* infer_meta_interface_{nullptr};

  bool fluid_op{false};
  std::shared_ptr<RuntimeContext> runtime_ctx_{nullptr};
};

class Instruction {
 public:
  Instruction(size_t id,
              OpFuncNode&& op_func_node,
              const phi::DeviceContext& dev_ctx);

  bool IsArtificial() const { return is_artificial_; }

  const std::vector<size_t>& NextInstrsInDifferenceThread() const {
    return next_instrs_in_different_thread;
  }

  const std::vector<size_t>& NextInstrsInSameThread() const {
    return next_instrs_in_same_thread;
  }

  size_t Id() const { return id_; }

  void AddEventToRecord(std::shared_ptr<platform::DeviceEvent> event,
                        platform::DeviceType waiter_type) {
    event_to_record_ = std::make_shared<EventInter>(id_, event, waiter_type);
  }

  void AddEventToWait(size_t instr_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type) {
    events_to_wait_.emplace_back(instr_id, event, waiter_type);
  }

  void AddEventToWait(const EventInter* event_inter) {
    events_to_wait_.push_back(*event_inter);
  }

  const std::vector<EventInter>& EventsToWait() const {
    return events_to_wait_;
  }

  const std::shared_ptr<EventInter>& EventToRecord() const {
    return event_to_record_;
  }

  void AddNextInstrInDifferentThread(size_t id) {
    next_instrs_in_different_thread.push_back(id);
  }

  void AddNextInstrInSameThread(size_t id) {
    next_instrs_in_same_thread.push_back(id);
  }

  void RecordEvent(const Place& place) const;

  void WaitEvent(const Place& place) const;

  const std::map<std::string, std::vector<int>>& Inputs() const;

  const std::map<std::string, std::vector<int>>& Outputs() const;

  const std::unordered_set<int>& NoDataTransformVars() const;

  OpKernelComputeFunc KernelFunc() const;

  phi::Kernel* PhiKernel() const;

  OpFuncType KernelType() const;

  const std::map<int, int>& InplaceBackMap() const;

  OperatorBase* OpBase() const;

  bool OpBaseValid() const;

  void AddGCCheckVar(size_t id);

  const std::vector<size_t>& GCCheckVars() const;

  void ResetContext(const VariableValueMap& in_vars,
                    const VariableValueMap& out_vars,
                    const std::string& op_name);

  void ResetContextWithScope(const VariableValueMap& in_vars,
                             const VariableValueMap& out_vars,
                             const framework::Scope& scope,
                             const std::string& op_name);

  std::shared_ptr<RuntimeContext> InnerRuntimeContext() const;

  std::shared_ptr<RuntimeInferShapeContext> InnerInferShapeContext() const;

  const phi::InferMetaContext* InnerCompatInferMetaContext() const;

  std::shared_ptr<ExecutionContext> InnerExecutionContext() const;

  const phi::DeviceContext& DeviceContext() const;

  const std::vector<std::pair<const Variable*, Variable*>>& InplaceInfo() const;

  void AddInplace(const Variable* in, Variable* out);

  void ClearInplace();

  SchedulingPriority GetSchedulingPriority() const {
    return op_func_node_.scheduling_priority_;
  }

  bool PreDefineContext() const { return pre_define_context_; }

  const OpFuncNode* OpFunc() const { return &op_func_node_; }

  // record stream for gc
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  bool need_record_stream_for_gc_ = false;
  gpuStream_t stream_{nullptr};
  void UpdateRecordStreamForGcInfo();
#endif

  bool can_use_infermeta_ctx_ = false;

 private:
  bool is_artificial_;  // Instruction is artificial means that it is only used
                        // to assist scheduling and no need to be executed.

  size_t id_;

  std::vector<size_t> next_instrs_in_different_thread;
  std::vector<size_t> next_instrs_in_same_thread;

  std::shared_ptr<EventInter> event_to_record_;
  std::vector<EventInter> events_to_wait_;

  OpFuncNode op_func_node_;
  const phi::DeviceContext& dev_ctx_;  // not owned

  std::shared_ptr<RuntimeContext> runtime_ctx_;
  std::shared_ptr<RuntimeInferShapeContext> infershape_ctx_;
  paddle::framework::CompatInferMetaContext compat_infermeta_ctx_;
  std::shared_ptr<ExecutionContext> execution_ctx_;

  std::vector<size_t> gc_check_vars_;

  std::vector<std::pair<const Variable*, Variable*>> vec_inplace_in_to_out_;

  bool pre_define_context_{false};
};

namespace interpreter {
static constexpr char kMemcpyH2D[] = "memcpy_h2d";
static constexpr char kMemcpyD2H[] = "memcpy_d2h";
static constexpr char kFetchVarName[] = "fetch";

// static_ref_ is the number of last live ops calculated to statically after
// `build` the Instructions. dynamic_ref_  is the runtime version ref which will
// be decreased by one dynamically after the execution of an op (in last ops
// list). var_ is the related variable

// The dynamic_ref_ is initialized to static_ref_ first, and is decreased to 1
// during interpretercore's execution, after the interpretercore run, it `reset`
// all dynamic_ref_, i.e., dynamic_ref_ = static_ref_ see ResetAtomicGuard for
// details
class VarRefInfo {
 public:
  explicit VarRefInfo(size_t ref, Variable* var)
      : static_ref_(ref), dynamic_ref_(ref), var_(var) {}
  size_t DynamicRef() { return dynamic_ref_; }
  Variable* Var() { return var_; }
  void ResetDynamicRef() {
    if (static_ref_ != 1) {
      dynamic_ref_ = static_ref_;
    }
  }
  void ResetVariable(Variable* new_var) { var_ = new_var; }
  bool CheckAndDecrease() {
    return static_ref_ == 1 || (dynamic_ref_.fetch_sub(1) == 1);
  }

 private:
  const size_t static_ref_;
  std::atomic<size_t> dynamic_ref_;
  Variable* var_;
};

// static_dep_ is the number of dependencies (ops that must run before it) of
// each op which is calculated to statically. static_dep_  is the runtime
// version dep which will be decreased by one dynamically after the execution of
// one dependency op.

// The dynamic_dep_ is initialized to static_dep_ first, and is decreased to 1
// during interpretercore's execution, after the interpretercore run, it `reset`
// all dynamic_dep_, i.e., dynamic_dep_ = static_dep_ see ResetAtomicGuard for
// details

class OpDepInfo {
 public:
  explicit OpDepInfo(size_t dep) : static_dep_(dep), dynamic_dep_(dep) {}
  size_t DynamicDep() { return dynamic_dep_; }
  void ResetDynamicDep() {
    if (static_dep_ != 1) {
      dynamic_dep_ = static_dep_;
    }
  }
  bool CheckAndDecrease() {
    return static_dep_ == 1 || (dynamic_dep_.fetch_sub(1) == 1);
  }
  size_t StaticDep() const { return static_dep_; }
  size_t DynamicDep() const { return dynamic_dep_; }

 private:
  const size_t static_dep_;
  std::atomic<size_t> dynamic_dep_;
};

class ResetAtomicGuard {
 public:
  ResetAtomicGuard(std::vector<std::shared_ptr<OpDepInfo>>* deps,
                   std::vector<std::shared_ptr<VarRefInfo>>* refs)
      : deps_(deps), refs_(refs) {}

  ~ResetAtomicGuard() {
    VLOG(10) << "Reset DynamicDep";
    for (auto&& dep : *deps_) {
      dep->ResetDynamicDep();
    }
    VLOG(10) << "Reset DynamicRef";
    for (auto&& ref : *refs_) {
      ref->ResetDynamicRef();
    }
  }

 private:
  std::vector<std::shared_ptr<OpDepInfo>>* deps_;
  std::vector<std::shared_ptr<VarRefInfo>>* refs_;
};

}  // namespace interpreter

}  // namespace framework
}  // namespace paddle
