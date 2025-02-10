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

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"

#include <algorithm>

#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/io/save_load_tensor.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/data_transfer.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/interpreter/static_build.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/controlflow/pylayer_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/memory/stats.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

COMMON_DECLARE_bool(use_mkldnn);
COMMON_DECLARE_bool(check_nan_inf);
COMMON_DECLARE_string(static_runtime_data_save_path);
COMMON_DECLARE_bool(save_static_runtime_data);

namespace paddle::framework::interpreter {

using VariableIdMap = std::map<std::string, std::vector<int>>;

// NOTE(Ruibiao): SingleStreamGuard make some multi-stream op (i.e.,
// c_allreduce_sum) run in single stream. It is dedicated to BuildOpFuncList
// which run kernel without stream synchronization.
class SingleStreamGuard {
 public:
  explicit SingleStreamGuard(std::shared_ptr<OperatorBase>& op) : op_(op) {
    if (op_->Type() == "c_allreduce_sum" &&
        op_->Attr<bool>("use_calc_stream") == false) {
      VLOG(6) << "Set c_allredce_sum's attr use_calc_stream to true";
      op_->SetAttr("use_calc_stream", true);
      is_changed = true;
    }
  }

  ~SingleStreamGuard() {  // NOLINT
    if (!is_changed) {
      return;
    }

    if (op_->Type() == "c_allreduce_sum") {
      op_->SetAttr("use_calc_stream", false);
      VLOG(6) << "Set c_allredce_sum's attr use_calc_stream to false";
    }
  }

  DISABLE_COPY_AND_ASSIGN(SingleStreamGuard);

 private:
  bool is_changed{false};
  std::shared_ptr<OperatorBase> op_;
};

const std::vector<WorkQueueOptions> ConstructWorkQueueOptions(
    size_t host_num_threads, size_t device_num_threads, EventsWaiter* waiter) {
  std::vector<WorkQueueOptions> group_options;
  // for execute host Kernel
  group_options.emplace_back(/*name*/ "HostTasks",
                             /*num_threads*/ host_num_threads,
                             /*allow_spinning*/ true,
                             /*always_spinning*/ false,
                             /*track_task*/ false,
                             /*detached*/ true,
                             /*events_waiter*/ waiter);
  // for launch device Kernel
  group_options.emplace_back(/*name*/ "DeviceKernelLaunch",
                             /*num_threads*/ device_num_threads,
                             /*allow_spinning*/ true,
                             /*always_spinning*/ false,
                             /*track_task*/ false,
                             /*detached*/ true,
                             /*events_waiter*/ waiter);
  return group_options;
}

AsyncWorkQueue::AsyncWorkQueue(size_t host_num_threads,
                               size_t device_num_threads,
                               EventsWaiter* waiter)
    : host_num_thread_(host_num_threads),
      queue_group_(CreateWorkQueueGroup(ConstructWorkQueueOptions(
          host_num_threads, device_num_threads, waiter))) {}

void AsyncWorkQueue::AddTask(const OpFuncType& op_func_type,
                             std::function<void()> fn) {
  // queue_idx=0 : kCpuSync or kGpuSync
  // queue_idx=1 : kGPUAsync
  queue_group_->AddTask(op_func_type == OpFuncType::kGpuAsync, std::move(fn));
}

bool IsCommunicationOp(const OperatorBase* op) {
  const std::string& op_name = op->Type();
  const std::set<std::string> special_comm_op_set = {
      "send",
      "recv",
      "send_v2",
      "recv_v2",
  };
  const std::string communication_op_prefix = "c_";
  if (op_name.find(communication_op_prefix) != std::string::npos ||
      special_comm_op_set.count(op_name)) {
    return true;
  }
  if (op->HasAttr("ring_id")) {
    return true;
  }
  return false;
}

bool IsCommunicationOp(const Instruction& instr) {
  if (!instr.OpBaseValid()) {
    return false;
  }
  return IsCommunicationOp(instr.OpBase());
}

bool IsCommunicationOp(const ::pir::Operation* op) {
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }
  const std::set<std::string> special_comm_op_set = {
      paddle::dialect::SendV2Op::name(),
      paddle::dialect::RecvV2Op::name(),
  };
  const std::string communication_op_prefix = "c_";
  if (op_name.find(communication_op_prefix) != std::string::npos ||
      special_comm_op_set.count(op_name)) {
    return true;
  }
  if (op->attributes().count("ring_id") != 0) {
    return true;
  }
  return false;
}

bool IsCpuOp(const Instruction& instr) {
  return phi::is_cpu_place(instr.DeviceContext().GetPlace());
}

bool IsCpuOp(Instruction* instr) {
  return phi::is_cpu_place(instr->DeviceContext().GetPlace());
}

bool IsCpuOp(const paddle::framework::InstructionBase& instr) {
  return phi::is_cpu_place(instr.DeviceContext().GetPlace());
}

bool IsCpuOp(paddle::framework::InstructionBase* instr) {
  return phi::is_cpu_place(instr->DeviceContext().GetPlace());
}

bool IsGradOp(const std::string& op_name) {
  return paddle::string::ends_with(op_name, "_grad");
}

bool IsSupportedHeterPlace(const phi::Place& place) {
  return phi::is_gpu_place(place) || phi::is_xpu_place(place) ||
         phi::is_ipu_place(place) || phi::is_custom_place(place);
}

bool IsMemcpyD2H(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyD2H;
}

bool IsMemcpyH2D(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyH2D;
}

bool IsMemcpyH2D(Instruction* instr) {
  return instr->OpBase()->Type() == kMemcpyH2D;
}

bool IsMemcpyH2D(paddle::framework::InstructionBase* instr) {
  return instr->Name() == "pd_op.memcpy_h2d";
}

bool IsMemcpyOp(const Instruction& instr) {
  return IsMemcpyD2H(instr) || IsMemcpyH2D(instr);
}

void AddFetch(const std::vector<std::string>& fetch_names,
              framework::BlockDesc* block) {
  auto* fetch_holder = block->Var(kFetchVarName);
  fetch_holder->SetType(proto::VarType::FETCH_LIST);
  fetch_holder->SetPersistable(true);

  int i = 0;
  for (auto& fetch_name : fetch_names) {
    // append fetch op
    auto* op = block->AppendOp();
    op->SetType("fetch_v2");
    op->SetInput("X", {fetch_name});
    op->SetOutput("Out", {kFetchVarName});
    op->SetAttr("col", {static_cast<int>(i)});
    op->CheckAttrs();
    i++;
  }
}

bool var_can_be_deleted(const std::string& name, const BlockDesc& block) {
  auto* var_desc = block.FindVar(name);
  if (var_desc == nullptr || var_desc->Persistable()) {
    return false;
  }

  auto type = var_desc->Proto()->type().type();

  return type == proto::VarType::DENSE_TENSOR ||
         type == proto::VarType::SELECTED_ROWS ||
         type == proto::VarType::DENSE_TENSOR_ARRAY ||
         type == proto::VarType::SPARSE_COO ||
         type == proto::VarType::SPARSE_CSR;
}

std::unordered_map<const paddle::framework::OperatorBase*,
                   std::vector<std::string>>
GetUnusedVars(const BlockDesc& block,
              const std::vector<std::shared_ptr<OperatorBase>>& ops) {
  std::unordered_map<std::string, size_t> var_op_idx_map;

  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];

    OpInOutInfo info;
    for (auto& name_pair : op->Inputs()) {
      for (auto& name : name_pair.second) {
        if (!var_can_be_deleted(name, block)) {
          continue;
        }

        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(op.get());
        }

        if (info.IsInArgBufferNeeded(name)) {
          // Update the last living op of variable to current op
          var_op_idx_map[name] = i;
        } else {
          VLOG(10) << "Skip reference count computing of variable "
                   << name_pair.first << "(" << name << ") in Operator "
                   << op->Type();
        }
      }
    }

    for (auto& name_pair : op->Outputs()) {
      for (auto& name : name_pair.second) {
        if (var_can_be_deleted(name, block)) {
          // Update the last living op of variable to current op
          var_op_idx_map[name] = i;
        }
      }
    }
  }

  std::unordered_map<const OperatorBase*, std::vector<std::string>> result;
  for (auto& name_op_idx_pair : var_op_idx_map) {
    auto& name = name_op_idx_pair.first;
    size_t op_idx = name_op_idx_pair.second;
    auto op = ops[op_idx].get();
    result[op].emplace_back(name);
    VLOG(4) << op->Type() << " " << name;
  }
  VLOG(4) << "gc map size:" << result.size();
  return result;
}

OpFuncType AnalyseOpFuncType(const OpFuncNode& op_func_node,
                             const phi::Place& place) {
  if (phi::is_cpu_place(place)) {
    return OpFuncType::kCpuSync;
  }

  PADDLE_ENFORCE_EQ(
      IsSupportedHeterPlace(place),
      true,
      common::errors::Fatal("Unsupported current place %s", place));

  // Some GPU OPs do not launch CUDA Kernel, but spend a lot of time on CPU
  // computing. They execute serially in device thread and block CUDA kernel
  // launching in other GPU OPs. To improve performance, set them as kGpuSync
  // and so that they would be dispatched to host thread.
  std::shared_ptr<OperatorBase> op = op_func_node.operator_base_;
  if (op->Type() == kCoalesceTensor &&
      (!phi::is_xpu_place(place) ||
       op->Attr<bool>("persist_output") == false) &&
      op->Attr<bool>("set_constant") == false &&
      op->Attr<bool>("copy_data") == false) {
    return OpFuncType::kGpuSync;
  }

  // for memcpy explicitly called by user
  if (phi::is_gpu_place(place) && op->Type() == interpreter::kMemcpyD2H) {
    return OpFuncType::kGpuSync;
  }

  if (op->Type() == "shape") {
    return OpFuncType::kGpuSync;
  }
  return OpFuncType::kGpuAsync;
}

void CreateAllOps(const framework::BlockDesc& block,
                  std::vector<std::unique_ptr<OperatorBase>>* ops) {
  for (auto& op : block.AllOps()) {
    auto op_type = op->Type();
    VLOG(8) << "CreateOp from : " << op_type;

    auto& info = OpInfoMap::Instance().Get(op_type);

    const VariableNameMap& inputs_names = op->Inputs();
    const VariableNameMap& outputs_names = op->Outputs();

    AttributeMap op_attr_map = op->GetAttrMap();
    AttributeMap op_runtime_attr_map = op->GetRuntimeAttrMap();

    if (info.Checker() != nullptr) {
      info.Checker()->Check(&op_attr_map);
    }

    const auto& extra_attr_checkers =
        operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(op_type);
    for (const auto& checker : extra_attr_checkers) {
      checker(&op_runtime_attr_map, false);
    }

    auto op_base =
        info.Creator()(op_type, inputs_names, outputs_names, op_attr_map);
    op_base->SetRuntimeAttributeMap(op_runtime_attr_map);

#ifdef PADDLE_WITH_DNNL
    if (FLAGS_use_mkldnn) {
      if (op->HasAttr("use_mkldnn")) {
        VLOG(4) << "Set use_mkldnn=True for " << op_base->Type();
        op_base->SetAttr("use_mkldnn", true);
      }
    }
#endif

    ops->emplace_back(std::unique_ptr<OperatorBase>(op_base));
  }
}

std::tuple<VariableValueMap, VariableIdMap> BuildVariableMap(
    const VariableNameMap& var_name_map,
    VariableScope* var_scope,
    Scope* local_scope,
    bool find_var_recursively = false,
    bool allow_var_not_in_scope = false) {
  VariableValueMap name2var;
  VariableIdMap name2id;
  for (auto& item : var_name_map) {
    std::vector<Variable*> vars;
    std::vector<int> ids;
    vars.reserve(item.second.size());

    for (auto& var_name : item.second) {
      auto* var = local_scope->FindVar(var_name);

      if (!var_scope->HasVar(var_name)) {
        if (find_var_recursively && var) {
          VLOG(3) << "Add " << var_name << " to var_scope";
          var_scope->AddVar(var_name, nullptr);
        } else if (allow_var_not_in_scope) {
          VLOG(4) << var_name << " don't exist in variable scope, skip it!";
          continue;
        }
      }
      auto var_id = var_scope->VarId(var_name);
      vars.push_back(var);
      ids.push_back(var_id);
    }
    name2var[item.first] = std::move(vars);
    name2id[item.first] = std::move(ids);
  }
  return std::make_tuple(name2var, name2id);
}

void ApplyDeviceGuard(const OperatorBase* op_base,
                      const phi::Place& place,
                      OpKernelType* expected_kernel_key) {
  bool need_change_place =
      (op_base->HasAttr("op_device") &&
       (op_base->Attr<std::string>("op_device").length() > 0));
  if (need_change_place) {
    auto& op_device = op_base->Attr<std::string>("op_device");
    if (op_device == "cpu" || phi::is_cpu_place(place)) {
      VLOG(3) << "Switch into CPUPlace by device_guard.";
      expected_kernel_key->place_ = phi::CPUPlace();
    } else if (op_device.find("gpu") != std::string::npos &&
               phi::is_gpu_place(place)) {
      // when the Op that does not have GPUKernel is assigned to GPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      if (op_base->SupportGPU()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = phi::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << op_base->Type()
            << ") has no CUDA implementation. It will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else if (op_device.find("xpu") != std::string::npos &&
               phi::is_xpu_place(place)) {
      // when the Op that does not have XPUKernel is assigned to XPU, the
      // CPUKernel will be executed and a warning will be given at the same
      // time.
      if (op_base->SupportXPU()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = phi::CPUPlace();
        LOG_FIRST_N(WARNING, 1)
            << "Op(" << op_base->Type()
            << ") has no XPU implementation. It will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else if (phi::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
      bool is_custom_device_op = false;
      for (auto dev_type : device_types) {
        if (op_device.find(dev_type) != std::string::npos) {
          is_custom_device_op = true;
          break;
        }
      }
      PADDLE_ENFORCE_EQ(
          is_custom_device_op,
          true,
          common::errors::Unimplemented(
              "Unsupported current device %s with Paddle CustomDevice ",
              op_device));
#else
      VLOG(1) << string::Sprintf(
          "Cannot use get_all_custom_device_type because you have installed "
          "CPU/GPU version PaddlePaddle.\n"
          "If you want to use get_all_custom_device_type, please try to "
          "install CustomDevice version "
          "PaddlePaddle by: pip install paddlepaddle\n");
#endif
      if (op_base->SupportCustomDevice()) {
        expected_kernel_key->place_ = place;
      } else {
        expected_kernel_key->place_ = phi::CPUPlace();
        LOG_FIRST_N(WARNING, 1) << "Op(" << op_base->Type()
                                << ") has no Custom Place implementation. It "
                                   "will be assigned to CPUPlace.";
      }
      VLOG(3) << "Switch into " << expected_kernel_key->place_
              << " by device_guard.";
    } else {
      PADDLE_THROW(
          common::errors::Fatal("Unsupported current place %s", op_device));
    }
  }
}

phi::DeviceContext* ConstructDeviceContext(const OperatorBase* op,
                                           const phi::Place& place) {
  auto& pool = phi::DeviceContextPool::Instance();
  auto* default_dev_ctx = pool.Get(place);

  // Replace the default_dev_ctx according to dst_place_type for memcpy op if
  // needed.

  // NOTE(liudongxue01):
  //  Please apply the following logic in other Executor/Interpreter modules
  //  likewise.
  //

  // NOTE(liudongxue01):
  // The following code aims to fixup the memcpy kernel which does not handle
  // some rare case. The case is:
  //  1. The default place in the current execution context is not CUDAPlace,
  //  such as CPUPlace,
  //  2. The dst_place_type is 1 which means CUDAPlace,
  //  3. The expected result place is CUDAPlace but the actual result is
  //  CPUPlace.
  // When the default place is CPUPlace, we call the tensor.cuda() would
  // simply hit such case.
  //
  // Q: Why we do not add such logic in the memcpy kernel?
  // A: (1) To fixup the memcpy kernel, we need to construct a CUDAPlace() and
  //    corresponding DeviceContext instance which used by the phi::Copy(...)
  //    api to perform the real memcpy action. (2) We should not access the
  //    singleton of the DeviceContextPool object in the PHI framework which
  //    is designed as a standalone module and all context data should passed
  //    into the kernel API through arguments. (3) So we have no way to
  //    construct a CUDAPlace() in the memcpy kernel and then pass it
  //     to the phi::Copy(...) api.
  if (!phi::is_gpu_place(place)) {
    const auto& op_type = op->Type();
    if (op_type == "memcpy") {
      int dst_place_type = op->Attr<int>("dst_place_type");
      if (dst_place_type == 1) {  // 1 : CUDAPlace
        auto dev_ctx = pool.Get(paddle::DefaultGPUPlace());
        VLOG(4) << "Change the device context for memcpy OP: ("
                << default_dev_ctx->type_info().name() << ") -> ("
                << dev_ctx->type_info().name() << ")";
        return dev_ctx;
      }
    }
  }

  return default_dev_ctx;
}

void HandleOperatorBase(
    const phi::Place& place,
    std::shared_ptr<OperatorBase> op,
    OpFuncNode* op_func_node,
    Scope* scope,
    bool static_build,
    std::vector<std::shared_ptr<OperatorBase>> following_ops) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  // input, output is prepared. set the other attributes.
  op_func_node->operator_base_ = op;
  op_func_node->type_ = AnalyseOpFuncType(*op_func_node, place);
  op_func_node->kernel_func_ = nullptr;
  if (static_build) {
    if (OperatorBasesMustRunInStaticBuild.count(op->Type())) {
      op->Run(*scope, place);
    }

    FakeInitializeOutputsForOperatorBase(*op, place, scope, following_ops);
  } else {
    op->Run(*scope, place);  // Run without data transformer.
  }

  op_func_node->dev_ctx_ = dev_ctx;
}

void BuildOpFuncList(const phi::Place& place,
                     const framework::BlockDesc& block,
                     const std::set<std::string>& skip_gc_vars,
                     std::vector<OpFuncNode>* vec_func_list,
                     VariableScope* var_scope,
                     const ExecutionConfig& execution_config,
                     const std::vector<HookFunc>& input_hookfuncs,
                     const std::vector<HookFunc>& output_hookfuncs,
                     bool use_local_scope,
                     bool static_build) {
  Scope* local_scope = use_local_scope ? var_scope->GetMutableLocalScope()
                                       : var_scope->GetMutableScope();
  std::vector<std::unique_ptr<OperatorBase>>
      ops_unique;  // its elements will be moved to vec_func_list
  // Step 1: create all ops for current block.
  CreateAllOps(block, &ops_unique);

  VLOG(1) << "Static build: " << static_build;

  if (!execution_config.used_for_jit) {
    // If gc is enabled and block size > 1
    const ProgramDesc& main_program = *block.Program();
    operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
        main_program, block.ID(), ops_unique);
    operators::PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
        main_program, block.ID(), ops_unique);
    operators::PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
        main_program, block.ID(), ops_unique);
  }

#ifdef PADDLE_WITH_DNNL
  platform::RegisterModelLayout(ops_unique, place);
#endif
  // its elements will be moved to vec_func_list
  std::vector<std::shared_ptr<OperatorBase>> ops;
  for (auto& op_unique : ops_unique) {
    ops.emplace_back(std::move(op_unique));
  }
  auto unused_var_map = GetUnusedVars(block, ops);

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();

  bool flag_log_is_printed = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto op = ops[i].get();
    op->SetId(i);

    const std::string& op_type = op->Type();
    if (execution_config.used_for_inference) {
      if (op_type == "feed" || op_type == "fetch") {
        continue;
      }
      for (auto& hook : input_hookfuncs) {
        hook(op, local_scope);
      }

      if (op->Type() == "while" || op->Type() == "conditional_block") {
        op->SetInputHooks(input_hookfuncs);
        op->SetOutputHooks(output_hookfuncs);
        auto runtime_attrs = op->RuntimeAttrs();
        runtime_attrs.insert(std::make_pair("used_for_inference", true));
        op->SetRuntimeAttributeMap(runtime_attrs);
      }
    }

    VLOG(6) << "Build OpFuncNode from : " << op_type;

    // Print new executor log if grad op is used.
    // It's only for test and will be removed later.
    if (!flag_log_is_printed && op_type.find("_grad") != std::string::npos) {
      LOG_FIRST_N(INFO, 1) << "Standalone Executor is Used.";
      flag_log_is_printed = true;
    }

    // Hot fix for variables used in dataloader, like
    // 'lod_tensor_blocking_queue_0'. These variables may be created in scope,
    // and it is not existed as variable in program.
    const std::set<std::string> ops_with_var_not_in_program = {
        "create_py_reader"};
    const std::set<std::string> ops_with_var_not_in_scope = {
        "conditional_block",
        "conditional_block_grad",
        "pylayer",
        "pylayer_grad"
        "recurrent_grad",
        "while",
        "while_grad"};
    bool allow_var_not_in_program = ops_with_var_not_in_program.count(op_type);
    bool allow_var_not_in_scope = ops_with_var_not_in_scope.count(op_type);

    // ops in the control flow block may not find its inputs or outputs
    // in VariableScope of the sub-block, so we need search it in parent scope.

    framework::VariableNameMap& input_name_map = op->Inputs();
    VariableValueMap ins_map;
    VariableIdMap ins_name2id;
    std::tie(ins_map, ins_name2id) = BuildVariableMap(
        input_name_map,
        var_scope,
        local_scope,
        execution_config.used_for_control_flow_op || allow_var_not_in_program,
        allow_var_not_in_scope);

    framework::VariableNameMap& output_name_map = op->Outputs();
    VariableValueMap outs_map;
    VariableIdMap outs_name2id;
    std::tie(outs_map, outs_name2id) =
        BuildVariableMap(output_name_map,
                         var_scope,
                         local_scope,
                         execution_config.used_for_control_flow_op,
                         allow_var_not_in_scope);

    // step 1: build OpFuncNode
    OpFuncNode op_func_node;
    op_func_node.operator_base_ = ops[i];
    op_func_node.input_index = ins_name2id;
    op_func_node.output_index = outs_name2id;

    const OperatorDistAttr* dist_attr =
        block.Op(static_cast<int>(i))->DistAttr();
    if (dist_attr) {
      if (dist_attr->execution_stream() !=
          distributed::auto_parallel::kDefault) {
        op_func_node.execution_stream_ = dist_attr->execution_stream();
      }
      op_func_node.stream_priority_ = dist_attr->stream_priority();
      op_func_node.scheduling_priority_ = dist_attr->scheduling_priority();
      // set manual event information
      op_func_node.force_record_event_ = dist_attr->force_record_event();
      op_func_node.events_to_wait_ = dist_attr->events_to_wait();
      op_func_node.event_to_record_ = dist_attr->event_to_record();
    } else {
      if (interpreter::IsCommunicationOp(op)) {
        // NOTE(Ruibiao): Dispatching computation before communication improves
        // multi-stream overlap when the time cost of communication less than
        // that of the calculation (e.g., ResNet50_bs128_pure_fp16 N4C32
        // training).
        op_func_node.scheduling_priority_ = 1;
      }
    }

    VLOG(6) << op_type
            << " : [execution_stream, stream_priority, scheduling_priority] = ["
            << op_func_node.execution_stream_ << ", "
            << op_func_node.stream_priority_ << ", "
            << op_func_node.scheduling_priority_ << "]";

    SingleStreamGuard single_stream_guard(ops[i]);

    VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

    try {
      if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr) {
        VLOG(4) << "HandleOperatorBase";
        // op is not a operatorwithkernel, so directly run OperatorBase::Run()

        std::vector<std::shared_ptr<OperatorBase>> following_ops(
            ops.begin() + static_cast<int>(i) + 1, ops.end());
        HandleOperatorBase(place,
                           ops[i],
                           &op_func_node,
                           local_scope,
                           static_build,
                           following_ops);
        vec_func_list->emplace_back(op_func_node);
      } else {
        VLOG(4) << "OP is not null";
        auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
            static_cast<const framework::OperatorWithKernel*>(op));
        VLOG(4) << "get op_with_kernel";
        // construct RuntimeContext and analysis KernelType
        RuntimeContext runtime_context({}, {});
        runtime_context.inputs.swap(ins_map);
        runtime_context.outputs.swap(outs_map);
        VLOG(4) << "get RuntimeContext";

        Scope scope, *runtime_scope = &scope;

        // construct the device context
        auto* dev_ctx = ConstructDeviceContext(op, place);

        SetDeviceCommContext(op, dev_ctx);
        auto exec_ctx = ExecutionContext(
            *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
        auto expected_kernel_key = framework::TransPhiKernelKeyToOpKernelType(
            op_with_kernel->GetExpectedKernelType(exec_ctx));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        if (op_with_kernel->CanCUDNNBeUsed(exec_ctx,
                                           expected_kernel_key.data_type_)) {
          expected_kernel_key.library_type_ = framework::LibraryType::kCUDNN;
        }
#endif
        VLOG(4) << "expected_kernel_key : " << expected_kernel_key;
        // change device by the device_guard()
        ApplyDeviceGuard(op, place, &expected_kernel_key);
        if (phi::places_are_same_class(exec_ctx.GetPlace(),
                                       expected_kernel_key.place_)) {
          expected_kernel_key.place_ = exec_ctx.GetPlace();
        }

        // step 2. select op kernel
        auto run_phi_kernel = false;
        if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
                op_with_kernel->Type())) {
          auto phi_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
          auto phi_kernel_name = op_with_kernel->PhiKernelSignature()->name;
          bool in_custom_back_list = false;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
          in_custom_back_list =
              phi::backends::custom_device::is_in_custom_black_list(
                  phi_kernel_name);
#endif
          if (op_with_kernel->PhiKernel()->IsValid() && !in_custom_back_list) {
            run_phi_kernel = true;
          } else {
            if ((!op_with_kernel->SupportsKernelType(expected_kernel_key,
                                                     exec_ctx)) ||
                in_custom_back_list) {
              std::string info = in_custom_back_list ? "fluid in black list "
                                                     : "fluid missing ";
              VLOG(3) << info << phi_kernel_key
                      << " kernel: " << phi_kernel_name;
              auto phi_cpu_kernel_key =
                  FallBackToCpu(phi_kernel_key, *op_with_kernel);
              op_with_kernel->ResetPhiKernel(
                  new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
                      phi_kernel_name, phi_cpu_kernel_key)));
              if (op_with_kernel->PhiKernel()->IsValid()) {
                VLOG(6) << "Static graph mode PrepareImpl - kernel name: "
                        << phi_kernel_name
                        << " | kernel key: " << phi_cpu_kernel_key
                        << " | kernel: " << *(op_with_kernel->PhiKernel());
                op_with_kernel->ResetKernelType(new OpKernelType(
                    TransPhiKernelKeyToOpKernelType(phi_cpu_kernel_key)));
                run_phi_kernel = true;
              }
            }
          }
        }

        VLOG(4) << "if run phi kernel? : " << run_phi_kernel;
        if (!run_phi_kernel) {
          op_with_kernel->ChooseKernel(exec_ctx);
          op_func_node.kernel_func_ = *op_with_kernel->kernel_func();
        } else {
          op_func_node.phi_kernel_ = op_with_kernel->PhiKernel();
        }
        auto kernel_type = *(op_with_kernel->kernel_type());
        if (kernel_type.place_ != dev_ctx->GetPlace()) {
          dev_ctx = pool.Get(kernel_type.place_);
        }
        op_func_node.dev_ctx_ = dev_ctx;
        op_func_node.type_ =
            AnalyseOpFuncType(op_func_node, kernel_type.place_);

        VLOG(3) << op_with_kernel->Type()
                << " : finally selected kernel_key: " << kernel_type;

        // step 3. data transform
        VariableValueMap& ins_map_temp = runtime_context.inputs;
        VariableValueMap& outs_map_temp = runtime_context.outputs;
        ApplyDataTransform(kernel_type,
                           place,
                           &ins_map_temp,
                           &outs_map_temp,
                           var_scope,
                           &op_func_node,
                           vec_func_list,
                           use_local_scope,
                           static_build);
        VLOG(4) << "apply data transform done. ";

        // step 4. infershape
        if (!static_build) {
          VLOG(4) << "infer shape";
          // see kAllKernelsMustComputeRuntimeShape in operator.h for why
          if (!(op->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
                op->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
            RuntimeInferShapeContext infer_shape_ctx(*op, runtime_context);
            // TODO(Aurelius84): In case of control flow ops, they are NOT
            // inherited from OperatorWithKernel.
            op_with_kernel->Info().infer_shape_(&infer_shape_ctx);
          }
        }

        // step 5. run kernel
        if (run_phi_kernel &&
            op_func_node.phi_kernel_->GetKernelRegisteredType() ==
                phi::KernelRegisteredType::FUNCTION) {
          VLOG(6) << op_type << " run function kernel";
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
          auto attrs = op->Attrs();
          if (attrs.find("ring_id") != attrs.end()) {
            auto ring_id_attr = attrs.at("ring_id");
            int ring_id = PADDLE_GET(int, ring_id_attr);
            auto map = distributed::ProcessGroupMapFromGid::getInstance();
            if (map->has(ring_id)) {
              auto original_stream =
                  static_cast<phi::GPUContext*>(dev_ctx)->cuda_stream();
              distributed::ProcessGroup* pg = map->get(ring_id);
              auto comm_context =
                  static_cast<paddle::distributed::ProcessGroupNCCL*>(pg)
                      ->GetOrCreateCommContext(place);
              dev_ctx =
                  static_cast<phi::distributed::NCCLCommContext*>(comm_context)
                      ->GetDevContext();
              dev_ctx->SetCommContext(comm_context);

              static_cast<phi::GPUContext*>(dev_ctx)->SetCUDAStream(
                  original_stream, false);
              auto& instance =
                  paddle::memory::allocation::AllocatorFacade::Instance();
              dev_ctx->SetAllocator(
                  instance
                      .GetAllocator(
                          place,
                          static_cast<phi::GPUContext*>(dev_ctx)->stream())
                      .get());
            } else {
              VLOG(3) << "ring_id " << ring_id
                      << " not found in ProcessGroupMapFromGid ";
            }
          }
#endif
          if (static_build) {
            FakeInitializeOutputsForFunctionKernel(
                *op,
                *(op_func_node.phi_kernel_),
                *(op_with_kernel->PhiKernelSignature()),
                runtime_context,
                *dev_ctx);
          } else {
            phi::KernelContext phi_kernel_context;
            op_with_kernel->BuildPhiKernelContext(
                runtime_context, dev_ctx, &phi_kernel_context);
            (*op_func_node.phi_kernel_)(&phi_kernel_context);
          }
        } else if (run_phi_kernel &&
                   op_func_node.phi_kernel_->GetKernelRegisteredType() ==
                       phi::KernelRegisteredType::STRUCTURE) {
          VLOG(6) << op_type << " run structure kernel";
          ExecutionContext execution_context(
              *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
          if (static_build) {
            FakeInitializeOutputsForStructureKernel(kernel_type,
                                                    &execution_context);
          } else {
            (*op_func_node.phi_kernel_)(&execution_context);
          }
        } else {
          VLOG(6) << op_type << " run fluid kernel";
          // the place of exec_ctx maybe has changed.
          ExecutionContext execution_context(
              *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
          if (static_build) {
            FakeInitializeOutputsForStructureKernel(kernel_type,
                                                    &execution_context);
          } else {
            op_func_node.kernel_func_(execution_context);
          }
        }

        // for debug nan/inf

        vec_func_list->emplace_back(op_func_node);

        if (!op_func_node.inplace_back_map.empty()) {
          auto& m = op_func_node.inplace_back_map;
          // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in
          // operator.cc
          for (auto& p : m) {
            auto* transformed_tensor =
                GetMutableDenseTensorOrSelectedRowsValueFromVar(
                    local_scope->FindVar(var_scope->GetNameById(p.first)));
            auto* original_tensor =
                GetMutableDenseTensorOrSelectedRowsValueFromVar(
                    local_scope->FindVar(var_scope->GetNameById(p.second)));

            // avoid overwriting valid data
            if (static_build && original_tensor->initialized()) {
              const phi::Place& target_place = transformed_tensor->place();
              phi::DeviceContext* dev_ctx_for_copy = nullptr;
              if (target_place.GetType() != AllocationType::CPU) {
                dev_ctx_for_copy = pool.Get(target_place);
              } else {
                dev_ctx_for_copy = pool.Get(original_tensor->place());
              }

              phi::Copy(*dev_ctx_for_copy,
                        *original_tensor,
                        target_place,
                        /*blocking=*/true,
                        original_tensor);
              original_tensor->set_type(transformed_tensor->dtype());
              original_tensor->set_layout(transformed_tensor->layout());
            } else {
              original_tensor->ShareDataWith(*transformed_tensor);
            }
            VLOG(4) << "Transfer inplace variable back form "
                    << var_scope->GetNameById(p.first) << " to "
                    << var_scope->GetNameById(p.second);
          }
        }

        // post-process grad_op.outputs if need cast complex grad into real
        // grad.
        // NOTE(Aurelius84): insert a transfer_dtype_op inplacely to cast it.
        if (IsGradOp(op_type) &&
            framework::IsComplexType(kernel_type.data_type_)) {
          interpreter::HandleComplexGradToRealGrad(op_func_node,
                                                   place,
                                                   output_name_map,
                                                   &runtime_context.outputs,
                                                   var_scope,
                                                   vec_func_list,
                                                   local_scope,
                                                   static_build);
        }
      }
    } catch (platform::EnforceNotMet& ex) {
      framework::InsertCallStackInfo(op_type, op->Attrs(), &ex);
      throw ex;
    } catch (platform::EOFException&) {
      std::rethrow_exception(std::current_exception());
    } catch (std::exception& ex) {
      LOG(WARNING) << op_type << " raises an exception "
                   << common::demangle(typeid(ex).name()) << ", " << ex.what();
      std::rethrow_exception(std::current_exception());
    } catch (...) {
      LOG(WARNING) << op_type << " raises an unknown exception";
      std::rethrow_exception(std::current_exception());
    }

    if (FLAGS_save_static_runtime_data) {
      VLOG(6) << "start to save paddle variable";
      auto root_path = FLAGS_static_runtime_data_save_path;
      for (auto& vname : op->InputVars()) {
        auto* var = local_scope->FindVar(vname);
        if (var == nullptr) continue;
        const phi::DenseTensor* tensor{nullptr};
        if (var->IsType<phi::DenseTensor>()) {
          tensor = &var->Get<phi::DenseTensor>();
        } else {
          VLOG(6) << vname << " is not DenseTensor";
          continue;
        }
        if (!tensor->IsInitialized()) continue;
        paddle::framework::SaveTensor(
            *tensor,
            root_path + "/saved_tensors/" + op_type + "-input-" + vname,
            false);
      }
      for (auto& vname : op->OutputVars(true)) {
        auto* var = local_scope->FindVar(vname);
        if (var == nullptr) continue;
        const phi::DenseTensor* tensor{nullptr};
        if (var->IsType<phi::DenseTensor>()) {
          tensor = &var->Get<phi::DenseTensor>();
        } else {
          VLOG(6) << vname << "  is not DenseTensor";
          continue;
        }
        if (!tensor->IsInitialized()) continue;
        paddle::framework::SaveTensor(
            *tensor,
            root_path + "/saved_tensors/" + op_type + "-output-" + vname,
            false);
      }
      VLOG(6) << "end save paddle variable";
    }

    if (FLAGS_check_nan_inf) {
      VLOG(4) << "Check nan/inf";
      try {
        framework::details::CheckOpHasNanOrInf(*op, *local_scope, place);
      } catch (...) {
        const std::vector<std::string>* callstack = nullptr;
        auto attrs = op->Attrs();
        auto iter =
            attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
        if (iter != attrs.end()) {
          callstack = &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
          if (callstack->empty()) callstack = nullptr;
        }
        std::ostringstream sout;
        if (callstack) {
          if (FLAGS_call_stack_level > 1) {
            sout << "\n\n  Compile Traceback (most recent call last):";
          } else {
            sout << "In user code:\n";
          }
          for (auto& line : *callstack) {
            sout << "\n  " << line;
          }
        }
        std::cout << sout.str() << std::endl;
        std::rethrow_exception(std::current_exception());
      }
    }

    if (execution_config.used_for_inference) {
      for (auto& hook : output_hookfuncs) {
        hook(op, local_scope);
      }
    }

    VLOG(4) << "End run " << place << " "
            << op_func_node.operator_base_->DebugStringEx(local_scope);

    if (!static_build) {
      // gc---------------------------------------------
      auto iter = unused_var_map.find(op);
      if (iter == unused_var_map.end()) {
        memory::LogDeviceMemoryStats(place, op_type);
        continue;
      }

      auto& delete_vars = iter->second;
      std::deque<std::shared_ptr<memory::Allocation>>* garbages =
          new std::deque<std::shared_ptr<memory::Allocation>>();

      for (auto& var_name : delete_vars) {
        auto* var = local_scope->FindVar(var_name);
        if (var == nullptr ||
            skip_gc_vars.find(var_name) != skip_gc_vars.end()) {
          continue;
        }

        VLOG(6) << "Erase variable " << var_name;
        if (var->IsType<phi::DenseTensor>()) {
          garbages->emplace_back(
              var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
        } else if (var->IsType<phi::SelectedRows>()) {
          garbages->emplace_back(var->GetMutable<phi::SelectedRows>()
                                     ->mutable_value()
                                     ->MoveMemoryHolder());
          var->GetMutable<phi::SelectedRows>()->mutable_rows()->clear();
        } else if (var->IsType<phi::TensorArray>()) {
          auto* tensor_arr = var->GetMutable<phi::TensorArray>();
          for (auto& t : *tensor_arr) {
            garbages->emplace_back(t.MoveMemoryHolder());
          }
        } else if (var->IsType<phi::SparseCooTensor>()) {
          garbages->emplace_back(var->GetMutable<phi::SparseCooTensor>()
                                     ->mutable_indices()
                                     ->MoveMemoryHolder());
          garbages->emplace_back(var->GetMutable<phi::SparseCooTensor>()
                                     ->mutable_values()
                                     ->MoveMemoryHolder());
        } else if (var->IsType<phi::SparseCsrTensor>()) {
          garbages->emplace_back(var->GetMutable<phi::SparseCsrTensor>()
                                     ->mutable_cols()
                                     ->MoveMemoryHolder());
          garbages->emplace_back(var->GetMutable<phi::SparseCsrTensor>()
                                     ->mutable_crows()
                                     ->MoveMemoryHolder());
          garbages->emplace_back(var->GetMutable<phi::SparseCsrTensor>()
                                     ->mutable_values()
                                     ->MoveMemoryHolder());
        }
      }
      delete garbages;  // free mem

      memory::LogDeviceMemoryStats(place, op_type);
    }
  }

  // in the unused_var_map, we did not record the variable who are created in
  // ApplyDataTransform. So we erase these variables here.
  std::deque<std::shared_ptr<memory::Allocation>>* garbages =
      new std::deque<std::shared_ptr<memory::Allocation>>();
  for (const auto& transferred_var : var_scope->DataTransferAddedVars()) {
    const auto& var_name = transferred_var.first;
    auto* var = local_scope->FindVar(var_name);
    if (var == nullptr) continue;
    VLOG(6) << "Erase variable " << var_name;
    if (var->IsType<phi::DenseTensor>()) {
      garbages->emplace_back(
          var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
    } else if (var->IsType<phi::SelectedRows>()) {
      garbages->emplace_back(var->GetMutable<phi::SelectedRows>()
                                 ->mutable_value()
                                 ->MoveMemoryHolder());
      var->GetMutable<phi::SelectedRows>()->mutable_rows()->clear();
    } else if (var->IsType<phi::TensorArray>()) {
      auto* tensor_arr = var->GetMutable<phi::TensorArray>();
      for (auto& t : *tensor_arr) {
        garbages->emplace_back(t.MoveMemoryHolder());
      }
    } else if (var->IsType<phi::SparseCooTensor>()) {
      garbages->emplace_back(var->GetMutable<phi::SparseCooTensor>()
                                 ->mutable_indices()
                                 ->MoveMemoryHolder());
      garbages->emplace_back(var->GetMutable<phi::SparseCooTensor>()
                                 ->mutable_values()
                                 ->MoveMemoryHolder());
    } else if (var->IsType<phi::SparseCsrTensor>()) {
      garbages->emplace_back(var->GetMutable<phi::SparseCsrTensor>()
                                 ->mutable_cols()
                                 ->MoveMemoryHolder());
      garbages->emplace_back(var->GetMutable<phi::SparseCsrTensor>()
                                 ->mutable_crows()
                                 ->MoveMemoryHolder());
      garbages->emplace_back(var->GetMutable<phi::SparseCsrTensor>()
                                 ->mutable_values()
                                 ->MoveMemoryHolder());
    }
  }
  delete garbages;
}

void BuildVariableScope(const framework::BlockDesc& block,
                        const ExecutionConfig& execution_config,
                        VariableScope* var_scope) {
  VLOG(3) << "Creating Variables";
  auto inner_scope = var_scope->GetMutableScope();

  // NOTE(zhiqiu): if create_local_scope_ is true, the persistable is
  // created in var_scope.scope_ , and other scope is created in local scope.
  Scope* local_scope = execution_config.create_local_scope
                           ? var_scope->GetMutableLocalScope()
                           : var_scope->GetMutableScope();

  for (auto& var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    // TODO(xiongkun): user may create a variable with name that exists before.
    // under such circumstances, we should raise a error. Currently we can't
    // get the var_desc of startup_program, so leave it later.
    if (var_name == framework::kEmptyVarName) {
      continue;
    }

    if (var_desc->Persistable() ||
        execution_config.force_root_scope_vars.count(var_name)) {
      // In principle, we should put all trainable parameters in global scope,
      // which means the root of the scope tree. Some cases like quantization
      // will look up these parameters in global scope.
      const Scope* ancestor_scope = inner_scope;
      while (ancestor_scope->parent()) {
        ancestor_scope = ancestor_scope->parent();
      }
      auto* ptr = const_cast<Scope*>(ancestor_scope)->Var(var_name);

      // NOTE(zhiqiu): if var exists in scope and the type is right,
      // InitializeVariable will not create a new variable.
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(3) << "Create Variable " << var_name << " global, which pointer is "
              << ptr << " type is " << static_cast<int>(var_desc->GetType());
    } else {
      auto* ptr = local_scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(3) << "Create Variable " << var_name << " locally, which pointer is "
              << ptr << " type is " << static_cast<int>(var_desc->GetType());
    }
    var_scope->AddVar(var_name, var_desc);
  }
}

void SetDeviceCommContext(framework::OperatorBase* operator_base,
                          phi::DeviceContext* dev_ctx) {
  if (operator_base->HasAttr("ring_id")) {
    int ring_id = operator_base->Attr<int>("ring_id");
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(std::to_string(ring_id))) {
      auto comm_context = comm_context_manager.Get(std::to_string(ring_id));
      if (!dev_ctx->GetCommContext()) {
        dev_ctx->SetCommContext(comm_context);
      }
    } else {
      VLOG(3) << "op: " << operator_base->Type() << ", ring_id: " << ring_id
              << ", get comm_context failed!";
    }
  }
}

void SetDeviceCommContext(pir::Operation* op, phi::DeviceContext* dev_ctx) {
  auto op_attributes = op->attributes();
  if (op_attributes.count("ring_id") != 0) {
    int ring_id =
        op_attributes.at("ring_id").dyn_cast<pir::Int32Attribute>().data();
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(std::to_string(ring_id))) {
      auto comm_context = comm_context_manager.Get(std::to_string(ring_id));
      if (!dev_ctx->GetCommContext()) {
        dev_ctx->SetCommContext(comm_context);
      }
    } else {
      VLOG(3) << "op: "
              << op_attributes.at("op_name")
                     .dyn_cast<pir::StrAttribute>()
                     .AsString()
              << ", ring_id: " << ring_id << ", get comm_context failed!";
    }
  }
}

std::unordered_set<std::string> GetSpecialOpNames() {
  return {
      "builtin.combine",
      "builtin.slice",
      "pd_op.feed",
      "builtin.set_parameter",
      "builtin.parameter",
      "builtin.constant",
      "pd_op.data",
      "builtin.shadow_output",
  };
}

void BuildId2VarName(const std::map<std::string, int>& var_name_2_id,
                     std::unordered_map<int, std::string>* id_2_var_name) {
  PADDLE_ENFORCE_NOT_NULL(
      id_2_var_name,
      common::errors::InvalidArgument("id2_var_name can not be null"));
  for (auto [var_name, id] : var_name_2_id) {
    id_2_var_name->insert({id, var_name});
  }
}

const paddle::framework::Variable* GetVariableByName(
    const std::string& var_name,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  for (auto kv : variable_2_var_name) {
    if (kv.second == var_name) {
      return kv.first;
    }
  }
  VLOG(8) << "cannot find var: " << var_name;
  return nullptr;
}

std::vector<std::string> GetOriginInputNames(const std::string& op_name) {
  std::vector<std::string> ret;
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  if (op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()) {
    paddle::dialect::OpYamlInfoParser yaml_parser(
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
            ->get_op_info_(op_name));
    ret = yaml_parser.InputNames();
  }
  return ret;
}

std::vector<std::string> GetOriginOutputNames(const std::string& op_name) {
  std::vector<std::string> ret;
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  if (op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()) {
    paddle::dialect::OpYamlInfoParser yaml_parser(
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
            ->get_op_info_(op_name));
    ret = yaml_parser.OutputNames();
  }
  return ret;
}

void PrintValuesAndVariables(
    const pir::Block& block,
    const std::unordered_map<pir::Value, std::string>& value_2_var_name,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  for (auto& op : block) {
    std::stringstream ss;
    VLOG(6) << "-----------------------------";
    op.Print(ss);
    VLOG(6) << ss.str();

    std::string op_name = op.name();
    if (op.attributes().count("op_name")) {
      op_name = op.attributes()
                    .at("op_name")
                    .dyn_cast<pir::StrAttribute>()
                    .AsString();
    }
    std::vector<std::string> origin_input_names = GetOriginInputNames(op_name);
    std::vector<std::string> origin_output_names =
        GetOriginOutputNames(op_name);

    // 1. output string
    VLOG(10) << "Generate output string ...";
    std::string ret_value_str = "Value   : (";
    std::string ret_variable_str = "Variable: (";
    if (!op.results().empty()) {
      for (size_t i = 0; i < op.num_results(); ++i) {
        pir::Value out_value = op.result(i);
        if (value_2_var_name.count(out_value)) {
          // get Variable by Value
          auto& var_name = value_2_var_name.at(out_value);
          const paddle::framework::Variable* out_variable =
              GetVariableByName(var_name, variable_2_var_name);

          // get origin name
          std::string origin_name;
          if (!origin_output_names.empty())
            origin_name = origin_output_names[i];
          else
            origin_name = var_name;

          // process info
          ss.str("");
          ss << out_value.impl();
          ret_value_str +=
              (std::string(origin_name.length(), ' ') + "[" + ss.str() + "]");
          ss.str("");
          if (out_variable) {
            ss << out_variable;
            ret_variable_str += (origin_name + "[" + ss.str() + "]");
          } else {
            ret_variable_str += (origin_name + "[NULL]");
          }
        } else {
          ret_value_str += "NULL";
          ret_variable_str += "NULL";
        }
        ret_value_str += ", ";
        ret_variable_str += ", ";
      }
      ret_value_str = ret_value_str.substr(0, ret_value_str.length() - 2);
      ret_variable_str =
          ret_variable_str.substr(0, ret_variable_str.length() - 2);
    }
    ret_value_str += ") = ";
    ret_variable_str += ") = ";

    // 2. op name
    VLOG(10) << "Generate op name ...";
    ret_value_str += op_name;
    ret_variable_str += op_name;

    // 3. input string
    VLOG(10) << "Generate input string ...";
    ret_value_str += "(";
    ret_variable_str += "(";
    if (!op.operands().empty()) {
      for (size_t i = 0; i < op.num_operands(); ++i) {
        ::pir::Value in_value = op.operand(i).source();
        if (value_2_var_name.count(in_value)) {
          // get Variable by Value
          auto& var_name = value_2_var_name.at(in_value);
          const paddle::framework::Variable* in_variable =
              GetVariableByName(var_name, variable_2_var_name);

          // get origin name
          std::string origin_name;
          if (!origin_input_names.empty())
            origin_name = origin_input_names[i];
          else
            origin_name = var_name;

          // process info
          ss.str("");
          ss << in_value.impl();
          ret_value_str +=
              (std::string(origin_name.length(), ' ') + "[" + ss.str() + "]");
          ss.str("");
          if (in_variable) {
            ss << in_variable;
            ret_variable_str += (origin_name + "[" + ss.str() + "]");
          } else {
            ret_variable_str += (origin_name + "[NULL]");
          }
        } else {
          ret_value_str += "NULL";
          ret_variable_str += "NULL";
        }
        ret_value_str += ", ";
        ret_variable_str += ", ";
      }
      ret_value_str = ret_value_str.substr(0, ret_value_str.length() - 2);
      ret_variable_str =
          ret_variable_str.substr(0, ret_variable_str.length() - 2);
    }
    ret_value_str += ")";
    ret_variable_str += ")";
    VLOG(6) << ret_value_str;
    VLOG(6) << ret_variable_str;
  }
  return;
}

const std::vector<std::string> GetInstructionCallStack(
    const std::string& type, const pir::AttributeMap& attrs) {
  std::vector<std::string> vec_str;
  if (attrs.count("sub_block") != 0) {
    return vec_str;
  }
  auto iter = attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
  if (iter != attrs.end()) {
    auto attr = iter->second;
    PADDLE_ENFORCE(
        attr.isa<pir::ArrayAttribute>(),
        common::errors::InvalidArgument(
            "%s: Callstack attributes of %s is not ArrayAttribute type", type));
    pir::ArrayAttribute array_attribute = attr.dyn_cast<pir::ArrayAttribute>();
    std::vector<pir::Attribute> vec_attr = array_attribute.AsVector();
    for (auto value : vec_attr) {
      PADDLE_ENFORCE(
          value.isa<pir::StrAttribute>(),
          common::errors::InvalidArgument(
              "%s: Callstack attributes of %s is not StrAttribute type", type));
      vec_str.emplace_back(value.dyn_cast<pir::StrAttribute>().AsString());
    }
  }
  return vec_str;
}

bool IsNoNeedBuffer(pir::Operation* op, pir::Value value) {
  paddle::dialect::OpYamlInfoInterface op_info_interface =
      op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
  std::unique_ptr<paddle::dialect::OpYamlInfoParser> info_parser(nullptr);
  if (op_info_interface) {
    info_parser = std::make_unique<paddle::dialect::OpYamlInfoParser>(
        op_info_interface.GetOpInfo(), paddle::dialect::IsLegacyOp(op->name()));
    auto& no_need_buffer_ids = info_parser->NoNeedBufferIds();
    for (auto no_need_buffer_id : no_need_buffer_ids) {
      if (value == op->operand_source(no_need_buffer_id)) {
        return true;
      }
    }
  }
  return false;
}

std::unordered_map<std::string, std::set<std::string>> GetNoNeedBufferValues(
    const std::unordered_map<std::string, std::shared_ptr<::pir::Program>>&
        type_to_ir_program) {
  std::unordered_map<std::string, std::set<std::string>> shadow_output_values;
  std::set<std::string> no_need_buffer_vars;

  for (auto& pair : type_to_ir_program) {
    std::shared_ptr<::pir::Program> program = pair.second;
    // Iterate over the block_args and data_op output, and if all ops in all
    // programs using this value are of the no_need_buffer type, then insert
    // this value into the no_need_buffer set.
    for (const auto& [name, value] : program->block()->kwargs()) {
      for (auto it = value.use_begin(); it != value.use_end(); ++it) {
        if (IsNoNeedBuffer(it->owner(), value)) {
          no_need_buffer_vars.insert(name);
        } else {
          no_need_buffer_vars.erase(name);
        }
      }
    }
    for (auto& op : *program->block()) {
      if (op.isa<paddle::dialect::DataOp>()) {
        pir::Value value = op.result(0);
        std::string name = op.attribute<pir::StrAttribute>("name").AsString();
        for (auto it = value.use_begin(); it != value.use_end(); ++it) {
          if (IsNoNeedBuffer(it->owner(), value)) {
            no_need_buffer_vars.insert(name);
          } else {
            no_need_buffer_vars.erase(name);
          }
        }
      }
      // Retrieve the value names of all shadow_output_op for each program.
      if (op.isa<pir::ShadowOutputOp>()) {
        shadow_output_values[pair.first].insert(
            op.attribute<pir::StrAttribute>("output_name").AsString());
      }
    }
  }

  for (auto& pair : shadow_output_values) {
    std::set<std::string>& value_set = pair.second;
    std::set<std::string> intersection;
    std::set_intersection(value_set.begin(),
                          value_set.end(),
                          no_need_buffer_vars.begin(),
                          no_need_buffer_vars.end(),
                          std::inserter(intersection, intersection.begin()));
    value_set = std::move(intersection);
  }

  return shadow_output_values;
}

}  // namespace paddle::framework::interpreter
