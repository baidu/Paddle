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

#include "paddle/fluid/framework/new_executor/interpreter/static_build.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/operators/controlflow/control_flow_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/phi/core/framework/reader.h"
#include "paddle/phi/core/operators/reader/buffered_reader.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

COMMON_DECLARE_bool(cache_inference_while_scope);

std::set<std::string> OperatorBasesMustRunInStaticBuild = {
    "create_double_buffer_reader", "create_py_reader"};

std::set<std::string> OpsHandledInStaticBuild = {"conditional_block",
                                                 "fused_rms_norm",
                                                 "fused_rms_norm_grad",
                                                 "read",
                                                 "while"};

std::set<std::string> OpsCanSkippedFakeAllocInStaticBuild = {
    "c_comm_init",
    "comm_init_all",
    "c_comm_init_multitrainer",
    "c_gen_bkcl_id",
    "c_gen_nccl_id",
    "sync_calc_stream",
    "c_sync_calc_stream",
    "sync_comm_stream",
    "c_sync_comm_stream",
    "c_wait_comm",
    "c_wait_compute",
    "create_double_buffer_reader",
    "create_py_reader",
    "depend",
    "fetch_v2",
    "print",
    "send_v2",
    "nop"};

std::set<std::string> StaticBuildBlackList = {
    "sparse_sparse_coo_tensor" /*: to handle sparse output*/,
    "distributed_fused_lamb_init"};

namespace paddle::framework::interpreter {

using InterpreterCore = framework::InterpreterCore;

static VarMetaInfo GetVarMetaInfo(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  phi::DataType dtype = phi::DataType::UNDEFINED;
  phi::Place place = phi::Place();
  if (var == nullptr) {
    return VarMetaInfo(name, dtype, place);
  }

  if (var->IsType<phi::DenseTensor>()) {
    const phi::DenseTensor& tensor = var->Get<phi::DenseTensor>();
    if (!UNLIKELY(!tensor.IsInitialized())) {
      dtype = tensor.dtype();
      place = tensor.place();
    }
  } else if (var->IsType<phi::SelectedRows>()) {
    auto tensor = var->Get<phi::SelectedRows>().value();
    if (!UNLIKELY(!tensor.IsInitialized())) {
      dtype = tensor.dtype();
      place = tensor.place();
    }
  }
  return VarMetaInfo(name, dtype, place);
}

std::vector<VarMetaInfo> GetVarsInfo(const Scope* scope,
                                     VariableNameMap var_map,
                                     const OperatorBase& op) {
  std::vector<VarMetaInfo> var_info;

  const std::unordered_set<std::string>* no_need_buffer_vars = nullptr;
  if (op.Info().NoNeedBufferVarsInferer()) {
    no_need_buffer_vars = &(op.Info().NoNeedBufferVarsInferer()(
        op.Inputs(), op.Outputs(), op.Attrs()));
    if (no_need_buffer_vars->empty()) no_need_buffer_vars = nullptr;
  }
  for (auto it = var_map.begin(); it != var_map.end();) {
    auto& var = *it;
    bool is_no_need_buffer_var =
        (no_need_buffer_vars && no_need_buffer_vars->count(var.first) > 0);
    std::string var_name;
    var_info.reserve(var_info.size() + var.second.size());
    for (size_t i = 0; i < var.second.size(); ++i) {
      auto var_name = var.second[i];
      if (scope && is_no_need_buffer_var) {
        var_info.emplace_back(GetVarMetaInfo(*scope, var_name));
      } else {
        var_info.emplace_back(var_name);
      }
    }
    ++it;
  }
  return var_info;
}

bool BlockCanBeStaticBuilt(const framework::BlockDesc& block) {
  // in_black_list = (kernelCode >> 5) & 1
  // is_operator_base = (kernelCode >> 4) & 1
  // is_custom_op = (kernelCode >> 3) & 1
  // use_mkldnn = (kernelCode >> 2) & 1
  // sub_block_can_not_static_build = (kernelCode >> 1) & 1
  using KernelCode = int8_t;
  std::set<std::pair<std::string, KernelCode>> invalid_ops;
  for (auto& op : block.AllOps()) {
    auto op_type = op->Type();
    if (OpsCanSkippedFakeAllocInStaticBuild.count(op_type) ||
        OpsHandledInStaticBuild.count(op_type)) {
      continue;
    }

    const framework::OpInfo& info = OpInfoMap::Instance().Get(op_type);
    auto op_base =
        info.Creator()(op_type, op->Inputs(), op->Outputs(), op->GetAttrMap());

    bool in_black_list = StaticBuildBlackList.count(op_type);
    bool is_operator_base =
        (dynamic_cast<framework::OperatorWithKernel*>(op_base) == nullptr);
    bool is_custom_op =
        egr::Controller::Instance().GetOpMetaInfoMap().count(op_type);
    bool use_mkldnn = false;
    if (op->HasAttr("use_mkldnn")) {
      Attribute attr = op->GetAttr("use_mkldnn");
      use_mkldnn = attr.index() == 1 ? PADDLE_GET_CONST(int, attr)
                                     : PADDLE_GET_CONST(bool, attr);
    }

    bool sub_block_can_not_static_build = false;
    if (op->HasAttr("sub_block")) {
      auto* sub_block =
          PADDLE_GET_CONST(framework::BlockDesc*, op->GetAttr("sub_block"));
      sub_block_can_not_static_build = !BlockCanBeStaticBuilt(*sub_block);
    }

    KernelCode kernel_code = static_cast<KernelCode>(
        (in_black_list << 5) + (is_operator_base << 4) + (is_custom_op << 3) +
        (use_mkldnn << 2) + (sub_block_can_not_static_build << 1));

    if (in_black_list || is_operator_base || is_custom_op || use_mkldnn ||
        sub_block_can_not_static_build) {
      invalid_ops.insert(std::make_pair(op_type, kernel_code));
    }
  }

  if (!invalid_ops.empty()) {
    std::stringstream ss;
    ss << "The following OPs are unable to static build:\n";
    for (auto& item : invalid_ops) {
      ss << item.first << " [in_black_list = " << (item.second >> 5 & 1)
         << ", is_operator_base = " << (item.second >> 4 & 1)
         << ", is_custom_op = " << (item.second >> 3 & 1)
         << ", use_mkldnn = " << (item.second >> 2 & 1)
         << ", sub_block_can_not_static_build = " << (item.second >> 1 & 1)
         << "]\n";
    }
    VLOG(1) << ss.str();
  }

  return invalid_ops.empty();
}

inline bool IsExtendedTensor(const phi::TensorBase& tensor) {
  return phi::RawTensor::classof(&tensor) || phi::Strings::classof(&tensor) ||
         phi::Vocab::classof(&tensor);
}

bool TensorShouldBeFakeInitialized(const OperatorBase& op,
                                   const std::string& parameter_name,
                                   const phi::TensorBase* tensor) {
  const std::string& op_type = op.Type();
  if (OpsCanSkippedFakeAllocInStaticBuild.count(op_type)) {
    return false;
  }

  if (op_type == "adam" || op_type == "adamw" || op_type == "merged_adam") {
    if (op.Attr<bool>("use_global_beta_pow") &&
        (parameter_name == "Beta1PowOut" || parameter_name == "Beta2PowOut")) {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if (op_type == "batch_norm" && parameter_name == "ReserveSpace") {
    if (dynamic_cast<const OperatorWithKernel*>(&op)->kernel_type()->place_ ==
        phi::CPUPlace()) {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if (op_type == "coalesce_tensor" && parameter_name == "Output") {
    VLOG(2) << "Skip fake initialization for: " << parameter_name;
    return false;
  }

  if (op_type == "dgc" && parameter_name == "k") {
    VLOG(2) << "Skip fake initialization for: " << parameter_name;
    return false;
  }

  if (op_type == "fused_bias_residual_layernorm" &&
      parameter_name == "residual_out") {
    if (op.HasInputs("residual")) {
      bool is_residual_empty = op.Input("residual") == kEmptyVarName;
      bool is_norm_weight_empty = op.Input("norm_weight") == kEmptyVarName;
      bool is_norm_bias_empty = op.Input("norm_bias") == kEmptyVarName;
      if (!is_residual_empty) {
        if (is_norm_weight_empty && is_norm_bias_empty) {
          VLOG(2) << "Skip fake initialization for: " << parameter_name;
          return false;
        }
      } else {
        VLOG(2) << "Skip fake initialization for: " << parameter_name;
        return false;
      }
    } else {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if (op_type == "fake_quantize_range_abs_max") {
    if (op.Attr<bool>("is_test") &&
        (parameter_name == "OutScale" || parameter_name == "OutScales")) {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if ((op_type == "flatten" || op_type == "flatten_contiguous_range") &&
      parameter_name == "XShape") {
    VLOG(2) << "Skip fake initialization for: " << parameter_name;
    return false;
  }

  if (op_type == "segment_pool" && parameter_name == "SummedIds") {
    return op.Attr<std::string>("pooltype") == "MEAN" &&
           dynamic_cast<const OperatorWithKernel*>(&op)
                   ->kernel_type()
                   ->place_ != phi::CPUPlace();
  }

  return tensor && !IsExtendedTensor(*tensor);
}

phi::TensorBase* GetTensorFormVar(framework::Variable* var) {
  if (var) {
    if (var->template IsType<phi::DenseTensor>()) {
      return var->template GetMutable<phi::DenseTensor>();
    } else if (var->template IsType<phi::SelectedRows>()) {
      return var->template GetMutable<phi::SelectedRows>();
    } else if (var->template IsType<phi::SparseCooTensor>()) {
      return var->template GetMutable<phi::SparseCooTensor>();
    } else if (var->template IsType<phi::TensorArray>()) {
      return var->template GetMutable<phi::TensorArray>();
    } else if (var->template IsType<phi::Strings>()) {
      return var->template GetMutable<phi::Strings>();
    } else if (var->template IsType<phi::Vocab>()) {
      return var->template GetMutable<phi::Vocab>();
    } else if (var->template IsType<phi::RawTensor>() ||
               !var->IsInitialized()) {
      return var->template GetMutable<phi::RawTensor>();
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported `%s` type when get tensor.",
          framework::ToTypeName(var->Type())));
    }
  } else {
    VLOG(4) << "Var is nullptr";
    return nullptr;
  }
}

template <class TensorType>
void FakeInitializeTensor(const phi::DeviceContext& dev_ctx,
                          const phi::Place& place,
                          const phi::DataType& dtype,
                          const phi::DataLayout& layout,
                          TensorType* tensor) {
  PADDLE_ENFORCE_NE(
      place.GetType(),
      phi::AllocationType::UNDEFINED,
      common::errors::InvalidArgument(
          "The place %s to fake initialize is not valid.", place));
  PADDLE_ENFORCE_NE(
      dtype,
      phi::DataType::UNDEFINED,
      common::errors::InvalidArgument(
          "The dtype %s to fake initialize is not valid.", dtype));
  PADDLE_ENFORCE_NE(
      layout,
      phi::DataLayout::UNDEFINED,
      common::errors::InvalidArgument(
          "The layout %s to fake initialize is not valid.", layout));
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      common::errors::InvalidArgument(
          "The tensor to fake initialize should not be null."));

  if (tensor->initialized() && place == tensor->place() &&
      dtype == tensor->dtype() && tensor->layout() == layout) {
    return;
  }

  // set place
  if (tensor->initialized()) {  // avoid overwriting valid data
    phi::DeviceContext* dev_ctx_for_copy = nullptr;
    if (place.GetType() != AllocationType::CPU) {
      dev_ctx_for_copy = phi::DeviceContextPool::Instance().Get(place);
    } else {
      dev_ctx_for_copy =
          phi::DeviceContextPool::Instance().Get(tensor->place());
    }
    phi::Copy(*dev_ctx_for_copy, *tensor, place, /*blocking=*/true, tensor);
  } else {
    if (place == phi::CPUPlace()) {
      dev_ctx.HostAlloc(tensor,
                        dtype,
                        /*requested_size=*/0,
                        /*fake_alloc=*/true);
    } else {
      PADDLE_ENFORCE_EQ(place,
                        dev_ctx.GetPlace(),
                        common::errors::Unavailable(
                            "The place %s for fack alloc is not equal to "
                            "the place %s of DeviceContext.",
                            place,
                            dev_ctx.GetPlace()));
      dev_ctx.Alloc(tensor,
                    dtype,
                    /*requested_size=*/0,
                    /*pinned=*/false,
                    /*fake_alloc=*/true);
    }
  }

  // set dtype and layout
  tensor->set_type(dtype);
  tensor->set_layout(layout);

  VLOG(4) << "Tensor " << tensor << " fake alloc with type = " << dtype
          << ", place = " << place << ", layout = " << layout;
}

void FakeInitializeTensorBase(const phi::DeviceContext& dev_ctx,
                              const phi::Place& place,
                              const phi::DataType& dtype,
                              const phi::DataLayout& layout,
                              phi::TensorBase* tensor) {
  if (phi::DenseTensor::classof(tensor)) {
    FakeInitializeTensor(
        dev_ctx, place, dtype, layout, dynamic_cast<phi::DenseTensor*>(tensor));
  } else if (phi::SelectedRows::classof(tensor)) {
    FakeInitializeTensor(dev_ctx,
                         place,
                         dtype,
                         layout,
                         dynamic_cast<phi::SelectedRows*>(tensor));
  } else if (phi::SparseCooTensor::classof(tensor)) {
    FakeInitializeTensor(dev_ctx,
                         place,
                         dtype,
                         layout,
                         dynamic_cast<phi::SparseCooTensor*>(tensor));
  } else if (phi::SparseCsrTensor::classof(tensor)) {
    FakeInitializeTensor(dev_ctx,
                         place,
                         dtype,
                         layout,
                         dynamic_cast<phi::SparseCsrTensor*>(tensor));
  } else if (phi::TensorArray::classof(tensor)) {
    FakeInitializeTensor(
        dev_ctx, place, dtype, layout, dynamic_cast<phi::TensorArray*>(tensor));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported `%s` type when fake initialize tensor.",
        tensor->type_info().name()));
  }
}

void RunConditionalBlockPreStaticBuild(const framework::Scope& scope,
                                       const phi::Place& dev_place,
                                       const OperatorBase& op) {
  auto* scope_var = scope.FindVar(op.Output("Scope"));
  PADDLE_ENFORCE_NOT_NULL(
      scope_var,
      common::errors::PreconditionNotMet(
          "Expect Scope variable to be set in conditional_block_op, but "
          "got a null Scope variable. Please set the Scope variable."));

  auto* scopes = scope_var->GetMutable<std::vector<framework::Scope*>>();
  scopes->resize(1);
  scopes->front() = &scope.NewScope();

  auto& cur_scope = *scopes->front();
#ifdef PADDLE_WITH_DNNL
  // Executor on being destroyed clears oneDNN cache and resets
  // registered model data layout. This is unwanted for nested
  // Executors (executors declared inside control ops)
  platform::DontClearMKLDNNCache(dev_place);
#endif
  auto* block = op.Attr<framework::BlockDesc*>("sub_block");
  VLOG(3) << "Conditional block.idx = " << block->ID()
          << ", scope = " << &cur_scope;

  auto& skip_vars =
      op.Attr<std::vector<std::string>>("skip_eager_deletion_vars");

  std::unique_ptr<InterpreterCore> core;
  LOG_FIRST_N(INFO, 1)
      << "[ControlFlow][ConditionalBlock] New Executor is Running.";

  VLOG(10) << "[interpreterCore cache]" << core.get();
  VLOG_IF(10, core) << phi::is_same_place(core->GetPlace(), dev_place);

  framework::interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_control_flow_op = true;
  execution_config.skip_gc_vars =
      std::set<std::string>(skip_vars.begin(), skip_vars.end());

  core.reset(
      new InterpreterCore(dev_place, *block, &cur_scope, execution_config));

  std::vector<paddle::framework::OpFuncNode> op_func_nodes;
  core->Build({}, &op_func_nodes);
}

void RunWhileBlockPreStaticBuild(const framework::Scope& scope,
                                 const phi::Place& dev_place,
                                 const OperatorBase& op) {
  PADDLE_ENFORCE_NOT_NULL(
      scope.FindVar(op.Input("Condition")),
      common::errors::NotFound("Input(Condition) of WhileOp is not found."));

#ifdef PADDLE_WITH_DNNL
  // Executor on being destroyed clears oneDNN cache and resets
  // registered model data layout. This is unwanted for nested
  // Executors (executors declared inside control ops)
  platform::DontClearMKLDNNCache(dev_place);
#endif
  auto* block = op.Attr<framework::BlockDesc*>("sub_block");

  // get device context from pool
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(dev_place);

  bool is_test = op.Attr<bool>("is_test");

  std::set<std::string> no_copy_var_names;
  if (!is_test) {
    // set all persistable parameters into no_copy_var_names.
    auto* global_block = block;

    while (global_block->ID() != 0) global_block = global_block->ParentBlock();
    auto all_vars = global_block->AllVars();
    std::for_each(all_vars.begin(),
                  all_vars.end(),
                  [&no_copy_var_names](framework::VarDesc* var) {
                    if (var->IsParameter())
                      no_copy_var_names.insert(var->Name());
                  });

    const std::vector<framework::OpDesc*>& all_ops = block->AllOps();
    for (const framework::OpDesc* item : all_ops) {
      const framework::VariableNameMap& input_var_names = item->Inputs();
      const framework::VariableNameMap& output_var_names = item->Outputs();
      for (auto& ipt : input_var_names) {
        for (const std::string& var_name : ipt.second) {
          if (operators::StrInVariableNameMap(var_name, output_var_names)) {
            no_copy_var_names.insert(var_name);
          }
        }
      }
    }
  }

  auto step_scopes = scope.FindVar(op.Output("StepScopes"))
                         ->GetMutable<std::vector<framework::Scope*>>();

  if (!step_scopes->empty()) {
    phi::DeviceContextPool::Instance().Get(dev_place)->Wait();
    for (auto& s : *step_scopes) {
      if (scope.HasKid(s)) {
        scope.DeleteScope(s);
      }
    }
    step_scopes->clear();
  }

  PADDLE_ENFORCE_EQ(step_scopes->size(),
                    0,
                    common::errors::PreconditionNotMet(
                        "The Output(StepScope) of WhileOp should be empty."));

  auto& skip_vars =
      op.Attr<std::vector<std::string>>("skip_eager_deletion_vars");

  // note(lvyongkang): The assign op in while loop may change the place of
  // variable. However, InterpreterCore fix the kernel of every ops during its
  // first run. A cpu tensor may become gpu tensor after first run. This will
  // lead to segmetation fault when it's used in a cpu kernel. Here we record
  // the place of every inputs and restore their place after
  // InterpreterCore.run().
  std::map<std::string, phi::Place> input_var_original_places;
  for (const auto& in_name : op.Inputs("X")) {
    framework::Variable* var = scope.FindVar(in_name);
    if (var == nullptr) {
      VLOG(4) << "[while op]"
              << "input not found:" << in_name;
    }

    if (var->Type() == framework::proto::VarType::DENSE_TENSOR) {
      input_var_original_places[in_name] =
          (var->Get<phi::DenseTensor>()).place();
    } else {
      VLOG(10) << "[while op]"
               << "skip backup input " << in_name << " type:"
               << phi::TransToPhiDataType(framework::ToVarType(var->Type()));
    }
  }

  LOG_FIRST_N(INFO, 1) << "[ControlFlow][WhileOp] New Executor is Running.";
  std::unique_ptr<InterpreterCore> core;

  framework::Scope placeholder;  // Don't care if it's valid, just for
                                 // initialize InterpreterCore
  framework::interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_control_flow_op = true;
  execution_config.skip_gc_vars =
      std::set<std::string>(skip_vars.begin(), skip_vars.end());

  core.reset(new framework::InterpreterCore(
      dev_place, *block, &placeholder, execution_config));

  if (!is_test) {
    auto& current_scope = scope.NewScope();
    step_scopes->push_back(&current_scope);

    std::vector<std::string> rename_vars;
    for (const std::string& input_var_name : op.Inputs("X")) {
      if (no_copy_var_names.find(input_var_name) == no_copy_var_names.end()) {
        std::string input_var_rename = input_var_name + "@TMP_COPY";
        framework::Variable* input_var = scope.FindVar(input_var_name);
        if (input_var->IsType<phi::DenseTensor>()) {
          rename_vars.push_back(input_var_rename);
          auto input_var_tensor = input_var->Get<phi::DenseTensor>();
          auto* rename_input_var_tensor = current_scope.Var(input_var_rename)
                                              ->GetMutable<phi::DenseTensor>();
          framework::TensorCopy(
              input_var_tensor, dev_place, rename_input_var_tensor);
          rename_input_var_tensor->set_lod(input_var_tensor.lod());
        }
      }
    }

    operators::BuildScopeForControlFlowOp(*core, *block, &current_scope);
    core->reset_scope(&current_scope);

    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    core->Build({}, &op_func_nodes);

    // restore inputs place
    for (const auto& n : input_var_original_places) {
      const std::string& in_name = n.first;
      const phi::Place& original_place = n.second;
      // input vars exist in `scope` not `current_scope`
      operators::TransferVariablePlace(
          &scope, in_name, original_place, dev_ctx);
    }

    for (auto& var_rename : rename_vars) {
      std::string input_var_name =
          var_rename.substr(0, var_rename.size() - strlen("@TMP_COPY"));
      current_scope.Rename(var_rename, input_var_name);
    }
  } else {
    framework::Scope* current_scope = nullptr;
    if (!FLAGS_cache_inference_while_scope) {
      current_scope = &(scope.NewScope());
      operators::BuildScopeForControlFlowOp(*core, *block, current_scope);
      core->reset_scope(current_scope);
    } else {
      auto cached_inference_scope = &(scope.NewScope());
      operators::BuildScopeForControlFlowOp(
          *core, *block, cached_inference_scope);
      core->reset_scope(cached_inference_scope);
      current_scope = cached_inference_scope;
    }

    for (auto& name : current_scope->LocalVarNames()) {
      auto* var = current_scope->Var(name);
      if (var->IsType<phi::DenseTensor>()) {
        // Clear all lod information for all lod_tensors.
        auto* t = var->GetMutable<phi::DenseTensor>();
        phi::LegacyLoD empty_lod;
        t->set_lod(empty_lod);
      } else if (var->IsType<phi::TensorArray>()) {
        // Clear elements of all tensor arrays.
        auto* t = var->GetMutable<phi::TensorArray>();
        t->clear();
      }
    }

    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    core->Build({}, &op_func_nodes);

    if (!FLAGS_cache_inference_while_scope) {
      scope.DeleteScope(current_scope);
    }
  }
}

void FakeInitializeOutputsForOperatorBase(
    const OperatorBase& op,
    const phi::Place& place,
    Scope* scope,
    std::vector<std::shared_ptr<OperatorBase>> following_ops) {
  const std::string& op_type = op.Type();
  if (OpsCanSkippedFakeAllocInStaticBuild.count(op_type)) {
    return;
  }

  phi::DeviceContext* dev_ctx = phi::DeviceContextPool::Instance().Get(place);

  if (op_type == "conditional_block" || op_type == "while") {
    // Note(sonder): skip fake init for conditional_block when there is no
    // op with kernel after it.
    bool skip_fake_init = true;
    std::unordered_set<std::string> following_input_vars;

    for (size_t i = 0; i < following_ops.size(); ++i) {
      if (dynamic_cast<framework::OperatorWithKernel*>(
              following_ops[i].get()) != nullptr) {
        VLOG(4) << "Find op with kernel after " << op_type << ": "
                << following_ops[i]->Type();
        skip_fake_init = false;
        auto input_vars_info = GetVarsInfo(
            scope, following_ops[i]->Inputs(), *following_ops[i].get());
        for (auto& input_var_info : input_vars_info) {
          following_input_vars.insert(input_var_info.name_);
        }
      }
    }

    if (skip_fake_init) {
      return;
    }

    const std::vector<VarMetaInfo> out_var_info_before_build =
        GetVarsInfo(scope, op.Outputs(), op);

    if (op_type == "conditional_block") {
      RunConditionalBlockPreStaticBuild(*scope, place, op);
    } else {
      RunWhileBlockPreStaticBuild(*scope, place, op);
    }

    const std::vector<VarMetaInfo> out_var_info_after_build =
        GetVarsInfo(scope, op.Outputs(), op);

    // Note(sonder): static_build is not supported if the output of
    // conditional_block is changed after static build.
    for (size_t i = 0; i < out_var_info_before_build.size(); ++i) {
      // static build is supported in case of the output's dtype/place
      // is changed but the following op is not use this output
      if (out_var_info_before_build[i] != out_var_info_after_build[i]) {
        auto var_name = out_var_info_before_build[i].name_;
        if (following_input_vars.count(var_name)) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "The output %s s' dtype/place of %s is "
              "changed after static build. Befer static build, the "
              "dtype is %s, place is %s. After static "
              "build, the dtype is %s, place is %s.",
              op_type,
              var_name,
              out_var_info_before_build[i].dtype_,
              out_var_info_before_build[i].place_,
              out_var_info_after_build[i].dtype_,
              out_var_info_after_build[i].place_));
        }
      }
    }
  } else if (op_type == "read") {
    const std::string& reader_name = op.Input("Reader");
    framework::ReaderHolder* reader =
        GET_DATA_SAFELY(scope->FindVar(reader_name), "Input", "Reader", "Read")
            .GetMutable<framework::ReaderHolder>();

    std::shared_ptr<operators::reader::BufferedReader> buffered_reader =
        std::dynamic_pointer_cast<operators::reader::BufferedReader>(
            reader->Get());
    phi::Place target_place =
        buffered_reader ? buffered_reader->GetPlace() : phi::CPUPlace();

    auto& outputs = op.Outputs("Out");
    auto& var_types = reader->VarTypes();
    PADDLE_ENFORCE_EQ(outputs.size(),
                      var_types.size(),
                      common::errors::Unavailable(
                          "The output size of read_op (%d) should equal "
                          "to the var_types size of ReaderHolder (%d).",
                          outputs.size(),
                          var_types.size()));

    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string& parameter_name = outputs[i];
      phi::TensorBase* out_tensor =
          GetTensorFormVar(scope->FindVar(parameter_name));
      if (TensorShouldBeFakeInitialized(op, parameter_name, out_tensor)) {
        phi::DataType dtype = phi::TransToPhiDataType(var_types[i]);
        FakeInitializeTensorBase(
            *dev_ctx, target_place, dtype, out_tensor->layout(), out_tensor);
      }
    }
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Can not static build for op: %s", op_type));
  }
}

phi::DataType GetInputDType(const RuntimeContext& runtime_ctx,
                            const std::string parameter_name) {
  phi::TensorBase* in_tensor =
      GetTensorFormVar(runtime_ctx.inputs.find(parameter_name)->second.at(0));
  return in_tensor->dtype();
}

bool InputExisted(const RuntimeContext& runtime_ctx,
                  const std::string& parameter_name) {
  auto it = runtime_ctx.inputs.find(parameter_name);
  if (it == runtime_ctx.inputs.end() || it->second.empty()) {
    return false;
  }
  return true;
}

phi::DataType InferDTypeFromAttr(const framework::OperatorBase& op,
                                 const RuntimeContext& runtime_ctx,
                                 const std::string& attr_name) {
  int dtype_attr = op.Attr<int>(attr_name);
  if (dtype_attr == -1) {  // -1 means the dtype is same as input
    return GetInputDType(runtime_ctx, "X");
  }
  return phi::TransToPhiDataType(dtype_attr);
}

phi::DataType InferMPDType(const RuntimeContext& runtime_ctx,
                           const std::string parameter_name) {
  phi::DataType in_dtype = GetInputDType(runtime_ctx, parameter_name);
  return (in_dtype == phi::DataType::BFLOAT16 ||
          in_dtype == phi::DataType::FLOAT16)
             ? phi::DataType::FLOAT32
             : in_dtype;
}

void FakeInitializeOutputsForFunctionKernel(
    const framework::OperatorBase& op,
    const phi::Kernel& phi_kernel,
    const phi::KernelSignature& kernel_sig,
    const RuntimeContext& runtime_ctx,
    const phi::DeviceContext& dev_ctx) {
  const std::string& op_type = op.Type();
  auto output_names = kernel_sig.output_names;
  auto output_defs = phi_kernel.args_def().output_defs();
  PADDLE_ENFORCE_EQ(output_names.size(),
                    output_defs.size(),
                    common::errors::InvalidArgument(
                        "The size of outputs_args names (%d) must be equal to "
                        "the size of kernel output_defs (%d).",
                        output_names.size(),
                        output_defs.size()));
  size_t start_idx = 0;
  for (size_t i = 0; i < output_names.size(); ++i) {
    const std::string& parameter_name = output_names[i];
    auto it = runtime_ctx.outputs.find(parameter_name);
    // Deal with the case that some outputs are not found or be NULL when run
    // the kernel. For example : the outputs of matmul_grad are dx and dy,
    // sometimes dx or dy may be NULL.
    if (it == runtime_ctx.outputs.end() || it->second.empty()) {
      VLOG(4) << "Output " << parameter_name << " not found";
      ++start_idx;
      continue;
    }
    auto& outs_vector = it->second;
    for (auto out_var : outs_vector) {
      phi::TensorBase* out_tensor = GetTensorFormVar(out_var);
      if (TensorShouldBeFakeInitialized(op, parameter_name, out_tensor)) {
        phi::TensorArgDef& tensor_arg_def = output_defs[i];

        // analyze place
        phi::Backend backend = tensor_arg_def.backend;
        if (backend == phi::Backend::UNDEFINED) {
          if (op_type == "adam" || op_type == "adamw" ||
              op_type == "merged_adam") {
            phi::TensorBase* beta1_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta1Pow")->second.at(0));
            phi::TensorBase* beta2_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta2Pow")->second.at(0));
            if (beta1_pow->place() == beta2_pow->place()) {
              backend = phi::TransToPhiBackend(beta1_pow->place());
            }
          } else if (op_type == "lamb") {
            phi::TensorBase* beta1_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta1Pow")->second.at(0));
            phi::TensorBase* beta2_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta2Pow")->second.at(0));
            if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU &&
                beta1_pow->place().GetType() == AllocationType::CPU &&
                beta2_pow->place().GetType() == AllocationType::CPU) {
              backend = phi::Backend::CPU;
            } else {
              backend = phi::TransToPhiBackend(dev_ctx.GetPlace());
            }
          } else if (op_type == "reshape2") {
            phi::TensorBase* x =
                GetTensorFormVar(runtime_ctx.inputs.find("X")->second.at(0));
            backend = phi::TransToPhiBackend(x->place());
          } else {
            PADDLE_THROW(common::errors::Unimplemented(
                "Unsupported UNDEFINED backend for op: %s, parameter: %s",
                op_type,
                parameter_name));
          }
        }
        phi::Place place = backend == phi::Backend::CUSTOM
                               ? dev_ctx.GetPlace()
                               : phi::TransToPhiPlace(backend);

        // analyze dtype
        phi::DataType dtype = tensor_arg_def.dtype;
        if (dtype == DataType::UNDEFINED) {
          // Some OP's InferMeta is sensitive to DDim, so we cannot get their
          // output dtype from InferMeta
          if (op_type == "adam" || op_type == "adamw") {
            dtype = InferMPDType(runtime_ctx, "Param");
          } else if (op_type == "arg_min" || op_type == "arg_max" ||
                     op_type == "coalesce_tensor" || op_type == "one_hot_v2" ||
                     op_type == "unique") {
            dtype = InferDTypeFromAttr(op, runtime_ctx, "dtype");
          } else if (op_type == "bincount" || op_type == "reduce_sum_grad") {
            dtype = GetInputDType(runtime_ctx, "X");
          } else if (op_type == "dequantize_linear") {
            dtype = GetInputDType(runtime_ctx, "Scale");
          } else if (op_type == "quantize_linear") {
            dtype = GetInputDType(runtime_ctx, "Scale");
          } else if (op_type == "lamb") {
            bool multi_precision = op.Attr<bool>("multi_precision");
            dtype = GetInputDType(runtime_ctx, "Moment1");
            if (multi_precision && dtype == phi::DataType::FLOAT16) {
              dtype = phi::DataType::FLOAT32;
            }
          } else if (op_type == "layer_norm") {
            dtype = InferMPDType(runtime_ctx, "X");
          } else if (op_type == "reduce_sum") {
            phi::DataType in_dtype = GetInputDType(runtime_ctx, "X");
            int dtype_attr = op.Attr<int>("out_dtype");
            if (dtype_attr != -1) {
              dtype = phi::TransToPhiDataType(dtype_attr);
              if (dtype == DataType::UNDEFINED) {
                dtype = in_dtype;
              }
            } else {
              dtype =
                  (in_dtype == DataType::BOOL || in_dtype == DataType::INT32)
                      ? DataType::INT64
                      : in_dtype;
            }
          } else if (op_type == "searchsorted") {
            bool out_int32 = op.Attr<bool>("out_int32");
            if (out_int32) {
              dtype = DataType::INT32;
            } else {
              dtype = DataType::INT64;
            }
          } else if (op_type == "fused_bias_residual_layernorm") {
            auto in_dtype = GetInputDType(runtime_ctx, "x");
            float quant_scale = op.Attr<float>("quant_scale");
            if (InputExisted(runtime_ctx, "residual") &&
                !InputExisted(runtime_ctx, "norm_weight") &&
                !InputExisted(runtime_ctx, "norm_bias")) {
              dtype = in_dtype;
            } else {
              if (quant_scale > 0.0f) {
                dtype = DataType::INT8;
              } else {
                dtype = in_dtype;
              }
            }
          } else {
            VLOG(4) << "Get dtype result from InferMeta";
            RuntimeInferShapeContext infer_shape_ctx(op, runtime_ctx);
            dynamic_cast<const framework::OperatorWithKernel*>(&op)
                ->Info()
                .infer_shape_(&infer_shape_ctx);
            dtype = out_tensor->dtype();  // dtype from InferMeta
          }
        }

        // analyze layout
        phi::DataLayout layout = tensor_arg_def.layout;
        FakeInitializeTensorBase(dev_ctx, place, dtype, layout, out_tensor);
      }
    }
    start_idx += outs_vector.size();
  }
}

void FakeInitializeOutputsForStructureKernel(
    const framework::OpKernelType& op_kernel_type,
    ExecutionContext* execution_context) {
  const framework::OperatorBase& op = execution_context->GetOp();
  if (OpsCanSkippedFakeAllocInStaticBuild.count(op.Type())) {
    return;
  }

  const VariableNameMap& outputs = op.Outputs();
  for (auto& item : outputs) {
    const std::string& output_parameter_name = item.first;
    auto multi_output_var =
        execution_context->MultiOutputVar(output_parameter_name);
    for (Variable* var : multi_output_var) {
      phi::TensorBase* out_tensor = GetTensorFormVar(var);
      if (TensorShouldBeFakeInitialized(
              op, output_parameter_name, out_tensor)) {
        phi::Place place = execution_context->GetPlace();
        phi::DataType dtype =
            phi::TransToPhiDataType(op_kernel_type.data_type_);
        // temporarily hack for extern op fused_rms_norm
        if (op.Type() == "fused_rms_norm") {
          if (output_parameter_name == "invvar") {
            dtype = DataType::FLOAT32;
          } else if (output_parameter_name == "y") {
            dtype = GetInputDType(execution_context->Context(), "x");
          }
          VLOG(4) << "Set fused_rms_norm output " << output_parameter_name
                  << " to " << dtype;
        }
        if (op.Type() == "fused_rms_norm_grad") {
          if (output_parameter_name == paddle::Grad("x")) {
            dtype = GetInputDType(execution_context->Context(), "x");
          } else if (output_parameter_name == paddle::Grad("scale")) {
            dtype = GetInputDType(execution_context->Context(), "scale");
          }
          VLOG(4) << "Set fused_rms_norm_grad output " << output_parameter_name
                  << " to " << dtype;
        }

        phi::DataLayout layout = out_tensor->layout();
        FakeInitializeTensorBase(execution_context->device_context(),
                                 place,
                                 dtype,
                                 layout,
                                 out_tensor);
      }
    }
  }
}

}  // namespace paddle::framework::interpreter
