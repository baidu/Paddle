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

#pragma once

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/feed_hook.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/pir/utils/name_analysis.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

COMMON_DECLARE_bool(enable_pir_with_pt_in_dy2st);
COMMON_DECLARE_bool(enable_pir_in_executor);
COMMON_DECLARE_bool(use_mkldnn);

namespace details {
using Tensor = paddle::Tensor;

static void Trans2ContiguousTensorsInplace(
    const std::vector<paddle::Tensor> &tensors) {
  std::vector<Tensor> res;
  for (auto &t : tensors) {
    if (t.initialized() && t.is_dense_tensor() &&
        !std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())
             ->meta()
             .is_contiguous()) {
      auto tmp = paddle::experimental::Trans2Contiguous(
          *(std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())));
      auto holder = tmp.MoveMemoryHolder();
      std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())->ResetHolder(
          holder);
      std::dynamic_pointer_cast<phi::DenseTensor>(t.impl())->set_meta(
          tmp.meta());
    }
  }
}

static std::vector<Tensor> DereferenceTensors(
    const std::vector<Tensor *> &tensor_ptr) {
  std::vector<Tensor> res;
  for (auto *t : tensor_ptr) {
    res.emplace_back(*t);
  }
  return res;
}

static std::vector<std::string> GetTensorsName(const std::vector<Tensor> &ins) {
  std::vector<std::string> in_names;
  for (auto &in_t : ins) {
    in_names.emplace_back(in_t.name());
  }
  return in_names;
}

static std::vector<std::string> GetTensorsName(
    const std::vector<Tensor *> &ins) {
  std::vector<std::string> in_names;
  for (auto *in_t : ins) {
    in_names.emplace_back(in_t->name());
  }
  return in_names;
}

static bool IsVariableRefArray(const Tensor &tensor) {
  return paddle::framework::VariableRefArray::classof(tensor.impl().get());
}

static auto GetNameFromValue(const std::vector<::pir::Value> &values) {
  std::vector<std::string> names;
  std::transform(
      values.begin(),
      values.end(),
      std::back_inserter(names),
      [](const ::pir::Value &v) {
        return pir::utils::name_analysis::TryGetValueFirstName(v).value_or(
            std::string(paddle::framework::kFakeVarName));
      });
  return names;
}

static void CheckInputVarStatus(const Tensor &tensor) {
  PADDLE_ENFORCE_EQ(tensor.defined() &&
                        (tensor.is_dense_tensor() ||
                         IsVariableRefArray(tensor) || tensor.is_dist_tensor()),
                    true,
                    common::errors::InvalidArgument(
                        "The input tensor %s of RunProgram(Grad)Op holds "
                        "wrong type. Expect type is DenseTensor or "
                        "VariableRefArray or DistTensor.",
                        tensor.name()));
}

static void CheckOutputVarStatus(const paddle::framework::Variable &src_var,
                                 const Tensor &dst_tensor) {
  auto name = dst_tensor.name();
  PADDLE_ENFORCE_EQ(dst_tensor.defined(),
                    true,
                    common::errors::InvalidArgument(
                        "dst_tensor `%s` shall be defined.", name));

  if (dst_tensor.is_dense_tensor() || dst_tensor.is_dist_tensor()) {
    auto &src_tensor = src_var.Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::DenseTensor::classof(&src_tensor),
                      true,
                      common::errors::InvalidArgument(
                          "The output tensor %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is DenseTensor",
                          name));
  } else if (dst_tensor.is_selected_rows()) {
    auto &src_tensor = src_var.Get<phi::SelectedRows>();
    PADDLE_ENFORCE_EQ(phi::SelectedRows::classof(&src_tensor),
                      true,
                      common::errors::InvalidArgument(
                          "The output tensor %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is SelectedRows",
                          name));
  } else if (IsVariableRefArray(dst_tensor)) {
    auto &src_tensor = src_var.Get<paddle::framework::VariableRefArray>();
    PADDLE_ENFORCE_EQ(paddle::framework::VariableRefArray::classof(&src_tensor),
                      true,
                      common::errors::InvalidArgument(
                          "The output tensor %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is VariableRefArray",
                          name));
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type DenseTensor, SelectedRows or VariableRefArray",
        name));
  }
}

static void ShareTensorsIntoScopeWithName(
    const std::vector<Tensor> &tensors,
    const std::vector<std::string> &tensor_names,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto name = tensor_names[i];
    VLOG(4) << "Share Tensor Into Scope: " << name;
    if (name == paddle::framework::kFakeVarName ||
        name == paddle::framework::kEmptyVarName) {
      continue;
    }
    auto *var = scope->Var(name);
    CheckInputVarStatus(tensors[i]);
    // share tensor
    auto tensor_base = tensors[i].impl();
    if (phi::DenseTensor::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor_base);
      *dst_tensor = *t;
    } else if (phi::SelectedRows::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::SelectedRows>();
      auto t = std::dynamic_pointer_cast<phi::SelectedRows>(tensor_base);
      *dst_tensor = *t;
    } else if (paddle::framework::VariableRefArray::classof(
                   tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<paddle::framework::VariableRefArray>();
      auto t = std::dynamic_pointer_cast<paddle::framework::VariableRefArray>(
          tensor_base);
      *dst_tensor = *t;
    } else if (phi::distributed::DistTensor::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t =
          std::dynamic_pointer_cast<phi::distributed::DistTensor>(tensor_base);
      *dst_tensor = t->value();
    }
  }
}

static void ShareTensorsIntoScope(const std::vector<Tensor> &tensors,
                                  paddle::framework::Scope *scope) {
  const std::vector<std::string> names =
      [&](const std::vector<Tensor> &tensors) {
        std::vector<std::string> names;
        for (auto &t : tensors) {
          names.push_back(t.name());
        }
        return names;
      }(tensors);

  ShareTensorsIntoScopeWithName(tensors, names, scope);
}

static void ShareTensorsIntoScopeByValue(
    const std::vector<Tensor> &tensors,
    const std::vector<::pir::Value> &values,
    paddle::framework::Scope *scope) {
  auto names = GetNameFromValue(values);
  ShareTensorsIntoScopeWithName(tensors, names, scope);
}

static void ShareTensorsFromScopeByValue(
    const std::vector<Tensor *> &tensors,
    const std::vector<::pir::Value> &values,
    paddle::framework::Scope *scope) {
  auto names = GetNameFromValue(values);
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &name = names[i];
    auto &value = values[i];
    VLOG(4) << "Share Tensor From Scope: " << name;

    if (value.impl() == nullptr) {
      // skip stop_gradient.
      continue;
    }
    auto *var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound("The output tensor %s is not in "
                                 "RunProgram(Grad)Op'"
                                 "s internal scope.",
                                 name));
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    if (var->IsType<phi::DenseTensor>()) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      if (tensors[i]->is_dist_tensor()) {
        auto *dst_tensor =
            std::dynamic_pointer_cast<phi::distributed::DistTensor>(
                tensors[i]->impl())
                ->unsafe_mutable_value();
        VLOG(2) << "actually do sharing " << name << " from scope";
        *dst_tensor = src_tensor;
      } else {
        auto *dst_tensor = const_cast<phi::DenseTensor *>(
            dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
        VLOG(2) << "actually do sharing " << name << " from scope";
        *dst_tensor = src_tensor;
      }
    } else if (var->IsType<phi::SelectedRows>()) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    } else if (var->IsType<paddle::framework::VariableRefArray>()) {
      auto &src_tensor = var->Get<paddle::framework::VariableRefArray>();
      auto *dst_tensor = const_cast<paddle::framework::VariableRefArray *>(
          dynamic_cast<const paddle::framework::VariableRefArray *>(
              tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The RunProgram(Grad)Op only support output "
          "variable of type DenseTensor, SelectedRows or VariableRefArray",
          name));
    }
  }
}

static void ShareTensorsFromScopeWithPartialBlock(
    const std::vector<Tensor *> &tensors,
    const paddle::framework::BlockDesc &forward_global_block,
    const paddle::framework::BlockDesc *backward_global_block,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &name = tensors[i]->name();
    auto *var = scope->FindVar(name);
    if (name == paddle::framework::kEmptyVarName ||
        name == paddle::framework::kFakeVarName || var == nullptr) {
      VLOG(2) << "find tensor name is " << name << ", skip it!";
      continue;
    }
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    if (var->IsType<phi::DenseTensor>()) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
      VLOG(2) << "share " << name << " from scope";
      *dst_tensor = src_tensor;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    } else if (var->IsType<paddle::framework::VariableRefArray>()) {
      auto &src_tensor = var->Get<paddle::framework::VariableRefArray>();
      auto *dst_tensor = const_cast<paddle::framework::VariableRefArray *>(
          dynamic_cast<const paddle::framework::VariableRefArray *>(
              tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The RunProgram(Grad)Op only support output "
          "variable of type DenseTensor, SelectedRows or VariableRefArray",
          name));
    }
  }
}

static void BuildScopeByBlock(
    const paddle::framework::InterpreterCore &interpreter_core,
    const paddle::framework::BlockDesc &block,
    paddle::framework::Scope *scope) {
  for (auto &var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    if (var_name == paddle::framework::kEmptyVarName) {
      continue;
    }
    if (!scope->FindLocalVar(var_name)) {
      auto *ptr = scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(2) << "Initialize Block Variable " << var_name;
    }
  }
  auto &data_transfer_added_vars =
      interpreter_core.GetVariableScope()->DataTransferAddedVars();
  for (size_t i = 0; i < data_transfer_added_vars.size(); i++) {
    auto *ptr = scope->Var(data_transfer_added_vars[i].first);
    InitializeVariable(ptr,
                       static_cast<paddle::framework::proto::VarType::Type>(
                           data_transfer_added_vars[i].second));
    VLOG(2) << "Initialize Transfer Added Variable "
            << data_transfer_added_vars[i].first;
  }
}

static void GcScope(paddle::framework::Scope *scope) {
  std::deque<std::shared_ptr<paddle::memory::Allocation>> *garbages =
      new std::deque<std::shared_ptr<paddle::memory::Allocation>>();

  for (auto &var : scope->LocalVars()) {
    if (var != nullptr) {
      if (var->IsType<phi::DenseTensor>()) {
        garbages->emplace_back(
            var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
      }
      if (var->IsType<phi::SelectedRows>()) {
        garbages->emplace_back(var->GetMutable<phi::SelectedRows>()
                                   ->mutable_value()
                                   ->MoveMemoryHolder());
      }
      if (var->IsType<phi::TensorArray>()) {
        auto *lod_tensor_arr = var->GetMutable<phi::TensorArray>();
        for (auto &t : *lod_tensor_arr) {
          garbages->emplace_back(t.MoveMemoryHolder());
        }
        lod_tensor_arr->clear();
      }
    }
  }
  delete garbages;  // free mem
}

template <class T>
void print_collection(const T &t) {
  VLOG(5) << "Print collection start :";
  for (auto s : t) {
    VLOG(5) << s;
  }
  VLOG(5) << "Print collection end.";
}

}  // namespace details

inline void PirRunProgramAPI(
    const std::vector<paddle::Tensor> &x,
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::Tensor *> &out,                   // NOLINT
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    bool require_any_grad,
    const paddle::framework::AttributeMap &attrs,
    const int64_t &place_hash_key) {
  VLOG(2) << "RunProgramOpKernel Compute";
  // In the original run_program OP, the default value of the is_test
  // attribute is false, we should check if there is is_test parameter
  // in attrs
  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  int64_t program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));
  bool in_sot_mode = false;
  if (attrs.count("in_sot_mode")) {
    in_sot_mode = PADDLE_GET_CONST(bool, attrs.at("in_sot_mode"));
  }
  auto place = egr::Controller::Instance().GetExpectedPlace();

  // NOTE(chenweihang): In order not to add new variable type, use vector
  // here. Originally, here can use scope directly.
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      common::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  VLOG(2) << "RunProgramOp use interpretercore to execute program.";

  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

  VLOG(4) << "global_inner_scope:" << global_inner_scope;

  auto input_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fx"));
  auto output_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fo"));
  auto middle_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fm"));
  auto param_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("fp"));

  std::shared_ptr<::pir::Program> forward_program = PADDLE_GET_CONST(
      std::shared_ptr<::pir::Program>, attrs.at("forward_program"));
  std::shared_ptr<::pir::Program> backward_program = PADDLE_GET_CONST(
      std::shared_ptr<::pir::Program>, attrs.at("backward_program"));

  VLOG(10) << is_test << program_id;

  auto &cache = paddle::framework::InterpreterCoreInfoCache::Instance();
  std::shared_ptr<paddle::framework::InterpreterCore> interpreter_core =
      nullptr;
  if (!cache.Has(program_id,
                 global_inner_scope,
                 place_hash_key,
                 /*is_grad=*/false,
                 /*in_pir_mode=*/true)) {
    phi::RecordEvent record_event(
        "create_new_interpretercore", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "No interpretercore cache, so create a new interpretercore "
               "for program: "
            << program_id;

    // Step 1. Get no need buffer vars for inplace pass and gc
    auto no_need_buffer_values = PADDLE_GET_CONST(std::vector<::pir::Value>,
                                                  attrs.at("no_need_buffers"));
    const auto no_need_buffer_names =
        details::GetNameFromValue(no_need_buffer_values);
    const auto no_need_buffer_name_set = std::set<std::string>(
        no_need_buffer_names.begin(), no_need_buffer_names.end());
    // Step 2. share input_vars & parameters into scope
    details::ShareTensorsIntoScopeByValue(x, input_values, global_inner_scope);
    details::ShareTensorsIntoScopeByValue(
        params, param_values, global_inner_scope);
    // Step 3. create new interpretercore
    auto passed_kernel_program = paddle::framework::ApplyIrPass(
        forward_program.get(), place, no_need_buffer_name_set);
    interpreter_core = paddle::framework::CreatePirInterpreterCoreInfoToCache(
        std::move(passed_kernel_program),
        place,
        /*is_grad=*/false,
        program_id,
        global_inner_scope,
        place_hash_key,
        in_sot_mode);
    // Step 4. get all eager gc vars (skip_names = backward_inputs -
    // no_need_buffers + outputs)
    std::vector<std::string> skip_names;
    // update interpretercore skip_gc_var
    for (auto &kwarg : backward_program->block()->kwargs()) {
      skip_names.push_back(kwarg.first);
    }
    auto skip_names_set =
        std::set<std::string>(skip_names.begin(), skip_names.end());
    for (auto &name : no_need_buffer_names) {
      VLOG(4) << "Find no need buffer vars with name:" << name;
      skip_names_set.erase(name);
    }
    skip_names = details::GetNameFromValue(output_values);
    skip_names_set.insert(skip_names.begin(), skip_names.end());

    details::print_collection(skip_names_set);
    interpreter_core->SetSkipGcVars(skip_names_set);

    // std::set<std::string> input_vars;
    // input_vars.insert(input_names.begin(), input_names.end());
    // interpreter_core->SetJitInputVars(input_vars);

    // cache.UpdateSkipEagerDeleteVars(
    // program_id, global_inner_scope, false, skip_eager_delete_vars);
  } else {
    phi::RecordEvent record_event(
        "get_interpretercore_cache", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "Get interpretercore cache by program:" << program_id;
    // Step 1. get cache interpretercore
    auto &cached_value = cache.GetMutable(program_id,
                                          global_inner_scope,
                                          place_hash_key,
                                          /*is_grad=*/false,
                                          /*in_pir_mode=*/true);
    interpreter_core = cached_value.core_;
    // Step 2. update scope for cache interpretercore
    details::ShareTensorsIntoScopeByValue(x, input_values, global_inner_scope);
    details::ShareTensorsIntoScopeByValue(
        params, param_values, global_inner_scope);
    // TODO(xiongkun): new ir how to build scope.
    // if (interpreter_core->GetVariableScope()->GetMutableScope() !=
    // global_inner_scope) {
    // details::BuildScopeByBlock(
    // *interpreter_core.get(), *forward_global_block, global_inner_scope);
    // interpreter_core->reset_scope(global_inner_scope);
    //}
  }

  paddle::framework::RunFeedHooks(*forward_program, *global_inner_scope);
  // interpretercore run
  if (!forward_program->block()->empty()) {
    phi::RecordEvent record_event(
        "interpreter_core_run", phi::TracerEventType::UserDefined, 1);
    interpreter_core->Run({});
  }

  {
    phi::RecordEvent record_event(
        "fetch_and_gc", phi::TracerEventType::UserDefined, 1);
    // Get Output, and Middle Outputs
    details::ShareTensorsFromScopeByValue(
        out, output_values, global_inner_scope);

    VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());

    if (is_test || !require_any_grad) {
      VLOG(4) << "don't require any grad, set this scope can reused";
      VLOG(4) << "is_test: " << is_test
              << ", require_any_grad: " << require_any_grad;
      global_inner_scope->SetCanReused(true);
      details::GcScope(global_inner_scope);
    } else {
      VLOG(4) << "not test, set this scope can not reused";
      global_inner_scope->SetCanReused(false);
    }
  }

#ifdef PADDLE_WITH_DNNL
  if (FLAGS_use_mkldnn) paddle::platform::DontClearMKLDNNCache(place);
#endif
}

inline void RunProgramAPI(
    const std::vector<paddle::Tensor> &x,
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::Tensor *> &out,                   // NOLINT
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    bool require_any_grad,
    const paddle::framework::AttributeMap &attrs,
    const int64_t &place_hash_key) {
  VLOG(2) << "RunProgramOpKernel Compute";
  // In the original run_program OP, the default value of the is_test
  // attribute is false, we should check if there is is_test parameter
  // in attrs
  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  auto need_grad = !is_test && require_any_grad;
  int64_t program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));
  auto place = egr::Controller::Instance().GetExpectedPlace();

  bool in_pir_pt_mode = FLAGS_enable_pir_with_pt_in_dy2st;
  if (attrs.count("in_pir_pt_mode")) {
    in_pir_pt_mode = PADDLE_GET_CONST(bool, attrs.at("in_pir_pt_mode"));
  }
  in_pir_pt_mode = in_pir_pt_mode || FLAGS_enable_pir_in_executor;

  // NOTE(chenweihang): In order not to add new variable type, use vector
  // here. Originally, here can use scope directly.
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      common::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  VLOG(2) << "RunProgramOp use interpretercore to execute program.";

  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

  VLOG(4) << "global_inner_scope:" << global_inner_scope;

  auto input_names =
      PADDLE_GET_CONST(std::vector<std::string>, attrs.at("x_names"));
  auto output_names = details::GetTensorsName(out);
  auto param_names = details::GetTensorsName(params);

  if (VLOG_IS_ON(6)) {
    std::stringstream s;
    s << "input_names: ";
    for (auto name : input_names) {
      s << name << " ";
    }
    s << std::endl;
    s << "param_names: ";
    for (auto name : param_names) {
      s << name << " ";
    }
    s << std::endl;
    s << "output_names: ";
    for (auto name : output_names) {
      s << name << " ";
    }
    s << std::endl;
    VLOG(6) << s.str();
  }

  auto *forward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("forward_global_block"));
  auto *forward_program = forward_global_block->Program();

  paddle::framework::BlockDesc *backward_global_block = nullptr;
  paddle::framework::ProgramDesc *backward_program = nullptr;

  if (need_grad) {
    backward_global_block = PADDLE_GET_CONST(paddle::framework::BlockDesc *,
                                             attrs.at("backward_global_block"));
    backward_program = backward_global_block->Program();
  }

  auto &cache = paddle::framework::InterpreterCoreInfoCache::Instance();
  std::shared_ptr<paddle::framework::InterpreterCore> interpreter_core =
      nullptr;
  if (!cache.Has(program_id,
                 global_inner_scope,
                 place_hash_key,
                 /*is_grad=*/false,
                 /*in_pir_mode=*/in_pir_pt_mode)) {
    phi::RecordEvent record_event(
        "create_new_interpretercore", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "No interpretercore cache, so create a new interpretercore "
               "for program: "
            << program_id;
    // Step 1. share input_vars & parameters into scope
    details::ShareTensorsIntoScopeWithName(x, input_names, global_inner_scope);
    details::ShareTensorsIntoScope(params, global_inner_scope);
    // Step 2. create new interpretercore

    if (in_pir_pt_mode) {
      // build new ir program
      auto ir_program =
          paddle::framework::ConstructForwardIrProgram(forward_global_block,
                                                       backward_global_block,
                                                       output_names,
                                                       x,
                                                       input_names,
                                                       params,
                                                       place);
      interpreter_core = paddle::framework::CreatePirInterpreterCoreInfoToCache(
          std::move(ir_program),
          place,
          /*is_grad=*/false,
          program_id,
          global_inner_scope,
          place_hash_key,
          /*used_for_sot=*/false);  // Simply pass false in PT mode
    } else {
      interpreter_core =
          paddle::framework::CreateProgramInterpreterCoreInfoToCache(
              *forward_program,
              place,
              /*is_grad=*/false,
              program_id,
              global_inner_scope,
              place_hash_key);
    }
    // Step 3. get all eager gc vars
    std::set<std::string> skip_eager_delete_vars;
    if (need_grad) {
      skip_eager_delete_vars =
          paddle::framework::details::ParseSafeEagerDeletionSkipVarsSet(
              *backward_program);
    }

    // all out_vars are skip_eager_var
    skip_eager_delete_vars.insert(output_names.begin(), output_names.end());
    // update interpretercore skip_gc_var
    details::print_collection(skip_eager_delete_vars);
    interpreter_core->SetSkipGcVars(skip_eager_delete_vars);

    std::set<std::string> input_vars;
    input_vars.insert(input_names.begin(), input_names.end());
    interpreter_core->SetJitInputVars(input_vars);

    if (VLOG_IS_ON(6)) {
      std::stringstream s;
      s << "skip_eager_delete_vars: ";
      for (auto name : skip_eager_delete_vars) {
        s << name << " ";
      }
      VLOG(6) << s.str();
    }

    cache.UpdateSkipEagerDeleteVars(program_id,
                                    global_inner_scope,
                                    place_hash_key,
                                    false,
                                    in_pir_pt_mode,
                                    skip_eager_delete_vars);
    VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
  } else {
    phi::RecordEvent record_event(
        "get_interpretercore_cache", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "Get interpretercore cache by program:" << program_id;
    // Step 1. get cache interpretercore
    auto &cached_value = cache.GetMutable(program_id,
                                          global_inner_scope,
                                          place_hash_key,
                                          /*is_grad=*/false,
                                          /*in_pir_mode=*/in_pir_pt_mode);
    interpreter_core = cached_value.core_;
    // Step 2. update scope for cache interpretercore
    details::ShareTensorsIntoScopeWithName(x, input_names, global_inner_scope);
    details::ShareTensorsIntoScope(params, global_inner_scope);
    if (interpreter_core->GetVariableScope()->GetMutableScope() !=
        global_inner_scope) {
      details::BuildScopeByBlock(
          *interpreter_core.get(), *forward_global_block, global_inner_scope);
      interpreter_core->reset_scope(global_inner_scope);
    }
  }

  // interpretercore run
  if (forward_global_block->OpSize() > 0) {
    phi::RecordEvent record_event(
        "interpreter_core_run", phi::TracerEventType::UserDefined, 1);
    interpreter_core->Run({});
  }
  VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());
  {
    phi::RecordEvent record_event(
        "fetch_and_gc", phi::TracerEventType::UserDefined, 1);
    // Get Output
    details::ShareTensorsFromScopeWithPartialBlock(
        out, *forward_global_block, backward_global_block, global_inner_scope);

    if (!need_grad) {
      VLOG(4) << "don't require any grad, set this scope can reused";
      VLOG(4) << "is_test: " << is_test
              << ", require_any_grad: " << require_any_grad;
      global_inner_scope->SetCanReused(true);
      details::GcScope(global_inner_scope);
    } else {
      VLOG(4) << "not test, set this scope can not reused";
      global_inner_scope->SetCanReused(false);
    }
  }

#ifdef PADDLE_WITH_DNNL
  if (FLAGS_use_mkldnn) paddle::platform::DontClearMKLDNNCache(place);
#endif
}

inline void RunProgramGradAPI(
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    const paddle::framework::AttributeMap &attrs,
    std::vector<paddle::Tensor *> &x_grad,       // NOLINT
    std::vector<paddle::Tensor *> &params_grad,  // NOLINT
    const int64_t &place_hash_key) {
  // if all output vars are set to stop_gradient, grad op no need to executed
  if (x_grad.empty() && params_grad.empty()) return;
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      common::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));
  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

  int64_t program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));

  bool in_pir_pt_mode = FLAGS_enable_pir_with_pt_in_dy2st;
  if (attrs.count("in_pir_pt_mode")) {
    in_pir_pt_mode = PADDLE_GET_CONST(bool, attrs.at("in_pir_pt_mode"));
  }
  in_pir_pt_mode = in_pir_pt_mode || FLAGS_enable_pir_in_executor;

  auto place = egr::Controller::Instance().GetExpectedPlace();
  VLOG(2) << "RunProgramGradOp use interpretercore to execute program.";

  VLOG(4) << "global_inner_scope:" << global_inner_scope;

  auto *forward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("forward_global_block"));
  auto *backward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("backward_global_block"));
  auto *backward_program = backward_global_block->Program();
  details::Trans2ContiguousTensorsInplace(out_grad);

  auto out_grad_names = details::GetTensorsName(out_grad);
  auto &cache = paddle::framework::InterpreterCoreInfoCache::Instance();
  std::shared_ptr<paddle::framework::InterpreterCore> interpreter_core =
      nullptr;
  if (!cache.Has(program_id,
                 global_inner_scope,
                 place_hash_key,
                 /*is_grad=*/true,
                 /*in_pir_mode=*/in_pir_pt_mode)) {
    phi::RecordEvent record_event(
        "create_new_interpretercore", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "No interpretercore cache, so create a new interpretercore"
               "for program: "
            << program_id;
    details::ShareTensorsIntoScope(out_grad, global_inner_scope);

    if (in_pir_pt_mode) {
      auto res =
          paddle::framework::ConstructBackwardIrProgram(backward_global_block,
                                                        out_grad,
                                                        x_grad,
                                                        params_grad,
                                                        global_inner_scope,
                                                        place);

      interpreter_core = paddle::framework::CreatePirInterpreterCoreInfoToCache(
          std::move(res),
          place,
          /*is_grad=*/true,
          program_id,
          global_inner_scope,
          place_hash_key,
          /*used_for_sot=*/false);  // Simply pass false in PT mode
    } else {
      interpreter_core =
          paddle::framework::CreateProgramInterpreterCoreInfoToCache(
              *backward_program,
              place,
              /*is_grad=*/true,
              program_id,
              global_inner_scope,
              place_hash_key);
    }

    // share threadpool
    // NOTE(zhiqiu): this only works interpreter_core is executed strictly
    // after the related fwd_interpreter_core.
    if (cache.Has(program_id,
                  global_inner_scope,
                  place_hash_key,
                  /*is_grad=*/false,
                  /*in_pir_mode=*/in_pir_pt_mode)) {
      auto fwd_interpreter_core =
          cache
              .GetMutable(program_id,
                          global_inner_scope,
                          place_hash_key,
                          /*is_grad=*/false,
                          /*in_pir_mode=*/in_pir_pt_mode)
              .core_;
      interpreter_core->ShareWorkQueueFrom(fwd_interpreter_core);
      VLOG(4) << "Share workqueue from " << fwd_interpreter_core.get() << " to "
              << interpreter_core.get();
    }

    std::vector<std::string> x_grad_names;
    std::vector<std::string> param_grad_names;
    if (!x_grad.empty()) {
      x_grad_names = details::GetTensorsName(x_grad);
    }
    if (!params_grad.empty()) {
      param_grad_names = details::GetTensorsName(params_grad);
    }
    // get all eager gc vars
    std::set<std::string> skip_eager_delete_vars;
    // all out_vars are skip_eager_var
    skip_eager_delete_vars.insert(x_grad_names.begin(), x_grad_names.end());
    // initialize skip gc vars by forward_program and backward_program
    paddle::framework::details::AppendSkipDeletionVars(param_grad_names,
                                                       &skip_eager_delete_vars);
    interpreter_core->SetSkipGcVars(skip_eager_delete_vars);
    cache.UpdateSkipEagerDeleteVars(program_id,
                                    global_inner_scope,
                                    place_hash_key,
                                    /*is_grad=*/true,
                                    in_pir_pt_mode,
                                    skip_eager_delete_vars);
    VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
  } else {
    phi::RecordEvent record_event(
        "get_interpretercore_cache", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "Get interpretercore cache by program:" << program_id;
    auto &cached_value = cache.GetMutable(program_id,
                                          global_inner_scope,
                                          place_hash_key,
                                          /*is_grad=*/true,
                                          /*in_pir_mode=*/in_pir_pt_mode);
    interpreter_core = cached_value.core_;

    // update scope
    details::ShareTensorsIntoScope(out_grad, global_inner_scope);
    if (interpreter_core->GetVariableScope()->GetMutableScope() !=
        global_inner_scope) {
      details::BuildScopeByBlock(
          *interpreter_core.get(), *backward_global_block, global_inner_scope);
      interpreter_core->reset_scope(global_inner_scope);
    }
  }

  if (backward_global_block->OpSize() > 0) {
    phi::RecordEvent record_event(
        "interpreter_core_run", phi::TracerEventType::UserDefined, 1);
    // Debug info: scope info when run end
    VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());
    interpreter_core->Run({});
  }

  {
    phi::RecordEvent record_event(
        "fetch_and_gc", phi::TracerEventType::UserDefined, 1);
    // Step 4. get outputs
    details::ShareTensorsFromScopeWithPartialBlock(x_grad,
                                                   *forward_global_block,
                                                   backward_global_block,
                                                   global_inner_scope);
    details::ShareTensorsFromScopeWithPartialBlock(params_grad,
                                                   *forward_global_block,
                                                   backward_global_block,
                                                   global_inner_scope);
    VLOG(4) << "after backward gc all vars";
    global_inner_scope->SetCanReused(true);
    details::GcScope(global_inner_scope);
  }
}

inline void PirRunProgramGradAPI(
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    const paddle::framework::AttributeMap &attrs,
    std::vector<paddle::Tensor *> &x_grad,       // NOLINT
    std::vector<paddle::Tensor *> &params_grad,  // NOLINT
    const int64_t &place_hash_key) {
  // if all output vars are set to stop_gradient, grad op no need to executed
  if (x_grad.empty() && params_grad.empty()) return;
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      common::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));
  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

  int64_t program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));

  bool in_sot_mode = false;
  if (attrs.count("in_sot_mode")) {
    in_sot_mode = PADDLE_GET_CONST(bool, attrs.at("in_sot_mode"));
  }

  auto place = egr::Controller::Instance().GetExpectedPlace();
  VLOG(2) << "RunProgramGradOp use interpretercore to execute program.";

  VLOG(4) << "global_inner_scope:" << global_inner_scope;

  std::shared_ptr<::pir::Program> backward_program = PADDLE_GET_CONST(
      std::shared_ptr<::pir::Program>, attrs.at("backward_program"));

  auto output_grad_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bo_g"));
  auto forward_input_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bx"));
  auto forward_middle_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bm"));
  auto parameter_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bp"));
  auto forward_output_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bo"));
  auto x_grad_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bx_g"));
  auto p_grad_values =
      PADDLE_GET_CONST(std::vector<::pir::Value>, attrs.at("bp_g"));

  details::Trans2ContiguousTensorsInplace(out_grad);

  // share x, param, middles, output_grads, out into scope.
  details::ShareTensorsIntoScopeByValue(
      out_grad, output_grad_values, global_inner_scope);

  auto &cache = paddle::framework::InterpreterCoreInfoCache::Instance();
  std::shared_ptr<paddle::framework::InterpreterCore> interpreter_core =
      nullptr;
  if (!cache.Has(program_id,
                 global_inner_scope,
                 place_hash_key,
                 /*is_grad=*/true,
                 /*in_pir_mode=*/true)) {
    phi::RecordEvent record_event(
        "create_new_interpretercore", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "No interpretercore cache, so create a new interpretercore";
    // Step 1. share input_vars & parameters into scope
    auto passed_kernel_program =
        paddle::framework::ApplyIrPass(backward_program.get(), place, {});

    const auto &new_block = passed_kernel_program->block();
    passed_kernel_program = paddle::framework::ApplyRemoveShadowFeedPass(
        std::move(passed_kernel_program), new_block, place, global_inner_scope);

    interpreter_core = paddle::framework::CreatePirInterpreterCoreInfoToCache(
        std::move(passed_kernel_program),
        place,
        /*is_grad=*/true,
        program_id,
        global_inner_scope,
        place_hash_key,
        in_sot_mode);
    // share threadpool
    // NOTE(zhiqiu): this only works interpreter_core is executed strictly
    // after the related fwd_interpreter_core.
    if (cache.Has(program_id,
                  global_inner_scope,
                  place_hash_key,
                  /*is_grad=*/false,
                  /*in_pir_mode=*/true)) {
      auto fwd_interpreter_core = cache
                                      .GetMutable(program_id,
                                                  global_inner_scope,
                                                  place_hash_key,
                                                  /*is_grad=*/false,
                                                  /*in_pir_mode=*/true)
                                      .core_;
      interpreter_core->ShareWorkQueueFrom(fwd_interpreter_core);
      VLOG(4) << "Share workqueue from " << fwd_interpreter_core.get() << " to "
              << interpreter_core.get();
    }

    // get all eager gc vars
    std::set<std::string> skip_eager_delete_vars;
    auto skip_names = details::GetNameFromValue(x_grad_values);
    skip_eager_delete_vars.insert(skip_names.begin(), skip_names.end());
    skip_names = details::GetNameFromValue(p_grad_values);
    skip_eager_delete_vars.insert(skip_names.begin(), skip_names.end());
    interpreter_core->SetSkipGcVars(skip_eager_delete_vars);
    cache.UpdateSkipEagerDeleteVars(program_id,
                                    global_inner_scope,
                                    place_hash_key,
                                    /*is_grad=*/true,
                                    /*in_pir_mode=*/true,
                                    skip_eager_delete_vars);
    VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
    details::print_collection(skip_eager_delete_vars);
  } else {
    phi::RecordEvent record_event(
        "get_interpretercore_cache", phi::TracerEventType::UserDefined, 1);
    VLOG(2) << "Get interpretercore cache by program:" << program_id;
    auto &cached_value = cache.GetMutable(program_id,
                                          global_inner_scope,
                                          place_hash_key,
                                          /*is_grad=*/true,
                                          /*in_pir_mode=*/true);
    interpreter_core = cached_value.core_;

    if (interpreter_core->GetVariableScope()->GetMutableScope() !=
        global_inner_scope) {
      // update scope (TODO(xiongkun): do we need this??)
      // details::BuildScopeByBlock(
      // *interpreter_core.get(), *backward_global_block, global_inner_scope);
      interpreter_core->reset_scope(global_inner_scope);
    }
  }

  paddle::framework::RunFeedHooks(*backward_program, *global_inner_scope);
  if (!backward_program->block()->empty()) {
    phi::RecordEvent record_event(
        "interpreter_core_run", phi::TracerEventType::UserDefined, 1);
    // Debug info: scope info when run end
    VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());
    interpreter_core->Run({});
  }

  {
    phi::RecordEvent record_event(
        "fetch_and_gc", phi::TracerEventType::UserDefined, 1);
    // Step 4. get outputs
    details::ShareTensorsFromScopeByValue(
        x_grad, x_grad_values, global_inner_scope);
    details::ShareTensorsFromScopeByValue(
        params_grad, p_grad_values, global_inner_scope);
    VLOG(4) << "after backward gc all vars";
    global_inner_scope->SetCanReused(true);
    details::GcScope(global_inner_scope);
  }
}

class GradNodeRunProgram : public egr::GradNodeBase {
 public:
  GradNodeRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(4) << "GradNodeRunProgram";
  }

  ~GradNodeRunProgram() override {
    if (!(*executed_)) {
      auto *out_scope_vec = &step_scope_;
      VLOG(4) << "~GradNodeRunProgram: " << this;
      // Normally out_scope_vec.size() == 1. for safety, we add for-loop here.
      for (size_t i = 0; i < out_scope_vec->size(); ++i) {
        paddle::framework::Scope *global_inner_scope = out_scope_vec->at(i);
        global_inner_scope->SetCanReused(true);
        details::GcScope(global_inner_scope);
        VLOG(4) << "global_inner_scope SetCanReused";
      }
    }
  }
  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize> &grads,  // NOLINT
             bool create_graph UNUSED,
             bool is_new_grad UNUSED) override {
    VLOG(3) << "Running Eager Backward Node: GradNodeRunProgram";
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        hooked_grads = GradNodeRunProgram::ApplyGradientHooks(grads);
    PADDLE_ENFORCE_EQ(hooked_grads.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The hooked_grads.size() of RunProgramGradOp should "
                          "be equal to 1."));

    std::vector<paddle::Tensor> x_grad;
    std::vector<paddle::Tensor> params_grad;
    std::vector<paddle::Tensor *> x_grad_ptr;
    std::vector<paddle::Tensor *> params_grad_ptr;
    {
      phi::RecordEvent record_event(
          "construct_grad_tensor", phi::TracerEventType::UserDefined, 1);

      egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&hooked_grads[0],
                                                         this->InputMeta()[0]);
      VLOG(3) << "hooked_grads[0].size() : " << hooked_grads[0].size();
      ConstructXGradTensors(x_, &x_grad);
      ConstructParamGradTensors(params_, &params_grad);
      for (auto &i : x_grad) {
        x_grad_ptr.emplace_back(&i);
      }
      for (auto &i : params_grad) {
        if (i.defined()) {
          params_grad_ptr.emplace_back(&i);
        }
      }
    }

    auto out_grad_names =
        PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("out_grad_names"));
    PADDLE_ENFORCE_EQ(hooked_grads[0].size(),
                      out_grad_names.size(),
                      common::errors::InvalidArgument(
                          "The hooked_grads[0].size() and "
                          "out_grad_names.size() should be equal."));
    for (size_t i = 0; i < out_grad_names.size(); ++i) {
      hooked_grads[0][i].set_name(out_grad_names[i]);
    }
    RunProgramGradAPI(hooked_grads[0],
                      step_scope_,
                      attrs_,
                      x_grad_ptr,
                      params_grad_ptr,
                      place_hash_key_);
    VLOG(3) << "End Eager Backward Node: GradNodeRunProgram: Ptr " << this;

    *executed_ = true;
    egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&x_grad,
                                                        this->OutputMeta()[0]);
    egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&params_grad,
                                                        this->OutputMeta()[1]);
    return {x_grad, params_grad};
  }

  void ClearTensorWrappers() override {
    x_.clear();
    params_.clear();
    SetIsTensorWrappersCleared(true);
  }

  // SetAttrMap
  void SetAttrMap(const paddle::framework::AttributeMap &attrs) {
    attrs_ = attrs;
  }

  void SetFwdX(const std::vector<paddle::Tensor> &tensors) { x_ = tensors; }

  void SetFwdParams(const std::vector<paddle::Tensor> &tensors) {
    params_ = tensors;
  }

  void SetStepScope(const std::vector<paddle::framework::Scope *> &scopes) {
    step_scope_ = scopes;
  }

  void SetPlaceHashKey(const int64_t &place_hash_key) {
    place_hash_key_ = place_hash_key;
  }

 protected:
  void ConstructXGradTensors(const std::vector<paddle::Tensor> &x,
                             std::vector<paddle::Tensor> *x_grad) {
    auto x_grad_names =
        PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("x_grad_names"));
    PADDLE_ENFORCE_EQ(
        x.size(),
        x_grad_names.size(),
        common::errors::InvalidArgument(
            "The x.size() and x_grad_names.size() should be equal. "
            "But received x.size() = %d, x_grad_names.size() = %d",
            x.size(),
            x_grad_names.size()));

    // TODO(dev): Need an elegant way to determine information of grad_tensor,
    // such as: name, tensor type(DenseTensor or SelectedRows).
    for (size_t i = 0; i < x.size(); i++) {
      if (x[i].is_dense_tensor()) {
        x_grad->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (x[i].is_selected_rows()) {
        x_grad->emplace_back(std::make_shared<phi::SelectedRows>());
      }
      x_grad->back().set_name(x_grad_names[i]);
    }
  }

  void ConstructParamGradTensors(const std::vector<paddle::Tensor> &params,
                                 std::vector<paddle::Tensor> *param_grads) {
    auto param_grad_names = PADDLE_GET_CONST(std::vector<std::string>,
                                             attrs_.at("param_grad_names"));
    PADDLE_ENFORCE_EQ(params.size(),
                      param_grad_names.size(),
                      common::errors::InvalidArgument(
                          "The param.size() and "
                          "param_grad_names.size() should be equal."));

    for (size_t i = 0; i < params.size(); ++i) {
      auto &p = params[i];
      auto &p_grad = egr::EagerUtils::unsafe_autograd_meta(p)->Grad();
      // In eager mode, the number of param_grad should be the same as
      // param, so here an empty Tensor is added for the param with
      // stop_gradient=True
      if (!p_grad.defined()) {
        param_grads->emplace_back();
      } else if (p_grad.is_dense_tensor()) {
        param_grads->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (p_grad.is_selected_rows()) {
        param_grads->emplace_back(std::make_shared<phi::SelectedRows>());
      }
      param_grads->back().set_name(param_grad_names[i]);
    }
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GradNodeRunProgram>(new GradNodeRunProgram(*this));
    return copied_node;
  }

 private:
  // TensorWrappers
  std::vector<paddle::Tensor> x_;
  std::vector<paddle::Tensor> params_;
  std::vector<paddle::framework::Scope *> step_scope_;

  // Attribute Map
  paddle::framework::AttributeMap attrs_;

  int64_t place_hash_key_;

  // why use shared_ptr. because paddle.grad will copy GradNode, if
  // we use bool, the copied node have different executed states.
  std::shared_ptr<bool> executed_ = std::make_shared<bool>(false);
};

class PirGradNodeRunProgram : public egr::GradNodeBase {
 public:
  PirGradNodeRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}

  ~PirGradNodeRunProgram() override {
    if (!(*executed_)) {
      auto *out_scope_vec = &step_scope_;
      VLOG(4) << "~PirGradNodeRunProgram";
      // Normally out_scope_vec.size() == 1. for safety, we add for-loop here.
      for (size_t i = 0; i < out_scope_vec->size(); ++i) {
        paddle::framework::Scope *global_inner_scope = out_scope_vec->at(i);
        global_inner_scope->SetCanReused(true);
        details::GcScope(global_inner_scope);
        VLOG(4) << "global_inner_scope SetCanReused";
      }
    }
  }
  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize> &grads,  // NOLINT
             bool create_graph UNUSED,
             bool is_new_grad UNUSED) override {
    VLOG(3) << "Running Eager Backward Node: PirGradNodeRunProgram";
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        hooked_grads = PirGradNodeRunProgram::ApplyGradientHooks(grads);
    PADDLE_ENFORCE_EQ(hooked_grads.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The hooked_grads.size() of RunProgramGradOp should "
                          "be equal to 1."));

    std::vector<paddle::Tensor> x_grad;
    std::vector<paddle::Tensor> params_grad;
    std::vector<paddle::Tensor *> x_grad_ptr;
    std::vector<paddle::Tensor *> params_grad_ptr;
    {
      phi::RecordEvent record_event(
          "construct_grad_tensor", phi::TracerEventType::UserDefined, 1);

      egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&hooked_grads[0],
                                                         this->InputMeta()[0]);
      VLOG(3) << "hooked_grads[0].size() : " << hooked_grads[0].size();
      ConstructXGradTensors(x_, &x_grad);
      ConstructParamGradTensors(params_, &params_grad);
      for (auto &i : x_grad) {
        x_grad_ptr.emplace_back(&i);
      }
      for (auto &i : params_grad) {
        params_grad_ptr.emplace_back(&i);
      }
    }

    auto out_grad_values =
        PADDLE_GET_CONST(std::vector<::pir::Value>, attrs_.at("bo_g"));
    PADDLE_ENFORCE_EQ(hooked_grads[0].size(),
                      out_grad_values.size(),
                      common::errors::InvalidArgument(
                          "The hooked_grads[0].size() and "
                          "out_grad_values.size() should be equal."));

    PirRunProgramGradAPI(hooked_grads[0],
                         step_scope_,
                         attrs_,
                         x_grad_ptr,
                         params_grad_ptr,
                         place_hash_key_);
    VLOG(3) << "End Eager Backward Node: PirGradNodeRunProgram";

    *executed_ = true;
    egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&x_grad,
                                                        this->OutputMeta()[0]);
    egr::EagerUtils::FillZeroForEmptyOptionalGradOutput(&params_grad,
                                                        this->OutputMeta()[1]);
    return {x_grad, params_grad};
  }

  void ClearTensorWrappers() override {
    x_.clear();
    params_.clear();
    SetIsTensorWrappersCleared(true);
  }

  // SetAttrMap
  void SetAttrMap(const paddle::framework::AttributeMap &attrs) {
    attrs_ = attrs;
  }

  void SetFwdX(const std::vector<paddle::Tensor> &tensors) { x_ = tensors; }

  void SetFwdParams(const std::vector<paddle::Tensor> &tensors) {
    params_ = tensors;
  }

  void SetStepScope(const std::vector<paddle::framework::Scope *> &scopes) {
    step_scope_ = scopes;
  }

  void SetPlaceHashKey(const int64_t &place_hash_key) {
    place_hash_key_ = place_hash_key;
  }

 protected:
  void ConstructXGradTensors(const std::vector<paddle::Tensor> &x,
                             std::vector<paddle::Tensor> *x_grad) {
    auto x_grad_values =
        PADDLE_GET_CONST(std::vector<::pir::Value>, attrs_.at("bx_g"));
    PADDLE_ENFORCE_EQ(
        x.size(),
        x_grad_values.size(),
        common::errors::InvalidArgument(
            "The x.size() and x_grad_names.size() should be equal. "
            "But received x.size() = %d, x_grad_names.size() = %d",
            x.size(),
            x_grad_values.size()));

    // TODO(dev): Need an elegant way to determine information of grad_tensor,
    // such as: name, tensor type (DenseTensor, SelectedRows or
    // VariableRefArray).
    for (size_t i = 0; i < x.size(); i++) {
      if (x[i].is_dense_tensor()) {
        x_grad->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (x[i].is_selected_rows()) {
        x_grad->emplace_back(std::make_shared<phi::SelectedRows>());
      } else if (details::IsVariableRefArray(x[i])) {
        x_grad->emplace_back(
            std::make_shared<paddle::framework::VariableRefArray>());
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "The grad tensor type is not supported."));
      }
    }
  }

  void ConstructParamGradTensors(const std::vector<paddle::Tensor> &params,
                                 std::vector<paddle::Tensor> *param_grads) {
    auto p_grad_values =
        PADDLE_GET_CONST(std::vector<::pir::Value>, attrs_.at("bp_g"));
    PADDLE_ENFORCE_EQ(params.size(),
                      p_grad_values.size(),
                      common::errors::InvalidArgument(
                          "The param.size() and "
                          "param_grad_names.size() should be equal."));

    for (size_t i = 0; i < params.size(); ++i) {
      auto &p = params[i];
      auto &p_grad = egr::EagerUtils::unsafe_autograd_meta(p)->Grad();
      // In eager mode, the number of param_grad should be the same as
      // param, so here an empty Tensor is added for the param with
      // stop_gradient=True
      if (!p_grad.defined()) {
        param_grads->emplace_back();
      } else if (p_grad.is_dense_tensor()) {
        param_grads->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (p_grad.is_selected_rows()) {
        param_grads->emplace_back(std::make_shared<phi::SelectedRows>());
      }
    }
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<PirGradNodeRunProgram>(
        new PirGradNodeRunProgram(*this));
    return copied_node;
  }

 private:
  // TensorWrappers
  std::vector<paddle::Tensor> x_;
  std::vector<paddle::Tensor> params_;
  std::vector<paddle::framework::Scope *> step_scope_;

  // Attribute Map
  paddle::framework::AttributeMap attrs_;

  int64_t place_hash_key_;

  std::shared_ptr<bool> executed_ = std::make_shared<bool>(false);
};
