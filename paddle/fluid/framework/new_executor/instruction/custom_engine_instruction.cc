// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/custom_engine_instruction.h"
#include "paddle/fluid/custom_engine/custom_engine_ext.h"
#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"

namespace paddle {
namespace framework {
namespace {
pir::Value GetVarNameToValue(const std::string &var_name,
                             ValueExecutionInfo *value_exec_info) {
  for (auto kv : value_exec_info->GetValue2VarName()) {
    if (kv.second == var_name) {
      return kv.first;
    }
  }
  PADDLE_THROW(
      common::errors::NotFound("Not found value for [%s].", var_name.c_str()));
}

void BuildEngineInputOutputValue(pir::Operation *op,
                                 ValueExecutionInfo *value_exec_info,
                                 std::vector<pir::Value> &inputs,     // NOLINT
                                 std::vector<pir::Value> &outputs) {  // NOLINT
  auto build = [&](bool is_input) {
    auto op_value = is_input ? (op->operand_source(0)) : (op->result(0));
    PADDLE_ENFORCE_EQ(value_exec_info->GetValue2VarName().count(op_value),
                      true,
                      common::errors::PreconditionNotMet(
                          "Input of custom engine op is not in name map"));

    auto op_var = value_exec_info->GetVarByValue(op_value);
    auto variable_array = op_var->Get<VariableRefArray>();

    for (uint64_t idx = 0; idx < variable_array.size(); ++idx) {
      PADDLE_ENFORCE_EQ(
          value_exec_info->GetVar2VarName().count(variable_array[idx]),
          true,
          common::errors::PreconditionNotMet(
              "[%d] the variable in engine op "
              "%s MUST in variable name map",
              idx,
              (is_input ? ("input") : ("output"))));
      std::string var_name =
          value_exec_info->GetVar2VarName().at(variable_array[idx]);
      pir::Value value = GetVarNameToValue(var_name, value_exec_info);
      if (is_input) {
        inputs.emplace_back(value);
      } else {
        outputs.emplace_back(value);
      }
      VLOG(6) << "Build engine " << (is_input ? ("input[") : ("output[")) << idx
              << "]: " << var_name;
    }
  };

  // inputs
  build(true);
  // outputs
  build(false);
}

void BuildEngineValueMap(
    pir::Operation *op,
    ValueExecutionInfo *value_exec_info,
    std::unordered_map<pir::Value, std::vector<phi::DenseTensor *>>
        &engine_value_to_tensors,
    std::unordered_map<pir::Value, std::vector<std::string>>
        &engine_value_to_var_names) {
  for (auto kv : value_exec_info->GetValue2VarName()) {
    pir::Value value = kv.first;
    std::string var_name = kv.second;

    auto var = value_exec_info->GetVarByValue(value);
    if (var->IsType<phi::DenseTensor>()) {
      const phi::DenseTensor *tensor = &(var->Get<phi::DenseTensor>());
      engine_value_to_tensors[value] = {const_cast<phi::DenseTensor *>(tensor)};
      engine_value_to_var_names[value] = {var_name};
      VLOG(6) << "Build engine value map for " << var_name;
    } else if (var->IsType<VariableRefArray>()) {
      std::vector<phi::DenseTensor *> tensors;
      std::vector<std::string> var_names;
      auto &variable_array = var->Get<VariableRefArray>();
      for (size_t i = 0; i < variable_array.size(); ++i) {
        if (variable_array[i]->IsType<phi::DenseTensor>()) {
          const phi::DenseTensor *tensor =
              &(variable_array[i]->Get<phi::DenseTensor>());
          tensors.emplace_back(const_cast<phi::DenseTensor *>(tensor));
          auto var_name_i = value_exec_info->GetVarName(variable_array[i]);
          var_names.emplace_back(var_name_i);
          VLOG(6) << "Build engine value map for Variable[" << i
                  << "]: " << var_name_i;
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "Only support Vector<DenseTensor> now, "
              "not support vector<%d>.",
              variable_array[i]->Type()));
        }
      }
      engine_value_to_tensors[value] = tensors;
      engine_value_to_var_names[value] = var_names;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Only support DenseTensor and Vector<DenseTensor> now, "
          "not support .",
          var->Type()));
    }
  }
}
}  // namespace

CustomEngineInstruction::CustomEngineInstruction(
    size_t id,
    const phi::Place &place,
    ::pir::Operation *op,
    ValueExecutionInfo *value_exec_info,
    paddle::framework::interpreter::ExecutionConfig execution_config)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  PADDLE_ENFORCE_EQ(paddle::dialect::IsCustomEngineOp(op),
                    true,
                    ::common::errors::InvalidArgument(
                        "The Op to construct CustomEngineInstruction must be a "
                        "custom engine op.  "
                        "but got op is %d",
                        op->name()));

  auto op_attributes = op->attributes();
  op_ = op;

  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();

  pir::Region &region = op->region(0);
  PADDLE_ENFORCE_EQ(region.empty(),
                    false,
                    ::common::errors::Unavailable(
                        "Required CustomEngineOp's region must not be empty."));
  pir::Block *block = &(region.front());

  paddle::framework::Scope *subgraph_scope =
      &(value_exec_info->GetScope()->NewScope());
  std::shared_ptr<ValueExecutionInfo> value_exec_info_child =
      value_exec_info->NewChild(subgraph_scope);
  std::string subgraph_prefix = op_name + "_subgraph";
  paddle::framework::BuildScope(
      *block, subgraph_prefix, execution_config, value_exec_info_child.get());
  VLOG(6) << "finish build sub scope";

  // must do this after building sub scope
  BuildEngineInputOutputValue(
      op, value_exec_info_child.get(), engine_inputs_, engine_outputs_);
  BuildEngineValueMap(op,
                      value_exec_info_child.get(),
                      engine_value_to_tensors_,
                      engine_value_to_var_names_);
  VLOG(6) << "finish build data info";

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(place),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      common::errors::PreconditionNotMet(
          "Can not find OpYamlInfoInterface for [%s]", op_name));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));
  VLOG(6) << "finish process yaml_info_parser";

  BuildPhiContext<phi::KernelContext,
                  const phi::TensorBase *,
                  phi::TensorBase *,
                  paddle::small_vector<const phi::TensorBase *>,
                  paddle::small_vector<phi::TensorBase *>,
                  true>(
      op, *value_exec_info_, yaml_info_parser, &kernel_context_);

  kernel_context_.SetDeviceContext(dev_ctx_);
  VLOG(6) << "finish process kernel context";

  InitInputsOutputsIds(op, *value_exec_info_);
  VLOG(6) << "finish process inputs outputs index";
}

void CustomEngineInstruction::Run() {
  if (!is_builed_) {
    VLOG(6) << "Start Build custom engine";
    interface_ = paddle::custom_engine::CustomEngineManager::Instance()
                     ->GetCustomEngineInterface();
    if (interface_ && interface_->graph_engine_build) {
      interface_->graph_engine_build(
          reinterpret_cast<C_CustomEngineInstruction>(this));
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "CustomEngineInstruction's "
          "C_CustomEngineInterface->graph_engine_build "
          "not implemented"));
    }

    PADDLE_ENFORCE_NOT_NULL(
        custom_engine_,
        ::common::errors::InvalidArgument(
            "custom_engine_ should not be nullptr after graph_engine_build"));

    VLOG(6) << "Finish build engine for: " << op_name_;
    is_builed_ = true;
  }

  if (interface_ && interface_->graph_engine_execute) {
    interface_->graph_engine_execute(
        reinterpret_cast<C_CustomEngineInstruction>(this));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "CustomEngineInstruction's C_CustomEngineInterface->graph_engine_run "
        "not implemented"));
  }
}

}  // namespace framework
}  // namespace paddle
