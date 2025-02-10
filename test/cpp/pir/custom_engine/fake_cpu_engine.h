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

#pragma once
#include <unordered_map>

#include "paddle/fluid/custom_engine/custom_engine_ext.h"
#include "paddle/fluid/framework/new_executor/instruction/custom_engine_instruction.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "test/cpp/pir/custom_engine/custom_engine_op.h"
#include "test/cpp/pir/custom_engine/fake_cpu_engine_base.h"

C_Status RegisterCustomEngineOp() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Dialect* custom_engine_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::CustomEngineDialect>();
  EXPECT_EQ(custom_engine_dialect != nullptr, true);
  ctx->RegisterOpInfo(custom_engine_dialect,
                      pir::TypeId::get<paddle::dialect::FakeEngineOp>(),
                      paddle::dialect::FakeEngineOp::name(),
                      paddle::dialect::FakeEngineOp::interface_set(),
                      paddle::dialect::FakeEngineOp::GetTraitSet(),
                      paddle::dialect::FakeEngineOp::attributes_num,
                      paddle::dialect::FakeEngineOp::attributes_name,
                      paddle::dialect::FakeEngineOp::VerifySigInvariants,
                      paddle::dialect::FakeEngineOp::VerifyRegionInvariants);
  return C_SUCCESS;
}

C_Status CustomEngineOpLower(C_CustomEngineLowerParams* lower_param) {
  // get lower params
  pir::IrContext* ctx =
      reinterpret_cast<pir::IrContext*>(lower_param->ir_context);
  pir::Operation* op_item =
      reinterpret_cast<pir::Operation*>(lower_param->operation);
  phi::KernelKey* kernel_key =
      reinterpret_cast<phi::KernelKey*>(lower_param->kernel_key);
  phi::Place* place = reinterpret_cast<phi::Place*>(lower_param->place);
  std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair =
      reinterpret_cast<std::unordered_map<pir::Operation*, pir::Operation*>*>(
          lower_param->map_op_pair);
  std::unordered_map<pir::Value, pir::Value>* map_value_pair =
      reinterpret_cast<std::unordered_map<pir::Value, pir::Value>*>(
          lower_param->map_value_pair);
  pir::Block* block = reinterpret_cast<pir::Block*>(lower_param->block);

  // Prepare output types
  std::vector<pir::Type> op_output_types;
  for (size_t i = 0; i < op_item->num_results(); ++i) {
    PushBackOutputTypes(ctx,
                        op_item,
                        op_item->result(i).type(),
                        *place,
                        *kernel_key,
                        &op_output_types);
  }
  // Prepare input
  std::vector<pir::Value> vec_inputs;

  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);
    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", i, op_item->name()));

    auto new_in = map_value_pair->at(cur_in);

    vec_inputs.push_back(new_in);
  }
  // Prepare attr
  std::unordered_map<std::string, pir::Attribute> op_attribute;
  auto op_attr_map = op_item->attributes();
  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }

  pir::OpInfo custom_engine_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::FakeEngineOp::name());
  pir::Operation* op = pir::Operation::Create(vec_inputs,
                                              op_attribute,
                                              op_output_types,
                                              custom_engine_op_info,
                                              1,
                                              {},
                                              true);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));
  op->set_attribute("op_name", pir::StrAttribute::get(ctx, op->name()));

  VLOG(3) << "CustomEngineOpLower get op_item subgraph block.";
  pir::Region& op_item_region = op_item->region(0);
  PADDLE_ENFORCE_EQ(op_item_region.empty(),
                    false,
                    ::common::errors::Unavailable(
                        "Required CustomEngineOp's region must not be empty."));
  pir::Block* sub_graph_block = &(op_item_region.front());

  VLOG(3) << "CustomEngineOpLower set new op subgraph block.";
  pir::Region& region = op->region(0);
  if (region.empty()) {
    region.emplace_back();
  }
  pir::Block* op_block = &(region.front());

  // process subgraph block
  paddle::dialect::ProcessBlock(
      *place, sub_graph_block, op_block, ctx, map_op_pair, map_value_pair);

  if (VLOG_IS_ON(3)) {
    std::stringstream ss;
    ss << "CustomEngineOpLower new op:";
    op->Print(ss);
    VLOG(3) << ss.str();
  }

  (*map_op_pair)[op_item] = op;
  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  block->push_back(op);
  return C_SUCCESS;
}

class CustomEngine {
 public:
  CustomEngine(std::vector<phi::DenseTensor*> tensor_args,
               std::vector<phi::DenseTensor*> return_tensor)
      : tensor_args_(tensor_args), return_tensor_(return_tensor) {}
  ~CustomEngine() {}

  void Run(const phi::DeviceContext& device_ctx, const phi::Place& place) {
    PADDLE_ENFORCE_EQ(
        tensor_args_.size(),
        2u,
        common::errors::PreconditionNotMet("tensor_args.size != 2"));
    PADDLE_ENFORCE_EQ(
        return_tensor_.size(),
        1u,
        common::errors::PreconditionNotMet("return_tensor.size != 1"));
    // phi::AddKernel<float, phi::DeviceContext>(device_ctx, *(tensor_args_[0]),
    // *(tensor_args_[1]),return_tensor_[0]);
    phi::Copy(device_ctx, *(tensor_args_[0]), place, true, return_tensor_[0]);
    return;
  }

 private:
  std::vector<phi::DenseTensor*> tensor_args_;
  std::vector<phi::DenseTensor*> return_tensor_;
  std::vector<phi::DenseTensor*> template_tensor_;
};

C_Status GraphEngineExecute(C_CustomEngineInstruction instruction) {
  paddle::framework::CustomEngineInstruction* instruction_ =
      reinterpret_cast<paddle::framework::CustomEngineInstruction*>(
          instruction);
  CustomEngine* customengine =
      reinterpret_cast<CustomEngine*>(instruction_->CustomEngine());

  customengine->Run(instruction_->DeviceContext(),
                    instruction_->DeviceContext().GetPlace());
  return C_SUCCESS;
}

C_Status GraphEngineBuild(C_CustomEngineInstruction instruction) {
  paddle::framework::CustomEngineInstruction* instruction_ =
      reinterpret_cast<paddle::framework::CustomEngineInstruction*>(
          instruction);
  pir::Operation* op = instruction_->Operation();
  const paddle::framework::ValueExecutionInfo* value_exec_info =
      instruction_->GetValueExecutionInfo();
  // prepare input tensors
  std::vector<phi::DenseTensor*> tensor_args;
  PADDLE_ENFORCE_EQ(
      op->num_operands(),
      1u,
      phi::errors::PreconditionNotMet("custom engine op should has 1 operand"));
  auto vec_in = op->operand_source(0).defining_op()->operands_source();
  for (auto in : vec_in) {
    auto var_name = value_exec_info->GetVarName(in);
    auto tensor = value_exec_info->GetScope()
                      ->FindVar(var_name)
                      ->GetMutable<phi::DenseTensor>();
    tensor_args.push_back(tensor);
  }

  // prepare output tensors
  std::vector<phi::DenseTensor*> return_tensor;
  PADDLE_ENFORCE_EQ(
      op->num_results(),
      1u,
      phi::errors::PreconditionNotMet("custom engine op should has 1 result"));
  pir::Value vec_result = op->result(0);
  PADDLE_ENFORCE_EQ(vec_result.type().isa<pir::VectorType>(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "custom engine op result should be vectortype"));
  auto vec_out = op->result(0).first_use().owner()->results();

  for (auto out : vec_out) {
    bool check =
        out && out.type() && out.type().isa<paddle::dialect::DenseTensorType>();
    PADDLE_ENFORCE_EQ(
        check,
        true,
        phi::errors::PreconditionNotMet(
            "customEngine instruction only support DenseTensorType"));
    auto var_name = value_exec_info->GetVarName(out);

    auto tensor = value_exec_info->GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    return_tensor.push_back(tensor);
    auto alloc_tensor_type =
        out.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    tensor->set_type(
        paddle::dialect::TransToPhiDataType(alloc_tensor_type.dtype()));
    tensor->Resize(alloc_tensor_type.dims());
  }

  CustomEngine* fake_engine = new CustomEngine(tensor_args, return_tensor);
  auto customEngineDeleter = [](void* ptr) {
    CustomEngine* customEngine = static_cast<CustomEngine*>(ptr);

    if (customEngine != nullptr) {
      delete customEngine;
    } else {
      PADDLE_THROW(phi::errors::PreconditionNotMet("customEngine is nullptr"));
    }
  };
  instruction_->SetCustomEngine(reinterpret_cast<void*>(fake_engine));
  instruction_->SetCustomEngineDeleter(customEngineDeleter);

  return C_SUCCESS;
}

void InitPluginCustomEngine(CustomEngineParams* params) {
  memset(reinterpret_cast<void*>(params->interface),
         0,
         sizeof(C_CustomEngineInterface));

  params->interface->register_custom_engine_op = RegisterCustomEngineOp;
  params->interface->graph_engine_build = GraphEngineBuild;
  params->interface->graph_engine_execute = GraphEngineExecute;
  params->interface->custom_engine_op_lower = CustomEngineOpLower;
}
