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

#include <stack>

#include "paddle/fluid/framework/new_executor/instruction/control_flow/tuple_pop_instruction.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

namespace paddle::framework {
TuplePopInstruction::TuplePopInstruction(size_t id,
                                         const phi::Place& place,
                                         ::pir::Operation* op,
                                         ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place), op_(op), value_exe_info_(value_exe_info) {
  tuple_pop_op_ = op->dyn_cast<pir::TuplePopOp>();
  VLOG(6) << "construct tuple_pop instruction for: " << tuple_pop_op_->name();
  auto outlet_value = tuple_pop_op_.outlet();
  auto var_array = value_exe_info_->GetVarByValue(outlet_value);
  stack_element_var_array_ = var_array->GetMutable<VariableRefArray>();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  inputs.emplace(tuple_pop_op_.outlet(),
                 std::initializer_list<int>{
                     value_exe_info_->GetVarId(tuple_pop_op_.outlet())});
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
    auto outlet_element_value = tuple_pop_op_.outlet_element(i);
    outputs.emplace(outlet_element_value,
                    GetValueIds(outlet_element_value, *value_exe_info_));
  }

  // NOTE(zhangbo): TuplePop will change the variables corresponding to the
  // outlet, so it needs to be marked as output.
  outputs.emplace(tuple_pop_op_.outlet(),
                  std::initializer_list<int>{
                      value_exe_info_->GetVarId(tuple_pop_op_.outlet())});
  SetOutputs(outputs);

  type_ = OpFuncType::kCpuSync;
}

// NOTE(zhangbo): TuplePop is an instruction used in conjunction with TuplePush.
// Each TuplePush pushes 0-n variables into a
// VariableRefArray(vector<Variable*>), which may be executed any number of
// times; TuplePop needs to retrieve variables from
// VariableRefArray(vector<Variable*>) before and after, corresponding to 0-n
// variables at a time. Therefore, a Stack data structure is designed here to
// implement the function of extracting variables from vector<Variable*>
// mentioned above.
static std::stack<const Variable*> PopElements(VariableRefArray* var_array,
                                               uint64_t num) {
  std::stack<const Variable*> rtn;
  for (uint64_t i = 0; i < num; i++) {
    rtn.push(var_array->back());
    var_array->pop_back();
    VLOG(6) << "tuple pop " << rtn.top() << " from : " << var_array;
  }
  return rtn;
}
void ShareVarData(const Variable* src_var, Variable* dst_var) {
  if (src_var->IsType<phi::DenseTensor>()) {
    auto& src_tensor = src_var->Get<phi::DenseTensor>();
    auto* tmp_dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
    if (src_tensor.numel() == 0) {
      tmp_dst_tensor->set_meta(src_tensor.meta());
      if (!src_tensor.IsInitialized()) {
        return;
      }
    }
    tmp_dst_tensor->ShareDataWith(src_tensor);
  } else if (src_var->IsType<phi::SelectedRows>()) {
    auto* tmp_dst_slr = dst_var->GetMutable<phi::SelectedRows>();
    auto* dst_t = tmp_dst_slr->mutable_value();
    auto& src_slr = src_var->Get<phi::SelectedRows>();
    auto& src_t = src_slr.value();
    if (src_t.numel() == 0) {
      dst_t->set_meta(src_t.meta());
      return;
    }
    dst_t->ShareDataWith(src_t);
  } else if (src_var->IsType<phi::TensorArray>()) {
    auto src_tensor_array = src_var->Get<phi::TensorArray>();
    auto* dst_tensor_array = dst_var->GetMutable<phi::TensorArray>();
    if (src_tensor_array.size() == 0) return;
    dst_tensor_array->resize(src_tensor_array.size());
    for (size_t i = 0; i < src_tensor_array.size(); ++i) {
      phi::DenseTensor& tmp_dst_tensor = dst_tensor_array->at(i);
      if (src_tensor_array.at(i).numel() == 0) {
        tmp_dst_tensor.set_meta(src_tensor_array.at(i).meta());
      } else {
        tmp_dst_tensor.ShareDataWith(src_tensor_array.at(i));
      }
    }
  } else if (src_var->IsType<VariableRefArray>()) {
    auto src_var_array = src_var->Get<VariableRefArray>();
    auto* dst_var_array = dst_var->GetMutable<VariableRefArray>();
    for (size_t i = 0; i < src_var_array.size(); ++i) {
      Variable* copy_var = const_cast<Variable*>(dst_var_array->at(i));
      ShareVarData(src_var_array.at(i), copy_var);
    }
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Output only support DenseTensorType "
        "or SelectedRowsType or TensorArrayType"));
  }
}

void TuplePopInstruction::Run() {
  VLOG(6) << "run tuple_pop instruction";
  if (tuple_pop_op_.tuple_size() == 0) {
    stack_element_var_array_->pop_back();
  } else {
    std::stack<const Variable*> var_elements =
        PopElements(stack_element_var_array_, tuple_pop_op_.tuple_size());
    // TODO(zhangbo): Performance optimization: static acquisition of TuplePoop
    // output variables.
    for (size_t i = 0; i < tuple_pop_op_.tuple_size(); ++i) {
      auto front_var = var_elements.top();
      var_elements.pop();
      VLOG(6) << "pop back var: " << front_var;
      auto outlet_element_value = tuple_pop_op_.outlet_element(i);
      auto grad_var = value_exe_info_->GetVarByValue(outlet_element_value);
      ShareVarData(front_var, grad_var);
      Variable* gc_front_var = const_cast<Variable*>(front_var);
      AddEagerGCVar(gc_front_var);
    }
  }
}

}  // namespace paddle::framework
