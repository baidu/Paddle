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

#include <glog/logging.h>
#include "paddle/phi/core/enforce.h"

#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace pir {

void YieldOp::Build(Builder &builder,
                    OperationArgument &argument,
                    const std::vector<Value> &inputs) {
  argument.AddInputs(inputs);
}

void TuplePushOp::Build(Builder &builder,             // NOLINT
                        OperationArgument &argument,  // NOLINT
                        Value inlet,
                        const std::vector<Value> &elements) {
  argument.AddInput(inlet);
  argument.AddInputs(elements);
}

void TuplePushOp::Build(Builder &builder,             // NOLINT
                        OperationArgument &argument,  // NOLINT
                        Value inlet,
                        std::initializer_list<Value> element_list) {
  argument.AddInput(inlet);
  argument.AddInputs(element_list);
}

void TuplePushOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes for: TuplePushOp.";
  // Verify inputs:
  PADDLE_ENFORCE_GE(num_operands(),
                    1u,
                    common::errors::InvalidArgument(
                        "The size of inputs must no less than 1."));
  PADDLE_ENFORCE_EQ(
      operand_source(0).type().isa<InletType>(),
      true,
      common::errors::InvalidArgument(
          "The first input of cf.tuple_push must be inlet_type."));

  // No attributes should be verify.

  // Verify outputs:
  PADDLE_ENFORCE_EQ(num_results(),
                    0u,
                    common::errors::InvalidArgument(
                        "The size of outputs must be equal to 0."));
  VLOG(4) << "End Verifying for TuplePushOp.";
}

void TuplePushOp::VerifyRegion() {
  // Note(winter-wang):Constraints on the number of uses can only can be placed
  // in VerifyRegion, Otherwise cloning would fail.
  PADDLE_ENFORCE_EQ(
      operand_source(0).HasOneUse(),
      true,
      common::errors::InvalidArgument(
          "The inlet value of cf.tuple_push can only be used once."));
}

size_t TuplePushOp::tuple_size() {
  auto operands_size = num_operands();
  PADDLE_ENFORCE_GE(operands_size,
                    1u,
                    common::errors::InvalidArgument(
                        "The operands of push op must no less than 1."));
  return operands_size - 1u;
}

TuplePopOp TuplePushOp::tuple_pop_op() {
  return container_interface().tuple_pop_op();
}

void TuplePopOp::Build(Builder &builder,             // NOLINT
                       OperationArgument &argument,  // NOLINT
                       Value outlet) {
  argument.AddInput(outlet);

  auto push_op = outlet.defining_op<ContainerOpInterface>().tuple_push_op();

  auto elements_size = push_op.tuple_size();

  for (size_t index = 0; index < elements_size; ++index) {
    argument.AddOutput(push_op.inlet_element(index).type());
  }
}

void TuplePopOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes  and stack validity for: "
             "TuplePopOp.";
  // Verify inputs:
  PADDLE_ENFORCE_EQ(
      num_operands(),
      1u,
      common::errors::InvalidArgument("The size of inputs must equal to 1."));
  PADDLE_ENFORCE_EQ(
      operand_source(0).type().isa<OutletType>(),
      true,
      common::errors::InvalidArgument(
          "The first input of cf.tuple_pop must be outlet_type."));

  // No attributes should be verify.

  // Verify outputs:
}

void TuplePopOp::VerifyRegion() {
  PADDLE_ENFORCE_EQ(
      operand_source(0).HasOneUse(),
      true,
      common::errors::InvalidArgument(
          "The outlet value of cf.tuple_pop can only be used once."));

  // Verify stack validity:
  if (has_container()) {
    // can be verified only if TuplePopOp and TuplePushOp are in the same
    // sub_program
    auto pop_op = container_interface().tuple_pop_op();
    PADDLE_ENFORCE(
        *this == pop_op,
        common::errors::InvalidArgument(
            "The pop_op of tuple_pop_op must be this tuple_pop_op self."));

    auto inlet_size = tuple_push_op().tuple_size();
    PADDLE_ENFORCE(
        inlet_size == tuple_size(),
        common::errors::InvalidArgument(
            "The pop elements size must equal to push elements size."));
    for (size_t index = 0; index < inlet_size; ++index) {
      PADDLE_ENFORCE(
          outlet_element(index).type() == inlet_element(index).type(),
          common::errors::InvalidArgument(
              "The %d element's push type (%s) isn't equal to pop type (%s)",
              index,
              outlet_element(index).type(),
              inlet_element(index).type()));
    }
  }
  VLOG(4) << "End Verifying for TuplePopOp.";
}

void StackCreateOp::Build(Builder &builder, OperationArgument &argument) {
  auto stack_type = StackType::get(builder.ir_context());
  auto inlet_type = InletType::get(builder.ir_context());
  auto outlet_type = OutletType::get(builder.ir_context());
  argument.AddOutputs({stack_type, inlet_type, outlet_type});
}

void StackCreateOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: StackCreateOp.";
  // Verify inputs:
  PADDLE_ENFORCE_EQ(num_operands(),
                    0u,
                    common::errors::InvalidArgument(
                        "The size of inputs must be equal to 0."));

  // No attributes should be verify.

  // Verify outputs:
  PADDLE_ENFORCE_EQ(num_results(),
                    3u,
                    common::errors::InvalidArgument(
                        "The size of outputs must be equal to 3."));

  PADDLE_ENFORCE_EQ(
      result(0).type().isa<StackType>(),
      true,
      common::errors::InvalidArgument(
          "The first outputs of cf.stack_create must be stack_type."));
  PADDLE_ENFORCE_EQ(
      result(1).type().isa<InletType>(),
      true,
      common::errors::InvalidArgument(
          "The first outputs of cf.stack_create must be inlet_type."));
  PADDLE_ENFORCE_EQ(
      result(2).type().isa<OutletType>(),
      true,
      common::errors::InvalidArgument(
          "The first outputs of cf.stack_create must be outlet_type."));

  VLOG(4) << "End Verifying for StackCreateOp.";
}

bool StackCreateOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext *infer_context) {
  const auto &null_shape_or_data =
      symbol::ShapeOrDataDimExprs(symbol::NullShapeOrDataDimExpr());
  infer_context->SetShapeOrDataForValue(result(0), null_shape_or_data);
  infer_context->SetShapeOrDataForValue(result(1), null_shape_or_data);
  infer_context->SetShapeOrDataForValue(result(2), null_shape_or_data);
  return true;
}

size_t StackCreateOp::tuple_size() { return tuple_push_op().tuple_size(); }

Value StackCreateOp::inlet_element(size_t index) {
  return tuple_push_op().inlet_element(index);
}

Value StackCreateOp::outlet_element(size_t index) {
  return tuple_pop_op().outlet_element(index);
}

TuplePushOp StackCreateOp::tuple_push_op() {
  auto inlet_value = inlet();
  PADDLE_ENFORCE_EQ(
      inlet_value.HasOneUse(),
      true,
      common::errors::InvalidArgument("The inlet value must has one use."));
  return inlet_value.first_use().owner()->dyn_cast<TuplePushOp>();
}

TuplePopOp StackCreateOp::tuple_pop_op() {
  auto outlet_value = outlet();
  PADDLE_ENFORCE_EQ(
      outlet_value.HasOneUse(),
      true,
      common::errors::InvalidArgument("The outlet value must has one use."));
  return outlet_value.first_use().owner()->dyn_cast<TuplePopOp>();
}

void StackCreateOp::Print(IrPrinter &printer) {  // NOLINT
  static std::unordered_map<uint64_t, std::unordered_map<Operation *, size_t>>
      kCounters;
  auto &counter = kCounters[printer.id()];
  auto iter = counter.insert({*this, counter.size()});
  auto index = iter.first->second;
  if (iter.second) {
    printer.AddValueAlias(stack(), "%stack_" + std::to_string(index));
    printer.AddValueAlias(inlet(), "%inlet_" + std::to_string(index));
    printer.AddValueAlias(outlet(), "%outlet_" + std::to_string(index));
  }
  printer.PrintGeneralOperation(**this);
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::YieldOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::StackCreateOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::TuplePushOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::TuplePopOp)
