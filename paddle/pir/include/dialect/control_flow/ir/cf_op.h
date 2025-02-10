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

#include <functional>

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_interface.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/cache_grad_op_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/utils/original_attributes_filter.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
namespace pir {
class IR_API YieldOp : public Op<YieldOp, SideEffectTrait> {
 public:
  using Op::Op;
  static const char *name() { return "cf.yield"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<Value> &Value);
  void VerifySig() {}
};

///
/// \brief Push a value tuple to a container.
///
class IR_API TuplePushOp : public Op<TuplePushOp,
                                     SideEffectTrait,
                                     CacheGradOpSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "cf.tuple_push"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value inlet,
                    const std::vector<Value> &elements);
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value inlet,
                    std::initializer_list<Value> element_list);
  void VerifySig();
  void VerifyRegion();

  Value container() { return container_interface().container(); }
  Value inlet() { return operand_source(0); }
  Value outlet() { return container_interface().outlet(); }

  size_t tuple_size();
  Value inlet_element(size_t index) { return operand_source(index + 1u); }
  Value outlet_element(size_t index) {
    return container_interface().outlet_element(index);
  }
  ContainerOpInterface container_interface() {
    return inlet().defining_op<ContainerOpInterface>();
  }
  TuplePopOp tuple_pop_op();

  void CacheGradOpSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class IR_API TuplePopOp : public Op<TuplePopOp, SideEffectTrait> {
 public:
  using Op::Op;
  static const char *name() { return "cf.tuple_pop"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value outlet);
  void VerifySig();
  void VerifyRegion();

  bool has_container() { return outlet().defining_op(); }
  Value container() { return container_interface().container(); }
  Value inlet() { return container_interface().inlet(); }
  Value outlet() { return operand_source(0); }

  size_t tuple_size() { return num_results(); }
  Value inlet_element(size_t index) {
    return tuple_push_op().inlet_element(index);
  }
  Value outlet_element(size_t index) { return result(index); }
  ContainerOpInterface container_interface() {
    return outlet().defining_op<ContainerOpInterface>();
  }
  TuplePushOp tuple_push_op() { return container_interface().tuple_push_op(); }
};

class IR_API StackCreateOp : public Op<StackCreateOp,
                                       ContainerOpInterface,
                                       pir::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "cf.stack_create"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(Builder &builder,              // NOLINT
                    OperationArgument &argument);  // NOLINT
  void VerifySig();

  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);

  Value container() { return result(0); }
  Value stack() { return result(0); }
  Value inlet() { return result(1); }
  Value outlet() { return result(2); }
  std::tuple<Value, Value, Value> out() { return {stack(), inlet(), outlet()}; }

  size_t tuple_size();
  Value inlet_element(size_t index);
  Value outlet_element(size_t index);
  TuplePushOp tuple_push_op();
  TuplePopOp tuple_pop_op();

  void Print(pir::IrPrinter &printer);  // NOLINT
};
}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::YieldOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::StackCreateOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::TuplePushOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::TuplePopOp);
