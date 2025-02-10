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
#ifdef GET_OP_LIST
#undef GET_OP_LIST
paddle::dialect::IfOp, paddle::dialect::WhileOp, paddle::dialect::HasElementsOp,
    paddle::dialect::AssertOp, paddle::dialect::SelectInputOp,
    paddle::dialect::SelectOutputOp
#else
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"

#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

using pir::TuplePopOp;
using pir::TuplePushOp;
constexpr char kStopGradientAttrName[] = "stop_gradient";  // NOLINT

namespace paddle::dialect {

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::vector<pir::Type> &&output_types) {
  VLOG(4) << "Start build IfOp";
  argument.AddInput(cond);
  argument.output_types.swap(output_types);
  argument.AddRegion().emplace_back();
  argument.AddRegion().emplace_back();
  cond.set_attribute(kStopGradientAttrName, builder.bool_attr(true));
}

void IfOp::Build(pir::Builder &builder,             // NOLINT
                 pir::OperationArgument &argument,  // NOLINT
                 pir::Value cond,
                 std::unique_ptr<pir::Block> &&true_block,
                 std::unique_ptr<pir::Block> &&false_block) {
  VLOG(4) << "Start build IfOp";
#ifdef PADDLE_WITH_DISTRIBUTE
  std::vector<pir::Value> values{cond};
  if (true_block && !true_block->empty() &&
      true_block->back().isa<pir::YieldOp>()) {
    for (auto value : true_block->back().operands_source()) {
      values.push_back(value);
    }
  }
  if (false_block && !false_block->empty() &&
      false_block->back().isa<pir::YieldOp>()) {
    for (auto value : false_block->back().operands_source()) {
      values.push_back(value);
    }
  }
  ProcessMeshAttribute op_mesh;
  if (HasDistInput(values, &op_mesh)) {
    CvtAllInputsToDist(values, op_mesh);
  }
#endif

  if (true_block && !true_block->empty() &&
      true_block->back().isa<pir::YieldOp>()) {
    auto &op = true_block->back();

    std::vector<pir::Attribute> outs_stop_gradient;
    for (size_t i = 0; i < op.num_operands(); ++i) {
      argument.AddOutput(op.operand(i).type());
      auto bool_attr = op.operand_source(i).attribute<pir::BoolAttribute>(
          kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
    }

    argument.AddAttribute(
        kStopGradientAttrName,
        pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));
  }
  if (false_block && !false_block->empty() &&
      false_block->back().isa<pir::YieldOp>()) {
    auto &op = false_block->back();
    auto size = op.num_operands();
    PADDLE_ENFORCE_EQ(size,
                      argument.output_types.size(),
                      common::errors::PreconditionNotMet(
                          "The output size of true block and false block must "
                          "be equal. but they are %u and %u, respectively",
                          argument.output_types.size(),
                          size));
    for (size_t i = 0; i < size; ++i) {
      if (op.operand(i).type() != argument.output_types[i]) {
        auto l_type = op.operand(i).type().dyn_cast<pir::DenseTensorType>();
        auto r_type = argument.output_types[i].dyn_cast<pir::DenseTensorType>();
        PADDLE_ENFORCE_EQ(l_type && r_type,
                          true,
                          common::errors::PreconditionNotMet(
                              "The output[%d] of true_block&false_block must "
                              "be dense tensor type.",
                              i));
        PADDLE_ENFORCE_EQ(l_type.dtype(),
                          r_type.dtype(),
                          common::errors::PreconditionNotMet(
                              "The dtype in output[%d] of "
                              "true_block&false_block must be equal.",
                              i));
        if (l_type.data_layout() != phi::DataLayout::UNDEFINED &&
            r_type.data_layout() != phi::DataLayout::UNDEFINED) {
          PADDLE_ENFORCE_EQ(
              l_type.data_layout(),
              r_type.data_layout(),
              common::errors::PreconditionNotMet(
                  "The data_layout in output[%d] of "
                  "true_block (%s) & false_block (%s) must be equal.",
                  i,
                  l_type.data_layout(),
                  r_type.data_layout()));
        }
        PADDLE_ENFORCE_EQ(l_type.lod(),
                          r_type.lod(),
                          common::errors::PreconditionNotMet(
                              "The lod in output[%d] of true_block&false_block "
                              "must be equal.",
                              i));
        PADDLE_ENFORCE_EQ(l_type.offset(),
                          r_type.offset(),
                          common::errors::PreconditionNotMet(
                              "The offset in output[%d] of "
                              "true_block&false_block must be equal.",
                              i));
        auto dim = common::ComputeCompatibleDim(l_type.dims(), r_type.dims());
        auto new_type = DenseTensorType::get(builder.ir_context(),
                                             l_type.dtype(),
                                             dim,
                                             l_type.data_layout(),
                                             l_type.lod(),
                                             l_type.offset());
        argument.output_types[i] = new_type;
      }
    }
  } else {
    PADDLE_ENFORCE(argument.output_types.empty(),
                   common::errors::PreconditionNotMet(
                       "The output size of true block and false block must be "
                       "equal. but they are %u and 0, respectively",
                       argument.output_types.size()));
  }
  argument.AddRegion().push_back(true_block.release());
  argument.AddRegion().push_back(false_block.release());
  argument.AddInput(cond);
  cond.set_attribute(kStopGradientAttrName, builder.bool_attr(true));
}

pir::Block &IfOp::true_block() {
  pir::Region &region = true_region();
  if (region.empty()) region.emplace_back();
  return region.front();
}
pir::Block &IfOp::false_block() {
  pir::Region &region = false_region();
  if (region.empty()) region.emplace_back();
  return region.front();
}

void IfOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(*op);
  os << " = ";
  printer.PrintOpName(*op);
  printer.PrintOpId(*op);

  printer.PrintOpOperands(*op);
  printer.PrintAttributeMap(*op);
  os << " -> ";
  printer.PrintOpReturnType(*op);
  os << " {\n";
  printer.AddIndentation();
  for (auto &item : true_block()) {
    printer.PrintOperation(item);
    os << "\n";
  }
  printer.DecreaseIndentation();
  os << printer.indentation() << "} else {\n";
  printer.AddIndentation();
  for (auto &item : false_block()) {
    printer.PrintOperation(item);
    os << "\n";
  }
  printer.DecreaseIndentation();
  os << printer.indentation() << "}";
}

void IfOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: IfOp.";
  auto input_size = num_operands();
  PADDLE_ENFORCE_EQ(
      input_size,
      1u,
      common::errors::PreconditionNotMet(
          "The size %d of inputs must be equal to 1.", input_size));

  if ((*this)->operand_source(0).type().isa<pir::DenseTensorType>()) {
    PADDLE_ENFORCE(
        (*this)
            ->operand_source(0)
            .type()
            .dyn_cast<pir::DenseTensorType>()
            .dtype()
            .isa<pir::BoolType>(),
        common::errors::PreconditionNotMet(
            "Type validation failed for the 1th input, it should be a "
            "bool DenseTensorType."));
  }

  PADDLE_ENFORCE_EQ((*this)->num_regions(),
                    2u,
                    common::errors::PreconditionNotMet(
                        "The size %d of regions must be equal to 2.",
                        (*this)->num_regions()));
}

void IfOp::VerifyRegion() {
  VLOG(4) << "Start Verifying sub regions for: IfOp.";
  VLOG(4) << "Start Verifying true branch.";
  PADDLE_ENFORCE_EQ(
      (*this)->region(0).size(),
      1u,
      common::errors::PreconditionNotMet(
          "The size %d of true_region must be 1.", (*this)->region(0).size()));
  if ((*this)->num_results() != 0) {
    auto &true_block = (*this)->region(0).front();
    PADDLE_ENFORCE_GT(
        true_block.size(),
        0u,
        common::errors::PreconditionNotMet(
            "The true block must have at least one op yield op."));
    auto &true_last_op = true_block.back();
    PADDLE_ENFORCE_EQ(true,
                      true_last_op.isa<pir::YieldOp>(),
                      common::errors::PreconditionNotMet(
                          "The last of true block must be YieldOp"));
    PADDLE_ENFORCE_EQ(true_last_op.num_operands(),
                      (*this)->num_results(),
                      common::errors::PreconditionNotMet(
                          "The size of last of true block op's input must be "
                          "equal to IfOp's outputs num."));
    VLOG(4) << "Start Verifying false branch.";
    PADDLE_ENFORCE_EQ((*this)->region(1).size(),
                      1u,
                      common::errors::PreconditionNotMet(
                          "The size %d of false_region must be 1.",
                          (*this)->region(0).size()));
    auto &false_block = (*this)->region(1).front();
    PADDLE_ENFORCE_GT(
        false_block.size(),
        0u,
        common::errors::PreconditionNotMet(
            "The false block must have at least one op yield op."));
    auto &false_last_op = false_block.back();
    PADDLE_ENFORCE_EQ(true,
                      false_last_op.isa<pir::YieldOp>(),
                      common::errors::PreconditionNotMet(
                          "The last of false block must be YieldOp"));
    PADDLE_ENFORCE_EQ(false_last_op.num_operands(),
                      (*this)->num_results(),
                      common::errors::PreconditionNotMet(
                          "The size of last of false block op's input must be "
                          "equal to IfOp's outputs num."));
  }
}

std::vector<std::vector<pir::Value>> IfOp::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs_,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE_EQ(inputs_.size() >= 1u,
                    true,
                    common::errors::InvalidArgument(
                        "if op's inputs' size should greater_equal "
                        "to 1, and all the inputs[i] "
                        "should be 1 size. "
                        "Now the inputs's size is %d .",
                        inputs_.size()));

  VLOG(6) << "Prepare inputs for if_grad";
  auto cond_val = inputs_[0][0];
  VLOG(6) << "Prepare attributes for if_grad";

  VLOG(6) << "Prepare outputs for if_grad";

  std::vector<pir::Type> output_types;
  for (size_t i = 1; i < inputs_.size(); ++i) {
    if (!stop_gradients[i - 1][0]) {
      output_types.push_back(inputs_[i][0].type());
    }
  }

  auto if_grad = ApiBuilder::Instance().GetBuilder()->Build<IfOp>(
      cond_val, std::move(output_types));

  std::vector<std::vector<pir::Value>> res{inputs_.size() - 1};
  for (size_t i = 1, j = 0; i < inputs_.size(); ++i) {
    res[i - 1].resize(1);
    if (!stop_gradients[i - 1][0]) {
      res[i - 1][0] = if_grad->result(j++);
    }
  }
  return res;
}

bool IfOp::InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context) {
  // infer true block
  pir::InferSymExprForBlock(true_block(), infer_context);

  // infer false block
  pir::InferSymExprForBlock(false_block(), infer_context);

  auto GetSymExprForBlockResult =
      [infer_context](const pir::Operation &op,
                      uint32_t idx) -> const std::vector<symbol::DimExpr> & {
    return infer_context->GetShapeOrDataForValue(op.operand_source(idx))
        .shape();
  };

  // TODO(lanxianghit): for llama, `if` op's result num always > 0, but
  // result_num == 0 should be supported in future
  if (num_results() > 0) {
    for (uint32_t rst_idx = 0; rst_idx < num_results(); rst_idx++) {
      const auto &true_dims =
          GetSymExprForBlockResult(true_block().back(), rst_idx);
      const auto &false_dims =
          GetSymExprForBlockResult(false_block().back(), rst_idx);

      // merge shape for true and false block, new symbol will be assigned when
      // the dims is not equal in true and false block, even if the dims are all
      // constant, since we don't know which will be returned in compile time
      // examples:
      // true_block    false_block    return
      // [1, 128]       [1, 256]      [1, S0]
      // [1, S0]        [1, S1]       [1, S2]
      // [1, S0]        [S1, S2]      [S1, S3]
      // [1, S0]        [1, S0]       [1, S0]

      std::vector<symbol::DimExpr> out_dims = true_dims;
      if (false_dims.size() != 0) {
        // now only support results of true and false block have same rank.
        PADDLE_ENFORCE_EQ(true_dims.size(),
                          false_dims.size(),
                          common::errors::PreconditionNotMet(
                              "The true and false block should have same rank, "
                              "but got true_rank(%d) and false_rank(%d)",
                              true_dims.size(),
                              false_dims.size()));
        for (size_t i = 0; i < true_dims.size(); i++) {
          if (true_dims[i] != false_dims[i]) {
            out_dims[i] = symbol::DimExpr{infer_context->GetNextSymName()};
          }
        }
      }

      infer_context->SetShapeOrDataForValue(
          result(rst_idx),
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(out_dims)});
    }

    return true;
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("IfOp::InferSymbolicShape: now only "
                                      "support num_results() == 1."));
  }
}

void WhileOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    const std::vector<pir::Value> &inputs,
                    bool construct_body) {
  argument.AddInput(cond);
  argument.AddInputs(inputs);
  std::vector<pir::Attribute> outs_stop_gradient;
  if (construct_body) {
    auto &body = argument.AddRegion().emplace_back();
    for (auto val : inputs) {
      argument.AddOutput(val.type());
      auto arg = body.AddArg(val.type());
      auto bool_attr = val.attribute<pir::BoolAttribute>(kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
      arg.set_attribute(kStopGradientAttrName,
                        bool_attr ? bool_attr : builder.bool_attr(false));
    }
  } else {
    argument.AddRegion(nullptr);
    for (auto val : inputs) {
      argument.AddOutput(val.type());
      auto bool_attr = val.attribute<pir::BoolAttribute>(kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
    }
  }

  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));

  cond.set_attribute(kStopGradientAttrName, builder.bool_attr(true));
}
pir::Block &WhileOp::body() {
  pir::Region &body_region = (*this)->region(0);
  if (body_region.empty()) body_region.emplace_back();
  return body_region.front();
}

pir::Value WhileOp::cond() { return (*this)->operand_source(0); }

void WhileOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(*op);
  os << " = \"" << name() << "\"";
  if (VLOG_IS_ON(1)) {
    os << " [id:" << op->id() << "]";
  }
  os << " (cond=";
  printer.PrintValue(cond());
  os << ", inputs=";
  auto operands = (*this)->operands_source();
  pir::detail::PrintInterleave(
      operands.begin() + 1,
      operands.end(),
      [&](pir::Value v) { printer.PrintValue(v); },
      [&]() { os << ", "; });
  os << ") { \n";
  os << printer.indentation() << "^";
  pir::detail::PrintInterleave(
      body().args_begin(),
      body().args_end(),
      [&](pir::Value v) { printer.PrintValue(v); },
      [&]() { os << ", "; });
  os << "\n";
  printer.AddIndentation();
  for (auto &item : body()) {
    printer.PrintOperation(item);
    os << "\n";
  }
  printer.DecreaseIndentation();
  os << printer.indentation() << "}";
}

void WhileOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: WhileOp.";
  auto input_size = num_operands();
  PADDLE_ENFORCE_GE(
      input_size,
      1u,
      common::errors::PreconditionNotMet(
          "The size %d of inputs must be greater or equal to 1.", input_size));

  if (auto cond_type = operand_type(0).dyn_cast<pir::DenseTensorType>()) {
    PADDLE_ENFORCE_EQ(
        cond_type.dtype().isa<pir::BoolType>(),
        true,
        common::errors::PreconditionNotMet(
            "Type validation failed for the 0th input, it should be a "
            "bool DenseTensorType."));
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Currently,  the while op cond input only support bool dense_tensor "
        "and bool allocated_dense_tensor."));
  }
  PADDLE_ENFORCE_EQ((*this)->num_regions(),
                    1u,
                    common::errors::PreconditionNotMet(
                        "The size %d of regions must be equal to 1.",
                        (*this)->num_regions()));
  auto output_size = num_results();
  PADDLE_ENFORCE_EQ(output_size + 1,
                    input_size,
                    common::errors::PreconditionNotMet(
                        "The result size (%d) not equal to input size(%d) + 1.",
                        num_results(),
                        input_size));
  for (size_t index = 0; index < output_size; ++index) {
    auto input_type = operand_type(index + 1);
    auto output_type = result_type(index);
    if (input_type.isa<pir::DenseTensorType>()) {
      // Support the case that the output tensor has -1 shape.
      pir::DenseTensorType input_tensor_type =
          input_type.dyn_cast<pir::DenseTensorType>();
      pir::DenseTensorType output_tensor_type =
          output_type.dyn_cast<pir::DenseTensorType>();

      auto GetCheckType = [&](const pir::DenseTensorType &type) {
        const auto &input_dims = input_tensor_type.dims();
        const auto &output_dims = output_tensor_type.dims();
        auto result_dims = type.dims();
        for (int i = 0; i < result_dims.size(); i++) {
          if (input_dims[i] == -1 || output_dims[i] == -1) {
            result_dims[i] = -1;
          }
        }
        return pir::DenseTensorType::get(pir::IrContext::Instance(),
                                         type.dtype(),
                                         result_dims,
                                         type.data_layout(),
                                         type.lod(),
                                         type.offset());
      };
      pir::DenseTensorType check_input_tensor_type =
          GetCheckType(input_tensor_type);
      pir::DenseTensorType check_output_tensor_type =
          GetCheckType(output_tensor_type);

      PADDLE_ENFORCE_EQ(
          check_input_tensor_type,
          check_output_tensor_type,
          common::errors::PreconditionNotMet(
              "The (%d) result and operand type is not equal.", index));
    } else {
      PADDLE_ENFORCE_EQ(
          input_type,
          output_type,
          common::errors::PreconditionNotMet(
              "The (%d) result and operand type is not equal.", index));
    }
  }
}

void WhileOp::VerifyRegion() {
  VLOG(4) << "Start verifying sub regions for: WhileOp.";
  PADDLE_ENFORCE_EQ(
      (*this)->region(0).size(),
      1u,
      common::errors::PreconditionNotMet(
          "The size %d of body_region must be 1.", (*this)->region(0).size()));
  auto &body_block = body();
  auto output_size = num_results();
  PADDLE_ENFORCE_EQ(
      body_block.args_size(),
      output_size,
      common::errors::PreconditionNotMet(
          "The result size (%d) not equal to block args size(%d) + 1.",
          output_size,
          body_block.args_size()));

  PADDLE_ENFORCE_EQ(
      body_block.empty(),
      false,
      common::errors::PreconditionNotMet("The body block is empty."));

  auto yield_op = body_block.back().dyn_cast<pir::YieldOp>();
  auto input_size = num_operands();
  PADDLE_ENFORCE_EQ(
      yield_op && yield_op.num_operands() == input_size,
      true,
      common::errors::PreconditionNotMet(
          "The body block yield size not equal to operands size."));
  // Todo: fix other bugs and make the following code work.
  // for (size_t index = 0; index < input_size; ++index) {
  //   PADDLE_ENFORCE_EQ(
  //       operand_type(index),
  //       yield_op.operand_type(index),
  //       common::errors::PreconditionNotMet(
  //           "The (%d) operand and block yield type is not equal.", index));
  // }
  VLOG(4) << "Successful end verifying sub regions for: WhileOp.";
}

std::vector<std::vector<pir::Value>> WhileOp::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  auto fwd_op = WhileOp::dyn_cast(op);
  PADDLE_ENFORCE_NE(fwd_op,
                    nullptr,
                    common::errors::InvalidArgument(
                        "The input op used to called WhileOp::vjp "
                        "must be non-nullptr while_op"));
  TuplePushOp push_op;
  for (auto iter = fwd_op.body().rbegin(); iter != fwd_op.body().rend();
       ++iter) {
    if (iter->isa<TuplePushOp>()) {
      push_op = iter->dyn_cast<TuplePushOp>();
      PADDLE_ENFORCE_EQ(push_op.container().use_empty(),
                        true,
                        common::errors::InvalidArgument(
                            "The last container in forward while op must used "
                            "empty while construct while_grad op"));
      break;
    }
  }
  PADDLE_ENFORCE_NE(push_op,
                    nullptr,
                    common::errors::InvalidArgument(
                        "The forward WhileOp must include TuplePushOp, denying "
                        "that we can't construct a reverse loop condition."));

  PADDLE_ENFORCE_GT(inputs.size(),
                    outputs.size(),
                    common::errors::InvalidArgument(
                        "while op's inputs' size should greater than "
                        "outputs' size, Now the inputs's size is %d ."
                        "the outputs size is %d.",
                        inputs.size(),
                        outputs.size()));
  PADDLE_ENFORCE_EQ(inputs.size(),
                    out_grads.size() + 1,
                    common::errors::InvalidArgument(
                        "while op's inputs' size should equal to "
                        "output_grads' size + 1, Now the inputs's size is %d ."
                        "the output_grads size is %d.",
                        inputs.size(),
                        out_grads.size()));
  PADDLE_ENFORCE_EQ(stop_gradients[0][0],
                    true,
                    common::errors::InvalidArgument(
                        "The stop_gradient of condition input must be true."));

  auto &builder = *ApiBuilder::Instance().GetBuilder();
  auto cond_val = builder.Build<HasElementsOp>(push_op.container()).out();

  std::vector<pir::Type> output_types;
  std::vector<pir::Value> loop_vars;

  for (size_t index = 0; index < out_grads.size(); ++index) {
    if (!stop_gradients[index + 1][0]) {
      loop_vars.push_back(out_grads[index][0]);
    }
  }
  auto while_grad = builder.Build<WhileOp>(cond_val, loop_vars);

  std::vector<std::vector<pir::Value>> res(inputs.size());
  for (size_t i = 0, j = 0; i < inputs.size(); ++i) {
    res[i].push_back(stop_gradients[i][0] ? nullptr : while_grad.result(j++));
  }
  return res;
}

void InitBlockArgSymbolicShape(const pir::Value &origin_input,
                               const pir::Value &block_arg,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &origin_input_shape_or_data =
      infer_context->GetShapeOrDataForValue(origin_input);
  origin_input_shape_or_data.Match(
      [&](const symbol::TensorShapeOrDataDimExprs &impl) {
        infer_context->SetSymbolForValueByStaticShape(block_arg);
        const auto &origin_data = impl.data();
        const DenseTensorType &type_info =
            block_arg.type().dyn_cast<DenseTensorType>();
        bool need_to_set_data = [&]() {
          if (!origin_data.has_value()) {
            return false;
          }
          if (!type_info.dtype().isa<pir::Int32Type>() &&
              !type_info.dtype().isa<pir::Int64Type>()) {
            return false;
          }
          if (common::contain_unknown_dim(type_info.dims())) {
            return false;
          }
          if (common::product(type_info.dims()) > 9) {
            return false;
          }
          return true;
        }();
        if (need_to_set_data) {
          const auto &block_arg_shape =
              infer_context->GetShapeOrDataForValue(block_arg).shape();
          std::vector<symbol::DimExpr> block_arg_data;
          for (size_t i = 0; i < origin_data.value().size(); ++i) {
            block_arg_data.emplace_back(infer_context->GetNextSymName());
          }
          infer_context->SetShapeOrDataForValue(
              block_arg,
              symbol::ShapeOrDataDimExprs(symbol::TensorShapeOrDataDimExprs(
                  block_arg_shape, block_arg_data)));
        }
      },
      [&](const symbol::TensorListShapeOrDataDimExprs &impl) {
        PADDLE_THROW(common::errors::Fatal(
            "Dead code, TensorList should not be handled in while args."));
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs &impl) {
        const auto &input_shape_hint = impl.GetShapeHint();
        std::vector<symbol::DimExpr> block_arg_shape_hint;
        for (size_t i = 0; i < input_shape_hint.size(); ++i) {
          block_arg_shape_hint.emplace_back(infer_context->GetNextSymName());
        }
        infer_context->SetShapeOrDataForValue(
            block_arg,
            symbol::ShapeOrDataDimExprs(
                symbol::RankedTensorArrayShapeOrDataDimExprs(
                    block_arg_shape_hint)));
      },
      [&](const symbol::NullShapeOrDataDimExpr &impl) {
        PADDLE_THROW(common::errors::Fatal(
            "Dead code, Null value should not be handled in while args."));
      });
}

void AddCstrForArgs(const pir::Value &origin_input,
                    const pir::Value &yield_value,
                    const pir::Value &block_arg,
                    const int &arg_index,
                    pir::InferSymbolicShapeContext *infer_context) {
  const auto &block_arg_shape_or_data =
      infer_context->GetShapeOrDataForValue(block_arg);
  block_arg_shape_or_data.Match(
      [&](const symbol::TensorShapeOrDataDimExprs &impl) {
        const auto &block_arg_shape = impl.shape();
        const auto &yield_value_shape =
            infer_context->GetShapeOrDataForValue(yield_value).shape();
        PADDLE_ENFORCE_EQ(block_arg_shape.size(),
                          yield_value_shape.size(),
                          common::errors::InvalidArgument(
                              "while op's input[%d] rank should equal to "
                              "output[%d]'s rank, Now the rank of input is %d,"
                              "the rank of output is %d.",
                              arg_index,
                              arg_index + 1,
                              block_arg_shape.size(),
                              yield_value_shape.size()));
        const auto &original_input_shape =
            infer_context->GetShapeOrDataForValue(origin_input).shape();
        if (original_input_shape.size() != block_arg_shape.size()) {
          return;
        }
        // GTOne
        for (size_t j = 0; j < original_input_shape.size(); ++j) {
          if (infer_context->IsGreatThanOne(original_input_shape[j])) {
            infer_context->AddGreatThanOneCstr(block_arg_shape[j]);
          }
        }

        // Equal
        for (size_t j = 0; j < block_arg_shape.size(); ++j) {
          if (block_arg_shape[j].isa<int64_t>()) {
            continue;
          }
          if (block_arg_shape[j] ==
              yield_value_shape[j]) {  // Dim isn't changed in while
            infer_context->AddEqualCstr(original_input_shape[j],
                                        block_arg_shape[j]);
            continue;
          }
          if (original_input_shape.size() == yield_value_shape.size()) {
            if (original_input_shape[j] == yield_value_shape[j]) {
              infer_context->AddEqualCstr(original_input_shape[j],
                                          block_arg_shape[j]);
              continue;
            }
            symbol::DimExprBuilder builder;
            if (yield_value_shape[j] ==
                    builder.Broadcast(block_arg_shape[j],
                                      original_input_shape[j]) ||
                yield_value_shape[j] ==
                    builder.Broadcast(original_input_shape[j],
                                      block_arg_shape[j])) {
              infer_context->AddEqualCstr(original_input_shape[j],
                                          block_arg_shape[j]);
              continue;
            }
          }
        }
      },
      [&](const symbol::TensorListShapeOrDataDimExprs &impl) {
        PADDLE_THROW(common::errors::Fatal(
            "Dead code, TensorList should not be handled in while args."));
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs &impl) {
        // TensorArray no need to add constraints
        return;
      },
      [&](const symbol::NullShapeOrDataDimExpr &impl) {
        PADDLE_THROW(common::errors::Fatal(
            "Dead code, Null value should not be handled in while args."));
      });
}

void AddCstrForOutputs(const pir::Value &origin_input,
                       const pir::Value &output,
                       const pir::Value &block_arg,
                       pir::InferSymbolicShapeContext *infer_context) {
  const auto &origin_input_shape_or_data =
      infer_context->GetShapeOrDataForValue(origin_input);
  origin_input_shape_or_data.Match(
      [&](const symbol::TensorShapeOrDataDimExprs &impl) {
        const auto &origin_input_shape = impl.shape();
        const auto &output_shape =
            infer_context->GetShapeOrDataForValue(output).shape();
        const auto &block_arg_shape =
            infer_context->GetShapeOrDataForValue(block_arg).shape();
        if (origin_input_shape.size() !=
            block_arg_shape
                .size()) {  // there is a trick, so the size may vary.
          return;
        }
        for (size_t j = 0; j < output_shape.size(); j++) {
          if (infer_context->IsEqual(output_shape[j], block_arg_shape[j])) {
            infer_context->AddEqualCstr(output_shape[j], origin_input_shape[j]);
          }
        }
      },
      [&](const symbol::TensorListShapeOrDataDimExprs &impl) {
        PADDLE_THROW(common::errors::Fatal(
            "Dead code, TensorList should not be handled in while args."));
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs &impl) {
        // TensorArray no need to add constraints
        return;
      },
      [&](const symbol::NullShapeOrDataDimExpr &impl) {
        PADDLE_THROW(common::errors::Fatal(
            "Dead code, Null value should not be handled in while args."));
      });
}

bool WhileOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext *infer_context) {
  const auto &body_args = block_args();
  PADDLE_ENFORCE_EQ(num_operands() - 1,
                    body_args.size(),
                    common::errors::InvalidArgument(
                        "The num_operands-1 and body_args.size is not equal"));
  for (size_t i = 0; i < body_args.size(); ++i) {
    InitBlockArgSymbolicShape(
        operand_source(i + 1), body_args[i], infer_context);
  }

  pir::InferSymExprForBlock(body(), infer_context);

  for (size_t i = 0; i < body_args.size(); ++i) {
    AddCstrForArgs(operand_source(i + 1),
                   body().back().operand_source(i + 1),
                   body_args[i],
                   i,
                   infer_context);
  }

  // Set ShapeOrDataDimExpr for results
  PADDLE_ENFORCE_EQ(body_args.size(),
                    num_results(),
                    common::errors::InvalidArgument(
                        "The body_args.size and num_results is not equal"));
  const auto &yield_op = body().back();
  bool need_infer_block_again = false;
  for (size_t i = 1; i < yield_op.num_operands(); ++i) {
    const symbol::ShapeOrDataDimExprs &yield_op_input_shape_or_data =
        infer_context->GetShapeOrDataForValue(yield_op.operand_source(i));
    if (!yield_op_input_shape_or_data
             .isa<symbol::TensorShapeOrDataDimExprs>()) {
      continue;
    }
    const std::vector<symbol::DimExpr> &yield_op_input_shape =
        yield_op_input_shape_or_data.shape();
    const std::vector<symbol::DimExpr> &block_arg_shape =
        infer_context->GetShapeOrDataForValue(body_args[i - 1]).shape();
    std::vector<symbol::DimExpr> new_block_arg_dims = block_arg_shape;

    bool need_set_block_args_again = false;
    for (size_t j = 0; j < yield_op_input_shape.size(); j++) {
      if (block_arg_shape[j].isa<int64_t>() &&
          (yield_op_input_shape[j] != block_arg_shape[j])) {
        need_infer_block_again = true;
        need_set_block_args_again = true;
        new_block_arg_dims[j] =
            symbol::DimExpr{infer_context->GetNextSymName()};
      }
    }
    // Reset block_args.
    if (need_set_block_args_again) {
      infer_context->SetShapeOrDataForValue(
          body_args[i - 1],
          symbol::ShapeOrDataDimExprs(
              symbol::TensorShapeOrDataDimExprs(new_block_arg_dims)));
    }
  }

  if (need_infer_block_again) {
    pir::InferSymExprForBlock(body(), infer_context);
  }

  for (size_t i = 0; i < num_results(); ++i) {
    infer_context->SetShapeOrDataForValue(
        result(i),
        infer_context->GetShapeOrDataForValue(yield_op.operand_source(i + 1)));
  }

  for (size_t i = 0; i < num_results(); ++i) {
    AddCstrForOutputs(
        operand_source(i + 1), result(i), body_args[i], infer_context);
  }

  return true;
}

std::vector<std::vector<pir::Value>> TuplePushOpVjpInterfaceModel::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs.size() >= 1u,
      true,
      common::errors::InvalidArgument("tuple_push op's inputs' size should be "
                                      "greater_equal than 1, and the "
                                      "inputs[i] should be non-empty. "
                                      "Now the inputs's size is %d.",
                                      inputs.size()));
  auto pop_op = ApiBuilder::Instance().GetBuilder()->Build<TuplePopOp>(
      TuplePushOp::dyn_cast(op).outlet());
  std::vector<std::vector<pir::Value>> res{inputs.size()};
  res[0].resize(1);
  for (size_t i = 1u; i < inputs.size(); ++i) {
    res[i].resize(1);
    res[i][0] = pop_op.result(i - 1);
  }
  return res;
}

void HasElementsOp::Build(pir::Builder &builder,             // NOLINT
                          pir::OperationArgument &argument,  // NOLINT
                          pir::Value container) {
  argument.AddInput(container);
  argument.AddOutput(
      DenseTensorType::get(builder.ir_context(), builder.bool_type(), {1}));
  std::vector<pir::Attribute> outs_stop_gradient{builder.bool_attr(true)};
  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}
void HasElementsOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes for: HasElementsOp.";
  // Verify inputs:
  PADDLE_ENFORCE_EQ(
      num_operands(),
      1u,
      common::errors::InvalidArgument("The size of inputs must equal to 1."));
  PADDLE_ENFORCE_EQ(
      operand_type(0).isa<pir::ContainerType>(),
      true,
      common::errors::InvalidArgument(
          "The first input of cf.has_elements must be container type."));

  // No attributes should be verify.

  // Verify outputs:
  PADDLE_ENFORCE_EQ(num_results(),
                    1u,
                    common::errors::InvalidArgument(
                        "The size of outputs must be equal to 1."));
  PADDLE_ENFORCE_EQ((*this)->result_type(0).isa<DenseTensorType>(),
                    true,
                    common::errors::InvalidArgument(
                        "The type of cf.has_elements' output is not correct."));
}

bool HasElementsOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      out(),
      symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs({symbol::DimExpr(1)})));
  return true;
}
const char *AssertOp::attributes_name[1] = {"summarize"};    // NOLINT
const char AssertOp::ERROR_INFO_ATTR_NAME[] = "error_info";  // NOLINT

void AssertOp::Build(pir::Builder &builder,             // NOLINT
                     pir::OperationArgument &argument,  // NOLINT
                     pir::Value cond_,
                     pir::Value data_,
                     int64_t summarize) {
  VLOG(4) << "Start build AssertOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {cond_, data_};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute attr_summarize =
      pir::Int64Attribute::get(pir::IrContext::Instance(), summarize);
  argument.AddAttribute("summarize", attr_summarize);
}

OpInfoTuple AssertOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      paddle::dialect::OpInputInfo("cond",
                                   "paddle::dialect::DenseTensorType",
                                   false,
                                   false,
                                   false,
                                   false),
      paddle::dialect::OpInputInfo(
          "data",
          "pir::VectorType<paddle::dialect::DenseTensorType>",
          false,
          false,
          false,
          false)};
  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo("summarize", "pir::Int64Attribute", "")};
  std::vector<paddle::dialect::OpOutputInfo> outputs = {};
  paddle::dialect::OpRunTimeInfo run_time_info = paddle::dialect::OpRunTimeInfo(
      "", {""}, "assert", {"cond", "data", "summarize"}, {"cond"}, {}, {}, {});
  return std::make_tuple(inputs, attributes, outputs, run_time_info, "assert");
}

void AssertOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: AssertOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        2u,
        common::errors::InvalidArgument(
            "The size %d of inputs must be equal to 2.", input_size));

    if ((*this)->operand_source(0).type().isa<pir::DenseTensorType>()) {
      PADDLE_ENFORCE_EQ(
          (*this)
              ->operand_source(0)
              .type()
              .dyn_cast<pir::DenseTensorType>()
              .dtype()
              .isa<pir::BoolType>(),
          true,
          common::errors::InvalidArgument(
              "Type validation failed for the 0th input, it should be a "
              "bool DenseTensorType."));
    }

    if (auto vec_type =
            (*this)->operand(1).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            vec_type[i].isa<paddle::dialect::DenseTensorType>() ||
                vec_type[i].isa<paddle::dialect::SelectedRowsType>(),
            true,
            common::errors::InvalidArgument(
                "Type validation failed for the 1th input."));
      }
    } else {
      PADDLE_ENFORCE_EQ(
          (*this)->operand(1).type().isa<paddle::dialect::DenseTensorType>() ||
              (*this)
                  ->operand(1)
                  .type()
                  .isa<paddle::dialect::SelectedRowsType>(),
          true,
          common::errors::InvalidArgument(
              "Type validation failed for the 1th input."));
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    PADDLE_ENFORCE_GT(
        attributes.count("summarize"),
        0,
        common::errors::InvalidArgument("summarize does not exist."));
    PADDLE_ENFORCE_EQ(
        attributes.at("summarize").isa<pir::Int64Attribute>(),
        true,
        common::errors::InvalidArgument(
            "Type of attribute: summarize is not pir::Int64Attribute."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        0u,
        common::errors::InvalidArgument(
            "The size %d of outputs must be equal to 0.", output_size));
    // Outputs num is 0, not need to check outputs type.
  }
  VLOG(4) << "End Verifying for: AssertOp.";
}

void SelectInputOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SelectInputOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto in_size = num_operands();
    PADDLE_ENFORCE_EQ(in_size,
                      3u,
                      common::errors::InvalidArgument(
                          "Size %d of inputs must be 3.", in_size));
    auto input1 = (*this)->operand_source(1).type();
    auto input2 = (*this)->operand_source(2).type();
    if (input1.isa<paddle::dialect::DenseTensorType>() &&
        input2.isa<paddle::dialect::DenseTensorType>()) {
      auto tensor1 = input1.dyn_cast<paddle::dialect::DenseTensorType>();
      auto tensor2 = input2.dyn_cast<paddle::dialect::DenseTensorType>();
      PADDLE_ENFORCE_EQ(
          tensor1.dtype(),
          tensor2.dtype(),
          common::errors::InvalidArgument(
              "The 1st input dtype %s should be equal to 2ed input dtype %s.",
              tensor1.dtype(),
              tensor2.dtype()));
      PADDLE_ENFORCE_EQ(
          tensor1.data_layout(),
          tensor2.data_layout(),
          common::errors::InvalidArgument(
              "The 1st input data_layout %s should be equal to 2ed input "
              "data_layout %s.",
              tensor1.data_layout(),
              tensor2.data_layout()));
      PADDLE_ENFORCE_EQ(
          tensor1.lod(),
          tensor2.lod(),
          common::errors::InvalidArgument(
              "The 1st input lod %s should be equal to 2ed input lod %s.",
              tensor1.lod(),
              tensor2.lod()));
      PADDLE_ENFORCE_EQ(
          tensor1.offset(),
          tensor2.offset(),
          common::errors::InvalidArgument(
              "The 1st input offset %s should be equal to 2ed input offset %s.",
              tensor1.offset(),
              tensor2.offset()));
    } else if (input1.isa<paddle::dialect::AllocatedDenseTensorType>() &&
               input2.isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto tensor1 =
          input1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      auto tensor2 =
          input1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      PADDLE_ENFORCE_EQ(
          tensor1.dtype(),
          tensor2.dtype(),
          common::errors::InvalidArgument(
              "The 1st input dtype %s should be equal to 2ed input dtype %s.",
              tensor1.dtype(),
              tensor2.dtype()));
      PADDLE_ENFORCE_EQ(
          tensor1.data_layout(),
          tensor2.data_layout(),
          common::errors::InvalidArgument(
              "The 1st input data_layout %s should be equal to 2ed input "
              "data_layout %s.",
              tensor1.data_layout(),
              tensor2.data_layout()));
      PADDLE_ENFORCE_EQ(
          tensor1.lod(),
          tensor2.lod(),
          common::errors::InvalidArgument(
              "The 1st input lod %s should be equal to 2ed input lod %s.",
              tensor1.lod(),
              tensor2.lod()));
      PADDLE_ENFORCE_EQ(
          tensor1.offset(),
          tensor2.offset(),
          common::errors::InvalidArgument(
              "The 1st input offset %s should be equal to 2ed input offset %s.",
              tensor1.offset(),
              tensor2.offset()));
      PADDLE_ENFORCE_EQ(
          tensor1.place(),
          tensor2.place(),
          common::errors::InvalidArgument(
              "The 1st input place %s should be equal to 2ed input place %s.",
              tensor1.place(),
              tensor2.place()));
    } else {
      PADDLE_ENFORCE_EQ(
          input1,
          input2,
          common::errors::InvalidArgument(
              "The 1st input type %s should be equal to 2ed input type %s.",
              input1,
              input2));
    }
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto out_size = num_results();
    PADDLE_ENFORCE_EQ(out_size,
                      1u,
                      common::errors::InvalidArgument(
                          "Size %d of outputs must be equal to 1.", out_size));
  }
  VLOG(4) << "End Verifying for: AssignArray_Op.";
}

bool SelectInputOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext *infer_context) {
  const auto &input1_dims =
      infer_context->GetShapeOrDataForValue(operand_source(1)).shape();
  const auto &input2_dims =
      infer_context->GetShapeOrDataForValue(operand_source(2)).shape();

  // for compatibility, we just return second_shape.
  if (input1_dims.size() != input2_dims.size()) {
    infer_context->SetShapeOrDataForValue(
        result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(input2_dims)});
    return true;
  }

  std::vector<symbol::DimExpr> out_dims = input1_dims;
  // merge shape for input1 and input2, since we don't know which will be
  // selected in compile time, the strategy is same with IfOp, see IfOp's
  // comments for details and examples
  if (input2_dims.size() != 0) {
    for (size_t i = 0; i < input1_dims.size(); i++) {
      if (input1_dims[i] != input2_dims[i]) {
        out_dims[i] = symbol::DimExpr{infer_context->GetNextSymName()};
      }
    }
  }

  infer_context->SetShapeOrDataForValue(
      result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

  return true;
}

void SelectOutputOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SelectOutputOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto in_size = num_operands();
    PADDLE_ENFORCE_EQ(in_size,
                      2u,
                      common::errors::InvalidArgument(
                          "Size %d of inputs must be 2.", in_size));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto out_size = num_results();
    PADDLE_ENFORCE_EQ(out_size,
                      2u,
                      common::errors::InvalidArgument(
                          "Size %d of outputs must be equal to 2.", out_size));

    auto out1 = (*this)->result(0).type();
    auto out2 = (*this)->result(1).type();
    if (out1.isa<paddle::dialect::DenseTensorType>() &&
        out2.isa<paddle::dialect::DenseTensorType>()) {
      auto tensor1 = out1.dyn_cast<paddle::dialect::DenseTensorType>();
      auto tensor2 = out2.dyn_cast<paddle::dialect::DenseTensorType>();
      PADDLE_ENFORCE_EQ(
          tensor1.dtype(),
          tensor2.dtype(),
          common::errors::InvalidArgument(
              "The 1st input dtype %s should be equal to 2ed input dtype %s.",
              tensor1.dtype(),
              tensor2.dtype()));
      PADDLE_ENFORCE_EQ(
          tensor1.data_layout(),
          tensor2.data_layout(),
          common::errors::InvalidArgument(
              "The 1st input data_layout %s should be equal to 2ed input "
              "data_layout %s.",
              tensor1.data_layout(),
              tensor2.data_layout()));
      PADDLE_ENFORCE_EQ(
          tensor1.lod(),
          tensor2.lod(),
          common::errors::InvalidArgument(
              "The 1st input lod %s should be equal to 2ed input lod %s.",
              tensor1.lod(),
              tensor2.lod()));
      PADDLE_ENFORCE_EQ(
          tensor1.offset(),
          tensor2.offset(),
          common::errors::InvalidArgument(
              "The 1st input offset %s should be equal to 2ed input offset %s.",
              tensor1.offset(),
              tensor2.offset()));
    } else if (out1.isa<paddle::dialect::AllocatedDenseTensorType>() &&
               out2.isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto tensor1 = out1.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      auto tensor2 = out2.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      PADDLE_ENFORCE_EQ(
          tensor1.dtype(),
          tensor2.dtype(),
          common::errors::InvalidArgument(
              "The 1st input dtype %s should be equal to 2ed input dtype %s.",
              tensor1.dtype(),
              tensor2.dtype()));
      PADDLE_ENFORCE_EQ(
          tensor1.data_layout(),
          tensor2.data_layout(),
          common::errors::InvalidArgument(
              "The 1st input data_layout %s should be equal to 2ed input "
              "data_layout %s.",
              tensor1.data_layout(),
              tensor2.data_layout()));
      PADDLE_ENFORCE_EQ(
          tensor1.lod(),
          tensor2.lod(),
          common::errors::InvalidArgument(
              "The 1st input lod %s should be equal to 2ed input lod %s.",
              tensor1.lod(),
              tensor2.lod()));
      PADDLE_ENFORCE_EQ(
          tensor1.offset(),
          tensor2.offset(),
          common::errors::InvalidArgument(
              "The 1st input offset %s should be equal to 2ed input offset %s.",
              tensor1.offset(),
              tensor2.offset()));
      PADDLE_ENFORCE_EQ(
          tensor1.place(),
          tensor2.place(),
          common::errors::InvalidArgument(
              "The 1st input place %s should be equal to 2ed input place %s.",
              tensor1.place(),
              tensor2.place()));
    } else {
      PADDLE_ENFORCE_EQ(
          out1,
          out2,
          common::errors::InvalidArgument(
              "The 1st input type %s should be equal to 2ed input type %s.",
              out1,
              out2));
    }
  }
  VLOG(4) << "End Verifying for: AssignArray_Op.";
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::HasElementsOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AssertOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SelectInputOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SelectOutputOp)

#endif
