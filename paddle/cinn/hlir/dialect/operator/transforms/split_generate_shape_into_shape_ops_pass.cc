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

#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace details {

struct TensorDimInShape {
  pir::Value value;
  int axis;
};

struct TensorDimInData {
  pir::Value value;
  int axis;
};

using TensorDim = std::variant<TensorDimInShape, TensorDimInData>;

using TensorDim4SymbolNameT =
    std::function<std::optional<TensorDim>(const std::string& symbol_name)>;

using SymbolName2CachedValue = std::unordered_map<symbol::DimExpr, pir::Value>;

struct CachedDimExprToValueConverter {
  CachedDimExprToValueConverter(
      const TensorDim4SymbolNameT& TensorDim4SymbolNameVal,
      pir::PatternRewriter* rewriter_val)
      : TensorDim4SymbolName(TensorDim4SymbolNameVal), rewriter(rewriter_val) {}

  TensorDim4SymbolNameT TensorDim4SymbolName;
  pir::PatternRewriter* rewriter;

  pir::Value ConvertToValue(const symbol::DimExpr& dim_expr) {
    pir::Value value =
        std::visit([&](const auto& impl) { return ConvertToValueImpl(impl); },
                   dim_expr.variant());
    return value;
  }

  pir::Value GetInputShapeByInputTensor(pir::Value input_tensor) {
    auto iter = tensor2shape_.find(input_tensor);
    if (iter == tensor2shape_.end()) {
      pir::Value shape =
          rewriter->Build<paddle::dialect::Shape64Op>(input_tensor).out();
      pir::Value cast_shape =
          rewriter->Build<paddle::dialect::CastOp>(shape, phi::DataType::INT64)
              .out();
      iter = tensor2shape_.emplace(input_tensor, cast_shape).first;
    }
    return iter->second;
  }

 private:
  std::unordered_map<pir::Value /*input tensor*/,
                     pir::Value /*input shape tensor*/>
      tensor2shape_;

  pir::Value ConvertToValueImpl(int64_t dim_expr) {
    return rewriter
        ->Build<paddle::dialect::FullIntArrayOp>(std::vector{dim_expr},
                                                 phi::DataType::INT64)
        .out();
  }

  pir::Value ConvertToValueImpl(const std::string& symbol_name) {
    const auto& tensor_dim = TensorDim4SymbolName(symbol_name);
    PADDLE_ENFORCE(
        tensor_dim.has_value(),
        ::common::errors::PreconditionNotMet(
            "symbol [%s] are not bound to any input of generate_shape op",
            symbol_name));
    return std::visit(
        [&](const auto& impl) { return ConvertTensorDimToValue(impl); },
        tensor_dim.value());
  }

  pir::Value ConvertTensorDimToValue(const TensorDimInShape& tensor_dim) {
    pir::Value input_shape = GetInputShapeByInputTensor(tensor_dim.value);
    return ConvertTensorDimToValue(
        TensorDimInData{.value = input_shape, .axis = tensor_dim.axis});
  }

  pir::Value ConvertTensorDimToValue(const TensorDimInData& tensor_dim) {
    auto CastToInt64IfNeed = [&](pir::Value value) {
      if (value.type()
              .dyn_cast<paddle::dialect::DenseTensorType>()
              .dtype()
              .isa<pir::Int64Type>()) {
        return value;
      }
      return rewriter
          ->Build<paddle::dialect::CastOp>(value, phi::DataType::INT64)
          .out();
    };
    auto FlattenValueIfNeed = [&](pir::Value value) {
      const auto& dims =
          value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
      if (dims.size() <= 1) {
        return value;
      }
      return rewriter
          ->Build<paddle::dialect::FlattenOp>(value, 0, dims.size() - 1)
          .out();
    };
    const auto& ddim = tensor_dim.value.type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims();
    if (ddim.size() == 0 || (ddim.size() == 1 && ddim[0] == 1)) {
      return CastToInt64IfNeed(tensor_dim.value);
    }
    return CastToInt64IfNeed(rewriter
                                 ->Build<paddle::dialect::SliceOp>(
                                     FlattenValueIfNeed(tensor_dim.value),
                                     std::vector<int64_t>{0LL},
                                     std::vector<int64_t>{tensor_dim.axis},
                                     std::vector<int64_t>{tensor_dim.axis + 1},
                                     std::vector<int64_t>{},
                                     std::vector<int64_t>{})
                                 .out());
  }

  pir::Value ConvertToValueImpl(
      const symbol::Negative<symbol::DimExpr>& dim_expr) {
    PADDLE_THROW(::common::errors::Fatal(
        "Dead code. This logical should handled by "
        "ConvertToValueImpl(symbol::Add<symbol::DimExpr>)"));
  }

  pir::Value ConvertToValueImpl(const symbol::Add<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value acc = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      if (operands->at(i).isa<symbol::Negative<symbol::DimExpr>>()) {
        const auto& [operand] =
            *operands->at(i).dyn_cast<symbol::Negative<symbol::DimExpr>>();
        pir::Value operand_value = ConvertToValue(operand);
        acc = rewriter->Build<paddle::dialect::SubtractOp>(acc, operand_value)
                  .out();
      } else {
        pir::Value operand_value = ConvertToValue(operands->at(i));
        acc = rewriter->Build<paddle::dialect::AddOp>(acc, operand_value).out();
      }
    }
    return acc;
  }

  pir::Value ConvertToValueImpl(const symbol::Mul<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value prod = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      prod = rewriter->Build<paddle::dialect::MultiplyOp>(prod, operand_value)
                 .out();
    }
    return prod;
  }

  pir::Value ConvertToValueImpl(const symbol::Div<symbol::DimExpr>& dim_expr) {
    const auto& lhs = dim_expr->lhs;
    const auto& rhs = dim_expr->rhs;
    pir::Value lhs_value = ConvertToValue(lhs);
    pir::Value rhs_value = ConvertToValue(rhs);
    return rewriter->Build<paddle::dialect::DivideOp>(lhs_value, rhs_value)
        .out();
  }

  pir::Value ConvertToValueImpl(const symbol::Max<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value max = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      max =
          rewriter->Build<paddle::dialect::MaximumOp>(max, operand_value).out();
    }
    return max;
  }

  pir::Value ConvertToValueImpl(const symbol::Min<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value min = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      min =
          rewriter->Build<paddle::dialect::MinimumOp>(min, operand_value).out();
    }
    return min;
  }

  pir::Value ConvertToValueImpl(
      const symbol::Broadcast<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value broadcasted = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      broadcasted = rewriter
                        ->Build<paddle::dialect::ShapeBroadcastOp>(
                            broadcasted, operand_value)
                        .out();
    }
    return broadcasted;
  }
};

void InsertSymbolBindingImpl(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::DataSymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim) {
  (*symbol_name2tensor_dim)[symbol_binding.symbol_name] = TensorDimInData{
      .value = op.operand_source(symbol_binding.input_tensor_idx),
      .axis = symbol_binding.input_tensor_dim_idx};
}

void InsertSymbolBindingImpl(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::ShapeSymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim) {
  (*symbol_name2tensor_dim)[symbol_binding.symbol_name] = TensorDimInShape{
      .value = op.operand_source(symbol_binding.input_tensor_idx),
      .axis = symbol_binding.input_tensor_dim_idx};
}

void InsertSymbolBinding(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::SymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim) {
  return std::visit(
      [&](const auto& impl) {
        return InsertSymbolBindingImpl(op, impl, symbol_name2tensor_dim);
      },
      symbol_binding);
}

TensorDim4SymbolNameT MakeGetterTensorDim4SymbolName(
    cinn::dialect::GenerateShapeOp op) {
  std::unordered_map<std::string, TensorDim> symbol_name2tensor_dim{};
  const auto& attr_map = op->attributes();
  const auto& iter = attr_map.find("symbol_bindings");
  PADDLE_ENFORCE((iter != attr_map.end()),
                 ::common::errors::PreconditionNotMet(
                     "attr symbol_bindings MUST in attribute map for [%s] op",
                     op->name()));
  pir::Attribute attr = iter->second;
  auto* Convert =
      &cinn::dialect::GenerateShapeOp::ConvertAttributeToSymbolBindings;
  const auto& symbol_bindings = Convert(attr);
  PADDLE_ENFORCE(symbol_bindings.has_value(),
                 ::common::errors::PreconditionNotMet(
                     "attr symbol_bindings in op [%s] can "
                     "not be converted to symbol bindings",
                     op->name()));
  for (const auto& symbol_binding : symbol_bindings.value()) {
    InsertSymbolBinding(op, symbol_binding, &symbol_name2tensor_dim);
  }
  return [map = std::move(symbol_name2tensor_dim)](
             const std::string& symbol_name) -> std::optional<TensorDim> {
    auto iter = map.find(symbol_name);
    if (iter == map.end()) return std::nullopt;
    return iter->second;
  };
}
std::vector<symbol::DimExpr> GetOutDimExprs(cinn::dialect::GenerateShapeOp op) {
  const auto& attr_map = op->attributes();
  const auto& iter = attr_map.find("output_dim_exprs");
  PADDLE_ENFORCE((iter != attr_map.end()),
                 ::common::errors::PreconditionNotMet(
                     "attr output_dim_exprs MUST in attribute map for [%s] op",
                     op->name()));
  pir::Attribute output_dim_exprs_attr = iter->second;
  PADDLE_ENFORCE(
      output_dim_exprs_attr.isa<pir::ArrayAttribute>(),
      ::common::errors::PreconditionNotMet(
          "attr output_dim_exprs for [%s] op must be an pir::ArrayAttribute",
          op->name()));
  std::vector<symbol::DimExpr> ret{};
  const auto& output_dim_exprs =
      output_dim_exprs_attr.dyn_cast<pir::ArrayAttribute>();
  for (int i = 0; i < output_dim_exprs.size(); ++i) {
    const auto& attr = output_dim_exprs.at(i);
    const auto& opt_dim_expr = cinn::dialect::ConvertAttributeToDimExpr(attr);
    PADDLE_ENFORCE(opt_dim_expr.has_value(),
                   ::common::errors::InvalidArgument(
                       "opt_dim_expr is empty, it should have value"));
    ret.emplace_back(opt_dim_expr.value());
  }
  return ret;
}

std::vector<pir::Value> GetValuesOfRewrittenOps(
    const std::vector<symbol::DimExpr>& dim_exprs,
    CachedDimExprToValueConverter* converter) {
  std::vector<pir::Value> ret;
  for (const auto& dim_expr : dim_exprs) {
    const auto& simplified = symbol::SimplifyDimExpr(dim_expr);
    pir::Value value = converter->ConvertToValue(simplified);
    ret.push_back(value);
  }
  return ret;
}

pir::Value GetValueOfRewrittenOps(const std::vector<symbol::DimExpr>& dim_exprs,
                                  CachedDimExprToValueConverter* converter) {
  const std::vector<pir::Value>& values_from_dim_exprs =
      GetValuesOfRewrittenOps(dim_exprs, converter);
  if (values_from_dim_exprs.size() == 1) return values_from_dim_exprs.at(0);

  auto IsZeroDim = [](const pir::Value& value) {
    return value.type()
               .dyn_cast<paddle::dialect::DenseTensorType>()
               .dims()
               .size() == 0;
  };
  pir::Value combine_values = [&] {
    const int zero_dim_value_count = std::count_if(
        values_from_dim_exprs.begin(), values_from_dim_exprs.end(), IsZeroDim);
    if (zero_dim_value_count != 0 &&
        zero_dim_value_count != values_from_dim_exprs.size()) {
      std::vector<pir::Value> new_values = values_from_dim_exprs;
      for (size_t i = 0; i < new_values.size(); ++i) {
        if (IsZeroDim(new_values[i])) {
          new_values[i] = converter->rewriter
                              ->Build<paddle::dialect::ReshapeOp>(
                                  new_values[i], std::vector<int64_t>{1})
                              .out();
        }
      }
      return converter->rewriter->Build<pir::CombineOp>(new_values).out();
    }
    return converter->rewriter->Build<pir::CombineOp>(values_from_dim_exprs)
        .out();
  }();

  return converter->rewriter->Build<paddle::dialect::ConcatOp>(combine_values)
      .out();
}

std::optional<pir::Value> GetOutReplacement(cinn::dialect::GenerateShapeOp op,
                                            pir::PatternRewriter* rewriter) {
  std::vector<symbol::DimExpr> dim_exprs = GetOutDimExprs(op);
  TensorDim4SymbolNameT TensorDim4SymbolName =
      MakeGetterTensorDim4SymbolName(op);
  if (!TensorDim4SymbolName) return std::nullopt;
  CachedDimExprToValueConverter converter{TensorDim4SymbolName, rewriter};
  std::optional<pir::Value> new_output =
      GetValueOfRewrittenOps(dim_exprs, &converter);
  if (!new_output.has_value()) return std::nullopt;
  const auto& gs_out_type =
      op->result(0).type().dyn_cast<paddle::dialect::DenseTensorType>();
  const auto& new_output_type =
      new_output->type().dyn_cast<paddle::dialect::DenseTensorType>();
  if (gs_out_type.dtype() != new_output_type.dtype()) {
    new_output =
        rewriter
            ->Build<paddle::dialect::CastOp>(
                new_output.value(),
                paddle::dialect::TransToPhiDataType(gs_out_type.dtype()))
            .out();
  }
  if (gs_out_type.dims() != new_output_type.dims()) {
    PADDLE_ENFORCE_EQ(
        ::common::contain_unknown_dim(gs_out_type.dims()),
        false,
        ::common::errors::PreconditionNotMet(
            "The shape of the output tensor of generate_shape_op should not "
            "contain unknown dim, but received [%s].",
            gs_out_type.dims().to_str()));
    PADDLE_ENFORCE_LE(
        ::common::product(gs_out_type.dims()),
        9,
        ::common::errors::PreconditionNotMet(
            "The numel of the output tensor of generate_shape_op should be "
            "less than 9, but received [%s].",
            ::common::product(gs_out_type.dims())));
    return rewriter
        ->Build<paddle::dialect::ReshapeOp>(
            new_output.value(),
            ::common::vectorize<int64_t>(gs_out_type.dims()))
        .out();
  }
  return new_output;
}

}  // namespace details

class SplitGenerateShapeIntoShapeOps
    : public pir::OpRewritePattern<cinn::dialect::GenerateShapeOp> {
 public:
  using OpRewritePattern<cinn::dialect::GenerateShapeOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::GenerateShapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    std::optional<pir::Value> out_replacement =
        details::GetOutReplacement(op, &rewriter);
    if (!out_replacement.has_value()) return false;
    rewriter.ReplaceAllUsesWith(op->result(0), out_replacement.value());
    if (op->use_empty()) {
      rewriter.EraseOp(op);
    }
    return true;
  }
};

class SplitGenerateShapeIntoShapeOpsPass : public pir::PatternRewritePass {
 public:
  SplitGenerateShapeIntoShapeOpsPass()
      : pir::PatternRewritePass("split_generate_shape_into_shape_ops_pass", 1) {
  }

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<SplitGenerateShapeIntoShapeOps>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateSplitGenerateShapeIntoShapeOpsPass() {
  return std::make_unique<SplitGenerateShapeIntoShapeOpsPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
