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
#include "paddle/cinn/common/dim_expr_converter.h"
#include <unordered_map>
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn::common {
using namespace symbol;  // NOLINT

namespace {

struct DimExprToIrExprVisitor {
  ir::Expr ConvertToIrExpr(const DimExpr& dim_expr) {
    return std::visit(*this, dim_expr.variant());
  }

  ir::Expr operator()(const int64_t& dim) { return ir::Expr(dim); }

  virtual ir::Expr operator()(const std::string& dim_expr) {
    // The dimension must be greater equal than 1, and due to the extensive use
    // of int32 in CAS, the upper bound here is temporarily INT32_MAX, otherwise
    // there may be a risk of overflow.
    Var x = ir::_Var_::Make(ir::Expr(static_cast<int64_t>(1)),
                            ir::Expr(INT32_MAX),
                            dim_expr,
                            /* is_reduce  = */ false,
                            /* is_symbolic_constant = */ true);
    return x;
  }

  ir::Expr operator()(const Negative<DimExpr>& dim_expr) {
    const auto& [operand] = *dim_expr;
    return ir::Sub::Make(ir::Expr(std::int64_t(0)), ConvertToIrExpr(operand));
  }

  ir::Expr operator()(const Add<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    if (operands->empty()) {
      return ir::Expr(std::int64_t(0));
    }
    ir::Expr sum = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      sum = ir::Add::Make(sum, ConvertToIrExpr(operands->at(i)));
    }
    return sum;
  }

  ir::Expr operator()(const Mul<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    if (operands->empty()) {
      return ir::Expr(std::int64_t(1));
    }
    ir::Expr product = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      product = ir::Mul::Make(product, ConvertToIrExpr(operands->at(i)));
    }
    return product;
  }

  ir::Expr operator()(const Div<DimExpr>& dim_expr) {
    const auto& lhs = ConvertToIrExpr(dim_expr->lhs);
    const auto& rhs = ConvertToIrExpr(dim_expr->rhs);
    return ir::Div::Make(lhs, rhs);
  }

  ir::Expr operator()(const Max<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_EQ(
        !operands->empty(),
        true,
        ::common::errors::InvalidArgument("The value in dim_expr is empty"));
    ir::Expr max = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      max = ir::Max::Make(max, ConvertToIrExpr(operands->at(i)));
    }
    return max;
  }

  ir::Expr operator()(const Min<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_EQ(
        !operands->empty(),
        true,
        ::common::errors::InvalidArgument("The value in dim_expr is empty"));
    ir::Expr min = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      min = ir::Min::Make(min, ConvertToIrExpr(operands->at(i)));
    }
    return min;
  }

  // convert Broadcast to Max
  ir::Expr operator()(const Broadcast<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_EQ(
        !operands->empty(),
        true,
        ::common::errors::InvalidArgument("The value in dim_expr is empty"));
    ir::Expr max = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      max = ir::Max::Make(max, ConvertToIrExpr(operands->at(i)));
    }
    return max;
  }
};

}  // namespace

struct DimExprConverterWithSymbolBindings::
    DimExprToIrExprVisitorWithSymbolBinding : public DimExprToIrExprVisitor {
  using SymbolBinding = cinn::dialect::SymbolBinding;
  using ShapeSymbolBinding = cinn::dialect::ShapeSymbolBinding;
  using DataSymbolBinding = cinn::dialect::DataSymbolBinding;

  const std::vector<ir::Tensor>& inputs_;
  std::unordered_map<std::string, cinn::dialect::SymbolBinding>
      symbol_binding_map_;

  ir::Expr operator()(const std::string& dim_expr) override {
    PADDLE_ENFORCE_EQ(symbol_binding_map_.count(dim_expr),
                      true,
                      ::common::errors::InvalidArgument(
                          "symbol_binding_map_ does not contain dim_expr"));
    auto symbol_binding = symbol_binding_map_[dim_expr];
    auto [input_idx, input_dim_idx] = std::visit(
        [](auto&& symbol_binding) -> std::pair<int64_t, int64_t> {
          return {symbol_binding.input_tensor_idx,
                  symbol_binding.input_tensor_dim_idx};
        },
        symbol_binding);
    if (std::holds_alternative<ShapeSymbolBinding>(symbol_binding)) {
      return inputs_[input_idx]->sym_shape[input_dim_idx]->GetDimExpr();
    }
    // for data binding [S0, a, b], inputs[a] is Tensor A, return A(b)
    return ir::Cast::Make(cinn::common::I64(),
                          inputs_[input_idx](cinn::ir::Expr(input_dim_idx)));
  }

  DimExprToIrExprVisitorWithSymbolBinding(
      const std::vector<ir::Tensor>& inputs,
      const std::vector<SymbolBinding>& symbol_bindings)
      : inputs_(inputs) {
    for (const auto& symbol_binding : symbol_bindings) {
      const auto& symbol_name = std::visit(
          [](auto&& symbol_binding) -> std::string {
            return symbol_binding.symbol_name;
          },
          symbol_binding);
      symbol_binding_map_[symbol_name] = symbol_binding;
    }
  }
};

ir::Expr DimExprConverter::ConvertToIrExpr(const DimExpr& dim_expr) const {
  return DimExprToIrExprVisitor().ConvertToIrExpr(dim_expr);
}

ir::Expr DimExprConverterWithSymbolBindings::ConvertToIrExpr(
    const DimExpr& dim_expr) const {
  return visitor_->ConvertToIrExpr(dim_expr);
}

DimExprConverterWithSymbolBindings::DimExprConverterWithSymbolBindings(
    const std::vector<ir::Tensor>& inputs,
    const cinn::dialect::SymbolBindings& symbol_bindings) {
  visitor_ = std::make_shared<DimExprToIrExprVisitorWithSymbolBinding>(
      inputs, symbol_bindings);
}

}  // namespace cinn::common
