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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_static_dim_to_dynamic_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

PD_DECLARE_string(cinn_convert_static_dim_to_dynamic_dim);

namespace cinn::dialect::ir {

namespace {

template <typename DoEachT>
void ForEachRawStaticDimToDynamicPair(const DoEachT& DoEach) {
  const std::string& env_var = FLAGS_cinn_convert_static_dim_to_dynamic_dim;
  size_t start = 0;
  while (true) {
    size_t end = env_var.find(",", start);
    DoEach(env_var.substr(start, end));
    if (end == std::string::npos) return;
    start = end + 1;
  }
}

std::optional<std::pair<int64_t, std::string>> ParseRawStaticDimToDynamicPair(
    const std::string& raw_pair) {
  size_t pos = raw_pair.find(":", 0);
  if (pos == std::string::npos) return std::nullopt;
  std::int64_t constant = 0;
  try {
    constant = std::stoll(raw_pair.substr(0, pos), nullptr);
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }
  if (constant <= 0) return std::nullopt;
  std::string symbol = raw_pair.substr(pos + 1, -1);
  if (symbol == "") return std::nullopt;
  const auto& IsWordOrUnderLine = [&](const char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_');
  };
  const auto& IsDigit = [&](const char ch) { return ch >= '0' && ch <= '9'; };
  if (!IsWordOrUnderLine(symbol[0])) return std::nullopt;
  for (int i = 1; i < symbol.size(); ++i) {
    if (!(IsWordOrUnderLine(symbol[i]) || IsDigit(symbol[i])))
      return std::nullopt;
  }
  return std::pair{int64_t{constant}, symbol};
}

std::unordered_map<int64_t, std::string> GetStaticDimToDynamicFromFlag() {
  std::unordered_map<int64_t, std::string> map;
  ForEachRawStaticDimToDynamicPair([&](const std::string& raw_pair) {
    if (auto pair = ParseRawStaticDimToDynamicPair(raw_pair)) {
      map.insert(pair.value());
    }
  });
  return map;
}

using GlobalStaticDimToDynamicMapT =
    std::vector<std::pair<int64_t, std::string>>;

std::optional<GlobalStaticDimToDynamicMapT> CalcGlobalStaticDimToDynamicMap() {
  std::unordered_map<int64_t, std::string> map =
      GetStaticDimToDynamicFromFlag();
  if (map.empty()) return std::nullopt;
  auto DividedByOther = [&](int64_t constant) {
    for (const auto& [other_constant, _] : map) {
      if (constant != other_constant && constant % other_constant == 0) {
        return true;
      }
    }
    return false;
  };
  GlobalStaticDimToDynamicMapT ret;
  for (const auto& pair : map) {
    if (DividedByOther(pair.first)) continue;
    ret.push_back(pair);
  }
  return ret;
}

const std::optional<GlobalStaticDimToDynamicMapT>*
GetGlobalStaticDimToDynamicMap() {
  static std::optional<GlobalStaticDimToDynamicMapT> map(
      CalcGlobalStaticDimToDynamicMap());
  return &map;
}

struct StaticDimToDynamicConverter {
  cinn::dialect::FusionOp fusion_op;

  bool Convert() {
    bool converted_once = false;
    RewriteEachDimExpr([&](const symbol::DimExpr& dim_expr,
                           int64_t c,
                           const std::string& symbol) {
      std::optional<symbol::DimExpr> converted =
          ConvertDimExpr(dim_expr, c, symbol);
      converted_once |= converted.has_value();
      return converted;
    });
    VLOG(4) << "Finish StaticDimToDynamic Convert, Begin UpdateValueDim";
    UpdateValueDim();
    return converted_once;
  }

 private:
  std::vector<std::int64_t> GetOriginValueShape(pir::Value value) {
    auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
    return ::common::vectorize(dim);
  }

  std::vector<std::int64_t> GetTargetValueShape(
      const std::vector<symbol::DimExpr>& exprs) {
    std::vector<std::int64_t> ret{};
    for (const auto& expr : exprs) {
      if (expr.Has<std::int64_t>()) {
        ret.emplace_back(expr.Get<std::int64_t>());
      } else {
        ret.emplace_back(-1);
      }
    }
    return ret;
  }

  void UpdateValueDim() {
    pir::ShapeConstraintIRAnalysis* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(
            this->fusion_op->GetParentProgram());
    ForEachValue([&](pir::Value value) {
      const auto& origin_shape = GetOriginValueShape(value);
      const auto& target_shape = GetTargetValueShape(
          shape_analysis->GetShapeOrDataForValue(value).shape());
      PADDLE_ENFORCE_EQ(
          origin_shape.size(),
          target_shape.size(),
          ::common::errors::InvalidArgument(
              "The size of origin shape and target shape is not equal,"
              "where the size of origin shape:%d but the size of target "
              "shape:%d.",
              origin_shape.size(),
              target_shape.size()));
      const auto& origin_type = value.type().dyn_cast<::pir::DenseTensorType>();
      pir::DenseTensorType target_type =
          pir::DenseTensorType::get(pir::IrContext::Instance(),
                                    origin_type.dtype(),
                                    ::common::make_ddim(target_shape),
                                    origin_type.data_layout(),
                                    origin_type.lod(),
                                    origin_type.offset());
      value.set_type(target_type);
    });
  }

  bool AppliedOnce(const symbol::DimExpr& dim_expr, const std::string& symbol) {
    return std::visit(
        [&](const auto& impl) { return AppliedOnceImpl(impl, symbol); },
        dim_expr.variant());
  }

  bool AppliedOnceImpl(int64_t dim_expr, const std::string& symbol) {
    return false;
  }

  bool AppliedOnceImpl(const std::string& dim_expr, const std::string& symbol) {
    return dim_expr == symbol;
  }

  template <typename T>
  bool AppliedOnceUnaryImpl(const T& dim_expr, const std::string& symbol) {
    return AppliedOnce(dim_expr->data, symbol);
  }

  bool AppliedOnceImpl(const symbol::Negative<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceUnaryImpl(dim_expr, symbol);
  }

  template <typename T>
  bool AppliedOnceBinaryImpl(const T& dim_expr, const std::string& symbol) {
    const auto& lhs = dim_expr->lhs;
    const auto& rhs = dim_expr->rhs;
    return AppliedOnce(lhs, symbol) || AppliedOnce(rhs, symbol);
  }

  bool AppliedOnceImpl(const symbol::Div<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceBinaryImpl(dim_expr, symbol);
  }

  template <typename T>
  bool AppliedOnceListImpl(const T& dim_expr, const std::string& symbol) {
    const auto& [operands] = dim_expr;
    for (const auto& operand : *operands) {
      if (AppliedOnce(operand, symbol)) return true;
    }
    return false;
  }

  bool AppliedOnceImpl(const symbol::Add<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceListImpl(dim_expr, symbol);
  }

  bool AppliedOnceImpl(const symbol::Mul<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceListImpl(dim_expr, symbol);
  }

  bool AppliedOnceImpl(const symbol::Min<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceListImpl(dim_expr, symbol);
  }

  bool AppliedOnceImpl(const symbol::Max<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceListImpl(dim_expr, symbol);
  }

  bool AppliedOnceImpl(const symbol::Broadcast<symbol::DimExpr>& dim_expr,
                       const std::string& symbol) {
    return AppliedOnceListImpl(dim_expr, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExpr(const symbol::DimExpr& dim_expr,
                                                int64_t c,
                                                const std::string& symbol) {
    if (AppliedOnce(dim_expr, symbol)) return std::nullopt;
    return std::visit(
        [&](const auto& impl) { return ConvertDimExprImpl(impl, c, symbol); },
        dim_expr.variant());
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(int64_t dim_expr,
                                                    int64_t c,
                                                    const std::string& symbol) {
    if (c <= 0) return std::nullopt;
    if (dim_expr == c) return symbol::DimExpr{symbol};
    if ((dim_expr > c) && (dim_expr % c == 0)) {
      return symbol::DimExpr{symbol} * symbol::DimExpr{dim_expr / c};
    }
    return std::nullopt;
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(const std::string& dim_expr,
                                                    int64_t c,
                                                    const std::string& symbol) {
    return std::nullopt;
  }

  template <typename T>
  std::optional<symbol::DimExpr> ConvertUnaryDimExprImpl(
      const T& dim_expr, int64_t c, const std::string& symbol) {
    const auto& operand = dim_expr->data;
    const auto& converted_operand = ConvertDimExpr(operand, c, symbol);
    if (!converted_operand.has_value()) return std::nullopt;
    return T{converted_operand.value()};
  }

  template <typename T>
  std::optional<symbol::DimExpr> ConvertBinaryDimExprImpl(
      const T& dim_expr, int64_t c, const std::string& symbol) {
    const auto& lhs = dim_expr->lhs;
    const auto& rhs = dim_expr->rhs;
    const auto& converted_lhs = ConvertDimExpr(lhs, c, symbol);
    const auto& converted_rhs = ConvertDimExpr(rhs, c, symbol);
    if (!converted_lhs.has_value() && !converted_rhs.has_value())
      return std::nullopt;
    if (converted_lhs.has_value() && converted_rhs.has_value()) {
      return T{converted_lhs.value(), converted_rhs.value()};
    }
    if (converted_lhs.has_value()) {
      return T{converted_lhs.value(), rhs};
    }
    return T{lhs, converted_rhs.value()};
  }

  template <typename T>
  std::optional<symbol::DimExpr> ConvertListDimExprImpl(
      const T& dim_expr, int64_t c, const std::string& symbol) {
    const auto& [operands] = dim_expr;
    symbol::List<symbol::DimExpr> ret_operands{};
    ret_operands->reserve(operands->size());
    bool converted_once = false;
    for (const auto& operand : *operands) {
      const auto& converted_operand = ConvertDimExpr(operand, c, symbol);
      converted_once |= converted_operand.has_value();
      ret_operands->emplace_back(
          converted_operand.has_value() ? converted_operand.value() : operand);
    }
    if (!converted_once) return std::nullopt;
    return T{ret_operands};
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Negative<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertUnaryDimExprImpl(dim_expr, c, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Add<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertListDimExprImpl(dim_expr, c, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Mul<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertListDimExprImpl(dim_expr, c, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Div<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertBinaryDimExprImpl(dim_expr, c, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Max<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertListDimExprImpl(dim_expr, c, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Min<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertListDimExprImpl(dim_expr, c, symbol);
  }

  std::optional<symbol::DimExpr> ConvertDimExprImpl(
      const symbol::Broadcast<symbol::DimExpr>& dim_expr,
      int64_t c,
      const std::string& symbol) {
    return ConvertListDimExprImpl(dim_expr, c, symbol);
  }

  template <typename DoEachT>
  void RewriteEachDimExpr(const DoEachT& DoEach) {
    pir::ShapeConstraintIRAnalysis* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(
            fusion_op->GetParentProgram());
    ForEachConstantToSymbol([&](int64_t c, const std::string& symbol) {
      ForEachValue([&](pir::Value value) {
        const auto& opt_converted = ConvertShapeOrDataDimExprs(
            DoEach, shape_analysis, value, c, symbol);
        if (!opt_converted.has_value()) return;
        UpdateShapeOrDataDimExprs(
            &*shape_analysis, value, opt_converted.value());
      });
    });
  }

  void UpdateShapeOrDataDimExprs(
      pir::ShapeConstraintIRAnalysis* shape_analysis,
      pir::Value value,
      const symbol::ShapeOrDataDimExprs& shape_or_data_dim_exprs) {
    shape_analysis->SetShapeOrDataForValue(value, shape_or_data_dim_exprs);
  }

  template <typename ConverterT>
  std::optional<symbol::ShapeOrDataDimExprs> ConvertShapeOrDataDimExprs(
      const ConverterT& Converter,
      pir::ShapeConstraintIRAnalysis* shape_analysis,
      pir::Value value,
      int64_t constant,
      const std::string& symbol) {
    const auto& old = shape_analysis->GetShapeOrDataForValue(value).shape();
    return ConvertShapeOrDataDimExprs(Converter, old, constant, symbol);
  }

  template <typename ConverterT>
  std::optional<symbol::ShapeOrDataDimExprs> ConvertShapeOrDataDimExprs(
      const ConverterT& Converter,
      const std::vector<symbol::DimExpr>& dim_exprs,
      int64_t constant,
      const std::string& symbol) {
    bool converted_once = false;
    const auto& TryConvert = [&](const auto& dim_expr) {
      const auto& converted_dim_expr = Converter(dim_expr, constant, symbol);
      converted_once |= converted_dim_expr.has_value();
      return converted_dim_expr.has_value() ? converted_dim_expr.value()
                                            : dim_expr;
    };
    std::vector<symbol::DimExpr> ret_shape{};
    ret_shape.reserve(dim_exprs.size());
    for (const auto& dim_expr : dim_exprs) {
      ret_shape.emplace_back(TryConvert(dim_expr));
    }

    if (!converted_once) return std::nullopt;
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(ret_shape)};
  }

  template <typename DoEachT>
  void ForEachConstantToSymbol(const DoEachT& DoEach) {
    const auto& map = *GetGlobalStaticDimToDynamicMap();
    PADDLE_ENFORCE_EQ(map.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "map is empty, it should have value"));
    for (const auto& [constant, symbol] : map.value()) {
      DoEach(constant, symbol);
    }
  }

  template <typename DoEachT>
  void ForEachValue(const DoEachT& DoEach) {
    ForEachOp([&](::pir::Operation* op) {
      for (int i = 0; i < op->num_operands(); ++i) {
        DoEach(op->operand_source(i));
      }
      for (int i = 0; i < op->num_results(); ++i) {
        DoEach(op->result(i));
      }
    });
  }

  template <typename DoEachT>
  void ForEachOp(const DoEachT& DoEach) {
    for (auto* op : this->fusion_op.GetOperators()) {
      DoEach(op);
    }
  }
};

class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    return StaticDimToDynamicConverter{fusion_op}.Convert();
  }

 private:
};

class ConvertStaticDimToDynamicPass : public pir::PatternRewritePass {
 public:
  ConvertStaticDimToDynamicPass()
      : pir::PatternRewritePass("convert_static_dim_to_dynamic_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context);
    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::optional<std::unique_ptr<::pir::Pass>>
CreateConvertStaticDimToDynamicPass() {
  if (!GetGlobalStaticDimToDynamicMap()->has_value()) return std::nullopt;
  return std::make_unique<ConvertStaticDimToDynamicPass>();
}

}  // namespace cinn::dialect::ir
