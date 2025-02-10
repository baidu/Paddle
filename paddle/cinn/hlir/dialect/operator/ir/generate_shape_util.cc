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

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include <unordered_set>
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace cinn::dialect {
using namespace symbol;  // NOLINT

namespace {

template <typename T>
std::string GetSerializedTag();

template <>
std::string GetSerializedTag<Negative<DimExpr>>() {
  return "Negative";
}

template <>
std::string GetSerializedTag<Reciprocal<DimExpr>>() {
  return "Reciprocal";
}

template <>
std::string GetSerializedTag<Add<DimExpr>>() {
  return "Add";
}

template <>
std::string GetSerializedTag<Mul<DimExpr>>() {
  return "Mul";
}

template <>
std::string GetSerializedTag<Max<DimExpr>>() {
  return "Max";
}

template <>
std::string GetSerializedTag<Min<DimExpr>>() {
  return "Min";
}

template <>
std::string GetSerializedTag<Broadcast<DimExpr>>() {
  return "Broadcast";
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx,
                                               const std::int64_t& dim_expr) {
  return pir::Int64Attribute::get(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx,
                                               const std::string& dim_expr) {
  return pir::StrAttribute::get(ctx, dim_expr);
}

template <typename T>
::pir::Attribute ConvertUnaryDimExprToAttributeImpl(::pir::IrContext* ctx,
                                                    const T& dim_expr) {
  std::vector<::pir::Attribute> attr_vecs{};
  attr_vecs.push_back(pir::StrAttribute::get(ctx, GetSerializedTag<T>()));
  const auto& operand = dim_expr->data;
  attr_vecs.push_back(ConvertDimExprToAttribute(ctx, operand));
  return pir::ArrayAttribute::get(ctx, attr_vecs);
}

::pir::Attribute ConvertDimExprToAttributeImpl(
    ::pir::IrContext* ctx, const Negative<DimExpr>& dim_expr) {
  return ConvertUnaryDimExprToAttributeImpl(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(
    ::pir::IrContext* ctx, const Reciprocal<DimExpr>& dim_expr) {
  return ConvertUnaryDimExprToAttributeImpl(ctx, dim_expr);
}

template <typename T>
::pir::Attribute ConvertVariadicDimExprToAttribute(::pir::IrContext* ctx,
                                                   const T& dim_expr) {
  std::vector<::pir::Attribute> attr_vecs{};
  attr_vecs.push_back(pir::StrAttribute::get(ctx, GetSerializedTag<T>()));
  const auto& operands = *(dim_expr.operands);
  for (const auto& operand : operands) {
    attr_vecs.push_back(ConvertDimExprToAttribute(ctx, operand));
  }
  return pir::ArrayAttribute::get(ctx, attr_vecs);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx,
                                               const Add<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx,
                                               const Mul<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx,
                                               const Max<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(::pir::IrContext* ctx,
                                               const Min<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

::pir::Attribute ConvertDimExprToAttributeImpl(
    ::pir::IrContext* ctx, const Broadcast<DimExpr>& dim_expr) {
  return ConvertVariadicDimExprToAttribute(ctx, dim_expr);
}

std::optional<DimExpr> ConvertInt64AttributeToDimExpr(
    const ::pir::Int64Attribute& attribute) {
  return DimExpr{attribute.data()};
}

std::optional<DimExpr> ConvertStrAttributeToDimExpr(
    const ::pir::StrAttribute& attribute) {
  return DimExpr{attribute.AsString()};
}

template <typename T>
std::optional<DimExpr> ConvertArrayAttributeToUnaryDimExpr(
    const ::pir::ArrayAttribute& attribute) {
  if (attribute.size() != 2) {
    return std::nullopt;
  }
  std::optional<DimExpr> operand = ConvertAttributeToDimExpr(attribute.at(1));
  if (!operand.has_value()) {
    return std::nullopt;
  }
  return T{operand.value()};
}

template <typename T>
std::optional<DimExpr> ConvertArrayAttributeToVariadicDimExpr(
    const ::pir::ArrayAttribute& attribute) {
  if (attribute.size() < 2) {
    return std::nullopt;
  }
  List<DimExpr> operands{};
  for (std::size_t i = 1; i < attribute.size(); ++i) {
    std::optional<DimExpr> operand = ConvertAttributeToDimExpr(attribute.at(i));
    if (!operand.has_value()) {
      return std::nullopt;
    }
    operands->push_back(operand.value());
  }
  return T{operands};
}

typedef std::optional<DimExpr> (*ArrayAttributeConverterT)(
    const ::pir::ArrayAttribute& attribute);

std::optional<ArrayAttributeConverterT> GetArrayAttributeConverter(
    const std::string& tag) {
  static std::unordered_map<std::string, ArrayAttributeConverterT> map{
      {GetSerializedTag<Negative<DimExpr>>(),
       &ConvertArrayAttributeToUnaryDimExpr<Negative<DimExpr>>},
      {GetSerializedTag<Reciprocal<DimExpr>>(),
       &ConvertArrayAttributeToUnaryDimExpr<Reciprocal<DimExpr>>},
      {GetSerializedTag<Add<DimExpr>>(),
       &ConvertArrayAttributeToVariadicDimExpr<Add<DimExpr>>},
      {GetSerializedTag<Mul<DimExpr>>(),
       &ConvertArrayAttributeToVariadicDimExpr<Mul<DimExpr>>},
      {GetSerializedTag<Max<DimExpr>>(),
       &ConvertArrayAttributeToVariadicDimExpr<Max<DimExpr>>},
      {GetSerializedTag<Min<DimExpr>>(),
       &ConvertArrayAttributeToVariadicDimExpr<Min<DimExpr>>},
      {GetSerializedTag<Broadcast<DimExpr>>(),
       &ConvertArrayAttributeToVariadicDimExpr<Broadcast<DimExpr>>},
  };
  const auto& iter = map.find(tag);
  if (iter == map.end()) {
    return std::nullopt;
  }
  return iter->second;
}

std::optional<DimExpr> ConvertArrayAttributeToDimExpr(
    const ::pir::ArrayAttribute& attribute) {
  if (attribute.empty()) {
    return std::nullopt;
  }
  if (!attribute.at(0).isa<::pir::StrAttribute>()) {
    return std::nullopt;
  }
  const auto& tag = attribute.at(0).dyn_cast<::pir::StrAttribute>().AsString();
  auto opt_func = GetArrayAttributeConverter(tag);
  if (!opt_func.has_value()) {
    return std::nullopt;
  }
  return opt_func.value()(attribute);
}

}  // namespace

::pir::Attribute ConvertDimExprToAttribute(pir::IrContext* ctx,
                                           const DimExpr& dim_expr) {
  return std::visit(
      [&](const auto& impl) {
        return ConvertDimExprToAttributeImpl(ctx, impl);
      },
      dim_expr.variant());
}

std::optional<DimExpr> ConvertAttributeToDimExpr(::pir::Attribute attribute) {
  if (attribute.isa<::pir::Int64Attribute>()) {
    return ConvertInt64AttributeToDimExpr(
        attribute.dyn_cast<::pir::Int64Attribute>());
  }
  if (attribute.isa<::pir::StrAttribute>()) {
    return ConvertStrAttributeToDimExpr(
        attribute.dyn_cast<::pir::StrAttribute>());
  }
  if (attribute.isa<::pir::ArrayAttribute>()) {
    return ConvertArrayAttributeToDimExpr(
        attribute.dyn_cast<::pir::ArrayAttribute>());
  }
  return std::nullopt;
}

std::optional<std::vector<symbol::DimExpr>> ConvertAttributeToDimExprs(
    ::pir::Attribute attribute) {
  if (!attribute.isa<pir::ArrayAttribute>()) return std::nullopt;
  auto array = attribute.dyn_cast<pir::ArrayAttribute>();
  std::vector<symbol::DimExpr> dim_exprs;
  for (int i = 0; i < array.size(); ++i) {
    const auto& dim_expr = ConvertAttributeToDimExpr(array.at(i));
    if (!dim_expr.has_value()) return std::nullopt;
    dim_exprs.push_back(dim_expr.value());
  }
  return dim_exprs;
}

class SubstituteDimExprHelper final {
 public:
  using DimExpr4SymbolNameT =
      std::function<std::optional<DimExpr>(const std::string& symbol_name)>;

  explicit SubstituteDimExprHelper(
      const DimExpr4SymbolNameT& DimExpr4SymbolName)
      : DimExpr4SymbolName_(DimExpr4SymbolName) {}

  std::optional<DimExpr> Substitute(const DimExpr& dim_expr) {
    return std::visit([&](const auto& impl) { return SubstituteImpl(impl); },
                      dim_expr.variant());
  }

 private:
  std::optional<DimExpr> SubstituteImpl(const std::int64_t& dim_expr) {
    return dim_expr;
  }
  std::optional<DimExpr> SubstituteImpl(const std::string& dim_expr) {
    return DimExpr4SymbolName_(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Negative<DimExpr>& dim_expr) {
    return SubstituteUnary(dim_expr);
  }
  std::optional<DimExpr> SubstituteImpl(const Reciprocal<DimExpr>& dim_expr) {
    return SubstituteUnary(dim_expr);
  }

  template <typename T>
  std::optional<DimExpr> SubstituteUnary(const T& dim_expr) {
    const auto& operand = dim_expr->data;
    const auto& substituted_operand = Substitute(operand);
    if (!substituted_operand.has_value()) {
      return std::nullopt;
    }
    return T{substituted_operand.value()};
  }

  std::optional<DimExpr> SubstituteImpl(const Add<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Mul<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Max<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Min<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Broadcast<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  template <typename T>
  std::optional<DimExpr> SubstituteVariadic(const T& dim_expr) {
    const auto& operands = *(dim_expr.operands);
    List<DimExpr> substituted_operands{};
    for (const auto& operand : operands) {
      const auto& substituted_operand = Substitute(operand);
      if (!substituted_operand.has_value()) {
        return std::nullopt;
      }
      substituted_operands->push_back(substituted_operand.value());
    }
    return T{substituted_operands};
  }

  DimExpr4SymbolNameT DimExpr4SymbolName_;
};

DimExpr SubstituteDimExpr(
    const DimExpr& dim_expr,
    const std::function<std::optional<DimExpr>(const std::string& symbol_name)>&
        DimExpr4SymbolName) {
  const auto& opt_substituted =
      SubstituteDimExprHelper(DimExpr4SymbolName).Substitute(dim_expr);
  if (opt_substituted.has_value()) return opt_substituted.value();
  return dim_expr;
}

namespace {

std::optional<DimExpr> GetDimExprBySymbolBindingImpl(
    const DataSymbolBinding& symbol_binding,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>&
        DimExpr4InputDim) {
  const symbol::ShapeOrDataDimExprs& shape_or_data_dim_expr =
      DimExpr4InputDim(symbol_binding.input_tensor_idx);
  if (!shape_or_data_dim_expr.data().has_value()) return std::nullopt;
  int dim_idx = symbol_binding.input_tensor_dim_idx;
  if (dim_idx >= shape_or_data_dim_expr.data().value().size())
    return std::nullopt;
  return shape_or_data_dim_expr.data().value().at(dim_idx);
}

std::optional<DimExpr> GetDimExprBySymbolBindingImpl(
    const ShapeSymbolBinding& symbol_binding,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>&
        DimExpr4InputDim) {
  const symbol::ShapeOrDataDimExprs& shape_or_data_dim_expr =
      DimExpr4InputDim(symbol_binding.input_tensor_idx);
  int dim_idx = symbol_binding.input_tensor_dim_idx;
  if (dim_idx >= shape_or_data_dim_expr.shape().size()) return std::nullopt;
  return shape_or_data_dim_expr.shape().at(dim_idx);
}

std::string GetSymbolNameBySymbolBinding(const SymbolBinding& symbol_binding) {
  return std::visit([](const auto& impl) { return impl.symbol_name; },
                    symbol_binding);
}

}  // namespace

std::function<std::optional<DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const SymbolBindings& symbol_bindings,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>&
        DimExpr4InputDim) {
  std::unordered_map<std::string, std::vector<SymbolBinding>>
      symbol_name2symbol_bindings{};
  for (const auto& symbol_binding : symbol_bindings) {
    symbol_name2symbol_bindings[GetSymbolNameBySymbolBinding(symbol_binding)]
        .emplace_back(symbol_binding);
  }
  return [map = std::move(symbol_name2symbol_bindings), DimExpr4InputDim](
             const std::string& symbol_name) -> std::optional<DimExpr> {
    const auto& iter = map.find(symbol_name);
    if (iter == map.end()) return std::nullopt;
    std::optional<DimExpr> ret = std::nullopt;
    for (const auto& symbol_binding : iter->second) {
      const auto& current = std::visit(
          [&](const auto& impl) {
            return GetDimExprBySymbolBindingImpl(impl, DimExpr4InputDim);
          },
          symbol_binding);
      if (!current.has_value()) return std::nullopt;
      if (ret.has_value()) {
        // Same names, same DimExprs.
        if (ret.value() != current.value()) return std::nullopt;
      } else {
        ret = current;
      }
    }
    return ret;
  };
}

namespace {

bool IsAtomicImpl(int64_t) { return true; }

bool IsAtomicImpl(const std::string&) { return true; }

bool IsAtomicImpl(const symbol::Negative<symbol::DimExpr>&) { return false; }

bool IsAtomicImpl(const symbol::Reciprocal<symbol::DimExpr>&) { return false; }

bool IsAtomicImpl(const symbol::Add<symbol::DimExpr>&) { return false; }

bool IsAtomicImpl(const symbol::Mul<symbol::DimExpr>&) { return false; }

bool IsAtomicImpl(const symbol::Max<symbol::DimExpr>&) { return false; }

bool IsAtomicImpl(const symbol::Min<symbol::DimExpr>&) { return false; }

bool IsAtomicImpl(const symbol::Broadcast<symbol::DimExpr>&) { return false; }

bool IsAtomic(const symbol::DimExpr& dim_expr) {
  return std::visit([](const auto& impl) { return IsAtomicImpl(impl); },
                    dim_expr.variant());
}

bool InputDimExprsAllSupported(
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::vector<pir::Value>& input_tensors) {
  const auto& AllSupported =
      [](const std::vector<symbol::DimExpr>& dim_exprs) -> bool {
    for (const auto& dim_expr : dim_exprs) {
      if (!IsAtomic(dim_expr)) return false;
    }
    return true;
  };
  for (const auto& input_tensor : input_tensors) {
    const auto& dim_exprs = ShapeOrDataDimExprs4Value(input_tensor);
    if (!AllSupported(dim_exprs.shape())) return false;
    if (dim_exprs.data().has_value()) {
      if (!AllSupported(dim_exprs.data().value())) return false;
    }
  }
  return true;
}

void ConvertDimExprToAttributes(pir::IrContext* ir_context,
                                const std::vector<symbol::DimExpr>& dim_exprs,
                                std::vector<pir::Attribute>* attrs) {
  attrs->clear();
  attrs->reserve(dim_exprs.size());
  for (const auto& dim_expr : dim_exprs) {
    attrs->emplace_back(ConvertDimExprToAttribute(ir_context, dim_expr));
  }
}

void CollectSymbolNames(const symbol::DimExpr& dim_expr,
                        std::set<std::string>* ret);

void CollectSymbolNamesImpl(const int64_t& dim_expr,
                            std::set<std::string>* ret) {
  // do nothing.
}

void CollectSymbolNamesImpl(const std::string& dim_expr,
                            std::set<std::string>* ret) {
  ret->insert(dim_expr);
}

template <typename T>
void CollectSymbolNamesImplForUnary(const T& dim_expr,
                                    std::set<std::string>* ret) {
  const auto& [operand] = *dim_expr;
  CollectSymbolNames(operand, ret);
}

void CollectSymbolNamesImpl(const symbol::Negative<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForUnary(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Reciprocal<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForUnary(dim_expr, ret);
}

template <typename T>
void CollectSymbolNamesImplForVariadic(const T& dim_expr,
                                       std::set<std::string>* ret) {
  const auto& operands = *(dim_expr.operands);
  for (const auto& operand : operands) {
    CollectSymbolNames(operand, ret);
  }
}

void CollectSymbolNamesImpl(const symbol::Add<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Mul<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Max<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Min<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Broadcast<symbol::DimExpr>& dim_expr,
                            std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNames(const symbol::DimExpr& dim_expr,
                        std::set<std::string>* ret) {
  return std::visit(
      [&](const auto& impl) { return CollectSymbolNamesImpl(impl, ret); },
      dim_expr.variant());
}

void CollectSymbolNames(const std::vector<symbol::DimExpr>& dim_exprs,
                        std::set<std::string>* ret) {
  for (const auto& dim_expr : dim_exprs) {
    CollectSymbolNames(dim_expr, ret);
  }
}

template <typename SymbolBindingsT>
void AppendSymbolBindings(const std::vector<symbol::DimExpr>& dim_exprs,
                          std::set<std::string>* remain_symbol_names_to_bind,
                          int in_tensor_idx,
                          SymbolBindings* symbol_bindings) {
  for (int in_tensor_dim_idx = 0; in_tensor_dim_idx < dim_exprs.size();
       ++in_tensor_dim_idx) {
    const auto& dim_expr = dim_exprs.at(in_tensor_dim_idx);
    PADDLE_ENFORCE(IsAtomic(dim_expr),
                   ::common::errors::InvalidArgument(
                       "The type of dim_expr is not atomic"));
    if (!dim_expr.isa<std::string>()) continue;
    const auto& sym_name = dim_expr.dyn_cast<std::string>();
    if (remain_symbol_names_to_bind->find(sym_name) ==
        remain_symbol_names_to_bind->end())
      continue;
    remain_symbol_names_to_bind->erase(sym_name);
    symbol_bindings->emplace_back(SymbolBindingsT{
        /*.symbol_name=*/sym_name,
        /*.input_tensor_idx=*/in_tensor_idx,
        /*.input_tensor_dim_idx=*/in_tensor_dim_idx,
    });
  }
}

void GenerateSymbolBindings(
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::vector<pir::Value>& input_tensors,
    const std::set<std::string>& symbol_names,
    SymbolBindings* symbol_bindings) {
  std::set<std::string> remain_symbol_names_to_bind = symbol_names;
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& input_tensor = input_tensors.at(i);
    const auto& dim_exprs = ShapeOrDataDimExprs4Value(input_tensor);
    AppendSymbolBindings<ShapeSymbolBinding>(
        dim_exprs.shape(), &remain_symbol_names_to_bind, i, symbol_bindings);
    if (dim_exprs.data().has_value()) {
      AppendSymbolBindings<DataSymbolBinding>(dim_exprs.data().value(),
                                              &remain_symbol_names_to_bind,
                                              i,
                                              symbol_bindings);
    }
  }
}

std::vector<pir::Value> GetMinimalInputs(
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::vector<pir::Value>& input_tensors) {
  std::unordered_set<symbol::DimExpr> handled_dim_exprs;
  std::unordered_set<pir::Value> first_occurred_input_tensors;
  auto TryCollectFirstOccurredInput_tensor =
      [&](pir::Value input_tensor,
          const std::vector<symbol::DimExpr>& dim_exprs) {
        for (const auto& dim_expr : dim_exprs) {
          if (!dim_expr.isa<std::string>()) continue;
          if (handled_dim_exprs.insert(dim_expr).second) {
            first_occurred_input_tensors.insert(input_tensor);
          }
        }
      };
  for (pir::Value input_tensor : input_tensors) {
    const auto& shape_or_data_dim_exprs =
        ShapeOrDataDimExprs4Value(input_tensor);
    if (shape_or_data_dim_exprs.data().has_value()) {
      TryCollectFirstOccurredInput_tensor(
          input_tensor, shape_or_data_dim_exprs.data().value());
    }
    TryCollectFirstOccurredInput_tensor(input_tensor,
                                        shape_or_data_dim_exprs.shape());
  }
  std::vector<pir::Value> ret{};
  ret.reserve(input_tensors.size());
  for (pir::Value input_tensor : input_tensors) {
    if (first_occurred_input_tensors.count(input_tensor) > 0) {
      ret.emplace_back(input_tensor);
    }
  }
  return ret;
}

}  // namespace

bool MakeGenerateShapeOpAttribute(
    pir::IrContext* ir_context,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::vector<symbol::DimExpr>& out_dim_exprs,
    const std::vector<pir::Value>& origin_inputs,
    std::vector<pir::Value>* minimal_inputs,
    std::vector<pir::Attribute>* output_dim_expr_attrs,
    SymbolBindings* symbol_bindings) {
  *minimal_inputs = GetMinimalInputs(ShapeOrDataDimExprs4Value, origin_inputs);
  if (!InputDimExprsAllSupported(ShapeOrDataDimExprs4Value, *minimal_inputs)) {
    VLOG(4) << "input dim_exprs are not as simple as symbols, please make sure "
               "they are handled by other passes";
    return false;
  }
  // When minimal_inputs is empty, it means that all of out dim_expr must be
  // int64.
  if (minimal_inputs->empty()) {
    for (const auto& dim_expr : out_dim_exprs) {
      if (!dim_expr.isa<int64_t>()) return false;
    }
  }
  // generate output_dim_expr_attrs
  ConvertDimExprToAttributes(
      ir_context, out_dim_exprs, /*out*/ output_dim_expr_attrs);
  // generate symbol_bindings
  std::set<std::string> symbol_names_in_out_dim_exprs{};
  CollectSymbolNames(out_dim_exprs, &symbol_names_in_out_dim_exprs);
  GenerateSymbolBindings(ShapeOrDataDimExprs4Value,
                         *minimal_inputs,
                         symbol_names_in_out_dim_exprs,
                         /*out*/ symbol_bindings);

  // check all dim exprs have symbol binding
  for (const auto& symbol_name : symbol_names_in_out_dim_exprs) {
    bool has_symbol_binding = false;
    for (const auto& symbol_binding : *symbol_bindings) {
      const std::string& symbol_binding_name = std::visit(
          [&](const auto& symbol_binding) -> const std::string& {
            return symbol_binding.symbol_name;
          },
          symbol_binding);
      if (symbol_name == symbol_binding_name) {
        has_symbol_binding = true;
        break;
      }
    }
    if (!has_symbol_binding) {
      VLOG(2) << "no symbol binding found for dim expr: " << symbol_name;
      return false;
    }
  }
  return true;
}

}  // namespace cinn::dialect
