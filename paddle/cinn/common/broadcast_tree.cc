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

#include "paddle/cinn/common/broadcast_tree.h"

#include <optional>
#include <unordered_map>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace cinn::common {

namespace {

template <typename DoEachT>
bool SearchBroadcast(const symbol::DimExpr& dim_expr, const DoEachT& DoEach);

template <typename DoEachT>
bool SearchBroadcastImpl(int64_t, const DoEachT& DoEach) {
  return false;
}

template <typename DoEachT>
bool SearchBroadcastImpl(const std::string&, const DoEachT& DoEach) {
  return false;
}

template <typename T, typename DoEachT>
bool SearchBroadcastImplForUnary(const T& unary, const DoEachT& DoEach) {
  const auto& operand = unary->data;
  return SearchBroadcast(operand, DoEach);
}

template <typename T, typename DoEachT>
bool SearchBroadcastImplForBinary(const T& binary, const DoEachT& DoEach) {
  const auto& lhs = binary->lhs;
  const auto& rhs = binary->rhs;
  if (SearchBroadcast(lhs, DoEach)) return true;
  if (SearchBroadcast(rhs, DoEach)) return true;
  return false;
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Negative<symbol::DimExpr>& unary,
                         const DoEachT& DoEach) {
  return SearchBroadcastImplForUnary(unary, DoEach);
}

template <typename T, typename DoEachT>
bool SearchBroadcastImplForVariadic(const T& variadic, const DoEachT& DoEach) {
  const auto& operands = *(variadic.operands);
  for (const auto& operand : operands) {
    if (SearchBroadcast(operand, DoEach)) return true;
  }
  return false;
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Add<symbol::DimExpr>& variadic,
                         const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic, DoEach);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Mul<symbol::DimExpr>& variadic,
                         const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic, DoEach);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Div<symbol::DimExpr>& binary,
                         const DoEachT& DoEach) {
  return SearchBroadcastImplForBinary(binary, DoEach);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Max<symbol::DimExpr>& variadic,
                         const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic, DoEach);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Min<symbol::DimExpr>& variadic,
                         const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic, DoEach);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Broadcast<symbol::DimExpr>& variadic,
                         const DoEachT& DoEach) {
  const auto& operands = *(variadic.operands);
  for (const auto& operand : operands) {
    PADDLE_ENFORCE_EQ(!operand.isa<int64_t>(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Invalid operand type. Expected operand "
                          "not to be of type int64_t."));
    if (SearchBroadcast(operand, DoEach)) return true;
  }
  return DoEach(variadic);
}

template <typename DoEachT>
bool SearchBroadcast(const symbol::DimExpr& dim_expr, const DoEachT& DoEach) {
  return std::visit(
      [&](const auto& impl) { return SearchBroadcastImpl(impl, DoEach); },
      dim_expr.variant());
}

template <typename DoEachT>
void ForEachBroadcastDimExpr(const BroadcastLeaf& leaves,
                             const DoEachT& DoEach) {
  for (const auto& dim_exprs : *leaves) {
    for (const auto& dim_expr : dim_exprs) {
      if (SearchBroadcast(dim_expr, DoEach)) return;
    }
  }
}

using Pattern2Placement = std::unordered_map<symbol::DimExpr, symbol::DimExpr>;

Pattern2Placement ConstructCstrLhsEqRhsReplacement(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition) {
  auto [lhs, rhs] = *broadcastable_condition;
  if (SubstituteDimExpr(rhs, Pattern2Placement{{lhs, rhs}}) != rhs) {
    return Pattern2Placement{{rhs, lhs}};
  }
  if (SubstituteDimExpr(lhs, Pattern2Placement{{rhs, lhs}}) != lhs) {
    return Pattern2Placement{{lhs, rhs}};
  }
  if (rhs.isa<std::string>()) return Pattern2Placement{{rhs, lhs}};
  if (lhs.isa<std::string>()) return Pattern2Placement{{lhs, rhs}};
  return Pattern2Placement{{lhs, rhs}};
}

Pattern2Placement ConstructCstrLhsEqOneReplacement(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition) {
  const auto& [lhs, rhs] = *broadcastable_condition;
  return Pattern2Placement{{lhs, symbol::DimExpr{1}}};
}

Pattern2Placement ConstructCstrRhsEqOneReplacement(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition) {
  const auto& [lhs, rhs] = *broadcastable_condition;
  return Pattern2Placement{{rhs, symbol::DimExpr{1}}};
}

symbol::DimExpr GetCstrLhsEqRhsDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  const auto& pattern2replacement =
      ConstructCstrLhsEqRhsReplacement(broadcastable_condition);
  return symbol::SimplifyDimExpr(
      symbol::SubstituteDimExpr(dim_expr, pattern2replacement));
}

symbol::DimExpr GetCstrLhsEqOneDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  const auto& pattern2replacement =
      ConstructCstrLhsEqOneReplacement(broadcastable_condition);
  return symbol::SimplifyDimExpr(
      symbol::SubstituteDimExpr(dim_expr, pattern2replacement));
}

symbol::DimExpr GetCstrRhsEqOneDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  const auto& pattern2replacement =
      ConstructCstrRhsEqOneReplacement(broadcastable_condition);
  return symbol::SimplifyDimExpr(
      symbol::SubstituteDimExpr(dim_expr, pattern2replacement));
}

typedef symbol::DimExpr (*ConvertDimExprT)(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr);

template <ConvertDimExprT ConvertDimExpr>
BroadcastLeaf ConvertBroadcastLeaf(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  BroadcastLeaf ret{};
  for (const auto& dim_exprs : *leaves) {
    std::vector<symbol::DimExpr> converted{};
    converted.reserve(dim_exprs.size());
    for (const auto& dim_expr : dim_exprs) {
      converted.push_back(ConvertDimExpr(broadcastable_condition, dim_expr));
    }
    ret->emplace_back(std::move(converted));
  }
  return ret;
}

BroadcastLeaf GetCstrLhsEqRhsLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrLhsEqRhsDimExpr>(broadcastable_condition,
                                                       leaves);
}

BroadcastLeaf GetCstrLhsEqOneLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrLhsEqOneDimExpr>(broadcastable_condition,
                                                       leaves);
}

BroadcastLeaf GetCstrRhsEqOneLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrRhsEqOneDimExpr>(broadcastable_condition,
                                                       leaves);
}

BroadcastBranch<BroadcastTree> ConstructBroadcastBranch(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves,
    int* num_of_leaves) {
  BroadcastLeaf cstr_lhs_eq_rhs_leaves =
      GetCstrLhsEqRhsLeaves(broadcastable_condition, leaves);
  BroadcastLeaf cstr_lhs_eq_one_leaves =
      GetCstrLhsEqOneLeaves(broadcastable_condition, leaves);
  BroadcastLeaf cstr_rhs_eq_one_leaves =
      GetCstrRhsEqOneLeaves(broadcastable_condition, leaves);
  return BroadcastBranch<BroadcastTree>{
      /*broadcastable_condition*/ broadcastable_condition,
      /*cstr_lhs_eq_rhs_branch*/
      ConstructBroadcastTree(cstr_lhs_eq_rhs_leaves, num_of_leaves),
      /*cstr_lhs_eq_one_branch*/
      ConstructBroadcastTree(cstr_lhs_eq_one_leaves, num_of_leaves),
      /*cstr_rhs_eq_one_branch*/
      ConstructBroadcastTree(cstr_rhs_eq_one_leaves, num_of_leaves)};
}

}  // namespace

std::optional<symbol::Broadcastable<symbol::DimExpr>> GetFirstCstrBroadcastable(
    const BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>> ret;
  ForEachBroadcastDimExpr(leaves, [&](const auto& broadcast) -> bool {
    const auto& operands = broadcast.operands;
    std::optional<symbol::DimExpr> lhs_symbol;
    std::optional<symbol::DimExpr> rhs_symbol;
    size_t i = 0;
    for (; i < operands->size(); ++i) {
      if (operands->at(i).template isa<std::string>()) {
        lhs_symbol = operands->at(i);
        break;
      }
    }
    for (i++; i < operands->size(); ++i) {
      if (operands->at(i).template isa<std::string>()) {
        rhs_symbol = operands->at(i);
        break;
      }
    }
    if (lhs_symbol.has_value() && rhs_symbol.has_value()) {
      PADDLE_ENFORCE_NE(lhs_symbol,
                        rhs_symbol,
                        ::common::errors::InvalidArgument(
                            "Symbols should not be equal. "
                            "Received lhs_symbol = %s, rhs_symbol = %s.",
                            lhs_symbol.value(),
                            rhs_symbol.value()));
      ret = symbol::Broadcastable<symbol::DimExpr>{lhs_symbol.value(),
                                                   rhs_symbol.value()};
      return true;
    }
    return false;
  });
  if (ret.has_value()) return ret.value();
  ForEachBroadcastDimExpr(leaves, [&](const auto& broadcast) -> bool {
    const auto& operands = broadcast.operands;
    std::optional<symbol::DimExpr> lhs_symbol;
    std::optional<symbol::DimExpr> rhs;
    for (const auto& operand : *operands) {
      if (operand.template isa<std::string>()) {
        lhs_symbol = operand;
        break;
      }
    }
    for (const auto& operand : *operands) {
      if (operand != lhs_symbol) {
        rhs = operand;
        break;
      }
    }
    if (lhs_symbol.has_value() && rhs.has_value()) {
      ret = symbol::Broadcastable<symbol::DimExpr>{lhs_symbol.value(),
                                                   rhs.value()};
      return true;
    }
    return false;
  });
  if (ret.has_value()) return ret.value();
  ForEachBroadcastDimExpr(leaves, [&](const auto& broadcast) -> bool {
    const auto& operands = broadcast.operands;
    PADDLE_ENFORCE_GE(operands->size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The operands size should be greater than 2."));
    PADDLE_ENFORCE_NE(
        operands->at(0),
        operands->at(1),
        ::common::errors::InvalidArgument("Operands should not be equal. "));
    ret = symbol::Broadcastable<symbol::DimExpr>{operands->at(0),
                                                 operands->at(1)};
    return true;
  });
  return ret;
}

BroadcastTree ConstructBroadcastTree(const BroadcastLeaf& leaves,
                                     int* num_of_leaves) {
  if (*num_of_leaves > FLAGS_pir_broadcast_tree_limit) {
    return leaves;
  }
  std::optional<symbol::Broadcastable<symbol::DimExpr>>
      broadcastable_condition = GetFirstCstrBroadcastable(leaves);
  if (!broadcastable_condition.has_value()) {
    (*num_of_leaves)++;
    return leaves;
  }
  return ConstructBroadcastBranch(
      broadcastable_condition.value(), leaves, num_of_leaves);
}

namespace {

std::string ToTxtStringImpl(const BroadcastBranch<BroadcastTree>& branch) {
  std::stringstream ss;
  const auto& [cstr, lhs_eq_rhs, lhs_eq_one, rhs_eq_one] = branch.tuple();
  const auto& [lhs, rhs] = *cstr;
  const auto& Put = [&](const std::string& key, const auto& value) {
    ss << "\"" << key << "\": ";
    ss << ToTxtString(value);
    ss << ",\n ";
  };
  ss << "{";
  ss << "\"$lhs\": " << lhs << ",\n ";
  ss << "\"$rhs\": " << rhs << ",\n ";
  Put("$lhs == $rhs", lhs_eq_rhs);
  Put("$lhs == 1", lhs_eq_one);
  Put("$rhs == 1", rhs_eq_one);
  ss << "}";
  return ss.str();
}

std::string ToTxtStringImpl(const BroadcastLeaf& leaf) {
  std::stringstream ss;
  ss << "[";
  for (const auto& dim_exprs : *leaf) {
    ss << "[";
    int j = 0;
    for (const auto& dim_expr : dim_exprs) {
      if (j++) {
        ss << ",";
      }
      ss << dim_expr;
    }
    ss << "]";
  }
  ss << "]";
  return ss.str();
}

}  // namespace

std::string ToTxtString(const BroadcastTree& tree) {
  return std::visit([&](const auto& impl) { return ToTxtStringImpl(impl); },
                    tree.variant());
}

}  // namespace cinn::common
