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

#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include <numeric>

namespace symbol {

namespace {

template <typename T>
DimExpr TrySimplifyPass(const DimExpr& expr) {
  if (!expr.Has<typename T::dim_expr_type>()) {
    return expr;
  }
  return T().Rewrite(expr);
}

DimExpr Simplify(const DimExpr& expr);

/*
 * Simplify Example:
 * Negative(S0) => Negative(Simplify(S0))
 */
template <template <typename> class Op>
struct SimplifyOneOperand {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    auto [operand] = *expr.Get<Op<DimExpr>>();
    const auto& ret_operand = Simplify(operand);
    if (ret_operand == operand) {
      return expr;
    } else {
      return Op<DimExpr>{ret_operand};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

template <template <typename> class Op>
struct SimplifyOneOperandTrait;

/*
 * Simplify Example:
 * Negative(0) => 0
 */
template <template <typename> class Op>
struct SimplifyUnitOneOperand {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operand] = *expr.Get<Op<DimExpr>>();
    if (operand.template Has<std::int64_t>() &&
        operand.template Get<std::int64_t>() ==
            SimplifyOneOperandTrait<Op>::unit) {
      return DimExpr{SimplifyOneOperandTrait<Op>::unit};
    } else {
      return expr;
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

/*
 * Simplify Example:
 * Negative(Negative(S0)) => S0
 * Negative(int) => -int
 */
struct SimplifyDoubleNeg {
  using dim_expr_type = Negative<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& inner_expr = expr.Get<Negative<DimExpr>>()->data;
    if (inner_expr.Has<Negative<DimExpr>>()) {
      const auto& ret_expr = inner_expr.Get<Negative<DimExpr>>()->data;
      return ret_expr;
    } else if (inner_expr.Has<std::int64_t>()) {
      return -inner_expr.Get<std::int64_t>();
    } else {
      return expr;
    }
  }
};

template <>
struct SimplifyOneOperandTrait<Negative> {
  static constexpr std::int64_t unit = 0;
};

/*
 * Simplify Example:
 * Add(S0, S1, ...) =>
 * Add(Simplify(S0), Simplify(S1), ...)
 */
template <template <typename> class Op>
struct SimplifyOperands {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Op<DimExpr>>();
    List<DimExpr> mut_operands{};
    for (const auto& operand : *operands) {
      mut_operands->emplace_back(Simplify(operand));
    }
    if (mut_operands == operands) {
      return expr;
    } else {
      return Op<DimExpr>{mut_operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

template <>
struct SimplifyOperands<Div> {
  using dim_expr_type = Div<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& div_expr = expr.Get<Div<DimExpr>>();
    const auto& lhs = div_expr->lhs;
    const auto& rhs = div_expr->rhs;
    const auto& ret_lhs = Simplify(lhs);
    const auto& ret_rhs = Simplify(rhs);
    if (lhs == ret_lhs && rhs == ret_rhs) {
      return expr;
    } else {
      return Div<DimExpr>{ret_lhs, ret_rhs};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

template <typename T>
struct GetOrderValue;

template <>
struct GetOrderValue<Broadcast<DimExpr>> {
  static constexpr int value = 10;
};

template <>
struct GetOrderValue<Mul<DimExpr>> {
  static constexpr int value = 20;
};

template <>
struct GetOrderValue<Div<DimExpr>> {
  static constexpr int value = 30;
};

template <>
struct GetOrderValue<Add<DimExpr>> {
  static constexpr int value = 40;
};

template <>
struct GetOrderValue<std::string> {
  static constexpr int value = 50;
};

template <>
struct GetOrderValue<std::int64_t> {
  static constexpr int value = 60;
};

template <>
struct GetOrderValue<Negative<DimExpr>> {
  static constexpr int value = 70;
};

template <>
struct GetOrderValue<Max<DimExpr>> {
  static constexpr int value = 80;
};

template <>
struct GetOrderValue<Min<DimExpr>> {
  static constexpr int value = 90;
};

bool IsLhsBeforeRhs(const DimExpr& lhs, const DimExpr& rhs);

template <template <typename> class Op>
struct IsListLhsBeforeListRhsStruct {
  static bool Call(const Op<DimExpr>& lhs, const Op<DimExpr>& rhs) {
    const auto& [lhs_operands] = lhs;
    const auto& [rhs_operands] = rhs;
    if (lhs_operands->empty() || rhs_operands->empty()) {
      throw std::runtime_error("Operands are uninitialized.");
    }
    if (lhs_operands->size() < rhs_operands->size()) {
      return true;
    }
    if (lhs_operands->size() > rhs_operands->size()) {
      return false;
    }
    for (std::size_t i = 0; i < lhs_operands->size(); ++i) {
      if (lhs_operands->at(i) == rhs_operands->at(i)) continue;
      return IsLhsBeforeRhs(lhs_operands->at(i), rhs_operands->at(i));
    }
    return true;
  }
};

template <typename T0, typename T1>
struct IsLhsBeforeRhsStruct {
  static bool Call(const T0& lhs, const T1& rhs) {
    return GetOrderValue<T0>::value < GetOrderValue<T1>::value;
  }
};

template <>
struct IsLhsBeforeRhsStruct<std::int64_t, std::int64_t> {
  static bool Call(std::int64_t lhs, std::int64_t rhs) { return lhs < rhs; }
};

template <>
struct IsLhsBeforeRhsStruct<std::string, std::string> {
  static bool Call(const std::string& lhs, const std::string& rhs) {
    return lhs.compare(rhs) < 0;
  }
};

template <>
struct IsLhsBeforeRhsStruct<Negative<DimExpr>, Negative<DimExpr>> {
  static bool Call(const Negative<DimExpr>& lhs, const Negative<DimExpr>& rhs) {
    const auto& lhs_operand = lhs->data;
    const auto& rhs_operand = rhs->data;
    return IsLhsBeforeRhs(lhs_operand, rhs_operand);
  }
};

template <>
struct IsLhsBeforeRhsStruct<Add<DimExpr>, Add<DimExpr>> final
    : public IsListLhsBeforeListRhsStruct<Add> {};

template <>
struct IsLhsBeforeRhsStruct<Mul<DimExpr>, Mul<DimExpr>> final
    : public IsListLhsBeforeListRhsStruct<Mul> {};

template <>
struct IsLhsBeforeRhsStruct<Div<DimExpr>, Div<DimExpr>> {
  static bool Call(const Div<DimExpr>& lhs, const Div<DimExpr>& rhs) {
    const auto& lhs_lhs = lhs->lhs;
    const auto& lhs_rhs = lhs->rhs;
    const auto& rhs_lhs = rhs->lhs;
    const auto& rhs_rhs = rhs->rhs;

    return IsLhsBeforeRhs(lhs_lhs, rhs_lhs) && IsLhsBeforeRhs(lhs_rhs, rhs_rhs);
  }
};

template <>
struct IsLhsBeforeRhsStruct<Broadcast<DimExpr>, Broadcast<DimExpr>> final
    : public IsListLhsBeforeListRhsStruct<Broadcast> {};

bool IsLhsBeforeRhs(const DimExpr& lhs, const DimExpr& rhs) {
  return std::visit(
      [&](const auto& lhs, const auto& rhs) {
        return IsLhsBeforeRhsStruct<std::decay_t<decltype(lhs)>,
                                    std::decay_t<decltype(rhs)>>::Call(lhs,
                                                                       rhs);
      },
      lhs.variant(),
      rhs.variant());
}

/*
 * Sort operands in DimExpr
 */
template <template <typename> class Op>
struct SortOperands {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Op<DimExpr>>();
    if (operands->size() == 1) {
      return operands->at(0);
    }
    bool is_sorted = IsSorted(operands);
    if (is_sorted) {
      return expr;
    }
    List<DimExpr> mut_operands{};
    mut_operands->insert(
        mut_operands->end(), operands->begin(), operands->end());
    std::sort(mut_operands->begin(), mut_operands->end(), &IsLhsBeforeRhs);
    return Op<DimExpr>{mut_operands};
  }

  bool IsSorted(const List<DimExpr>& operands) {
    PADDLE_ENFORCE_EQ(
        !operands->empty(),
        true,
        common::errors::InvalidArgument("input op is empty, please check!"));
    for (std::size_t i = 0; i < operands->size() - 1; ++i) {
      if (IsLhsBeforeRhs(operands->at(i + 1), operands->at(i))) {
        return false;
      }
    }
    return true;
  }
};

std::int64_t GetInteger(const DimExpr& expr) {
  if (expr.Has<Negative<DimExpr>>()) {
    const auto& integer = expr.Get<Negative<DimExpr>>()->data;
    PADDLE_ENFORCE_EQ(integer.Has<std::int64_t>(),
                      true,
                      common::errors::InvalidArgument(
                          "input expression's member `data` has no attribution "
                          "`int64_t`, maybe input dim is empty"));
    return -integer.Get<std::int64_t>();
  }
  PADDLE_ENFORCE_EQ(
      expr.Has<std::int64_t>(),
      true,
      common::errors::InvalidArgument(
          "input has no attribution `int64_t`, maybe input dim is empty"));
  return expr.Get<std::int64_t>();
}

template <template <typename> class Op, template <typename> class Inversed>
struct VisitEachInversableOperandStruct {
  template <typename DoEachT>
  static void Call(const DimExpr& expr,
                   const DoEachT& DoEach,
                   std::size_t depth,
                   bool is_inversed) {
    if (expr.Has<Op<DimExpr>>()) {
      const auto& [operands] = expr.Get<Op<DimExpr>>();
      for (const auto& operand : *operands) {
        Call(operand, DoEach, depth + 1, is_inversed);
      }
    } else if (expr.Has<Inversed<DimExpr>>()) {
      const auto& [operand] = expr.Get<Inversed<DimExpr>>();
      Call(operand->data, DoEach, depth, !is_inversed);
    } else {
      DoEach(expr, depth, is_inversed);
    }
  }
};

template <template <typename> class>
struct VisitEachOperandStruct;

template <>
struct VisitEachOperandStruct<Add>
    : public VisitEachInversableOperandStruct<Add, Negative> {};

template <>
struct VisitEachOperandStruct<Mul> {
  template <typename DoEachT>
  static void Call(const DimExpr& expr,
                   const DoEachT& DoEach,
                   std::size_t depth,
                   bool is_inversed) {
    if (expr.Has<Mul<DimExpr>>()) {
      const auto& [operands] = expr.Get<Mul<DimExpr>>();
      for (const auto& operand : *operands) {
        Call(operand, DoEach, depth + 1, false);
      }
    } else {
      DoEach(expr, depth, false);
    }
  }
};

template <>
struct VisitEachOperandStruct<Div> {
  template <typename DoEachT>
  static void Call(const DimExpr& expr,
                   const DoEachT& DoEach,
                   std::size_t depth,
                   bool is_inversed) {
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

template <>
struct VisitEachOperandStruct<Broadcast> {
  template <typename DoEachT>
  static void Call(const DimExpr& expr,
                   const DoEachT& DoEach,
                   std::size_t depth,
                   bool is_inversed) {
    if (expr.Has<Broadcast<DimExpr>>()) {
      const auto& [operands] = expr.Get<Broadcast<DimExpr>>();
      for (const auto& operand : *operands) {
        Call(operand, DoEach, depth + 1, false);
      }
    } else {
      DoEach(expr, depth, false);
    }
  }
};

template <template <typename> class Op, typename DoEachT>
void VisitEachOperand(const DimExpr& expr, const DoEachT& DoEach) {
  if (expr.Has<Op<DimExpr>>()) {
    VisitEachOperandStruct<Op>::Call(
        expr, DoEach, /*depth=*/0, /*is_inversed=*/false);
  } else {
    // Do nothing;
  }
}

template <template <typename> class Op>
bool HasNested(const DimExpr& expr) {
  bool has_nested = false;
  VisitEachOperand<Op>(
      expr, [&](const DimExpr& operand, std::size_t depth, bool is_negative) {
        has_nested = has_nested || (depth > 0);
      });
  return has_nested;
}

template <template <typename> class Op>
struct GetInversed {};

template <>
struct GetInversed<Add> {
  static DimExpr Call(const DimExpr& expr) { return Negative<DimExpr>(expr); }
};

template <>
struct GetInversed<Mul> {
  static DimExpr Call(const DimExpr& expr) {
    PADDLE_THROW(common::errors::Fatal(
        "Integer multiplication and integer division are not reciprocal."));
  }
};

template <>
struct GetInversed<Div> {
  static DimExpr Call(const DimExpr& expr) {
    PADDLE_THROW(common::errors::Fatal(
        "Integer division and integer multiplication are not reciprocal."));
  }
};

template <>
struct GetInversed<Broadcast> {
  static DimExpr Call(const DimExpr& expr) {
    PADDLE_THROW(common::errors::Fatal("Broadcast is not a group in math."));
  }
};

/*
 * Simplify Example:
 * Add(S0, Add(S1, S2)) =>
 * Add(S0, S1, S2)
 */
template <template <typename> class Op>
struct FlattenOperands {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    if (!HasNested<Op>(expr)) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    VisitEachOperand<Op>(
        expr,
        [&](const DimExpr& dim_expr, std::size_t depth, bool is_negative) {
          if (is_negative) {
            ret_operands->emplace_back(GetInversed<Op>::Call(dim_expr));
          } else {
            ret_operands->emplace_back(dim_expr);
          }
        });
    return Op<DimExpr>{ret_operands};
  }
};

template <template <typename> class Op>
struct FoldOperandTrait;

template <template <typename> class Op>
size_t GetConstDimCount(const List<DimExpr>& exprs) {
  std::size_t cnt = 0;
  for (const auto& expr : *exprs) {
    cnt += FoldOperandTrait<Op>::IsConstPattern(expr);
  }
  return cnt;
}

/*
 * Simplify Example:
 * Add(S0, 0) => S0
 */
template <template <typename> class Op>
struct FoldUnitConstant {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto [operands] = expr.Get<Op<DimExpr>>();
    if (GetConstDimCount<Op>(operands) == 0) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    for (const auto& operand : *operands) {
      if (FoldOperandTrait<Op>::IsUnitDimExpr(operand)) {
        continue;
      } else {
        ret_operands->emplace_back(operand);
      }
    }
    if (ret_operands->empty()) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(
          FoldOperandTrait<Op>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Op<DimExpr>{ret_operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

/*
 * Simplify Example:
 * Add(S0, 0, 1, 2) => Add(S0, 3)
 */
template <template <typename> class Op>
struct FoldConstants {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto [operands] = expr.Get<Op<DimExpr>>();
    if (GetConstDimCount<Op>(operands) <= 1) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    typename FoldOperandTrait<Op>::const_value_type const_dim =
        FoldOperandTrait<Op>::MakeUnit();
    for (const auto& operand : *operands) {
      if (FoldOperandTrait<Op>::IsConstPattern(operand)) {
        FoldOperandTrait<Op>::Accumulate(&const_dim, operand);
      } else {
        ret_operands->emplace_back(operand);
      }
    }
    if (!FoldOperandTrait<Op>::IsUnit(const_dim)) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(const_dim, &ret_operands);
    }
    if (ret_operands->empty()) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(
          FoldOperandTrait<Op>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Op<DimExpr>{ret_operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

template <>
struct FoldOperandTrait<Add> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return 0; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = *value + GetInteger(expr);
  }
  static bool IsUnit(const const_value_type& value) { return value == 0; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 0;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    if (lhs.Has<Negative<DimExpr>>()) {
      const auto& lhs_operand = lhs.Get<Negative<DimExpr>>()->data;
      return lhs_operand == rhs;
    }
    if (rhs.Has<Negative<DimExpr>>()) {
      const auto& [rhs_operand] = *rhs.Get<Negative<DimExpr>>();
      return rhs_operand == lhs;
    }
    return false;
  }
};

template <>
struct FoldOperandTrait<Max> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return INT64_MIN; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = std::max(*value, GetInteger(expr));
  }
  static bool IsUnit(const const_value_type& value) {
    return value == INT64_MIN;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == INT64_MIN;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

template <>
struct FoldOperandTrait<Min> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return INT64_MAX; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = std::min(*value, GetInteger(expr));
  }
  static bool IsUnit(const const_value_type& value) {
    return value == INT64_MAX;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == INT64_MAX;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

template <>
struct FoldOperandTrait<Mul> {
  using const_value_type = int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    return false;
  }

  static const_value_type MakeUnit() { return 1; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = *value * GetInteger(expr);
  }
  static bool IsUnit(const const_value_type& value) { return value == 1; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }

  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }
  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

template <>
struct FoldUnitConstant<Mul> {
  using dim_expr_type = Mul<DimExpr>;
  static bool IsZeroDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 0;
  }
  DimExpr Rewrite(const DimExpr& expr) {
    const auto [operands] = expr.Get<Mul<DimExpr>>();
    if (GetConstDimCount<Mul>(operands) == 0) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    for (const auto& operand : *operands) {
      if (FoldOperandTrait<Mul>::IsUnitDimExpr(operand)) {
        continue;
      } else if (IsZeroDimExpr(operand)) {
        // Mul(S0,0) => 0
        return DimExpr{0};
      } else {
        ret_operands->emplace_back(operand);
      }
    }
    if (ret_operands->empty()) {
      FoldOperandTrait<Mul>::MakeAndAppendDimExpr(
          FoldOperandTrait<Mul>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Mul<DimExpr>{ret_operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }
};

template <>
struct FoldOperandTrait<Broadcast> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    return false;
  }

  static const_value_type MakeUnit() { return 1; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    PADDLE_ENFORCE_EQ(expr.Has<std::int64_t>(),
                      true,
                      common::errors::InvalidArgument(
                          "Input constant `expr`(second argument) has no "
                          "attribution `int64_T`, please check"));
    std::int64_t expr_value = expr.Get<std::int64_t>();
    if (*value == 1) {
      *value = expr_value;
    } else if (expr_value != 1) {
      PADDLE_ENFORCE_EQ(*value,
                        expr_value,
                        common::errors::InvalidArgument(
                            "The value (%d) should be equal to expr "
                            "(%d) when they are both not 1.",
                            *value,
                            expr_value));
    } else {
      // do nothing.
    }
  }
  static bool IsUnit(const const_value_type& value) { return value == 1; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }
  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

/*
 * Simplify Example:
 * Div(S0, 1) => S0
 * Div(0, S0) => 0
 */
template <>
struct FoldUnitConstant<Div> {
  using dim_expr_type = Div<DimExpr>;
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static bool IsZeroDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 0;
  }

  DimExpr Rewrite(const DimExpr& expr) {
    const auto div_expr = expr.Get<Div<DimExpr>>();
    if (IsZeroDimExpr(div_expr->lhs)) {
      return DimExpr{0};
    }

    if (IsUnitDimExpr(div_expr->rhs)) {
      return div_expr->lhs;
    }
    return expr;
  }
};

/*
 * Simplify Example:
 * Mul(Div(1, S0), S0) => S0
 * Mul(S0, Div(1, S0) => S0
 */
template <template <typename> class Op>
struct FoldInversedPairToUnit {
  using dim_expr_type = Op<DimExpr>;

  struct SearchResult {
    std::size_t value_pos;
    std::size_t inverse_value_pos;
  };

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Op<DimExpr>>();
    const auto& opt_searched = SearchInversedPair(operands);
    if (!opt_searched.has_value()) {
      return expr;
    }
    const auto& [i, j] = opt_searched.value();
    List<DimExpr> ret_operands{};
    ret_operands->insert(ret_operands->end(),
                         operands->begin(),
                         std::next(operands->begin(), i));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), i + 1),
                         std::next(operands->begin(), j));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), j + 1),
                         operands->end());
    if (ret_operands->empty()) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(
          FoldOperandTrait<Op>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Op<DimExpr>{ret_operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }

  std::optional<SearchResult> SearchInversedPair(
      const List<DimExpr>& operands) {
    for (std::size_t i = 0; i < operands->size(); ++i) {
      for (std::size_t j = 0; j < operands->size(); ++j) {
        if (i == j) {
          continue;
        }
        if (FoldOperandTrait<Op>::IsInversedPair(operands->at(i),
                                                 operands->at(j))) {
          return SearchResult{i, j};
        }
      }
    }
    return std::nullopt;
  }
};

/*
 * Simplify Example:
 * Broadcast(2, S0) => 2
 */
struct FoldRedundantSymbolicBroadcast {
  using dim_expr_type = Broadcast<DimExpr>;

  struct MaxInt64 {
    std::int64_t value;
    int value_pos;
  };

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Broadcast<DimExpr>>();
    const auto& opt_max_int64 = SearchMaxInt64(operands);
    if (!opt_max_int64.has_value()) {
      return expr;
    }
    const auto& [value, i] = opt_max_int64.value();
    if (value != 1) {
      return value;
    }
    List<DimExpr> ret_operands{};
    ret_operands->insert(ret_operands->end(),
                         operands->begin(),
                         std::next(operands->begin(), i));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), i + 1),
                         operands->end());
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Broadcast<DimExpr>{ret_operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }

  std::optional<MaxInt64> SearchMaxInt64(const List<DimExpr>& operands) {
    std::optional<MaxInt64> ret;
    int operands_size = static_cast<int>(operands->size());
    for (int i = 0; i < operands_size; ++i) {
      const auto& expr = operands->at(i);
      if (!expr.Has<std::int64_t>()) {
        continue;
      }
      std::int64_t int64_value = expr.Get<std::int64_t>();
      if (ret.has_value()) {
        if (int64_value > 1) {
          if (ret.value().value > 1) {
            PADDLE_ENFORCE_EQ(
                ret.value().value,
                int64_value,
                common::errors::InvalidArgument(
                    "The value of return (%d) should be equal to expr (%d) of "
                    "operands at index (%d) when they are both > 1.",
                    ret.value().value,
                    int64_value,
                    i));
          }
          ret = MaxInt64{int64_value, i};
        }
      } else {
        ret = MaxInt64{int64_value, i};
      }
    }
    return ret;
  }
};

/*
 * Simplify Example:
 * Broadcast(S0,S0,S1) => Broadcast(S0,S1)
 */
struct FoldRedundantBroadcast {
  using dim_expr_type = Broadcast<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Broadcast<DimExpr>>();
    while (operands->size() > 1) {
      int pos_index = SearchSameIndex(operands);
      if (pos_index < 0) {
        break;
      }
      operands->erase(operands->begin() + pos_index);
    }
    if (operands->size() == 1) {
      return operands->at(0);
    } else {
      return Broadcast<DimExpr>{operands};
    }
    PADDLE_THROW(common::errors::Fatal("Dead code."));
  }

  int SearchSameIndex(const List<DimExpr>& operands) {
    for (std::size_t i = 0; i < operands->size(); ++i) {
      for (std::size_t j = 0; j < operands->size(); ++j) {
        if (i == j) {
          continue;
        }
        if (operands->at(i) == operands->at(j)) {
          return i;
        }
      }
    }
    return -1;
  }
};

/*
 * Simplify Example:
 * Broadcast(S0, Mul(S0, S1)) => Mul(S0, S1)
 */
struct SimplifyBroadcast {
  using dim_expr_type = Broadcast<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    auto [operands] = expr.Get<Broadcast<DimExpr>>();
    while (operands->size() > 1) {
      int pos_erasable = SearchErasable(operands);
      if (pos_erasable < 0) break;
      operands->erase(operands->begin() + pos_erasable);
    }
    if (operands->size() == 1) {
      return operands->at(0);
    } else {
      return Broadcast<DimExpr>{operands};
    }
  }

  bool IsLhsGreatThanRhs(const DimExpr& lhs, const DimExpr& rhs) {
    auto LhsOperandsVisitor = common::Overloaded{
        [&](const Mul<DimExpr>& mul) {
          bool lhs_great_than_rhs = false;
          for (const auto& expr : *mul.operands) {
            if (expr == rhs)
              lhs_great_than_rhs = true;
            else if (!expr.isa<std::int64_t>() && !expr.isa<std::string>())
              return false;
          }
          return lhs_great_than_rhs;
        },
        [&](const Add<DimExpr>& add) {
          bool lhs_great_than_rhs = false;
          for (const auto& expr : *add.operands) {
            if (expr == rhs)
              lhs_great_than_rhs = true;
            else if (!expr.isa<std::int64_t>() && !expr.isa<std::string>())
              return false;
          }
          return lhs_great_than_rhs;
        },
        [&](const auto& lhs) { return false; }};
    return std::visit(LhsOperandsVisitor, lhs.variant());
  }

  int SearchErasable(const List<DimExpr>& operands) {
    for (std::size_t i = 0; i < operands->size() - 1; ++i) {
      for (std::size_t j = i + 1; j < operands->size(); ++j) {
        if (IsLhsGreatThanRhs(operands->at(i), operands->at(j))) {
          return j;
        } else if (IsLhsGreatThanRhs(operands->at(j), operands->at(i))) {
          return i;
        }
      }
    }
    return -1;
  }
};

std::pair<List<DimExpr>, List<DimExpr>> GetDivExprNumAndDem(
    const DimExpr& expr) {
  const auto div_expr = expr.Get<Div<DimExpr>>();
  const auto lhs = div_expr->lhs;
  const auto rhs = div_expr->rhs;
  PADDLE_ENFORCE_NE(rhs,
                    symbol::DimExpr{0},
                    common::errors::InvalidArgument(
                        "In SimplifyDiv Pass dem cannot be zero."));
  auto GetOperandsList = [](const DimExpr& expr) -> List<DimExpr> {
    return expr.Has<Mul<DimExpr>>() ? expr.Get<Mul<DimExpr>>().operands
                                    : List<DimExpr>{expr};
  };
  return {GetOperandsList(lhs), GetOperandsList(rhs)};
}

std::pair<int64_t, int64_t> SimplifiedConstRational(int64_t num, int64_t dem) {
  if (num == 0 && dem == 0) {
    PADDLE_THROW(
        common::errors::InvalidArgument("num and dem cannot both be zero."));
  }
  std::int64_t gcd = std::gcd(num, dem);
  return std::pair{num / gcd, dem / gcd};
}

DimExpr SetDivSimpliedRes(const List<DimExpr>& lhs_operands,
                          const List<DimExpr>& rhs_operands) {
  if (lhs_operands->empty() && rhs_operands->empty()) {
    return DimExpr{1};
  }

  if (rhs_operands->empty()) {
    return lhs_operands->size() == 1 ? lhs_operands->at(0)
                                     : Mul<DimExpr>{lhs_operands};
  }
  if (lhs_operands->empty()) {
    return Div<DimExpr>{1,
                        rhs_operands->size() == 1 ? rhs_operands->at(0)
                                                  : Mul<DimExpr>{rhs_operands}};
  }
  DimExpr last_lhs = lhs_operands->size() == 1 ? lhs_operands->at(0)
                                               : Mul<DimExpr>{lhs_operands};
  DimExpr last_rhs = rhs_operands->size() == 1 ? rhs_operands->at(0)
                                               : Mul<DimExpr>{rhs_operands};

  return Div<DimExpr>{last_lhs, last_rhs};
}

/*
 * Simplify Example:
 * Div(Mul(S0, 8), 4) => Mul(S0, 2)
 */
template <>
struct FoldConstants<Div> {
  using dim_expr_type = Div<DimExpr>;

  std::pair<List<DimExpr>, List<DimExpr>> FilterIntOperands(
      const List<DimExpr>& operands) {
    List<DimExpr> NoIntSymList{}, IntSymList{};
    for (const auto& operand : *operands) {
      if (operand.Has<std::int64_t>()) {
        IntSymList->emplace_back(operand);
      } else {
        NoIntSymList->emplace_back(operand);
      }
    }
    return {NoIntSymList, IntSymList};
  }

  DimExpr Rewrite(const DimExpr& expr) {
    auto [lhs_operands, rhs_operands] = GetDivExprNumAndDem(expr);
    auto [lhs_no_int_list, lhs_int_list] = FilterIntOperands(lhs_operands);
    auto [rhs_no_int_list, rhs_int_list] = FilterIntOperands(rhs_operands);
    if (!lhs_int_list->empty() && !rhs_int_list->empty()) {
      PADDLE_ENFORCE_EQ(lhs_int_list->size() == rhs_int_list->size() &&
                            lhs_int_list->size() == 1,
                        true,
                        common::errors::InvalidArgument(
                            "Int should be fold by FoldUnitConstant<Mul>"));
      auto [lhs_int, rhs_int] =
          SimplifiedConstRational(lhs_int_list->at(0).Get<int64_t>(),
                                  rhs_int_list->at(0).Get<int64_t>());
      if (lhs_int != 1) {
        lhs_no_int_list->emplace_back(lhs_int);
      }
      if (rhs_int != 1) {
        rhs_no_int_list->emplace_back(rhs_int);
      }
      return SetDivSimpliedRes(lhs_no_int_list, rhs_no_int_list);
    } else {
      return expr;
    }
  }
};

/*
 * Simplify Example:
 * Div(Mul(S0, S1), S1) => S0
 */
struct SimplifyDiv {
  using dim_expr_type = Div<DimExpr>;

  std::pair<List<DimExpr>, List<DimExpr>> FoldSameOperand(
      const List<DimExpr>& lhsList, const List<DimExpr>& rhsList) {
    auto sorted_lhs = lhsList;
    auto sorted_rhs = rhsList;
    List<DimExpr> result_lhs{};
    List<DimExpr> result_rhs{};
    std::sort(sorted_lhs->begin(), sorted_lhs->end(), &IsLhsBeforeRhs);
    std::sort(sorted_rhs->begin(), sorted_rhs->end(), &IsLhsBeforeRhs);
    size_t i = 0, j = 0;
    while (i < sorted_lhs->size() && j < sorted_rhs->size()) {
      if (sorted_lhs->at(i) == sorted_rhs->at(j)) {
        ++i;
        ++j;
      } else if (IsLhsBeforeRhs(sorted_lhs->at(i), sorted_rhs->at(j))) {
        result_lhs->push_back(sorted_lhs->at(i));
        ++i;
      } else {
        result_rhs->push_back(sorted_rhs->at(j));
        ++j;
      }
    }
    while (i < sorted_lhs->size()) {
      result_lhs->push_back(sorted_lhs->at(i));
      ++i;
    }
    while (j < sorted_rhs->size()) {
      result_rhs->push_back(sorted_rhs->at(j));
      ++j;
    }
    return std::make_pair(result_lhs, result_rhs);
  }

  DimExpr Rewrite(const DimExpr& expr) {
    auto [lhs_operands, rhs_operands] = GetDivExprNumAndDem(expr);
    auto [new_lhs_operands, new_rhs_operands] =
        FoldSameOperand(lhs_operands, rhs_operands);
    return SetDivSimpliedRes(new_lhs_operands, new_rhs_operands);
  }
};

template <typename PassT>
void DoPass(bool* rewritten, DimExpr* expr) {
  const auto old_expr = *expr;
  *expr = TrySimplifyPass<PassT>(*expr);
  *rewritten = *rewritten || (old_expr != *expr);
}

DimExpr Simplify(const DimExpr& expr) {
  DimExpr ret = expr;
  for (bool keep_rewrite = true; keep_rewrite;) {
    keep_rewrite = false;
    const DimExpr expr_before_run_pipeline = ret;
    DoPass<SimplifyOneOperand<Negative>>(&keep_rewrite, &ret);
    DoPass<SimplifyUnitOneOperand<Negative>>(&keep_rewrite, &ret);
    DoPass<SimplifyDoubleNeg>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Add>>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Mul>>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Div>>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Broadcast>>(&keep_rewrite, &ret);
    DoPass<SortOperands<Add>>(&keep_rewrite, &ret);
    DoPass<SortOperands<Mul>>(&keep_rewrite, &ret);
    DoPass<SortOperands<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FlattenOperands<Add>>(&keep_rewrite, &ret);
    DoPass<FlattenOperands<Mul>>(&keep_rewrite, &ret);
    DoPass<FlattenOperands<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Add>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Mul>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Div>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Add>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Mul>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Div>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Max>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Min>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FoldInversedPairToUnit<Add>>(&keep_rewrite, &ret);
    DoPass<FoldRedundantBroadcast>(&keep_rewrite, &ret);
    DoPass<FoldRedundantSymbolicBroadcast>(&keep_rewrite, &ret);
    DoPass<SimplifyBroadcast>(&keep_rewrite, &ret);
    DoPass<SimplifyDiv>(&keep_rewrite, &ret);
    if (expr_before_run_pipeline == ret) break;
  }
  return ret;
}

}  // namespace

DimExpr SimplifyDimExpr(const DimExpr& expr) { return Simplify(expr); }

}  // namespace symbol

namespace symbol {

namespace {

class SubstituteDimExprHelper final {
 public:
  explicit SubstituteDimExprHelper(
      const std::unordered_map<DimExpr, DimExpr>& pattern_to_replacement)
      : pattern_to_replacement_(pattern_to_replacement) {}

  std::optional<DimExpr> Substitute(const DimExpr& dim_expr) {
    auto iter = pattern_to_replacement_.find(dim_expr);
    if (iter != pattern_to_replacement_.end()) return iter->second;
    return std::visit([&](const auto& impl) { return SubstituteImpl(impl); },
                      dim_expr.variant());
  }

 private:
  std::optional<DimExpr> SubstituteImpl(const std::int64_t& value) {
    // `Substitute` has handled the case that `value` is matched.
    return std::nullopt;
  }
  std::optional<DimExpr> SubstituteImpl(const std::string& value) {
    // `Substitute` has handled the case that `value` is matched.
    return std::nullopt;
  }

  std::optional<DimExpr> SubstituteImpl(const Negative<DimExpr>& dim_expr) {
    return SubstituteUnary(dim_expr);
  }

  template <typename T>
  std::optional<DimExpr> SubstituteUnary(const T& dim_expr) {
    const auto& operand = dim_expr->data;
    const auto& substituted_operand = Substitute(operand);
    if (!substituted_operand.has_value()) return std::nullopt;
    return T{substituted_operand.value()};
  }

  std::optional<DimExpr> SubstituteImpl(const Add<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Mul<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Div<DimExpr>& dim_expr) {
    return SubstituteBinary(dim_expr);
  }

  template <typename T>
  std::optional<DimExpr> SubstituteBinary(const T& dim_expr) {
    const auto& lhs = dim_expr->lhs;
    const auto& rhs = dim_expr->rhs;
    const auto& substituted_lhs = Substitute(lhs);
    const auto& substituted_rhs = Substitute(rhs);
    if (!substituted_lhs.has_value() && !substituted_rhs.has_value())
      return std::nullopt;
    if (!substituted_lhs.has_value()) return T{lhs, substituted_rhs.value()};
    if (!substituted_rhs.has_value()) return T{substituted_lhs.value(), rhs};
    return T{substituted_lhs.value(), substituted_rhs.value()};
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

  template <template <typename> class OpT>
  std::optional<DimExpr> SubstituteVariadic(const OpT<DimExpr>& dim_expr) {
    auto opt_result = SubstituteEntireExpr(dim_expr);

    if (opt_result.has_value()) {
      if (opt_result->template isa<OpT<DimExpr>>()) {
        auto new_result = SubstituteSubOperands(
            opt_result->template dyn_cast<OpT<DimExpr>>());
        if (new_result.has_value()) {
          return new_result;
        }
      }
      return opt_result;
    } else {
      return SubstituteSubOperands(dim_expr);
    }
  }

  template <template <typename> class OpT>
  std::optional<DimExpr> SubstituteEntireExpr(const OpT<DimExpr>& dim_expr) {
    const auto& operands = *(dim_expr.operands);
    List<DimExpr> substituted_operands{};
    size_t replace_cnt = 0;
    for (const auto& operand : operands) {
      const auto& substituted_operand = Substitute(operand);
      replace_cnt += substituted_operand.has_value();
      substituted_operands->push_back(substituted_operand.has_value()
                                          ? substituted_operand.value()
                                          : operand);
    }
    if (replace_cnt == 0) return std::nullopt;
    return SimplifyDimExpr(OpT<DimExpr>{substituted_operands});
  }

  template <template <typename> class OpT>
  std::optional<DimExpr> SubstituteSubOperands(const OpT<DimExpr>& dim_expr) {
    const std::unordered_set<DimExpr> operands_set{dim_expr.operands->begin(),
                                                   dim_expr.operands->end()};

    auto CanReplaceSubOperands = [&operands_set](const OpT<DimExpr>& dim_expr) {
      for (const auto& operand : *dim_expr.operands) {
        if (operands_set.find(operand) == operands_set.end()) return false;
      }
      return true;
    };

    for (const auto& kv : pattern_to_replacement_) {
      if (!kv.first.isa<OpT<DimExpr>>()) continue;
      const auto& dim_expr_pattern = kv.first.dyn_cast<OpT<DimExpr>>();
      if (!CanReplaceSubOperands(dim_expr_pattern)) continue;

      List<DimExpr> ret_operands{kv.second};
      for (const auto& operand : operands_set) {
        if (std::find(dim_expr_pattern.operands->begin(),
                      dim_expr_pattern.operands->end(),
                      operand) == dim_expr_pattern.operands->end()) {
          ret_operands->push_back(operand);
        }
      }
      return SimplifyDimExpr(OpT<DimExpr>{ret_operands});
    }

    return std::nullopt;
  }

  std::unordered_map<DimExpr, DimExpr> pattern_to_replacement_;
};

}  // namespace

DimExpr SubstituteDimExpr(
    const DimExpr& dim_expr,
    const std::unordered_map<DimExpr, DimExpr>& pattern_to_replacement) {
  const auto& opt_replaced =
      SubstituteDimExprHelper(pattern_to_replacement).Substitute(dim_expr);
  return opt_replaced.has_value() ? opt_replaced.value() : dim_expr;
}

}  // namespace symbol

namespace symbol {

IR_API int GetDimExprPriority(const DimExpr& dim_expr) {
  return std::visit(common::Overloaded{
                        [&](std::int64_t) { return 0; },
                        [&](const std::string&) { return 1; },
                        [&](const Negative<DimExpr>&) { return 2; },
                        [&](const Add<DimExpr>&) { return 2; },
                        [&](const Mul<DimExpr>&) { return 2; },
                        [&](const Div<DimExpr>&) { return 2; },
                        [&](const Max<DimExpr>&) { return 2; },
                        [&](const Min<DimExpr>&) { return 2; },
                        [&](const Broadcast<DimExpr>&) { return 2; },
                    },
                    dim_expr.variant());
}

IR_API PriorityComparisonStatus CompareDimExprPriority(const DimExpr& lhs,
                                                       const DimExpr& rhs) {
  int lhs_priority = GetDimExprPriority(lhs);
  int rhs_priority = GetDimExprPriority(rhs);

  if (lhs_priority != rhs_priority) {
    return lhs_priority < rhs_priority ? PriorityComparisonStatus::HIGHER
                                       : PriorityComparisonStatus::LOWER;
  }

  auto CompareForEqualPriority = common::Overloaded{
      [](const std::string& lhs, const std::string& rhs) {
        if (lhs.size() != rhs.size()) {
          return lhs.size() < rhs.size() ? PriorityComparisonStatus::HIGHER
                                         : PriorityComparisonStatus::LOWER;
        }
        int compare_result = lhs.compare(rhs);
        if (compare_result == 0)
          return PriorityComparisonStatus::EQUAL;
        else if (compare_result < 0)
          return PriorityComparisonStatus::HIGHER;
        else
          return PriorityComparisonStatus::LOWER;
      },
      [](const auto& lhs, const auto& rhs) {
        return PriorityComparisonStatus::EQUAL;
      }};
  return std::visit(CompareForEqualPriority, lhs.variant(), rhs.variant());
}

DimExprCompareResult Compare(const DimExpr& lhs, const DimExpr& rhs) {
  auto CompareFunc = common::Overloaded{
      [](const int& lhs, const int& rhs) {
        return lhs == rhs ? DimExprCompareResult::EQ
                          : (lhs < rhs ? DimExprCompareResult::LT
                                       : DimExprCompareResult::GT);
      },
      [](const Add<DimExpr>& lhs, const Add<DimExpr>& rhs) {
        DimExpr simplified_result =
            SimplifyDimExpr(DimExpr{lhs} - DimExpr{rhs});
        if (simplified_result.isa<int64_t>()) {
          int64_t const_result = simplified_result.dyn_cast<int64_t>();
          return const_result == 0  ? DimExprCompareResult::EQ
                 : const_result < 0 ? DimExprCompareResult::LT
                                    : DimExprCompareResult::GT;
        }
        return DimExprCompareResult::UNKNOWN;
      },
      [](const auto& lhs, const auto& rhs) {
        return DimExprCompareResult::UNKNOWN;
      }};
  return std::visit(CompareFunc, lhs.variant(), rhs.variant());
}

}  // namespace symbol

namespace symbol {
namespace {

void CollectUnaryDimExprSymbolsImpl(const DimExpr& dim_expr,
                                    std::unordered_set<std::string>* ret) {
  std::unordered_set<std::string> symbols = CollectDimExprSymbols(dim_expr);
  ret->insert(symbols.begin(), symbols.end());
}

void CollectListDimExprSymbolsImpl(const List<DimExpr>& dim_exprs,
                                   std::unordered_set<std::string>* ret) {
  for (const auto& dim_expr : *dim_exprs) {
    std::unordered_set<std::string> symbols = CollectDimExprSymbols(dim_expr);
    ret->insert(symbols.begin(), symbols.end());
  }
}
}  // namespace

std::unordered_set<std::string> CollectDimExprSymbols(const DimExpr& dim_expr) {
  std::unordered_set<std::string> symbols;
  // clang-format off
  auto lambdas = common::Overloaded{
      [&](std::int64_t dim_expr) { return; },
      [&](const std::string& dim_expr) { symbols.insert(dim_expr); },
      [&](const Negative<DimExpr>& dim_expr) {
        CollectUnaryDimExprSymbolsImpl(dim_expr->data, &symbols);
      },
      [&](const Add<DimExpr>& dim_expr) {
        CollectListDimExprSymbolsImpl(dim_expr.operands, &symbols);
      },
      [&](const Mul<DimExpr>& dim_expr) {
        CollectListDimExprSymbolsImpl(dim_expr.operands, &symbols);
      },
      [&](const Div<DimExpr>& dim_expr) {
        CollectUnaryDimExprSymbolsImpl(dim_expr->lhs, &symbols);
        CollectUnaryDimExprSymbolsImpl(dim_expr->rhs, &symbols);
      },
      [&](const Max<DimExpr>& dim_expr) {
        CollectListDimExprSymbolsImpl(dim_expr.operands, &symbols);
      },
      [&](const Min<DimExpr>& dim_expr) {
        CollectListDimExprSymbolsImpl(dim_expr.operands, &symbols);
      },
      [&](const Broadcast<DimExpr>& dim_expr) {
        CollectListDimExprSymbolsImpl(dim_expr.operands, &symbols);
      }};
  // clang-format on
  std::visit(lambdas, dim_expr.variant());
  return symbols;
}

}  // namespace symbol
