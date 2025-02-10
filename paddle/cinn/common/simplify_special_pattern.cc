// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/common/simplify_special_pattern.h"
#include <list>
#include <optional>
#include <stack>
#include <unordered_map>
#include <vector>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/op/ir_operators.h"
namespace cinn {
namespace common {

static void MergeMulModInsertElements(
    const std::vector<ir::IndexExpr>& elems,
    std::list<ir::IndexExpr>* mult_exprs,
    std::list<std::pair<ir::IndexExpr, ir::IndexExpr>>* mod_exprs,
    ir::IndexExpr* no_opt_sum,
    bool* has_mult,
    bool* has_mod) {
  *has_mult = false;
  *has_mod = false;
  for (const ir::IndexExpr ele : elems) {
    auto mod_ptr = ele.As<ir::Mod>();
    auto mult_ptr = ele.As<ir::Mul>();
    if (mod_ptr) {
      *has_mod = true;
      mod_exprs->emplace_back(
          std::make_pair(std::move(mod_ptr->a().as_index()),
                         std::move(mod_ptr->b().as_index())));
    } else if (mult_ptr) {
      *has_mult = true;
      mult_exprs->emplace_back(ele);
    } else {
      *no_opt_sum = no_opt_sum->get() ? *no_opt_sum + ele : ele;
    }
  }
}

// (S0 + (S1 + S2 / (S3 * S4) * S3)) * S4 + S2 % (S3 * S4)
// ==> (S0 + S1 * S3) * S4 + S2
static std::optional<ir::IndexExpr> MergeMulModInner(
    const ir::IndexExpr& expr,
    const ir::IndexExpr& overall_mult,
    const ir::IndexExpr& mod_l_expr,
    const ir::IndexExpr& mod_r_expr) {
  // The multiplier must always remain divisible by the modulo right
  // operand. because the final hit condition is that the two are equal.
  if (!ProveDivisible(mod_r_expr, overall_mult)) return std::nullopt;
  if (auto mult_ptr = expr.As<ir::Mul>()) {
    return MergeMulModInner(mult_ptr->a().as_index(),
                            overall_mult * mult_ptr->b().as_index(),
                            mod_l_expr,
                            mod_r_expr);
  } else if (auto div_ptr = expr.As<ir::Div>()) {
    VLOG(5) << "---- DEBUG SpecialPattern: MergeMulModInner Start ----";
    VLOG(5) << "div_ptr_b: " << div_ptr->b().as_index();
    VLOG(5) << "overall_mult: " << overall_mult;
    VLOG(5) << "mod_r_expr: " << mod_r_expr;
    VLOG(5) << "div_ptr_a - mod_l_expr: "
            << div_ptr->a().as_index() - mod_l_expr;
    VLOG(5) << "ProveDivisible: "
            << ProveDivisible(div_ptr->a().as_index() - mod_l_expr, mod_r_expr);
    VLOG(5) << "div_ptr_a - mod_l_expr % overall_mult: "
            << div_ptr->a().as_index() % overall_mult;
    VLOG(5) << "---- DEBUG SpecialPattern: MergeMulModInner End ----";

    // f % (S0 * S1) / S0 * S0 + f % S0 ==> f % (S0 + S1),
    // because f - f % (S0 * S1) == f / (S0 * S1) * (S0 * S1) can be divisible
    // by S0.
    if (overall_mult == div_ptr->b().as_index() && overall_mult == mod_r_expr &&
        (ProveDivisible(div_ptr->a().as_index() - mod_l_expr, mod_r_expr) ||
         div_ptr->a().as_index() % overall_mult == mod_l_expr % mod_r_expr)) {
      // Found!
      return div_ptr->a().as_index();
    } else {
      return std::nullopt;
    }
  } else if (auto add_ptr = expr.As<ir::Add>()) {
    auto lhs = add_ptr->a().as_index();
    auto rhs = add_ptr->b().as_index();
    if (auto lhs_result =
            MergeMulModInner(lhs, overall_mult, mod_l_expr, mod_r_expr)) {
      return rhs * overall_mult + lhs_result.value();
    } else if (auto rhs_result = MergeMulModInner(
                   rhs, overall_mult, mod_l_expr, mod_r_expr)) {
      return lhs * overall_mult + rhs_result.value();
    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
}

ir::IndexExpr MergeMulMod(const ir::IndexExpr& base) {
  std::vector<ir::IndexExpr> elems = GetFlattenExprs<ir::Add>(base);
  std::list<ir::IndexExpr> mult_exprs;
  std::list<std::pair<ir::IndexExpr, ir::IndexExpr>> mod_exprs;
  ir::IndexExpr no_opt_sum;
  bool has_mult;
  bool has_mod;
  MergeMulModInsertElements(
      elems, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult, &has_mod);
  bool find_opt = false;
  auto search_mod_it = mod_exprs.begin();

  while (search_mod_it != mod_exprs.end()) {
    auto mult_it = mult_exprs.begin();
    bool inner_find_opt = false;
    while (mult_it != mult_exprs.end()) {
      auto ret = MergeMulModInner(*mult_it,
                                  ir::IndexExpr(1),
                                  search_mod_it->first,
                                  search_mod_it->second);
      if (!ret.has_value()) {
        ++mult_it;
        continue;
      }
      inner_find_opt = true;
      auto temp_mod_it = search_mod_it;
      ++search_mod_it;
      mod_exprs.erase(temp_mod_it);
      mult_exprs.erase(mult_it);
      std::vector<ir::IndexExpr> ret_elems =
          GetFlattenExprs<ir::Add>(ret.value());
      MergeMulModInsertElements(
          ret_elems, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult, &has_mod);
      if (has_mult) {
        search_mod_it = mod_exprs.begin();
      } else if (has_mod && search_mod_it == mod_exprs.end()) {
        search_mod_it--;
      }
      break;
    }
    find_opt = find_opt || inner_find_opt;
    if (!inner_find_opt) {
      ++search_mod_it;
    }
  }
  if (!find_opt) {
    return base;
  }
  for (const auto& it : mult_exprs) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + it : it;
  }

  for (const auto& it : mod_exprs) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + it.first % it.second
                                  : it.first % it.second;
  }
  return no_opt_sum;
}

// S0 / (S1 * S2) * S1 * S2 + S4 % (S1 * S2) ==> S0
// s.t. (S4 - S0) % (S1 * S2) == 0
std::optional<ir::IndexExpr> DivMulAddModCornerCase(const ir::IndexExpr& lhs,
                                                    const ir::IndexExpr& rhs) {
  auto lhsMul = lhs.As<ir::Mul>();
  auto rhsMod = rhs.As<ir::Mod>();
  if (!lhsMul || !rhsMod) return std::nullopt;

  // Why inner is lhs of Mul? because we sort by expr length, and the length of
  // inner is longer in this case.
  auto inner = lhsMul->a().as_index();
  auto mult_outer = lhsMul->b().as_index();

  // Calculate the outer multiplier
  while (true) {
    auto mulPtr = inner.As<ir::Mul>();
    if (mulPtr) {
      inner = mulPtr->a().as_index();
      mult_outer = mulPtr->b().as_index() * mult_outer;
    } else {
      break;
    }
  }

  // Check if the inner expression is a div
  auto innerDiv = inner.As<ir::Div>();
  if (!innerDiv) return std::nullopt;
  if (innerDiv->b().as_index() == rhsMod->b().as_index() &&
      innerDiv->b().as_index() == mult_outer) {
    // The second condition is to adapt to the dynamic shape:
    // f % (S0 * S1) / S0 * S0 + f % S0 ==> f % (S0 * S1)
    if (ProveDivisible(rhsMod->a().as_index() - innerDiv->a().as_index(),
                       mult_outer) ||
        innerDiv->a().as_index() % mult_outer == rhs)
      return innerDiv->a().as_index();
  }
  return std::nullopt;
}

// (S0 * 8 + S1 * 2 + S2) + (S1 * 2 + S2) * (-1) ===> 0
std::optional<ir::IndexExpr> AddMulCornerCase(
    const ir::IndexExpr& lhs,
    const ir::IndexExpr& rhs,
    const ir::IndexExpr& scale = ir::IndexExpr(1)) {
  auto rhsMul = rhs.As<ir::Mul>();
  if (!rhsMul) return std::nullopt;
  if (!rhsMul->b().is_constant()) return std::nullopt;

  auto scale_ = scale * rhsMul->b().as_index();
  auto flatten = GetFlattenExprs<ir::Add>(rhsMul->a());
  std::optional<ir::IndexExpr> resOpt;
  ir::IndexExpr res = lhs;
  for (const auto& expr : flatten) {
    if (auto innerMul = expr.As<ir::Mul>()) {
      if (!innerMul->b().is_constant()) return std::nullopt;
      auto resOpt = AddMulCornerCase(res, expr, scale_);
      if (!resOpt.has_value())
        return std::nullopt;
      else
        res = resOpt.value();
    } else {
      if (!IsSumPartialBySymbol(res, expr)) return std::nullopt;
    }
  }

  for (const auto& expr : flatten) {
    if (expr.As<ir::Mul>()) continue;
    if (expr.is_constant()) {
      res = res + expr * scale_;
      continue;
    }
    res = SimplifySymbolicAdd(res, expr, scale_);
  }
  return res;
}

// (S0 + S1 - (S0 + S1) % S2) % S2 == 0
// (S0 + S1 - (S0 + S1) % S2) / S2 == (S0 + S1) / S2
std::optional<ir::IndexExpr> SubModCornerCase(const ir::IndexExpr& lhs,
                                              const ir::IndexExpr& rhs,
                                              bool isDiv) {
  auto flatten = GetFlattenExprs<ir::Add>(lhs);

  if (flatten.size() < 2) return std::nullopt;

  for (int64_t i = 0, e = flatten.size(); i < e; ++i) {
    // Check if negation
    ir::IndexExpr beforeNegation = flatten[i];
    auto isNeg = IsNegatedIndexExpr(flatten[i], beforeNegation);

    // Check if the negation term is a mod
    auto innerMod = beforeNegation.As<ir::Mod>();
    if (!innerMod) continue;
    if (!ProveDivisible(innerMod->b().as_index(), rhs)) continue;

    // Check if the sum of all other terms is equal to the lhs of mod
    auto diff = ir::IndexExpr(0);
    for (int64_t j = 0; j < e; ++j)
      if (i != j) diff = diff + flatten[j];
    diff = isNeg ? diff - innerMod->a().as_index()
                 : diff + innerMod->a().as_index();
    if (IsZero(diff)) {
      if (!isDiv) return ir::IndexExpr(0);
      return isNeg ? innerMod->a().as_index() / rhs
                   : -(innerMod->a().as_index() / rhs);
    }

    // For simplify mod case: ((S0 * 256 + S1) % 512 - S1) % 32 == 0
    if (!isDiv) {
      auto diffBeforeNegation = diff;
      auto isDiffNeg = IsNegatedIndexExpr(diff, diffBeforeNegation);
      if (isDiffNeg) diff = diffBeforeNegation;
      auto flatten_diff = GetFlattenExprs<ir::Add>(diff);
      bool isDivisible = true;
      for (const auto& expr : flatten_diff) {
        if (!isDivisible) break;
        if (!ProveDivisible(expr, rhs)) isDivisible = false;
      }
      if (isDivisible) return ir::IndexExpr(0);
    }
  }
  return std::nullopt;
}

// (S0 + S1) / (S0 + S1) == 1
// (S0 + S1) % (S0 + S1) == 0
std::optional<ir::IndexExpr> MultiArgsDivAndMod(const ir::IndexExpr& lhs,
                                                const ir::IndexExpr& rhs,
                                                bool isDiv) {
  // TODO(liujinnan): Dealing with multiple relationships.
  if (lhs == rhs) {
    return isDiv ? ir::IndexExpr(1) : ir::IndexExpr(0);
  }
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyCornerCase(const ir::IndexExpr& expr) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm:
    case ir::IrNodeTy::_Var_:
      return expr;
    case ir::IrNodeTy::Add:
      return SimplifyAddCornerCase(expr.operand(0), expr.operand(1));
    case ir::IrNodeTy::Mul:
      return SimplifyMulCornerCase(expr.operand(0), expr.operand(1));
    case ir::IrNodeTy::Div:
      return SimplifyDivCornerCase(expr.operand(0), expr.operand(1));
    case ir::IrNodeTy::Mod:
      return SimplifyModCornerCase(expr.operand(0), expr.operand(1));
  }
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyAddCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  if (auto res = DivMulAddModCornerCase(lhs, rhs)) return res.value();
  if (auto res = AddMulCornerCase(lhs, rhs)) return res.value();
  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyMulCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyDivCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  if (auto res = SubModCornerCase(lhs, rhs, true)) return res.value();
  if (auto res = MultiArgsDivAndMod(lhs, rhs, true)) return res.value();
  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyModCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  if (auto res = SubModCornerCase(lhs, rhs, false)) return res.value();
  // Add other corner cases
  if (auto res = MultiArgsDivAndMod(lhs, rhs, false)) return res.value();
  return std::nullopt;
}

}  // namespace common
}  // namespace cinn
