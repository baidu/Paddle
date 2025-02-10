// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace common {

Expr IndiceToAbsOffset(const std::vector<Expr> &shape,
                       const std::vector<Expr> &indices);
Expr IndiceToAbsOffset(const std::vector<int> &shape,
                       const std::vector<Expr> &indices);

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape,
                              int preceding_n_axis);

Expr CastIfNeeded(Expr body, Type type);

//! Substitute vars to other expressions.
//! @param expr The expression to do modification.
//! @param var_map The map from variables to the target expressions.
void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map);

//! Get a stack of forloops(For and PolyFor nodes) to a Store node target to \p
//! tensor_name
std::vector<Expr *> GetForloopStackToStore(Expr *expr,
                                           const std::string &tensor_name);

// make const
// @{
inline Expr make_const(int32_t x) { return Expr(static_cast<int32_t>(x)); }
inline Expr make_const(int64_t x) { return Expr(static_cast<int64_t>(x)); }
inline Expr make_const(bfloat16 x) { return Expr(static_cast<bfloat16>(x)); }
inline Expr make_const(float16 x) { return Expr(static_cast<float16>(x)); }
inline Expr make_const(float x) { return Expr(static_cast<float>(x)); }
inline Expr make_const(double x) { return Expr(static_cast<double>(x)); }
inline Expr make_const(bool x) { return Expr(static_cast<bool>(x)); }
// @}

//! maker for some general consts.
// @{
template <typename T = int32_t>
inline Expr make_zero() {
  return make_const(static_cast<T>(0));
}
template <typename T = int32_t>
inline Expr make_one() {
  return make_const(static_cast<T>(1));
}
inline Expr make_bool(bool x) {
  return cinn::common::make_shared<ir::UIntImm>(Bool(), x);
}
inline Expr make_bool(bool x, int lanes) {
  return cinn::common::make_shared<ir::UIntImm>(Bool(lanes), x);
}
// @}

/**
 * \brief Check all the tensors are unique in an expression.
 */
void CheckTensorUniqueInExpr(Expr expr);

std::vector<std::string> GatherItersToTensorProducer(
    const std::string &target_tensor_name, Expr *expr);

bool is_zero(Expr v);

bool MathEqual(const Expr &a, const Expr &b);

//! helper function to get a ir::Select node.
Expr select(Expr cond, Expr true_value, Expr false_value);

//! helper function to get the And of all the conditions.
Expr and_all(const std::vector<Expr> &conds);

//! helper function to get the Or of all the conditions.
Expr or_any(const std::vector<Expr> &conds);

//! Cast the expression \p e to type \type.
Expr cast(Expr e, Type type);

Expr max(Expr a, Expr b);

Expr min(Expr a, Expr b);

template <typename T>
Expr make_const(Type t, T v) {
  if (t.is_vector()) {
    if (t.is_int()) {
      return ir::Broadcast::Make(
          make_shared<ir::IntImm>(t.ElementOf(), static_cast<int64_t>(v)),
          t.lanes());
    } else if (t.is_uint()) {
      return ir::Broadcast::Make(
          make_shared<ir::UIntImm>(t.ElementOf(), static_cast<uint64_t>(v)),
          t.lanes());
    } else if (t.is_float()) {
      return ir::Broadcast::Make(
          make_shared<ir::FloatImm>(t.ElementOf(), static_cast<double>(v)),
          t.lanes());
    } else if (t.is_bool()) {
      return ir::Broadcast::Make(
          make_shared<ir::UIntImm>(t.ElementOf(), static_cast<bool>(v)),
          t.lanes());
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    if (t.is_int()) {
      return make_shared<ir::IntImm>(t, static_cast<int64_t>(v));
    } else if (t.is_uint()) {
      return make_shared<ir::UIntImm>(t, static_cast<uint64_t>(v));
    } else if (t.is_float()) {
      return make_shared<ir::FloatImm>(t, static_cast<double>(v));
    } else if (t.is_bool()) {
      return make_shared<ir::UIntImm>(t, static_cast<bool>(v));
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }
  return Expr();
}

template <typename FuncOp>
Expr FoldExpr(FuncOp func_op, const std::vector<Expr> &values) {
  Expr init_value;
  for (const Expr &val : values) {
    if (!init_value.defined()) {
      init_value = val;
    } else {
      init_value = func_op(val, init_value);
    }
  }
  return init_value;
}

inline bool IsIterExpr(const Expr &a, const Expr &b) {
  return a.As<ir::IterSplit>() || a.As<ir::IterSum>() ||
         b.As<ir::IterSplit>() || b.As<ir::IterSum>();
}

inline bool IsOne(const Expr &expr) {
  if (expr.is_constant() && expr.get_constant() == 1) {
    return true;
  }
  return false;
}
inline bool IsZero(const Expr &expr) {
  if (expr.is_constant() && expr.get_constant() == 0) {
    return true;
  }
  return false;
}

/*!
 * \brief Apply func `fleaf` into each leaf node of `expr`.
 * which leaf node is the most outside node that has TNode type.
 * \param expr The expression to be applied.
 * \param fleaf The function to be applied.
 */
template <typename TNode, typename FLeaf>
inline void UnpackReduction(const ir::IndexExpr &expr, FLeaf fleaf) {
  if (const TNode *node = expr.As<TNode>()) {
    UnpackReduction<TNode, FLeaf>(node->a(), fleaf);
    UnpackReduction<TNode, FLeaf>(node->b(), fleaf);
  } else {
    fleaf(expr);
  }
}

/*!
 * \brief Flatten the expression into a vector of expressions splited by `Add`
 * or `Mul`.
 *
 * For example (Add):
 * 1. `S0 + S1` ==> {S0, S1}
 * 2. `S0 + S1 * S2` ==> {S0, S1 * S2}
 * 3. `S0 + S1 * (S2 + S3)` ==> {S0, S1 * (S2 + S3)}
 *
 * \param lhs The left hand side expression to be compared.
 * \param rhs The right hand side expression to be compared.
 * \return A boolean value indicating whether the priority of `lhs` is higher
 * than `rhs`.
 */
template <typename T>
inline std::vector<ir::IndexExpr> GetFlattenExprs(const ir::IndexExpr &expr) {
  std::vector<ir::IndexExpr> result;
  auto fcollect = [&](ir::IndexExpr val) { result.push_back(val); };
  UnpackReduction<T>(expr, fcollect);
  return result;
}

/*!
 * \brief Compare the priority of the two expressions. this func follows the
 * above rules:
 * 1. if lhs = var, rhs = const,    return true;
 * 2. if lhs = const, rhs = var,    return false;
 * 3. if lhs = var, rhs = var,      return lhs_var_name <= lhs_var_name;
 * 4. if lhs.length > rhs.length,   return true;
 * 5. if lhs.length == rhs.length,  return lhs_type <= rhs_type; (Add < Mul <
 * Div < Mod)
 * 6. if lhs.length < rhs.length    return false;
 *
 * For example:
 * 1. `ComparePriority(S0, 2)` return true;
 * 2. `ComparePriority(S0, S0)` return true;
 * 2. `ComparePriority(S0, S1)` return false;
 * 3. `ComparePriority(S0, S1 + 1)` return false;
 * 4. `ComparePriority(S0 % 2, S1 + 1)` return false;
 *
 * \param lhs The left hand side expression to be compared.
 * \param rhs The right hand side expression to be compared.
 * \return A boolean value indicating whether the priority of `lhs` is higher
 * than `rhs`.
 */
bool ComparePriority(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs);

/*!
 * \brief Determines whether there are sub-parts in the `expr` that can be
 * simplified by `Add` operation with the input `symbol`. If true is returned,
 * the operation will be attempted on each subpart in outer
 * `SimplifySymbolicAdd` function.
 *
 * For example:
 * 1. `IsSumPartialBySymbol(5, S0)` return false;
 * 2. `IsSumPartialBySymbol(S0, S0)` return true;
 * 3. `IsSumPartialBySymbol(S0 + S1, S1)` return true;
 * 4. `IsSumPartialBySymbol(S0 * 5 + S1, S0)` return true;
 * 5. `IsSumPartialBySymbol(S0 / 3, S0)` return true;
 * 6. `IsSumPartialBySymbol(S0 / 3 + S1, S0)` return true;
 * 7. `IsSumPartialBySymbol(S0 % 3, S0)` return false;
 *
 * Note: For performance reasons, special patterns will not be matched here.
 * This can be allowed for extreme optimization.
 * For example:
 * `IsSumPartialBySymbol((S0 + S1 / 5 * 25) / 5, S1 % 5)` return false;
 *
 * \param expr The expression to be checked.
 * \param symbol  The symbol to be checked.
 * \return True means there are sub-parts in the `expr` that can be simplified.
 */
bool IsSumPartialBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol);

/*!
 * \brief Simplify the `lhs` by symbol `sym`. Usually run after
 * `IsSumPartialBySymbol`
 *
 * \param lhs The expression to be simplified.
 * \param sym  The symbol to be checked.
 *    it may be `i, j ..` or  `S0, S1 ..` or other symbolic expr.
 * \param outer_mul_factor The scale of symbolic expr.
 *    e.g. `S0 * 4` ===> sym == S0, outer_mul_factor == 4
 * \return The expr after simplification.
 */
ir::IndexExpr SimplifySymbolicAdd(
    const ir::IndexExpr &lhs,
    const ir::IndexExpr &sym,
    const ir::IndexExpr &outer_mul_factor = ir::IndexExpr(1));

/*!
 * \brief Determines whether there are sub-parts in the `expr` that can be
 * simplified by `Div` operation with the input `symbol`. If true is returned,
 * the operation will be attempted on each subpart in outer
 * `SimplifySymbolicDivide` function.
 *
 * For example:
 * 1. `IsDivisibleBySymbol(5, S0, div)` return false;
 * 2. `IsDivisibleBySymbol(S0, S0, div)` return true;
 * 3. `IsDivisibleBySymbol(S0 + S1, S1, div)` return false;
 * 4. `IsDivisibleBySymbol(S0 * 5 + S1 * S2, S0, div)` return true;
 * 5. `IsDivisibleBySymbol(S0 / 3, S0, div)` return true;
 * 6. `IsDivisibleBySymbol(S0 * 4 / 3, S0, div)` return true;
 * 7. `IsDivisibleBySymbol(S0 % 3, S0, div)` return false;
 * 8. `IsDivisibleBySymbol(S0 / 3, S0, mod)` return false;
 *
 * \param expr The expression to be checked.
 * \param symbol  The symbol to be checked.
 * \param ty ty is `Mod` or `Div`.
 * \return True means there are sub-parts in the `expr` that can be simplified.
 * \note this func dont deal the corner case, please use `ProveDivisible` for
 * exact result. e.g. `IsDivisibleBySymbol(f % S0 - f, S0, div)` is false
 */
bool IsDivisibleBySymbol(const ir::IndexExpr &expr,
                         const ir::IndexExpr &symbol,
                         const ir::IrNodeTy &ty);

/*!
 * \brief Simplify the `lhs` by symbol `sym`. Usually run after
 * `IsDivisibleBySymbol`
 *
 * \param lhs The expression to be simplified.
 * \param sym  The symbol to be checked.
 *    it may be `i, j ..` or  `S0, S1 ..` or other symbolic expr.
 * \param ty ty is `Mod` or `Div`.
 * \return The expr after simplification.
 */
ir::IndexExpr SimplifySymbolicDivide(const ir::IndexExpr &lhs,
                                     const ir::IndexExpr &sym,
                                     const ir::IrNodeTy &ty);

/*!
 * \brief Determine whether `lhs` is divisible by `rhs`, regardless of whether
 * `rhs` is a constant or a symbol.
 * \param lhs lhs is dividend.
 * \param rhs rhs is divisor.
 * \return A boolean value indicating whether the `lhs` is divisible by `rhs`
 */
bool ProveDivisible(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs);

/*!
 * \brief Judge whether `candidate` is a negated index expression.
 * \param candidate The expression to be checked.
 * \param expr The positive part
 * \return A boolean value indicating whether `candidate` is negative.
 */
bool IsNegatedIndexExpr(const ir::IndexExpr &candidate,
                        ir::IndexExpr &expr);  // NOLINT

/*!
 * \brief Construct index expression by node type with or without simplify.
 * \param ty The node type of index expression.
 * \param lhs left operand.
 * \param rhs right operand.
 * \param simplify_flag Whether to simplify the result.
 * \return The constructed index expression.
 */
ir::IndexExpr ConstructIndexExprByNodeType(const ir::IrNodeTy &ty,
                                           const ir::IndexExpr &lhs,
                                           const ir::IndexExpr &rhs,
                                           bool simplify_flag = true);

/*!
 * \brief Change the sequence of `Div` and `Mod` in index expression.
 * Mathematical formula: `(a / b) % c = (a % (b * c)) / b`
 * For example:
 * 1. i / 4 % 8 => i % 32 / 4
 * 2. i / S0 % S1 => i % (S0 * S1) / S0
 * 3. (i * 32 + j) / 4 % 8 => (i * 32 + j) % 32 / 4
 *
 * \param expr The `IndexExpr` to be change
 * \return `IndexExpr` after change.
 */
ir::IndexExpr ChangeSeqOfDivMod(const ir::IndexExpr &expr);
enum IndexType {
  kInvalid = 0,  // invalid expr
  kValid = 1,    // valid expr
  kLoad = 2,     // exist Load
  kCast = 3      // exist cast
};

/*!
 * \brief Judge type of `expr` is valid type of `IndexExpr` or not.
 * \param expr The expression to be checked.
 * \return A enum IndexType value indicating whether the type of `expr` is valid
 * IndexExpr.
 *
 * Note: Although load and cast are also legal IndexExpr, we need to know this
 * information in some scenarios.
 */
IndexType VerifyIndex(const ir::Expr &expr);

/*!
 * \brief The multiplication in rhs is broken down and each sub-part is
 * independently determined to be divisible.
 * \param lhs The dividend.
 * \param rhs The divisor.
 * \param ty  ty is `Mod` or `Div`.
 * \return A optional index expression indicating whether the `lhs`
 * is divisible, nullopt indicating not divisible.
 *
 * For example:
 * 1. i * S0 * S1 * S2 / (S0 * S1) ==> i / S2
 * 2. i * S0 * S1 / S0 ==> i * S1
 * 3. i * S0 / (S0 + 1) ==> nullopt
 */
std::optional<ir::IndexExpr> DivByPartMul(const ir::IndexExpr &lhs,
                                          const ir::IndexExpr &rhs,
                                          ir::IrNodeTy ty);

/*!
 * \brief Simplify complex modulo expressions.
 * \param lhs The dividend.
 * \param rhs The divisor.
 * \return A optional index expression indicating whether simplified
 *
 * For example:
 * 1. (i / S0 * S0 + i % (S0 * S1)) % S0 ==> i % S0
 * 2. (i / S0 * S0 * S1 + i % (S0 * S1 * S2)) % (S0 * S1) ==> i % (S0 * S1)
 * 3. i % (S0 * S1) % S0 ==> i % S0
 * 4. i * S0 * S1 % (S0 * S1) ==> 0
 */
std::optional<ir::IndexExpr> SimplifyComplexMod(const ir::IndexExpr &lhs,
                                                const ir::IndexExpr &rhs);
}  // namespace common
}  // namespace cinn
