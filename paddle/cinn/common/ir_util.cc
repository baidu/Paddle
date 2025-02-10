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

#include "paddle/cinn/common/ir_util.h"

#include <algorithm>
#include <stack>
#include <unordered_set>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/const_fold.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

namespace {

// ramp + scalar or broadcast
Expr RampRelatedMul(ir::Ramp *ramp, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().ElementOf(),
      Int(32),
      ::common::errors::InvalidArgument("The type of other should be int32."));
  PADDLE_ENFORCE_EQ(ramp->base.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The type of ramp->base should be int32."));
  PADDLE_ENFORCE_EQ(ramp->stride.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The type of ramp->stride should be int32."));
  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    PADDLE_ENFORCE_EQ(ramp->lanes,
                      other_broadcast->lanes,
                      ::common::errors::InvalidArgument(
                          "The lanes of ramp and other should be equal."));
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base * other, ramp->stride * other, ramp->lanes);
}

Expr RampRelatedMul(ir::Broadcast *broadcast, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().lanes(),
      1,
      ::common::errors::InvalidArgument("The lanes of other should be 1."));
  return ir::Broadcast::Make(broadcast->value * other, broadcast->lanes);
}
// ramp * ramp
Expr RampRelatedMul(ir::Ramp *ramp, ir::Ramp *other) {
  CINN_NOT_IMPLEMENTED
  return Expr();
}
// ramp + scalar
Expr RampRelatedAdd(ir::Ramp *ramp, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().ElementOf(),
      Int(32),
      ::common::errors::InvalidArgument("The type of other should be int32."));

  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    PADDLE_ENFORCE_EQ(ramp->lanes,
                      other_broadcast->lanes,
                      ::common::errors::InvalidArgument(
                          "The lanes of ramp and other should be equal."));
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base + other, ramp->stride, ramp->lanes);
}
Expr RampRelatedAdd(ir::Broadcast *broadcast, Expr other) {
  PADDLE_ENFORCE_EQ(
      other.type().lanes(),
      1,
      ::common::errors::InvalidArgument("The lanes of other should be 1."));
  return ir::Broadcast::Make(broadcast->value + other, broadcast->lanes);
}
// ramp + ramp
Expr RampRelatedAdd(ir::Ramp *ramp, ir::Ramp *other) {
  PADDLE_ENFORCE_NOT_NULL(
      ramp,
      ::common::errors::InvalidArgument("Ramp pointer should not be null."));
  PADDLE_ENFORCE_NOT_NULL(other,
                          ::common::errors::InvalidArgument(
                              "Other ramp pointer should not be null."));
  if (ramp->lanes == other->lanes) {
    Expr base_add = optim::ArithSimplify(ramp->base + other->base);
    Expr stride_add = optim::ArithSimplify(ramp->stride + other->stride);
    VLOG(2) << base_add;
    VLOG(2) << stride_add;
    return ir::Ramp::Make(base_add, stride_add, ramp->lanes);
  }
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr RampRelatedAdd(Expr a, Expr b) {
  auto *a_ramp = a.As<ir::Ramp>();
  auto *b_ramp = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (b->type().lanes() == 1 || b_broadcast)) {
    return RampRelatedAdd(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().lanes() == 1 || a_broadcast)) {
    return RampRelatedAdd(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() &&
             !b->type().is_vector()) {
    return a + b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedAdd(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedAdd(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedAdd(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    PADDLE_ENFORCE_EQ(
        a_broadcast->lanes,
        b_broadcast->lanes,
        ::common::errors::InvalidArgument(
            "The lanes of a_broadcast and b_broadcast should be equal."));
    return ir::Broadcast::Make(a_broadcast->value + b_broadcast->value,
                               a_broadcast->lanes);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

Expr RampRelatedMul(Expr a, Expr b) {
  auto *a_ramp = a.As<ir::Ramp>();
  auto *b_ramp = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (!b->type().is_vector() || b_broadcast)) {
    return RampRelatedMul(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().is_vector() || a_broadcast)) {
    return RampRelatedMul(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() &&
             !b->type().is_vector()) {
    return a * b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedMul(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedMul(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedMul(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    PADDLE_ENFORCE_EQ(
        a_broadcast->lanes,
        b_broadcast->lanes,
        ::common::errors::InvalidArgument(
            "The lanes of a_broadcast and b_broadcast should be equal."));
    return ir::Broadcast::Make(a_broadcast->value * b_broadcast->value,
                               a_broadcast->lanes);
  } else {
    VLOG(3) << "a,b: " << a << " " << b;
    CINN_NOT_IMPLEMENTED
  }
}

}  // namespace

Expr IndiceToAbsOffset(const std::vector<Expr> &shape,
                       const std::vector<Expr> &indices) {
  VLOG(3) << "Begin IndiceToAbsOffset";
  VLOG(3) << "shape is : " << utils::Join(shape, ",");
  VLOG(3) << "indices is : " << utils::Join(indices, ",");
  PADDLE_ENFORCE_LE(shape.size(),
                    indices.size(),
                    ::common::errors::InvalidArgument(
                        "The size of shape should be less than or "
                        "equal to the size of indices."));
  Expr res(0);

  for (int32_t i = 0; i < shape.size(); i++) {
    PADDLE_ENFORCE_EQ(
        shape[i].type() == Int(64) || shape[i].type() == Int(32),
        true,
        ::common::errors::InvalidArgument(
            "The shape data type currently supports only int32 or int64, but "
            "the current data type of shape[{}] is {}",
            i,
            shape[i].type()));

    Expr indice_cast = indices[i];
    optim::SimplifyCast(&indice_cast);
    res = RampRelatedAdd(RampRelatedMul(res, shape[i]), indice_cast);
    if (res.is_index()) {
      res = res.as_index().Normalize(ir::IndexExpr::OptLevel::Level2);
    } else {
      VLOG(8) << "**** expr is not index ****: " << res;
    }
  }

  return res;
}

Expr IndiceToAbsOffset(const std::vector<int> &shape,
                       const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return IndiceToAbsOffset(shape, indices);
}

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape,
                              int preceding_n_axis) {
  std::vector<Expr> indices;
  for (int i = 0; i < preceding_n_axis; i++) indices.push_back(shape[i]);
  return IndiceToAbsOffset(shape, indices);
}

namespace {

class SubstituteMutator : ir::IRMutator<ir::Expr *> {
 public:
  explicit SubstituteMutator(const std::map<const ir::_Var_ *, Expr> &var_map) {
    for (auto &item : var_map) {
      var_map_[item.first->name] = item.second;
    }
  }

  void operator()(ir::Expr *expr) { Visit(expr); }

 private:
  void Visit(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Var_ *op, ir::Expr *expr) override {
    auto it = var_map_.find(op->name);
    if (it == var_map_.end()) return;
    *expr = it->second;
  }

  Expr *expr_{};
  std::map<std::string, Expr> var_map_;
};

}  // namespace

void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map) {
  SubstituteMutator mutator(var_map);
  mutator(expr);
}

bool is_zero(Expr v) {
  v = AutoSimplify(v);
  auto *int_n = v.As<ir::IntImm>();
  auto *float_n = v.As<ir::FloatImm>();

  if (int_n) return int_n->value == 0;
  if (float_n) return float_n->value == 0.f;
  return false;
}

Expr CastIfNeeded(Expr body, Type type) {
  if (body.type() == type) return body;
  return ir::Cast::Make(type, body);
}

bool MathEqual(const Expr &a, const Expr &b) {
  auto c = a - b;
  c = AutoSimplify(c);
  return is_zero(c);
}

Expr select(Expr cond, Expr true_value, Expr false_value) {
  return ir::Select::Make(cond, true_value, false_value);
}

Expr and_all(const std::vector<Expr> &conds) {
  PADDLE_ENFORCE_NE(conds.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The conditions vector should not be empty."));
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::And::Make(res, conds[i]);
  }
  return res;
}

Expr or_all(const std::vector<Expr> &conds) {
  PADDLE_ENFORCE_NE(conds.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The conditions vector should not be empty."));
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::Or::Make(res, conds[i]);
  }
  return res;
}

void CheckTensorUniqueInExpr(Expr expr) {
  auto tensor_uniq = ir::ir_utils::CollectIRNodes(
      expr, [](const Expr *x) { return x->as_tensor(); });
  absl::flat_hash_map<std::string, const ir::_Tensor_ *> tensor_names;
  for (auto &t : tensor_uniq) {
    auto *tp = t.as_tensor();
    if (!tensor_names.count(tp->name)) {
      tensor_names[tp->name] = tp;
    } else {
      PADDLE_ENFORCE_EQ(
          tensor_names[tp->name],
          tp,
          ::common::errors::InvalidArgument(
              "Found tensor not unique, The original express is %d .", expr));
    }
  }
}

Expr cast(Expr e, Type type) {
  if (e.is_constant()) {
    if (type.is_bool()) {
      return Expr(static_cast<bool>(e.get_constant()));
    } else if (type.is_int(8)) {
      return Expr(static_cast<int8_t>(e.get_constant()));
    } else if (type.is_int(16)) {
      return Expr(static_cast<int16_t>(e.get_constant()));
    } else if (type.is_int(32)) {
      return Expr(static_cast<int32_t>(e.get_constant()));
    } else if (type.is_int(64)) {
      return Expr(static_cast<int64_t>(e.get_constant()));
    } else if (type.is_uint(8)) {
      return Expr(static_cast<uint8_t>(e.get_constant()));
    } else if (type.is_uint(16)) {
      return Expr(static_cast<uint16_t>(e.get_constant()));
    } else if (type.is_uint(32)) {
      return Expr(static_cast<uint32_t>(e.get_constant()));
    } else if (type.is_uint(64)) {
      return Expr(static_cast<uint64_t>(e.get_constant()));
    } else if (type.is_float(32)) {
      return Expr(static_cast<float>(e.get_constant()));
    } else if (type.is_float(64)) {
      return Expr(static_cast<double>(e.get_constant()));
    } else if (type.is_bfloat16()) {
      return Expr(static_cast<cinn::common::bfloat16>(e.get_constant()));
    } else if (type.is_float16()) {
      return Expr(static_cast<cinn::common::float16>(e.get_constant()));
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }

  return ir::Cast::Make(type, e);
}

std::vector<std::string> GatherItersToTensorProducer(
    const std::string &target_tensor_name, Expr *expr) {
  struct Visitor : public ir::IRMutator<> {
    std::vector<std::string> iters;
    const std::string &target_tensor_name;

    explicit Visitor(const std::string &target_tensor_name)
        : target_tensor_name(target_tensor_name) {}

    std::vector<std::string> operator()(Expr *expr) {
      ir::IRMutator<>::Visit(expr, expr);
      return iters;
    }

    void Visit(const ir::Store *op, Expr *expr) {
      if (op->tensor.as_tensor()->name == target_tensor_name) {
        PADDLE_ENFORCE_EQ(iters.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The iterators vector should be empty."));
        for (auto &e : for_stack) {
          auto *for_n = e->As<ir::For>();
          auto *polyfor_n = e->As<ir::PolyFor>();
          if (for_n) {
            iters.push_back(for_n->loop_var->name);
          } else {
            iters.push_back(polyfor_n->iterator->name);
          }
        }
      }
    }

    void Visit(const ir::For *op, Expr *expr) {
      for_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      for_stack.pop_back();
    }
    void Visit(const ir::PolyFor *op, Expr *expr) {
      for_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      for_stack.pop_back();
    }

    std::vector<Expr *> for_stack;
  };

  return Visitor(target_tensor_name)(expr);
}

std::vector<Expr *> GetForloopStackToStore(Expr *expr,
                                           const std::string &tensor_name) {
  VLOG(4) << "search store " << tensor_name << " in expr:\n";
  VLOG(4) << *expr;
  struct Mutator : public ir::IRMutator<> {
    std::vector<Expr *> forloop_stack;
    bool found{false};

    std::string tensor_name;

    explicit Mutator(const std::string &tensor_name)
        : tensor_name(tensor_name) {}

    std::vector<Expr *> operator()(Expr *expr) {
      ir::IRMutator<>::Visit(expr, expr);
      return forloop_stack;
    }

    void Visit(const ir::For *op, Expr *expr) {
      auto *node = expr->As<ir::For>();
      forloop_stack.push_back(expr);
      ir::IRMutator<>::Visit(&node->body, &node->body);
      if (!found) forloop_stack.pop_back();
    }

    void Visit(const ir::PolyFor *op, Expr *expr) {
      auto *node = expr->As<ir::PolyFor>();
      forloop_stack.push_back(expr);
      ir::IRMutator<>::Visit(&node->body, &node->body);
      if (!found) forloop_stack.pop_back();
    }

    void Visit(const ir::Store *op, Expr *expr) {
      found = op->tensor.as_tensor()->name == tensor_name;
    }
  };

  return Mutator(tensor_name)(expr);
}

Expr max(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be equal."));
  return ir::Max::Make(a, b);
}

Expr min(Expr a, Expr b) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be equal."));
  return ir::Min::Make(a, b);
}

bool ComparePriority(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs) {
  if (lhs.node_type() == ir::IrNodeTy::IntImm &&
      rhs.node_type() != ir::IrNodeTy::IntImm)
    return false;
  if (rhs.node_type() == ir::IrNodeTy::IntImm &&
      lhs.node_type() != ir::IrNodeTy::IntImm)
    return true;
  if (auto lhsVar = lhs.As<ir::_Var_>())
    if (auto rhsVar = rhs.As<ir::_Var_>())
      return std::make_tuple(lhsVar->name.length(), lhsVar->name) <=
             std::make_tuple(rhsVar->name.length(), rhsVar->name);
  auto lhsLen = lhs.length();
  auto rhsLen = rhs.length();
  if (lhsLen < rhsLen) return false;
  // Add < Mul < Div < Mod < Min < Max < Cast < Load.
  else if (lhsLen == rhsLen)
    return lhs.node_type() <= rhs.node_type();
  else
    return true;
}

bool IsSumPartialBySymbol(const ir::IndexExpr &expr,
                          const ir::IndexExpr &symbol) {
  if (expr == symbol) return true;
  // TODO(liujinnan): Check Ty
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm: {
      return false;
    }
    case ir::IrNodeTy::_Var_:
      return expr == symbol;
    case ir::IrNodeTy::Add:
      return IsSumPartialBySymbol(expr.operand(0), symbol) ||
             IsSumPartialBySymbol(expr.operand(1), symbol);
    case ir::IrNodeTy::Mul: {
      if (expr.operand(1).is_constant() && expr.operand(1).get_constant() == -1)
        return IsSumPartialBySymbol(expr.operand(0), symbol);
      else
        return expr.operand(0) == symbol || expr.operand(1) == symbol;
    }

    case ir::IrNodeTy::Div: {
      return IsSumPartialBySymbol(expr.operand(0), symbol);
    }
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast:
      return false;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in IsSumPartialBySymbol which is: %s",
          expr));
  }
}
ir::IndexExpr SimplifySymbolicAdd(const ir::IndexExpr &lhs,
                                  const ir::IndexExpr &sym,
                                  const ir::IndexExpr &outer_mul_factor) {
  if (lhs == sym) return sym * (outer_mul_factor + ir::IndexExpr(1));
  switch (lhs.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = lhs.As<ir::IntImm>();
      if (imm->value != 0)
        PADDLE_THROW(::common::errors::Fatal("Error in SimplifySymbolicAdd!"));
      return ir::IndexExpr(0);
    }
    case ir::IrNodeTy::_Var_: {
      return sym * (outer_mul_factor + ir::IndexExpr(1));
    }
    case ir::IrNodeTy::Add: {
      if (!common::IsSumPartialBySymbol(lhs.operand(0), sym))
        return lhs.operand(0) +
               SimplifySymbolicAdd(lhs.operand(1), sym, outer_mul_factor);
      return SimplifySymbolicAdd(lhs.operand(0), sym, outer_mul_factor) +
             lhs.operand(1);
    }
    case ir::IrNodeTy::Mul: {
      if (lhs.operand(1).is_constant() && lhs.operand(1).as_int64() == -1) {
        return SimplifySymbolicAdd(lhs.operand(0), sym, -outer_mul_factor) *
               lhs.operand(1);
      }
      if (lhs.operand(0) == sym)
        return lhs.operand(0) * (lhs.operand(1) + outer_mul_factor);
      return (lhs.operand(0) + outer_mul_factor) * lhs.operand(1);
    }
    case ir::IrNodeTy::Mod:
      PADDLE_THROW(::common::errors::Fatal("Error in SimplifySymbolicAdd!"));
    case ir::IrNodeTy::Div: {
      return SimplifySymbolicAdd(
                 lhs.operand(0), sym, lhs.operand(1) * outer_mul_factor) /
             lhs.operand(1);
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of lhs in SimplifySymbolicAdd which is: %s", lhs));
  }
}

bool IsDivisibleBySymbol(const ir::IndexExpr &expr,
                         const ir::IndexExpr &symbol,
                         const ir::IrNodeTy &ty) {
  if (expr == symbol) return true;
  // TODO(liujinnan): Check Ty
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = expr.As<ir::IntImm>();
      return imm->value == 0;
    }
    case ir::IrNodeTy::_Var_:
      return expr == symbol;
    case ir::IrNodeTy::Add:
      return IsDivisibleBySymbol(expr.operand(0), symbol, ty) &&
             IsDivisibleBySymbol(expr.operand(1), symbol, ty);
    case ir::IrNodeTy::Mul:
      // Because (S0 / 7 * 100) / S0 is not divisible by S0, so we push
      // `expr.node_type()` into third parameter.
      return IsDivisibleBySymbol(expr.operand(0), symbol, expr.node_type()) ||
             IsDivisibleBySymbol(expr.operand(1), symbol, expr.node_type());
    case ir::IrNodeTy::Mod:
      // Because S0 % 3 + S0 % 5 is not divisible by S0, so we push
      // `expr.node_type()` into third parameter.
      return IsDivisibleBySymbol(expr.operand(0), symbol, expr.node_type()) &&
             IsDivisibleBySymbol(expr.operand(1), symbol, expr.node_type());
    case ir::IrNodeTy::Div: {
      if (ty != expr.node_type()) return false;
      return IsDivisibleBySymbol(expr.operand(0), symbol, expr.node_type());
    }
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast:
      return false;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in IsDivisibleBySymbol which is: %s",
          expr));
  }
}

ir::IndexExpr SimplifySymbolicDivide(const ir::IndexExpr &lhs,
                                     const ir::IndexExpr &sym,
                                     const ir::IrNodeTy &ty) {
  if (lhs == sym) return ir::IndexExpr(1);
  switch (lhs.node_type()) {
    case ir::IrNodeTy::IntImm: {
      auto imm = lhs.As<ir::IntImm>();
      if (imm->value != 0)
        PADDLE_THROW(
            ::common::errors::Fatal("Error in SimplifySymbolicDivide!"));
      return ir::IndexExpr(0);
    }
    case ir::IrNodeTy::_Var_:
      return ir::IndexExpr(1);
    case ir::IrNodeTy::Add:
      return SimplifySymbolicDivide(lhs.operand(0), sym, ty) +
             SimplifySymbolicDivide(lhs.operand(1), sym, ty);
    case ir::IrNodeTy::Mul: {
      if (!common::IsDivisibleBySymbol(lhs.operand(0), sym, ty))
        return lhs.operand(0) * SimplifySymbolicDivide(lhs.operand(1), sym, ty);
      return SimplifySymbolicDivide(lhs.operand(0), sym, ty) * lhs.operand(1);
    }
    case ir::IrNodeTy::Mod:
      return SimplifySymbolicDivide(lhs.operand(0), sym, lhs.node_type()) %
             SimplifySymbolicDivide(lhs.operand(1), sym, lhs.node_type());
    case ir::IrNodeTy::Div: {
      return SimplifySymbolicDivide(lhs.operand(0), sym, lhs.node_type()) /
             lhs.operand(1);
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of lhs in SimplifySymbolicDivide which is: %s",
          lhs));
  }
}

bool ProveDivisible(const ir::IndexExpr &lhs, const ir::IndexExpr &rhs) {
  if (IsZero(lhs % rhs)) return true;
  if (IsZero(optim::ArithSimplify(lhs % rhs))) return true;
  return false;
}

bool IsNegatedIndexExpr(const ir::IndexExpr &candidate,
                        ir::IndexExpr &expr) {  // NOLINT
  if (auto mul = candidate.As<ir::Mul>()) {
    if (mul->b().is_constant() && mul->b().get_constant() == -1) {
      expr = mul->a();
      return true;
    }
  }
  return false;
}

IndexType VerifyIndex(const ir::Expr &expr) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::IntImm: {
      return expr.type().is_index_type() ? IndexType::kValid
                                         : IndexType::kInvalid;
    }
    case ir::IrNodeTy::Load: {
      return expr.type().is_index_type() ? IndexType::kLoad
                                         : IndexType::kInvalid;
    }
    case ir::IrNodeTy::Cast: {
      IndexType result = VerifyIndex(expr->operand(0));
      return result == IndexType::kValid && expr.type().is_index_type()
                 ? IndexType::kCast
                 : IndexType::kInvalid;
    }
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Sub:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Min: {
      IndexType left = VerifyIndex(expr->operand(0));
      IndexType right = VerifyIndex(expr->operand(1));
      if (left == IndexType::kInvalid || right == IndexType::kInvalid)
        return IndexType::kInvalid;
      return std::max(left, right);
    }
  }
  return IndexType::kInvalid;
}

ir::IndexExpr ConstructIndexExprByNodeType(const ir::IrNodeTy &ty,
                                           const ir::IndexExpr &lhs,
                                           const ir::IndexExpr &rhs,
                                           bool simplify_flag) {
  switch (ty) {
    case ir::IrNodeTy::Add:
      return simplify_flag ? lhs + rhs : ir::Add::Make(lhs, rhs);
    case ir::IrNodeTy::Sub:
      return simplify_flag ? lhs - rhs : ir::Sub::Make(lhs, rhs);
    case ir::IrNodeTy::Mul:
      return simplify_flag ? lhs * rhs : ir::Mul::Make(lhs, rhs);
    case ir::IrNodeTy::Div:
      return simplify_flag ? lhs / rhs : ir::Div::Make(lhs, rhs);
    case ir::IrNodeTy::Mod:
      return simplify_flag ? lhs % rhs : ir::Mod::Make(lhs, rhs);
    case ir::IrNodeTy::Min:
      return ir::Min::Make(lhs, rhs);
    case ir::IrNodeTy::Max:
      return ir::Max::Make(lhs, rhs);
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type in Constructir::IndexExprByNodeType, which is: %s",
          ty));
  }
}

ir::IndexExpr ChangeSeqOfDivMod(const ir::IndexExpr &expr) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm:
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::Cast:
    case ir::IrNodeTy::Load: {
      return expr;
    }
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Sub:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Div: {
      auto lhs = ChangeSeqOfDivMod(expr.operand(0));
      auto rhs = ChangeSeqOfDivMod(expr.operand(1));
      return ConstructIndexExprByNodeType(expr.node_type(), lhs, rhs, false);
    }
    case ir::IrNodeTy::Mod: {
      if (expr.operand(0).node_type() == ir::IrNodeTy::Div) {
        auto div_lhs = ChangeSeqOfDivMod(expr.operand(0).operand(0));
        auto div_rhs = ChangeSeqOfDivMod(expr.operand(0).operand(1));
        auto mod_rhs = ChangeSeqOfDivMod(expr.operand(1));
        return div_lhs % (div_rhs * mod_rhs) / div_rhs;
      } else {
        auto lhs = ChangeSeqOfDivMod(expr.operand(0));
        auto rhs = ChangeSeqOfDivMod(expr.operand(1));
        if (lhs.node_type() == ir::IrNodeTy::Div) {
          return (lhs.operand(0) % (lhs.operand(1) * rhs)) / lhs.operand(1);
        }
        return ConstructIndexExprByNodeType(expr.node_type(), lhs, rhs, false);
      }
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in ChangeSeqOfDivMod which is: %s", expr));
  }
}
std::optional<ir::IndexExpr> DivByPartMul(const ir::IndexExpr &lhs,
                                          const ir::IndexExpr &rhs,
                                          ir::IrNodeTy ty) {
  std::vector<ir::IndexExpr> elems = common::GetFlattenExprs<ir::Mul>(rhs);

  ir::IndexExpr result = ir::ir_utils::IRCopy(lhs);

  for (const auto &elem : elems) {
    if (common::IsDivisibleBySymbol(result, elem, ty)) {
      result = common::SimplifySymbolicDivide(result, elem, ty);
    } else {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<ir::IndexExpr> SimplifyComplexMod(const ir::IndexExpr &lhs,
                                                const ir::IndexExpr &rhs) {
  if (lhs == rhs) return ir::IndexExpr(lhs.type(), 0);
  switch (lhs.node_type()) {
    case ir::IrNodeTy::Add: {
      auto simplify_lhs = SimplifyComplexMod(lhs.operand(0), rhs);
      auto simplify_rhs = SimplifyComplexMod(lhs.operand(1), rhs);
      if (simplify_lhs.has_value() && simplify_rhs.has_value())
        return (simplify_lhs.value() + simplify_rhs.value());
      return std::nullopt;
    }
    case ir::IrNodeTy::Mul: {
      // (S0 % 4 * S1 % 8) % 4 != S0 % 4 * S1 % 4;
      if (DivByPartMul(lhs, rhs, ir::IrNodeTy::Mod))
        return ir::IndexExpr(lhs.type(), 0);
      return std::nullopt;
    }
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::IntImm:
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast: {
      return std::nullopt;
    }
    case ir::IrNodeTy::Mod: {
      if (DivByPartMul(lhs.operand(1), rhs, ir::IrNodeTy::Mod)) {
        return lhs.operand(0) % rhs;
      }
      return std::nullopt;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in SimplifyComplexMod which is: %s", lhs));
  }
  return std::nullopt;
}
}  // namespace common
}  // namespace cinn
