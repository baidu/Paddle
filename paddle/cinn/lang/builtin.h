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

#include <vector>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace lang {

//! Get the ALL of the conditions.
Expr logic_and(const std::vector<Expr>& conds);
Expr logic_or(const std::vector<Expr>& conds);

Expr Zero(const Type& type);
Expr One(const Type& type);
Expr min_value(const Type& type);
Expr max_value(const Type& type);
Expr Epsilon(const Type& type);

//! extern call op
#define EXTERN_CALL_DCL(name__) Expr name__(Expr e);

EXTERN_CALL_DCL(Exp);
EXTERN_CALL_DCL(Erf);
EXTERN_CALL_DCL(Sqrt);
EXTERN_CALL_DCL(Rsqrt);
EXTERN_CALL_DCL(Log);
EXTERN_CALL_DCL(Log2);
EXTERN_CALL_DCL(Log10);
EXTERN_CALL_DCL(Floor);
EXTERN_CALL_DCL(Ceil);
EXTERN_CALL_DCL(Round);
EXTERN_CALL_DCL(Trunc);
EXTERN_CALL_DCL(Cos);
EXTERN_CALL_DCL(Cosh);
EXTERN_CALL_DCL(Tan);
EXTERN_CALL_DCL(Sin);
EXTERN_CALL_DCL(Sinh);
EXTERN_CALL_DCL(Acos);
EXTERN_CALL_DCL(Acosh);
EXTERN_CALL_DCL(Asin);
EXTERN_CALL_DCL(Asinh);
EXTERN_CALL_DCL(Atan);
EXTERN_CALL_DCL(Atanh);
EXTERN_CALL_DCL(Tanh);
EXTERN_CALL_DCL(Cbrt);
EXTERN_CALL_DCL(Clz);
EXTERN_CALL_DCL(Popc);

#undef EXTERN_CALL_DCL

//! extern call binary op
#define EXTERN_BINARY_CALL_DCL(name__) Expr name__(Expr a, Expr b);

EXTERN_BINARY_CALL_DCL(FloorDivide);
EXTERN_BINARY_CALL_DCL(Remainder);
EXTERN_BINARY_CALL_DCL(Mod);
EXTERN_BINARY_CALL_DCL(LogicalRightShift);
EXTERN_BINARY_CALL_DCL(Pow);

#undef EXTERN_BINARY_CALL_DCL

inline Expr Sigmoid(Expr e) {
  auto one = One(e->type());
  return one / (one + Exp(-e));
}

inline Expr Sign(Expr e) {
  auto zero = Zero(e->type());
  auto one = One(e->type());
  auto neg_one = ir::Cast::Make(e->type(), Expr(-1));
  auto ret0 = ir::Select::Make(ir::EQ::Make(e, zero), zero, e);
  auto ret1 = ir::Select::Make(e > zero, one, ret0);
  auto ret2 = ir::Select::Make(e < zero, neg_one, ret1);
  return ret2;
}

Expr Abs(Expr e);

inline Expr Negative(Expr e) { return -e; }
inline Expr Identity(Expr e) { return e; }
inline Expr LogicalNot(Expr e) { return !e; }
inline Expr BitwiseNot(Expr e) { return ~e; }
inline Expr BitwiseAnd(Expr a, Expr b) { return a & b; }
inline Expr BitwiseOr(Expr a, Expr b) { return a | b; }
inline Expr BitwiseXor(Expr a, Expr b) { return a ^ b; }
inline Expr LeftShift(Expr a, Expr b) { return a << b; }
inline Expr RightShift(Expr a, Expr b) { return a >> b; }

inline Expr Relu(Expr e, double threshold = 0.0) {
  return ir::Max::Make(e, ir::Cast::Make(e->type(), Expr(threshold)));
}

inline Expr Relu6(Expr e, double threshold = 0.0) {
  return ir::Min::Make(
      ir::Max::Make(e, ir::Cast::Make(e->type(), Expr(threshold))),
      ir::Cast::Make(e->type(), Expr(6.0)));
}

inline Expr LeakyRelu(Expr e, double alpha) {
  auto zero = Zero(e->type());
  return ir::Select::Make(
      e > zero, e, e * ir::Cast::Make(e->type(), Expr(alpha)));
}

inline Expr LeakyRelu(Expr e, Expr alpha) {
  auto zero = Zero(e->type());
  return ir::Select::Make(e > zero, e, e * alpha);
}

inline Expr ReduceSum(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = Zero(e->type());
  }
  return ir::Reduce::Make(ir::Reduce::kSum, initial, e, reduce_axis);
}

inline Expr ReduceMul(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = One(e->type());
  }
  return ir::Reduce::Make(ir::Reduce::kMul, initial, e, reduce_axis);
}

inline Expr ReduceMax(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = min_value(e.type());
  }
  return ir::Reduce::Make(ir::Reduce::kMax, initial, e, reduce_axis);
}
inline Expr ReduceMin(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = max_value(e.type());
  }
  return ir::Reduce::Make(ir::Reduce::kMin, initial, e, reduce_axis);
}
inline Expr ReduceAll(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = Expr(true);
  }
  return ir::Reduce::Make(ir::Reduce::kAll, initial, e, reduce_axis);
}
inline Expr ReduceAny(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = Expr(false);
  }
  return ir::Reduce::Make(ir::Reduce::kAny, initial, e, reduce_axis);
}

inline Expr ReduceVar(Expr e,
                      const std::vector<Var>& reduce_axis,
                      Expr initial = Expr()) {
  if (!initial.defined()) {
    initial = Zero(e->type());
  }
  return ir::Reduce::Make(ir::Reduce::kVar, initial, e, reduce_axis);
}

Expr IsNan(Expr e);

Expr Infinity(const Type& type);

Expr IsInf(Expr e);

Expr IsFinite(Expr e);

}  // namespace lang
}  // namespace cinn
