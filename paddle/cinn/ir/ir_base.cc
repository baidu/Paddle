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

#include "paddle/cinn/ir/ir_base.h"
#include <sstream>
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/const_fold.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

using cinn::common::bfloat16;
using cinn::common::float16;

//! Implementations for Ir Expr Nodes.
// @{
#define __m(t__)                                             \
  template <>                                                \
  void ExprNode<t__>::Accept(cinn::ir::IRVisitor *v) const { \
    v->Visit(const_self());                                  \
  }
#undef __m
// @}

std::ostream &operator<<(std::ostream &os, IrNodeTy type) {
  switch (type) {
    case IrNodeTy::IterMark:
      os << "<node: IterMark>";
      break;
    case IrNodeTy::IterSplit:
      os << "<node: IterSplit>";
      break;
    case IrNodeTy::IterSum:
      os << "<node: IterSum>";
      break;

#define __m(t__)                    \
  case IrNodeTy::t__:               \
    os << "<node: " << #t__ << ">"; \
    break;

      NODETY_FORALL(__m)
#undef __m

    default:
      PADDLE_THROW(::common::errors::InvalidArgument("unknown IrNodeTy found"));
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, StmtNodeTy type) {
  switch (type) {
#define __m(t__)                         \
  case StmtNodeTy::t__:                  \
    os << "<stmt node: " << #t__ << ">"; \
    break;

    NODETY_FORALL_STMT(__m)
#undef __m
    default:
      PADDLE_THROW(
          ::common::errors::InvalidArgument("unknown StmtNodeTy found"));
  }
}

Expr Zero(const Type &type) {
  if (type.is_bfloat16()) return Expr(bfloat16(0.f));
  if (type.is_float16()) return Expr(float16(0.f));
  if (type.is_float(32)) return Expr(0.f);
  if (type.is_float(64)) return Expr(double(0.));  // NOLINT

  if (type.is_bool()) return Expr(false);

  if (type.is_int(8)) return Expr(int8_t(0));
  if (type.is_int(16)) return Expr(int16_t(0));
  if (type.is_int(32)) return Expr(int32_t(0));
  if (type.is_int(64)) return Expr(int64_t(0));

  if (type.is_uint(8)) return Expr(uint8_t(0));
  if (type.is_uint(16)) return Expr(uint16_t(0));
  if (type.is_uint(32)) return Expr(uint32_t(0));
  if (type.is_uint(64)) return Expr(uint64_t(0));
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr One(const Type &type) {
  if (type.is_bfloat16()) return Expr(bfloat16(1.f));
  if (type.is_float16()) return Expr(float16(1.f));
  if (type.is_float(32)) return Expr(1.f);
  if (type.is_float(64)) return Expr(double(1.));  // NOLINT

  if (type.is_bool()) return Expr(true);

  if (type.is_int(8)) return Expr(int8_t(1));
  if (type.is_int(16)) return Expr(int16_t(1));
  if (type.is_int(32)) return Expr(int32_t(1));
  if (type.is_int(64)) return Expr(int64_t(1));

  if (type.is_uint(8)) return Expr(uint8_t(1));
  if (type.is_uint(16)) return Expr(uint16_t(1));
  if (type.is_uint(32)) return Expr(uint32_t(1));
  if (type.is_uint(64)) return Expr(uint64_t(1));
  CINN_NOT_IMPLEMENTED
  return Expr();
}

Expr::Expr(const Var &var) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&var);
}

Expr::Expr(const IndexExpr &e) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&e);
}

bool Expr::as_bool() const {
  PADDLE_ENFORCE_EQ(
      type().is_uint(1),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be a 1-bit unsigned integer type."));
  return As<UIntImm>()->value;
}

int8_t Expr::as_int8() const {
  PADDLE_ENFORCE_EQ(
      type().is_int(8),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be an 8-bit integer type."));
  return As<IntImm>()->value;
}
int16_t Expr::as_int16() const {
  PADDLE_ENFORCE_EQ(
      type().is_int(16),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be an 16-bit integer type."));
  return As<IntImm>()->value;
}
int32_t Expr::as_int32() const {
  PADDLE_ENFORCE_EQ(
      type().is_int(32),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be an 32-bit integer type. %s",
          ::common::GetCurrentTraceBackString()));
  return As<IntImm>()->value;
}
int64_t Expr::as_int64() const {
  if (!type().is_int(64))
    PADDLE_ENFORCE_EQ(type().is_int(32),
                      true,
                      ::common::errors::InvalidArgument(
                          "Invalid type. The type must be an 32-bit "
                          "integer or 64-bit integer type."));
  return As<IntImm>()->value;
}

uint8_t Expr::as_uint8() const {
  PADDLE_ENFORCE_EQ(
      type().is_uint(8),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be a 8-bit unsigned integer type."));
  return As<UIntImm>()->value;
}
uint16_t Expr::as_uint16() const {
  PADDLE_ENFORCE_EQ(
      type().is_uint(16),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be a 16-bit unsigned integer type."));
  return As<UIntImm>()->value;
}
uint32_t Expr::as_uint32() const {
  PADDLE_ENFORCE_EQ(
      type().is_uint(32),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be a 32-bit unsigned integer type."));
  return As<UIntImm>()->value;
}
uint64_t Expr::as_uint64() const {
  PADDLE_ENFORCE_EQ(
      type().is_uint(64),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be a 64-bit unsigned integer type."));
  return As<UIntImm>()->value;
}

bfloat16 Expr::as_bfloat16() const {
  PADDLE_ENFORCE_EQ(type().is_bfloat16(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Invalid type. The type must be bfloat16() type."));
  return bfloat16(As<FloatImm>()->value);
}
float16 Expr::as_float16() const {
  PADDLE_ENFORCE_EQ(type().is_bfloat16(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Invalid type. The type must be bfloat16() type."));
  return float16(As<FloatImm>()->value);
}
float Expr::as_float() const {
  PADDLE_ENFORCE_EQ(type().is_float(32),
                    true,
                    ::common::errors::InvalidArgument(
                        "The type must be a 32-bit floating point type."));
  return As<FloatImm>()->value;
}
double Expr::as_double() const {
  PADDLE_ENFORCE_EQ(type().is_float(64),
                    true,
                    ::common::errors::InvalidArgument(
                        "The type must be a 64-bit floating point type."));
  return As<FloatImm>()->value;
}

Expr &Expr::operator=(const Expr &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

Expr &Expr::operator=(const IndexExpr &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

Expr &Expr::operator=(const Var &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

Expr::operator Var() {
  auto *x = As<ir::_Var_>();
  PADDLE_ENFORCE_NOT_NULL(
      x,
      ::common::errors::InvalidArgument("x is a nullptr. It must not be null"));
  return ir::Var(x);
}

bool Expr::is_constant() const {
  return As<IntImm>() || As<UIntImm>() || As<FloatImm>();
}

double Expr::get_constant() const {
  PADDLE_ENFORCE_EQ(is_constant(),
                    true,
                    ::common::errors::InvalidArgument(
                        "%s is not constant! Please check.", *this));
  auto *vi = As<IntImm>();
  auto *vf = As<FloatImm>();
  auto *vu = As<UIntImm>();
  if (vi) return vi->value;
  if (vu) return vu->value;
  return vf->value;
}

bool Expr::is_var() const { return As<_Var_>(); }

bool Expr::is_index() const {
  // Temporarily use `VerifyIndex`. because `get_index` depends on marking
  // `indexExpr` in For::make and sch
  return VerifyIndex(*this);
  // return get()->get_index();
}

Expr &Expr::set_index(bool flag) {
  if (flag && !VerifyIndex(*this)) {
    PADDLE_THROW(::common::errors::InvalidType(
        "Expr: %s is not IndexExpr! cannot be set as IndexExpr.", *this));
  }
  get()->set_index(flag);
  return *this;
}

const Expr &Expr::set_index(bool flag) const {
  if (flag && !VerifyIndex(*this)) {
    PADDLE_THROW(::common::errors::InvalidType(
        "Expr: %s is not IndexExpr! cannot be set as IndexExpr.", *this));
  }
  get()->set_index(flag);
  return *this;
}

const IndexExpr Expr::as_index() const {
  if (is_index()) {
    std::vector<ir::Expr> collection =
        ir::ir_utils::CollectIRNodesWithoutTensor(*this, [&](const Expr *x) {
          return x->node_type() == ir::IrNodeTy::Sub;
        });
    if (!collection.empty()) return IndexExpr(*this).Normalize();
    return IndexExpr(*this);
  }
  PADDLE_THROW(
      ::common::errors::InvalidType("Expr: %s is not IndexExpr!", *this));
}

IndexExpr Expr::as_index() {
  if (is_index()) {
    std::vector<ir::Expr> collection =
        ir::ir_utils::CollectIRNodesWithoutTensor(*this, [&](const Expr *x) {
          return x->node_type() == ir::IrNodeTy::Sub;
        });
    if (!collection.empty()) return IndexExpr(*this).Normalize();
    return IndexExpr(*this);
  }
  PADDLE_THROW(
      ::common::errors::InvalidType("Expr: %s is not IndexExpr!", *this));
}

_Buffer_ *Expr::as_buffer() { return As<_Buffer_>(); }
const _Buffer_ *Expr::as_buffer() const { return As<_Buffer_>(); }
Buffer Expr::as_buffer_ref() const { return Buffer(&Reference(as_buffer())); }

_Tensor_ *Expr::as_tensor() { return As<_Tensor_>(); }
const _Tensor_ *Expr::as_tensor() const { return As<_Tensor_>(); }
ir::Tensor Expr::as_tensor_ref() const {
  return ir::Tensor(&Reference(as_tensor()));
}

_Var_ *Expr::as_var() { return As<_Var_>(); }
const _Var_ *Expr::as_var() const { return As<_Var_>(); }
Var Expr::as_var_ref() const { return Var(&Reference(as_var())); }

bool Expr::is_cmp() const {
  switch (node_type()) {
    case ir::IrNodeTy::LE:
    case ir::IrNodeTy::LT:
    case ir::IrNodeTy::EQ:
    case ir::IrNodeTy::NE:
    case ir::IrNodeTy::GT:
    case ir::IrNodeTy::GE:
      return true;
    default:
      return false;
  }
}

const Expr &IrNode::operand(int i) {
  PADDLE_ENFORCE_LT(
      i,
      operands.size(),
      ::common::errors::InvalidArgument("The index %d is out of range", i));
  return operands[i];
}

IndexExpr::IndexExpr(const Expr &e) {
  if (!e.is_index())
    PADDLE_THROW(
        ::common::errors::InvalidType("Expr: %s is not IndexExpr!", e));
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&e);
}

IndexExpr &IndexExpr::operator=(const IndexExpr &other) {
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

IndexExpr &IndexExpr::operator=(const Expr &other) {
  if (!other.is_index()) {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "The Expr is not IndexExpr, which is: %s", other));
  }
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

IndexExpr &IndexExpr::operator=(const Var &other) {
  if (!other.is_index()) {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "The Expr is not IndexExpr, which is: %s", other));
  }
  *static_cast<IrNodeRef *>(this) = *static_cast<const IrNodeRef *>(&other);
  return *this;
}

const IndexExpr IndexExpr::operand(int32_t i) const {
  return get()->operand(i).as_index();
}

int64_t IndexExpr::GetLargestMultiplyPart() const {
  switch (node_type()) {
    case cinn::ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max:
    case ir::IrNodeTy::Load:
    case ir::IrNodeTy::Cast:
      return 1;
    case cinn::ir::IrNodeTy::Div: {
      if (operand(1).type().is_index_type()) {
        int64_t lhsDiv = operand(0).GetLargestMultiplyPart();
        int64_t rhsDiv = operand(1).GetLargestMultiplyPart();
        if (lhsDiv % rhsDiv == 0) return std::abs(lhsDiv / rhsDiv);
      }
      return 1;
    }
    case cinn::ir::IrNodeTy::IntImm: {
      auto int_imm = As<ir::IntImm>();
      return std::abs(int_imm->value);
    }
    case cinn::ir::IrNodeTy::Mul: {
      return operand(0).GetLargestMultiplyPart() *
             operand(1).GetLargestMultiplyPart();
    }
    case cinn::ir::IrNodeTy::Add:
    case cinn::ir::IrNodeTy::Mod: {
      return std::gcd(operand(0).GetLargestMultiplyPart(),
                      operand(1).GetLargestMultiplyPart());
    }
  }
  PADDLE_THROW(::common::errors::Unimplemented("Unsupported type of expr: %s",
                                               node_type()));
}

int32_t IndexExpr::length() const {
  switch (node_type()) {
    case ir::IrNodeTy::_Var_:
    case ir::IrNodeTy::IntImm:
    case ir::IrNodeTy::Load:
      return 1;
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max: {
      int lhs_count = operand(0).length();
      int rhs_count = operand(1).length();
      return lhs_count + rhs_count + 1;
    }
    case ir::IrNodeTy::Cast: {
      return operand(0).length() + 1;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type in length, which is: %s", node_type()));
  }
}

bool IndexExpr::IsDynamic() const {
  switch (node_type()) {
    case ir::IrNodeTy::_Var_:
      return as_var()->name.at(0) == 'S';
    case ir::IrNodeTy::Load:
      return true;
    case ir::IrNodeTy::IntImm:
      return false;
    case ir::IrNodeTy::Cast:
      return operand(0).IsDynamic();
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max: {
      auto lFlag = operand(0).IsDynamic();
      auto rFlag = operand(1).IsDynamic();
      return lFlag || rFlag;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type in IsDynamic, which is: %s", node_type()));
  }
}

IndexExpr Simplify(const IndexExpr &expr, IndexExpr::OptLevel level) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm:
      return expr;
    case ir::IrNodeTy::_Var_: {
      auto op = expr.As<ir::_Var_>();
      if (op->lower_bound.defined() && op->upper_bound.defined()) {
        if (!(op->lower_bound.is_constant() && op->upper_bound.is_constant()))
          return expr;
        auto l = op->lower_bound.as_int64();
        auto u = op->upper_bound.as_int64();
        if (l && u && l + 1 == u) return op->lower_bound;
        return expr;
      }
      return expr;
    }
    case ir::IrNodeTy::Load: {
      auto load = expr.As<ir::Load>();
      return Load::Make(load->tensor, load->indices).set_index(true);
    }
    case ir::IrNodeTy::Cast: {
      auto v = Simplify(expr.operand(0), level);
      return Cast::Make(expr.type(), v);
    }
    case ir::IrNodeTy::Add:
    case ir::IrNodeTy::Sub:
    case ir::IrNodeTy::Mul:
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod:
    case ir::IrNodeTy::Min:
    case ir::IrNodeTy::Max: {
      auto lhs = Simplify(expr.operand(0), level);
      auto rhs = Simplify(expr.operand(1), level);
      auto res = ConstructIndexExprByNodeType(expr.node_type(), lhs, rhs);
      if (level == IndexExpr::OptLevel::Level2 &&
          expr.node_type() == ir::IrNodeTy::Add)
        res = common::MergeMulMod(res);
      return res;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Unsupported type of expr in Simplify which is: %s", expr));
  }
}

IndexExpr IndexExpr::Normalize(OptLevel level) const {
  auto res = Simplify(*this, level);
  res = ChangeSeqOfDivMod(res);
  return Simplify(res, level);
}

int32_t IndexExpr::as_int32() const {
  PADDLE_ENFORCE_EQ(
      type().is_int(32),
      true,
      ::common::errors::InvalidArgument(
          "Invalid type. The type must be an 32-bit integer type. %s",
          ::common::GetCurrentTraceBackString()));
  return As<IntImm>()->value;
}
int64_t IndexExpr::as_int64() const {
  if (!type().is_int(64))
    PADDLE_ENFORCE_EQ(type().is_int(32),
                      true,
                      ::common::errors::InvalidArgument(
                          "Invalid type. The type must be an 32-bit "
                          "integer or 64-bit integer type."));
  return As<IntImm>()->value;
}

bool IndexExpr::is_constant() const { return As<IntImm>(); }
int64_t IndexExpr::get_constant() const { return As<IntImm>()->value; }

bool IndexExpr::is_var() const { return As<_Var_>(); }
_Var_ *IndexExpr::as_var() { return As<_Var_>(); }
const _Var_ *IndexExpr::as_var() const { return As<_Var_>(); }
Var IndexExpr::as_var_ref() const { return Var(&Reference(as_var())); }
void IrNode::set_type(Type type) { type_ = type; }

void IrNode::replace(Expr old_op, Expr new_op) {
  std::stringstream ss;
  ss << "Not Implemented, The node:(" << node_type() << ") has an old_op: ("
     << old_op.node_type() << ") should be replaced with new_op: ("
     << new_op.node_type() << ") but not Implemented";

  PADDLE_THROW(::common::errors::Unimplemented(ss.str()));
}

bool IrNode::get_index() const { return is_index_; }
void IrNode::set_index(bool flag) {
  if (is_index_ == flag) return;
  is_index_ = flag;
  if (flag) {
    for (Expr &operand : operands) {
      operand->set_index(flag);
    }
  }
}

void IrNode::convert_int32_to_int64() {
  if (type_ != Int(64) && type_ != UInt(64))
    if (type_ != Int(32) && type_ != UInt(32))
      PADDLE_ENFORCE_EQ(type_.is_unk(),
                        true,
                        ::common::errors::InvalidArgument(
                            "Current only support convert int32_t "
                            "to int64_t, but get type is: %s",
                            type_));

  if (type_ == Int(32)) type_ = Int(64);
  if (type_ == UInt(32)) type_ = UInt(64);

  for (Expr &operand : operands) {
    operand->convert_int32_to_int64();
  }
}

void IrNode::convert_int64_to_int32() {
  if (type_ != Int(64) && type_ != UInt(64))
    if (type_ != Int(32) && type_ != UInt(32))
      PADDLE_ENFORCE_EQ(type_.is_unk(),
                        true,
                        ::common::errors::InvalidArgument(
                            "Current only support convert int64_t "
                            "to int32_t, but get type is: %s, node type is: %s",
                            type_,
                            node_type()));

  if (node_type() == IrNodeTy::IntImm) {
    auto *int_imm = static_cast<IntImm *>(this);
    if (int_imm->value >= INT_MAX) return;
    int_imm->value = int32_t(int_imm->value);
  }

  if (type_ == Int(64)) type_ = Int(32);
  if (type_ == UInt(64)) type_ = UInt(32);

  for (Expr &operand : operands) {
    operand->convert_int64_to_int32();
  }
}

void TryElevateInt32ToInt64(const std::vector<Expr> &expr_vec) {
  Type type = expr_vec.front()->type();
  for (const Expr &expr : expr_vec) {
    if (expr->type() == Int(64)) {
      type = Int(64);
      break;
    }
  }

  // Not need Elevate to Int(64)
  if (type != Int(64)) {
    return;
  }
  for (const Expr &expr : expr_vec) {
    if (expr->type() != Int(64))
      if (expr->type() != Int(32))
        PADDLE_ENFORCE_EQ(expr->type().is_unk(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Current only support convert int32_t "
                              "to int64_t, but get type is: %s",
                              expr->type()));
    if (expr->type() == Int(32)) {
      expr->convert_int32_to_int64();
    }
  }
}

void TryElevateInt64ToInt32(const std::vector<Expr> &expr_vec) {
  for (const Expr &expr : expr_vec) {
    if (!expr.is_index()) return;
  }

  for (const Expr &expr : expr_vec) {
    if (expr->type() != Int(64))
      if (expr->type() != Int(32))
        PADDLE_ENFORCE_EQ(expr->type().is_unk(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Current only support convert int64_t "
                              "to int32_t, but get type is: %s",
                              expr->type()));
    if (expr->type() == Int(64)) {
      expr->convert_int64_to_int32();
    }
  }
}

}  // namespace ir
}  // namespace cinn
