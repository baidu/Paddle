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

#include "paddle/cinn/ir/ir.h"

#include <map>
#include <optional>
#include <string>
#include <vector>
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

namespace cinn {
namespace ir {

using cinn::common::make_shared;

Expr Cast::Make(Type t, Expr v) {
  PADDLE_ENFORCE_EQ(!t.is_unk(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The type is unknown. "
                        "A valid type is required for casting."));
  PADDLE_ENFORCE_EQ(
      !(t.is_void() && !t.is_cpp_handle()),
      true,
      ::common::errors::InvalidArgument(
          "Void is not allowed to cast. "
          "Ensure the type is not void unless it is a C++ handle."));
  PADDLE_ENFORCE_EQ(v.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The expression is not defined. "
                        "A defined expression is required for casting."));

#define __CAST_TO_TYPE(type__)                  \
  if (auto *i = v.As<ir::IntImm>()) {           \
    return Expr(static_cast<type__>(i->value)); \
  } else if (auto *u = v.As<ir::UIntImm>()) {   \
    return Expr(static_cast<type__>(u->value)); \
  }

  if (v.is_constant()) {
    if (t == type_of<int8_t>()) {
      __CAST_TO_TYPE(int8_t)
    } else if (t == type_of<int16_t>()) {
      __CAST_TO_TYPE(int16_t)
    } else if (t == type_of<int32_t>()) {
      __CAST_TO_TYPE(int32_t)
    } else if (t == type_of<int64_t>()) {
      __CAST_TO_TYPE(int64_t)
    } else if (t == type_of<uint8_t>()) {
      __CAST_TO_TYPE(uint8_t)
    } else if (t == type_of<uint16_t>()) {
      __CAST_TO_TYPE(uint16_t)
    } else if (t == type_of<uint32_t>()) {
      __CAST_TO_TYPE(uint32_t)
    } else if (t == type_of<uint64_t>()) {
      __CAST_TO_TYPE(uint64_t)
    }
  }
#undef __CAST_TO_TYPE

  // Cast indexExpr without `cast` and `load`
  if (common::VerifyIndex(v) == common::IndexType::kValid && t == Int(64)) {
    v->convert_int32_to_int64();
    return v;
  }
  if (common::VerifyIndex(v) == common::IndexType::kValid && t == Int(32)) {
    v->convert_int64_to_int32();
    return v;
  }

  auto node = make_shared<Cast>();
  node->v() = v;
  node->set_type(t);
  return Expr(node);
}

void Cast::Verify() const {
  if (v().type() == type())
    LOG(WARNING) << "Found a Cast Node casting a value to the same type, this "
                    "is not reasonable";
}

Expr Add::Make(Expr a, Expr b) {
  auto node = make_shared<Add>(a, b);
  // For performance reasons, do not perform frequent simplifications. Just set
  // the is_index() flag to true. For extreme optimization,
  // `node->set_index(true)` can be replaced by `a.as_index() + b.as_index()`
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Add::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Add>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void BinaryNodeVerify(const Expr &a, const Expr &b, absl::string_view ir_name) {
  PADDLE_ENFORCE_EQ(
      a.defined(),
      true,
      ::common::errors::InvalidArgument("The first operand is not defined. "
                                        "A valid expression is required."));
  PADDLE_ENFORCE_EQ(
      b.defined(),
      true,
      ::common::errors::InvalidArgument("The second operand is not defined. "
                                        "A valid expression is required."));
  TryElevateInt32ToInt64({a, b});
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The operands' types of the node [%s] don't match. "
                        "Received types: %s and %s",
                        ir_name.data(),
                        a.type().to_string().c_str(),
                        b.type().to_string().c_str()));
}

void Add::Verify() const { BinaryNodeVerify(a(), b(), "Add"); }

Expr Sub::Make(Expr a, Expr b) {
  auto node = make_shared<Sub>(a, b);
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Sub::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Sub>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void Sub::Verify() const { BinaryNodeVerify(a(), b(), "Sub"); }

Expr Mul::Make(Expr a, Expr b) {
  auto node = make_shared<Mul>(a, b);
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Mul::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Mul>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void Mul::Verify() const { BinaryNodeVerify(a(), b(), "Mul"); }

Expr Div::Make(Expr a, Expr b) {
  auto node = make_shared<Div>(a, b);
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Div::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Div>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void Div::Verify() const { BinaryNodeVerify(a(), b(), "Div"); }

Expr Mod::Make(Expr a, Expr b) {
  auto node = make_shared<Mod>(a, b);
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Mod::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Mod>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void Mod::Verify() const { BinaryNodeVerify(a(), b(), "Mod"); }

Expr Min::Make(Expr a, Expr b) {
  auto node = make_shared<Min>(a, b);
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Min::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Min>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void Min::Verify() const { BinaryNodeVerify(a(), b(), "Min"); }

Expr Max::Make(Expr a, Expr b) {
  auto node = make_shared<Max>(a, b);
  if (a.is_index() && b.is_index()) node->set_index(true);
  return Expr(node);
}

IndexExpr Max::Make(IndexExpr a, IndexExpr b) {
  auto node = make_shared<Max>(a, b);
  node->set_index(true);
  return IndexExpr(node);
}

void Max::Verify() const { BinaryNodeVerify(a(), b(), "Max"); }

Expr Minus::Make(Expr a) {
  auto node = make_shared<Minus>(a);
  return Expr(node);
}

void Minus::Verify() const {
  PADDLE_ENFORCE_EQ(
      v().defined(),
      true,
      ::common::errors::InvalidArgument(
          "The operand is not defined. "
          "A valid operand is required for the Minus operation."));
}

Expr EQ::Make(Expr a, Expr b) {
  auto node = make_shared<EQ>(a, b);
  return Expr(node);
}

void EQ::Verify() const { BinaryNodeVerify(a(), b(), "EQ"); }

Expr NE::Make(Expr a, Expr b) {
  auto node = make_shared<NE>(a, b);
  return Expr(node);
}

void NE::Verify() const { BinaryNodeVerify(a(), b(), "NE"); }

Expr LT::Make(Expr a, Expr b) {
  auto node = make_shared<LT>(a, b);
  return Expr(node);
}

void LT::Verify() const { BinaryNodeVerify(a(), b(), "LT"); }

Expr LE::Make(Expr a, Expr b) {
  auto node = make_shared<LE>(a, b);
  return Expr(node);
}

void LE::Verify() const { BinaryNodeVerify(a(), b(), "LE"); }

Expr GT::Make(Expr a, Expr b) {
  auto node = make_shared<GT>(a, b);
  return Expr(node);
}

void GT::Verify() const { BinaryNodeVerify(a(), b(), "GT"); }

Expr GE::Make(Expr a, Expr b) {
  auto node = make_shared<GE>(a, b);
  return Expr(node);
}

void GE::Verify() const { BinaryNodeVerify(a(), b(), "GE"); }

Expr And::Make(Expr a, Expr b) {
  auto node = make_shared<And>(a, b);
  return Expr(node);
}

void And::Verify() const {
  BinaryNodeVerify(a(), b(), "And");
  PADDLE_ENFORCE_EQ(
      a().type(),
      type_of<bool>(),
      ::common::errors::InvalidArgument(
          "The type of the operands of the node [And] should be bool"));
}

Expr Or::Make(Expr a, Expr b) {
  auto node = make_shared<Or>(a, b);
  return Expr(node);
}

void Or::Verify() const {
  BinaryNodeVerify(a(), b(), "Or");
  PADDLE_ENFORCE_EQ(
      a().type(),
      type_of<bool>(),
      ::common::errors::InvalidArgument(
          "The type of the operands of the node [Or] should be bool"));
}

Type Or::type() const { return type_; }

Expr Not::Make(Expr v) {
  auto node = make_shared<Not>(v);
  return Expr(node);
}

void Not::Verify() const {
  PADDLE_ENFORCE_EQ(
      v().type(),
      type_of<bool>(),
      ::common::errors::InvalidArgument(
          "The type of the operand of the node [Not] should be bool"));
}

Type Not::type() const { return type_; }

Expr Let::Make(Expr symbol, Expr body) {
  auto *n = make_shared<Let>();
  PADDLE_ENFORCE_EQ(
      symbol.type().valid(),
      true,
      ::common::errors::InvalidArgument(
          "The type of the symbol is not valid. "
          "A valid type for the symbol is required to create a Let node."));
  if (body.defined()) {
    PADDLE_ENFORCE_EQ(body.type().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The type of the body is not valid. "
                          "If a body is defined, it must have a valid type."));
  }
  n->symbol = symbol;
  n->body = body;
  n->set_type(n->symbol->type());
  return Expr(n);
}

void Let::Verify() const {
  PADDLE_ENFORCE_EQ(symbol.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The symbol is not defined. "
                        "A defined symbol is required for the Let node."));
  // The default value(contained in body) is not required.
  if (body.defined()) {
    TryElevateInt32ToInt64({symbol, body});
    PADDLE_ENFORCE_EQ(
        symbol.type(),
        body.type(),
        ::common::errors::InvalidArgument(
            "The type of the symbol and the body of "
            "the node [Let] should be the same. "
            "The types must match to ensure consistency within the Let node."));
  }
}

Type Let::type() const { return symbol.type(); }

Expr _Var_::Make(const std::string &name, const Type &type) {
  auto node = new _Var_(name, type);
  return Expr(node);
}

Expr _Var_::Make(Expr lower_bound,
                 Expr upper_bound,
                 const std::string &name,
                 bool is_reduce_axis,
                 bool is_symbolic_constant,
                 bool is_keepdim) {
  auto *n = make_shared<_Var_>();
  n->lower_bound = lower_bound;
  n->upper_bound = upper_bound;
  n->is_reduce_axis = is_reduce_axis;
  n->is_keepdim = is_keepdim;
  n->is_symbolic_constant = is_symbolic_constant;
  n->name = name;
  n->set_type(lower_bound.type());
  return Expr(n);
}

Expr _Var_::Copy() const {
  auto *n = make_shared<_Var_>();
  n->name = name;
  n->is_reduce_axis = is_reduce_axis;
  n->is_keepdim = is_keepdim;
  n->set_index(get_index());
  n->lower_bound = lower_bound;
  n->upper_bound = upper_bound;
  n->set_type(type());
  return Expr(n);
}

void _Var_::Verify() const {
  PADDLE_ENFORCE_EQ(!name.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The variable should have a name. "
                        "A valid name is required to identify the variable."));
}

const IndexExpr Var::as_index() const {
  if (is_index()) {
    return IndexExpr(*this);
  }
  PADDLE_THROW(
      ::common::errors::InvalidType("Var: %s is not IndexExpr!", *this));
}

IndexExpr Var::as_index() {
  if (is_index()) {
    return IndexExpr(*this);
  }
  PADDLE_THROW(
      ::common::errors::InvalidType("Var: %s is not IndexExpr!", *this));
}

Expr IterMark::Make(const Expr &source, const IndexExpr &extent) {
  auto *n = make_shared<IterMark>();
  n->source = source;
  n->extent = extent;
  n->set_type(source.type());
  return Expr(n);
}

IterMark &IterMark::operator=(const IterMark &other) {
  this->set_type(other.type());
  source = other.source;
  extent = other.extent;
  return *this;
}
Expr IterSplit::Make(const Expr &source,
                     const IndexExpr &lower_factor,
                     const IndexExpr &extent,
                     const IndexExpr &scale) {
  auto *n = make_shared<IterSplit>();
  n->set_type(source.type());
  n->source = source;
  n->lower_factor = lower_factor;
  n->extent = extent;
  n->scale = scale;
  return Expr(n);
}

Expr IterSplit::Make(const Expr &source) {
  auto *n = make_shared<IterSplit>();
  auto source_mark = source.As<IterMark>();
  n->set_type(source.type());
  n->source = source;
  n->extent = source_mark->extent;
  n->lower_factor = One(source.type());
  n->scale = One(source.type());
  return Expr(n);
}

Expr IterSplit::Make(const Expr &source, const IndexExpr &scale) {
  auto *n = make_shared<IterSplit>();
  auto source_mark = source.As<IterMark>();
  n->set_type(source.type());
  n->source = source;
  n->extent = source_mark->extent;
  n->lower_factor = One(source.type());
  n->scale = scale;
  return Expr(n);
}
IterSplit &IterSplit::operator=(const IterSplit &other) {
  this->set_type(other.type());
  source = other.source;
  lower_factor = other.lower_factor;
  extent = other.extent;
  scale = other.scale;
  return *this;
}

Expr IterSum::Make(const std::vector<Expr> &args, const IndexExpr &base) {
  auto *n = make_shared<IterSum>();
  n->set_type(base.type());
  n->args = std::move(args);
  n->base = base;
  return Expr(n);
}

Expr For::Make(Var loop_var,
               Expr min,
               Expr extent,
               ForType for_type,
               DeviceAPI device_api,
               Expr body,
               VectorizeInfo vector_info,
               BindInfo bind_info) {
  ir::TryElevateInt32ToInt64({loop_var, min, extent});
  auto node = make_shared<For>();

  PADDLE_ENFORCE_EQ(
      loop_var.defined(),
      true,
      ::common::errors::InvalidArgument("The loop variable is not defined. "
                                        "A valid loop variable is required."));
  PADDLE_ENFORCE_EQ(
      min.defined(),
      true,
      ::common::errors::InvalidArgument("The minimum value is not defined. "
                                        "A valid minimum value is required."));
  PADDLE_ENFORCE_EQ(
      extent.defined(),
      true,
      ::common::errors::InvalidArgument("The extent is not defined. "
                                        "A valid extent is required."));

  if (!(loop_var->lower_bound.defined())) loop_var->lower_bound = min;
  if (!(loop_var->upper_bound.defined())) loop_var->upper_bound = extent;

  node->loop_var = loop_var;
  node->min = min;
  node->extent = extent;
  node->device_api = device_api;
  node->body = body.As<ir::Block>() ? body : ir::Block::Make({body});
  node->set_for_type(for_type);
  node->set_vectorize_info(vector_info);
  node->set_bind_info(bind_info);

  if (node->is_vectorized()) {
    PADDLE_ENFORCE_EQ(node->vectorize_info().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The vectorize info is not valid. "
                          "Ensure that the vectorization "
                          "information is correctly specified."));
  }
  if (node->is_binded() && bind_info.offset >= 0) {
    PADDLE_ENFORCE_EQ(
        node->bind_info().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The bind info is not valid. "
            "Ensure that the binding information is correctly specified."));
  }

  return Expr(node);
}

std::vector<Expr *> For::expr_fields() { return {&min, &extent, &body}; }
std::vector<const Expr *> For::expr_fields() const {
  return {&min, &extent, &body};
}

Expr Block::Make(const std::vector<Expr> &stmts) {
  auto node = make_shared<Block>();
  node->stmts = stmts;
  return Expr(node);
}
std::vector<Expr *> Block::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : stmts) res.push_back(&x);
  return res;
}
std::vector<const Expr *> Block::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : stmts) res.push_back(&x);
  return res;
}

Expr ScheduleBlock::Make(const std::vector<Var> &iter_vars,
                         const std::vector<Expr> &read_buffers,
                         const std::vector<Expr> &write_buffers,
                         const std::string &name,
                         Expr body) {
  auto node = make_shared<ScheduleBlock>();
  node->iter_vars = iter_vars;
  node->read_buffers = read_buffers;
  node->write_buffers = write_buffers;
  node->name = name;
  node->body = body;
  return Expr(node);
}
void ScheduleBlock::Verify() const {
  PADDLE_ENFORCE_EQ(
      !name.empty(),
      true,
      ::common::errors::InvalidArgument(
          "The name is empty. A valid name is required for the ScheduleBlock "
          "to "
          "ensure proper identification and referencing within the code."));
  PADDLE_ENFORCE_EQ(body.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body is not defined. "
                        "A defined body is required for the ScheduleBlock."));
}

std::vector<Expr *> ScheduleBlock::expr_fields() {
  std::vector<Expr *> res;
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> ScheduleBlock::expr_fields() const {
  std::vector<const Expr *> res;
  res.push_back(&body);
  return res;
}

Expr ScheduleBlockRealize::Make(const std::vector<Expr> &iter_values,
                                const Expr &schedule_block) {
  auto node = make_shared<ScheduleBlockRealize>();
  node->iter_values = iter_values;
  node->schedule_block = schedule_block;
  return Expr(node);
}
void ScheduleBlockRealize::Verify() const {
  auto *schedule_block_ptr = schedule_block.As<ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(schedule_block_ptr,
                          ::common::errors::InvalidArgument(
                              "The schedule block pointer is null. "
                              "A valid schedule block pointer is required."));
  PADDLE_ENFORCE_EQ(
      schedule_block_ptr->iter_vars.size(),
      iter_values.size(),
      ::common::errors::InvalidArgument(
          "The size of iter_values should be equal to the size of iter_vars. "
          "Expected size: %d, but got: %d",
          schedule_block_ptr->iter_vars.size(),
          iter_values.size()));
}

std::vector<Expr *> ScheduleBlockRealize::expr_fields() {
  std::vector<Expr *> res;
  auto *schedule_block_ptr = schedule_block.As<ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(schedule_block_ptr,
                          ::common::errors::InvalidArgument(
                              "The schedule block pointer is null. "
                              "A valid schedule block pointer is required."));
  res.push_back(&schedule_block_ptr->body);
  return res;
}

std::vector<const Expr *> ScheduleBlockRealize::expr_fields() const {
  std::vector<const Expr *> res;
  auto *schedule_block_ptr = schedule_block.As<ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(schedule_block_ptr,
                          ::common::errors::InvalidArgument(
                              "The schedule block pointer is null. "
                              "A valid schedule block pointer is required."));
  res.push_back(&schedule_block_ptr->body);
  return res;
}

Expr IfThenElse::Make(Expr condition, Expr true_case, Expr false_case) {
  if (true_case.defined() && (!true_case.As<Block>()))
    true_case = ir::Block::Make({true_case});
  if (false_case.defined() && (!false_case.As<Block>()))
    false_case = ir::Block::Make({false_case});
  auto node = make_shared<IfThenElse>(condition, true_case, false_case);
  return Expr(node);
}

IfThenElse::IfThenElse(Expr condition, Expr true_case, Expr false_case)
    : ExprNode(Type()),
      condition(condition),
      true_case(true_case),
      false_case(false_case) {
  PADDLE_ENFORCE_EQ(
      condition.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The condition is not defined. "
          "A valid condition expression is required for IfThenElse."));
  PADDLE_ENFORCE_EQ(
      true_case.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The true_case is not defined. "
          "A valid true_case expression is required for IfThenElse."));
}

std::vector<Expr *> IfThenElse::expr_fields() {
  return {&condition, &true_case, &false_case};
}
std::vector<const Expr *> IfThenElse::expr_fields() const {
  return {&condition, &true_case, &false_case};
}

Expr Store::Make(Expr tensor, Expr value, const std::vector<Expr> &indices) {
  PADDLE_ENFORCE_NOT_NULL(tensor.As<_Tensor_>(),
                          ::common::errors::InvalidArgument(
                              "The tensor should be of type _Tensor_. "
                              "Ensure that the tensor is correctly defined."));
  auto node = make_shared<Store>();
  node->tensor = tensor;
  node->value = value;
  node->indices =
      utils::GetCompatibleStoreLoadIndices(tensor.as_tensor_ref(), indices);

  if (tensor->type() != Void()) {
    node->set_type(
        tensor->type().ElementOf().with_lanes(node->index().type().lanes()));
  }
  return Expr(node);
}

Expr Store::index() const {
  auto *tensor_n = tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(tensor_n,
                          ::common::errors::InvalidArgument(
                              "The tensor pointer is null. "
                              "Ensure that the tensor is correctly defined."));
  if (indices.size() == 1) {
    return indices[0];
  }
  Expr res = cinn::common::IndiceToAbsOffset(tensor_n->shape, indices);
  return res;
}

void Store::replace(Expr old_op, Expr new_op) {
  if (value == old_op) {
    value = new_op;
  }
  if (tensor == old_op) {
    tensor = new_op;
  }
  for (int i = 0; i < indices.size(); i++) {
    if (indices[i] == old_op) {
      indices[i] = new_op;
    }
  }
}

void Select::replace(Expr old_op, Expr new_op) {
  if (condition == old_op) {
    condition = new_op;
  }
  if (true_value == old_op) {
    true_value = new_op;
  }
  if (false_value == old_op) {
    false_value = new_op;
  }
}

void Cast::replace(Expr old_op, Expr new_op) {
  if (v() == old_op) {
    v() = new_op;
  }
}

const std::string &Store::name() const {
  auto *t = tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(
      t,
      ::common::errors::InvalidArgument(
          "The tensor pointer is null. "
          "A valid tensor pointer is required to get the name."));
  return t->name;
}

Type Store::type() const { return value.type(); }

std::vector<Expr *> Store::expr_fields() {
  std::vector<Expr *> exprs({&tensor, &value});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

std::vector<const Expr *> Store::expr_fields() const {
  std::vector<const Expr *> exprs({&tensor, &value});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

void Store::Verify() const {
  PADDLE_ENFORCE_EQ(
      tensor.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The tensor is not defined. "
          "A defined tensor is required for the Store operation."));
}

Expr Alloc::Make(Expr dest,
                 Type type,
                 const std::vector<Expr> &extents,
                 Expr condition,
                 Expr body) {
  auto node = make_shared<Alloc>();
  PADDLE_ENFORCE_NOT_NULL(dest.As<_Buffer_>(),
                          ::common::errors::InvalidArgument(
                              "Alloc destination only supports Buffer. "
                              "Ensure the destination is of type Buffer."));
  node->destination = dest;
  node->extents = extents;
  node->condition = condition;
  node->body = body;
  node->set_type(type);
  return Expr(node);
}

int32_t Alloc::ConstantAllocationSize() const {
  return ConstantAllocationSize(extents);
}

int32_t Alloc::ConstantAllocationSize(const std::vector<Expr> &extents) {
  int32_t res{1};
  for (auto &e : extents) {
    auto *p = e.As<IntImm>();
    PADDLE_ENFORCE_NOT_NULL(p,
                            ::common::errors::InvalidArgument(
                                "Extent should be IntImm. "
                                "Each extent must be an instance of IntImm."));
    res *= p->value;
  }
  return res;
}

std::vector<Expr *> Alloc::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : extents) res.push_back(&x);
  res.push_back(&condition);
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> Alloc::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : extents) res.push_back(&x);
  res.push_back(&condition);
  res.push_back(&body);
  return res;
}

Expr Free::Make(Expr dest) {
  auto node = make_shared<Free>();
  PADDLE_ENFORCE_NOT_NULL(dest.As<_Buffer_>(),
                          ::common::errors::InvalidArgument(
                              "Free destination only supports Buffer. "
                              "Ensure the destination is of type Buffer."));
  node->destination = dest;
  return Expr(node);
}

Expr Call::Make(Type type,
                const std::string &name,
                const std::vector<Expr> &read_args,
                const std::vector<Expr> &write_args,
                CallType call_type,
                FunctionRef func,
                int value_index,
                const std::map<std::string, attr_t> &attrs) {
  for (size_t i = 0; i < read_args.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        read_args[i].defined(),
        true,
        ::common::errors::InvalidArgument("Read argument %d is not defined. "
                                          "All read arguments must be defined.",
                                          i));
  }

  auto node = cinn::common::make_shared<Call>(type);
  node->name = name;
  node->read_args = read_args;
  node->write_args = write_args;
  node->call_type = call_type;
  node->func = func;
  node->value_index = value_index;
  node->set_type(type);
  node->attrs = attrs;
  return Expr(node);
}

void Call::replace(Expr old_op, Expr new_op) {
  for (int i = 0; i < read_args.size(); i++) {
    if (read_args[i] == old_op) {
      read_args[i] = new_op;
    }
  }
  for (int i = 0; i < write_args.size(); i++) {
    if (read_args[i] == old_op) {
      read_args[i] = new_op;
    }
  }
}

std::vector<Expr *> Call::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : read_args) res.push_back(&x);
  for (auto &x : write_args) res.push_back(&x);
  return res;
}
std::vector<const Expr *> Call::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : read_args) res.push_back(&x);
  for (auto &x : write_args) res.push_back(&x);
  return res;
}
void Call::Verify() const {}

Expr PolyFor::Make(Var iterator,
                   Expr init_val,
                   Expr condition,
                   Expr inc,
                   ForType for_type,
                   DeviceAPI device_api,
                   Expr body,
                   VectorizeInfo vectorize_info,
                   BindInfo bind_info) {
  auto n = make_shared<PolyFor>();
  n->iterator = iterator;
  n->init = init_val;
  n->condition = condition;
  n->inc = inc;
  n->device_api = device_api;
  n->body = body.As<ir::Block>() ? body : ir::Block::Make({body});
  n->set_for_type(for_type);
  n->set_vectorize_info(vectorize_info);
  n->set_bind_info(bind_info);

  if (n->is_vectorized()) {
    PADDLE_ENFORCE_EQ(n->vectorize_info().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The vectorize info is not valid. "
                          "Ensure that the vectorization "
                          "information is correctly specified."));
  }
  if (n->is_binded() && bind_info.offset >= 0) {
    PADDLE_ENFORCE_EQ(
        n->bind_info().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The bind info is not valid. "
            "Ensure that the binding information is correctly specified."));
  }

  return Expr(n);
}

std::vector<Expr *> PolyFor::expr_fields() {
  return {&init, &condition, &inc, &body};
}
std::vector<const Expr *> PolyFor::expr_fields() const {
  return {&init, &condition, &inc, &body};
}

Expr PolyFor::ExtractExtent() const {
  auto nodes = ir::ir_utils::CollectIRNodes(condition, [&](const Expr *e) {
    return e->As<NE>() ||   //
           e->As<EQ>() ||   //
           e->As<Min>() ||  //
           e->As<Max>();
  });

  if (!nodes.empty()) {
    return Expr();
  }

  auto *le_n = condition.As<LE>();
  auto *lt_n = condition.As<LT>();
  if (!(le_n || lt_n)) return Expr();

  if (le_n) {
    if (le_n->a() != Expr(iterator)) return Expr();
    auto *le_b_int = le_n->b().As<IntImm>();
    if (le_b_int)
      return Expr(make_shared<IntImm>(Int(32), le_b_int->value + 1));
    return Add::Make(le_n->b(), Expr(1));
  }

  if (lt_n) {
    if (lt_n->a() != Expr(iterator)) return Expr();
    return lt_n->b();
  }
  return Expr();
}

bool Var::operator==(const Var &o) const {
  return o->name == operator->()->name;
}
bool Var::operator!=(const Var &o) const { return !(*this == o); }

Var &Var::operator=(_Var_ *x) {
  *this = Var(x);
  return *this;
}

Var &Var::operator=(const _Var_ *x) {
  *this = x->Copy();
  return *this;
}

Expr Load::Make(Expr tensor, const std::vector<Expr> &origin_indices) {
  PADDLE_ENFORCE_EQ(
      tensor->type().valid(),
      true,
      ::common::errors::InvalidArgument("The tensor type is not valid. "
                                        "A valid tensor type is required."));
  const auto indices = utils::GetCompatibleStoreLoadIndices(
      tensor.as_tensor_ref(), origin_indices);
  PADDLE_ENFORCE_EQ(
      !indices.empty(),
      true,
      ::common::errors::InvalidArgument("The indices should not be empty. "
                                        "At least one index is required."));
  TryElevateInt32ToInt64(indices);
  for (auto &idx : indices) {
    PADDLE_ENFORCE_EQ(
        idx.type().ElementOf() == Int(64) || idx.type().ElementOf() == Int(32),
        true,
        ::common::errors::InvalidArgument(
            "The index type should be either int64 or int32. "
            "Received index type: %s",
            idx.type().to_string().c_str()));
  }
  auto node = make_shared<Load>();
  node->tensor = tensor;
  node->indices = indices;
  node->set_type(node->type());
  return Expr(node);
}

void Load::convert_int32_to_int64() {
  IrNode::convert_int32_to_int64();
  for (auto &indice : indices) {
    indice->convert_int32_to_int64();
  }
  tensor->convert_int32_to_int64();
}

void Load::convert_int64_to_int32() {
  IrNode::convert_int64_to_int32();
  for (auto &indice : indices) {
    indice->convert_int64_to_int32();
  }
  tensor->convert_int64_to_int32();
}

Type Load::type() const {
  PADDLE_ENFORCE_EQ(
      tensor.defined(),
      true,
      ::common::errors::InvalidArgument("The tensor is not defined. "
                                        "A defined tensor is required."));
  PADDLE_ENFORCE_EQ(
      tensor.type().valid(),
      true,
      ::common::errors::InvalidArgument("The tensor type is not valid. "
                                        "A valid tensor type is required."));

  int lanes = 0;
  for (auto &idx : indices) {
    lanes = std::max(lanes, idx.type().lanes());
  }
  auto type = tensor.type().ElementOf().with_lanes(lanes);
  if (type.is_cpp_handle()) {
    return type.set_cpp_handle(false);
  }
  if (type.is_cpp_handle2()) {
    return type.set_cpp_handle(true);
  }
  return type;
}

std::vector<Expr *> Load::expr_fields() {
  std::vector<Expr *> exprs({&tensor});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

std::vector<const Expr *> Load::expr_fields() const {
  std::vector<const Expr *> exprs({&tensor});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

Expr Load::index() const {
  if (is_addr_tensor()) {
    auto *tensor_n = tensor.As<_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(tensor_n,
                            ::common::errors::InvalidArgument(
                                "The tensor pointer is null. "
                                "A valid tensor pointer is required."));
    VLOG(3) << "Begin Load::index IndiceToAbsOffset of tensor: "
            << this->name();
    if (indices.size() == 1) {
      return indices[0];
    }
    Expr res = cinn::common::IndiceToAbsOffset(tensor_n->shape, indices);
    return res;
  } else {
    PADDLE_ENFORCE_EQ(indices.size(),
                      1UL,
                      ::common::errors::InvalidArgument(
                          "The indices size of Load node should be 1"));
    return indices[0];
  }
}

const std::string &Load::name() const {
  auto *t = tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(
      t,
      ::common::errors::InvalidArgument("The tensor pointer is null. "
                                        "A valid tensor pointer is required."));
  return t->name;
}

void Load::Verify() const {
  PADDLE_ENFORCE_EQ(
      tensor.defined(),
      true,
      ::common::errors::InvalidArgument("The tensor is not defined."));
  PADDLE_ENFORCE_EQ(
      !indices.empty(),
      true,
      ::common::errors::InvalidArgument("At least one index is needed."));
  for (auto &indice : indices) {
    PADDLE_ENFORCE_EQ(indice.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "One of the indices is not defined."));
    PADDLE_ENFORCE_EQ(
        indice.type().ElementOf() == type_of<int32_t>() ||
            indice.type().ElementOf() == type_of<int64_t>(),
        true,
        ::common::errors::InvalidArgument(
            "The index type should be either int32 or int64. Received type: %s",
            indice.type().to_string().c_str()));
  }
}

bool LoadStoreAddrMnger::is_addr_tensor() const {
  return tensor.As<_Tensor_>();
}
bool LoadStoreAddrMnger::is_addr_scalar() const { return !is_addr_tensor(); }

Expr Ramp::Make(Expr base, Expr stride, int lanes) {
  PADDLE_ENFORCE_EQ(
      base.defined(),
      true,
      ::common::errors::InvalidArgument("The base expression is not defined."));
  PADDLE_ENFORCE_EQ(stride.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The stride expression is not defined."));
  PADDLE_ENFORCE_EQ(
      base.type().valid(),
      true,
      ::common::errors::InvalidArgument("The base type is not valid."));
  PADDLE_ENFORCE_EQ(
      stride.type().valid(),
      true,
      ::common::errors::InvalidArgument("The stride type is not valid."));
  PADDLE_ENFORCE_EQ(stride.type(),
                    Int(32),
                    ::common::errors::InvalidArgument(
                        "The stride of the node [Ramp] should be int32"));
  PADDLE_ENFORCE_GT(
      lanes,
      0,
      ::common::errors::InvalidArgument(
          "The lanes of the node [Ramp] should be greater than 0"));

  auto *n = make_shared<Ramp>();
  n->base = base;
  n->stride = stride;
  n->lanes = lanes;
  Type type(base.type().type(), base.type().bits(), lanes);
  n->set_type(type);
  return Expr(n);
}

Expr Broadcast::Make(Expr value, int lanes) {
  PADDLE_ENFORCE_EQ(value.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The value expression is not defined."));
  PADDLE_ENFORCE_EQ(
      value.type().valid(),
      true,
      ::common::errors::InvalidArgument("The value type is not valid."));

  auto *n = make_shared<Broadcast>();
  n->value = value;
  n->lanes = lanes;

  Type type(value.type().type(), value.type().bits(), lanes);
  n->set_type(type);

  return Expr(n);
}

Type Broadcast::type() const {
  return value.type().ElementOf().with_lanes(lanes);
}

Expr Sum::Make(const std::vector<Expr> &vs) {
  PADDLE_ENFORCE_EQ(
      !vs.empty(),
      true,
      ::common::errors::InvalidArgument("The vector of operands is empty. "
                                        "At least one operand is required."));
  if (vs.size() == 1) return vs.front();

  auto *n = make_shared<Sum>();
  TryElevateInt32ToInt64(vs);
  auto type = vs.front().type();
  for (auto &v : vs) {
    PADDLE_ENFORCE_EQ(v.type(),
                      type,
                      ::common::errors::InvalidArgument(
                          "The operands' types of the node [Sum] don't match. "
                          "Expected type: %s, but got type: %s",
                          type.to_string().c_str(),
                          v.type().to_string().c_str()));
  }

  n->operands() = vs;
  n->set_type(vs.front()->type());

  return Expr(n);
}

Expr Product::Make(const std::vector<Expr> &vs) {
  PADDLE_ENFORCE_GE(
      vs.size(),
      1,
      ::common::errors::InvalidArgument("The operands of the node [Product] "
                                        "should have at least one element"));

  auto *n = make_shared<Product>();
  TryElevateInt32ToInt64(vs);
  auto type = vs.front().type();
  for (auto &v : vs)
    PADDLE_ENFORCE_EQ(
        v.type(),
        type,
        ::common::errors::InvalidArgument("The operands' types of the node "
                                          "[Product] don't match"));

  n->operands() = vs;

  n->set_type(vs.front()->type());

  return Expr(n);
}

Expr FracOp::Make(Expr n, Expr d) {
  auto *node = make_shared<FracOp>();
  node->a() = n;
  node->b() = d;
  return Expr(node);
}

ir::Module _Module_::Make(const std::string &name, Target target) {
  auto n = make_shared<_Module_>();
  n->name = name;
  n->target = target;
  return ir::Module(n);
}

Expr PrimitiveNode::Make(const std::string &name,
                         const std::map<std::string, attr_t> &attrs) {
  auto *n = make_shared<PrimitiveNode>();
  n->name = name;
  n->attrs = attrs;
  return Expr(n);
}

Expr Reduce::Make(Reduce::ReduceType reduce_type,
                  Expr init,
                  Expr body,
                  const std::vector<Var> &reduce_axis) {
  PADDLE_ENFORCE_EQ(
      body.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The body expression is not defined. "
          "A valid body expression is required for the Reduce node."));
  PADDLE_ENFORCE_EQ(
      init.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The init expression is not defined. "
          "A valid init expression is required for the Reduce node."));

  auto n = cinn::common::make_shared<Reduce>();
  n->init = init;
  n->body = body;
  n->reduce_type = reduce_type;
  n->reduce_axis.append(reduce_axis.begin(), reduce_axis.end());

  PADDLE_ENFORCE_EQ(body.type().valid(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body type is not valid. "
                        "Ensure that the body expression has a valid type."));

  if (init.defined()) {
    PADDLE_ENFORCE_EQ(init.type().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The init type is not valid. "
                          "Ensure that the init expression has a valid type."));
    PADDLE_ENFORCE_EQ(init.type(),
                      body.type(),
                      ::common::errors::InvalidArgument(
                          "The type of the init and the body of the "
                          "node [Reduce] should be the same. "
                          "Received init type: %s, body type: %s",
                          init.type().to_string().c_str(),
                          body.type().to_string().c_str()));
  }

  n->set_type(body.type());

  return Expr(n);
}

Type Reduce::type() const { return body.type().ElementOf(); }

std::vector<Expr *> Reduce::expr_fields() {
  std::vector<Expr *> res;
  if (init.defined()) {
    res.push_back(&init);
  }
  PADDLE_ENFORCE_EQ(body.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body expression is not defined. "
                        "The Reduce node requires a defined body expression."));
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> Reduce::expr_fields() const {
  std::vector<const Expr *> res;
  if (init.defined()) {
    res.push_back(&init);
  }
  PADDLE_ENFORCE_EQ(body.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body expression is not defined. "
                        "The Reduce node requires a defined body expression."));
  res.push_back(&body);
  return res;
}

void Reduce::Verify() const {
  PADDLE_ENFORCE_EQ(init.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The init expression is not defined. "
                        "The Reduce node requires a defined init expression."));
  PADDLE_ENFORCE_EQ(body.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body expression is not defined. "
                        "The Reduce node requires a defined body expression."));
  PADDLE_ENFORCE_EQ(!reduce_axis.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "At least one reduce axis is needed. "
                        "Ensure that the reduce_axis vector is not empty."));
  PADDLE_ENFORCE_EQ(init.type(),
                    body.type(),
                    ::common::errors::InvalidArgument(
                        "The type of the init and the body of the "
                        "node [Reduce] should be the same. "
                        "Received init type: %s, body type: %s",
                        init.type().to_string().c_str(),
                        body.type().to_string().c_str()));
}

Select::Select(Expr condition, Expr true_value, Expr false_value)
    : ExprNode<Select>(true_value.type()),
      condition(condition),
      true_value(true_value),
      false_value(false_value) {
  TryElevateInt32ToInt64({true_value, false_value});
  PADDLE_ENFORCE_EQ(true_value.type(),
                    false_value.type(),
                    ::common::errors::InvalidArgument(
                        "The type of true_value and false_value should be the "
                        "same. T: %s, F: %s",
                        true_value,
                        false_value));
  PADDLE_ENFORCE_EQ(condition.type().is_bool(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "The condition must be of boolean type."));
  type_ = true_value.type();
}

Type Select::type() const {
  PADDLE_ENFORCE_EQ(
      true_value.type(), false_value.type(), "Type of Select must be same");
  return type_;
}

void Select::Verify() const {
  PADDLE_ENFORCE_EQ(
      condition.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The condition expression is not defined. "
          "A valid condition expression is required for the Select node."));
  PADDLE_ENFORCE_EQ(
      true_value.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The true_value expression is not defined. "
          "A valid true_value expression is required for the Select node."));
  PADDLE_ENFORCE_EQ(
      false_value.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The false_value expression is not defined. "
          "A valid false_value expression is required for the Select node."));
  PADDLE_ENFORCE_EQ(
      condition.type().is_bool(),
      true,
      ::common::errors::InvalidArgument(
          "The condition of the Select Node should be a boolean."));
  PADDLE_ENFORCE_EQ(
      true_value.type(),
      false_value.type(),
      ::common::errors::InvalidArgument(
          "The true_value and false_value of the Select Node should have "
          "the same type. Received true_value type: %s, false_value type: %s",
          true_value.type().to_string().c_str(),
          false_value.type().to_string().c_str()));
}

void Free::Verify() const {
  PADDLE_ENFORCE_EQ(destination.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The destination is not defined. "
                        "A valid destination is required for the Free node."));
}

void Alloc::Verify() const {
  PADDLE_ENFORCE_EQ(destination.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The destination is not defined. "
                        "A valid destination is required for the Alloc node."));
}

void For::Verify() const {
  PADDLE_ENFORCE_EQ(loop_var.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The loop variable is not defined. "
                        "A valid loop variable is required for the For node."));
  PADDLE_ENFORCE_EQ(min.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The minimum value is not defined. "
                        "A valid minimum value is required for the For node."));
  PADDLE_ENFORCE_EQ(extent.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The extent is not defined. "
                        "A valid extent is required for the For node."));
  PADDLE_ENFORCE_EQ(body.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body is not defined. "
                        "A valid body is required for the For node."));

  PADDLE_ENFORCE_EQ((loop_var->type() == type_of<int32_t>()) ||
                        (loop_var->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "The loop variable's type must be int32 or int64. "
                        "Received type: %s",
                        loop_var->type().to_string().c_str()));
  PADDLE_ENFORCE_EQ((min->type() == type_of<int32_t>()) ||
                        (min->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "The minimum value's type must be int32 or int64. "
                        "Received type: %s",
                        min->type().to_string().c_str()));
  PADDLE_ENFORCE_EQ((extent->type() == type_of<int32_t>()) ||
                        (extent->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "The extent's type must be int32 or int64. "
                        "Received type: %s",
                        extent->type().to_string().c_str()));
}

void PolyFor::Verify() const {
  PADDLE_ENFORCE_EQ(iterator.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The iterator is not defined. "
                        "A valid iterator is required for the PolyFor node."));
  PADDLE_ENFORCE_EQ(
      init.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The init expression is not defined. "
          "A valid init expression is required for the PolyFor node."));
  PADDLE_ENFORCE_EQ(
      condition.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The condition expression is not defined. "
          "A valid condition expression is required for the PolyFor node."));
  PADDLE_ENFORCE_EQ(
      inc.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The increment expression is not defined. "
          "A valid increment expression is required for the PolyFor node."));
  PADDLE_ENFORCE_EQ(body.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body is not defined. "
                        "A valid body is required for the PolyFor node."));

  PADDLE_ENFORCE_EQ(
      iterator->type(),
      type_of<int32_t>(),
      ::common::errors::InvalidArgument("The iterator's type must be int32. "
                                        "Received type: %s",
                                        iterator->type().to_string().c_str()));
  PADDLE_ENFORCE_EQ(init.type(),
                    type_of<int32_t>(),
                    ::common::errors::InvalidArgument(
                        "The init expression's type must be int32. "
                        "Received type: %s",
                        init.type().to_string().c_str()));
  PADDLE_ENFORCE_EQ(condition.type(),
                    type_of<bool>(),
                    ::common::errors::InvalidArgument(
                        "The condition expression's type must be bool. "
                        "Received type: %s",
                        condition.type().to_string().c_str()));
  PADDLE_ENFORCE_EQ(inc.type(),
                    type_of<int32_t>(),
                    ::common::errors::InvalidArgument(
                        "The increment expression's type must be int32. "
                        "Received type: %s",
                        inc.type().to_string().c_str()));
}

void Ramp::Verify() const {
  PADDLE_ENFORCE_EQ(
      base.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The base expression is not defined. "
          "A valid base expression is required for the Ramp node."));
  PADDLE_ENFORCE_EQ(
      stride.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The stride expression is not defined. "
          "A valid stride expression is required for the Ramp node."));
}

void FracOp::Verify() const {
  PADDLE_ENFORCE_EQ(
      a().defined(),
      true,
      ::common::errors::InvalidArgument(
          "The operand 'a' is not defined. "
          "A valid operand 'a' is required for the FracOp node."));
  PADDLE_ENFORCE_EQ(
      b().defined(),
      true,
      ::common::errors::InvalidArgument(
          "The operand 'b' is not defined. "
          "A valid operand 'b' is required for the FracOp node."));
  PADDLE_ENFORCE_EQ(
      a().type(),
      b().type(),
      ::common::errors::InvalidArgument(
          "The type of the operands of the node [FracOp] should be the same. "
          "Received 'a' type: %s, 'b' type: %s",
          a().type().to_string().c_str(),
          b().type().to_string().c_str()));
}

void Broadcast::Verify() const {
  PADDLE_ENFORCE_EQ(value.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The value is not defined. "
                        "A valid value is required for the Broadcast node."));
}

void MultiOperandVerify(llvm::ArrayRef<Expr> operands) {
  Type operand_type = operands.front().type();
  PADDLE_ENFORCE_EQ(
      operand_type.valid(),
      true,
      ::common::errors::InvalidArgument("The operand type is not valid. "
                                        "A valid operand type is required."));
  for (int i = 1; i < operands.size(); i++) {
    PADDLE_ENFORCE_EQ(operands[i].defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "One of the operands is not defined. "
                          "All operands must be defined for the node."));
    PADDLE_ENFORCE_EQ(operands[i].type(),
                      operand_type,
                      ::common::errors::InvalidArgument(
                          "The operands' types of the node don't match. "
                          "Expected type: %s, but got type: %s",
                          operand_type.to_string().c_str(),
                          operands[i].type().to_string().c_str()));
  }
}

Type Product::type() const { return operands().front().type(); }

void Product::Verify() const {
  PADDLE_ENFORCE_GT(operands().size(),
                    1UL,
                    ::common::errors::InvalidArgument(
                        "Product node should have more than 1 operands"));
  MultiOperandVerify(operands());
}

Type Sum::type() const { return operands().front().type(); }

void Sum::Verify() const {
  PADDLE_ENFORCE_GT(operands().size(),
                    1UL,
                    ::common::errors::InvalidArgument(
                        "Sum node should have more than 1 operands"));
  MultiOperandVerify(operands());
}

void Block::Verify() const {}

void PrimitiveNode::Verify() const {}
}  // namespace ir
}  // namespace cinn
