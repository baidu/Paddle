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

#include "paddle/cinn/ir/stmt.h"

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_utils.h"

namespace cinn {
namespace ir {
namespace stmt {

using cinn::common::make_shared;

const BlockRef _Stmt_::GetParentBlockRef() const { return BlockRef{parent_}; }

void _Stmt_::set_block_fields(const std::vector<BlockRef> &blocks) {
  blocks_ = blocks;
  for (auto &block : blocks_) {
    block->set_parent(this);
  }
}

BlockRef _Block_::Make(const std::vector<StmtRef> &stmts) {
  BlockRef ref(new _Block_());
  ref->set_stmts(stmts);
  return ref;
}

Let _Let_::Make(Expr symbol, Expr body) {
  Let ref(new _Let_());
  PADDLE_ENFORCE_EQ(
      symbol.type().valid(),
      true,
      ::common::errors::InvalidArgument(
          "The type of the symbol is not valid. "
          "A valid type for the symbol is required to create a _Let_."));
  if (body.defined()) {
    PADDLE_ENFORCE_EQ(body.type().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The type of the body is not valid. "
                          "If a body is defined, it must have a valid type."));
  }
  ref->set_symbol(symbol);
  ref->set_body(body);
  ref->set_type(ref->symbol()->type());
  return ref;
}

void _Let_::Verify() const {
  PADDLE_ENFORCE_EQ(symbol_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The symbol is not defined. "
                        "A defined symbol is required for the _Let_."));
  // The default value(contained in body) is not required.
  if (body_.defined()) {
    TryElevateInt32ToInt64({symbol_, body_});
    PADDLE_ENFORCE_EQ(symbol_.type(),
                      body_.type(),
                      ::common::errors::InvalidArgument(
                          "The type of the symbol and the body of "
                          "the node [LetStmt] should be the same. "
                          "The types must match to ensure consistency within "
                          "the _Let_."));
  }
}

Type _Let_::type() const { return symbol_.type(); }

Store _Store_::Make(Expr tensor, Expr value, const std::vector<Expr> &indices) {
  PADDLE_ENFORCE_NOT_NULL(tensor.As<_Tensor_>(),
                          ::common::errors::InvalidArgument(
                              "The tensor should be of type _Tensor_. "
                              "Ensure that the tensor is correctly defined."));
  Store ref(new _Store_());
  ref->set_tensor(tensor);
  ref->set_value(value);
  ref->set_indices(
      utils::GetCompatibleStoreLoadIndices(tensor.as_tensor_ref(), indices));

  if (tensor->type() != Void()) {
    ref->set_type(
        tensor->type().ElementOf().with_lanes(ref->index().type().lanes()));
  }
  return ref;
}

Expr _Store_::index() const {
  auto *tensor_n = addr_mnger_.tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(tensor_n,
                          ::common::errors::InvalidArgument(
                              "The tensor pointer is null. "
                              "Ensure that the tensor is correctly defined."));
  if (indices_.size() == 1) {
    return indices_[0];
  }
  Expr res = cinn::common::IndiceToAbsOffset(tensor_n->shape, indices_);
  return res;
}

void _Store_::replace(Expr old_op, Expr new_op) {
  if (value_ == old_op) {
    value_ = new_op;
  }
  if (addr_mnger_.tensor == old_op) {
    addr_mnger_.tensor = new_op;
  }
  for (int i = 0; i < indices_.size(); i++) {
    if (indices_[i] == old_op) {
      indices_[i] = new_op;
    }
  }
}

const std::string &_Store_::name() const {
  auto *t = addr_mnger_.tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NOT_NULL(
      t,
      ::common::errors::InvalidArgument(
          "The tensor pointer is null. "
          "A valid tensor pointer is required to get the name."));
  return t->name;
}

Type _Store_::type() const { return value_.type(); }

void _Store_::Verify() const {
  PADDLE_ENFORCE_EQ(
      addr_mnger_.tensor.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The tensor is not defined. "
          "A defined tensor is required for the Store operation."));
}

Alloc _Alloc_::Make(Expr dest,
                    Type type,
                    const std::vector<Expr> &extents,
                    Expr condition,
                    Expr body) {
  Alloc ref(new _Alloc_());
  PADDLE_ENFORCE_NOT_NULL(dest.As<_Buffer_>(),
                          ::common::errors::InvalidArgument(
                              "Alloc destination only supports Buffer. "
                              "Ensure the destination is of type Buffer."));
  ref->set_destination(dest);
  ref->set_extents(extents);
  ref->set_condition(condition);
  ref->set_body(body);
  ref->set_type(type);
  return ref;
}

int32_t _Alloc_::ConstantAllocationSize() const {
  return ConstantAllocationSize(extents_);
}

int32_t _Alloc_::ConstantAllocationSize(const std::vector<Expr> &extents) {
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

void _Alloc_::Verify() const {
  PADDLE_ENFORCE_EQ(destination_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The destination is not defined. "
                        "A valid destination is required for the _Alloc_."));
}

Free _Free_::Make(Expr dest) {
  Free ref(new _Free_());
  PADDLE_ENFORCE_NOT_NULL(dest.As<_Buffer_>(),
                          ::common::errors::InvalidArgument(
                              "Free destination only supports Buffer. "
                              "Ensure the destination is of type Buffer."));
  ref->set_destination(dest);
  return ref;
}

void _Free_::Verify() const {
  PADDLE_ENFORCE_EQ(destination_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The destination is not defined. "
                        "A valid destination is required for the _Free_."));
}

IfThenElse _IfThenElse_::Make(Expr condition,
                              BlockRef true_case,
                              BlockRef false_case) {
  IfThenElse ref(new _IfThenElse_());
  PADDLE_ENFORCE_EQ(
      condition.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The condition is not defined. "
          "A valid condition expression is required for _IfThenElse_."));
  PADDLE_ENFORCE_EQ(
      true_case.defined(),
      true,
      ::common::errors::InvalidArgument(
          "The true_case is not defined. "
          "A valid true_case expression is required for _IfThenElse_."));
  ref->set_condition(condition);
  ref->set_block_fields({true_case, false_case});
  return ref;
}

void _IfThenElse_::Verify() const {
  PADDLE_ENFORCE_EQ(
      condition_.defined(),
      true,
      ::common::errors::PreconditionNotMet("The condition must be defined."));
  PADDLE_ENFORCE_EQ(
      blocks_.size(),
      2,
      ::common::errors::PreconditionNotMet("IfThenElse requires two blocks."));
  PADDLE_ENFORCE_EQ(
      blocks_[0].defined(),
      true,
      ::common::errors::PreconditionNotMet("The true_case must be defined."));
  PADDLE_ENFORCE_EQ(blocks_[1].defined(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "The false_case can be empty but must be defined."));
  PADDLE_ENFORCE_EQ(
      condition_.type(),
      type_of<bool>(),
      ::common::errors::InvalidArgument("condition should be a bool"));
}

For _For_::Make(Var loop_var,
                Expr min,
                Expr extent,
                ForType for_type,
                DeviceAPI device_api,
                BlockRef body,
                VectorizeInfo vector_info,
                BindInfo bind_info) {
  ir::TryElevateInt32ToInt64({loop_var, min, extent});
  For ref(new _For_());

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

  ref->set_loop_var(loop_var);
  ref->set_min(min);
  ref->set_extent(extent);
  ref->set_device_api(device_api);
  ref->set_body(body);
  ref->set_for_type(for_type);
  ref->set_vectorize_info(vector_info);
  ref->set_bind_info(bind_info);

  if (ref->is_vectorized()) {
    PADDLE_ENFORCE_EQ(ref->vectorize_info().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The vectorize info is not valid. "
                          "Ensure that the vectorization "
                          "information is correctly specified."));
  }
  if (ref->is_binded() && bind_info.offset >= 0) {
    PADDLE_ENFORCE_EQ(
        ref->bind_info().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The bind info is not valid. "
            "Ensure that the binding information is correctly specified."));
  }

  return ref;
}

void _For_::Verify() const {
  PADDLE_ENFORCE_EQ(loop_var_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The loop variable is not defined. "
                        "A valid loop variable is required for the _For_."));
  PADDLE_ENFORCE_EQ(min_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The minimum value is not defined. "
                        "A valid minimum value is required for the _For_."));
  PADDLE_ENFORCE_EQ(extent_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The extent is not defined. "
                        "A valid extent is required for the _For_."));
  PADDLE_ENFORCE_EQ(
      blocks_.size(),
      1,
      ::common::errors::InvalidArgument("For requires a single body."));
  PADDLE_ENFORCE_EQ(blocks_[0].defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body is not defined. "
                        "A valid body is required for the _For_."));

  PADDLE_ENFORCE_EQ((loop_var_->type() == type_of<int32_t>()) ||
                        (loop_var_->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "The loop variable's type must be int32 or int64. "
                        "Received type: %s",
                        loop_var_->type().to_string().c_str()));
  PADDLE_ENFORCE_EQ((min_->type() == type_of<int32_t>()) ||
                        (min_->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "The minimum value's type must be int32 or int64. "
                        "Received type: %s",
                        min_->type().to_string().c_str()));
  PADDLE_ENFORCE_EQ((extent_->type() == type_of<int32_t>()) ||
                        (extent_->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "The extent's type must be int32 or int64. "
                        "Received type: %s",
                        extent_->type().to_string().c_str()));
}

Schedule _Schedule_::Make(const std::vector<Var> &iter_vars,
                          const std::vector<Expr> &iter_values,
                          const std::vector<Expr> &read_buffers,
                          const std::vector<Expr> &write_buffers,
                          const std::string &name,
                          const BlockRef &body,
                          const std::map<std::string, attr_t> &attrs,
                          const ReduceMethod &reduce_method) {
  Schedule ref(new _Schedule_());
  ref->set_iter_vars(iter_vars);
  ref->set_iter_values(iter_values);
  ref->set_read_buffers(read_buffers);
  ref->set_write_buffers(write_buffers);
  ref->set_body(body);
  ref->set_name(name);
  ref->set_attrs(attrs);
  ref->set_reduce_method(reduce_method);
  return ref;
}

void _Schedule_::Verify() const {
  PADDLE_ENFORCE_EQ(
      !name_.empty(),
      true,
      ::common::errors::InvalidArgument(
          "The name is empty. A valid name is required for the "
          "_Schedule_ "
          "to "
          "ensure proper identification and referencing within the code."));
  PADDLE_ENFORCE_EQ(
      blocks_.size(),
      1,
      ::common::errors::InvalidArgument("Schedule requires a single body."));
  PADDLE_ENFORCE_EQ(blocks_[0].defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The body is not defined. "
                        "A defined body is required for the _Schedule_."));
  PADDLE_ENFORCE_EQ(
      iter_vars_.size(),
      iter_values_.size(),
      ::common::errors::InvalidArgument(
          "The size of iter_values should be equal to the size of iter_vars. "
          "Expected size: %d, but got: %d",
          iter_vars_.size(),
          iter_values_.size()));
}

Evaluate _Evaluate_::Make(Expr value) {
  Evaluate ref(new _Evaluate_());
  ref->set_value(value);
  return ref;
}

void _Evaluate_::Verify() const {
  PADDLE_ENFORCE_EQ(value_.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The value is not defined. "
                        "A defined value is required for the _Evaluate_."));
}

}  // namespace stmt
}  // namespace ir
}  // namespace cinn
