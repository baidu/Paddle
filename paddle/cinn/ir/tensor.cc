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

#include "paddle/cinn/ir/tensor.h"

#include <cstring>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/arithmetic.h"
#include "paddle/cinn/common/axis.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace ir {

Tensor _Tensor_::Make(const std::string &name,
                      Type dtype,
                      const std::vector<Expr> &shape,
                      const std::vector<Expr> &domain,
                      FunctionRef fn,
                      const std::vector<Var> &reduce_axis) {
  PADDLE_ENFORCE_EQ(name.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Required tensor name shall not be empty."));
  auto n = make_shared<_Tensor_>();
  n->name = name;
  n->shape = utils::GetCompatibleShape(shape);
  n->domain = domain;
  n->reduce_axis = reduce_axis;
  n->set_type(dtype);
  n->operation = fn;
  n->InitAxis();

  return Tensor(n);
}
Tensor _Tensor_::Make(const std::string &name,
                      Type dtype,
                      const std::vector<Expr> &shape,
                      const std::vector<Expr> &domain,
                      const std::vector<Var> &reduce_axis) {
  PADDLE_ENFORCE_EQ(name.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Required tensor name shall not be empty."));
  auto n = make_shared<_Tensor_>();
  n->name = name;
  n->shape = utils::GetCompatibleShape(shape);
  n->domain = domain;
  n->reduce_axis = reduce_axis;
  n->operation = PlaceholderOp::Make(n->name, n->shape, Float(32));
  n->set_type(dtype);
  n->InitAxis();

  return Tensor(n);
}

Tensor _Tensor_::Make(const std::string &name,
                      Type dtype,
                      const std::vector<Dim> &sym_shape,
                      const std::vector<Dim> &sym_domain,
                      FunctionRef fn,
                      const std::vector<Var> &reduce_axis) {
  PADDLE_ENFORCE_EQ(name.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Required tensor name shall not be empty."));
  PADDLE_ENFORCE_EQ(sym_shape.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Required tensor sym_shape shall not be empty."));
  auto n = make_shared<_Tensor_>();
  n->name = name;
  n->sym_shape = sym_shape;
  for (int i = 0; i < sym_shape.size(); i++) {
    n->shape.emplace_back(sym_shape[i]->dim_expr);
  }
  n->sym_domain = sym_domain;
  for (int i = 0; i < sym_domain.size(); i++) {
    n->domain.emplace_back(sym_domain[i]->dim_expr);
  }
  n->reduce_axis = reduce_axis;
  n->set_type(dtype);
  n->operation = fn;
  n->InitAxis();

  return Tensor(n);
}
Tensor _Tensor_::Make(const std::string &name,
                      Type dtype,
                      const std::vector<Dim> &sym_shape,
                      const std::vector<Dim> &sym_domain,
                      const std::vector<Var> &reduce_axis) {
  PADDLE_ENFORCE_EQ(name.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Required tensor name shall not be empty."));
  PADDLE_ENFORCE_EQ(sym_shape.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Required tensor sym_shape shall not be empty."));
  auto n = make_shared<_Tensor_>();
  n->name = name;
  n->sym_shape = sym_shape;
  for (int i = 0; i < sym_shape.size(); i++) {
    n->shape.emplace_back(sym_shape[i]->dim_expr);
  }
  n->sym_domain = sym_domain;
  for (int i = 0; i < sym_domain.size(); i++) {
    n->domain.emplace_back(sym_domain[i]->dim_expr);
  }
  n->reduce_axis = reduce_axis;
  n->operation = PlaceholderOp::Make(n->name, n->shape, Float(32));
  n->set_type(dtype);
  n->InitAxis();

  return Tensor(n);
}

size_t Tensor::ndims() const { return operator->()->shape.size(); }

std::set<std::string> _Tensor_::GetDependTensorNames() const {
  std::set<std::string> names;

  auto add_depend_tensors_from_expr = [&](Expr expr) {
    auto tensors = ir::ir_utils::CollectIRNodes(expr, [&](const Expr *x) {
      return x->as_tensor() && x->as_tensor()->name != this->name;
    });
    for (auto &e : tensors) {
      names.insert(e.as_tensor()->name);
    }
  };

  if (is_compute_node()) {
    add_depend_tensors_from_expr(body());
  } else if (is_call_node()) {
    add_depend_tensors_from_expr(body());
  } else if (is_extern_call_node()) {
    add_depend_tensors_from_expr(body());
  } else if (is_placeholder_node()) {
    return names;
  } else {
    CINN_NOT_IMPLEMENTED
  }

  return names;
}

Expr Tensor::operator()(const std::vector<Expr> &indices) const {
  PADDLE_ENFORCE_EQ(self()->is_tuple(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Required tensor shall not be tuple type."));
  auto *node = operator->();
  const auto compatible_indices =
      utils::GetCompatibleStoreLoadIndices(*this, indices);

  PADDLE_ENFORCE_EQ(compatible_indices.size(),
                    ndims(),
                    ::common::errors::PreconditionNotMet(
                        "number of indices not match the dimension"));
  return Load::Make(*this, compatible_indices);
}

Expr _Tensor_::inline_expanded(const std::vector<Expr> &indices) {
  PADDLE_ENFORCE_EQ(is_compute_node(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Required tensor shall be compute node."));
  return get_compute_op()->producer_fn(indices);
}

const char *_Tensor_::operation_type() const {
  if (!operation.defined()) return "";
  return operation->as<ir::_Operation_>()->func_type();
}

bool _Tensor_::is_compute_node() const {
  return std::strcmp(operation_type(), ir::ComputeOp::__func_type__) == 0;
}
bool _Tensor_::is_placeholder_node() const {
  return std::strcmp(operation_type(), ir::PlaceholderOp::__func_type__) == 0;
}
bool _Tensor_::is_call_node() const {
  return std::strcmp(operation_type(), ir::CallOp::__func_type__) == 0;
}
bool _Tensor_::is_extern_call_node() const {
  if (std::strcmp(operation_type(), ir::CallOp::__func_type__) == 0) {
    auto *op = operation->as<ir::CallOp>();
    auto *call = op->call_expr.As<ir::Call>();
    if (call) {
      return call->is_extern_call();
    }
  }
  return false;
}
bool _Tensor_::is_buffer_shared_node() const {
  return std::strcmp(operation_type(), ir::BufferShareOp::__func_type__) == 0;
}

bool _Tensor_::is_preceding_view_node() const {
  return std::strcmp(operation_type(), ir::PrecedingViewOp::__func_type__) == 0;
}

ComputeOp *_Tensor_::get_compute_op() const {
  if (!is_compute_node()) return nullptr;
  return operation->as<ComputeOp>();
}

PlaceholderOp *_Tensor_::get_placeholder_op() const {
  if (!is_placeholder_node()) return nullptr;
  return operation->as<PlaceholderOp>();
}

void _Tensor_::InitAxis() const {
  axis_ = cinn::common::GenDefaultAxis(domain_without_reduce_axis().size());
}

bool _Tensor_::has_expression() const {
  return (!is_placeholder_node()) && (!is_tuple_get()) &&
         (!is_buffer_shared_node());
}

std::vector<Expr *> _Tensor_::expr_fields() {
  std::vector<Expr *> res;
  const char *func_type = operation->as<ir::_Operation_>()->func_type();
  if (operation.defined()) {
    if (is_compute_node()) {
      auto *op = operation->as<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
    } else if (is_placeholder_node()) {
      auto *op = operation->as<ir::PlaceholderOp>();
    } else if (is_call_node()) {
      auto *op = operation->as<ir::CallOp>();
      for (auto &expr : op->read_args()) res.push_back(&expr);
    } else if (is_buffer_shared_node()) {
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }

  for (auto &e : shape) {
    res.push_back(&e);
  }
  for (auto &e : domain) {
    res.push_back(&e);
  }
  return res;
}

std::vector<const Expr *> _Tensor_::expr_fields() const {
  std::vector<const Expr *> res;
  const char *func_type = operation->as<ir::_Operation_>()->func_type();
  if (operation.defined()) {
    if (is_compute_node()) {
      auto *op = operation->as<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
    } else if (is_placeholder_node()) {
      auto *op = operation->as<ir::PlaceholderOp>();
    } else if (is_call_node()) {
      auto *op = operation->as<ir::CallOp>();
      for (auto &expr : op->read_args()) res.push_back(&expr);
    } else if (is_buffer_shared_node()) {
    } else {
      LOG(ERROR) << "func_type: " << func_type;
      CINN_NOT_IMPLEMENTED
    }
  }

  for (auto &e : shape) {
    res.push_back(&e);
  }
  for (auto &e : domain) {
    res.push_back(&e);
  }

  return res;
}

_Tensor_::~_Tensor_() {}

Expr _Tensor_::body() const {
  if (is_placeholder_node()) return Expr();
  if (is_buffer_shared_node()) return Expr();
  if (is_compute_node()) return operation->as<ir::ComputeOp>()->body.front();
  if (is_call_node()) return operation->as<ir::CallOp>()->call_expr;
  CINN_NOT_IMPLEMENTED;
}

Expr *_Tensor_::mutable_body() {
  if (is_placeholder_node()) return nullptr;
  if (is_buffer_shared_node()) return nullptr;
  if (is_compute_node()) return &operation->as<ir::ComputeOp>()->body.front();
  if (is_call_node()) return &operation->as<ir::CallOp>()->call_expr;
  CINN_NOT_IMPLEMENTED
}

Expr _Tensor_::tensor_store_expanded_body() {
  PADDLE_ENFORCE_EQ(is_placeholder_node(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Placeholder should not expand store."));

  Expr final_body = body();
  if (shape.empty()) return final_body;

  std::vector<Expr> g_axis = cinn::common::GenDefaultAxisAsExpr(shape.size());
  if (!new_indices.empty()) {
    g_axis = new_indices;
  }

  auto *reduce_node = body().As<ir::Reduce>();
  if (reduce_node) {
    final_body = reduce_node->body;
    switch (reduce_node->reduce_type) {
      case ir::Reduce::kSum:
        final_body = Tensor(this)(g_axis) + final_body;
        break;
      case ir::Reduce::kMul:
        final_body = Tensor(this)(g_axis) * final_body;
        break;
      case ir::Reduce::kMax:
        final_body = Max::Make(Tensor(this)(g_axis), final_body);
        break;
      case ir::Reduce::kMin:
        final_body = Min::Make(Tensor(this)(g_axis), final_body);
        break;
      case ir::Reduce::kAll:
        final_body = Tensor(this)(g_axis) && final_body;
        break;
      case ir::Reduce::kAny:
        final_body = Tensor(this)(g_axis) || final_body;
        break;
      default:
        CINN_NOT_IMPLEMENTED
    }
  }

  if (is_tuple()) return final_body;

  return ir::Store::Make(Expr(Buffer(this)), final_body, g_axis);
}

void _Tensor_::Bind(lang::Buffer &buffer) {
  PADDLE_ENFORCE_EQ(buffer->type().is_void(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Required buffer type shall not be void()."));
  if (this->buffer.defined()) {
    // remove the old buffer
    if (this->buffer == buffer.buffer()) return;
    this->buffer->Unbind(this);
  }
  // Extract the tensors those has binded to this buffer.
  buffer_depended_tensor_names_ = buffer.buffer()->binded_tensor_names();

  buffer.buffer()->BindTo(this);
  PADDLE_ENFORCE_EQ(buffer->binded_tensor_names().empty(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Required binded_tensor_names shall not be empty."));
  this->buffer = buffer.buffer();
  PADDLE_ENFORCE_EQ(this->buffer.defined(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Required buffer shall be defined."));
}

void _Tensor_::Bind(const Buffer &buffer) {
  lang::Buffer buf(buffer);
  Bind(buf);
}

void _Tensor_::WithBuffer(const Type &type) {
  Type buf_type = type.is_void() ? type_ : type;
  lang::Buffer buf(buf_type);
  buf->target = cinn::common::DefaultHostTarget();
  Bind(buf);
}

void _Tensor_::WithBuffer(const std::string &memory_type,
                          const std::string &buffer_name,
                          const Type &type) {
  Type buf_type = type.is_void() ? type_ : type;
  if (this->buffer.defined()) {
    this->buffer->dtype = buf_type;
    this->buffer->name = buffer_name;
    if (memory_type == "shared") {
      this->buffer->memory_type = MemoryType::GPUShared;
    } else if (memory_type == "local") {
      this->buffer->memory_type = MemoryType::GPULocal;
    } else if (memory_type == "global") {
      this->buffer->memory_type = MemoryType::Heap;
    } else {
      std::stringstream ss;
      ss << "Not supported memory type " << memory_type;
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
  } else {
    lang::Buffer buf(buf_type, buffer_name);
    buf->target = cinn::common::DefaultHostTarget();
    Bind(buf);

    if (memory_type == "shared") {
      buf->memory_type = MemoryType::GPUShared;
    } else if (memory_type == "local") {
      buf->memory_type = MemoryType::GPULocal;
    } else if (memory_type == "global") {
      buf->memory_type = MemoryType::Heap;
    } else {
      std::stringstream ss;
      ss << "Not supported memory type " << memory_type;
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
  }
}

bool _Tensor_::HasSameShapeWith(const Tensor &other) const {
  if (shape.size() != other->shape.size()) return false;

  for (int i = 0; i < shape.size(); i++) {
    Expr dim0 = optim::ArithSimplify(shape[i]);
    Expr dim1 = optim::ArithSimplify(other->shape[i]);

    if (dim0 != dim1) return false;
  }
  return true;
}

Tensor _Tensor_::TupleGet(int offset) const {
  PADDLE_ENFORCE_EQ(is_tuple(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Required Tensor shall be tuple type."));
  auto *call = body().As<ir::Call>();
  PADDLE_ENFORCE_LT(
      offset,
      call->write_args.size(),
      ::common::errors::PreconditionNotMet(
          "Required offset shall be less than call->write_args.size()."));
  auto tensor = call->write_args[offset].as_tensor_ref();
  tensor->WithBuffer();
  return tensor;
}

bool _Tensor_::is_tuple() const {
  if (!has_expression()) return false;
  auto *call = body().As<ir::Call>();
  if (call && call->is_extern_call() && !call->write_args.empty()) return true;
  return false;
}

std::vector<Expr> _Tensor_::domain_with_reduce_axis() const {
  if (reduce_axis.empty()) return domain;
  auto res = domain;
  for (const Var &axis : reduce_axis) {
    PADDLE_ENFORCE_EQ(axis->upper_bound.type().is_int(32) ||
                          axis->upper_bound.type().is_int(64),
                      true,
                      ::common::errors::PreconditionNotMet(
                          "Required upper_bound shall be int32 or int64."));
    res.push_back(axis->upper_bound);
  }
  return res;
}

bool operator<(const Tensor &a, const Tensor &b) { return a->name < b->name; }

Tensor::Tensor(const std::string &name,
               Type dtype,
               const std::vector<Expr> &shape,
               const std::vector<Expr> &domain,
               FunctionRef fn,
               const std::vector<Var> &reduce_axis)
    : IrNodeRef(
          _Tensor_::Make(name, dtype, shape, domain, fn, reduce_axis).self()) {}

Tensor::Tensor(const std::string &name,
               Type dtype,
               const std::vector<Dim> &sym_shape,
               const std::vector<Dim> &sym_domain,
               FunctionRef fn,
               const std::vector<Var> &reduce_axis)
    : IrNodeRef(
          _Tensor_::Make(name, dtype, sym_shape, sym_domain, fn, reduce_axis)
              .self()) {}

bool _Tensor_::is_tuple_get() const {
  return is_call_node() && operation.defined() &&
         operation->as<ir::_Operation_>()->func_type() ==
             ir::CallOp::__func_type__ &&
         operation->as<ir::CallOp>()->is_tuple_get;
}

bool _Tensor_::IsDependOnStatement(absl::string_view statement) {
  if (!is_compute_node()) {
    return false;
  }

  auto depend_tensors = DependingTensorNames();
  for (const auto &x : depend_tensors) {
    if (x == statement) return true;
  }
  return false;
}

std::set<std::string> _Tensor_::DependingTensorNames() {
  std::set<std::string> res;
  if (body().defined()) {
    auto depend_tensors = ir::ir_utils::CollectIRNodes(
        body(), [](const Expr *x) -> bool { return x->as_tensor(); });
    for (const auto &x : depend_tensors) {
      if (x.get() != this) {
        res.insert(x.as_tensor()->name);
      }
    }
  }
  return res;
}

const std::vector<Var> &_Tensor_::axis() const {
  PADDLE_ENFORCE_EQ(axis_.size(),
                    domain_without_reduce_axis().size(),
                    ::common::errors::PreconditionNotMet(
                        "Required axis_ shall have same size with "
                        "domain_without_reduce_axis."));
  return axis_;
}

std::vector<Var> _Tensor_::axis_with_reduce() const {
  auto axis = axis_;
  axis.insert(axis.end(), reduce_axis.begin(), reduce_axis.end());
  return axis;
}

bool _Tensor_::Uses(const Tensor &other) const {
  auto loads = ir::ir_utils::CollectIRNodes(body(), [&](const Expr *x) {
    auto *loadn = x->As<ir::Load>();
    if (!loadn) return false;
    return loadn->tensor.as_tensor()->name == other->name;
  });
  return !loads.empty();
}

ir::Tensor _Tensor_::Reshape(const std::vector<Expr> &shape) const {
  auto op = BufferShareOp::Make();
  auto n = make_shared<_Tensor_>();
  auto selft = Tensor(const_cast<ir::_Tensor_ *>(this));

  {
    int32_t this_num_elements = 1;
    for (auto &e : this->shape) {
      this_num_elements = this_num_elements * e.as_int32();
    }

    int32_t num_elements = 1;
    for (auto &e : shape) {
      num_elements = num_elements * e.as_int32();
    }

    PADDLE_ENFORCE_EQ(
        this_num_elements,
        num_elements,
        ::common::errors::PreconditionNotMet(
            "Required this_num_elements shall be equal to num_elements."));
  }

  n->name = Context::Global().NewName(name + "_reshape");
  n->shape = shape;
  n->domain = shape;
  n->set_type(type());
  n->operation = op;
  n->InitAxis();

  auto t = Tensor(n);
  return t;
}

ir::Tensor _Tensor_::ReshapeCopied(const std::vector<Expr> &shape) const {
  auto t = ir::Tensor(const_cast<ir::_Tensor_ *>(this));
  auto copied = Compute(
      domain,
      [=](const std::vector<Expr> &axis) { return t(axis); },
      Context::Global().NewName(this->name + "_copied"));
  auto res = copied->Reshape(shape);
  return res;
}

Shared<poly::Stage> CreateStage(Tensor tensor) {
  isl::set isl_domain;
  // We will remove isl, and the subsequent compilation process will no longer
  // use it. But it has not been completely removed in the process. it cannot be
  // supported here under dynamic shape. Therefore, we temporarily use fake
  // domain.
  poly::Domain fake_domain(Context::isl_ctx(), "fake_domain", {});
  isl_domain = fake_domain.to_isl();

  return poly::Stage::New(isl_domain, tensor->body(), tensor.self());
}

static constexpr char kReduceInitSuffix[] = "__reduce_init";

std::string GenReduceInitTensorNameOf(const std::string &tensor_name) {
  return tensor_name + kReduceInitSuffix;
}

bool IsReduceInitTensorName(const std::string &tensor_name) {
  std::string reduce_init_suffix(kReduceInitSuffix);
  return tensor_name.length() > reduce_init_suffix.size() &&
         tensor_name.substr(tensor_name.length() - reduce_init_suffix.size(),
                            reduce_init_suffix.size()) == reduce_init_suffix;
}

bool IsSplitTransformTensorName(const std::string &tensor_name) {
  return tensor_name.find("_split_transform") != std::string::npos;
}

std::string GetOriginalReduceTensorName(const std::string &tensor_name) {
  std::string reduce_init_suffix(kReduceInitSuffix);
  if (IsReduceInitTensorName(tensor_name)) {
    return tensor_name.substr(0,
                              tensor_name.length() - reduce_init_suffix.size());
  }
  return tensor_name;
}

bool _Tensor_::is_reduce_sum() const {
  if (!contains_reduce_axis()) return false;
  return body().As<ir::Reduce>() &&
         body().As<ir::Reduce>()->reduce_type == ir::Reduce::ReduceType::kSum;
}
bool _Tensor_::is_reduce_mul() const {
  if (!contains_reduce_axis()) return false;
  return body().As<ir::Reduce>() &&
         body().As<ir::Reduce>()->reduce_type == ir::Reduce::ReduceType::kMul;
}

Expr _Tensor_::GetReduceInitVal() const {
  PADDLE_ENFORCE_EQ(is_reduce_tensor(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Required tensor is a reduce type."));
  return body().As<ir::Reduce>()->init;
}

void _Tensor_::Verify() const {
  PADDLE_ENFORCE_EQ(shape.empty(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Required shape shall not be empty."));
  PADDLE_ENFORCE_EQ(domain.empty(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Required domain shall not be empty."));
  PADDLE_ENFORCE_EQ(name.empty(),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "Required name shall not be empty."));
}

}  // namespace ir
}  // namespace cinn
