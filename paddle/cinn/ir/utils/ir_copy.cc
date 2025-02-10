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

#include "paddle/cinn/ir/utils/ir_copy.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace ir {
namespace ir_utils {
namespace {
struct IRCopyVisitor : public ir::IRVisitorRequireReImpl<Expr> {
 public:
  explicit IRCopyVisitor(bool copy_buffer_node)
      : copy_buffer_node(copy_buffer_node) {}

  // Use maps to unify all the copied tensors and buffers.
  std::map<std::string, ir::_Tensor_*> tensor_map;
  std::map<std::string, ir::_Buffer_*> buffer_map;
  // whether to deep copy Buffer node.
  bool copy_buffer_node;

  Expr Visit(const Expr* op) override {
    bool is_index = op->is_index();
    auto copy = IRVisitorRequireReImpl::Visit(op);
    return copy.set_index(is_index);
  }

  Expr Visit(const IndexExpr* op) override {
    Expr e = *op;
    auto copy = Visit(&e);
    return copy.set_index(true);
  }

  Module Visit(const ir::_Module_* op) {
    std::vector<Expr> buffers;
    std::vector<LoweredFunc> functions;
    std::vector<Module> submodules;
    std::vector<Expr> predicates;
    std::vector<int> priorities;
    LoweredFunc infer_shape_func;
    for (auto& expr : op->buffers) {
      buffers.push_back(Visit(&expr));
    }

    for (auto& expr : op->functions) {
      functions.push_back(Visit(expr.As<_LoweredFunc_>()));
    }

    for (auto& expr : op->submodules) {
      submodules.push_back(Visit(expr.As<ir::_Module_>()));
    }

    for (auto& expr : op->predicates) {
      predicates.push_back(Visit(&expr));
    }

    for (int priority : op->priorities) {
      priorities.push_back(priority);
    }

    if (op->infer_shape_func.defined()) {
      infer_shape_func = Visit(op->infer_shape_func.As<_LoweredFunc_>());
    }

    auto res = ir::_Module_::Make(op->name, op->target);
    res->buffers = buffers;
    res->functions = functions;
    res->submodules = submodules;
    res->predicates = predicates;
    res->priorities = priorities;
    res->infer_shape_func = infer_shape_func;

    return res;
  }

  LoweredFunc Visit(const _LoweredFunc_* op) {
    auto func = make_shared<_LoweredFunc_>();

    func->name = op->name;
    func->args = op->args;
    func->body = Visit(&op->body);
    func->temp_bufs = op->temp_bufs;
    func->temp_spaces = op->temp_spaces;
    func->num_output_tensors = op->num_output_tensors;

    func->device_api = op->device_api;

    func->cuda_axis_info = op->cuda_axis_info;

    std::vector<Expr> alloc_output_buffer_exprs;
    std::vector<Expr> dealloc_output_buffer_exprs;
    std::vector<Expr> buffer_data_cast_exprs;
    std::vector<Expr> argument_prepare_exprs;

#define COPY_ADD_FIELD(field__)      \
  for (auto& expr : op->field__) {   \
    field__.push_back(Visit(&expr)); \
  }                                  \
  func->field__ = std::move(field__);

    COPY_ADD_FIELD(alloc_output_buffer_exprs);
    COPY_ADD_FIELD(dealloc_output_buffer_exprs);
    COPY_ADD_FIELD(buffer_data_cast_exprs);
    COPY_ADD_FIELD(argument_prepare_exprs);

    return LoweredFunc(func);
  }

 protected:
  // The methods of ir nodes follows the order defined in node.h
  Expr Visit(const ir::IntImm* op) override {
    return Expr(make_shared<IntImm>(op->type(), op->value));
  }
  Expr Visit(const ir::UIntImm* op) override {
    return Expr(make_shared<UIntImm>(op->type(), op->value));
  }
  Expr Visit(const ir::FloatImm* op) override {
    return Expr(make_shared<FloatImm>(op->type(), op->value));
  }
  Expr Visit(const ir::StringImm* op) override {
    return Expr(cinn::common::make_shared<StringImm>(op->value));
  }

  Expr Visit(const ir::Cast* op) override {
    auto v = Visit(&op->v());
    return Cast::Make(op->type(), v);
  }

  Expr Visit(const Select* op) override {
    auto condition = Visit(&op->condition);
    auto true_value = Visit(&op->true_value);
    auto false_value = Visit(&op->false_value);
    return Select::Make(condition, true_value, false_value);
  }

  Expr Visit(const IfThenElse* op) override {
    auto condition = Visit(&op->condition);
    auto true_case = Visit(&op->true_case);
    Expr false_case;
    if (op->false_case.defined()) false_case = Visit(&op->false_case);
    return IfThenElse::Make(condition, true_case, false_case);
  }

  Expr Visit(const Block* op) override {
    std::vector<Expr> stmts;
    for (auto& s : op->stmts) {
      stmts.push_back(Visit(&s));
    }
    return Block::Make(stmts);
  }

  Expr Visit(const Call* op) override {
    auto read_args = Visit(op->read_args);
    auto write_args = Visit(op->write_args);
    return Call::Make(op->type(),
                      op->name,
                      read_args,
                      write_args,
                      op->call_type,
                      FunctionRef(),
                      0,
                      op->attrs);
  }

  Expr Visit(const _Var_* op) override {
    auto* n = make_shared<_Var_>();

    n->name = op->name;
    n->is_reduce_axis = op->is_reduce_axis;
    n->is_symbolic_constant = op->is_symbolic_constant;
    n->set_type(op->type());

    if (op->lower_bound.defined()) {
      n->lower_bound = Visit(&op->lower_bound);
    }
    if (op->upper_bound.defined()) {
      n->upper_bound = Visit(&op->upper_bound);
    }

    return Expr(n);
  }

  Expr Visit(const Load* op) override {
    auto tensor = Visit(&op->tensor);
    std::vector<Expr> indices;
    for (auto& idx : op->indices) {
      indices.push_back(Visit(&idx));
    }
    return Load::Make(tensor, indices);
  }

  Expr Visit(const Store* op) override {
    auto tensor = Visit(&op->tensor);
    auto value = Visit(&op->value);
    std::vector<Expr> indices;
    for (auto& idx : op->indices) indices.push_back(Visit(&idx));

    return Store::Make(tensor, value, indices);
  }

  Expr Visit(const Alloc* op) override {
    auto extents = Visit(op->extents);
    Expr condition;
    Expr body;
    if (op->condition.defined()) condition = Visit(&op->condition);
    if (op->body.defined()) body = Visit(&op->body);

    return Alloc::Make(op->destination, op->type(), extents, condition, body);
  }

  Expr Visit(const Free* op) override { return Free::Make(op->destination); }

  Expr Visit(const _Buffer_* op) override {
    if (buffer_map.count(op->name)) {
      return buffer_map[op->name];
    }

    auto shape = Visit(op->shape);
    auto strides = Visit(op->strides);
    auto name = op->name;
    auto scope = op->scope;
    int data_alignment = op->data_alignment;
    auto elem_offset = Visit(&op->elem_offset);
    int offset_factor = op->offset_factor;
    Target target = op->target;

    auto new_node = _Buffer_::Make(name, shape);
    new_node->strides = strides;
    new_node->dtype = op->dtype;  // copy data element's type.
    new_node->name = name;
    new_node->scope = scope;
    new_node->data_alignment = data_alignment;
    new_node->elem_offset = elem_offset;
    new_node->offset_factor = offset_factor;
    new_node->target = target;
    new_node->memory_type = op->memory_type;
    new_node->set_type(op->type());
    op->CopyMeta(new_node.As<ir::_Buffer_>());

    buffer_map[op->name] = new_node->self();

    return Expr(ir::Buffer(new_node));
  }

  Expr Visit(const _Tensor_* op) override {
    if (tensor_map.count(op->name)) {
      return tensor_map[op->name];
    }

    auto shape = Visit(op->shape);
    auto domain = Visit(op->domain);
    auto buffer_expr = Expr(op->buffer);
    // TODO(Superjomn) copy the operation.
    auto operation = op->operation;
    auto name = op->name;
    auto tensor = make_shared<_Tensor_>();

    // tensor->buffer = op->buffer;
    if (buffer_expr.defined()) {
      if (copy_buffer_node) {
        auto buffer = Visit(&buffer_expr);
        tensor->buffer = buffer.as_buffer_ref();
      } else {
        tensor->buffer = op->buffer;
      }
    }
    tensor->domain = domain;
    tensor->shape = shape;
    tensor->reduce_axis = op->reduce_axis;
    tensor->operation = operation;
    tensor->name = name;
    tensor->set_type(op->type());
    tensor->axis_ = op->axis_;

    tensor_map[tensor->name] = tensor;

    return tensor;
  }

  Expr Visit(const For* op) override {
    auto extent = Visit(&op->extent);
    auto min = Visit(&op->min);
    auto body = Visit(&op->body);

    return ir::For::Make(op->loop_var,
                         min,
                         extent,
                         op->for_type(),
                         op->device_api,
                         body,
                         op->vectorize_info(),
                         op->bind_info());
  }

  Expr Visit(const ir::PolyFor* op) override {
    auto init = Visit(&op->init);
    auto condition = Visit(&op->condition);
    auto inc = Visit(&op->inc);
    auto body = Visit(&op->body);
    auto expr = PolyFor::Make(op->iterator,
                              init,
                              condition,
                              inc,
                              op->for_type(),
                              op->device_api,
                              body,
                              op->vectorize_info(),
                              op->bind_info());
    return expr;
  }

  Expr Visit(const Let* op) override {
    auto value = Visit(&op->symbol);
    auto body = Visit(&op->body);

    return Let::Make(value, body);
  }

  Expr Visit(const Reduce* op) override {
    auto init = Visit(&op->init);
    auto body = Visit(&op->body);
    std::vector<Var> reduce_axis(op->reduce_axis.begin(),
                                 op->reduce_axis.end());
    return Reduce::Make(op->reduce_type, init, body, reduce_axis);
  }

  Expr Visit(const Ramp* op) override {
    auto base = Visit(&op->base);
    auto stride = Visit(&op->stride);
    int lanes = op->lanes;
    return Ramp::Make(base, stride, lanes);
  }

  Expr Visit(const Broadcast* op) override {
    auto value = Visit(&op->value);
    int lanes = op->lanes;
    PADDLE_ENFORCE_EQ(value.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Broadcasting value is not defined."));
    PADDLE_ENFORCE_EQ(
        value.type().valid(),
        true,
        ::common::errors::InvalidArgument("Broadcasting value type is invalid. "
                                          "Expected a valid type, but got: %s",
                                          value.type()));

    auto* n = make_shared<Broadcast>();
    n->value = value;
    n->lanes = lanes;
    return Expr(n);
  }

  Expr Visit(const FracOp* op) override {
    auto a = Visit(&op->a());
    auto b = Visit(&op->b());
    PADDLE_ENFORCE_EQ(a.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The first operand of FracOp is not defined."));
    PADDLE_ENFORCE_EQ(b.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The second operand of FracOp is not defined."));

    auto* n = make_shared<FracOp>();
    n->a() = a;
    n->b() = b;
    return Expr(n);
  }

  Expr Visit(const Product* op) override {
    std::vector<Expr> operands;
    for (auto& v : op->operands()) {
      operands.push_back(Visit(&v));
    }
    return Product::Make(operands);
  }

  Expr Visit(const Sum* op) override {
    std::vector<Expr> operands;
    for (auto& v : op->operands()) {
      operands.push_back(Visit(&v));
    }
    return Sum::Make(operands);
  }

  Expr Visit(const ir::PrimitiveNode* op) override {
    std::vector<std::vector<Expr>> arguments;
    for (auto& args : op->arguments) {
      arguments.push_back(Visit(args));
    }

    auto n = cinn::common::make_shared<ir::PrimitiveNode>();
    n->name = op->name;
    n->attrs = op->attrs;  // attrs are PODs
    n->arguments = arguments;
    return Expr(n);
  }

  Expr Visit(const ir::_BufferRange_* op) {
    std::vector<Var> ranges;
    for (auto& range_var : op->ranges) {
      auto* var = range_var.As<_Var_>();
      ranges.push_back(Visit(var));
    }
    return ir::_BufferRange_::Make(Visit(&op->buffer), ranges);
  }

  Expr Visit(const ir::ScheduleBlock* op) {
    std::vector<Var> iter_vars;
    for (auto iter_var : op->iter_vars) {
      auto* var = iter_var.As<_Var_>();
      PADDLE_ENFORCE_NE(
          var,
          nullptr,
          ::common::errors::InvalidArgument(
              "ScheduleBlock iter_var is not a valid _Var_ type."));
      iter_vars.push_back(Visit(var));
    }
    std::vector<Expr> read_buffers;
    for (auto buffer_range : op->read_buffers) {
      read_buffers.push_back(Visit(&buffer_range));
    }
    std::vector<Expr> write_buffers;
    for (auto buffer_range : op->write_buffers) {
      write_buffers.push_back(Visit(&buffer_range));
    }
    Expr res = ir::ScheduleBlock::Make(
        iter_vars, read_buffers, write_buffers, op->name, Visit(&op->body));
    res.As<ScheduleBlock>()->attrs = op->attrs;
    res.As<ScheduleBlock>()->reduce_method = op->reduce_method;
    return res;
  }

  Expr Visit(const ir::ScheduleBlockRealize* op) {
    std::vector<Expr> iter_values;
    for (auto iter_value : op->iter_values) {
      iter_values.push_back(Visit(&iter_value));
    }
    return ir::ScheduleBlockRealize::Make(iter_values,
                                          Visit(&op->schedule_block));
  }

  Expr Visit(const ir::_Dim_* op) override {
    return ir::_Dim_::Make(op->name, op->sym_dim);
  }
  Expr Visit(const ir::IterMark* op) override {
    Expr source = Visit(&(op->source));
    Expr extent = Visit(&(op->extent));

    return IterMark::Make(source, extent);
  }
  Expr Visit(const ir::IterSplit* op) override {
    Expr source = Visit(&(op->source));
    Expr lower_factor = Visit(&(op->lower_factor));
    Expr extent = Visit(&(op->extent));
    Expr scale = Visit(&(op->scale));

    return IterSplit::Make(source, lower_factor, extent, scale);
  }
  Expr Visit(const ir::IterSum* op) override {
    std::vector<Expr> args;
    for (const auto& v : op->args) {
      args.push_back(Visit(&v));
    }
    Expr base = Visit(&(op->base));
    return IterSum::Make(args, base);
  }

#define __(x__) Expr Visit(const ir::intrinsics::x__* op);
  INTRINSIC_KIND_FOR_EACH(__)
#undef __

  Expr Visit(const ir::IntrinsicOp* op) override {
    switch (op->getKind()) {
#define __(x__)                   \
  case ir::IntrinsicKind::k##x__: \
    return Visit(llvm::dyn_cast<ir::intrinsics::x__>(op));
      INTRINSIC_KIND_FOR_EACH(__)
#undef __
    }
  }

#define OP_BINARY_HANDLE(op__)                        \
  Expr Visit(const ir::op__* op) override {           \
    auto a = IRVisitorRequireReImpl::Visit(&op->a()); \
    auto b = IRVisitorRequireReImpl::Visit(&op->b()); \
    return op__::Make(a, b);                          \
  }
  NODETY_BINARY_OP_FOR_EACH(OP_BINARY_HANDLE)
#undef OP_BINARY_HANDLE

#define OP_UNARY_HANDLE(op__)                         \
  Expr Visit(const op__* op) override {               \
    auto v = IRVisitorRequireReImpl::Visit(&op->v()); \
    return op__::Make(v);                             \
  }
  NODETY_UNARY_OP_FOR_EACH(OP_UNARY_HANDLE)
#undef OP_UNARY_HANDLE

  std::vector<Expr> Visit(const std::vector<Expr>& vs) {
    std::vector<Expr> copied;
    for (auto& e : vs) {
      copied.push_back(Visit(&e));
    }
    return copied;
  }
};

Expr IRCopyVisitor::Visit(const ir::intrinsics::BufferGetDataHandle* op) {
  return intrinsics::BufferGetDataHandle::Make(Visit(&op->buffer));
}
Expr IRCopyVisitor::Visit(const ir::intrinsics::BufferGetDataConstHandle* op) {
  return intrinsics::BufferGetDataConstHandle::Make(Visit(&op->buffer));
}
Expr IRCopyVisitor::Visit(const ir::intrinsics::PodValueToX* op) {
  return intrinsics::PodValueToX::Make(Visit(&op->pod_value_ptr),
                                       op->GetOutputType(0));
}
Expr IRCopyVisitor::Visit(const ir::intrinsics::BufferCreate* op) {
  return intrinsics::BufferCreate::Make(Visit(&op->buffer));
}
Expr IRCopyVisitor::Visit(const ir::intrinsics::GetAddr* op) {
  return intrinsics::GetAddr::Make(Visit(&op->data));
}
Expr IRCopyVisitor::Visit(const ir::intrinsics::ArgsConstruct* op) {
  llvm::SmallVector<Expr, 7> args;
  for (auto& arg : op->args) {
    args.push_back(Visit(&arg));
  }
  return intrinsics::ArgsConstruct::Make(op->var, args);
}
Expr IRCopyVisitor::Visit(const ir::intrinsics::BuiltinIntrin* op) {
  return intrinsics::BuiltinIntrin::Make(
      op->name, op->args, op->id, op->arg_nums, op->type());
}
}  // namespace
Expr IRCopy(const Expr& x, bool copy_buffer_node) {
  IRCopyVisitor visitor(copy_buffer_node);
  auto copied = visitor.Visit(&x);
  return copied;
}

std::vector<Expr> IRCopy(const std::vector<Expr>& x, bool copy_buffer_node) {
  std::vector<Expr> res;
  for (auto& i : x) {
    res.emplace_back(IRCopy(i, copy_buffer_node));
  }
  return res;
}

ir::ModuleExpr IRCopy(const ir::ModuleExpr& x, bool copy_buffer_node) {
  return ir::ModuleExpr(IRCopy(x.GetExprs(), copy_buffer_node));
}

ir::Module IRCopy(const Module& m, bool copy_buffer_node) {
  IRCopyVisitor visitor(copy_buffer_node);
  return visitor.Visit(m.As<ir::_Module_>());
}

ir::LoweredFunc IRCopy(const ir::LoweredFunc& x, bool copy_buffer_node) {
  IRCopyVisitor visitor(copy_buffer_node);
  auto copied = visitor.Visit(x.As<ir::_LoweredFunc_>());
  // TODO(Dmovic): Update ir copy when remove expr body.
  copied->body_block = x->body_block;
  return copied;
}

// TODO(zhhsplendid): make IRCopy of std::vector a template function
std::vector<ir::LoweredFunc> IRCopy(const std::vector<ir::LoweredFunc>& x,
                                    bool copy_buffer_node) {
  std::vector<ir::LoweredFunc> res;
  for (const auto& i : x) {
    res.emplace_back(IRCopy(i, copy_buffer_node));
  }
  return res;
}
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
