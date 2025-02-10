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

#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include <glog/logging.h>

#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt_visitors.h"

namespace cinn {
namespace ir {

namespace ir_utils {
namespace {
struct IrNodesCollector : public IRVisitorRequireReImpl<void> {
  using teller_t = std::function<bool(const Expr*)>;
  using handler_t = std::function<void(const Expr*)>;

  teller_t teller;
  handler_t handler;
  bool uniq_target_;
  bool find_target_{false};

  IrNodesCollector(teller_t&& teller, handler_t&& handler, bool uniq_target)
      : teller(teller), handler(handler), uniq_target_(uniq_target) {}

  void Visit(const Expr* expr) override {
    if (!expr->defined() || find_target_) return;
    if (visited_.count(expr->get())) return;

    if (teller(expr)) {
      handler(expr);
      if (uniq_target_) {
        find_target_ = true;
        return;
      }
    }
    visited_.insert(expr->get());

    switch (expr->node_type()) {
#define __(op__)                 \
  case ir::IrNodeTy::op__:       \
    Visit(expr->As<ir::op__>()); \
    break;

      NODETY_FORALL(__)

      default:
        PADDLE_THROW(::common::errors::InvalidArgument("not supported NodeTy"));
#undef __
    }
  }

#define __m(t__)                       \
  void Visit(const t__* x) override {  \
    for (auto* n : x->expr_fields()) { \
      if (n->defined()) {              \
        Visit(n);                      \
      }                                \
    }                                  \
  }

  NODETY_FORALL_EXCEPT_INTRINSIC(__m)
#undef __m

  void Visit(const ir::IntrinsicOp* op) {
    switch (op->getKind()) {
#define __(x)                                     \
  case ir::IntrinsicKind::k##x:                   \
    Visit(llvm::dyn_cast<ir::intrinsics::x>(op)); \
    break;

      INTRINSIC_KIND_FOR_EACH(__)
#undef __
    }
  }

  void Visit(const ir::intrinsics::GetAddr* x) {
    if (x->data.defined()) {
      Visit(&(x->data));
    }
  }

  void Visit(const ir::intrinsics::BufferGetDataHandle* x) {
    if (x->buffer.defined()) {
      Visit(&(x->buffer));
    }
  }

  void Visit(const ir::intrinsics::BufferGetDataConstHandle* x) {
    if (x->buffer.defined()) {
      Visit(&(x->buffer));
    }
  }

  void Visit(const ir::intrinsics::PodValueToX* x) {
    if (x->pod_value_ptr.defined()) {
      Visit(&(x->pod_value_ptr));
    }
  }

  void Visit(const ir::intrinsics::BufferCreate* x) {
    if (x->buffer.defined()) {
      Visit(&(x->buffer));
    }
  }

  void Visit(const ir::intrinsics::ArgsConstruct* x) {
    if (x->var.defined()) {
      Expr convert = Expr(x->var);
      Visit(&convert);
    }
    for (int i = 0; i < x->args.size(); ++i) {
      if (x->args[i].defined()) {
        Visit(&(x->args[i]));
      }
    }
  }

  void Visit(const ir::intrinsics::BuiltinIntrin* x) {
    for (int i = 0; i < x->args.size(); ++i) {
      if (x->args[i].defined()) {
        Visit(&(x->args[i]));
      }
    }
  }

  std::set<void*> visited_;
};

struct IrNodesWithoutTensorCollector : public IrNodesCollector {
  using teller_t = std::function<bool(const Expr*)>;
  using handler_t = std::function<void(const Expr*)>;
  IrNodesWithoutTensorCollector(teller_t teller,
                                handler_t handler,
                                bool uniq_target)
      : IrNodesCollector(std::move(teller), std::move(handler), uniq_target) {}

  void Visit(const _Tensor_* expr) override {
    for (auto& e : expr->shape) {
      IrNodesCollector::Visit(&e);
    }
  }
  void Visit(const Expr* expr) override { IrNodesCollector::Visit(expr); }
};

}  // namespace

std::set<Expr> CollectIRNodes(Expr expr,
                              std::function<bool(const Expr*)>&& teller,
                              bool uniq_target) {
  std::set<Expr> exprs;
  IrNodesCollector::handler_t handler = [&](const Expr* x) {
    exprs.insert(*x);
  };
  IrNodesCollector collector(
      std::move(teller), std::move(handler), uniq_target);
  collector.Visit(&expr);
  return exprs;
}

std::set<Expr> CollectIRNodes(ir::LoweredFunc f,
                              std::function<bool(const Expr*)>&& teller,
                              bool uniq_target) {
  return CollectIRNodes(f->body, std::move(teller), uniq_target);
}

std::set<Expr> CollectIRNodes(ir::stmt::BlockRef block,
                              std::function<bool(const Expr*)>&& teller,
                              bool uniq_target) {
  std::set<Expr> collect_nodes;
  const auto& CollectInSubExpr = [&](const Expr& e) {
    const auto& sub_expr_res =
        CollectIRNodes(e, std::move(teller), uniq_target);
    collect_nodes.insert(sub_expr_res.begin(), sub_expr_res.end());
  };
  const auto& CollectInSubExprArr = [&](const std::vector<Expr>& exprs) {
    for (const auto& expr : exprs) {
      CollectInSubExpr(expr);
    }
  };
  const auto& CollectInStmt = [&](const stmt::StmtRef& stmt) {
    if (stmt.isa<stmt::Let>()) {
      const auto& let_stmt = stmt.as<stmt::Let>();
      CollectInSubExpr(let_stmt->symbol());
      if (let_stmt->body().defined()) {
        CollectInSubExpr(let_stmt->body());
      }
    } else if (stmt.isa<stmt::Store>()) {
      const auto& store_stmt = stmt.as<stmt::Store>();
      CollectInSubExpr(store_stmt->tensor());
      CollectInSubExpr(store_stmt->value());
      CollectInSubExprArr(store_stmt->indices());
    } else if (stmt.isa<stmt::Alloc>()) {
      const auto& alloc_stmt = stmt.as<stmt::Alloc>();
      CollectInSubExprArr(alloc_stmt->extents());
      if (alloc_stmt->condition().defined()) {
        CollectInSubExpr(alloc_stmt->condition());
      }
      if (alloc_stmt->body().defined()) {
        CollectInSubExpr(alloc_stmt->body());
      }
    } else if (stmt.isa<stmt::Free>()) {
      const auto& free_stmt = stmt.as<stmt::Free>();
      CollectInSubExpr(free_stmt->destination());
    } else if (stmt.isa<stmt::IfThenElse>()) {
      const auto& if_stmt = stmt.as<stmt::IfThenElse>();
      CollectInSubExpr(if_stmt->condition());
    } else if (stmt.isa<stmt::For>()) {
      const auto& for_stmt = stmt.as<stmt::For>();
      CollectInSubExpr(for_stmt->min());
      CollectInSubExpr(for_stmt->extent());
    } else if (stmt.isa<stmt::Schedule>()) {
      // Body is a BlockRef which will be visited automatically, do nothing
      // here.
    } else if (stmt.isa<stmt::Evaluate>()) {
      const auto& evaluate_stmt = stmt.as<stmt::Evaluate>();
      CollectInSubExpr(evaluate_stmt->value());
    } else {
      PADDLE_THROW(::common::errors::Fatal(
          "Dead code. Stmt type %d is not supported to collect ir.",
          stmt->stmt_type()));
    }
  };
  stmt::Visit(block, CollectInStmt, [](const stmt::StmtRef& stmt) {});
  return collect_nodes;
}

std::vector<Expr> CollectIRNodesInOrder(
    Expr expr, std::function<bool(const Expr*)>&& teller) {
  std::vector<Expr> exprs;
  IrNodesWithoutTensorCollector::handler_t handler = [&](const Expr* x) {
    exprs.push_back(*x);
  };
  IrNodesWithoutTensorCollector collector(
      std::move(teller), std::move(handler), false);
  collector.Visit(&expr);
  return exprs;
}

std::vector<Expr> CollectIRNodesWithoutTensor(
    Expr expr, std::function<bool(const Expr*)>&& teller, bool uniq_target) {
  std::vector<Expr> exprs;
  IrNodesWithoutTensorCollector::handler_t handler = [&](const Expr* x) {
    exprs.push_back(*x);
  };
  IrNodesWithoutTensorCollector collector(
      std::move(teller), std::move(handler), uniq_target);
  collector.Visit(&expr);
  return exprs;
}

std::map<std::string, Expr> CollectTensorMap(
    Expr x, std::function<bool(const Expr*)>&& extra_teller) {
  std::map<std::string, Expr> tensor_map;

  auto tensors = CollectIRNodes(
      x, [&](const Expr* x) { return x->as_tensor() && extra_teller(x); });
  for (auto& e : tensors) {
    auto* t = e.as_tensor();
    tensor_map[t->name] = e;
  }
  return tensor_map;
}

std::set<Expr> CollectLoadTensors(Expr x,
                                  std::function<bool(const Expr*)>&& teller) {
  if (!x.defined()) return std::set<Expr>();
  struct Mutator : public ir::IRMutator<const Expr*> {
    std::function<bool(const Expr*)> teller;
    std::set<Expr> exprs;
    explicit Mutator(std::function<bool(const Expr*)>&& teller)
        : teller(std::move(teller)) {}

    void operator()(const Expr* expr) {
      ir::IRMutator<const Expr*>::Visit(expr, expr);
    }

    void Visit(const Load* op, const Expr* expr) override {
      if (teller(&op->tensor)) {
        exprs.insert(op->tensor);
      }
    }
  };

  Mutator mutator(std::move(teller));
  mutator(&x);
  return mutator.exprs;
}

std::set<Expr> CollectStoreTensors(Expr x,
                                   std::function<bool(const Expr*)>&& teller) {
  struct Mutator : public ir::IRMutator<const Expr*> {
    std::function<bool(const Expr*)> teller;
    std::set<Expr> exprs;
    explicit Mutator(std::function<bool(const Expr*)>&& teller)
        : teller(std::move(teller)) {}

    void operator()(const Expr* expr) {
      ir::IRMutator<const Expr*>::Visit(expr, expr);
    }

    void Visit(const Store* op, const Expr* expr) override {
      if (teller(&op->tensor)) {
        exprs.insert(op->tensor);
      }
    }
  };

  Mutator mutator(std::move(teller));
  mutator(&x);
  return mutator.exprs;
}

std::set<Expr> CollectReferencedTensors(
    Expr x, const std::function<bool(const Expr*)>& teller) {
  auto handle0 = teller;
  auto handle1 = teller;

  auto ts0 = CollectLoadTensors(x, std::move(handle0));
  auto ts1 = CollectLoadTensors(x, std::move(handle1));

  for (auto& item : ts1) {
    ts0.insert(item);
  }
  return ts0;
}

std::vector<std::string> CollectUndefinedVars(const Expr* e) {
  struct Mutator : public ir::IRMutator<const Expr*> {
    using ir::IRMutator<const Expr*>::Visit;
    std::vector<std::string> undefined_vars;
    std::set<std::string> defined_vars;
    std::set<std::string> used_vars;

    void CollectVarDef(const std::string& var) {
      PADDLE_ENFORCE_EQ(defined_vars.count(var),
                        false,
                        ::common::errors::InvalidArgument(
                            "var %s has been defined, please check", var));
      PADDLE_ENFORCE_EQ(used_vars.count(var),
                        false,
                        ::common::errors::InvalidArgument(
                            "var %s is wrongly used before definition", var));
      defined_vars.insert(var);
    }

    void ClearVar(const std::string& var) {
      defined_vars.erase(var);
      used_vars.erase(var);
    }

    void CollectVarUse(const std::string& var) {
      used_vars.insert(var);
      if (defined_vars.count(var) == 0) {
        undefined_vars.push_back(var);
      }
    }

    void Visit(const ir::Let* op, const Expr* expr) override {
      Expr symbol = op->symbol;
      auto var = symbol.as_var_ref();
      PADDLE_ENFORCE_EQ(var.defined(),
                        true,
                        ::common::errors::InvalidArgument(
                            "var %s is not defined, please check.", var->name));
      CollectVarDef(var->name);
      auto* node = expr->As<ir::Let>();
      Visit(&node->body, &node->body);
    }

    void Visit(const ir::For* op, const Expr* expr) override {
      CollectVarDef(op->loop_var->name);
      auto* node = expr->As<ir::For>();
      Visit(&node->min, &node->min);
      Visit(&node->extent, &node->extent);
      Visit(&node->body, &node->body);
      ClearVar(op->loop_var->name);
    }

    void Visit(const ir::Load* op, const Expr* expr) override {
      auto tensor = op->tensor.as_tensor_ref();
      CollectVarUse(tensor->name);
      auto* node = expr->As<ir::Load>();
      for (auto& idx : node->indices) Visit(&idx, &idx);
    }

    void Visit(const ir::Store* op, const Expr* expr) override {
      auto tensor = op->tensor.as_tensor_ref();
      CollectVarUse(tensor->name);
      auto* node = expr->As<ir::Store>();
      for (auto& idx : node->indices) Visit(&idx, &idx);
      Visit(&node->value, &node->value);
    }

    void Visit(const ir::_Var_* op, const Expr* expr) override {
      CollectVarUse(op->name);
      auto* node = expr->As<ir::_Var_>();
      if (node->lower_bound.defined()) {
        Visit(&node->lower_bound, &node->lower_bound);
      }
      if (node->upper_bound.defined()) {
        Visit(&node->upper_bound, &node->upper_bound);
      }
    }

    void Visit(const ir::Reduce* op, const Expr* expr) override {
      for (auto& axis : op->reduce_axis) {
        CollectVarDef(axis->name);
      }
      auto* node = expr->As<ir::Reduce>();
      if (node->init.defined()) Visit(&node->init, &node->init);
      Visit(&node->body, &node->body);
    }
  };

  Mutator mutator;
  mutator.Visit(e, e);
  return mutator.undefined_vars;
}

std::set<std::string> CollectTensorNeedsWrite(const Expr* e) {
  std::set<std::string> tensor_written;
  IrNodesCollector::handler_t handler = [&](const Expr* x) {
    if (x->As<ir::Store>()) {
      tensor_written.insert(
          x->As<ir::Store>()->tensor.As<ir::_Tensor_>()->name);
    }
    if (x->As<ir::_Tensor_>()) {
      tensor_written.insert(x->As<ir::_Tensor_>()->name);
    }
  };
  IrNodesCollector::teller_t teller = [](const Expr* x) {
    if (x->As<ir::Store>() && x->As<ir::Store>()->tensor.As<ir::_Tensor_>()) {
      return true;
    }
    if (x->As<ir::_Tensor_>() && x->As<ir::_Tensor_>()->is_call_node()) {
      return true;
    }
    return false;
  };
  IrNodesCollector collector(std::move(teller), std::move(handler), false);
  collector.Visit(e);
  return tensor_written;
}

std::set<std::string> CollectTensorNeedsWrite(const stmt::BlockRef& block) {
  std::set<std::string> tensor_written;
  const auto& CollectInSubExpr = [&](const Expr* e) {
    const auto& sub_expr_res = CollectTensorNeedsWrite(e);
    tensor_written.insert(sub_expr_res.begin(), sub_expr_res.end());
  };
  const auto& CollectInStmt = [&](const stmt::StmtRef& stmt) {
    if (stmt.isa<stmt::Store>()) {
      const auto& store_stmt = stmt.as<stmt::Store>();
      tensor_written.insert(store_stmt->tensor().As<ir::_Tensor_>()->name);
      CollectInSubExpr(&(store_stmt->value()));
    }
    if (stmt.isa<stmt::Let>()) {
      CollectInSubExpr(&(stmt.as<stmt::Let>()->body()));
    }
  };
  stmt::Visit(block, CollectInStmt, [](const stmt::StmtRef& stmt) {});
  return tensor_written;
}
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
