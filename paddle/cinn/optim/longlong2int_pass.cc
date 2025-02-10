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

#include "paddle/cinn/optim/longlong2int_pass.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/pass/pass_manager.h"

namespace cinn {
namespace optim {
namespace {
using ir::stmt::BlockRef;
using ir::stmt::For;
using ir::stmt::IfThenElse;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

void CastVarWithBound(cinn::ir::Var& var) {  // NOLINT
  if (!var.defined()) return;
  if (var->is_symbolic_constant) return;
  var->convert_int64_to_int32();
  auto lb = var->lower_bound;
  auto ub = var->upper_bound;
  if (lb.defined()) ir::TryElevateInt64ToInt32({lb});
  if (ub.defined()) ir::TryElevateInt64ToInt32({ub});
}
void CastBufferMeta(cinn::ir::Buffer& bf) {  // NOLINT
  if (!bf.defined()) return;
  std::for_each(bf->shape.begin(), bf->shape.end(), [&](cinn::ir::Expr& e) {
    ir::TryElevateInt64ToInt32({e});
  });
  std::for_each(bf->strides.begin(), bf->strides.end(), [&](cinn::ir::Expr& e) {
    ir::TryElevateInt64ToInt32({e});
  });
  ir::TryElevateInt64ToInt32({bf->elem_offset});
}

class CheckOverflow : public ir::stmt::StmtVisitor<> {
 public:
  bool operator()(const StmtRef& stmt) {
    VisitStmt(stmt);
    return is_overflow_;
  }
  bool operator()(const BlockRef& block) {
    VisitBlock(block);
    return is_overflow_;
  }

 private:
  void VisitStmt(const StmtRef& stmt) override {
    if (is_overflow_) return;
    ir::stmt::StmtVisitor<>::VisitStmt(stmt);
  }

  void VisitStmt(const For& for_stmt) override {
    if (!for_stmt->extent().is_constant()) is_overflow_ = true;
    if (!for_stmt->extent().type().is_index_type()) is_overflow_ = true;
    if (curr_product_ > INT_MAX) is_overflow_ = true;

    if (is_overflow_) return;

    curr_product_ *= for_stmt->extent().as_int64();
    VisitBlock(for_stmt->body());
    curr_product_ /= for_stmt->extent().as_int64();
  }

  void VisitStmt(const Schedule& schedule_stmt) override {
    VisitBlock(schedule_stmt->body());
  }

  void VisitStmt(const IfThenElse& stmt) override {
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(const ir::stmt::Let& stmt) override { return; }
  void VisitStmt(const ir::stmt::Store& stmt) override { return; }
  void VisitStmt(const ir::stmt::Alloc& stmt) override { return; }
  void VisitStmt(const ir::stmt::Free& stmt) override { return; }
  void VisitStmt(const ir::stmt::Evaluate& stmt) override { return; }

 private:
  int64_t curr_product_ = 1;
  bool is_overflow_ = false;
};

class CastLonglong2IntMutator : public ir::IRMutator<> {
 public:
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto node = expr->As<ir::_Tensor_>();
    std::for_each(node->shape.begin(),
                  node->shape.end(),
                  [&](cinn::ir::Expr& e) { ir::TryElevateInt64ToInt32({e}); });
    CastBufferMeta(node->buffer);
  }
  void Visit(const ir::Load* op, Expr* expr) override {
    auto node = expr->As<ir::Load>();
    std::for_each(node->indices.begin(),
                  node->indices.end(),
                  [&](cinn::ir::Expr& e) { ir::TryElevateInt64ToInt32({e}); });
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }

  void Visit(const ir::Select* op, Expr* expr) override {
    auto node = expr->As<ir::Select>();
    auto cond = node->condition;
    if (cond.is_cmp()) {
      ir::TryElevateInt64ToInt32({cond->operand(0), cond->operand(1)});
    }
    ir::IRMutator<>::Visit(&node->true_value, &node->true_value);
    ir::IRMutator<>::Visit(&node->false_value, &node->false_value);
  }
};

class LongLong2IntStmtPass : public StmtPass {
 public:
  LongLong2IntStmtPass() : StmtPass("longlong2int_stmt") {}
  LogicalResult Run(ir::stmt::StmtRef stmt) override;
};

class LongLong2IntExprPass : public ExprPass {
 public:
  LongLong2IntExprPass() : ExprPass("longlong2int_expr") {}
  LogicalResult Run(ir::Expr* expr) override;
};
}  // namespace

LogicalResult LongLong2IntStmtPass::Run(ir::stmt::StmtRef stmt) {
  auto CastStore = [](StmtRef stmt) {
    Store store_stmt = stmt.as<Store>();
    for (Expr index : store_stmt->indices()) {
      ir::TryElevateInt64ToInt32({index});
    }
  };

  auto CastIfThenElse = [](StmtRef stmt) {
    IfThenElse if_stmt = stmt.as<IfThenElse>();
    Expr cond = if_stmt->condition();
    if (cond.is_cmp()) {
      ir::TryElevateInt64ToInt32({cond->operand(0), cond->operand(1)});
    }
  };

  auto CastFor = [](StmtRef stmt) {
    For for_stmt = stmt.as<For>();
    ir::Var loop_var = for_stmt->loop_var();
    CastVarWithBound(loop_var);
    ir::TryElevateInt64ToInt32({for_stmt->min()});
    ir::TryElevateInt64ToInt32({for_stmt->extent()});
  };

  auto CastSchedule = [](StmtRef stmt) {
    Schedule schedule_stmt = stmt.as<Schedule>();
    std::vector<Var> iter_vars = schedule_stmt->iter_vars();
    std::for_each(iter_vars.begin(), iter_vars.end(), [&](cinn::ir::Var& v) {
      CastVarWithBound(v);
    });

    std::vector<Expr> iter_values = schedule_stmt->iter_values();
    std::for_each(iter_values.begin(),
                  iter_values.end(),
                  [&](cinn::ir::Expr& e) { ir::TryElevateInt64ToInt32({e}); });

    for (auto& buffer_range : schedule_stmt->read_buffers()) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::vector<Var> ranges = range->ranges;
        std::for_each(ranges.begin(), ranges.end(), [&](cinn::ir::Var& v) {
          CastVarWithBound(v);
        });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }

    for (auto& buffer_range : schedule_stmt->write_buffers()) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::vector<Var> ranges = range->ranges;

        std::for_each(ranges.begin(), ranges.end(), [&](cinn::ir::Var& v) {
          CastVarWithBound(v);
        });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }
  };

  switch (stmt->stmt_type()) {
    case ir::StmtNodeTy::Store:
      CastStore(stmt);
      break;

    case ir::StmtNodeTy::IfThenElse:
      CastIfThenElse(stmt);
      break;

    case ir::StmtNodeTy::For:
      CastFor(stmt);
      break;

    case ir::StmtNodeTy::Schedule:
      CastSchedule(stmt);
      break;
    default:
      break;
  }
  return LogicalResult::success();
}

LogicalResult LongLong2IntExprPass::Run(ir::Expr* expr) {
  CastLonglong2IntMutator narrow;
  narrow(expr);
  return LogicalResult::success();
}
std::unique_ptr<StmtPass> CreateLongLong2IntStmtPass() {
  return std::make_unique<LongLong2IntStmtPass>();
}

std::unique_ptr<ExprPass> CreateLongLong2IntExprPass() {
  return std::make_unique<LongLong2IntExprPass>();
}

// Check if the given block can be converted from long long to int,
// A.K.A. the product of the extents of all possible nested loops is within
// INT_MAX
bool CanApplyLongLong2Int(ir::stmt::BlockRef block) {
  CheckOverflow check_overflow;
  return !check_overflow(block);
}

void TryCastLonglong2Int(ir::stmt::BlockRef block) {
  if (CanApplyLongLong2Int(block)) {
    StmtPassManager stmt_pass_manager;
    stmt_pass_manager.AddPass(CreateLongLong2IntStmtPass());
    ExprPassManager expr_pass_manager;
    expr_pass_manager.AddPass(CreateLongLong2IntExprPass());

    stmt_pass_manager.Run(block);
    expr_pass_manager.Run(block);
  }
}

}  // namespace optim
}  // namespace cinn
