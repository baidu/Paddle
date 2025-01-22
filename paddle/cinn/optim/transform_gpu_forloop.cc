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

#include "paddle/cinn/optim/transform_gpu_forloop.h"

#include <algorithm>
#include <map>
#include <stack>
#include <string>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/optim/eliminate_common_factor_of_local_index.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/longlong2int_pass.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/optim/resize_buffer.h"
#include "paddle/cinn/optim/update_buffer_axis_pass.h"
#include "paddle/cinn/pass/pass_manager.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"

PD_DECLARE_bool(cinn_longlong2int);
namespace cinn {
namespace optim {

class GPUForLoopsMutator {
 public:
  void operator()(ir::stmt::BlockRef block) { VisitBlock(block); }

  explicit GPUForLoopsMutator(const ir::CudaAxisInfo &cuda_axis_info)
      : cuda_axis_info_(cuda_axis_info) {}

 private:
  void VisitBlock(ir::stmt::BlockRef block) {
    std::vector<ir::stmt::StmtRef> stmts = block->stmts();
    std::vector<ir::stmt::StmtRef> new_stmts;
    for (ir::stmt::StmtRef &stmt : stmts) {
      switch (stmt->stmt_type()) {
        case ir::StmtNodeTy::For: {
          ir::stmt::For for_stmt = stmt.as<ir::stmt::For>();
          switch (VisitStmt(for_stmt)) {
            case 0: {
              ReplaceForloopWithIfThenElse(stmt);
              ir::stmt::IfThenElse if_stmt = stmt.as<ir::stmt::IfThenElse>();
              // Visit true case only
              VisitBlock(if_stmt->true_case());
              new_stmts.push_back(if_stmt);
              break;
            }
            case 1: {
              VisitBlock(for_stmt->body());
              for (const auto &stmt : for_stmt->body()->stmts()) {
                new_stmts.push_back(stmt);
              }
              break;
            }
            case 2: {
              VisitBlock(for_stmt->body());
              new_stmts.push_back(for_stmt);
              break;
            }
            default:
              break;
          }
          break;
        }
        case ir::StmtNodeTy::Schedule: {
          ir::stmt::Schedule schedule = stmt.as<ir::stmt::Schedule>();
          VisitBlock(schedule->body());
          new_stmts.push_back(stmt);
          break;
        }
        case ir::StmtNodeTy::IfThenElse: {
          ir::stmt::IfThenElse if_then_else = stmt.as<ir::stmt::IfThenElse>();
          VisitBlock(if_then_else->true_case());
          if (if_then_else->false_case().defined()) {
            VisitBlock(if_then_else->true_case());
          }
          new_stmts.push_back(stmt);
          break;
        }
        default:
          new_stmts.push_back(stmt);
          break;
      }
    }
    block->set_stmts(new_stmts);
  }

  // NOLINTNEXTLINE(runtime/references)
  int VisitStmt(const ir::stmt::For &stmt) {
    if (stmt->for_type() == ir::ForType::GPUBlock ||
        stmt->for_type() == ir::ForType::GPUThread) {
      if (NeedToReplaceForloopWithIfThenElse(stmt)) {
        // Replace the GPU For loop with an IfThenElse.
        return 0;
      } else {
        // Replace the GPU For loop with its body.
        return 1;
      }
    }
    // Keep this For loop, traverse the body of it.
    return 2;
  }

  bool NeedToReplaceForloopWithIfThenElse(const ir::stmt::For &stmt) const {
    // If the loop doesn't start from 0.
    if (stmt->min() != cinn::common::make_const(0)) {
      return true;
    }

    // Get dim_size from the functions's cuda_axis_info as pre-condition.
    ir::Expr dim_size;
    switch (stmt->bind_info().for_type) {
      case ir::ForType::GPUThread:
        dim_size = cuda_axis_info_.block_dim(stmt->bind_info().offset);
        break;
      case ir::ForType::GPUBlock:
        dim_size = cuda_axis_info_.grid_dim(stmt->bind_info().offset);
        break;
    }
    if (!dim_size.defined()) {
      return true;
    }

    // If we can prove the loop's extent >= dim_size, then it's safe not
    // to add the IfThenElse guard.
    common::cas_intervals_t var_intervals =
        common::CollectVarIntervalsOfExprs({stmt->extent(), dim_size});
    common::SymbolicExprAnalyzer analyzer{var_intervals};
    std::optional<bool> proved_ge = analyzer.ProveGE(stmt->extent(), dim_size);
    if (proved_ge.value_or(false)) {
      return false;
    }
    return true;
  }

  // NOLINTNEXTLINE(runtime/references)
  void ReplaceForloopWithIfThenElse(ir::stmt::StmtRef &stmt) {
    ir::stmt::For for_n = stmt.as<ir::stmt::For>();

    Expr condition;
    const auto AppendCondition = [&](Expr new_cond) {
      if (condition.defined()) {
        condition = ir::And::Make(condition, new_cond);
      } else {
        condition = new_cond;
      }
    };

    // for(i, 2, 100);
    //        ^
    if (for_n->min() != cinn::common::make_const(0)) {
      AppendCondition(ir::GE::Make(for_n->loop_var(), for_n->min()));
    }
    // for(i, 2, min(M/2, 20)
    //            ^
    AppendCondition(ir::LT::Make(for_n->loop_var(), for_n->extent()));

    PADDLE_ENFORCE_EQ(condition.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Condition is not defined, please check."));

    stmt = ir::stmt::IfThenElse(condition, for_n->body());
  }

  ir::CudaAxisInfo cuda_axis_info_;
};

LogicalResult RemoveGpuForLoopsPass::Run(ir::LoweredFunc fn) {
  GPUForLoopsMutator mutator(fn->cuda_axis_info);
  mutator(fn->body_block);
  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateRemoveGpuForLoopsPass() {
  return std::make_unique<RemoveGpuForLoopsPass>();
}

/**
 * The generated __syncthreads call will be wrapped with a `if (xxxx == 0) { }`,
 * this is the problem of isl AST output, drop it to make it run in all the
 * threads.
 */
class DropIfThenElseMutator : public ir::IRMutator<>,
                              public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::Expr expr) {
    ir::Expr _expr = expr;
    ir::IRMutator<>::Visit(&_expr, &_expr);
  }

  void operator()(ir::stmt::BlockRef block) { VisitBlock(block); }

 private:
  void VisitBlock(ir::stmt::BlockRef block) override {
    std::vector<ir::stmt::StmtRef> stmts = block->stmts();
    std::vector<ir::stmt::StmtRef> new_stmts;
    for (ir::stmt::StmtRef &stmt : stmts) {
      if (stmt->stmt_type() == ir::StmtNodeTy::IfThenElse) {
        const ir::stmt::IfThenElse &node = stmt.as<ir::stmt::IfThenElse>();
        VisitStmt(node);
        if (!blocked_statement_stack.empty() &&
            blocked_statement_stack.back().count(node) &&
            blocked_statement_stack.back()[node]) {
          // If the IfThenElse statement is marked, replace it with its
          // true_case statements.
          const ir::stmt::BlockRef true_case_block = node->true_case();
          for (const auto &true_stmt : true_case_block->stmts()) {
            new_stmts.push_back(true_stmt);
          }
        } else {
          new_stmts.push_back(stmt);
        }
      } else {
        new_stmts.push_back(stmt);
      }
    }
    block->set_stmts(new_stmts);
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    // Initialize the current IfThenElse statement as not marked.
    if (blocked_statement_stack.empty()) {
      std::map<ir::stmt::IfThenElse, bool> map({{stmt, false}});
      blocked_statement_stack.push_back(map);
    }
    blocked_statement_stack.back()[stmt] = false;

    // Traverse the condition and true_case.
    operator()(stmt->condition());
    operator()(stmt->true_case());

    // Pop the current IfThenElse statement from the stack.
    blocked_statement_stack.back().erase(stmt);
    if (blocked_statement_stack.back().empty()) {
      blocked_statement_stack.pop_back();
    }
  }

  void VisitStmt(ir::stmt::For stmt) override {
    operator()(stmt->min());
    operator()(stmt->extent());
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::Schedule stmt) override { VisitBlock(stmt->body()); }

  void VisitStmt(ir::stmt::Let stmt) override { operator()(stmt->body()); }

  void VisitStmt(ir::stmt::Free stmt) override {}

  void VisitStmt(ir::stmt::Store stmt) override {}

  void VisitStmt(ir::stmt::Alloc) override {}

  void VisitStmt(ir::stmt::Evaluate) override {}

  void Visit(const ir::Call *op, Expr *expr) override {
    if (op->name == runtime::intrinsic::cuda_sync_threads) {
      if (!blocked_statement_stack.empty()) {
        auto &current_if_map = blocked_statement_stack.back();
        for (auto &[if_stmt, marked] : current_if_map) {
          if (auto *eq_n = if_stmt->condition().As<ir::EQ>()) {
            if (eq_n->b() == cinn::common::make_const(0)) {
              marked = true;
            }
          }
        }
      }
    }
  }

  // Collect all the statements with Block(include Block) to the statement.
  std::vector<std::map<ir::stmt::IfThenElse, bool>> blocked_statement_stack;
};

LogicalResult CudaSyncThreadsDropIfThenElsePass::Run(ir::LoweredFunc fn) {
  DropIfThenElseMutator mutator;
  mutator(fn->body_block);
  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateCudaSyncThreadsDropIfThenElsePass() {
  return std::make_unique<CudaSyncThreadsDropIfThenElsePass>();
}

class RestructureVarNodes : public ir::IRMutator<>,
                            public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::stmt::BlockRef block) { VisitBlock(block); }

 private:
  void Visit(const ir::Load *load, Expr *op) override {
    std::vector<ir::Expr> indices_copied;
    for (const ir::Expr &indice : load->indices) {
      indices_copied.push_back(ir::ir_utils::IRCopy(indice));
    }
    op->As<ir::Load>()->indices = indices_copied;

    IRMutator::Visit(load, op);
  }

  void VisitStmt(ir::stmt::Store stmt) override {
    std::vector<ir::Expr> indices_copied;
    for (const ir::Expr &indice : stmt->indices()) {
      indices_copied.push_back(ir::ir_utils::IRCopy(indice));
    }
    stmt->set_indices(indices_copied);

    ir::Expr value = stmt->value();
    IRMutator::Visit(&value, &value);
    stmt->set_value(value);
  }

  void VisitStmt(ir::stmt::For stmt) override { operator()(stmt->body()); }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    operator()(stmt->true_case());
    if (stmt->false_case().defined()) {
      operator()(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override { operator()(stmt->body()); }

  void VisitStmt(ir::stmt::Let stmt) override {
    ir::Expr body = stmt->body();
    IRMutator::Visit(&body, &body);
    stmt->set_body(body);
  }

  void VisitStmt(ir::stmt::Alloc) override {}

  void VisitStmt(ir::stmt::Evaluate) override {}

  void VisitStmt(ir::stmt::Free) override {}
};

class ReplaceIndexToBindExpr {
 public:
  void operator()(ir::stmt::BlockRef block) {
    for (ir::stmt::StmtRef stmt : block->stmts()) {
      switch (stmt->stmt_type()) {
        case ir::StmtNodeTy::For: {
          operator()(stmt.as<ir::stmt::For>()->body());
          break;
        }
        case ir::StmtNodeTy::Schedule: {
          VisitStmt(stmt.as<ir::stmt::Schedule>());
          break;
        }
        case ir::StmtNodeTy::IfThenElse: {
          ir::stmt::IfThenElse if_node = stmt.as<ir::stmt::IfThenElse>();
          operator()(if_node->true_case());
          if (if_node->false_case().defined()) {
            operator()(if_node->false_case());
          }
          break;
        }
        default:
          break;
      }
    }
  }

 private:
  void VisitStmt(ir::stmt::Schedule stmt) {
    std::vector<ir::Expr> iter_values = stmt->iter_values();
    std::vector<ir::Var> iter_vars = stmt->iter_vars();
    ir::stmt::BlockRef body = stmt->body();

    PADDLE_ENFORCE_EQ(iter_values.size(),
                      iter_vars.size(),
                      ::common::errors::InvalidArgument(
                          "The size of iter values and iter vars is not equal,"
                          "where iter values:%d but iter vars:%d.",
                          iter_values.size(),
                          iter_vars.size()));
    for (int idx = 0; idx < iter_values.size(); ++idx) {
      ReplaceVarWithExpr<ir::stmt::BlockRef>(
          body, iter_vars[idx], iter_values[idx]);
    }
    stmt->set_body(body);
    operator()(stmt->body());
  }
};

class ReplaceLoopVarToGpu {
 public:
  void operator()(ir::stmt::BlockRef block) {
    std::vector<ir::stmt::StmtRef> stmts = block->stmts();
    for (ir::stmt::StmtRef stmt : stmts) {
      switch (stmt->stmt_type()) {
        case ir::StmtNodeTy::For: {
          VisitStmt(stmt.as<ir::stmt::For>());
          break;
        }
        case ir::StmtNodeTy::Schedule: {
          operator()(stmt.as<ir::stmt::Schedule>()->body());
          break;
        }
        case ir::StmtNodeTy::IfThenElse: {
          ir::stmt::IfThenElse if_node = stmt.as<ir::stmt::IfThenElse>();
          operator()(if_node->true_case());
          if (if_node->false_case().defined()) {
            operator()(if_node->false_case());
          }
          break;
        }
        default:
          break;
      }
    }
    block->set_stmts(stmts);
  }

 private:
  void VisitStmt(ir::stmt::For stmt) {
    auto bind_info = stmt->bind_info();

    std::string var_name = "";
    if (bind_info.offset <= 0)
      var_name = "x";
    else if (bind_info.offset == 1)
      var_name = "y";
    else if (bind_info.offset == 2)
      var_name = "z";
    if (stmt->is_gpu_block_binded()) {
      var_name = "blockIdx." + var_name;
      optim::ReplaceVarWithExpr<ir::stmt::StmtRef>(
          stmt, stmt->loop_var(), ir::Expr(ir::Var(var_name)));
    } else if (stmt->is_gpu_thread_binded()) {
      var_name = "threadIdx." + var_name;
      optim::ReplaceVarWithExpr<ir::stmt::StmtRef>(
          stmt, stmt->loop_var(), ir::Expr(ir::Var(var_name)));
    }

    operator()(stmt->body());
  }
};

class SharedAxisVisitor : public ir::IRMutator<>,
                          public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }
  void operator()(ir::stmt::BlockRef block) {
    ir::stmt::StmtMutator<>::VisitBlock(block);
  }

 private:
  void VisitStmt(ir::stmt::Store stmt) override {
    if (!stmt->tensor().as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (stmt->tensor().as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPUShared) {
      std::vector<ir::Expr> indices = stmt->indices();
      for (ir::Expr &index : indices) {
        for (const std::string &axis : gpu_axis) {
          optim::ReplaceVarWithExpr<ir::Expr *>(
              &index, ir::Var(axis), ir::Expr(0));
        }
        index = cinn::optim::ArithSimplify(index);
      }
      stmt->set_indices(indices);
    }
    ir::Expr value = stmt->value();
    ir::IRMutator<>::Visit(&value, &value);
    stmt->set_value(value);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (load->is_addr_scalar()) {
      return;
    }
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPUShared) {
      for (auto &index : load->indices) {
        for (const std::string &axis : gpu_axis) {
          optim::ReplaceVarWithExpr<ir::Expr *>(
              &index, ir::Var(axis), ir::Expr(0));
        }
        index = cinn::optim::ArithSimplify(index);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void VisitStmt(ir::stmt::For stmt) override {
    ir::Expr min = stmt->min();
    ir::Expr extent = stmt->extent();
    operator()(&min);
    operator()(&extent);
    stmt->set_min(min);
    stmt->set_extent(extent);
    operator()(stmt->body());
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    ir::Expr condition = stmt->condition();
    operator()(&condition);
    stmt->set_condition(condition);

    operator()(stmt->true_case());
    if (stmt->false_case().defined()) {
      operator()(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    std::vector<ir::Expr> iter_values = stmt->iter_values();
    for (ir::Expr &iter_value : iter_values) {
      operator()(&iter_value);
    }
    stmt->set_iter_values(iter_values);
    operator()(stmt->body());
  }

  void VisitStmt(ir::stmt::Let stmt) override {
    ir::Expr body = stmt->body();
    ir::IRMutator<>::Visit(&body, &body);
    stmt->set_body(body);
  }

  void VisitStmt(ir::stmt::Alloc) override {}

  void VisitStmt(ir::stmt::Evaluate) override {}

  void VisitStmt(ir::stmt::Free) override {}

  const std::vector<std::string> gpu_axis = {
      "blockIdx.x", "blockIdx.y", "blockIdx.z"};
};

class LocalAxisVisitor : public ir::IRMutator<>,
                         public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }
  void operator()(ir::stmt::BlockRef block) {
    ir::stmt::StmtMutator<>::VisitBlock(block);
  }

 private:
  void VisitStmt(ir::stmt::Store stmt) override {
    ir::Expr value = stmt->value();
    operator()(&value);
    stmt->set_value(value);

    if (!stmt->tensor().as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (stmt->tensor().as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      std::vector<ir::Expr> indices = stmt->indices();
      for (ir::Expr &index : indices) {
        for (const std::string &axis : gpu_axis) {
          optim::ReplaceVarWithExpr<ir::Expr *>(
              &index, ir::Var(axis), ir::Expr(0));
        }
        index = cinn::optim::ArithSimplify(index);
      }
      stmt->set_indices(indices);
    }
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();

    if (load->is_addr_scalar()) {
      return;
    }
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      for (ir::Expr &index : load->indices) {
        for (const std::string &axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&index, ir::Var(axis), ir::Expr(0));
        }
        index = cinn::optim::ArithSimplify(index);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void VisitStmt(ir::stmt::For stmt) override { operator()(stmt->body()); }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    operator()(stmt->true_case());
    if (stmt->false_case().defined()) {
      operator()(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    std::vector<ir::Expr> iter_values = stmt->iter_values();
    for (ir::Expr &iter_value : iter_values) {
      operator()(&iter_value);
    }
    stmt->set_iter_values(iter_values);
    operator()(stmt->body());
  }

  void VisitStmt(ir::stmt::Let stmt) override {
    ir::Expr body = stmt->body();
    ir::IRMutator<>::Visit(&body, &body);
    stmt->set_body(body);
  }

  void VisitStmt(ir::stmt::Alloc) override {}

  void VisitStmt(ir::stmt::Evaluate) override {}

  void VisitStmt(ir::stmt::Free) override {}

  const std::vector<std::string> gpu_axis = {"blockIdx.x",
                                             "blockIdx.y",
                                             "blockIdx.z",
                                             "threadIdx.x",
                                             "threadIdx.y",
                                             "threadIdx.z"};
};

class ReplaceUnitVarToZero : public ir::IRMutator<>,
                             public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }
  void operator()(ir::stmt::BlockRef block) {
    ir::stmt::StmtMutator<>::VisitBlock(block);
  }

 private:
  void VisitStmt(ir::stmt::Store stmt) override {
    if (!stmt->tensor().as_tensor_ref()->buffer.defined()) {
      return;
    }

    std::vector<ir::Expr> indices = stmt->indices();
    for (ir::Expr &index : indices) {
      for (const std::string &var_ : loop_var_) {
        optim::ReplaceVarWithExpr<ir::Expr *>(
            &index, ir::Var(var_), ir::Expr(0));
      }
      index = cinn::optim::ArithSimplify(index);
    }
    stmt->set_indices(indices);
    ir::Expr value = stmt->value();
    operator()(&value);
    stmt->set_value(value);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    auto &indices = load->indices;
    for (auto &index : indices) {
      for (const std::string &var_ : loop_var_) {
        optim::ReplaceVarWithExpr<ir::Expr *>(
            &index, ir::Var(var_), ir::Expr(0));
      }
      index = cinn::optim::ArithSimplify(index);
    }

    ir::IRMutator<>::Visit(op, expr);
  }

  void VisitStmt(ir::stmt::For stmt) override {
    auto var_name = stmt->loop_var()->name;
    auto extent_i = stmt->extent();

    if (extent_i.is_constant() && extent_i.as_int64() == 1)
      loop_var_.insert(var_name);
    operator()(stmt->body());
    loop_var_.erase(var_name);
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    operator()(stmt->true_case());
    if (stmt->false_case().defined()) {
      operator()(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    std::vector<ir::Expr> iter_values = stmt->iter_values();
    for (ir::Expr &iter_value : iter_values) {
      operator()(&iter_value);
    }
    stmt->set_iter_values(iter_values);
    operator()(stmt->body());
  }

  void VisitStmt(ir::stmt::Let stmt) override {
    ir::Expr body = stmt->body();
    ir::IRMutator<>::Visit(&body, &body);
    stmt->set_body(body);
  }

  void VisitStmt(ir::stmt::Alloc) override {}

  void VisitStmt(ir::stmt::Evaluate) override {}

  void VisitStmt(ir::stmt::Free) override {}

  std::unordered_set<std::string> loop_var_;
};

// void OptimizeExprGPU(Expr *expr) {
void OptimizeExprGPU(ir::stmt::BlockRef block) {
  VLOG(4) << "Before Optimize Expr:\n" << block;

  // ir::stmt::BlockRef block = ir::ConvertExprBlockToStmtBlock(*expr);
  // Make independent copies for each load/store's indices to prevent cross
  // modification in later passes.
  RestructureVarNodes restructure_var_nodes;
  restructure_var_nodes(block);

  // Replace iter_vars used in ScheduleBlocks to their corresponding iter_values
  // in ScheduleBlockRealizes.
  ReplaceIndexToBindExpr replace_index_to_bind_expr;
  replace_index_to_bind_expr(block);

  // resize buffer axis
  BlockPassManager pass_manager;
  pass_manager.AddPass(optim::CreateUpdateBufferAxisPass());
  pass_manager.Run(block);
  ir::Expr new_expr = ir::ConvertStmtBlockToExprBlock(block);

  // Replace variables bound on block/thread to the actual blockIdx/threadIdx.
  ReplaceLoopVarToGpu replace_loop_var_to_gpu;
  replace_loop_var_to_gpu(block);

  // Replace blockIdx in shared memory's indices to zero, because shared memory
  // cannot be accessed from another block.
  SharedAxisVisitor shared_axis_visitor;
  shared_axis_visitor(block);

  // Replace blockIdx/threadIdx in local buffer's indices to zero, because local
  // buffers cannot be accessed from another block/thread.
  LocalAxisVisitor local_axis_visitor;
  local_axis_visitor(block);

  // Replace variables that are in range [0, 1) to zero.
  ReplaceUnitVarToZero replace_unit_var_to_zero;
  replace_unit_var_to_zero(block);

  EliminateCommonFactorOfLocalIndex(block);
  VLOG(10) << "After EliminateCommonFactorOfLocalIndex: \n" << block;

  ir::Expr expr = ir::ConvertStmtBlockToExprBlock(block);

  ResizeBufferToMaxVarRange(&expr);

  block = ir::ConvertExprBlockToStmtBlock(expr);

  if (FLAGS_cinn_longlong2int) {
    VLOG(10) << "Before CastLonglong2Int: \n" << block;
    TryCastLonglong2Int(block);
    VLOG(10) << "After CastLonglong2Int: \n" << block;
  }

  VLOG(4) << "After Optimize Expr: \n" << block;
}

}  // namespace optim
}  // namespace cinn
