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

void RemoveGpuForLoops(ir::LoweredFunc fn) {
  struct Mutator : public ir::IRMutator<Expr *> {
    using ir::IRMutator<>::Visit;
    void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

    explicit Mutator(const ir::CudaAxisInfo &cuda_axis_info)
        : cuda_axis_info_(cuda_axis_info) {}

   private:
    ir::CudaAxisInfo cuda_axis_info_;

    void Visit(const ir::For *op, Expr *expr) override {
      switch (op->for_type()) {
        case ir::ForType::GPUBlock:
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          break;
        case ir::ForType::GPUThread:
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          break;
        default:
          auto *node = expr->As<ir::For>();
          IRMutator<>::Visit(&node->body, &node->body);
          break;
      }
    }

    bool NeedToReplaceForloopWithIfThenElse(const ir::For *n) const {
      // If the loop doesn't start from 0.
      if (n->min != cinn::common::make_const(0)) {
        return true;
      }

      // Get dim_size from the functions's cuda_axis_info as pre-condition.
      ir::Expr dim_size;
      switch (n->bind_info().for_type) {
        case ir::ForType::GPUThread:
          dim_size = cuda_axis_info_.block_dim(n->bind_info().offset);
          break;
        case ir::ForType::GPUBlock:
          dim_size = cuda_axis_info_.grid_dim(n->bind_info().offset);
          break;
      }
      if (!dim_size.defined()) {
        return true;
      }

      // If we can prove the loop's extent >= dim_size, then it's safe not
      // to add the IfThenElse guard.
      common::cas_intervals_t var_intervals =
          common::CollectVarIntervalsOfExprs({n->extent, dim_size});
      common::SymbolicExprAnalyzer analyzer{var_intervals};
      std::optional<bool> proved_ge = analyzer.ProveGE(n->extent, dim_size);
      if (proved_ge.value_or(false)) {
        return false;
      }
      return true;
    }

    void ReplaceForloopWithIfThenElse(Expr *expr) {
      auto *for_n = expr->As<ir::For>();

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
      if (for_n->min != cinn::common::make_const(0)) {
        AppendCondition(ir::GE::Make(for_n->loop_var, for_n->min));
      }
      // for(i, 2, min(M/2, 20)
      //            ^
      AppendCondition(ir::LT::Make(for_n->loop_var, for_n->extent));

      PADDLE_ENFORCE_EQ(condition.defined(),
                        true,
                        ::common::errors::InvalidArgument(
                            "Condition is not defined, please check."));

      *expr = ir::IfThenElse::Make(condition, for_n->body);
    }

    void Visit(const ir::PolyFor *op, Expr *expr) override {
      const auto msg =
          "PolyFor is not allowed for GPU, only For nodes are allowed";
      PADDLE_ENFORCE_EQ(
          op->for_type() != ir::ForType::GPUBlock,
          true,
          ::common::errors::InvalidArgument(
              "PolyFor is not allowed for GPU, only For nodes are allowed."));
      PADDLE_ENFORCE_EQ(
          op->for_type() != ir::ForType::GPUThread,
          true,
          ::common::errors::InvalidArgument(
              "PolyFor is not allowed for GPU, only For nodes are allowed."));
      PADDLE_ENFORCE_EQ(
          op->for_type() != ir::ForType::GPULane,
          true,
          ::common::errors::InvalidArgument(
              "PolyFor is not allowed for GPU, only For nodes are allowed."));
    }
  };

  Mutator mutator(fn->cuda_axis_info);
  mutator(&fn->body);
}

/**
 * The generated __syncthreads call will be wrapped with a `if (xxxx == 0) { }`,
 * this is the problem of isl AST output, drop it to make it run in all the
 * threads.
 */
void CudaSyncThreadsDropIfThenElse(ir::LoweredFunc fn) {
  struct Mutator : public ir::IRMutator<> {
    using ir::IRMutator<>::Visit;
    void operator()(ir::LoweredFunc fn) { Visit(fn.As<ir::_LoweredFunc_>()); }

    void Visit(const ir::IfThenElse *op, Expr *expr) override {
      blocked_statement_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      blocked_statement_stack.pop_back();
    }

    void Visit(const ir::Call *op, Expr *expr) override {
      if (op->name == runtime::intrinsic::cuda_sync_threads) {
        if (!blocked_statement_stack.empty()) {
          auto *last_for = blocked_statement_stack.back()->As<ir::IfThenElse>();
          if (auto *eq_n = last_for->condition.As<ir::EQ>()) {
            if (eq_n->b() == cinn::common::make_const(0)) {
              *blocked_statement_stack.back() = *expr;
            }
          }
        }
      }
    }

    // Collect all the statements with Block(include Block) to the statement.
    std::vector<ir::Expr *> blocked_statement_stack;
  };

  Mutator()(fn);
}

class RestructureVarNodes : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Load *load, Expr *op) override {
    std::vector<ir::Expr> indices_copied;
    for (const ir::Expr &indice : load->indices) {
      indices_copied.push_back(ir::ir_utils::IRCopy(indice));
    }
    op->As<ir::Load>()->indices = indices_copied;

    IRMutator::Visit(load, op);
  }

  void Visit(const ir::Store *store, Expr *op) override {
    std::vector<ir::Expr> indices_copied;
    for (const ir::Expr &indice : store->indices) {
      indices_copied.push_back(ir::ir_utils::IRCopy(indice));
    }
    op->As<ir::Store>()->indices = indices_copied;

    IRMutator::Visit(store, op);
  }
};

class ReplaceIndexToBindExpr : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize *op, Expr *expr) override {
    ir::ScheduleBlockRealize *schedule_block_realize =
        expr->As<ir::ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(
        schedule_block_realize->schedule_block.As<ir::ScheduleBlock>(),
        ::common::errors::InvalidArgument(
            "The type of schedule block realize should be ScheduleBlock!"));
    std::vector<ir::Expr> iter_values = schedule_block_realize->iter_values;
    ir::Expr body =
        schedule_block_realize->schedule_block.As<ir::ScheduleBlock>()->body;
    std::vector<ir::Var> iter_vars =
        schedule_block_realize->schedule_block.As<ir::ScheduleBlock>()
            ->iter_vars;

    PADDLE_ENFORCE_EQ(iter_values.size(),
                      iter_vars.size(),
                      ::common::errors::InvalidArgument(
                          "The size of iter values and iter vars is not equal,"
                          "where iter values:%d but iter vars:%d.",
                          iter_values.size(),
                          iter_vars.size()));
    for (int idx = 0; idx < iter_values.size(); ++idx) {
      ReplaceVarWithExpr(&body, iter_vars[idx], iter_values[idx]);
    }
    ir::IRMutator<>::Visit(&body, &body);
  }
};

class ReplaceLoopVarToGpu : public ir::IRMutator<> {
 public:
  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::For *op, Expr *expr) override {
    auto for_ir = expr->As<ir::For>();
    PADDLE_ENFORCE_NOT_NULL(for_ir,
                            ::common::errors::InvalidArgument(
                                "The type of expression should be For!"));

    auto bind_info = for_ir->bind_info();

    std::string var_name = "";
    if (bind_info.offset <= 0)
      var_name = "x";
    else if (bind_info.offset == 1)
      var_name = "y";
    else if (bind_info.offset == 2)
      var_name = "z";
    if (for_ir->is_gpu_block_binded()) {
      var_name = "blockIdx." + var_name;
      optim::ReplaceVarWithExpr(
          expr, op->loop_var, ir::Expr(ir::Var(var_name)));
    } else if (for_ir->is_gpu_thread_binded()) {
      var_name = "threadIdx." + var_name;
      optim::ReplaceVarWithExpr(
          expr, op->loop_var, ir::Expr(ir::Var(var_name)));
    }

    ir::IRMutator<>::Visit(&for_ir->body, &for_ir->body);
  }
  void Visit(const ir::PolyFor *op, Expr *expr) override {
    PADDLE_THROW(::common::errors::InvalidArgument("Unknown PolyFor!"));
  }
};

class SharedAxisVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (store->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPUShared) {
      for (auto &indice : store->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = cinn::common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
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
      for (auto &indice : load->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = cinn::common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  const std::vector<std::string> gpu_axis = {
      "blockIdx.x", "blockIdx.y", "blockIdx.z"};
};

class LocalAxisVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();

    ir::IRMutator<>::Visit(op, expr);
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (store->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      for (auto &indice : store->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = cinn::common::AutoSimplify(indice);
      }
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
      for (auto &indice : load->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = cinn::common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  const std::vector<std::string> gpu_axis = {"blockIdx.x",
                                             "blockIdx.y",
                                             "blockIdx.z",
                                             "threadIdx.x",
                                             "threadIdx.y",
                                             "threadIdx.z"};
};

class ReplaceUnitVarToZero : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    auto &indices = store->indices;
    for (auto &indice : indices) {
      for (auto var_ : loop_var_) {
        optim::ReplaceVarWithExpr(&indice, ir::Var(var_), ir::Expr(0));
      }
      indice = cinn::common::AutoSimplify(indice);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    auto &indices = load->indices;
    for (auto &indice : indices) {
      for (auto var_ : loop_var_) {
        optim::ReplaceVarWithExpr(&indice, ir::Var(var_), ir::Expr(0));
      }
      indice = cinn::common::AutoSimplify(indice);
    }

    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For *op, Expr *expr) override {
    PADDLE_ENFORCE_NOT_NULL(expr->As<ir::For>(),
                            ::common::errors::InvalidArgument(
                                "The type of expression should be For!"));
    auto for_ir = expr->As<ir::For>();
    auto var_name = for_ir->loop_var->name;
    auto extent_i = for_ir->extent;

    if (extent_i.is_constant() && extent_i.as_int64() == 1)
      loop_var_.insert(var_name);
    ir::IRMutator<>::Visit(op, expr);
    loop_var_.erase(var_name);
  }
  std::unordered_set<std::string> loop_var_;
};

void OptimizeExprGPU(Expr *expr) {
  VLOG(4) << "Before Optimize Expr:\n" << *expr;

  // Make independent copies for each load/store's indices to prevent cross
  // modification in later passes.
  RestructureVarNodes restructure_var_nodes;
  restructure_var_nodes(expr);

  // Replace iter_vars used in ScheduleBlocks to their corresponding iter_values
  // in ScheduleBlockRealizes.
  ReplaceIndexToBindExpr replace_index_to_bind_expr;
  replace_index_to_bind_expr(expr);

  // resize buffer axis
  BlockPassManager pass_manager;
  ir::stmt::BlockRef _block = ir::ConvertExprBlockToStmtBlock(*expr);
  pass_manager.AddPass(optim::CreateUpdateBufferAxisPass());
  pass_manager.Run(_block);
  ir::Expr new_expr = ir::ConvertStmtBlockToExprBlock(_block);
  *expr = new_expr;

  // Replace variables bound on block/thread to the actual blockIdx/threadIdx.
  ReplaceLoopVarToGpu replace_loop_var_to_gpu;
  replace_loop_var_to_gpu(expr);

  // Replace blockIdx in shared memory's indices to zero, because shared memory
  // cannot be accessed from another block.
  SharedAxisVisitor shared_axis_visitor;
  shared_axis_visitor(expr);

  // Replace blockIdx/threadIdx in local buffer's indices to zero, because local
  // buffers cannot be accessed from another block/thread.
  LocalAxisVisitor local_axis_visitor;
  local_axis_visitor(expr);

  // Replace variables that are in range [0, 1) to zero.
  ReplaceUnitVarToZero replace_unit_var_to_zero;
  replace_unit_var_to_zero(expr);
  VLOG(10) << "After ReplaceUnitVarToZero: \n" << *expr;
  ir::stmt::BlockRef func_body = ir::ConvertExprBlockToStmtBlock(*expr);
  EliminateCommonFactorOfLocalIndex(func_body);
  *expr = ir::ConvertStmtBlockToExprBlock(func_body);
  VLOG(10) << "After EliminateCommonFactorOfLocalIndex: \n" << *expr;

  ResizeBufferToMaxVarRange(expr);
  VLOG(4) << "After Optimize Expr: \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
