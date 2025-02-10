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

#include "paddle/cinn/optim/rearrange_load_instruction_pass.h"
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/phi/core/enforce.h"

PD_DECLARE_bool(cinn_enable_rearrange_load);

namespace cinn {
namespace optim {

using ir::stmt::Alloc;
using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::For;
using ir::stmt::Free;
using ir::stmt::IfThenElse;
using ir::stmt::Let;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

namespace {
constexpr int MaxRearrangeLoadNum = 8;

template <typename NodeTy>
bool ContainsExprNodeInExpr(const ir::Expr& expr) {
  auto res = ir::ir_utils::CollectIRNodes(
      expr,
      [](const ir::Expr* x) { return x->As<NodeTy>(); },
      /* uniq_target = */ true);
  return !res.empty();
}

template <typename StmtType>
bool ContainsStmtInStmt(const StmtRef& stmt) {
  bool found = false;
  auto CheckStmt = [&found](const StmtRef& stmt) {
    if (!found && stmt.isa<StmtType>()) {
      found = true;
    }
  };
  ir::stmt::Visit(stmt, CheckStmt, [](const StmtRef&) {});
  return found;
}

/**
 * Calculate the buffer size as a constant. For dynamic dims, since they are
 * difficult to compare, we just estimate them to be 32.
 * Note: this is a heuristic optimization, so the exact number is not very
 * important.
 */
int64_t EstimateBufferSize(const ir::Buffer& buffer) {
  int64_t size = 1;
  for (auto& dim_size : buffer->shape) {
    if (dim_size.is_constant()) {
      size *= dim_size.as_int64();
    } else {
      size *= 32;
    }
  }
  return size;
}

std::vector<std::string> SortLoadsByBufferSizes(
    const std::unordered_map<std::string, const ir::Expr*>& load_map,
    std::vector<std::string> load_list) {
  // Calculate the buffer sizes of loads (with estimation).
  std::map<ir::Buffer, int64_t> buffer_size_map;
  for (auto& [_, load_expr] : load_map) {
    auto& buffer = load_expr->As<ir::Load>()->tensor.as_tensor()->buffer;
    if (buffer_size_map.count(buffer)) {
      continue;
    }
    buffer_size_map[buffer] = EstimateBufferSize(buffer);
  }

  const auto GetBufferSize = [&](const std::string& key) {
    auto& buffer = load_map.at(key)->As<ir::Load>()->tensor.as_tensor()->buffer;
    return buffer_size_map[buffer];
  };

  // Sort loads by their buffer sizes from large to small.
  // Note: we use stable sort here, because for equal-size loads, we want to
  // keep their original order.
  std::stable_sort(load_list.begin(),
                   load_list.end(),
                   [&](const std::string& key1, const std::string& key2) {
                     return GetBufferSize(key1) > GetBufferSize(key2);
                   });
  return load_list;
}

struct LoadCollector : public ir::IRMutator<> {
  explicit LoadCollector(const std::set<ir::Buffer>& locally_defined_buffers)
      : locally_defined_buffers_(locally_defined_buffers) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  // Collect loads that meet the following criteria:
  // 1) It is loading from global memory. Local loads are simply register
  //    reads and do not require rearrangement.
  // 2) The value being loaded is not defined locally by a previous store. In
  //    such cases, the value resides in a register rather than in memory,
  //    thus doesn't need rearrangement. This criteria also prevents
  //    data-dependency harzards.
  // 3) It doesn't contains indirect indices (i.e. loads within indices).
  //    Indirect indices are hard to manage and are seldom seem, so we choose
  //    not to handle them.
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto& buffer = op->tensor.as_tensor()->buffer;
    if (buffer->memory_type != ir::MemoryType::Heap) {
      return;
    }
    if (locally_defined_buffers_.count(buffer) > 0) {
      return;
    }
    for (auto& index_expr : op->indices) {
      if (ContainsExprNodeInExpr<ir::Load>(index_expr)) {
        return;
      }
    }
    std::string key = utils::GetStreamCnt(*expr);
    CollectLoad(key, expr);
  }

  // Handle Select as a special op.
  // Since Select evaluates only one of its two branches, we can rearrange a
  // load in Select only if the load appears in both branches, otherwise we
  // may violate the control dependency.
  void Visit(const ir::Select* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Select>();
    ir::IRMutator<>::Visit(&node->condition, &node->condition);

    LoadCollector true_collector(locally_defined_buffers_);
    true_collector(&node->true_value);
    LoadCollector false_collector(locally_defined_buffers_);
    false_collector(&node->false_value);

    for (auto& key : true_collector.load_list_) {
      if (false_collector.load_map_.count(key) > 0) {
        CollectLoad(key, true_collector.load_map_[key]);
      }
    }
  }

  void CollectLoad(const std::string& key, const ir::Expr* expr) {
    auto [_, is_first] = load_map_.emplace(key, expr);
    if (is_first) {
      load_list_.push_back(key);
    }
  }

 public:
  // map from the signatures of loads to the load nodes
  std::unordered_map<std::string, const ir::Expr*> load_map_;
  // list of the signatures of loads in the order they are visited
  std::vector<std::string> load_list_;

 private:
  const std::set<ir::Buffer>& locally_defined_buffers_;
};

struct LoadReplacer : public ir::IRMutator<>, public ir::stmt::StmtMutator<> {
  explicit LoadReplacer(const std::unordered_map<std::string, ir::Var>& var_map)
      : var_map_(var_map) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void operator()(StmtRef stmt) { ir::stmt::StmtMutator<>::VisitStmt(stmt); }

 private:
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    std::string key = utils::GetStreamCnt(*expr);
    if (var_map_.count(key) > 0) {
      *expr = Expr(var_map_.at(key));
    }
  }

  void VisitStmt(ir::stmt::Let stmt) override {
    if (stmt->body().defined()) {
      Expr body = stmt->body();
      ir::IRMutator<>::Visit(&body, &body);
      stmt->set_body(body);
    }
  }

  void VisitStmt(ir::stmt::Store stmt) override {
    auto* tensor = stmt->tensor().as_tensor();

    std::vector<Expr> new_indices = stmt->indices();
    for (Expr& index : new_indices) {
      ir::IRMutator<>::Visit(&index, &index);
    }
    stmt->set_indices(new_indices);

    Expr tensor_expr = stmt->tensor();
    ir::IRMutator<>::Visit(&tensor_expr, &tensor_expr);
    stmt->set_tensor(tensor_expr);

    Expr value = stmt->value();
    ir::IRMutator<>::Visit(&value, &value);
    stmt->set_value(value);
  }

  void VisitStmt(ir::stmt::For stmt) override {
    Expr min = stmt->min();
    ir::IRMutator<>::Visit(&min, &min);
    Expr extent = stmt->extent();
    ir::IRMutator<>::Visit(&extent, &extent);
    VisitBlock(stmt->body());
    ir::Expr loop_var = stmt->loop_var();
    ir::IRMutator<>::Visit(&loop_var, &loop_var);
    stmt->set_loop_var(loop_var);
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    Expr condition = stmt->condition();
    ir::IRMutator<>::Visit(&condition, &condition);
    ir::stmt::BlockRef true_case = stmt->true_case();
    VisitBlock(true_case);
    stmt->set_true_case(true_case);
    if (stmt->false_case().defined()) {
      ir::stmt::BlockRef false_case = stmt->false_case();
      VisitBlock(false_case);
      stmt->set_false_case(false_case);
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    std::vector<Var> vars = stmt->iter_vars();
    for (ir::Var& var : vars) {
      if (var->lower_bound.defined()) {
        ir::IRMutator<>::Visit(&var->lower_bound, &var->lower_bound);
      }
      if (var->upper_bound.defined()) {
        ir::IRMutator<>::Visit(&var->upper_bound, &var->upper_bound);
      }
    }
    std::vector<Expr> new_read_buffers = stmt->read_buffers();
    for (Expr& read_buffer : new_read_buffers) {
      ir::IRMutator<>::Visit(&read_buffer, &read_buffer);
    }
    stmt->set_read_buffers(new_read_buffers);

    std::vector<Expr> new_write_buffers = stmt->write_buffers();
    for (Expr& write_buffer : new_write_buffers) {
      ir::IRMutator<>::Visit(&write_buffer, &write_buffer);
    }
    stmt->set_write_buffers(new_write_buffers);
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::Alloc stmt) override { return; }

  void VisitStmt(ir::stmt::Free stmt) override { return; }

  void VisitStmt(ir::stmt::Evaluate) override { return; }

  const std::unordered_map<std::string, ir::Var>& var_map_;
};

struct RearrangeLoadInstructionMutator : public ir::stmt::StmtMutator<> {
  void operator()(BlockRef block) { VisitBlock(block); }

 private:
  // A block is a leaf block if it is inside at least one loop, and all of its
  // stmts are schedule blocks.
  bool IsLeafBlock(BlockRef block) {
    if (parent_loops_.empty()) return false;
    for (StmtRef stmt : block->stmts()) {
      if (!stmt.isa<Schedule>()) return false;
      Schedule schedule_stmt = stmt.as<Schedule>();
      if (schedule_stmt->name().substr(0, 4) == "root") return false;
    }
    return true;
  }

  // Local buffer initialization is like:
  //    var_1_local[0] = var_1[blockIdx.x],
  // where the lhs is a local buffer and the rhs is a single load.
  bool IsLocalBufferInit(Store store_stmt) {
    const ir::Buffer& store_buffer = store_stmt->tensor().as_tensor()->buffer;
    return store_buffer->memory_type == ir::MemoryType::GPULocal &&
           store_stmt->value().As<ir::Load>();
  }

  // If there is any vectorized loop in the block skip pass
  bool HasVectorizedLoop(BlockRef block) {
    bool is_vectorized = false;
    auto pre_callback = [&is_vectorized](const StmtRef& stmt) {
      if (stmt.isa<For>() && stmt.as<For>()->is_vectorized())
        is_vectorized = true;
    };

    ir::stmt::Visit(block, pre_callback, [](const StmtRef&) {});
    return is_vectorized;
  }

  void DoRearrangeLoadInstruction(BlockRef block) {
    auto GetStoreOfScheduleStmt = [](Schedule schedule_stmt) -> Store {
      bool found = false;
      Store ret;
      for (StmtRef stmt : schedule_stmt->body()->stmts()) {
        if (stmt.isa<Store>()) {
          PADDLE_ENFORCE(found == false,
                         ::common::errors::InvalidArgument(
                             "One schedule statement should only have one "
                             "store statement."));
          found = true;
          ret = stmt.as<Store>();
        }
      }
      PADDLE_ENFORCE(found == true,
                     ::common::errors::InvalidArgument(
                         "One schedule statement should have one store "
                         "statement, but not found."));
      return ret;
    };

    // Step 1. Collect loads in each schedule block under this block.
    // Requirements:
    // 1) The schedule block cannot contain IfThenElse, or we will violate the
    //    control dependency. Schedule blocks that have IfThenElse usually
    //    don't benefit from rearranging loads, so it's ok to skip them.
    // 2) The schedule block is not local buffer initialization, because when
    //    initializing the local buffer with a load, we have already
    //    rearranged that load.
    // 3) There are more constrains on the loads to collect, see LoadCollector
    //    for details.
    LoadCollector collector(locally_defined_buffers_);
    for (StmtRef stmt : block->stmts()) {
      if (ContainsStmtInStmt<IfThenElse>(stmt)) continue;
      if (!stmt.isa<Schedule>()) continue;
      Schedule schedule_stmt = stmt.as<Schedule>();
      Store store_stmt = GetStoreOfScheduleStmt(schedule_stmt);
      if (IsLocalBufferInit(store_stmt)) continue;
      collector(const_cast<ir::Expr*>(&store_stmt->value()));
    }

    // Step 2. Sort the loads by their buffer sizes from large to small, and
    //    only keep the first `MaxRearrangeLoadNum` loads.
    // Performance concerns:
    // 1) Larger buffers need more time to access, so we should issue their
    //    corresponding loads earlier.
    // 2) Rearranged loads will consume registers, so we should set a limit
    //    to prevent register overflow.
    std::vector<std::string> load_list =
        SortLoadsByBufferSizes(collector.load_map_, collector.load_list_);
    if (load_list.size() > MaxRearrangeLoadNum) {
      load_list.resize(MaxRearrangeLoadNum);
    }

    // Step 3. Create loads with Let at the beginning of the block.
    std::vector<StmtRef> new_stmts;
    std::unordered_map<std::string, ir::Var> var_map;
    for (std::string& key : load_list) {
      const ir::Expr* load_expr = collector.load_map_[key];
      const auto tensor = load_expr->As<ir::Load>()->tensor.as_tensor();
      ir::Var local_var = ir::Var(common::UniqName(tensor->name + "_local"),
                                  tensor->buffer->dtype);
      Let let_stmt = Let(local_var, *load_expr);
      new_stmts.push_back(let_stmt);
      var_map[key] = local_var;
    }

    // Step 4. Replace loads in schedule blocks with the above Let vars.
    LoadReplacer replacer(var_map);
    for (StmtRef stmt : block->stmts()) {
      if (stmt.isa<Schedule>()) {
        replacer(stmt);
      }
      new_stmts.push_back(stmt);
    }
    block->set_stmts(new_stmts);
  }

  void VisitBlock(BlockRef block) override {
    if (HasVectorizedLoop(block)) return;
    ir::stmt::StmtMutator<>::VisitBlock(block);
    if (IsLeafBlock(block)) {
      DoRearrangeLoadInstruction(block);
    }
  }

  void VisitStmt(Schedule stmt) override {
    if (stmt->name().substr(0, 4) == "root") {
      ir::stmt::StmtMutator<>::VisitBlock(stmt->body());
      return;
    }
    for (auto& buffer_range : stmt->write_buffers()) {
      auto& write_buffer = buffer_range.As<ir::_BufferRange_>()->buffer;
      locally_defined_buffers_.insert(write_buffer.as_buffer_ref());
    }
  }

  void VisitStmt(For stmt) override {
    parent_loops_.push_back(stmt);
    VisitBlock(stmt->body());
    parent_loops_.pop_back();
  }

  void VisitStmt(IfThenElse stmt) override {
    ir::stmt::BlockRef true_case = stmt->true_case();
    VisitBlock(true_case);
    stmt->set_true_case(true_case);
    if (stmt->false_case().defined()) {
      ir::stmt::BlockRef false_case = stmt->false_case();
      VisitBlock(false_case);
      stmt->set_false_case(false_case);
    }
  }

  void VisitStmt(Let stmt) override { return; }
  void VisitStmt(Store stmt) override { return; }
  void VisitStmt(Alloc stmt) override { return; }
  void VisitStmt(Free stmt) override { return; }
  void VisitStmt(Evaluate stmt) override { return; }

 private:
  std::set<ir::Buffer> locally_defined_buffers_;
  std::vector<For> parent_loops_;
};
}  // namespace

LogicalResult cinn::optim::RearrangeLoadInstructionPass::Run(
    ir::LoweredFunc func) {
  if (FLAGS_cinn_enable_rearrange_load) {
    BlockRef body = func->body_block;
    RearrangeLoadInstructionMutator mutator;
    mutator(body);
  }
  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateRearrangeLoadInstructionPass() {
  return std::make_unique<RearrangeLoadInstructionPass>();
}
}  // namespace optim
}  // namespace cinn
