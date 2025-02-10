// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace auto_schedule {

AutoInline::AutoInline(
    const cinn::common::Target& target,
    const std::unordered_set<std::string>& no_inline_output_names)
    : AutoGenRule(target), no_inline_output_names_(no_inline_output_names) {}

bool AutoInline::CanInlineIntoConsumer(const Expr& sche_block_realize_expr,
                                       ir::IRSchedule* ir_sch) const {
  const ir::ScheduleBlockRealize* sche_block_realize =
      sche_block_realize_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sche_block =
      sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
  ir::Expr compute_body = sche_block->body;
  ir::Expr root = ir_sch->GetRootBlock(sche_block_realize_expr);

  // Check the schedule block to be inlined is not a reduce tensor.
  for (const ir::Var& iter_var : sche_block->iter_vars) {
    if (iter_var->is_reduce_axis) {
      return false;
    }
  }
  std::vector<ir::Expr> find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Store>(); });
  if (find_store.size() != 1UL) {
    return false;
  }

  ir::Expr tensor_expr = (*find_store.begin()).As<ir::Store>()->tensor;
  ir::Tensor tensor = tensor_expr.as_tensor_ref();
  if (tensor->is_reduce_tensor()) {
    return false;
  }

  // LoweredFunc output can be tensor name or tensor buffer name
  if (no_inline_output_names_.find(tensor->name) !=
          no_inline_output_names_.end() ||
      no_inline_output_names_.find(tensor->buffer->name) !=
          no_inline_output_names_.end()) {
    return false;
  }

  // the xxx_reduce_init block cannot be inlined.
  if (ir::IsReduceInitTensorName(tensor->name)) {
    return false;
  }

  // Skip external calls
  std::vector<ir::Expr> consumers =
      ir::GetConsumers(sche_block_realize_expr, root);
  for (const ir::Expr& consumer : consumers) {
    std::vector<ir::Expr> find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
        consumer.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>()
            ->body,
        [&](const ir::Expr* x) {
          return x->As<ir::Load>() &&
                 x->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                     tensor->name;
        });
    if (find_load.empty()) {
      return false;
    }
  }

  // write_buffers.size() = 1 and read_buffers is empty, means const
  // we can inline to consumer
  if (sche_block->read_buffers.empty()) {
    return true;
  }

  // Check this schedule block is the only writer of the tensor.
  find_store =
      ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
        return x->As<ir::Store>() &&
               (x->As<ir::Store>()->tensor).as_tensor_ref()->name ==
                   tensor->name;
      });
  if (find_store.size() != 1UL) {
    return false;
  }
  // Check there is no overlap between the buffers the schedule block reads and
  // writes.
  std::vector<ir::Expr> find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) {
        return x->As<ir::Load>() && x->As<ir::Load>()->tensor == tensor_expr;
      });
  if (!find_load.empty()) {
    return false;
  }

  ir::Expr store = *(find_store.begin());

  ir::ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(),
                             store);
  if (!inliner.BodyPatternAllowInline()) {
    return false;
  }

  ir::LeafBlockRemovalPlan remove_plan(
      sche_block_realize_expr, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  if (!inliner.src_stmt.defined() || !inliner.tgt_stmt.defined()) {
    return false;
  }

  VLOG(6) << "Found store Expr " << store << ", which CanInlineIntoConsumer";
  return true;
}

AutoInlineType AutoInline::AnalyzeInlineType(
    const Expr& sche_block_realize_expr, ir::IRSchedule* ir_sch) const {
  const ir::ScheduleBlockRealize* sche_block_realize =
      sche_block_realize_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sche_block =
      sche_block_realize->schedule_block.As<ir::ScheduleBlock>();

  // Inline if the block has only 1 write buffer
  if (sche_block->write_buffers.size() != 1) {
    return AutoInlineType::kCannotInline;
  }

  std::unordered_set<ir::IrNodeTy> no_inline_node_types = {
      ir::IrNodeTy::IfThenElse};
  if (ContainsNodeType(sche_block->body, no_inline_node_types)) {
    return AutoInlineType::kCannotInline;
  }

  // InlineIntoConsumer other than above situations
  if (CanInlineIntoConsumer(sche_block_realize_expr, ir_sch)) {
    return AutoInlineType::kInlineIntoConsumer;
  }

  // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
  // so we just do kInlineIntoConsumer here. Add CanInlineIntoProducer
  // once ReverseComputeInline is ready.
  return AutoInlineType::kCannotInline;
}

RuleApplyType AutoInline::Init(ir::IRSchedule* ir_schedule) {
  ir_schedule_ = ir_schedule;
  all_block_realizes_ = ir_schedule_->GetAllBlocks();
  apply_indices_and_type_.clear();
  num_applicable_ = 0;

  for (size_t i = 0; i < all_block_realizes_.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize =
        all_block_realizes_[i].As<ir::ScheduleBlockRealize>();
    AnalyzeScheduleBlockReadWriteBuffer(
        sche_block_realize->schedule_block.As<ir::ScheduleBlock>());
    AutoInlineType type =
        AnalyzeInlineType(all_block_realizes_[i], ir_schedule_);
    if (type != AutoInlineType::kCannotInline) {
      ++num_applicable_;
      apply_indices_and_type_.push_back({i, type});
    }
  }

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndPruneOtherRules
                             : RuleApplyType::kCannotApply;
}

void AutoInline::Apply(int index) {
  PADDLE_ENFORCE_EQ(
      ir_schedule_ != nullptr,
      true,
      ::common::errors::InvalidArgument("Run AutoInline::Apply without Init"));

  PADDLE_ENFORCE_EQ(
      num_applicable_ > 0 && apply_indices_and_type_.size() == num_applicable_,
      true,
      ::common::errors::InvalidArgument(
          "AutoInline::Apply pre-condition doesn't meet"));

  PADDLE_ENFORCE_EQ(
      index >= 0 && num_applicable_ > index,
      true,
      ::common::errors::InvalidArgument(
          "Invalid index for AutoInline::Apply, the index needs 0 <= index && "
          "index < NumberApplicable(), "
          "Currently index = %d, NumberApplicable() = %d",
          index,
          num_applicable_));

  int apply_index = apply_indices_and_type_[index].first;
  Apply(ir_schedule_, all_block_realizes_[apply_index]);
  return;
}

std::string AutoInline::GetRuleName() const { return "AutoInline"; }

RuleApplyType AutoInline::AnalyseApplyType(
    SearchState state, const std::string& block_name) const {
  Expr block_expr = state->ir_schedule.GetBlock(block_name);
  auto* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(
      block_realize,
      ::common::errors::InvalidArgument(
          "stmt is not a ScheduleBlockRealize: %s", block_expr));

  AnalyzeScheduleBlockReadWriteBuffer(
      block_realize->schedule_block.As<ir::ScheduleBlock>());
  AutoInlineType type = AnalyzeInlineType(block_expr, &state->ir_schedule);

  return type == AutoInlineType::kCannotInline
             ? RuleApplyType::kCannotApply
             : RuleApplyType::kApplyAndPruneOtherRules;
}

std::vector<SearchState> AutoInline::ApplyOnBlock(
    SearchState state, const std::string& block_name) {
  SearchState new_state = state.Copy();
  Expr block_expr = new_state->ir_schedule.GetBlock(block_name);
  Apply(&new_state->ir_schedule, block_expr);

  return {new_state};
}

void AutoInline::Apply(ir::IRSchedule* ir_schedule, ir::Expr& block_expr) {
  auto* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(
      block_realize,
      ::common::errors::InvalidArgument(
          "stmt is not a ScheduleBlockRealize: %s", block_expr));

  AnalyzeScheduleBlockReadWriteBuffer(
      block_realize->schedule_block.As<ir::ScheduleBlock>());
  AutoInlineType type = AnalyzeInlineType(block_expr, ir_schedule);

  if (type == AutoInlineType::kInlineIntoConsumer) {
    VLOG(6) << "Apply ComputeInline on " << block_expr;
    ir_schedule->ComputeInline(block_expr);
    VLOG(6) << "After ComputeInline: " << block_expr;

  } else if (type == AutoInlineType::kInlineIntoProducer) {
    // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
    // so we just do kInlineIntoConsumer here. Add CanInlineIntoConsumer
    // once ReverseComputeInline is ready.

    // ir_schedule->ReverseComputeInline(all_block_realizes_[apply_index]);
  }

  // Make sure re-apply the AutoInline won't be error.
  // AutoInline changes the read and write buffers of schedule blocks,
  // we need to re-analyze
  all_block_realizes_ = ir_schedule->GetAllBlocks();
  for (size_t i = 0; i < all_block_realizes_.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize =
        all_block_realizes_[i].As<ir::ScheduleBlockRealize>();
    ir::ScheduleBlock* sche_block =
        sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
    sche_block->read_buffers = {};
    sche_block->write_buffers = {};
    AnalyzeScheduleBlockReadWriteBuffer(sche_block);
  }
}

}  // namespace auto_schedule
}  // namespace cinn
