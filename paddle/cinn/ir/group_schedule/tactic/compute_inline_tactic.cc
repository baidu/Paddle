// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"

#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

namespace cinn {
namespace ir {

/**
 * The types of the AutoInline
 */
enum class AutoInlineType : int {
  // The block cannot be inlined
  kCannotInline = 0,
  // Inline this block into the consumer
  kInlineIntoConsumer,
  // Inline this block into the producer
  kInlineIntoProducer,
};

class ComputeInlineTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "ComputeInlineTactic"; }

 private:
  AutoInlineType AnalyzeInlineType(const Expr& sche_block_realize_expr,
                                   ir::IRSchedule* ir_sch) const;
  bool CanInlineIntoConsumer(const Expr& sche_block_realize_expr,
                             ir::IRSchedule* ir_sch) const;

  std::unordered_set<std::string> output_names_;
  cinn::common::Target target_;
};

void ComputeInlineTactic::Init(ScheduleContext* context) {
  output_names_ = context->output_names;
  target_ = context->target;
}

bool ComputeInlineTactic::CanInlineIntoConsumer(
    const Expr& sche_block_realize_expr, ir::IRSchedule* ir_sch) const {
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
  if (output_names_.find(tensor->name) != output_names_.end() ||
      output_names_.find(tensor->buffer->name) != output_names_.end()) {
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

namespace {
bool ContainsNodeType(ir::Expr expr,
                      const std::unordered_set<ir::IrNodeTy>& node_types) {
  std::vector<ir::Expr> collection =
      ir::ir_utils::CollectIRNodesWithoutTensor(expr, [&](const Expr* x) {
        return node_types.find(x->node_type()) != node_types.end();
      });
  return !collection.empty();
}
}  // namespace

AutoInlineType ComputeInlineTactic::AnalyzeInlineType(
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

void ComputeInlineTactic::Apply(ir::IRSchedule* sch,
                                const std::string& block_id) {
  // TODO(LiuYang): Compute of ops will be rewritten so that we
  // don't use it in dynamic group_schedule rules temporarily.
  // if (IsProhibitScheduleExternCallBlock(node->Block())) {
  //    return;
  // }
  VLOG(6) << "try ComputeInline on: " << block_id
          << ", before ComputeInline, func body: "
          << sch->GetModule().GetExprs().front();
  ir::Expr schedule_block = sch->GetBlock(block_id);

  auto* block_realize = schedule_block.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(
      block_realize,
      ::common::errors::InvalidArgument(
          "stmt is not a ScheduleBlockRealize: %s", schedule_block));

  analyzer::AnalyzeScheduleBlockReadWriteBuffer(
      block_realize->schedule_block.As<ir::ScheduleBlock>());
  AutoInlineType type = AnalyzeInlineType(schedule_block, sch);

  if (type == AutoInlineType::kInlineIntoConsumer) {
    VLOG(6) << "Apply ComputeInline on " << schedule_block;
    sch->ComputeInline(schedule_block);
    VLOG(6) << "After ComputeInline: " << schedule_block;

  } else if (type == AutoInlineType::kInlineIntoProducer) {
    // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
    // so we just do kInlineIntoConsumer here. Add CanInlineIntoConsumer
    // once ReverseComputeInline is ready.
  }

  // Make sure re-apply the AutoInline won't be error.
  // AutoInline changes the read and write buffers of schedule blocks,
  // we need to re-analyze
  auto all_block_realizes = sch->GetAllBlocks();
  for (size_t i = 0; i < all_block_realizes.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize =
        all_block_realizes[i].As<ir::ScheduleBlockRealize>();
    ir::ScheduleBlock* sche_block =
        sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
    sche_block->read_buffers = {};
    sche_block->write_buffers = {};
    analyzer::AnalyzeScheduleBlockReadWriteBuffer(sche_block);
  }
  VLOG(6) << "try ComputeInline on: " << block_id
          << ", after ComputeInline, func body: "
          << sch->GetModule().GetExprs().front();
}

std::unique_ptr<ScheduleTactic> CreateComputeInlineTactic() {
  return std::make_unique<ComputeInlineTactic>();
}

}  // namespace ir
}  // namespace cinn
