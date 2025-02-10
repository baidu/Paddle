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

#include "paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
#include "paddle/cinn/ir/group_schedule/tactic/align_iter_space_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/compute_at_reduction_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/tile_broadcast_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace ir {

void DynamicShapeGroupScheduler::Init() {
  VLOG(4) << "=============================Start group "
             "schedule==============================";
  VLOG(4) << "original group func body: \n"
          << ir_sch_->GetModule().GetExprs()[0];
  InitBuckets();
  tactics_.emplace_back(CreateAlignIterSpaceTactic());
  tactics_.emplace_back(CreateTileBroadcastTactic());
  tactics_.emplace_back(CreateTileTransposeTactic());
  tactics_.emplace_back(CreateTileFirstGeneralTactic());
  tactics_.emplace_back(CreateComputeInlineTactic());
  tactics_.emplace_back(CreateComputeAtReductionTactic());
}

void DynamicShapeGroupScheduler::InitBuckets() {
  std::unordered_set<std::string> output_names = OutputTensorNames();

  auto InitBucket = [&](BucketInfo&& bucket_info, ScheduleConfig&& config) {
    std::unique_ptr<ir::IRSchedule> ir_sch =
        std::make_unique<ir::IRSchedule>(*ir_sch_);
    std::unique_ptr<ir::ScheduleBlockGraph> schedule_block_graph =
        std::make_unique<ir::ScheduleBlockGraph>(*ir_sch);
    ir::ScheduleBlockNode* global_master =
        FindGlobalMasterNode(schedule_block_graph);

    VLOG(4) << bucket_info.ToString();
    SymbolicPredicate predicate =
        MakeBucketPredicate(bucket_info, global_master);

    ScheduleContext schedule_context{output_names,
                                     target_,
                                     IterativeSpaceInfo(),
                                     std::move(bucket_info),
                                     std::move(config)};
    BucketContext bucket_context{std::move(predicate),
                                 bucket_info.bucket_priority,
                                 std::move(ir_sch),
                                 std::move(schedule_block_graph),
                                 std::move(schedule_context)};
    bucket_contexts_.emplace_back(std::move(bucket_context));
  };

  ScheduleConfigManager& schedule_config_manager =
      ScheduleConfigManager::Instance();
  std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash> configs =
      schedule_config_manager.ExtractConfigs(target_, group_info_);
  for (std::pair<BucketInfo, ScheduleConfig>&& config : configs) {
    InitBucket(std::move(config.first), std::move(config.second));
  }
}

void DynamicShapeGroupScheduler::Schedule() {
  VLOG(4) << "bucket_context_.size() = " << bucket_contexts_.size();
  for (BucketContext& bucket_context : bucket_contexts_) {
    VLOG(4) << "===========================Apply tactics on Bucket ["
            << bucket_context.predicate << "]==========================";
    ApplyTactics(&bucket_context);
  }
}

void DynamicShapeGroupScheduler::ApplyTactics(BucketContext* bucket_context) {
  bucket_context->schedule_block_graph->Update(*(bucket_context->ir_sch));
  for (const auto& tactic : tactics_) {
    VLOG(5) << "[Start " << tactic->TacticName() << "] func body:\n"
            << bucket_context->ir_sch->GetModule().GetExprs().front();
    auto ApplyTacticFunc = [&](ir::ScheduleBlockNode* node) {
      VLOG(6) << "before applying [" << tactic->TacticName()
              << "] on ScheduleBlockNode [" << node->id() << "] func body:\n"
              << bucket_context->ir_sch->GetModule().GetExprs().front();
      tactic->Apply(bucket_context->ir_sch.get(), node->id());
      VLOG(6) << "after applying [" << tactic->TacticName()
              << "] on ScheduleBlockNode [" << node->id() << "] func body:\n"
              << bucket_context->ir_sch->GetModule().GetExprs().front();
    };
    tactic->Init(&(bucket_context->schedule_context),
                 bucket_context->ir_sch.get());
    bucket_context->schedule_block_graph->DFSTopoWalk(ApplyTacticFunc);
    bucket_context->schedule_block_graph->Update(*(bucket_context->ir_sch));
    VLOG(5) << "[End " << tactic->TacticName() << "] func body: "
            << bucket_context->ir_sch->GetModule().GetExprs().front();
  }
}

std::vector<std::pair<SymbolicPredicate, ir::Expr>>
DynamicShapeGroupScheduler::GetIRs() {
  std::vector<std::pair<SymbolicPredicate, ir::Expr>> irs;
  for (BucketContext& context : bucket_contexts_) {
    irs.emplace_back(context.predicate,
                     context.ir_sch->GetModule().GetExprs()[0]);
  }
  return irs;
}

std::vector<int> DynamicShapeGroupScheduler::GetPriorities() {
  std::vector<int> priorities;
  for (BucketContext& context : bucket_contexts_) {
    priorities.emplace_back(context.priority);
  }
  return priorities;
}

std::vector<std::pair<SymbolicPredicate, ir::Expr>>
DynamicShapeGroupScheduler::GetCX86IRs() {
  std::vector<std::pair<SymbolicPredicate, ir::Expr>> irs(1);
  irs[0].first = ir::EQ::Make(ir::Expr(1), ir::Expr(1));
  irs[1].second = ir_sch_->GetModule().GetExprs()[0];
  return irs;
}

ir::ScheduleBlockNode* DynamicShapeGroupScheduler::FindGlobalMasterNode(
    const std::unique_ptr<ir::ScheduleBlockGraph>& schedule_block_graph) {
  ir::ScheduleBlockNode* master = nullptr;
  // 1. reduce
  auto FindReduce = [&](ir::ScheduleBlockNode* node) {
    if (analyzer::IsReductionSBlock(node->Block())) {
      master = node;
    }
  };
  schedule_block_graph->NodesWalk(FindReduce);
  if (master != nullptr) {
    VLOG(6) << "Find the global master node: " << master->id();
    return master;
  }
  // 2. broadcast
  auto FindBroadcast = [&](ir::ScheduleBlockNode* node) {
    if (analyzer::IsBroadcastSBlock(node->Block())) {
      master = node;
    }
  };
  schedule_block_graph->NodesWalk(FindBroadcast);
  if (master != nullptr) {
    VLOG(6) << "Find the global master node: " << master->id();
    return master;
  }
  // 3. end point
  master = schedule_block_graph->EndPoints().back();
  VLOG(6) << "Find the global master node: " << master->id();
  return master;
}

SymbolicPredicate DynamicShapeGroupScheduler::MakeBucketPredicate(
    const BucketInfo& bucket_info, ScheduleBlockNode* node) {
  auto [sp_extent, rd_extent] = [&]() -> std::pair<ir::Expr, ir::Expr> {
    std::vector<ir::Expr> loops = node->GetLoops();
    std::set<int> reduce_axis(group_info_->reduce_axis.begin(),
                              group_info_->reduce_axis.end());

    ir::Expr sp_extent = ir::Expr(1);
    ir::Expr rd_extent = ir::Expr(1);
    for (int i = 0; i < loops.size(); ++i) {
      auto& extent = loops[i].As<ir::For>()->extent;
      if (reduce_axis.count(i) == 0) {
        sp_extent = sp_extent * extent;
      } else {
        rd_extent = rd_extent * extent;
      }
    }

    sp_extent = optim::ArithSimplify(sp_extent);
    rd_extent = optim::ArithSimplify(rd_extent);
    return {sp_extent, rd_extent};
  }();

  auto MakeDimBoundPredicate = [](const ir::Expr& extent,
                                  const BucketInfo::Dimension& dim) {
    SymbolicPredicate lower_bound_predicate =
        ir::GE::Make(extent, ir::Expr(dim.lower_bound));
    if (dim.upper_bound == BucketInfo::kMaxNumel) {
      return lower_bound_predicate;
    }
    SymbolicPredicate upper_bound_predicate =
        ir::LE::Make(extent, ir::Expr(dim.upper_bound));
    return ir::And::Make(lower_bound_predicate, upper_bound_predicate);
  };

  std::set<std::string> iter_type_set;
  SymbolicPredicate predicate = ir::Expr(true);

  for (auto& dim : bucket_info.space) {
    PADDLE_ENFORCE_EQ(
        iter_type_set.count(dim.iter_type),
        0UL,
        ::common::errors::PreconditionNotMet(
            "There can be at most one occurrence of each iter_type in "
            "BucketInfo. However, got duplicate \"%s\" type.",
            dim.iter_type));
    iter_type_set.insert(dim.iter_type);

    ir::Expr extent = (dim.iter_type == "S") ? sp_extent : rd_extent;
    SymbolicPredicate curr_predicate = MakeDimBoundPredicate(extent, dim);
    predicate = ir::And::Make(predicate, curr_predicate);
  }
  return predicate;
}

}  // namespace ir
}  // namespace cinn
