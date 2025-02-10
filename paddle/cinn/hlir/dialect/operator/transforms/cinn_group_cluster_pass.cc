// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/attribute_storage.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/operator_fusion/cluster_interface.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

using cinn::hlir::framework::pir::ScheduleInfoNode;

std::vector<::pir::Value> GetListOutsideInput(
    const std::vector<::pir::Operation*>& ops) {
  std::vector<pir::Value> outside_inputs;
  std::unordered_set<pir::Value> visited;

  for (auto op : ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      visited.insert(op->result(i));
    }
  }

  for (const auto& op : ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      if (!visited.count(op->operand_source(i))) {
        visited.insert(op->operand_source(i));
        outside_inputs.push_back(op->operand_source(i));
      }
    }
  }
  return outside_inputs;
}

std::string BuildGroupId(const ::pir::GroupOpsVec& ops_list) {
  std::string group_id;
  for (const auto& op : ops_list) {
    if (group_id != "") {
      group_id += "_";
    }
    group_id += op->name();
  }

  return group_id;
}
struct GroupClusterNode {
  // all the ops in each Node
  std::vector<::pir::Operation*> ops;
  // group kind
  cinn::hlir::framework::OpPatternKind group_kind{
      cinn::hlir::framework::kElementWise};
  // reduce_axis if kind is Reduce else empty
  std::vector<int64_t> reduce_axis;
  // if kind is reduce, loop ranges equal input dim
  // if kind id elementwise or broadcast, loop ranges equal output dim
  std::vector<int64_t> loop_ranges;
  std::vector<symbol::DimExpr> loop_rangs_expr;

  std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>
      alignment_schedule_info;

  std::vector<::pir::Value> GetOutsideInput() const {
    return GetListOutsideInput(ops);
  }

  cinn::fusion::FusionTrackerPtr tracker;
};

std::vector<::pir::Value> GenerateOutputValue(
    const std::vector<::pir::Operation*>& ops,
    const std::unordered_map<::pir::Value, size_t>& outside_need_value) {
  std::vector<::pir::Value> temp_out;
  for (const auto& op : ops) {
    if (op->isa<pir::YieldOp>()) {
      continue;
    }

    std::unordered_set<::pir::Value> inserted_val;
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (outside_need_value.count(op->result(i))) {
        if (!inserted_val.count(op->result(i))) {
          temp_out.push_back(op->result(i));
          inserted_val.insert(op->result(i));
        }
      }
    }
  }
  std::sort(temp_out.begin(),
            temp_out.end(),
            [&outside_need_value](::pir::Value a, ::pir::Value b) {
              return outside_need_value.at(a) < outside_need_value.at(b);
            });

  return temp_out;
}

cinn::dialect::GroupInfo BuildGroupInfo(
    const ::pir::GroupOpsVec& vec_new_op_list,
    const GroupClusterNode& node,
    const std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>&
        new_align_info) {
  cinn::dialect::GroupInfo group_info(vec_new_op_list);
  group_info.group_id = BuildGroupId(vec_new_op_list);
  group_info.loop_ranges = node.loop_ranges;
  group_info.loop_ranges_expr = node.loop_rangs_expr;
  group_info.reduce_axis = node.reduce_axis;
  group_info.op_pattern_kind = node.group_kind;
  group_info.alignment_schedule_info = new_align_info;

  return group_info;
}

std::vector<pir::Type> BuildOutType(
    const std::vector<::pir::Value>& output_value) {
  std::vector<pir::Type> output_types;

  for (const auto& value : output_value) {
    output_types.emplace_back(value.type());
  }

  return output_types;
}

::pir::GroupOpsVec CloneOps(
    const ::pir::GroupOpsVec& group_ops,
    const GroupClusterNode& node,
    ::pir::IrMapping* ir_mapping,
    std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>*
        align_info) {
  std::vector<::pir::Operation*> vec_new_op_list;
  ::pir::CloneOptions clone_options(false, true, false);

  auto& alignment_schedule_info = node.alignment_schedule_info;
  for (auto op : group_ops) {
    auto new_op = op->Clone(*ir_mapping, clone_options);
    // TODO(Hongqing-work): delete this after fix bug of
    // cinn_dynamic_reshape_op_pass
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    for (size_t i = 0; i < op->num_results(); ++i) {
      shape_analysis.SetShapeOrDataForValue(
          new_op->result(i),
          shape_analysis.GetShapeOrDataForValue(op->result(i)));
    }

    vec_new_op_list.push_back(new_op);

    if (alignment_schedule_info.count(op)) {
      align_info->emplace(new_op, alignment_schedule_info.at(op));
    }
  }

  return vec_new_op_list;
}

::pir::Operation* ReplaceWithFusionOp(
    pir::PatternRewriter* rewriter,
    const ::pir::GroupOpsVec& group_ops,
    const GroupClusterNode& node,
    const std::vector<::pir::Value> output_value,
    ::pir::IrMapping* ir_mapping) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>
      new_align_info;

  auto vec_new_op_list = CloneOps(group_ops, node, ir_mapping, &new_align_info);

  auto group_info = BuildGroupInfo(vec_new_op_list, node, new_align_info);
  // step 2: Replace the old op with GroupOp.

  auto output_types = BuildOutType(output_value);
  auto new_fusion_op = rewriter->Build<cinn::dialect::FusionOp>(
      output_types, group_info, node.tracker);
  pir::Block* fusion_block = new_fusion_op.block();

  for (auto op : vec_new_op_list) {
    fusion_block->insert(fusion_block->end(), op);
  }

  // step 3: Replace outputs of inner ops
  auto group_outs = new_fusion_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());

  std::vector<::pir::Value> new_output;

  for (size_t i = 0; i < output_value.size(); ++i) {
    new_output.push_back(ir_mapping->Lookup<::pir::Value>(output_value[i]));
  }

  rewriter->SetInsertionPointToBlockEnd(fusion_block);
  rewriter->Build<::pir::YieldOp>(new_output);
  rewriter->SetInsertionPointAfter(new_fusion_op);

  return new_fusion_op;
}

std::vector<GroupClusterNode> GroupSplit(cinn::dialect::GroupOp group_op) {
  std::function<cinn::fusion::PatternContent(pir::Operation*)> func =
      [](pir::Operation* op) { return cinn::fusion::PatternContent(op); };
  const auto& contents = cinn::fusion::MapVector(group_op.GetOperators(), func);
  auto cluster_result = cinn::fusion::ClusterOps(contents, {});
  std::vector<std::vector<pir::Operation*>> op_sets;
  std::vector<cinn::fusion::FusionTrackerPtr> trackers;
  std::transform(cluster_result.begin(),
                 cluster_result.end(),
                 std::back_inserter(op_sets),
                 [](const cinn::fusion::PatternNodePtr node) {
                   return cinn::fusion::GetOpsInPattern(node->stmt_pattern());
                 });
  std::transform(cluster_result.begin(),
                 cluster_result.end(),
                 std::back_inserter(trackers),
                 [](const cinn::fusion::PatternNodePtr node)
                     -> cinn::fusion::FusionTrackerPtr {
                   return cinn::fusion::GetFusionTracker(node->stmt_pattern());
                 });

  // Each stmts corresponds to each fusion op(cluster node).
  // Concat all the ops of patterns in the stmts, and make them the op list of
  // cluster node.
  VLOG(4) << "Start Creating Cluster Nodes!";
  std::vector<GroupClusterNode> output_cluster_nodes;
  for (int i = 0; i < op_sets.size(); i++) {
    auto op_set = op_sets[i];
    GroupClusterNode cluster_node;
    for (const auto* op : op_set) {
      cluster_node.ops.push_back(const_cast<pir::Operation*>(op));
      auto op_kind = cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op);
      cluster_node.group_kind =
          cluster_node.group_kind > op_kind ? cluster_node.group_kind : op_kind;
    }
    // Deep copy trackers to avoid shared tracker conflict in different node
    cluster_node.tracker = trackers[i]->Clone();
    output_cluster_nodes.push_back(cluster_node);
  }
  VLOG(4) << "Finished Creating Cluster Nodes!";
  return output_cluster_nodes;
}

std::vector<::pir::Operation*> SortByOriginalOrderAndUniq(
    cinn::dialect::GroupOp group_op,
    const std::vector<::pir::Operation*>& ops) {
  size_t index = 0;
  std::unordered_map<pir::Operation*, size_t> op2order_value;

  for (auto op : group_op.GetOperators()) {
    op2order_value[op] = index++;
  }

  std::vector<pir::Operation*> tmp_ops(ops);
  std::sort(tmp_ops.begin(),
            tmp_ops.end(),
            [&op2order_value](pir::Operation* a, pir::Operation* b) {
              return op2order_value.at(a) < op2order_value.at(b);
            });

  std::unique(tmp_ops.begin(), tmp_ops.end());

  return tmp_ops;
}

std::unordered_map<::pir::Value, size_t> BuildValueOrderByYieldOp(
    const std::vector<GroupClusterNode>& node_list,
    cinn::dialect::GroupOp group_op) {
  std::unordered_map<::pir::Value, size_t> all_output_values;
  auto yield_op = group_op.GetOperators().back();
  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    size_t id = all_output_values.size();
    all_output_values.emplace(yield_op->operand_source(i), id);
  }

  for (size_t i = 0; i < node_list.size(); ++i) {
    auto node_outside_input = node_list[i].GetOutsideInput();
    for (const auto& val : node_outside_input) {
      size_t id = all_output_values.size();
      all_output_values.emplace(val, id);
    }
  }

  return all_output_values;
}

void UpdateTracker(std::vector<pir::Operation*> uniq_ops,
                   fusion::FusionTrackerPtr tracker) {
  std::map<pir::Operation*, int> op2idx;
  for (int i = 0; i < uniq_ops.size(); ++i) {
    op2idx[uniq_ops[i]] = i;
  }
  for (const auto& t : tracker->instructions_) {
    if (t->type() == fusion::T_InitPattern) {
      auto init_instr =
          cinn::fusion::dynamic_cast_instr_with_err<fusion::InitPatternInstr>(
              t);
      init_instr->set_idx(op2idx[init_instr->op_]);
    }
  }
}

}  // namespace

class CinnGroupClusterPattern
    : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::GroupOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrMapping ir_mapping;

    auto group_outside_input = GetListOutsideInput(group_op.GetOperators());
    // insert initial input to ir mapping
    for (auto val : group_outside_input) {
      ir_mapping.Add(val, val);
    }

    auto split_res = GroupSplit(group_op);

    auto all_output_values = BuildValueOrderByYieldOp(split_res, group_op);

    for (auto& node : split_res) {
      if (node.ops.size() == 0) {
        continue;
      }
      auto output_values = GenerateOutputValue(node.ops, all_output_values);
      VLOG(4) << "cluster node output size: " << output_values.size();
      auto uniq_ops = SortByOriginalOrderAndUniq(group_op, node.ops);

      UpdateTracker(uniq_ops, node.tracker);

      auto new_group_op = ReplaceWithFusionOp(
          &rewriter, uniq_ops, node, output_values, &ir_mapping);

      // TODO(Hongqing-work): delete this after fix bug of
      // cinn_dynamic_reshape_op_pass
      auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
          group_op->GetParentProgram());
      // update ir mapping
      for (size_t i = 0; i < output_values.size(); ++i) {
        ir_mapping.Add(output_values[i], new_group_op->result(i));
        shape_analysis.SetShapeOrDataForValue(
            new_group_op->result(i),
            shape_analysis.GetShapeOrDataForValue(output_values[i]));
      }
      for (size_t i = 0; i < output_values.size(); ++i) {
        auto find_it = all_output_values.find(output_values[i]);
        if ((find_it != all_output_values.end()) &&
            (find_it->second < group_op->num_results())) {
          // id < num_results means yield input
          rewriter.ReplaceAllUsesWith(group_op.result(find_it->second),
                                      new_group_op->result(i));
        }
      }
    }

    rewriter.EraseOp(group_op);

    return true;
  }
};

class CinnGroupClusterPass : public pir::PatternRewritePass {
 public:
  CinnGroupClusterPass()
      : pir::PatternRewritePass("cinn_group_cluster_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    ps.Add<CinnGroupClusterPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<cinn::dialect::FusionOp>()) {
      return false;
    }
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateCinnGroupClusterPass() {
  return std::make_unique<CinnGroupClusterPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
