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

#include "paddle/cinn/operator_fusion/pattern_graph.h"
#include <functional>
#include "paddle/cinn/operator_fusion/graph_transformer/matcher.h"
#include "paddle/cinn/operator_fusion/graph_transformer/operation.h"
#include "paddle/cinn/operator_fusion/graph_transformer/search_algorithm.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"

namespace cinn::fusion {

std::vector<PatternNodePtr> PatternGraph::ClusterOps() {
  VLOG(4) << "[Group Cluster] Initial Condition: ";
  PrintGraphInfo();

  VLOG(4) << "[Group Cluster] Start SinkTrivialPattern";
  SinkTrivialPattern();
  VLOG(4) << "[Group Cluster] After SinkTrivialPattern: ";
  PrintGraphInfo();

  // ReducePattern -> ReduceTreePattern
  VLOG(4) << "[Group Cluster] Start ReduceLiftReduceTree";
  ReduceLiftReduceTree();
  VLOG(4) << "[Group Cluster] After ReduceLiftReduceTree: ";
  PrintGraphInfo();

  // ReduceTreePattern + ReduceTreePattern fusion
  VLOG(4) << "[Group Cluster] Start ReduceTreeGrown";
  ReduceTreeGrown();
  VLOG(4) << "[Group Cluster] After ReduceTreeGrown: ";
  PrintGraphInfo();

  // ReduceTreePattern + TrivialPattern fusion.
  VLOG(4) << "[Group Cluster] Start ReduceTree_Trivial_Fusion";
  ReduceTree_Trivial_Fusion();
  VLOG(4) << "[Group Cluster] After ReduceTree_Trivial_Fusion: ";
  PrintGraphInfo();

  // All -> ItersPermutationPattern
  VLOG(4) << "[Group Cluster] Start LiftToItersPermutationPattern";
  LiftToItersPermutationPattern();
  VLOG(4) << "[Group Cluster] After LiftToItersPermutationPattern: ";
  PrintGraphInfo();

  // ItersPermutationPattern x ItersPermutationPattern Fusion
  VLOG(4) << "[Group Cluster] Start IdentityAnchorFusion";
  LimitedAnchorFusion();
  VLOG(4) << "[Group Cluster] After IdentityAnchorFusion: ";
  PrintGraphInfo();

  // Sink single trivial op pattern
  VLOG(4) << "[Group Cluster] Start SplitRecomputePattern";
  SplitRecomputePattern();
  VLOG(4) << "[Group Cluster] After SplitRecomputePattern: ";
  PrintGraphInfo();

  // ItersPermutationPattern x ItersPermutationPattern Fusion
  VLOG(4) << "[Group Cluster] Start ItersPermutationFusion";
  ItersPermutationFusion();
  VLOG(4) << "[Group Cluster] After ItersPermutationFusion: ";
  PrintGraphInfo();

  // Horizontal fusion.
  VLOG(4) << "[Group Cluster] Start HorizontalFusion";
  HorizontalFusion();
  VLOG(4) << "[Group Cluster] After HorizontalFusion: ";
  PrintGraphInfo();

  return ReturnFusionResults();
}

std::vector<PatternNodePtr> PatternGraph::ReturnFusionResults() {
  auto sorted_nodes = SortByTopoOrder();
  for (const auto& node : sorted_nodes) {
    node->set_return();
  }
  return sorted_nodes;
}

std::vector<PatternNodePtr> PatternGraph::SortByTopoOrder() const {
  // sort all_pattern_nodes_ by topo order.
  std::vector<PatternNodePtr> res;
  std::list<PatternNodePtr> topo_queue;
  std::map<PatternNodePtr, int> degree;
  for (const auto& node : all_pattern_nodes_) {
    degree[node] = node->upstream().size();
    if (degree[node] == 0) {
      topo_queue.push_back(node);
    }
  }
  while (!topo_queue.empty()) {
    PatternNodePtr node = topo_queue.front();
    topo_queue.pop_front();
    res.push_back(node);
    for (const auto& downstream_op : node->downstream()) {
      degree[downstream_op] = degree[downstream_op] - 1;
      if (degree[downstream_op] == 0) {
        topo_queue.push_back(downstream_op);
      }
    }
  }
  return res;
}

std::vector<PatternNodePtr> PatternGraph::SortByReverseTopoOrder() const {
  // sort all_pattern_nodes_ by reverse topo order.
  std::vector<PatternNodePtr> res;
  std::list<PatternNodePtr> reverse_topo_queue;
  std::map<PatternNodePtr, int> degree;

  for (const auto& node : all_pattern_nodes_) {
    degree[node] = node->downstream().size();
    if (degree[node] == 0) {
      reverse_topo_queue.push_back(node);
    }
  }

  while (!reverse_topo_queue.empty()) {
    PatternNodePtr node = reverse_topo_queue.front();
    reverse_topo_queue.pop_front();
    res.push_back(node);
    for (const auto& upstream : node->upstream()) {
      degree[upstream]--;
      if (degree[upstream] == 0) {
        reverse_topo_queue.push_back(upstream);
      }
    }
  }
  return res;
}

void PatternGraph::SinkTrivialPattern() {
  GraphTransformer<NodePattern,
                   And<StmtPatternGraphMatcher<TrivialPattern>,
                       OnlyOneDownstreamMatcher,
                       Not<ReshapeConnectionMatcher>>,
                   MergeTrivialPatternOperation>(this);

  // TODO(huangjiyi): remove sink multi downstream transpose after
  // supporting transpose plus reduce anchor fusion
  GraphTransformer<
      NodePattern,
      And<StmtPatternGraphMatcher<TrivialPattern>, TransposeOpMatcher>,
      MergeTrivialPatternOperation>(this);

  // Sink Trivial pattern whose downstream containing related iters.
  // TODO(huangjiyi): Only sink to the related iters pattern.
  GraphTransformer<NodePattern,
                   And<StmtPatternGraphMatcher<TrivialPattern>,
                       DownstreamHasItersRelationMatcher,
                       Not<ReshapeConnectionMatcher>>,
                   MergeTrivialPatternOperation>(this);

  // Sink non-leaf reshape pattern.
  GraphTransformer<
      NodePattern,
      And<StmtPatternGraphMatcher<TrivialPattern>,
          Or<OnlyOneDownstreamMatcher, DownstreamHasItersRelationMatcher>,
          Not<LeafReshapeConnectionMatcher>>,
      MergeTrivialPatternOperation>(this);

  // Align leaf reshape pattern to the input shape
  GraphTransformer<NodePattern,
                   And<StmtPatternGraphMatcher<TrivialPattern>,
                       ReshapeOpMatcher,
                       LeafReshapeConnectionMatcher>,
                   ReshapeAlignInputOperation>(this);
}

void PatternGraph::ReduceLiftReduceTree() {
  GraphTransformer<
      NodePattern,
      And<DownstreamSmallerThan<2>, StmtPatternGraphMatcher<ReducePattern>>,
      LiftReduceToReduceTreeOperation>(this);
}

void PatternGraph::HorizontalFusion() {
  GraphTransformer<NodePattern,
                   Or<StmtPatternGraphMatcher<TrivialPattern>,
                      StmtPatternGraphMatcher<ReduceTreePlusTrivialPattern>,
                      StmtPatternGraphMatcher<ReducePattern>,
                      StmtPatternGraphMatcher<ReduceTreePattern>,
                      StmtPatternGraphMatcher<ItersPermutationPattern>>,
                   LiftToHorizontalFusionPatternOperation>(this);

  GraphTransformer<NodePairPattern,
                   And<HorizontalFusionConstrain,
                       InputOutputMaximumConstrain,
                       HorizontalCheckMiddleOutputVar>,  // Avoid two many
                                                         // inputs and
                                                         // outputs.
                   HorizontalFusionOperation>(this);
}

void PatternGraph::ReduceTreeGrown() {
  GraphTransformer<NodePattern,
                   CanFuseReduceTreeMatcher,
                   MergeReduceTreeOperation>(this);
}

void PatternGraph::ReduceTree_Trivial_Fusion() {
  GraphTransformer<NodePattern,
                   CanFuseReduceTreeAndTrivialMatcher,
                   MergeReduceTreeAndTrivialOperation>(this);
}

void PatternGraph::LiftToItersPermutationPattern() {
  GraphTransformer<NodePattern,
                   Or<StmtPatternGraphMatcher<TrivialPattern>,
                      StmtPatternGraphMatcher<ReduceTreePlusTrivialPattern>,
                      StmtPatternGraphMatcher<ReducePattern>,
                      StmtPatternGraphMatcher<ReduceTreePattern>>,
                   LiftToItersPermutationPatternOperation>(this);
}

void PatternGraph::LimitedAnchorFusion() {
  iters_fusion_policy()
      ->DisableStrategy(ItersTransformType::ReuseIters)
      ->DisableStrategy(ItersTransformType::AppendIters);

  GraphTransformer<
      ReverseTopoNodePairPattern,
      And<CanFuseItersPermutationMatcher, InputOutputMaximumConstrain>,
      FuseItersPermutatioOperation>(this);
}

void PatternGraph::ItersPermutationFusion() {
  iters_fusion_policy()->EnableAllStrategies();

  GraphTransformer<
      ReverseTopoNodePairPattern,
      And<CanFuseItersPermutationMatcher, InputOutputMaximumConstrain>,
      FuseItersPermutatioOperation>(this);
}

void PatternGraph::SplitRecomputePattern() {
  GraphTransformer<NodePattern, RecomputeNodeMatcher, SplitRecomputeOperation>(
      this);
  GraphTransformer<NodePattern,
                   StmtPatternGraphMatcher<TrivialPattern>,
                   LiftToItersPermutationPatternOperation>(this);
}

PatternGraph::PatternGraph(const std::vector<PatternContent>& contents,
                           const std::vector<pir::Value>& outputs,
                           const PolicyManager policy_manager)
    : policy_manager_(policy_manager), outputs_(outputs) {
  std::unordered_map<pir::Operation*, PatternNodePtr> op_to_node_map;

  VLOG(4) << "len(outputs) = " << outputs_.size();
  for (const auto& v : outputs) {
    VLOG(4) << "output is" << OpsDebugStr({v.defining_op()});
  }

  for (const auto& content : contents) {
    const auto& fusion_iters =
        iters_fusion_policy()->iters_manager()->GetItersSignature(content.op);
    PatternNodePtr node = std::make_shared<PatternNode>(content, fusion_iters);
    op_to_node_map[content.op] = node;
    all_pattern_nodes_.emplace(node);
  }

  for (const auto& content : contents) {
    PatternNodePtr cur_node = op_to_node_map[content.op];

    // add upstream nodes
    for (int i = 0; i < content.op->num_operands(); ++i) {
      ::pir::Operation* input_op = content.op->operand_source(i).defining_op();
      if (op_to_node_map.find(input_op) != op_to_node_map.end()) {
        PatternNodePtr upstream_node = op_to_node_map[input_op];
        cur_node->AddNodeToUpstream(upstream_node);
      }
    }

    // add downstream nodes
    for (int i = 0; i < content.op->num_results(); ++i) {
      pir::Value related_value = content.op->result(i);
      for (auto consumer_it = related_value.use_begin();
           consumer_it != related_value.use_end();
           ++consumer_it) {
        ::pir::Operation* output_op = consumer_it->owner();
        if (op_to_node_map.find(output_op) != op_to_node_map.end()) {
          PatternNodePtr downstream_node = op_to_node_map[output_op];
          cur_node->AddNodeToDownstream(downstream_node);
        }
      }
    }

    // unique all upstream / downstream node.
    // c = a + a ; then add will have 2 same upstream.
    cur_node->UniqueUpstream();
    cur_node->UniqueDownstream();
  }

  VLOG(4) << "PatternGraph Created, pattern node size: "
          << all_pattern_nodes_.size();
}

void PatternGraph::RemoveNode(const PatternNodePtr& node) {
  VLOG(4) << "Start Remove: " << node->id() << "(" << node << ")";
  for (auto it = all_pattern_nodes_.begin(); it != all_pattern_nodes_.end();
       ++it) {
    // Here we use traversal instead of count() or find() builtin function
    // because all_pattern_nodes_ is sorted by node id when initialization
    // but node id may be changed in copy instruction that may destroy the
    // order of set.
    if ((*it)->id() == node->id()) {
      VLOG(4) << "Removed " << (*it)->id();
      all_pattern_nodes_.erase(it);
      break;
    }
  }
  for (const PatternNodePtr& upstream : node->upstream()) {
    upstream->RemoveNodeFromDownstream(node);
  }
  for (const PatternNodePtr& downstream : node->downstream()) {
    downstream->RemoveNodeFromUpstream(node);
  }
}

void PatternGraph::AppendNode(const PatternNodePtr& node) {
  all_pattern_nodes_.emplace(node);
}

void PatternGraph::PrintGraphInfo() const {
  VLOG(4) << "========= GraphInfo ===========";
  for (const auto& v : all_pattern_nodes_) {
    std::stringstream ss;
    ss << "\n##############################";
    ss << "\n" << v->DebugStr();
    ss << "\n    IsOutput: " << IsOutputNodeMatcher()(*this, v);
    VLOG(4) << ss.str();
  }
  VLOG(4) << "===============================";
}

PatternNodePtr PatternGraph::MergeNode(const PatternNodePtr& upstream,
                                       const PatternNodePtr& downstream,
                                       MergePatternFn merge_pattern_fn) {
  PatternNodePtr merged_node =
      std::make_shared<PatternNode>(upstream, downstream, merge_pattern_fn);

  // Update upstream and downstream nodes.
  for (const auto& upstream_node : merged_node->upstream()) {
    upstream_node->AddNodeToDownstream(merged_node);
    upstream_node->RemoveNodeFromDownstream(upstream);
    upstream_node->RemoveNodeFromDownstream(downstream);
  }
  for (const auto& downstream_node : merged_node->downstream()) {
    downstream_node->AddNodeToUpstream(merged_node);
    downstream_node->RemoveNodeFromDownstream(upstream);
    downstream_node->RemoveNodeFromDownstream(downstream);
  }

  const auto vec_unique = [](const std::vector<PatternNodePtr>& vec) {
    auto set = std::unordered_set(vec.begin(), vec.end());
    return set.size() == vec.size();
  };

  PADDLE_ENFORCE_EQ(
      vec_unique(merged_node->upstream()),
      true,
      ::common::errors::PreconditionNotMet(
          "The upstream nodes of the merged node are not unique."));
  PADDLE_ENFORCE_EQ(
      vec_unique(merged_node->downstream()),
      true,
      ::common::errors::PreconditionNotMet(
          "The downstream nodes of the merged node are not unique."));

  // deal with the graph storage.
  AppendNode(merged_node);
  return merged_node;
}

}  // namespace cinn::fusion
