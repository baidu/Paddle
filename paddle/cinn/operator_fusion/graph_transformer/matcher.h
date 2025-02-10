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

#pragma once
#include "paddle/cinn/operator_fusion/pattern_graph.h"

namespace cinn::fusion {
// Matcher

struct AlwaysTrue {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return true;
  }
};

struct NonSinkNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !node->downstream().empty();
  }
};

struct IsOutputNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    bool res = IsAnyFirstInSecond(node->sink_op()->results(), graph.outputs());
    return res;
  }
};

template <int N>
struct DownstreamSmallerThan {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream().size() < N;
  }
};

template <int N>
struct DownstreamGreaterThan {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream().size() > N;
  }
};

struct OnlyOneDownstreamMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return node->downstream().size() == 1;
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return GetPatternType(node->stmt_pattern()) == StmtPattern::type();
  }
};

struct CanFuseRxTMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return (std::holds_alternative<ReduceTreePattern>(node->stmt_pattern()) &&
            !node->downstream().empty() &&
            std::holds_alternative<TrivialPattern>(
                node->downstream().at(0)->stmt_pattern()));
  }
};

struct CanFuseReduceTreeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<ReduceTreePattern>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(node, node->downstream().at(0)) &&
           graph.policy_manager()
               .template GetPolicy<RelativeJudgePolicy>()
               ->CanFuse(node, node->downstream().at(0));
  }
};

struct CanFuseReduceTreeAndTrivialMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<TrivialPattern>(
               node->downstream().at(0)->stmt_pattern()) &&
           node->downstream().at(0)->downstream().size() == 0 &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(node, node->downstream().at(0)) &&
           graph.policy_manager()
               .template GetPolicy<RelativeJudgePolicy>()
               ->CanFuse(node, node->downstream().at(0));
  }
};

struct CanFuseTrivialAndReduce {
  bool operator()(PatternGraph graph,  // NOLINT
                  const PatternNodePtr& upstream,
                  const PatternNodePtr& downstream) {
    PatternNodePtr reduce_node;
    if (upstream->fusion_iters().reduce_iter_nums &&
        !downstream->fusion_iters().reduce_iter_nums) {
      reduce_node = upstream;
    } else if (!upstream->fusion_iters().reduce_iter_nums &&
               downstream->fusion_iters().reduce_iter_nums) {
      reduce_node = downstream;
    } else {
      return true;
    }
    auto reduce_dims_product =
        graph.iters_fusion_policy()->iters_manager()->GetReduceDimsProduct(
            reduce_node->fusion_iters());
    if (reduce_dims_product.isa<std::int64_t>()) {
      return reduce_dims_product.dyn_cast<std::int64_t>() <= 1024 * 10;
    } else {
      return true;
    }
  }
};

struct DownstreamHasItersRelationMatcher {
  bool operator()(PatternGraph graph, const PatternNodePtr& node) {  // NOLINT
    return std::any_of(
        node->downstream().begin(),
        node->downstream().end(),
        [&graph, &node](const PatternNodePtr& downstream) {
          return !graph.iters_fusion_policy()->CheckItersRelation(node,
                                                                  downstream);
        });
  }
};

struct CanFuseItersPermutationMatcher {
  bool operator()(PatternGraph graph,  // NOLINT
                  const PatternNodePtr& upstream,
                  const PatternNodePtr& downstream) {
    return StmtPatternGraphMatcher<ItersPermutationPattern>()(graph,
                                                              upstream) &&
           StmtPatternGraphMatcher<ItersPermutationPattern>()(graph,
                                                              downstream) &&
           graph.policy_manager()
               .template GetPolicy<GeneralTopoPolicy>()
               ->CanFuse(upstream, downstream) &&
           graph.iters_fusion_policy()->CheckItersRelation(upstream,
                                                           downstream) &&
           (graph.iters_fusion_policy()->CanFuseSource2Target(downstream,
                                                              upstream) ||
            graph.iters_fusion_policy()->CanFuseSource2Target(upstream,
                                                              downstream));
  }
};

struct RecomputeNodeMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    const auto can_recompute_fn = [](const PatternNodePtr& node) -> bool {
      // Current Algorithm:
      // An node can be recomputed if:
      // 1. It didn't go through any pattern merging during prior fusions,
      // which means it only has one output value.
      // 2. It only contains trivial ops.
      if (node->fusion_iters().output_values.size() > 1) {
        return false;
      }
      bool has_combine_fusion =
          std::any_of(node->fusion_tracker()->instructions_.begin(),
                      node->fusion_tracker()->instructions_.end(),
                      [](const FusionInstrPtr& instr) {
                        return instr->type() == T_Combine;
                      });
      if (has_combine_fusion) {
        return false;
      }
      for (const auto& op : GetOpsInPattern(node->stmt_pattern())) {
        const auto& op_kind = GetOpPatternKind(op);
        if (op_kind >= hlir::framework::kReduction) {
          return false;
        }
      }
      return true;
    };

    return StmtPatternGraphMatcher<ItersPermutationPattern>()(graph, node) &&
           node->downstream().size() >= 1 && can_recompute_fn(node);
  }
};

struct TransposeOpMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return (node->sink_op()->name() == "pd_op.transpose");
  }
};

struct ReshapeOpMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    auto has_dynamic_shape = [](const PatternNodePtr& node) {
      const auto in_value = node->sink_op()->operand_source(0);
      const auto out_value = node->sink_op()->result(0);
      const auto in_shape = GetDimExprsFromValue(in_value);
      const auto out_shape = GetDimExprsFromValue(out_value);
      return GetShapeProduct(in_shape, 0, in_shape.size())
                 .isa<std::int64_t>() &&
             GetShapeProduct(out_shape, 0, out_shape.size())
                 .isa<std::int64_t>();
    };
    return node->ops().size() == 1 &&
           node->sink_op()->name() == "cinn_op.reshape" &&
           has_dynamic_shape(node);
  }
};

struct ReshapeConnectionMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    bool upstream_match = node->downstream().size() == 1 &&
                          ReshapeOpMatcher()(graph, node->downstream()[0]) &&
                          node->downstream()[0]->downstream().size() == 1;
    bool downstream_match =
        ReshapeOpMatcher()(graph, node) && node->downstream().size() == 1;
    return upstream_match || downstream_match;
  }
};

struct LeafReshapeConnectionMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    const auto match_upstream = [&graph](const PatternNodePtr& upstream) {
      return StmtPatternGraphMatcher<TrivialPattern>()(graph, upstream) &&
             upstream->downstream().size() == 1 &&
             !upstream->upstream().empty() &&
             std::any_of(upstream->upstream().begin(),
                         upstream->upstream().end(),
                         [&graph](const PatternNodePtr& node) {
                           return DownstreamGreaterThan<1>()(graph, node);
                         });
    };
    const auto match_downstream = [&graph](const PatternNodePtr& downstream) {
      return ReshapeOpMatcher()(graph, downstream) &&
             downstream->downstream().size() == 1 &&
             downstream->downstream()[0]->downstream().empty() &&
             downstream->fusion_iters().loop_iters ==
                 downstream->downstream()[0]->fusion_iters().loop_iters;
    };
    bool upstream_match = match_upstream(node) &&
                          node->downstream().size() == 1 &&
                          match_downstream(node->downstream()[0]);
    bool downstream_match = match_downstream(node) &&
                            node->upstream().size() == 1 &&
                            match_upstream(node->upstream()[0]);
    return upstream_match || downstream_match;
  }
};

struct NotAllElementWiseDownstreamMatcher {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    size_t count = 0;
    for (const auto& downstream : node->downstream()) {
      if (StmtPatternGraphMatcher<TrivialPattern>()(graph, downstream)) {
        auto ops = std::get<TrivialPattern>(downstream->stmt_pattern()).ops();
        bool is_elementwise =
            std::all_of(ops.begin(), ops.end(), [](pir::Operation* op) {
              return GetOpPatternKind(op) == hlir::framework::kElementWise;
            });
        count += is_elementwise;
      }
    }
    return (count < node->downstream().size());
  }
};

template <typename... Args>
struct And {};

template <typename A>
struct And<A> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node);
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    return A()(graph, lhs, rhs);
  }
};

template <typename A, typename... Args>
struct And<A, Args...> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) && And<Args...>()(graph, node);
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    return A()(graph, lhs, rhs) && And<Args...>()(graph, lhs, rhs);
  }
};

template <typename... Args>
struct Or {};

template <typename A>
struct Or<A> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node);
  }
};

template <typename A, typename... Args>
struct Or<A, Args...> {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return A()(graph, node) || Or<Args...>()(graph, node);
  }
};

template <typename A>
struct Not {
  bool operator()(const PatternGraph& graph, const PatternNodePtr& node) {
    return !A()(graph, node);
  }
};

struct HorizontalFusionConstrain {
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern>()(graph, lhs)) {
      return false;
    }
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern>()(graph, rhs)) {
      return false;
    }
    const auto& lhs_pattern =
        std::get<HorizontalFusionPattern>(lhs->stmt_pattern());
    const auto& rhs_pattern =
        std::get<HorizontalFusionPattern>(rhs->stmt_pattern());

    return graph.policy_manager().GetPolicy<GeneralTopoPolicy>()->CanFuse(
               lhs, rhs) &&
           IsLoopFrameworkEqual(lhs_pattern, rhs_pattern);
  }
};

/*
 * We must limit the output + input + shape_info number and make sure
 * the number is smaller than 512.
 */
struct InputOutputMaximumConstrain {
  const int MAX_INPUT_OUTPUT_NUMBER = 480;  // cuda only support 512
  std::vector<pir::Value> GetInputValuesExceptMiddle(
      const std::vector<pir::Operation*>& ops) {
    return VectorDiff(GetInputsValue(ops), GetOutputsValue(ops));
  }
  std::vector<pir::Value> GetOutputValuesExceptMiddle(
      const std::vector<pir::Operation*>& ops) {
    return VectorDiff(GetOutputsValue(ops), GetInputsValue(ops));
  }
  std::vector<pir::Operation*> GetAllOps(const PatternNodePtr& lhs,
                                         const PatternNodePtr& rhs) {
    return UniqueVectorBySet(
        ConcatVector(GetOpsInPattern(lhs->stmt_pattern()),
                     GetOpsInPattern(rhs->stmt_pattern())));
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    const auto& all_ops = GetAllOps(lhs, rhs);
    int input_number = GetInputValuesExceptMiddle(all_ops).size();
    int output_number = GetOutputValuesExceptMiddle(all_ops).size();
    return input_number + output_number < MAX_INPUT_OUTPUT_NUMBER;
  }
};

struct HorizontalCheckMiddleOutputVar {
  bool DontHaveMiddleVariable(const PatternGraph& graph,
                              const PatternNodePtr& lhs,
                              const PatternNodePtr& rhs) {
    for (const auto& i : lhs->downstream()) {
      if (i == rhs) return false;
    }
    for (const auto& i : lhs->upstream()) {
      if (i == rhs) return false;
    }
    return true;
  }

  std::vector<ValueDim> SqueezedValueDim(const LoopValueDims& vdims) {
    return FilterVector(vdims, [](const ValueDim& v) {
      return !v.empty() && GetDimExprsFromValue(v.v_)[v.idx_] != 1;
    });
  }

  bool IdenticalDep(const PatternGraph& graph,
                    const LoopValueDims& lhs_dims,
                    const LoopValueDims& rhs_dims) {
    auto sp = graph.policy_manager().template GetPolicy<RelativeJudgePolicy>();
    auto get_axes_from_valuedim = [&](const ValueDim& vdim) {
      return (sp->GetAxesInfoManager()).GetAxes(vdim.v_).axis_names[vdim.idx_];
    };
    VLOG(4) << "origin lhs_dims.size() = " << lhs_dims.size();
    VLOG(4) << "origin rhs_dims.size() = " << rhs_dims.size();
    std::vector<ValueDim> lhs_squeeze_value_dim = SqueezedValueDim(lhs_dims);
    std::vector<ValueDim> rhs_squeeze_value_dim = SqueezedValueDim(rhs_dims);

    if (VLOG_IS_ON(4)) {
      VLOG(4) << "lhs_squeeze_value_dim is : ";
      for (int i = 0; i < lhs_squeeze_value_dim.size(); ++i) {
        VLOG(4) << "    " << i << " = " << lhs_squeeze_value_dim[i].DebugStr();
        VLOG(4) << "    "
                << "shardable axes: "
                << get_axes_from_valuedim(lhs_squeeze_value_dim[i]);
      }
      VLOG(4) << "lhs_squeeze_value_dim is : ";
      if (VLOG_IS_ON(4)) {
        for (int i = 0; i < rhs_squeeze_value_dim.size(); ++i) {
          VLOG(4) << "    " << i << " = "
                  << rhs_squeeze_value_dim[i].DebugStr();
          VLOG(4) << "    "
                  << "shardable axes: "
                  << get_axes_from_valuedim(rhs_squeeze_value_dim[i]);
        }
      }
    }

    // compare non_one value dims of
    PADDLE_ENFORCE_EQ(
        lhs_squeeze_value_dim.size(),
        rhs_squeeze_value_dim.size(),
        ::common::errors::PreconditionNotMet(
            "lhs squeezed dims is not equal to rhs squeezed dims"));
    for (int i = 0; i < lhs_squeeze_value_dim.size(); ++i) {
      if (get_axes_from_valuedim(lhs_squeeze_value_dim[i]) !=
          get_axes_from_valuedim(rhs_squeeze_value_dim[i]))
        return false;
    }
    return true;
  }
  bool IdenticalDepAll(const PatternGraph& graph,
                       const LoopValueDims& rhs_dims,
                       const std::vector<LoopValueDims> lhs_dims_vec) {
    std::function<bool(const LoopValueDims)> is_identical_dep =
        [&](const LoopValueDims out) {
          return IdenticalDep(graph, rhs_dims, out);
        };
    return All(MapVector(lhs_dims_vec, is_identical_dep));
  }
  bool operator()(const PatternGraph& graph,
                  const PatternNodePtr& lhs,
                  const PatternNodePtr& rhs) {
    // Middle Variable Must be ( id-dependent ) to support horizontal fusion.
    if (DontHaveMiddleVariable(graph, lhs, rhs)) return true;
    // TODO(huangjiyi): Support horizontal fusion for patterns with dependency
    // relationship
    // const auto& left_dims_vec = GetLoopValueDims(lhs->stmt_pattern());
    // const auto& right_dims_vec = GetLoopValueDims(rhs->stmt_pattern());
    // bool identical_dep = true;
    // for (const auto& right_dims : right_dims_vec) {
    // identical_dep &= IdenticalDepAll(graph, right_dims, left_dims_vec);
    //}
    // return identical_dep;
    return false;
  }
};

}  // namespace cinn::fusion
