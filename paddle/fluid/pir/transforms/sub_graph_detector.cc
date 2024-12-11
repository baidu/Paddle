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

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

#include <memory>

#include <iterator>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/string.h"
#endif

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/common/flags.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_onednn_dialect.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#endif
namespace pir {
std::vector<pir::Operation*> InverselyTopologicalSort(pir::Block* block) {
  std::vector<pir::Operation*> sort_ops;
  std::unordered_map<pir::Operation*, int> pending_count;
  // step 1: initialize pending_cout for defined op
  for (auto& op : *block) {
    if (pending_count.find(&op) == pending_count.end()) {
      pending_count[&op] = 0;
    }
    for (auto operand : GetUsedExternalValue(op)) {
      if (!operand || !operand.defining_op()) {
        continue;
      }
      auto* defined_op = operand.defining_op();
      if (pending_count.find(defined_op) != pending_count.end()) {
        ++pending_count[defined_op];
      } else {
        pending_count[defined_op] = 1;
      }
    }
  }

  std::queue<pir::Operation*> queue;
  for (auto& op : *block) {
    VLOG(4) << op.name() << " pending_count: " << pending_count[&op];
    if (pending_count[&op] == 0) {
      queue.push(&op);
    }
  }

  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
    VLOG(4) << "Pop Op: " << op->name();
    sort_ops.push_back(op);
    for (auto operand : GetUsedExternalValue(*op)) {
      if (!operand || !operand.defining_op()) {
        continue;
      }
      auto* defined_op = operand.defining_op();
      --pending_count[defined_op];
      if (defined_op && pending_count[defined_op] == 0 &&
          defined_op->GetParent() == block) {
        queue.push(defined_op);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      block->size(),
      sort_ops.size(),
      common::errors::InvalidArgument("sort_ops.size() must be equal to "
                                      "block.size(), but received %d != %d",
                                      block->size(),
                                      sort_ops.size()));

  return sort_ops;
}

std::vector<pir::Operation*> GetProducerOpsReverseSort(
    pir::Operation* op, const std::unordered_map<pir::Operation*, int>& op2id) {
  std::unordered_set<pir::Operation*> producers;

  std::vector<pir::Operation*> vec_res;
  for (auto operand : GetUsedExternalValue(*op)) {
    if (!operand || !operand.defining_op()) {
      continue;
    }
    auto* source_op = operand.defining_op();
    if (source_op && !producers.count(source_op) &&
        source_op->GetParent() == op->GetParent()) {
      producers.insert(source_op);
      PADDLE_ENFORCE(
          op2id.count(source_op),
          common::errors::PreconditionNotMet("source op MUST in op2id map"));
      vec_res.emplace_back(source_op);
    }
  }

  std::sort(vec_res.begin(),
            vec_res.end(),
            [&op2id](pir::Operation* a, pir::Operation* b) {
              return op2id.at(a) > op2id.at(b);
            });

  return vec_res;
}

std::unordered_set<pir::Operation*> GetProducerOps(pir::Operation* op) {
  std::unordered_set<pir::Operation*> producers;

  for (auto operand : GetUsedExternalValue(*op)) {
    if (!operand || !operand.defining_op()) {
      continue;
    }
    auto* source_op = operand.defining_op();
    if (source_op && source_op->GetParent() == op->GetParent()) {
      producers.insert(source_op);
    }
  }
  return producers;
}

std::unordered_set<pir::Operation*> GetConsumerOps(
    pir::Operation* op, const std::unordered_map<pir::Operation*, int>& op2id) {
  std::unordered_set<pir::Operation*> consumers;

  for (auto& result : op->results()) {
    for (auto it = result.use_begin(); it != result.use_end(); ++it) {
      auto parent_op = it->owner();
      while (parent_op) {
        if (op2id.count(parent_op)) {
          consumers.insert(parent_op);
          break;
        }
        parent_op = parent_op->GetParentOp();
      }
    }
  }
  return consumers;
}

static std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(*op);
    ss << "{" << op->id() << "}\n";
  }
  return ss.str();
}

struct SubGraph : public std::enable_shared_from_this<SubGraph> {
  using SubGraphPtr = std::shared_ptr<SubGraph>;
  SubGraph() = delete;
  SubGraph(pir::Operation* op, int id, bool subst)
      : substitute(subst), min_op_id(id), max_op_id(id), name(UniqueId()) {
    ops.push_back(op);
  }

  void Merge(const SubGraphPtr& other);

  static std::string UniqueId() {
    static std::atomic<size_t> counter{0};
    return std::string("Subgraph_") + std::to_string(counter++);
  }

  std::string DebugStr() const {
    std::stringstream ss;
    ss << name << " (substitute=" << substitute << ")";
    ss << "\nupstream: ";
    for (const auto& subgraph : upstreams) {
      ss << subgraph->name << ", ";
    }
    ss << "\ndownstream: ";
    for (const auto& subgraph : downstreams) {
      ss << subgraph->name << ", ";
    }
    ss << "\n" << OpsDebugStr(ops);
    return ss.str();
  }

  struct hash {
    size_t operator()(const SubGraphPtr& subgraph) const {
      return std::hash<std::string>()(subgraph->name);
    }
  };

  std::vector<pir::Operation*> ops;
  std::unordered_set<SubGraphPtr, hash> upstreams;
  std::unordered_set<SubGraphPtr, hash> downstreams;

  bool substitute{true};
  size_t min_op_id;
  size_t max_op_id;
  std::string name;
};
using SubGraphPtr = std::shared_ptr<SubGraph>;

void SubGraph::Merge(const SubGraphPtr& other) {
  SubGraphPtr self = shared_from_this();
  for (const auto& upstream : other->upstreams) {
    if (upstream == self) continue;
    upstream->downstreams.erase(other);
    upstream->downstreams.insert(self);
    upstreams.insert(upstream);
  }
  for (const auto& downstream : other->downstreams) {
    if (downstream == self) continue;
    downstream->upstreams.erase(other);
    downstream->upstreams.insert(self);
    downstreams.insert(downstream);
  }
  upstreams.erase(other);
  downstreams.erase(other);
  ops.insert(ops.begin(), other->ops.begin(), other->ops.end());
  min_op_id = std::min(self->min_op_id, other->min_op_id);
  max_op_id = std::max(self->max_op_id, other->max_op_id);
}

bool HasSinkRoute(const SubGraphPtr& source, const SubGraphPtr& target) {
  if (source == target) return true;
  if (source->min_op_id > target->max_op_id) {
    return false;
  }
  for (const auto& subgraph : source->downstreams) {
    if (HasSinkRoute(subgraph, target)) return true;
  }
  return false;
}

bool HasLiftRoute(const SubGraphPtr& source, const SubGraphPtr& target) {
  if (source == target) return true;
  if (source->max_op_id < target->min_op_id) {
    return false;
  }
  for (const auto& subgraph : source->upstreams) {
    if (HasLiftRoute(subgraph, target)) return true;
  }
  return false;
}

bool HasRoute(const SubGraphPtr& up, const SubGraphPtr& down) {
  return HasSinkRoute(up, down) || HasLiftRoute(down, up);
}

bool CanFuseUpstream2Downstream(const SubGraphPtr& upstream,
                                const SubGraphPtr& downstream) {
  PADDLE_ENFORCE(upstream->downstreams.count(downstream) &&
                     downstream->upstreams.count(upstream),
                 ::common::errors::InvalidArgument(
                     "Subgraphs to be fused must have direct relationship."));
  auto up_downstreams = upstream->downstreams;
  up_downstreams.erase(downstream);
  auto down_upstreams = downstream->upstreams;
  down_upstreams.erase(upstream);
  if (up_downstreams.empty() || down_upstreams.empty()) return true;
  for (const auto& subgraph : up_downstreams) {
    if (HasSinkRoute(subgraph, downstream)) return false;
  }
  for (const auto& subgraph : down_upstreams) {
    if (HasLiftRoute(subgraph, upstream)) return false;
  }
  return true;
}

class SubgraphDetector {
 public:
  SubgraphDetector(pir::Block* block, const OpClassifier& classifier);

  void SubgraphFusion();

  std::vector<GroupOpsVec> BuildGroups();

 private:
  SubGraphPtr GetOpSubgraph(pir::Operation* op) {
    PADDLE_ENFORCE(
        op2subgraph_.count(op),
        ::common::errors::InvalidArgument(
            "Can not find op in op2subgraph_: \n%s", OpsDebugStr({op})));
    return op2subgraph_.at(op);
  }

  std::unordered_map<pir::Operation*, int> op2id_;
  std::vector<pir::Operation*> sort_ops_;
  std::unordered_map<pir::Operation*, SubGraphPtr> op2subgraph_;
};

SubgraphDetector::SubgraphDetector(pir::Block* block,
                                   const OpClassifier& classifier) {
  // init sort_ops_ in reverse topo order
  sort_ops_ = InverselyTopologicalSort(block);
  // init op2id_ in topo order
  int index = 0;
  for (auto& op : *block) {
    VLOG(4) << index << " " << OpsDebugStr({&op});
    op2id_[&op] = index++;
  }
  // construct subgraphs and upstream/downstream relation
  for (const auto& op : sort_ops_) {
    bool substitute = classifier(*op);
    auto subgraph = std::make_shared<SubGraph>(op, op2id_[op], substitute);
    op2subgraph_[op] = subgraph;
  }
  for (const auto& op : sort_ops_) {
    auto subgraph = op2subgraph_[op];
    for (const auto& producer : GetProducerOps(op)) {
      if (!op2subgraph_.count(producer)) continue;
      subgraph->upstreams.insert(op2subgraph_[producer]);
      op2subgraph_[producer]->downstreams.insert(subgraph);
    }
    for (const auto& consumer : GetConsumerOps(op, op2id_)) {
      if (!op2subgraph_.count(consumer)) continue;
      subgraph->downstreams.insert(op2subgraph_[consumer]);
      op2subgraph_[consumer]->upstreams.insert(subgraph);
    }
  }
}

void SubgraphDetector::SubgraphFusion() {
  VLOG(4) << "Merge subgraphs with direct relation";
  for (const auto& op : sort_ops_) {
    auto downstream = GetOpSubgraph(op);
    if (!downstream->substitute) continue;
    for (const auto& producer : GetProducerOpsReverseSort(op, op2id_)) {
      auto upstream = GetOpSubgraph(producer);
      if (upstream == downstream || !upstream->substitute) continue;
      if (CanFuseUpstream2Downstream(upstream, downstream)) {
        downstream->Merge(upstream);
        for (auto upstream_op : upstream->ops) {
          op2subgraph_[upstream_op] = downstream;
        }
      }
    }
  }
  VLOG(4) << "Merge brother subgraphs with same upstream";
  for (const auto& op : sort_ops_) {
    auto subgraph = GetOpSubgraph(op);
    if (!subgraph->substitute) continue;
    for (auto producer : GetProducerOpsReverseSort(op, op2id_)) {
      for (auto consumer : GetConsumerOps(producer, op2id_)) {
        auto brother = GetOpSubgraph(consumer);
        if (brother == subgraph || !brother->substitute) continue;
        if (!HasRoute(subgraph, brother) && !HasRoute(brother, subgraph)) {
          subgraph->Merge(brother);
          for (auto brother_op : brother->ops) {
            op2subgraph_[brother_op] = subgraph;
          }
        }
      }
    }
  }
}

std::vector<GroupOpsVec> SubgraphDetector::BuildGroups() {
  std::unordered_set<SubGraphPtr> subgraph_set;
  std::vector<SubGraphPtr> subgraph_list;
  for (auto op : sort_ops_) {
    SubGraphPtr subgraph = GetOpSubgraph(op);
    if (subgraph_set.count(subgraph)) continue;
    subgraph_set.insert(subgraph);
    subgraph_list.push_back(subgraph);
  }
  std::reverse(subgraph_list.begin(), subgraph_list.end());
  VLOG(4) << "Subgraph list size: " << subgraph_list.size();

  std::vector<GroupOpsVec> groups;
  for (const auto& subgraph : subgraph_list) {
    if (!subgraph->substitute) {
      continue;
    }
    // sort group ops by natural increasing index.
    std::vector<pir::Operation*> group_ops(subgraph->ops.begin(),
                                           subgraph->ops.end());
    std::sort(group_ops.begin(),
              group_ops.end(),
              [this](pir::Operation* a, pir::Operation* b) {
                return this->op2id_.at(a) < this->op2id_.at(b);
              });
    groups.push_back(group_ops);
  }
  return groups;
}

std::vector<GroupOpsVec> DetectSubGraphs(pir::Block* block,
                                         const OpClassifier& classifier) {
  auto subgraph_detector = SubgraphDetector(block, classifier);
  subgraph_detector.SubgraphFusion();
  return subgraph_detector.BuildGroups();
}

std::vector<pir::Value> AnalysisOutputs(
    const GroupOpsVec& group_ops) {  // NOLINT
  // Get output by ud chain
  std::unordered_set<pir::Value> used_by_outside;
  std::unordered_set<pir::Operation*> op_set(group_ops.begin(),
                                             group_ops.end());

  std::vector<pir::Value> outputs;
  for (auto* op : group_ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);

      for (auto use_iter = result.use_begin(); use_iter != result.use_end();
           ++use_iter) {
        if (!op_set.count(use_iter->owner())) {
          outputs.push_back(result);
          break;
        }
      }
    }
  }

  // NOTE: If all value are not used outside, we mark last op's results
  // as outputs. But keep in mind that is risky.
  if (outputs.size() == 0) {
    for (size_t i = 0; i < group_ops.back()->num_results(); ++i) {
      outputs.push_back(group_ops.back()->result(i));
    }
  }

  return outputs;
}

namespace {

struct IncrementalOrder {
  bool operator()(const pir::Operation* lhs, const pir::Operation* rhs) const {
    PADDLE_ENFORCE_EQ(lhs->GetParent() == rhs->GetParent(),
                      true,
                      common::errors::PreconditionNotMet(
                          "lhs and rhs should have same parent block."));
    auto lhs_iter = lhs->operator Block::ConstIterator();
    auto rhs_iter = rhs->operator Block::ConstIterator();
    auto end_iter = lhs->GetParent()->end();
    while (lhs_iter != end_iter) {
      lhs_iter++;
      if (lhs_iter == rhs_iter) return true;
      if (lhs_iter == end_iter) return false;
    }
    PADDLE_ENFORCE_EQ(
        false,
        true,
        common::errors::InvalidArgument("rhs is not reachable from lhs."));
    return false;
  }
};

std::unordered_set<pir::Operation*> GetUpstreamOpsAfterPosition(
    const pir::Operation* position_op,
    const pir::Block* block,
    pir::Operation* op,
    std::unordered_set<pir::Operation*>* visited_ops) {
  std::unordered_set<pir::Operation*> ops;
  const auto& IsInBlock = [](const pir::Operation* src_op,
                             const pir::Block* block) {
    for (auto& item : *block) {
      if (src_op->id() == item.id()) return true;
    }
    return false;
  };
  std::vector<pir::Value> op_inputs = GetUsedExternalValue(*op);
  for (auto value : op_inputs) {
    if (!value || !value.defining_op()) continue;
    pir::Operation* defining_op = value.defining_op();
    if (visited_ops->count(defining_op)) continue;
    visited_ops->insert(defining_op);
    if (!IsInBlock(defining_op, block)) continue;
    if (IncrementalOrder()(defining_op, position_op)) continue;

    ops.insert(defining_op);
    auto recursive_ops = GetUpstreamOpsAfterPosition(
        position_op, block, defining_op, visited_ops);
    ops.insert(recursive_ops.begin(), recursive_ops.end());
  }
  return ops;
}
}  // namespace

void MoveUpstreamOpBeforeGroup(const GroupOpsVec& group_ops,
                               pir::Block* block,
                               pir::Operation* insert_point_op) {
  const auto moved_ops = [&]() {
    std::set<pir::Operation*, IncrementalOrder> ops_set;
    std::unordered_set<pir::Operation*> visited_ops;
    for (auto& op : group_ops) {
      auto upstream_ops =
          GetUpstreamOpsAfterPosition(insert_point_op, block, op, &visited_ops);
      ops_set.insert(upstream_ops.begin(), upstream_ops.end());
    }
    return ops_set;
  }();

  for (auto& op : moved_ops) {
    if (op == insert_point_op) continue;
    VLOG(4) << "Move " << op->id() << " " << op->name() << " before "
            << insert_point_op->id() << " " << insert_point_op->name();
    op->MoveTo(block, insert_point_op->operator Block::Iterator());
  }
}

pir::Operation* FindInsertPoint(const GroupOpsVec& group_ops,
                                const std::vector<pir::Value>& outputs) {
  // Regard last op as insert position if there are no downstream ops between in
  // group_ops.
  pir::Operation* first_op = group_ops.front();
  pir::Operation* insert_point_op = group_ops.back();
  auto order_info =
      [&]() -> std::unordered_map<const pir::Operation*, int64_t> {
    std::unordered_map<const pir::Operation*, int64_t> map;
    // initialize the position index with block size by default.
    auto block = insert_point_op->GetParent();
    int64_t order = 0;
    for (auto& op : *block) {
      map[&op] = order++;
    }
    return map;
  }();

  for (auto* op : group_ops) {
    if (order_info.at(op) > order_info.at(insert_point_op)) {
      insert_point_op = op;
    }
    if (order_info.at(op) < order_info.at(first_op)) {
      first_op = op;
    }
  }

  auto begin = first_op->operator Block::ConstIterator();
  auto end = ++(insert_point_op->operator Block::ConstIterator());
  const std::unordered_set<pir::Value> outputs_set(outputs.begin(),
                                                   outputs.end());
  const std::unordered_set<const pir::Operation*> group_ops_set(
      group_ops.begin(), group_ops.end());

  const auto& IsDownstreamOp = [&](const pir::Operation* op) -> bool {
    if (group_ops_set.find(op) != group_ops_set.end()) return false;
    for (auto& value : GetUsedExternalValue(*op)) {
      if (outputs_set.find(value) != outputs_set.end()) {
        return true;
      }
    }
    return false;
  };
  // Find first downstream op as final insert position.
  for (; begin != end; ++begin) {
    if (IsDownstreamOp(begin)) {
      insert_point_op = begin;
      break;
    }
  }
  return insert_point_op;
}

void ReplaceWithGroupOp(pir::Block* block,
                        const GroupOpsVec& group_ops) {  // NOLINT
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
#ifdef PADDLE_WITH_CINN
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
#endif
#ifdef PADDLE_WITH_DNNL
  ctx->GetOrRegisterDialect<paddle::dialect::OneDNNOperatorDialect>();
#endif
  ::pir::Builder builder = ::pir::Builder(ctx, block);
  const std::vector<pir::Value> outputs = AnalysisOutputs(group_ops);

  // step 1: Analysis and insert group op before insert_point.
  auto* insert_point = FindInsertPoint(group_ops, outputs);
  MoveUpstreamOpBeforeGroup(group_ops, block, insert_point);
  builder.set_insertion_point(insert_point);
  VLOG(6) << "Insert GroupOp after " << insert_point->name();

// step 2: Replace the old op with GroupOp.
#ifdef PADDLE_WITH_CINN

  auto new_group_op = [&]() -> cinn::dialect::GroupOp {
    std::vector<pir::Type> output_types;
    for (auto& value : outputs) output_types.emplace_back(value.type());

    auto group_op = builder.Build<cinn::dialect::GroupOp>(output_types);
    for (auto op : group_ops) {
      op->MoveTo(group_op.block(), group_op.block()->end());
    }
    return group_op;
  }();
#else
  auto new_group_op = [&]() -> pir::GroupOp {
    std::vector<pir::Type> output_types;
    for (auto& value : outputs) output_types.emplace_back(value.type());

    auto group_op = builder.Build<pir::GroupOp>(output_types);
    for (auto op : group_ops) {
      op->MoveTo(group_op.block(), group_op.block()->end());
    }
    return group_op;
  }();
#endif

  // step 3: Replace outputs of inner ops
  const std::vector<pir::Value> group_outs = new_group_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].ReplaceUsesWithIf(group_outs[i],
                                 [&inner_ops](pir::OpOperand op) {
                                   return !inner_ops.count(op.owner());
                                 });
  }

  // step 4: Insert YieldOp for outputs
  builder.SetInsertionPointToBlockEnd(new_group_op.block());
  builder.Build<::pir::YieldOp>(outputs);
}

}  // namespace pir
