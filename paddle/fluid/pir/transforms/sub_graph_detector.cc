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

#include <climits>
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
#include "paddle/common/macros.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_onednn_dialect.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#endif

REGISTER_FILE_SYMBOLS(sub_graph_detector);
namespace pir {
std::vector<pir::Operation*> InverselyTopologicalSort(pir::Block* block) {
  std::vector<pir::Operation*> sort_ops;
  std::unordered_map<pir::Operation*, size_t> pending_count;
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
    if (pending_count[&op] == 0) {
      queue.push(&op);
    }
  }

  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
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
    pir::Operation* op,
    const std::unordered_map<pir::Operation*, int>& op2index) {
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
          op2index.count(source_op),
          common::errors::PreconditionNotMet("source op MUST in op2index map"));
      vec_res.emplace_back(source_op);
    }
  }

  std::sort(vec_res.begin(),
            vec_res.end(),
            [&op2index](pir::Operation* a, pir::Operation* b) {
              return op2index.at(a) > op2index.at(b);
            });

  return vec_res;
}

std::vector<pir::Operation*> GetProducerOps(pir::Operation* op) {
  std::vector<pir::Operation*> producers;

  for (auto operand : GetUsedExternalValue(*op)) {
    if (!operand || !operand.defining_op()) {
      continue;
    }
    auto* source_op = operand.defining_op();
    if (source_op && source_op->GetParent() == op->GetParent()) {
      producers.push_back(source_op);
    }
  }
  return producers;
}

std::vector<pir::Operation*> GetConsumerOps(
    pir::Operation* op,
    const std::unordered_map<pir::Operation*, int>& op2index) {
  std::vector<pir::Operation*> consumers;

  for (auto& result : op->results()) {
    for (auto it = result.use_begin(); it != result.use_end(); ++it) {
      auto parent_op = it->owner();
      while (parent_op) {
        if (op2index.count(parent_op)) {
          consumers.push_back(parent_op);
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
  SubGraph(pir::Operation* op, int index, bool subst)
      : substitute(subst), topo_index(index), id(UniqueId()) {
    ops.push_back(op);
  }

  void Merge(const SubGraphPtr& other);

  static size_t UniqueId() {
    static std::atomic<size_t> counter{0};
    return counter++;
  }

  template <typename V>
  static std::string JointName(const V& subgraphs) {
    std::stringstream ss;
    for (const auto& subgraph : subgraphs) {
      ss << subgraph->name() << ", ";
    }
    auto str = ss.str();
    return str.empty() ? str : str.substr(0, str.size() - 2);
  }

  std::string DebugStr(bool print_ops = false) const {
    std::stringstream ss;
    ss << "=========================================\n";
    ss << name() << " (substitute=" << substitute << ", "
       << "index=" << topo_index << ", "
       << "size=" << ops.size() << ")\n";
    if (print_ops) ss << OpsDebugStr(ops);
    ss << "upstream: " << JointName(upstreams);
    ss << "\ndownstream: " << JointName(downstreams);
    return ss.str();
  }

  std::string name() const {
    return std::string("Subgraph_") + std::to_string(id);
  }

  struct compare {
    bool operator()(const SubGraphPtr& lhs, const SubGraphPtr& rhs) const {
      // sort by reverse order of topo id
      return lhs->id > rhs->id;
    }
  };

  std::vector<pir::Operation*> ops;
  std::set<SubGraphPtr, compare> upstreams;
  std::set<SubGraphPtr, compare> downstreams;

  bool substitute;  // whether this subgraph can be merged
  int topo_index;
  size_t id;
};
using SubGraphPtr = std::shared_ptr<SubGraph>;

void SubGraph::Merge(const SubGraphPtr& other) {
  // Merge other subgraph into this subgraph:
  // Inherit its upstreams/downstreams and ops
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
}

bool HasSinkRoute(const SubGraphPtr& source, const SubGraphPtr& target) {
  if (source == target) return true;
  std::unordered_set<SubGraphPtr> visited;
  std::queue<SubGraphPtr> queue;
  queue.push(source);
  visited.insert(source);
  while (!queue.empty()) {
    SubGraphPtr cur = queue.front();
    queue.pop();
    if (cur == target) return true;
    if (cur->topo_index > target->topo_index) continue;
    for (const auto& subgraph : cur->downstreams) {
      if (visited.count(subgraph)) continue;
      queue.push(subgraph);
      visited.insert(subgraph);
    }
  }
  return false;
}

bool HasLiftRoute(const SubGraphPtr& source, const SubGraphPtr& target) {
  if (source == target) return true;
  std::unordered_set<SubGraphPtr> visited;
  std::queue<SubGraphPtr> queue;
  queue.push(source);
  visited.insert(source);
  while (!queue.empty()) {
    SubGraphPtr cur = queue.front();
    queue.pop();
    if (cur == target) return true;
    if (source->topo_index < target->topo_index) continue;
    for (const auto& subgraph : cur->upstreams) {
      if (visited.count(subgraph)) continue;
      queue.push(subgraph);
      visited.insert(subgraph);
    }
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

std::optional<std::string> DetectCirclesInSubgraphs(
    const std::vector<SubGraphPtr>& subgraph_list) {
  std::set<SubGraphPtr, SubGraph::compare> subgraph_set(subgraph_list.begin(),
                                                        subgraph_list.end());
  std::unordered_map<SubGraphPtr, size_t> in_degree;
  std::unordered_map<SubGraphPtr, size_t> out_degree;
  for (const auto& subgraph : subgraph_set) {
    in_degree[subgraph] = subgraph->upstreams.size();
    out_degree[subgraph] = subgraph->downstreams.size();
  }
  // Recursively remove nodes with in_degree or out_degree = 0
  bool erase_flag = true;
  while (erase_flag) {
    erase_flag = false;
    for (const auto& subgraph : subgraph_list) {
      if (subgraph_set.count(subgraph) == 0) continue;
      if (in_degree[subgraph] == 0) {
        for (const auto& downstream : subgraph->downstreams) {
          in_degree[downstream]--;
        }
        subgraph_set.erase(subgraph);
        erase_flag = true;
        continue;
      }
      if (out_degree[subgraph] == 0) {
        for (const auto& upstream : subgraph->upstreams) {
          out_degree[upstream]--;
        }
        subgraph_set.erase(subgraph);
        erase_flag = true;
        continue;
      }
    }
  }
  if (subgraph_set.empty()) return std::nullopt;
  // If subgraph_set is not empty, there are circles in the subgraphs.
  auto circle_size = subgraph_set.size();
  std::stringstream ss;
  ss << "Circles detected in subgraphs (size=" << circle_size << "): \n";
  for (const auto& subgraph : subgraph_set) {
    ss << subgraph->DebugStr() << "\n";
  }
  return std::make_optional(ss.str());
}

class SubgraphDetector {
 public:
  SubgraphDetector(pir::Block* block, const OpClassifier& classifier);

  void SubgraphFusion();

  std::vector<GroupOpsVec> BuildGroups();

 private:
  void ReorderIndexOfSubgraphs();

  void MergeSource2Target(const SubGraphPtr& source, const SubGraphPtr& target);

  SubGraphPtr GetOpSubgraph(pir::Operation* op) {
    PADDLE_ENFORCE(
        op2subgraph_.count(op),
        ::common::errors::InvalidArgument(
            "Can not find op in op2subgraph_: \n%s", OpsDebugStr({op})));
    return op2subgraph_.at(op);
  }

  std::unordered_map<pir::Operation*, int> op2index_;
  std::vector<pir::Operation*> sort_ops_;
  std::unordered_map<pir::Operation*, SubGraphPtr> op2subgraph_;
  std::unordered_set<int> subgraph_index_set_;
};

void SubgraphDetector::ReorderIndexOfSubgraphs() {
  // After merging subgraphs with direct relation, brother subgraphs with
  // indirect relation may not be detected by index order. So we need to
  // reorder the index of subgraphs.
  std::queue<SubGraphPtr> queue;
  std::unordered_map<SubGraphPtr, int> in_degree;
  for (auto it = sort_ops_.rbegin(); it != sort_ops_.rend(); ++it) {
    auto subgraph = GetOpSubgraph(*it);
    if (in_degree.count(subgraph)) continue;
    in_degree[subgraph] = subgraph->upstreams.size();
    if (in_degree[subgraph] == 0) queue.push(subgraph);
  }
  int index = 0;
  while (!queue.empty()) {
    auto subgraph = queue.front();
    queue.pop();
    subgraph->topo_index = index++;
    for (const auto& downstream : subgraph->downstreams) {
      in_degree[downstream]--;
      if (in_degree[downstream] == 0) queue.push(downstream);
    }
  }
}

void SubgraphDetector::MergeSource2Target(const SubGraphPtr& source,
                                          const SubGraphPtr& target) {
  VLOG(6) << "Merge source: " << source->DebugStr();
  VLOG(6) << "Merge target: " << target->DebugStr();
  target->Merge(source);
  for (const auto& op : source->ops) {
    op2subgraph_[op] = target;
  }
  int max_index = std::max(source->topo_index, target->topo_index);
  int min_index = std::min(source->topo_index, target->topo_index);
  auto merged = target;
  // Check if merged subgraph and its related subgraphs
  // satisfy the topological order condition.
  int upstream_max_index = -1, downstream_min_index = INT_MAX;
  for (const auto& upstream : merged->upstreams) {
    upstream_max_index = std::max(upstream->topo_index, upstream_max_index);
  }
  for (const auto& downstream : merged->downstreams) {
    downstream_min_index =
        std::min(downstream->topo_index, downstream_min_index);
  }
  // 1. If satisfy the topological order after merging, just set max_index
  VLOG(6) << "Check if satisfy the topological order after merging";
  if (min_index > upstream_max_index && max_index < downstream_min_index) {
    merged->topo_index = max_index;
    subgraph_index_set_.erase(min_index);
    return;
  }
  // 2. If not satisfy the order, find a index between upstream_max_index
  // and downstream_min_index while not in subgraph_index_set_.
  VLOG(6) << "Try to find a valid index not in subgraph_index_set_";
  for (int i = upstream_max_index + 1; i < downstream_min_index; ++i) {
    if (!subgraph_index_set_.count(i)) {
      merged->topo_index = i;
      subgraph_index_set_.erase(min_index);
      subgraph_index_set_.erase(max_index);
      subgraph_index_set_.insert(i);
      return;
    }
  }
  // 3. If can not find a valid index, reorder topo index of all subgraphs.
  VLOG(6) << "Reorder topo index of all subgraphs";
  ReorderIndexOfSubgraphs();
}

SubgraphDetector::SubgraphDetector(pir::Block* block,
                                   const OpClassifier& classifier) {
  // init sort_ops_ in reverse topo order and op2index_ in topo order
  int index = 0;
  for (auto& op : *block) {
    sort_ops_.push_back(&op);
    op2index_[&op] = index++;
  }
  std::reverse(sort_ops_.begin(), sort_ops_.end());

  // construct subgraphs and upstream/downstream relation
  std::vector<SubGraphPtr> subgraph_list;
  for (const auto& op : sort_ops_) {
    bool substitute = classifier(*op);
    auto subgraph = std::make_shared<SubGraph>(op, op2index_[op], substitute);
    op2subgraph_[op] = subgraph;
    subgraph_index_set_.insert(op2index_[op]);
    subgraph_list.push_back(subgraph);
  }
  for (const auto& op : sort_ops_) {
    auto subgraph = op2subgraph_[op];
    for (const auto& producer : GetProducerOps(op)) {
      if (!op2subgraph_.count(producer)) continue;
      subgraph->upstreams.insert(op2subgraph_[producer]);
      op2subgraph_[producer]->downstreams.insert(subgraph);
    }
    for (const auto& consumer : GetConsumerOps(op, op2index_)) {
      if (!op2subgraph_.count(consumer)) continue;
      subgraph->downstreams.insert(op2subgraph_[consumer]);
      op2subgraph_[consumer]->upstreams.insert(subgraph);
    }
  }
  VLOG(6) << "Subgraphs before building groups: ";
  for (const auto& subgraph : subgraph_list) {
    VLOG(6) << subgraph->DebugStr();
  }
  auto circle_info = DetectCirclesInSubgraphs(subgraph_list);
  if (circle_info) {
    PADDLE_THROW(::common::errors::PreconditionNotMet(
        "Before building groups: %s", circle_info.value()));
  }
}

void SubgraphDetector::SubgraphFusion() {
  // Two subgraphs can be merged only if they have no route except direct
  // connection between them (brother subgraphs should have no any route),
  // otherwise a circle will be formed after merging them.
  VLOG(4) << "Merge subgraphs with direct relation";
  for (const auto& op : sort_ops_) {
    auto downstream = GetOpSubgraph(op);
    if (!downstream->substitute) continue;
    for (const auto& producer : GetProducerOpsReverseSort(op, op2index_)) {
      auto upstream = GetOpSubgraph(producer);
      if (upstream == downstream || !upstream->substitute) continue;
      if (CanFuseUpstream2Downstream(upstream, downstream)) {
        MergeSource2Target(upstream, downstream);
        VLOG(6) << "Merged subgraph: " << downstream->DebugStr();
      }
    }
  }

  VLOG(4) << "Merge brother subgraphs with same upstream";
  for (const auto& op : sort_ops_) {
    auto subgraph = GetOpSubgraph(op);
    if (!subgraph->substitute) continue;
    for (auto producer : GetProducerOpsReverseSort(op, op2index_)) {
      if (GetOpSubgraph(producer) == subgraph) continue;
      for (auto consumer : GetConsumerOps(producer, op2index_)) {
        auto brother = GetOpSubgraph(consumer);
        if (brother == subgraph || !brother->substitute) continue;
        if (!HasRoute(subgraph, brother) && !HasRoute(brother, subgraph)) {
          MergeSource2Target(brother, subgraph);
          VLOG(6) << "Merged subgraph: " << subgraph->DebugStr();
        }
      }
    }
  }
}

std::vector<GroupOpsVec> SubgraphDetector::BuildGroups() {
  // 1. Get subgraph list in topo order
  std::unordered_set<SubGraphPtr> subgraph_set;
  std::vector<SubGraphPtr> subgraph_list;
  for (const auto& op : sort_ops_) {
    SubGraphPtr subgraph = GetOpSubgraph(op);
    if (subgraph_set.count(subgraph)) continue;
    subgraph_set.insert(subgraph);
    subgraph_list.push_back(subgraph);
  }
  std::reverse(subgraph_list.begin(), subgraph_list.end());
  VLOG(6) << "Subgraphs after building groups: ";
  for (const auto& subgraph : subgraph_list) {
    VLOG(6) << subgraph->DebugStr();
  }
  auto circle_info = DetectCirclesInSubgraphs(subgraph_list);
  if (circle_info) {
    PADDLE_THROW(::common::errors::PreconditionNotMet(
        "After building groups: %s", circle_info.value()));
  }

  // 2. Build group ops in subgraph which can be substituted
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
                return this->op2index_.at(a) < this->op2index_.at(b);
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
