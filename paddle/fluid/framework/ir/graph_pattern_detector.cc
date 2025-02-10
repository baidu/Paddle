// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

size_t PDPattern::id_ = 0UL;

#ifdef PADDLE_WITH_TENSORRT
namespace patterns {
thread_local std::unordered_map<std::string, size_t> KeyCounter::dic_;
}
#endif

PDNode *PDPattern::NewNode(const std::string &name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(
        node_map_.count(name),
        0UL,
        common::errors::PreconditionNotMet(
            "PDNode's name should be unique, get duplicate [%s]", name));
  }

  nodes_.emplace_back(new PDNode(this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode *PDPattern::NewNode(PDNode::teller_t &&teller, const std::string &name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(
        node_map_.count(name),
        0UL,
        common::errors::PreconditionNotMet(
            "PDNode's name should be unique, get duplicate [%s]", name));
  }

  nodes_.emplace_back(new PDNode(std::move(teller), this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode *PDPattern::RetrieveNode(const std::string &id) const {
  auto it = node_map_.find(id);
  if (it == node_map_.end()) {
    return nullptr;
  }

  return it->second;
}

void PDPattern::AddEdge(PDNode *a, PDNode *b) {
  PADDLE_ENFORCE_NOT_NULL(a,
                          common::errors::NotFound("PDNode %s is not found.",
                                                   a->name()));  // NOLINT
  PADDLE_ENFORCE_NOT_NULL(b,
                          common::errors::NotFound("PDNode %s is not found.",
                                                   b->name()));  // NOLINT
  PADDLE_ENFORCE_NE(a,
                    b,
                    common::errors::PermissionDenied(
                        "Cannot connect the same node in the graph."));
  edges_.emplace_back(a, b);
}

void GraphPatternDetector::operator()(Graph *graph,
                                      GraphPatternDetector::handle_t handler) {
  if (!MarkPDNodesInGraph(*graph)) {
    return;
  }
  auto subgraphs = DetectPatterns();
  UniquePatterns(&subgraphs);
  SortSubgraphs(&subgraphs);
  RemoveOverlappedMatch(&subgraphs);
  ValidateByNodeRole(&subgraphs);

  if (subgraphs.empty()) return;
  int id = 0;
  for (auto &g : subgraphs) {
    VLOG(3) << "optimizing #" << id++ << " subgraph";
    handler(g, graph);
  }
}

bool GraphPatternDetector::MarkPDNodesInGraph(const ir::Graph &graph) {
  VLOG(3) << "mark pdnodes in graph";
  if (graph.Nodes().empty()) return false;

  for (auto &node : GraphTraits::DFS(graph)) {
    if (node.Name().rfind("__control_var") == 0) continue;
    for (const auto &pdnode : pattern_.nodes()) {
      if (pdnode->Tell(&node)) {
        VLOG(4) << "Node " << node.Name() << "(" << node.id() << ")"
                << " marked as " << pdnode->name();
        pdnodes2nodes_[pdnode.get()].insert(&node);
      }
    }
  }
  // Check to early stop if some PDNode can't find matched Node.
  for (auto &pdnode : pattern_.nodes()) {
    if (!pdnodes2nodes_.count(pdnode.get())) {
      VLOG(4) << pdnode->name() << " can't find matched Node, early stop";
      // return false;
    }
  }
  VLOG(3) << pdnodes2nodes_.size() << " nodes marked";

  return !pdnodes2nodes_.empty();
}

// The intermediate Nodes can only link to the nodes inside the pattern, or this
// subgraph will be dropped.
void GraphPatternDetector::ValidateByNodeRole(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  std::vector<GraphPatternDetector::subgraph_t> result;

  subgraphs->erase(
      std::remove_if(
          subgraphs->begin(),
          subgraphs->end(),
          [](const GraphPatternDetector::subgraph_t &subgraph) -> bool {
            // Collect the inputs and outputs.
            std::set<Node *> ios;
            for (auto &item : subgraph) {
              if (!item.first->IsIntermediate()) {
                ios.insert(item.second);
              }
            }
            for (auto &item : subgraph) {
              if (item.first->IsIntermediate()) {
                for (auto *x : item.second->inputs) {
                  if (!ios.count(x)) {
                    return true;
                  }
                }
                for (auto *x : item.second->outputs) {
                  if (!ios.count(x)) {
                    return true;
                  }
                }
              }
            }
            return false;
          }),
      subgraphs->end());
}

struct HitGroup {
  std::map<PDNode *, Node *> roles;
  HitGroup() : roles(), nodes_() {}
  bool Match(Node *node, PDNode *pat) {
    if (nodes_.count(node)) {
      if (roles.count(pat) && roles[pat] == node) return true;
      return false;
    } else {
      if (roles.count(pat) && roles[pat] != node) return false;
      return true;
    }
  }

  void Register(Node *node, PDNode *pat) {
    roles[pat] = node;
    nodes_.insert(node);
  }

 private:
  std::set<Node *> nodes_;
};

// Tell whether Node a links to b.
bool IsNodesLink(Node *a, Node *b) {
  for (auto *node : a->outputs) {
    if (b == node) {
      return true;
    }
  }
  return false;
}

std::vector<GraphPatternDetector::subgraph_t>
GraphPatternDetector::DetectPatterns() {
  // Init empty subgraphs.
  std::vector<GraphPatternDetector::subgraph_t> result;
  std::vector<HitGroup> init_groups;
  std::array<std::vector<HitGroup>, 2> bi_records;
  auto *first_pnode = pattern_.edges().empty() ? pattern().nodes().front().get()
                                               : pattern_.edges().front().first;
  if (!pdnodes2nodes_.count(first_pnode)) return result;
  for (auto *node : pdnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.roles[first_pnode] = node;
    init_groups.emplace_back(group);
  }

  int step = 0;
  bi_records[0] = std::move(init_groups);

  // Extend a PDNode to subgraphs by deducing the connection relations defined
  // in edges of PDNodes.
  for (const auto &edge : pattern_.edges()) {
    VLOG(4) << "check " << edge.first->name() << " -> " << edge.second->name();
    // TODO(Superjomn) Fix bug here, the groups might be duplicate here.
    // Each role has two PDNodes, which indicates two roles.
    // Detect two Nodes that can match these two roles and they are connected.
    auto &pre_groups = bi_records[step % 2];
    auto &cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    if (pre_groups.empty()) break;
    // source -> target
    for (Node *source : pdnodes2nodes_[edge.first]) {
      for (Node *target : pdnodes2nodes_[edge.second]) {
        VLOG(8) << "check " << source->Name() << "(" << source->id() << ")"
                << " -- " << target->Name() << "(" << target->id() << ")";
        // TODO(Superjomn) add some prune strategies.
        for (const auto &group : pre_groups) {
          if (IsNodesLink(source, target)) {
            HitGroup new_group = group;
            bool flag = new_group.Match(source, edge.first) &&
                        new_group.Match(target, edge.second);
            if (flag) {
              new_group.Register(source, edge.first);
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
              // TODO(Superjomn) need to unique
            }
          }
        }
      }
    }
    VLOG(3) << "step " << step << " get records: " << cur_groups.size();
    for (auto &group : cur_groups) {
      for (auto &item : group.roles) {
        VLOG(4) << "node " << item.second->Name() << "(" << item.second->id()
                << ")"
                << " as " << item.first->name();
      }
      VLOG(4) << "=========================================================";
    }
  }

  for (auto &group : bi_records[step % 2]) {
    GraphPatternDetector::subgraph_t subgraph;
    for (auto &role : group.roles) {
      subgraph.emplace(role.first, role.second);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

struct GraphItemLessThan {
  bool operator()(const std::pair<PDNode *, Node *> &a,
                  const std::pair<PDNode *, Node *> &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    } else {
      return a.second < b.second;
    }
  }
};

// TODO(Superjomn) enhance the function as it marks unique unique as duplicates
// see https://github.com/PaddlePaddle/Paddle/issues/13550
void GraphPatternDetector::UniquePatterns(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  if (subgraphs->empty()) return;
  std::vector<GraphPatternDetector::subgraph_t> result;

  std::set<size_t> set;
  std::hash<std::string> hasher;
  for (auto &g : *subgraphs) {
    // Sort the items in the sub-graph, and transform to a string key.
    std::vector<std::pair<PDNode *, Node *>> sorted_keys(g.begin(), g.end());
    std::sort(sorted_keys.begin(), sorted_keys.end(), GraphItemLessThan());
    std::stringstream ss;
    for (auto &item : sorted_keys) {
      ss << item.first << ":" << item.second;
    }
    auto key = hasher(ss.str());
    if (!set.count(key)) {
      result.emplace_back(g);
      set.insert(key);
    }
  }
  *subgraphs = result;
}

void GraphPatternDetector::SortSubgraphs(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  if (subgraphs->empty()) return;
  bool has_bn_add_act = false;
  for (auto &subgraph : *subgraphs) {
    for (auto &item : subgraph) {
      if (item.first->name().find("bn_add_act") != std::string::npos) {
        has_bn_add_act = true;
        break;
      }
    }
  }
  if (!has_bn_add_act) {
    return;
  }

  std::sort(
      subgraphs->begin(),
      subgraphs->end(),
      [](const GraphPatternDetector::subgraph_t &a,
         const GraphPatternDetector::subgraph_t &b) {
        for (auto &item : a) {
          if (item.first->name().find("bn_add_act") != std::string::npos &&
              item.first->name().find("bn_reserve_space") !=
                  std::string::npos) {
            auto it_b = b.find(item.first);
            if (it_b != b.end()) {
              if (item.second->Name() != it_b->second->Name()) {
                return item.second->Name() < it_b->second->Name();
              } else {
                return false;
              }
            } else {
              return false;
            }
          }
        }
        return false;
      });
}

void GraphPatternDetector::RemoveOverlappedMatch(
    std::vector<subgraph_t> *subgraphs) {
  std::vector<subgraph_t> result;
  std::set<Node *> node_set;

  for (const auto &subgraph : *subgraphs) {
    bool valid = true;
    for (auto &item : subgraph) {
      if (item.first->IsIntermediate() && node_set.count(item.second)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      for (auto &item : subgraph) {
        node_set.insert(item.second);
      }
      result.push_back(subgraph);
    }
  }
  *subgraphs = result;
}

std::string PDPattern::DotString() const {
  using inference::analysis::Dot;
  Dot dot;
  int id = 0;
  // Create Nodes
  std::unordered_map<PDNode *, std::string> node2dot;
  for (const auto &node : nodes()) {
    std::string node_id = "Node" + std::to_string(id++);
    dot.AddNode(node_id, {}, node->name());
    node2dot[node.get()] = node_id;
  }
  // Create Edges
  for (const auto &edge : edges()) {
    if (!node2dot.count(edge.first) || !node2dot.count(edge.second)) {
      continue;
    }
    auto &src = node2dot.at(edge.first);
    auto &trg = node2dot.at(edge.second);
    dot.AddEdge(src, trg, {});
  }
  return dot.Build();
}

PDNode &PDNode::LinksTo(const std::vector<PDNode *> &others) {
  // extend outlinks.
  for (PDNode *x : others) {
    pattern_->AddEdge(this, x);
  }
  return *this;
}

PDNode &PDNode::LinksFrom(const std::vector<PDNode *> &others) {
  // extend outlinks.
  for (PDNode *x : others) {
    pattern_->AddEdge(x, this);
  }
  return *this;
}

PDNode *PDNode::assert_is_op() {
  asserts_.emplace_back([](Node *x) { return x && x->IsOp(); });
  return this;
}

PDNode *PDNode::assert_is_op(const std::string &op_type) {
  asserts_.emplace_back([op_type](Node *x) {
    return x && x->IsOp() && x->Op()->Type() == op_type;
  });
  return this;
}

PDNode *PDNode::assert_is_not_op_type(const std::string &op_type) {
  asserts_.emplace_back([op_type](Node *x) {
    return x && x->IsOp() && x->Op()->Type() != op_type;
  });
  return this;
}

PDNode *PDNode::assert_is_var() {
  asserts_.emplace_back([](Node *x) { return x && x->IsVar(); });
  return this;
}

PDNode *PDNode::assert_var_dtype(proto::VarType::Type dtype) {
  assert_is_var();
  asserts_.emplace_back(
      [dtype](Node *x) { return x->Var()->GetDataType() == dtype; });
  return this;
}

PDNode *PDNode::assert_is_not_ctrl_var() {
  asserts_.emplace_back([](Node *x) { return x && !x->IsCtrlVar(); });
  return this;
}

PDNode *PDNode::assert_var_not_persistable() {
  assert_is_var();
  asserts_.emplace_back(
      [](Node *x) { return x->Var() && !x->Var()->Persistable(); });
  return this;
}

PDNode *PDNode::assert_is_persistable_var() {
  assert_is_var();
  asserts_.emplace_back(
      [=](Node *x) { return x->Var() && x->Var()->Persistable(); });
  return this;
}

PDNode *PDNode::assert_is_op_nth_input(const std::string &op_type,
                                       const std::string &argument,
                                       int nth) {
  assert_is_var();
  assert_is_op_input(op_type);
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op->IsOp() && op->Op()->Type() == op_type &&
          IsNthInput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_nth_output(const std::string &op_type,
                                        const std::string &argument,
                                        int nth) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op->IsOp() && op->Op()->Type() == op_type &&
          IsNthOutput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_input_of_op(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type &&
          op->inputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_output_of_op(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type &&
          op->outputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_output(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_output(const std::string &op_type,
                                    const std::string &argument) {
  assert_is_var();
  assert_is_op_nth_output(op_type, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_op_input(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_not_op_input(const std::string &argument) {
  assert_is_op();
  asserts_.emplace_back([=](Node *x) {
    auto &ins = x->Op()->Inputs();
    auto iter = ins.find(argument);
    return iter == ins.end() || iter->second.empty();
  });
  return this;
}

PDNode *PDNode::assert_is_op_input(const std::string &op_type,
                                   const std::string &argument) {
  assert_is_var();
  assert_is_op_nth_input(op_type, argument, 0);
  return this;
}

PDNode *PDNode::assert_op_has_n_inputs(const std::string &op_type, size_t n) {
  assert_is_op(op_type);
  asserts_.emplace_back([=](Node *x) { return x->inputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_op_has_n_outputs(const std::string &op_type, size_t n) {
  assert_is_op(op_type);
  asserts_.emplace_back([=](Node *x) { return x->outputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_has_n_inputs(size_t n) {
  asserts_.emplace_back([=](Node *x) { return x->inputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_has_n_outputs(size_t n) {
  asserts_.emplace_back([=](Node *x) { return x->outputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_more(PDNode::teller_t &&teller) {
  asserts_.emplace_back(std::move(teller));
  return this;
}

PDNode *PDNode::assert_is_ops(const std::unordered_set<std::string> &op_types) {
  asserts_.emplace_back([op_types](Node *x) {
    return x && x->IsOp() && op_types.count(x->Op()->Type());
  });
  return this;
}

PDNode *PDNode::assert_is_ops_nth_input(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument,
    int nth) {
  assert_is_var();
  assert_is_ops_input(op_types);
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op->IsOp() && op_types.count(op->Op()->Type()) &&
          IsNthInput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_nth_output(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument,
    int nth) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op->IsOp() && op_types.count(op->Op()->Type()) &&
          IsNthOutput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}
PDNode *PDNode::assert_is_ops_output(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type())) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_output(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument) {
  assert_is_var();
  assert_is_ops_nth_output(op_types, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_ops_input(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type())) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_input(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument) {
  assert_is_var();
  assert_is_ops_nth_input(op_types, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_only_input_of_ops(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type()) &&
          op->inputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_output_of_ops(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type()) &&
          op->outputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

bool VarLinksToOp(Node *node, const std::string &op_type) {
  for (auto *out : node->outputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

bool IsNthInput(Node *var, Node *op, const std::string &argument, size_t nth) {
  PADDLE_ENFORCE_EQ(
      var->IsVar(),
      true,
      common::errors::InvalidArgument(
          "First parameter of function IsNthInput must be Node::Var"));
  PADDLE_ENFORCE_EQ(
      op->IsOp(),
      true,
      common::errors::InvalidArgument(
          "Second parameter of function IsNthInput must be Node::Op"));
  if (!HasInput(op, argument) || op->Op()->Input(argument).size() <= nth)
    return false;
  return var->Name() == op->Op()->Input(argument)[nth];
}

bool HasInput(Node *op, const std::string &argument) {
  PADDLE_ENFORCE_EQ(
      op->IsOp(),
      true,
      common::errors::InvalidArgument(
          "First parameter of function HasInput must be Node::Op"));
  auto const &names = op->Op()->InputNames();
  if (std::find(names.begin(), names.end(), argument) == names.end())
    return false;
  return true;
}

bool HasOutput(Node *op, const std::string &argument) {
  PADDLE_ENFORCE_EQ(
      op->IsOp(),
      true,
      common::errors::InvalidArgument(
          "First parameter of function HasOutput must be Node::Op"));
  auto const &names = op->Op()->OutputNames();
  if (std::find(names.begin(), names.end(), argument) == names.end())
    return false;
  return true;
}

bool IsNthOutput(Node *var, Node *op, const std::string &argument, size_t nth) {
  PADDLE_ENFORCE_EQ(
      var->IsVar(),
      true,
      common::errors::InvalidArgument(
          "First parameter of function IsNthOutput must be Node::Var"));
  PADDLE_ENFORCE_EQ(
      op->IsOp(),
      true,
      common::errors::InvalidArgument(
          "Second parameter of function IsNthOutput must be Node::Op"));
  if (!HasOutput(op, argument) || op->Op()->Output(argument).size() <= nth)
    return false;
  return var->Name() == op->Op()->Output(argument)[nth];
}

void GraphSafeRemoveNodes(
    Graph *graph,
    const std::unordered_set<const Node *> &nodes,
    std::unordered_set<std::shared_ptr<Node>> *saved_nodes) {
  for (auto *node : nodes) {
    if (saved_nodes != nullptr) {
      // prevent unique_ptr node from being released
      saved_nodes->insert(graph->RemoveNode(const_cast<Node *>(node)));
    } else {
      graph->RemoveNode(const_cast<Node *>(node));
    }
  }

  for (auto *node : graph->Nodes()) {
    for (auto it = node->inputs.begin(); it != node->inputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node *>(node)->inputs.erase(it);
      } else {
        it++;
      }
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node *>(node)->outputs.erase(it);
      } else {
        it++;
      }
    }
  }
}

bool VarLinksFromOp(Node *node, const std::string &op_type) {
  for (auto *out : node->inputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

PDNode *patterns::ConvBN::operator()(paddle::framework::ir::PDNode *conv_input,
                                     const std::string &conv_type,
                                     bool with_eltwise_add) {
  // Create Operators
  conv_input->assert_is_op_input(conv_type, "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op(conv_type);

  PDNode *eltwise_op = nullptr;
  if (with_eltwise_add) {
    eltwise_op =
        pattern->NewNode(eltwise_repr())->assert_is_op("elementwise_add");
  }
  auto *batch_norm_op =
      pattern->NewNode(batch_norm_repr())->assert_is_op("batch_norm");
  // Create variables
  // Conv Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input(conv_type, "Filter");

  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op(conv_type);

  PDNode *eltwise_y_in_var = nullptr;
  PDNode *eltwise_out_var = nullptr;
  if (with_eltwise_add) {
    // Conv output as Bias input
    conv_out_var->assert_is_op_input("elementwise_add", "X");
    // Bias
    eltwise_y_in_var = pattern->NewNode(eltwise_y_in_repr())
                           ->assert_is_op_input("elementwise_add", "Y")
                           ->assert_is_persistable_var()
                           ->AsInput();
    eltwise_out_var = pattern->NewNode(eltwise_out_repr())
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("elementwise_add");
  } else {
    // Conv output as BN input
    conv_out_var->assert_is_op_input("batch_norm", "X");
  }

  // BN Scale
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->AsInput()
                           ->assert_is_persistable_var()
                           ->assert_is_op_input("batch_norm", "Scale")
                           ->assert_has_n_outputs(1);
  // BN Bias
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_is_op_input("batch_norm", "Bias")
                          ->assert_has_n_outputs(1);
  // BN Mean
  auto *bn_mean_var = pattern->NewNode(bn_mean_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_is_op_input("batch_norm", "Mean")
                          ->assert_has_n_outputs(1);
  // BN Variance
  auto *bn_variance_var = pattern->NewNode(bn_variance_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->assert_has_n_outputs(1);

  // BN output
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("batch_norm", "Y");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("batch_norm", "MeanOut")
                              ->assert_has_n_outputs(0);

  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->AsOutput()
          ->assert_is_op_output("batch_norm", "VarianceOut")
          ->assert_has_n_outputs(0);

  auto *bn_saved_mean_var = pattern->NewNode(bn_saved_mean_repr())
                                ->AsOutput()
                                ->assert_is_op_output("batch_norm", "SavedMean")
                                ->assert_has_n_outputs(0);

  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("batch_norm", "SavedVariance")
          ->assert_has_n_outputs(0);

  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});

  if (with_eltwise_add) {
    eltwise_op->LinksFrom({conv_out_var, eltwise_y_in_var})
        .LinksTo({eltwise_out_var});
    batch_norm_op
        ->LinksFrom({eltwise_out_var,
                     bn_scale_var,
                     bn_bias_var,
                     bn_mean_var,
                     bn_variance_var})
        .LinksTo({bn_out_var,
                  bn_mean_out_var,
                  bn_variance_out_var,
                  bn_saved_mean_var,
                  bn_saved_variance_var});
  } else {
    batch_norm_op
        ->LinksFrom({conv_out_var,
                     bn_scale_var,
                     bn_bias_var,
                     bn_mean_var,
                     bn_variance_var})
        .LinksTo({bn_out_var,
                  bn_mean_out_var,
                  bn_variance_out_var,
                  bn_saved_mean_var,
                  bn_saved_variance_var});
  }
  return bn_out_var;
}

PDNode *patterns::OperatorActivation::operator()(
    const std::string &operator_type, const std::string &activation_type) {
  auto *preceding_op =
      pattern->NewNode(preceding_op_repr())->assert_is_op(operator_type);
  auto *preceding_op_out = pattern->NewNode(preceding_op_out_repr())
                               ->AsIntermediate()
                               ->assert_is_only_output_of_op(operator_type)
                               ->assert_is_op_input(activation_type);
  auto *activation_op =
      pattern->NewNode(activation_repr())->assert_is_op(activation_type);
  auto *activation_out = pattern->NewNode(activation_out_repr())
                             ->AsOutput()
                             ->assert_is_op_output(activation_type);
  preceding_op->LinksTo({preceding_op_out});
  activation_op->LinksFrom({preceding_op_out}).LinksTo({activation_out});
  return activation_out;
}

PDNode *patterns::QuantTranspose::operator()(
    const std::string &transpose_type) {
  auto *quant_in = pattern->NewNode(quant_in_repr())
                       ->AsInput()
                       ->assert_is_op_input("quantize", "Input");
  auto *quant_op = pattern->NewNode(quant_op_repr())->assert_is_op("quantize");
  auto *quant_out = pattern->NewNode(quant_out_repr())
                        ->AsOutput()
                        ->AsIntermediate()
                        ->assert_has_n_outputs(1)
                        ->assert_is_op_output("quantize")
                        ->assert_is_op_input(transpose_type, "X");
  auto *transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op(transpose_type);

  quant_op->LinksFrom({quant_in}).LinksTo({quant_out});
  transpose_op->LinksFrom({quant_out});

  return transpose_op;
}

PDNode *patterns::TransposeDequant::operator()(
    const std::string &transpose_type) {
  auto *transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op(transpose_type);
  auto dequant_in = pattern->NewNode(dequant_in_repr())
                        ->AsIntermediate()
                        ->assert_has_n_inputs(1)
                        ->assert_is_op_input("dequantize", "Input");
  auto dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");
  auto dequant_out = pattern->NewNode(dequant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("dequantize", "Output");

  transpose_op->LinksTo({dequant_in});
  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  return dequant_out;
}

PDNode *patterns::Squeeze2Transpose2::operator()() {
  auto *squeeze2_op_in = pattern->NewNode(squeeze2_op_in_repr())
                             ->AsInput()
                             ->assert_has_n_outputs(1)
                             ->assert_is_op_input("squeeze2", "X");
  auto *squeeze2_op = pattern->NewNode(squeeze2_op_repr())
                          ->assert_is_op("squeeze2")
                          ->assert_has_n_outputs(2);
  auto *squeeze2_op_out = pattern->NewNode(squeeze2_op_out_repr())
                              ->AsIntermediate()
                              ->assert_is_op_output("squeeze2", "Out")
                              ->assert_is_op_input("transpose2", "X");
  auto *transpose2_op =
      pattern->NewNode(transpose2_op_repr())->assert_is_op("transpose2");

  squeeze2_op->LinksFrom({squeeze2_op_in}).LinksTo({squeeze2_op_out});
  transpose2_op->LinksFrom({squeeze2_op_out});
  return transpose2_op;
}

PDNode *patterns::OperatorUnsqueeze2::operator()(
    const std::string &operator_type, const int num_of_operator_outs) {
  auto *preceding_op = pattern->NewNode(preceding_op_repr())
                           ->assert_is_op(operator_type)
                           ->assert_has_n_outputs(num_of_operator_outs);
  auto *preceding_op_out = pattern->NewNode(preceding_op_out_repr())
                               ->AsIntermediate()
                               ->assert_is_op_output(operator_type, "Out")
                               ->assert_is_op_input("unsqueeze2");
  auto *unsqueeze2_op =
      pattern->NewNode(unsqueeze2_op_repr())->assert_is_op("unsqueeze2");
  auto *unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->AsOutput()
                             ->assert_is_op_output("unsqueeze2");
  preceding_op->LinksTo({preceding_op_out});
  unsqueeze2_op->LinksFrom({preceding_op_out}).LinksTo({unsqueeze2_out});
  return unsqueeze2_out;
}

PDNode *patterns::OperatorReshape2::operator()(const std::string &operator_type,
                                               const int num_of_operator_outs) {
  auto *preceding_op = pattern->NewNode(preceding_op_repr())
                           ->assert_is_op(operator_type)
                           ->assert_has_n_outputs(num_of_operator_outs);
  auto *preceding_op_out = pattern->NewNode(preceding_op_out_repr())
                               ->AsIntermediate()
                               ->assert_is_op_output(operator_type, "Out")
                               ->assert_is_op_input("reshape2");
  auto *reshape2_op =
      pattern->NewNode(reshape2_op_repr())->assert_is_op("reshape2");
  auto *reshape2_out = pattern->NewNode(reshape2_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("reshape2");
  preceding_op->LinksTo({preceding_op_out});
  reshape2_op->LinksFrom({preceding_op_out}).LinksTo({reshape2_out});
  return reshape2_out;
}

PDNode *patterns::SeqConvEltAddRelu::operator()(
    paddle::framework::ir::PDNode *seqconv_input) {
  // Create Operators
  seqconv_input->assert_is_op_input("sequence_conv", "X");
  auto *seqconv_op = pattern->NewNode(seqconv_repr())
                         ->assert_is_op("sequence_conv")
                         ->assert_has_n_inputs(2)
                         ->assert_op_attr<bool>("paddingTrainable", false)
                         ->assert_op_attr<int>("contextStride", 1);

  auto *eltadd_op =
      pattern->NewNode(eltadd_repr())->assert_is_op("elementwise_add");
  auto *relu_op = pattern->NewNode(relu_repr())->assert_is_op("relu");
  // Create variables
  // Filter
  auto *seqconv_weight_var =
      pattern->NewNode(seqconv_weight_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("sequence_conv", "Filter");
  // Bias
  auto *eltadd_bias_var = pattern->NewNode(eltadd_bias_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add");
  // intermediate variable, will be removed in the IR after fuse.
  auto *seqconv_out_var = pattern->NewNode(seqconv_out_repr())
                              ->AsIntermediate()
                              ->assert_is_only_output_of_op("sequence_conv")
                              ->assert_is_op_input("elementwise_add");
  auto *eltadd_out_var = pattern->NewNode(eltadd_out_repr())
                             ->AsIntermediate()
                             ->assert_is_only_output_of_op("elementwise_add")
                             ->assert_is_only_input_of_op("relu");
  // output
  auto *relu_out_var = pattern->NewNode(relu_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("relu");

  seqconv_op->LinksFrom({seqconv_input, seqconv_weight_var})
      .LinksTo({seqconv_out_var});
  eltadd_op->LinksFrom({seqconv_out_var, eltadd_bias_var})
      .LinksTo({eltadd_out_var});
  relu_op->LinksFrom({eltadd_out_var}).LinksTo({relu_out_var});
  return relu_out_var;
}

PDNode *patterns::FC::operator()(paddle::framework::ir::PDNode *x,
                                 bool with_bias,
                                 bool with_relu) {
  // Create shared nodes.
  x->assert_is_op_input("mul", "X");
  auto *mul = pattern->NewNode(mul_repr())->assert_is_op("mul");

  auto *mul_w_var = pattern->NewNode(w_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("mul", "Y");

  auto *mul_out_var =
      pattern->NewNode(mul_out_repr())->assert_is_op_output("mul");

  // Add links.
  mul->LinksFrom({x, mul_w_var}).LinksTo({mul_out_var});
  if (!with_bias) {  // not with bias
    return mul_out_var;
  } else {  // with bias
    mul_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
    // Create operators.
    auto *elementwise_add = pattern->NewNode(elementwise_add_repr())
                                ->assert_is_op("elementwise_add");
    // Create variables.
    auto *bias = pattern->NewNode(bias_repr())
                     ->assert_is_op_input("elementwise_add")
                     ->assert_is_persistable_var()
                     ->AsInput();

    auto *elementwise_add_out_var =
        pattern->NewNode(elementwise_add_out_repr())
            ->AsOutput()
            ->assert_is_op_output("elementwise_add");

    elementwise_add->LinksFrom({mul_out_var, bias})
        .LinksTo({elementwise_add_out_var});
    if (!with_relu) {
      return elementwise_add_out_var;
    } else {
      elementwise_add_out_var->AsIntermediate()->assert_is_op_input("relu");
      // Create operators.
      auto *relu = pattern->NewNode(relu_repr())->assert_is_op("relu");
      auto *relu_out_var = pattern->NewNode(relu_out_repr())
                               ->AsOutput()
                               ->assert_is_op_output("relu");

      relu->LinksFrom({elementwise_add_out_var}).LinksTo({relu_out_var});
      return relu_out_var;
    }
  }
}

PDNode *patterns::FCMKLDNN::operator()(bool with_residual_data) {
  auto *fc_op = pattern->NewNode(fc_repr())->assert_is_op("fc");
  // Create variables
  // Input
  auto *input_var = pattern->NewNode(input_repr())
                        ->AsInput()
                        ->assert_is_op_input("fc", "Input");
  // Filter
  auto *fc_weight_var = pattern->NewNode(weights_repr())
                            ->AsInput()
                            ->assert_is_op_input("fc", "W");
  // Bias
  auto *fc_bias_var = pattern->NewNode(bias_repr())
                          ->AsInput()
                          ->assert_is_op_input("fc", "Bias");
  // Output
  auto *fc_out_var = pattern->NewNode(output_repr())
                         ->AsOutput()
                         ->assert_is_op_output("fc", "Out")
                         ->assert_is_only_output_of_op("fc");

  std::vector<PDNode *> links_from{input_var, fc_weight_var, fc_bias_var};
  if (with_residual_data) {
    auto res_fc_var = pattern->NewNode(residual_data_repr())
                          ->AsInput()
                          ->assert_is_op_input("fc", "ResidualData");
    links_from.push_back(res_fc_var);
  } else {
    fc_op->assert_more([&](Node *x) {
      if (!HasInput(x, "ResidualData") ||
          x->Op()->Input("ResidualData").empty())
        return true;
      return false;
    });
  }

  fc_op->LinksFrom(links_from).LinksTo({fc_out_var});
  return fc_out_var;
}

PDNode *patterns::Embedding::operator()(PDNode *x) {
  x->assert_is_op_input("lookup_table", "Ids");
  auto *lookup_table_op =
      pattern->NewNode(lookup_table_repr())->assert_is_op("lookup_table");
#define NEW_NODE(arg__, io__)                    \
  auto *arg__ = pattern->NewNode(arg__##_repr()) \
                    ->assert_is_op_##io__("lookup_table", #arg__);

  NEW_NODE(W, input);

  NEW_NODE(Out, output);
#undef NEW_NODE

  lookup_table_op->LinksFrom({x, W});
  lookup_table_op->LinksTo({Out});
  return Out;
}

PDNode *patterns::LSTM::operator()(PDNode *x) {
  x->assert_is_op_input("lstm", "Input");
  auto *lstm_op = pattern->NewNode(lstm_repr())->assert_is_op("lstm");
#define NEW_NODE(arg__, io__) \
  auto *arg__ =               \
      pattern->NewNode(arg__##_repr())->assert_is_op_##io__("lstm", #arg__);

  // Currently, the H0 and C0 are optional
  // TODO(Superjomn) upgrade the fuse framework to support optional.
  // NEW_NODE(H0, input);
  // NEW_NODE(C0, input);
  NEW_NODE(Weight, input);
  NEW_NODE(Bias, input);

  NEW_NODE(Hidden, output);
  NEW_NODE(Cell, output);
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchCellPreAct, output);
#undef NEW_NODE

  lstm_op->LinksFrom({x, Weight, Bias});
  lstm_op->LinksTo({Hidden, Cell, BatchGate, BatchCellPreAct});
  return Hidden;
}

PDNode *patterns::GRU::operator()(PDNode *x) {
  x->assert_is_op_input("gru", "Input");
  auto *gru_op = pattern->NewNode(gru_repr())->assert_is_op("gru");
#define NEW_NODE(arg__, io__) \
  auto *arg__ =               \
      pattern->NewNode(arg__##_repr())->assert_is_op_##io__("gru", #arg__);

  NEW_NODE(Weight, input);
  // TODO(Superjomn): upgrade the fuse framework to support optional.
  // H0 and bias are optional
  NEW_NODE(Bias, input);  // also optional
  // NEW_NODE(H0, input);

  NEW_NODE(Hidden, output);
  // below are intermediate
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchResetHiddenPrev, output);
  NEW_NODE(BatchHidden, output);
#undef NEW_NODE

  BatchGate->AsIntermediate();
  BatchResetHiddenPrev->AsIntermediate();
  BatchHidden->AsIntermediate();

  gru_op->LinksFrom({x, Weight, Bias});
  gru_op->LinksTo({Hidden, BatchGate, BatchResetHiddenPrev, BatchHidden});
  return Hidden;
}

PDNode *patterns::ActElewiseAdd::operator()(
    paddle::framework::ir::PDNode *in_var,
    std::unordered_set<std::string> act_types) {
  in_var->assert_is_ops_input(act_types, "X");

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);
  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_not_ctrl_var()
                          ->assert_is_ops_output(act_types);
  act_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto *ele_x_var = pattern->NewNode(ele_x_repr())
                        ->assert_is_not_ctrl_var()
                        ->assert_is_op_input("elementwise_add")
                        ->AsInput();
  auto *elementwise_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *elewise_add_out = pattern->NewNode(elewise_add_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("elementwise_add", "Out");

  act->LinksFrom({in_var}).LinksTo({act_out_var});
  elementwise_add->LinksFrom({act_out_var, ele_x_var})
      .LinksTo({elewise_add_out});

  return elewise_add_out;
}

PDNode *patterns::BatchNormAct::operator()(
    paddle::framework::ir::PDNode *bn_x_var,
    std::unordered_set<std::string> act_types) {
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm", "Bias");
  auto *bn_variance_var = pattern->NewNode(bn_variance_repr())
                              ->assert_is_op_input("batch_norm", "Variance");
  auto *bn_mean_var = pattern->NewNode(bn_mean_repr())
                          ->assert_is_op_input("batch_norm", "Mean");

  auto *bn = pattern->NewNode(batch_norm_repr())
                 ->assert_is_op("batch_norm")
                 ->assert_is_not_op_input("MomentumTensor")
                 ->assert_op_attr<bool>("is_test", false)
                 ->assert_op_attr<bool>("use_global_stats", false)
                 ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->assert_is_op_output("batch_norm", "MeanOut");
  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->assert_is_op_output("batch_norm", "VarianceOut");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_output("batch_norm", "SavedVariance");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_output("batch_norm", "SavedMean");
  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_output("batch_norm", "ReserveSpace");
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->assert_is_op_output("batch_norm", "Y")
                         ->assert_has_n_outputs(1);

  bn_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  bn->LinksFrom(
        {bn_x_var, bn_scale_var, bn_bias_var, bn_variance_var, bn_mean_var})
      .LinksTo({bn_mean_out_var,
                bn_variance_out_var,
                bn_saved_variance_var,
                bn_saved_mean_var,
                bn_reserve_space,
                bn_out_var});
  act->LinksFrom({bn_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::BatchNormActGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_grad_types) {
  auto *act_grad =
      pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
  auto *bn_grad = pattern->NewNode(batch_norm_grad_repr())
                      ->assert_is_op("batch_norm_grad")
                      ->assert_op_attr<bool>("use_global_stats", false)
                      ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_ops_input(act_grad_types, "Out");
  auto *d_intermediate_var =
      pattern->NewNode(d_intermediate_out_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"))
          ->assert_has_n_outputs(1);
  auto *bn_x_var = pattern->NewNode(bn_x_repr())
                       ->assert_is_op_input("batch_norm_grad", "X")
                       ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm_grad", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm_grad", "Bias");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedMean");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedVariance");
  // ReserveSpace as the output is equal to:
  // data_layout == 'NHWC' && FLAGS_cudnn_batchnorm_spatial_persistent == true
  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_input("batch_norm_grad", "ReserveSpace");
  auto *d_bn_x_var =
      pattern->NewNode(d_bn_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("X"));
  auto *d_bn_scale_var =
      pattern->NewNode(d_bn_scale_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
  auto *d_bn_bias_var =
      pattern->NewNode(d_bn_bias_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));

  act_grad->LinksFrom({d_act_out_var, act_out_var})
      .LinksTo({d_intermediate_var});

  bn_grad
      ->LinksFrom({bn_x_var,
                   d_intermediate_var,
                   bn_scale_var,
                   bn_bias_var,
                   bn_saved_mean_var,
                   bn_saved_variance_var,
                   bn_reserve_space})
      .LinksTo({d_bn_x_var, d_bn_scale_var, d_bn_bias_var});

  return bn_grad;
}

PDNode *patterns::BatchNormActOneDNN::operator()(const std::string &act_type) {
  auto *bn_x = pattern->NewNode(bn_in_repr())
                   ->AsInput()
                   ->assert_is_op_input("batch_norm", "X");
  auto *bn = pattern->NewNode(batch_norm_repr())->assert_is_op("batch_norm");
  auto *bn_out = pattern->NewNode(bn_out_repr())
                     ->assert_is_op_output("batch_norm", "Y")
                     ->assert_is_op_input(act_type);
  auto *act =
      pattern->NewNode(act_repr())->assert_is_op(act_type)->AsIntermediate();
  auto *act_out = pattern->NewNode(act_out_repr())
                      ->assert_is_op_output(act_type, "Out")
                      ->AsOutput();

  bn->LinksFrom({bn_x}).LinksTo({bn_out});
  act->LinksFrom({bn_out}).LinksTo({act_out});

  return act_out;
}

PDNode *patterns::BatchNormAddAct::operator()(
    paddle::framework::ir::PDNode *bn_x_var,
    std::unordered_set<std::string> act_types) {
  bn_x_var->assert_is_op_input("batch_norm", "X")
      ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm", "Bias");

  auto *bn = pattern->NewNode(batch_norm_repr())
                 ->assert_is_op("batch_norm")
                 ->assert_is_not_op_input("MomentumTensor")
                 ->assert_op_attr<bool>("is_test", false)
                 ->assert_op_attr<bool>("use_global_stats", false)
                 ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->assert_is_op_output("batch_norm", "MeanOut");
  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->assert_is_op_output("batch_norm", "VarianceOut");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_output("batch_norm", "SavedVariance");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_output("batch_norm", "SavedMean");
  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_output("batch_norm", "ReserveSpace");
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->assert_is_op_output("batch_norm", "Y")
                         ->assert_var_dtype(proto::VarType::FP16);

  bn_out_var->assert_is_op_input("elementwise_add");

  auto *elewise_add =
      pattern->NewNode(elewise_add_repr())->assert_is_op("elementwise_add");

  auto *elewise_add_in_var = pattern->NewNode(elewise_add_in_repr())
                                 ->assert_is_not_ctrl_var()
                                 ->assert_is_op_input("elementwise_add")
                                 ->assert_var_dtype(proto::VarType::FP16);

  auto *elewise_add_out_var =
      pattern->NewNode(elewise_add_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_has_n_outputs(1);

  elewise_add_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  bn->LinksFrom({bn_x_var, bn_scale_var, bn_bias_var})
      .LinksTo({bn_mean_out_var,
                bn_variance_out_var,
                bn_saved_variance_var,
                bn_saved_mean_var,
                bn_reserve_space,
                bn_out_var});
  elewise_add->LinksFrom({elewise_add_in_var, bn_out_var})
      .LinksTo({elewise_add_out_var});
  act->LinksFrom({elewise_add_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::BatchNormAddActGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_grad_types) {
  auto *act_grad =
      pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
  auto *elewise_add_grad = pattern->NewNode(elewise_add_grad_repr())
                               ->assert_is_op("elementwise_add_grad");
  auto *bn_grad = pattern->NewNode(batch_norm_grad_repr())
                      ->assert_is_op("batch_norm_grad")
                      ->assert_op_attr<bool>("use_global_stats", false)
                      ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_ops_input(act_grad_types, "Out");
  auto *d_act_x_var =
      pattern->NewNode(d_act_x_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"))
          ->assert_has_n_outputs(1);  // d_act_x

  d_act_x_var->AsIntermediate()->assert_is_op_input("elementwise_add_grad");

  auto *d_elewise_add_in_var =
      pattern->NewNode(d_elewise_add_in_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad")
          ->assert_var_dtype(proto::VarType::FP16);  // d_add_in_1
  auto *d_bn_out_var =
      pattern->NewNode(d_bn_out_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad")
          ->assert_var_dtype(proto::VarType::FP16);  // d_add_in_2

  d_bn_out_var->assert_is_op_input("batch_norm_grad", GradVarName("Y"));

  auto *bn_x_var = pattern->NewNode(bn_x_repr())
                       ->assert_is_op_input("batch_norm_grad", "X")
                       ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm_grad", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm_grad", "Bias");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedMean");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedVariance");

  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_input("batch_norm_grad", "ReserveSpace");
  auto *d_bn_x_var =
      pattern->NewNode(d_bn_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("X"))
          ->assert_var_dtype(proto::VarType::FP16);
  auto *d_bn_scale_var =
      pattern->NewNode(d_bn_scale_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
  auto *d_bn_bias_var =
      pattern->NewNode(d_bn_bias_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));

  act_grad->LinksFrom({d_act_out_var, act_out_var}).LinksTo({d_act_x_var});

  elewise_add_grad->LinksFrom({d_act_x_var})
      .LinksTo({d_elewise_add_in_var, d_bn_out_var});

  bn_grad
      ->LinksFrom({bn_x_var,
                   d_bn_out_var,
                   bn_scale_var,
                   bn_bias_var,
                   bn_saved_mean_var,
                   bn_saved_variance_var,
                   bn_reserve_space})
      .LinksTo({d_bn_x_var, d_bn_scale_var, d_bn_bias_var});

  return bn_grad;
}

PDNode *patterns::ElewiseAddActInplaceGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_types) {
  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  auto *act_grad = pattern->NewNode(act_grad_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_input(act_types, "Out");

  auto *d_intermediate_var =
      pattern->NewNode(d_intermediate_out_repr())
          ->assert_is_ops_output(act_types, GradVarName("X"));

  act_grad->LinksFrom({d_act_out_var, act_out_var})
      .LinksTo({d_intermediate_var});

  auto *ele_y_var = pattern->NewNode(ele_y_repr())
                        ->assert_is_not_ctrl_var()
                        ->assert_is_op_input("elementwise_add_grad", "Y");

  auto *ele_add_grad = pattern->NewNode(ele_add_grad_repr())
                           ->assert_is_op("elementwise_add_grad");

  auto *d_ele_x_var =
      pattern->NewNode(d_ele_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));

  auto *d_ele_y_var =
      pattern->NewNode(d_ele_y_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));

  ele_add_grad->LinksFrom({d_intermediate_var, ele_y_var})
      .LinksTo({d_ele_x_var, d_ele_y_var});

  return ele_add_grad;
}

PDNode *patterns::ActElewiseAddInplaceGrad::operator()(
    paddle::framework::ir::PDNode *d_out_var,
    std::unordered_set<std::string> act_types) {
  VLOG(4) << "ActElewiseAddInplaceGrad::operator";

  auto *ele_add_grad_op = pattern->NewNode(ele_add_grad_op_repr())
                              ->assert_is_op("elementwise_add_grad");
  auto *act_grad_op =
      pattern->NewNode(act_grad_op_repr())->assert_is_ops(act_types);

  auto *d_intermediate_out_var =
      pattern->NewNode(d_intermediate_var_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"))
          ->assert_is_ops_input(act_types, GradVarName("Out"));
  auto *intermediate_out_var =
      pattern->NewNode(intermediate_var_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y")
          ->assert_is_ops_input(act_types, "Out");

  ele_add_grad_op->LinksFrom({d_out_var});
  d_intermediate_out_var->LinksFrom({ele_add_grad_op}).LinksTo({act_grad_op});
  intermediate_out_var->LinksTo({ele_add_grad_op});
  intermediate_out_var->LinksTo({act_grad_op});

  return act_grad_op;
}

PDNode *patterns::ElewiseAddAct::operator()(
    paddle::framework::ir::PDNode *ele_x_var,
    std::unordered_set<std::string> act_types) {
  auto *ele_y_var = pattern->NewNode(ele_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y");

  auto *ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *ele_out_var = pattern->NewNode(elewise_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");

  ele_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  ele_add->LinksFrom({ele_x_var, ele_y_var}).LinksTo({ele_out_var});
  act->LinksFrom({ele_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::LinearAct::operator()(
    paddle::framework::ir::PDNode *linear_x_var,
    const std::unordered_set<std::string> &act_types,
    bool with_grad_link,
    bool is_act_grad_x_from_act) {
  auto *matmul_w_var =
      pattern->NewNode(matmul_w_repr())->assert_is_op_input("matmul_v2", "Y");

  auto *matmul = pattern->NewNode(matmul_repr())->assert_is_op("matmul_v2");

  auto *matmul_out_var = pattern->NewNode(matmul_out_repr())
                             ->assert_is_op_output("matmul_v2", "Out");

  matmul_out_var->AsIntermediate()->assert_is_op_input("elementwise_add", "X");

  auto *ele_bias_var = pattern->NewNode(ele_bias_repr())
                           ->assert_is_op_input("elementwise_add", "Y");

  auto *ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *ele_out_var = pattern->NewNode(elewise_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");

  matmul->LinksFrom({linear_x_var, matmul_w_var}).LinksTo({matmul_out_var});
  ele_add->LinksFrom({matmul_out_var, ele_bias_var}).LinksTo({ele_out_var});

  if (with_grad_link) {
    matmul_out_var->assert_is_op_input("elementwise_add_grad", "X");
    auto *elementwise_add_grad_op = pattern->NewNode("elementwise_add_grad")
                                        ->assert_is_op("elementwise_add_grad");
    elementwise_add_grad_op->LinksFrom({matmul_out_var});
  }

  if (!act_types.empty()) {
    ele_out_var->AsIntermediate()->assert_is_ops_input(act_types);

    auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);
    auto *act_out_var = pattern->NewNode(act_out_repr())
                            ->assert_is_ops_output(act_types, "Out");

    act->LinksFrom({ele_out_var}).LinksTo({act_out_var});

    if (with_grad_link && !is_act_grad_x_from_act) {
      std::unordered_set<std::string> act_grad_types;
      for (const auto &act : act_types) {
        std::string act_grad(act);
        act_grad.append("_grad");
        act_grad_types.insert(act_grad);
      }

      ele_out_var->assert_is_ops_input(act_grad_types, "X");
      auto *act_grad_op =
          pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
      act_grad_op->LinksFrom({ele_out_var});
    }

    return act_out_var;
  }

  return ele_out_var;
}

PDNode *patterns::ElewiseAddMatmulAct::operator()(
    paddle::framework::ir::PDNode *dout_var,
    const std::unordered_set<std::string> &act_grad_types,
    bool without_x_gradient,
    bool is_act_grad_x_from_act) {
  auto *ele_grad_bias_var =
      pattern->NewNode(ele_grad_bias_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y");
  auto *ele_add_grad = pattern->NewNode(ele_add_grad_repr())
                           ->assert_is_op("elementwise_add_grad");
  auto *ele_grad_dx_var =
      pattern->NewNode(ele_grad_dx_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));
  auto *ele_grad_dbias_var =
      pattern->NewNode(ele_grad_dbias_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));
  ele_add_grad->LinksFrom({dout_var, ele_grad_bias_var})
      .LinksTo({ele_grad_dx_var, ele_grad_dbias_var});

  ele_grad_dx_var->AsIntermediate()->assert_is_op_input("matmul_v2_grad",
                                                        GradVarName("Out"));

  auto *matmul_grad_x_var = pattern->NewNode(matmul_grad_x_repr())
                                ->assert_is_op_input("matmul_v2_grad", "X");
  auto *matmul_grad_w_var = pattern->NewNode(matmul_grad_w_repr())
                                ->assert_is_op_input("matmul_v2_grad", "Y");
  auto *matmul_grad =
      pattern->NewNode(matmul_grad_repr())->assert_is_op("matmul_v2_grad");
  auto *matmul_grad_dx_var =
      pattern->NewNode(matmul_grad_dx_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("X"));
  auto *matmul_grad_dw_var =
      pattern->NewNode(matmul_grad_dw_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("Y"));
  matmul_grad->LinksFrom(
      {ele_grad_dx_var, matmul_grad_x_var, matmul_grad_w_var});
  if (without_x_gradient) {
    matmul_grad->LinksTo({matmul_grad_dw_var});
  } else {
    matmul_grad->LinksTo({matmul_grad_dx_var, matmul_grad_dw_var});
  }

  if (!without_x_gradient && !act_grad_types.empty()) {
    matmul_grad_dx_var->AsIntermediate()->assert_is_ops_input(
        act_grad_types, GradVarName("Out"));

    auto *act_grad =
        pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
    auto *act_grad_dx_var =
        pattern->NewNode(act_grad_dx_repr())
            ->assert_is_ops_output(act_grad_types, GradVarName("X"));

    auto *act_grad_x_var = matmul_grad_x_var;
    if (!is_act_grad_x_from_act) {
      auto *ele_out_var = pattern->NewNode(ele_out_repr())
                              ->assert_is_ops_input(act_grad_types, "X");
      act_grad_x_var = ele_out_var;
    }

    act_grad->LinksFrom({matmul_grad_dx_var, act_grad_x_var})
        .LinksTo({act_grad_dx_var});
    return act_grad;
  }

  return matmul_grad;
}

// conv_type: conv2d, conv3d, conv2d_transpose
PDNode *patterns::ConvBias::operator()(
    paddle::framework::ir::PDNode *conv_input, std::string conv_type) {
  // Create Operators
  conv_input->assert_is_op_input(conv_type, "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op(conv_type);
  auto *eltiwse_op =
      pattern->NewNode(eltwise_repr())->assert_is_op("elementwise_add");
  // Create variables
  // Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_op_input(conv_type, "Filter");
  // intermediate variable, will be removed in the IR after fuse.
  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op(conv_type)
                           ->assert_is_op_input("elementwise_add");
  // Bias stored in elementwise_add
  auto *eltwise_bias_var = pattern->NewNode(eltwise_bias_repr())
                               ->AsInput()
                               ->assert_is_persistable_var()
                               ->assert_is_op_input("elementwise_add", "Y");
  // output
  auto *eltwise_out_var = pattern->NewNode(eltwise_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("elementwise_add");
  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});
  eltiwse_op->LinksFrom({conv_out_var, eltwise_bias_var})
      .LinksTo({eltwise_out_var});
  return eltwise_out_var;
}

PDNode *patterns::Conv::operator()(const std::string &conv_type) {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op(conv_type);

  auto input_var = pattern->NewNode(conv_input_repr())
                       ->AsInput()
                       ->assert_is_op_input(conv_type, "Input");

  auto filter_var = pattern->NewNode(conv_filter_repr())
                        ->AsInput()
                        ->assert_is_op_input(conv_type, "Filter");

  auto output_var = pattern->NewNode(conv_output_repr())
                        ->AsOutput()
                        ->assert_is_op_output(conv_type, "Output");

  conv_op->LinksFrom({input_var, filter_var}).LinksTo({output_var});
  return output_var;
}

PDNode *patterns::Immutable::operator()(const std::string &immutable_type,
                                        const std::string &input_name) {
  auto prev_op = pattern->NewNode(prev_op_repr())->assert_is_op();

  auto immutable_op =
      pattern->NewNode(immutable_op_repr())->assert_is_op(immutable_type);

  auto immutable_in = pattern->NewNode(immutable_in_repr())
                          ->AsInput()
                          ->assert_is_op_input(immutable_type, input_name);
  auto immutable_out = pattern->NewNode(immutable_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output(immutable_type, "Out");

  prev_op->LinksTo({immutable_in});
  immutable_op->LinksFrom({immutable_in}).LinksTo({immutable_out});
  return immutable_out;
}

PDNode *patterns::Matmul::operator()() {
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");

  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y = pattern->NewNode(matmul_in_y_repr())
                         ->AsInput()
                         ->assert_is_persistable_var()
                         ->assert_is_op_input("matmul", "Y");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

// MatmulV2: tensor * weight
PDNode *patterns::MatmulV2Weight::operator()() {
  auto matmul_v2_op =
      pattern->NewNode(matmul_v2_op_repr())->assert_is_op("matmul_v2");

  auto matmul_v2_in_x = pattern->NewNode(matmul_v2_in_x_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "X");
  auto matmul_v2_in_y = pattern->NewNode(matmul_v2_in_y_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()  // Y is weight
                            ->assert_is_op_input("matmul_v2", "Y");
  auto matmul_v2_out = pattern->NewNode(matmul_v2_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("matmul_v2", "Out");

  matmul_v2_op->LinksFrom({matmul_v2_in_x, matmul_v2_in_y})
      .LinksTo({matmul_v2_out});
  return matmul_v2_out;
}

// MatmulV2: tensor * tensor or tensor * weight
PDNode *patterns::MatmulV2::operator()() {
  auto matmul_v2_op =
      pattern->NewNode(matmul_v2_op_repr())->assert_is_op("matmul_v2");

  auto matmul_v2_in_x = pattern->NewNode(matmul_v2_in_x_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "X");
  auto matmul_v2_in_y = pattern->NewNode(matmul_v2_in_y_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto matmul_v2_out = pattern->NewNode(matmul_v2_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("matmul_v2", "Out");

  matmul_v2_op->LinksFrom({matmul_v2_in_x, matmul_v2_in_y})
      .LinksTo({matmul_v2_out});
  return matmul_v2_out;
}

PDNode *patterns::MatmulScale::operator()() {
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y = pattern->NewNode(matmul_in_y_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "Y");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");
  auto scale_in_x = pattern->NewNode(scale_in_x_repr())
                        ->assert_is_op_output("matmul", "Out")
                        ->assert_is_op_input("scale", "X");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("scale", "Out");
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({scale_in_x});
  scale_op->LinksFrom({scale_in_x}).LinksTo({scale_out});
  return scale_out;
}

PDNode *patterns::MatmulV2Scale::operator()() {
  auto matmul_v2_op =
      pattern->NewNode(matmul_v2_op_repr())->assert_is_op("matmul_v2");
  auto matmul_v2_in_x = pattern->NewNode(matmul_v2_in_x_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "X");
  auto matmul_v2_in_y = pattern->NewNode(matmul_v2_in_y_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()  // Y is weight
                            ->assert_is_op_input("matmul_v2", "Y");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");
  auto scale_in_x = pattern->NewNode(scale_in_x_repr())
                        ->assert_is_op_output("matmul_v2", "Out")
                        ->assert_is_op_input("scale", "X");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("scale", "Out");
  matmul_v2_op->LinksFrom({matmul_v2_in_x, matmul_v2_in_y})
      .LinksTo({scale_in_x});
  scale_op->LinksFrom({scale_in_x}).LinksTo({scale_out});
  return scale_out;
}

PDNode *patterns::Squeeze2Matmul::operator()() {
  auto squeeze2_in_x = pattern->NewNode(squeeze2_in_x_repr())
                           ->assert_is_op_input("squeeze2", "X")
                           ->AsInput();
  auto squeeze2_op =
      pattern->NewNode(squeeze2_op_repr())->assert_is_op("squeeze2");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->assert_is_op_output("squeeze2", "Out")
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y =
      pattern->NewNode(matmul_in_y_repr())->assert_is_op_input("matmul", "Y");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  squeeze2_op->LinksFrom({squeeze2_in_x}).LinksTo({matmul_in_x});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::Reshape2Matmul::operator()() {
  auto reshape2_in_x = pattern->NewNode(reshape2_in_x_repr())
                           ->assert_is_op_input("reshape2", "X")
                           ->AsInput();
  auto reshape2_op =
      pattern->NewNode(reshape2_op_repr())->assert_is_op("reshape2");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->assert_is_op_output("reshape2", "Out")
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y =
      pattern->NewNode(matmul_in_y_repr())->assert_is_op_input("matmul", "Y");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  reshape2_op->LinksFrom({reshape2_in_x}).LinksTo({matmul_in_x});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::FusedMatmul::operator()(bool with_residual) {
  auto matmul_op =
      pattern->NewNode(matmul_op_repr())->assert_is_op("fused_matmul");

  if (!with_residual) {
    matmul_op->assert_more([&](Node *x) {
      return (!HasInput(x, "ResidualData") ||
              x->Op()->Input("ResidualData").empty());
    });
  }

  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->AsInput()
                         ->assert_is_op_input("fused_matmul", "X");
  auto matmul_in_y = pattern->NewNode(matmul_in_y_repr())
                         ->AsInput()
                         ->assert_is_op_input("fused_matmul", "Y");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("fused_matmul", "Out")
                        ->assert_is_only_output_of_op("fused_matmul");
  std::vector<PDNode *> links_from{matmul_in_x, matmul_in_y};

  if (with_residual) {
    auto matmul_residual_data =
        pattern->NewNode(matmul_residual_data_repr())
            ->AsInput()
            ->assert_is_op_input("fused_matmul", "ResidualData");
    links_from.push_back(matmul_residual_data);
  }

  matmul_op->LinksFrom(links_from).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::Flatten2Matmul::operator()() {
  auto flatten2_in_x = pattern->NewNode(flatten2_in_x_repr())
                           ->assert_is_op_input("flatten2", "X")
                           ->AsInput();
  auto flatten2_op =
      pattern->NewNode(flatten2_op_repr())->assert_is_op("flatten2");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->assert_is_op_output("flatten2", "Out")
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y =
      pattern->NewNode(matmul_in_y_repr())->assert_is_op_input("matmul", "Y");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  flatten2_op->LinksFrom({flatten2_in_x}).LinksTo({matmul_in_x});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::ConvResidual::operator()(const std::string &conv_type,
                                           bool with_residual_data) {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op(conv_type);

  if (!with_residual_data) {
    conv_op->assert_more([&](Node *x) {
      if (!HasInput(x, "ResidualData") ||
          x->Op()->Input("ResidualData").empty())
        return true;
      return false;
    });
  }

  auto input_var = pattern->NewNode(conv_input_repr())
                       ->AsInput()
                       ->assert_is_op_input(conv_type, "Input");

  auto filter_var = pattern->NewNode(conv_filter_repr())
                        ->AsInput()
                        ->assert_is_op_input(conv_type, "Filter");

  auto output_var = pattern->NewNode(conv_output_repr())
                        ->AsOutput()
                        ->assert_is_op_output(conv_type, "Output");

  std::vector<PDNode *> links_from{input_var, filter_var};

  if (with_residual_data) {
    auto res_conn_var = pattern->NewNode(conv_residual_data_repr())
                            ->AsInput()
                            ->assert_is_op_input(conv_type, "ResidualData");
    links_from.push_back(res_conn_var);
  }

  conv_op->LinksFrom(links_from).LinksTo({output_var});
  return output_var;
}

PDNode *patterns::Pool::operator()() {
  auto pool_op = pattern->NewNode(pool_op_repr())->assert_is_op("pool2d");

  auto input_var = pattern->NewNode(pool_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("pool2d", "X");

  auto output_var = pattern->NewNode(pool_output_repr())
                        ->AsOutput()
                        ->assert_is_op_output("pool2d", "Out");

  pool_op->LinksFrom({input_var}).LinksTo({output_var});
  return output_var;
}

PDNode *patterns::Elementwise::operator()(PDNode *x_var,
                                          PDNode *y_var,
                                          const std::string &elementwise_type) {
  auto elementwise_op =
      pattern->NewNode(elementwise_op_repr())->assert_is_op(elementwise_type);

  x_var->AsInput()->assert_is_op_input(elementwise_type, "X");
  y_var->AsInput()->assert_is_op_input(elementwise_type, "Y");
  auto out_var = pattern->NewNode(elementwise_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output(elementwise_type, "Out");

  elementwise_op->LinksFrom({x_var, y_var});
  elementwise_op->LinksTo({out_var});

  return out_var;
}

PDNode *patterns::ElementwiseOp::operator()(
    const std::string &elementwise_type) {
  auto elementwise_op =
      pattern->NewNode(elementwise_op_repr())->assert_is_op(elementwise_type);

  auto out_var = pattern->NewNode(elementwise_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output(elementwise_type, "Out");

  elementwise_op->LinksTo({out_var});

  return out_var;
}

PDNode *patterns::MatmulElementwiseAdd::operator()(
    const std::string &matmul_type, bool as_x) {
  auto matmul_op =
      pattern->NewNode(matmul_op_repr())->assert_is_op(matmul_type);
  auto matmul_out =
      pattern->NewNode(matmul_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output(matmul_type, "Out")
          ->assert_is_only_output_of_op(matmul_type)
          ->assert_is_op_input("elementwise_add", as_x ? "X" : "Y");
  auto elementwise_addend =
      pattern->NewNode(elementwise_addend_repr())
          ->AsInput()
          ->assert_is_op_input("elementwise_add", as_x ? "Y" : "X");
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_out =
      pattern->NewNode(elementwise_add_out_repr())
          ->AsOutput()
          ->assert_is_op_output("elementwise_add", "Out");

  matmul_op->LinksTo({matmul_out});
  elementwise_add_op->LinksFrom({matmul_out, elementwise_addend})
      .LinksTo({elementwise_add_out});
  return elementwise_add_out;
}

PDNode *patterns::ResidualElementwise::operator()(
    PDNode *op_var,
    PDNode *residual_var,
    const std::string &elementwise_type,
    bool as_x) {
  auto elementwise_op =
      pattern->NewNode(elementwise_op_repr())->assert_is_op(elementwise_type);

  if (as_x) {
    op_var->AsInput()->assert_is_op_input(elementwise_type, "X");
    residual_var->AsInput()->assert_is_op_input(elementwise_type, "Y");
  } else {
    op_var->AsInput()->assert_is_op_input(elementwise_type, "Y");
    residual_var->AsInput()->assert_is_op_input(elementwise_type, "X");
  }
  auto out_var = pattern->NewNode(elementwise_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output(elementwise_type, "Out");

  elementwise_op->LinksFrom({op_var, residual_var});
  elementwise_op->LinksTo({out_var});

  return out_var;
}

PDNode *patterns::Concat::operator()() {
  auto concat_op = pattern->NewNode(concat_op_repr())->assert_is_op("concat");

  auto output_var = pattern->NewNode(concat_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("concat", "Out");

  concat_op->LinksTo({output_var});
  return output_var;
}

PDNode *patterns::OpRequant::operator()() {
  auto any_op = pattern->NewNode(any_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      return (node->Op()->HasAttr("Scale_out") ||
                              node->Op()->HasAttr("scale_out"));
                    });
  auto requant_in = pattern->NewNode(requant_in_repr())
                        ->assert_is_op_input("requantize", "Input");
  auto requant_op =
      pattern->NewNode(requant_op_repr())->assert_is_op("requantize");
  auto requant_out = pattern->NewNode(requant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("requantize", "Output");

  any_op->LinksTo({requant_in});
  requant_op->LinksFrom({requant_in}).LinksTo({requant_out});
  return requant_out;
}

PDNode *patterns::RequantOp::operator()() {
  auto requant_in = pattern->NewNode(requant_in_repr())
                        ->assert_is_op_input("requantize", "Input");
  auto requant_op =
      pattern->NewNode(requant_op_repr())->assert_is_op("requantize");
  auto requant_out = pattern->NewNode(requant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("requantize", "Output");
  auto any_op = pattern->NewNode(any_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      return (node->Op()->HasAttr("Scale_in") ||
                              node->Op()->HasAttr("Scale_x") ||
                              node->Op()->HasAttr("Scale_y") ||
                              node->Op()->HasAttr("scale_in") ||
                              node->Op()->HasAttr("scale_x") ||
                              node->Op()->HasAttr("scale_y"));
                    });

  requant_op->LinksFrom({requant_in}).LinksTo({requant_out});
  any_op->LinksFrom({requant_out});
  return any_op;
}

PDNode *patterns::OpDequant::operator()() {
  auto any_op = pattern->NewNode(any_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      return (node->Op()->HasAttr("force_fp32_output") ||
                              node->Op()->HasProtoAttr("force_fp32_output"));
                    });
  auto dequant_in = pattern->NewNode(dequant_in_repr())
                        ->assert_is_op_input("dequantize", "Input");
  auto dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");
  auto dequant_out = pattern->NewNode(dequant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("dequantize", "Output");

  any_op->LinksTo({dequant_in});
  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  return dequant_out;
}

PDNode *patterns::DequantScale::operator()() {
  // Create Operators
  auto dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");

  auto dequant_out = pattern->NewNode(dequant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("dequantize", "Output");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("scale", "Out");

  dequant_op->LinksTo({dequant_out});
  scale_op->LinksFrom({dequant_out}).LinksTo({scale_out});

  return scale_out;
}

PDNode *patterns::ScaleQuant::operator()() {
  auto scale_in = pattern->NewNode(scale_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("scale", "X");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");

  auto quant_in = pattern->NewNode(quant_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("quantize", "Input");
  auto quant_op = pattern->NewNode(quant_op_repr())->assert_is_op("quantize");

  scale_op->LinksFrom({scale_in}).LinksTo({quant_in});
  quant_op->LinksFrom({quant_in});

  return quant_op;
}

PDNode *patterns::QuantConv::operator()(const std::string &conv_type) {
  auto quant_in = pattern->NewNode(quant_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("quantize", "Input");
  auto quant_op = pattern->NewNode(quant_op_repr())->assert_is_op("quantize");

  auto conv_in = pattern->NewNode(conv_in_repr())
                     ->AsInput()
                     ->assert_is_op_input(conv_type, "Input");
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op(conv_type);
  conv_op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });

  quant_op->LinksFrom({quant_in}).LinksTo({conv_in});
  conv_op->LinksFrom({conv_in});

  return quant_op;
}

PDNode *patterns::ScaleMatmul::operator()() {
  auto scale_in = pattern->NewNode(scale_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("scale", "X");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("scale", "Out");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");

  scale_op->LinksFrom({scale_in}).LinksTo({scale_out});
  matmul_op->LinksFrom({scale_out});
  return matmul_op;
}

PDNode *patterns::PriorBox::operator()() {
  auto prior_box_op =
      pattern->NewNode(prior_box_op_repr())->assert_is_op("prior_box");

  auto input_var = pattern->NewNode(prior_box_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("prior_box", "Input");

  auto image_var = pattern->NewNode(prior_box_image_repr())
                       ->AsInput()
                       ->assert_is_op_input("prior_box", "Image");

  auto boxes_var = pattern->NewNode(prior_box_boxes_repr())
                       ->AsOutput()
                       ->assert_is_op_output("prior_box", "Boxes");

  auto variances_var = pattern->NewNode(prior_box_variances_repr())
                           ->AsOutput()
                           ->assert_is_op_output("prior_box", "Variances");

  prior_box_op->LinksFrom({input_var, image_var})
      .LinksTo({boxes_var, variances_var});
  return boxes_var;
}

PDNode *patterns::ConvElementwiseaddAct::operator()(
    PDNode *conv_in, const std::unordered_set<std::string> &conv_act_set) {
  conv_in->AsInput();
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d", "Filter")
                         ->AsInput();
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y = pattern->NewNode(elementwise_add_in_y_repr())
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->AsInput();
  auto elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                 ->assert_is_op_output("elementwise_add")
                                 ->AsIntermediate();

  auto act_op = pattern->NewNode(act_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      auto op_type = node->Name();
                      return conv_act_set.count(op_type);
                    });

  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_var()
                     // is activation op's output.
                     ->assert_more([&](Node *node) {
                       for (auto *in_op : node->inputs) {
                         if (conv_act_set.count(in_op->Name())) {
                           return true;
                         }
                       }
                       return false;
                     })
                     ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter});
  conv_out->LinksFrom({conv_op});
  elementwise_add_op->LinksFrom({conv_out, elementwise_add_in_y})
      .LinksTo({elementwise_add_out});
  act_op->LinksFrom({elementwise_add_out}).LinksTo({act_out});

  return act_out;
}

PDNode *patterns::DotProductAttention::operator()(bool with_dropout) {
  // Attention Computing
  auto *attn_q = pattern->NewNode(attn_q_repr())
                     ->AsInput()
                     ->assert_is_op_output("reshape2", "Out")
                     ->assert_is_op_input("transpose2", "X");
  auto *attn_k = pattern->NewNode(attn_k_repr())
                     ->AsInput()
                     ->assert_is_op_output("reshape2", "Out")
                     ->assert_is_op_input("transpose2", "X");
  auto *attn_v = pattern->NewNode(attn_v_repr())
                     ->AsInput()
                     ->assert_is_op_output("reshape2", "Out")
                     ->assert_is_op_input("transpose2", "X");

  auto *attn_q_transpose =
      pattern->NewNode(attn_q_transpose_repr())->assert_is_op("transpose2");
  auto *attn_k_transpose =
      pattern->NewNode(attn_k_transpose_repr())->assert_is_op("transpose2");
  auto *attn_v_transpose =
      pattern->NewNode(attn_v_transpose_repr())->assert_is_op("transpose2");

  auto *attn_q_transpose_out_var =
      pattern->NewNode(attn_q_transpose_out_repr())
          ->assert_is_op_output("transpose2", "Out")
          ->assert_is_op_input("scale", "X");
  auto *attn_k_transpose_out_var =
      pattern->NewNode(attn_k_transpose_out_repr())
          ->assert_is_op_output("transpose2", "Out")
          ->assert_is_op_input("matmul_v2", "Y");
  auto *attn_v_transpose_out_var =
      pattern->NewNode(attn_v_transpose_out_repr())
          ->assert_is_op_output("transpose2", "Out")
          ->assert_is_op_input("matmul_v2", "Y");
  auto *attn_q_transpose_xshape_var =
      pattern->NewNode(attn_q_transpose_xshape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto *attn_k_transpose_xshape_var =
      pattern->NewNode(attn_k_transpose_xshape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto *attn_v_transpose_xshape_var =
      pattern->NewNode(attn_v_transpose_xshape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  attn_q_transpose->LinksFrom({attn_q}).LinksTo(
      {attn_q_transpose_out_var, attn_q_transpose_xshape_var});
  attn_k_transpose->LinksFrom({attn_k}).LinksTo(
      {attn_k_transpose_out_var, attn_k_transpose_xshape_var});
  attn_v_transpose->LinksFrom({attn_v}).LinksTo(
      {attn_v_transpose_out_var, attn_v_transpose_xshape_var});

  auto *attn_q_scale =
      pattern->NewNode(attn_q_scale_repr())->assert_is_op("scale");
  auto *attn_q_scale_out_var = pattern->NewNode(attn_q_scale_out_repr())
                                   ->assert_is_op_output("scale", "Out")
                                   ->assert_is_op_input("matmul_v2", "X");
  attn_q_scale->LinksFrom({attn_q_transpose_out_var})
      .LinksTo({attn_q_scale_out_var});

  auto *attn_qk_matmul = pattern->NewNode(attn_qk_matmul_repr())
                             ->assert_is_op("matmul_v2")
                             ->assert_op_attr<bool>("trans_x", false)
                             ->assert_op_attr<bool>("trans_y", true);
  auto *attn_qk_matmul_out_var =
      pattern->NewNode(attn_qk_matmul_out_repr())
          ->assert_is_op_output("matmul_v2", "Out")
          ->assert_is_op_input("elementwise_add", "X");
  attn_qk_matmul->LinksFrom({attn_q_scale_out_var, attn_k_transpose_out_var})
      .LinksTo({attn_qk_matmul_out_var});

  auto *attn_mask_var = pattern->NewNode(attn_mask_repr())
                            ->assert_is_op_input("elementwise_add", "Y");

  auto *attn_mask_eleadd = pattern->NewNode(attn_mask_eleadd_repr())
                               ->assert_is_op("elementwise_add");
  auto *attn_mask_eleadd_out_var =
      pattern->NewNode(attn_mask_eleadd_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_is_op_input("softmax", "X");
  attn_mask_eleadd->LinksFrom({attn_mask_var, attn_qk_matmul_out_var})
      .LinksTo({attn_mask_eleadd_out_var});

  auto *attn_softmax =
      pattern->NewNode(attn_softmax_repr())->assert_is_op("softmax");
  auto *attn_softmax_out_var = pattern->NewNode(attn_softmax_out_repr())
                                   ->assert_is_op_output("softmax", "Out");
  attn_softmax->LinksFrom({attn_mask_eleadd_out_var})
      .LinksTo({attn_softmax_out_var});

  auto *attn_context_matmul_input = attn_softmax_out_var;
  if (with_dropout) {
    attn_softmax_out_var->assert_is_op_input("dropout", "X");
    auto *attn_dropout =
        pattern->NewNode(attn_dropout_repr())->assert_is_op("dropout");
    auto *attn_dropout_out_var = pattern->NewNode(attn_dropout_out_repr())
                                     ->assert_is_op_output("dropout", "Out")
                                     ->assert_is_op_input("matmul_v2", "X");
    auto *attn_dropout_mask_var = pattern->NewNode(attn_dropout_mask_repr())
                                      ->assert_is_op_output("dropout", "Mask");
    attn_dropout->LinksFrom({attn_softmax_out_var})
        .LinksTo({attn_dropout_out_var, attn_dropout_mask_var});
    attn_context_matmul_input = attn_dropout_out_var;
  } else {
    attn_softmax_out_var->assert_is_op_input("matmul_v2", "X");
  }

  auto *attn_context_matmul =
      pattern->NewNode(attn_context_matmul_repr())->assert_is_op("matmul_v2");
  auto *attn_context_matmul_out_var =
      pattern->NewNode(attn_context_matmul_out_repr())
          ->assert_is_op_output("matmul_v2", "Out")
          ->assert_is_op_input("transpose2", "X");
  attn_context_matmul
      ->LinksFrom({attn_context_matmul_input, attn_v_transpose_out_var})
      .LinksTo({attn_context_matmul_out_var});

  auto *attn_transpose =
      pattern->NewNode(attn_transpose_repr())->assert_is_op("transpose2");
  auto *attn_transpose_out_var = pattern->NewNode(attn_transpose_out_repr())
                                     ->assert_is_op_output("transpose2", "Out")
                                     ->assert_is_op_input("reshape2", "X");
  attn_transpose->LinksFrom({attn_context_matmul_out_var})
      .LinksTo({attn_transpose_out_var});
  auto *attn_transpose_xshape_var =
      pattern->NewNode(attn_transpose_xshape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  attn_transpose->LinksFrom({attn_context_matmul_out_var})
      .LinksTo({attn_transpose_out_var, attn_transpose_xshape_var});

  return attn_transpose_out_var;
}

PDNode *patterns::DotProductAttentionGrad::operator()(bool with_dropout) {
  auto *attn_dout_var =
      pattern->NewNode(attn_dout_repr())
          ->AsInput()
          ->assert_is_op_input("transpose2_grad", GradVarName("Out"))
          ->assert_is_op_output("reshape2_grad", GradVarName("X"));
  auto *attn_transpose_grad = pattern->NewNode(attn_transpose_grad_repr())
                                  ->assert_is_op("transpose2_grad");
  auto *attn_transpose_grad_out_var =
      pattern->NewNode(attn_transpose_grad_out_repr())
          ->assert_is_op_output("transpose2_grad", GradVarName("X"));
  attn_transpose_grad->LinksFrom({attn_dout_var})
      .LinksTo({attn_transpose_grad_out_var});

  attn_transpose_grad_out_var->assert_is_op_input("matmul_v2_grad",
                                                  GradVarName("Out"));
  auto *attn_context_matmul_grad_x_var =
      pattern->NewNode(attn_context_matmul_grad_x_repr())
          ->assert_is_op_input("matmul_v2_grad", "X");
  auto *attn_context_matmul_grad_y_var =
      pattern->NewNode(attn_context_matmul_grad_y_repr())
          ->assert_is_op_input("matmul_v2_grad", "Y");
  auto *attn_context_matmul_grad =
      pattern->NewNode(attn_context_matmul_grad_repr())
          ->assert_is_op("matmul_v2_grad");
  auto *attn_context_matmul_grad_dx_var =
      pattern->NewNode(attn_context_matmul_grad_dx_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("X"));
  auto *attn_context_matmul_grad_dy_var =
      pattern->NewNode(attn_context_matmul_grad_dy_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("Y"));
  attn_context_matmul_grad
      ->LinksFrom({attn_transpose_grad_out_var,
                   attn_context_matmul_grad_x_var,
                   attn_context_matmul_grad_y_var})
      .LinksTo(
          {attn_context_matmul_grad_dx_var, attn_context_matmul_grad_dy_var});

  PDNode *attn_softmax_grad_input = nullptr;
  PDNode *attn_softmax_out_var = nullptr;
  if (with_dropout) {
    auto *attn_dropout_grad = pattern->NewNode(attn_dropout_grad_repr())
                                  ->assert_is_op("dropout_grad");
    auto *attn_dropout_grad_out_var =
        pattern->NewNode(attn_dropout_grad_out_repr())
            ->assert_is_op_output("dropout_grad", GradVarName("X"));
    attn_context_matmul_grad_dx_var->assert_is_op_input("dropout_grad",
                                                        GradVarName("Out"));
    attn_dropout_grad->LinksFrom({attn_context_matmul_grad_dx_var})
        .LinksTo({attn_dropout_grad_out_var});
    attn_softmax_grad_input = attn_dropout_grad_out_var;
    attn_softmax_out_var = pattern->NewNode(attn_softmax_out_repr());

  } else {
    attn_context_matmul_grad_dx_var->assert_is_op_input("softmax_grad",
                                                        GradVarName("Out"));
    attn_softmax_grad_input = attn_context_matmul_grad_dx_var;
    attn_softmax_out_var = attn_context_matmul_grad_x_var;
  }
  attn_softmax_out_var->assert_is_op_input("softmax_grad", "Out");

  auto *attn_softmax_grad =
      pattern->NewNode(attn_softmax_grad_repr())->assert_is_op("softmax_grad");
  auto *attn_softmax_grad_out_var =
      pattern->NewNode(attn_softmax_grad_out_repr())
          ->assert_is_op_output("softmax_grad", GradVarName("X"));
  attn_softmax_grad->LinksFrom({attn_softmax_out_var, attn_softmax_grad_input})
      .LinksTo({attn_softmax_grad_out_var});

  attn_softmax_grad_out_var->assert_is_op_input("elementwise_add_grad",
                                                GradVarName("Out"));
  auto *attn_mask_eleadd_grad_mask_var =
      pattern->NewNode(attn_mask_eleadd_grad_mask_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y");
  auto *attn_mask_eleadd_grad = pattern->NewNode(attn_mask_eleadd_grad_repr())
                                    ->assert_is_op("elementwise_add_grad");
  auto *attn_mask_eleadd_grad_dx_var =
      pattern->NewNode(attn_mask_eleadd_grad_dx_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));
  attn_mask_eleadd_grad
      ->LinksFrom({attn_softmax_grad_out_var, attn_mask_eleadd_grad_mask_var})
      .LinksTo({attn_mask_eleadd_grad_dx_var});

  attn_mask_eleadd_grad_dx_var->assert_is_op_input("matmul_v2_grad",
                                                   GradVarName("Out"));
  auto *attn_qk_matmul_grad_x_var =
      pattern->NewNode(attn_qk_matmul_grad_x_repr())
          ->assert_is_op_input("matmul_v2_grad", "X");
  auto *attn_qk_matmul_grad_y_var =
      pattern->NewNode(attn_qk_matmul_grad_y_repr())
          ->assert_is_op_input("matmul_v2_grad", "Y");
  auto *attn_qk_matmul_grad = pattern->NewNode(attn_qk_matmul_grad_repr())
                                  ->assert_is_op("matmul_v2_grad");
  auto *attn_qk_matmul_grad_dx_var =
      pattern->NewNode(attn_qk_matmul_grad_dx_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("X"));
  auto *attn_qk_matmul_grad_dy_var =
      pattern->NewNode(attn_qk_matmul_grad_dy_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("Y"));
  attn_qk_matmul_grad
      ->LinksFrom({attn_mask_eleadd_grad_dx_var,
                   attn_qk_matmul_grad_x_var,
                   attn_qk_matmul_grad_y_var})
      .LinksTo({attn_qk_matmul_grad_dx_var, attn_qk_matmul_grad_dy_var});

  attn_qk_matmul_grad_dx_var->assert_is_op_input("scale", "X");
  auto *attn_scale_grad =
      pattern->NewNode(attn_scale_grad_repr())->assert_is_op("scale");
  auto *attn_scale_grad_out_var = pattern->NewNode(attn_scale_grad_out_repr())
                                      ->assert_is_op_output("scale", "Out");
  attn_scale_grad->LinksFrom({attn_qk_matmul_grad_dx_var})
      .LinksTo({attn_scale_grad_out_var});

  attn_scale_grad_out_var->assert_is_op_input("transpose2_grad",
                                              GradVarName("Out"));

  // q -> transpose2_grad -> reshape2_grad
  auto *attn_q_transpose_grad = pattern->NewNode(attn_q_transpose_grad_repr())
                                    ->assert_is_op("transpose2_grad");
  auto *attn_dq = pattern->NewNode(attn_dq_repr())
                      ->assert_is_op_output("transpose2_grad", GradVarName("X"))
                      ->assert_is_op_input("reshape2_grad", GradVarName("Out"));
  attn_q_transpose_grad->LinksFrom({attn_scale_grad_out_var})
      .LinksTo({attn_dq});

  // k -> transpose2_grad -> reshape2_grad
  attn_qk_matmul_grad_dy_var->assert_is_op_input("transpose2_grad",
                                                 GradVarName("Out"));
  auto *attn_k_transpose_grad = pattern->NewNode(attn_k_transpose_grad_repr())
                                    ->assert_is_op("transpose2_grad");
  auto *attn_dk = pattern->NewNode(attn_dk_repr())
                      ->assert_is_op_output("transpose2_grad", GradVarName("X"))
                      ->assert_is_op_input("reshape2_grad", GradVarName("Out"));
  attn_k_transpose_grad->LinksFrom({attn_qk_matmul_grad_dy_var})
      .LinksTo({attn_dk});

  // v -> transpose2_grad -> slice_grad
  attn_context_matmul_grad_dy_var->assert_is_op_input("transpose2_grad",
                                                      GradVarName("Out"));
  auto *attn_v_transpose_grad = pattern->NewNode(attn_v_transpose_grad_repr())
                                    ->assert_is_op("transpose2_grad");
  auto *attn_dv = pattern->NewNode(attn_dv_repr())
                      ->assert_is_op_output("transpose2_grad", GradVarName("X"))
                      ->assert_is_op_input("reshape2_grad", GradVarName("Out"));
  attn_v_transpose_grad->LinksFrom({attn_context_matmul_grad_dy_var})
      .LinksTo({attn_dv});

  return attn_dq;
}

PDNode *patterns::VitAttention::operator()(PDNode *in) {
  in->AsInput();
  std::unordered_set<std::string> matmul_ops{"matrix_multiply"};

  auto matmul0_op =
      pattern->NewNode(matmul0_op_repr())->assert_is_ops(matmul_ops);
  auto matmul0_in_y = pattern->NewNode(matmul0_in_y_repr())
                          ->AsInput()
                          ->assert_is_ops_input(matmul_ops, "Y");
  auto matmul0_out = pattern->NewNode(matmul0_out_repr())
                         ->assert_is_ops_output(matmul_ops, "Out")
                         ->assert_is_op_input("elementwise_add", "X")
                         ->AsIntermediate();

  auto elementwise0_op =
      pattern->NewNode(elementwise0_op_repr())->assert_is_op("elementwise_add");
  auto elementwise0_in_y = pattern->NewNode(elementwise0_in_y_repr())
                               ->AsInput()
                               ->assert_is_op_input("elementwise_add", "Y");
  auto elementwise0_out = pattern->NewNode(elementwise0_out_repr())
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->assert_is_op_input("reshape2", "X")
                              ->AsIntermediate();

  auto reshape1_op =
      pattern->NewNode(reshape1_op_repr())->assert_is_op("reshape2");
  auto reshape1_out = pattern->NewNode(reshape1_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->assert_is_op_input("transpose2", "X")
                          ->AsIntermediate();

  auto transpose1_op =
      pattern->NewNode(transpose1_op_repr())->assert_is_op("transpose2");
  auto transpose1_out = pattern->NewNode(transpose1_out_repr())
                            ->assert_is_op_output("transpose2", "Out")
                            ->assert_is_op_input("slice", "Input")
                            ->AsIntermediate();

  auto slice1_op = pattern->NewNode(slice1_op_repr())->assert_is_op("slice");
  auto slice1_out = pattern->NewNode(slice1_out_repr())
                        ->assert_is_op_output("slice", "Out")
                        ->assert_is_op_input("matrix_multiply", "Y")
                        ->AsIntermediate();

  auto slice2_op = pattern->NewNode(slice2_op_repr())->assert_is_op("slice");
  auto slice2_out = pattern->NewNode(slice2_out_repr())
                        ->assert_is_op_output("slice", "Out")
                        ->assert_is_op_input("matrix_multiply", "X")
                        ->AsIntermediate();

  auto slice3_op = pattern->NewNode(slice3_op_repr())->assert_is_op("slice");
  auto slice3_out = pattern->NewNode(slice3_out_repr())
                        ->assert_is_op_output("slice", "Out")
                        ->assert_is_op_input("transpose2", "X")
                        ->AsIntermediate();

  auto transpose2_op =
      pattern->NewNode(transpose2_op_repr())->assert_is_op("transpose2");
  auto transpose2_out = pattern->NewNode(transpose2_out_repr())
                            ->assert_is_op_output("transpose2", "Out")
                            ->assert_is_op_input("matrix_multiply", "Y")
                            ->AsIntermediate();

  auto matmul1_op =
      pattern->NewNode(matmul1_op_repr())->assert_is_op("matrix_multiply");
  auto matmul1_out = pattern->NewNode(matmul1_out_repr())
                         ->assert_is_op_output("matrix_multiply", "Out")
                         ->assert_is_op_input("scale", "X")
                         ->AsIntermediate();

  auto scale1_op = pattern->NewNode(scale1_op_repr())->assert_is_op("scale");
  auto scale1_out = pattern->NewNode(scale1_out_repr())
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input("softmax", "X")
                        ->AsIntermediate();

  auto softmax1_op =
      pattern->NewNode(softmax1_op_repr())->assert_is_op("softmax");
  auto softmax1_out = pattern->NewNode(softmax1_out_repr())
                          ->assert_is_op_output("softmax", "Out")
                          ->assert_is_op_input("matrix_multiply", "X")
                          ->AsIntermediate();

  auto matmul2_op =
      pattern->NewNode(matmul2_op_repr())->assert_is_op("matrix_multiply");
  auto matmul2_out = pattern->NewNode(matmul2_out_repr())
                         ->assert_is_op_output("matrix_multiply", "Out")
                         ->assert_is_op_input("transpose2", "X")
                         ->AsIntermediate();

  auto transpose3_op =
      pattern->NewNode(transpose3_op_repr())->assert_is_op("transpose2");
  auto transpose3_out = pattern->NewNode(transpose3_out_repr())
                            ->assert_is_op_output("transpose2", "Out")
                            ->assert_is_op_input("reshape2", "X")
                            ->AsIntermediate();

  auto reshape2_op =
      pattern->NewNode(reshape2_op_repr())->assert_is_op("reshape2");
  auto reshape2_out = pattern->NewNode(reshape2_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->AsOutput();

  matmul0_op->LinksFrom({in, matmul0_in_y});
  matmul0_out->LinksFrom({matmul0_op});

  elementwise0_op->LinksFrom({matmul0_out, elementwise0_in_y});
  elementwise0_out->LinksFrom({elementwise0_op});

  reshape1_op->LinksFrom({elementwise0_out});
  reshape1_out->LinksFrom({reshape1_op});

  transpose1_op->LinksFrom({reshape1_out});
  transpose1_out->LinksFrom({transpose1_op});

  slice1_op->LinksFrom({transpose1_out});
  slice1_out->LinksFrom({slice1_op});

  slice2_op->LinksFrom({transpose1_out});
  slice2_out->LinksFrom({slice2_op});

  slice3_op->LinksFrom({transpose1_out});
  slice3_out->LinksFrom({slice3_op});

  transpose2_op->LinksFrom({slice3_out});
  transpose2_out->LinksFrom({transpose2_op});

  matmul1_op->LinksFrom({slice2_out, transpose2_out});
  matmul1_out->LinksFrom({matmul1_op});

  scale1_op->LinksFrom({matmul1_out});
  scale1_out->LinksFrom({scale1_op});

  softmax1_op->LinksFrom({scale1_out});
  softmax1_out->LinksFrom({softmax1_op});

  matmul2_op->LinksFrom({slice1_out, softmax1_out});
  matmul2_out->LinksFrom({matmul2_op});

  transpose3_op->LinksFrom({matmul2_out});
  transpose3_out->LinksFrom({transpose3_op});

  reshape2_op->LinksFrom({transpose3_out});
  reshape2_out->LinksFrom({reshape2_op});

  return reshape2_out;
}

PDNode *patterns::SelfAttention::operator()(PDNode *in) {
  in->AsInput();

  std::unordered_set<std::string> matmul_ops{"matmul", "matmul_v2"};
  auto transpose2_0_op =
      pattern->NewNode(transpose2_0_op_repr())->assert_is_op("transpose2");
  auto transpose2_0_out = pattern->NewNode(transpose2_0_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input("slice", "Input")
                              ->AsIntermediate();
  auto slice_0_op = pattern->NewNode(slice_0_op_repr())->assert_is_op("slice");
  auto slice_0_out = pattern->NewNode(slice_0_out_repr())
                         ->assert_is_op_output("slice", "Out")
                         ->assert_is_ops_input(matmul_ops, "X")
                         ->AsIntermediate();
  auto slice_1_op = pattern->NewNode(slice_1_op_repr())->assert_is_op("slice");
  auto slice_1_out = pattern->NewNode(slice_1_out_repr())
                         ->assert_is_op_output("slice", "Out")
                         ->assert_is_op_input("transpose2", "X")
                         ->AsIntermediate();
  auto slice_2_op = pattern->NewNode(slice_2_op_repr())->assert_is_op("slice");
  auto slice_2_out = pattern->NewNode(slice_2_out_repr())
                         ->assert_is_op_output("slice", "Out")
                         ->assert_is_ops_input(matmul_ops, "Y")
                         ->AsIntermediate();
  auto matmul_0_op =
      pattern->NewNode(matmul_0_op_repr())->assert_is_ops(matmul_ops);
  auto matmul_0_out = pattern->NewNode(matmul_0_out_repr())
                          ->assert_is_ops_output(matmul_ops, "Out")
                          ->assert_is_op_input("transpose2", "X")
                          ->AsIntermediate();
  auto matmul_1_op =
      pattern->NewNode(matmul_1_op_repr())->assert_is_ops(matmul_ops);
  auto matmul_1_out = pattern->NewNode(matmul_1_out_repr())
                          ->assert_is_ops_output(matmul_ops, "Out")
                          ->assert_is_op_input("softmax", "X")
                          ->AsIntermediate();
  auto transpose2_1_op =
      pattern->NewNode(transpose2_1_op_repr())->assert_is_op("transpose2");
  auto transpose2_1_out = pattern->NewNode(transpose2_1_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_ops_input(matmul_ops, "Y")
                              ->AsIntermediate();
  auto softmax_op =
      pattern->NewNode(softmax_op_repr())->assert_is_op("softmax");
  auto softmax_out = pattern->NewNode(softmax_out_repr())
                         ->assert_is_op_output("softmax", "Out")
                         ->assert_is_ops_input(matmul_ops, "X")
                         ->AsIntermediate();
  auto transpose2_2_op =
      pattern->NewNode(transpose2_2_op_repr())->assert_is_op("transpose2");
  auto transpose2_2_out = pattern->NewNode(transpose2_2_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->AsOutput();
  transpose2_0_op->LinksFrom({in});
  transpose2_0_out->LinksFrom({transpose2_0_op});
  slice_0_op->LinksFrom({transpose2_0_out});
  slice_0_out->LinksFrom({slice_0_op});
  slice_1_op->LinksFrom({transpose2_0_out});
  slice_1_out->LinksFrom({slice_1_op});
  slice_2_op->LinksFrom({transpose2_0_out});
  slice_2_out->LinksFrom({slice_2_op});
  transpose2_1_op->LinksFrom({slice_1_out});
  transpose2_1_out->LinksFrom({transpose2_1_op});
  matmul_1_op->LinksFrom({slice_0_out, transpose2_1_out});
  matmul_1_out->LinksFrom({matmul_1_op});
  softmax_op->LinksFrom({matmul_1_out});
  softmax_out->LinksFrom({softmax_op});
  matmul_0_op->LinksFrom({softmax_out, slice_2_out});
  matmul_0_out->LinksFrom({matmul_0_op});
  transpose2_2_op->LinksFrom({matmul_0_out});
  transpose2_2_out->LinksFrom({transpose2_2_op});
  return transpose2_2_out;
}

PDNode *patterns::ConvElementwiseAdd2Act::operator()(
    PDNode *conv_in, const std::unordered_set<std::string> &conv_act_set) {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d", "Filter")
                         ->AsInput();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y = pattern->NewNode(elementwise_add_in_y_repr())
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->AsInput();
  auto elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                 ->assert_is_op_output("elementwise_add")
                                 ->assert_is_op_input("elementwise_add", "Y")
                                 ->AsIntermediate();

  auto elementwise_add_op_1 = pattern->NewNode(elementwise_add_op_1_repr())
                                  ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y_1 = pattern->NewNode(elementwise_add_in_y_1_repr())
                                    ->assert_is_op_input("elementwise_add", "X")
                                    ->AsInput();
  auto elementwise_add_out_1 = pattern->NewNode(elementwise_add_out_1_repr())
                                   ->assert_is_op_output("elementwise_add")
                                   ->AsIntermediate();

  auto act_op = pattern->NewNode(act_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      auto op_type = node->Name();
                      return conv_act_set.count(op_type);
                    });
  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_var()
                     // is activation op's output.
                     ->assert_more([&](Node *node) {
                       for (auto *in_op : node->inputs) {
                         if (conv_act_set.count(in_op->Name())) {
                           return true;
                         }
                       }
                       return false;
                     })
                     ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter}).LinksTo({conv_out});
  elementwise_add_op->LinksFrom({conv_out, elementwise_add_in_y})
      .LinksTo({elementwise_add_out});
  elementwise_add_op_1->LinksFrom({elementwise_add_out, elementwise_add_in_y_1})
      .LinksTo({elementwise_add_out_1});
  act_op->LinksFrom({elementwise_add_out_1}).LinksTo({act_out});
  return act_out;
}

PDNode *patterns::ConvElementwiseadd::operator()(PDNode *conv_in) {
  conv_in->AsInput();
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d", "Filter")
                         ->AsInput();
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y = pattern->NewNode(elementwise_add_in_y_repr())
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->AsInput();
  auto elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                 ->assert_is_op_output("elementwise_add")
                                 ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter});
  conv_out->LinksFrom({conv_op});
  elementwise_add_op->LinksFrom({conv_out, elementwise_add_in_y})
      .LinksTo({elementwise_add_out});

  return elementwise_add_out;
}

PDNode *patterns::ConvAffineChannel::operator()(
    paddle::framework::ir::PDNode *conv_input,
    const std::string &conv_type,
    bool with_eltwise_add) {
  // Create Operators
  conv_input->assert_is_op_input(conv_type, "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op(conv_type);

  PDNode *eltwise_op = nullptr;
  if (with_eltwise_add) {
    eltwise_op =
        pattern->NewNode(eltwise_repr())->assert_is_op("elementwise_add");
  }

  auto *affine_channel_op =
      pattern->NewNode(affine_channel_repr())->assert_is_op("affine_channel");
  // Create variables
  // Conv Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input(conv_type, "Filter");

  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op(conv_type);

  PDNode *eltwise_y_in_var = nullptr;
  PDNode *eltwise_out_var = nullptr;
  if (with_eltwise_add) {
    // Conv output as Bias input
    conv_out_var->assert_is_op_input("elementwise_add", "X");
    // Bias
    eltwise_y_in_var = pattern->NewNode(eltwise_y_in_repr())
                           ->assert_is_op_input("elementwise_add", "Y")
                           ->AsInput();
    eltwise_out_var = pattern->NewNode(eltwise_out_repr())
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("elementwise_add");
  } else {
    // Conv output as AffineChannel input
    conv_out_var->assert_is_op_input("affine_channel", "X");
  }

  // AC Scale
  auto *ac_scale_var = pattern->NewNode(ac_scale_repr())
                           ->AsInput()
                           ->assert_is_persistable_var()
                           ->assert_has_n_outputs(1)
                           ->assert_is_op_input("affine_channel", "Scale");
  // AC Bias
  auto *ac_bias_var = pattern->NewNode(ac_bias_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_has_n_outputs(1)
                          ->assert_is_op_input("affine_channel", "Bias");

  // AC output
  auto *ac_out_var = pattern->NewNode(ac_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("affine_channel");

  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});

  if (with_eltwise_add) {
    eltwise_op->LinksFrom({conv_out_var, eltwise_y_in_var})
        .LinksTo({eltwise_out_var});
    affine_channel_op->LinksFrom({eltwise_out_var, ac_scale_var, ac_bias_var})
        .LinksTo({ac_out_var});
  } else {
    affine_channel_op->LinksFrom({conv_out_var, ac_scale_var, ac_bias_var})
        .LinksTo({ac_out_var});
  }
  return ac_out_var;
}

PDNode *patterns::DequantQuantAny::operator()() {
  auto *dequant_in = pattern->NewNode(dequant_in_repr())
                         ->AsInput()
                         ->assert_is_op_input("dequantize", "Input");

  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");

  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize", "Output");

  auto *quant_op = pattern->NewNode(quant_op_repr())
                       ->assert_is_op("quantize")
                       ->AsIntermediate();

  auto *quant_out = pattern->NewNode(quant_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("quantize");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  quant_op->LinksFrom({dequant_out}).LinksTo({quant_out});
  next_op->LinksFrom({quant_out});

  return quant_out;
}

PDNode *patterns::DequantAny::operator()() {
  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");

  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize", "Output");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  dequant_op->LinksTo({dequant_out});
  next_op->LinksFrom({dequant_out});

  return dequant_out;
}

PDNode *patterns::MultipleQuantize::operator()() {
  auto *prev_out = pattern->NewNode(prev_out_repr())->AsOutput();

  // find nodes that are inputs to quantize operators
  prev_out->assert_more([&](Node *node) {
    int counter = static_cast<int>(std::count_if(
        node->outputs.begin(), node->outputs.end(), [&](Node const *iter) {
          return iter && iter->IsOp() && iter->Op()->Type() == "quantize";
        }));
    return (counter > 1);
  });

  return prev_out;
}

PDNode *patterns::QuantizePlacement::operator()(
    const std::unordered_set<std::string> &quantize_enabled_op_types) {
  auto *op =
      pattern->NewNode(op_repr())->assert_is_ops(quantize_enabled_op_types);
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<bool>("use_mkldnn");
  });
  return op;
}

PDNode *patterns::Bfloat16Placement::operator()(
    const std::unordered_set<std::string> &bfloat16_enabled_op_types) {
  std::unordered_set<std::string> supported_op_types =
      std::unordered_set<std::string>({"bilinear_interp_v2",
                                       "cast",
                                       "clip",
                                       "concat",
                                       "conv2d",
                                       "fused_conv2d",
                                       "conv2d_transpose",
                                       "elementwise_add",
                                       "elementwise_mul",
                                       "expand_v2",
                                       "fc",
                                       "fusion_gru",
                                       "fusion_lstm",
                                       "gelu",
                                       "layer_norm",
                                       "matmul",
                                       "matmul_v2",
                                       "fused_matmul",
                                       "pool2d",
                                       "prelu",
                                       "relu",
                                       "reshape2",
                                       "scale",
                                       "sigmoid",
                                       "slice",
                                       "softmax",
                                       "split",
                                       "squeeze",
                                       "squeeze2",
                                       "sum",
                                       "transpose2"});
  if (!bfloat16_enabled_op_types.empty()) {
    supported_op_types = bfloat16_enabled_op_types;
  }
  auto *op_in = pattern->NewNode(op_in_repr())->AsInput();
  auto *op = pattern->NewNode(op_repr())->assert_is_ops(supported_op_types);
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<bool>("use_mkldnn") ||
           node->Op()->Type() == "reshape2";
  });
  op->LinksFrom({op_in});
  return op;
}

PDNode *patterns::OrphanedBfloat16::operator()() {
  auto *prev_op = pattern->NewNode(prev_op_repr())->assert_is_op();
  prev_op->assert_more([&](Node *node) {
    bool data_type_is_missing = !node->Op()->HasAttr("mkldnn_data_type");
    bool data_type_is_fp32 = node->Op()->GetAttrIfExists<std::string>(
                                 "mkldnn_data_type") == "float32";
    return data_type_is_missing || data_type_is_fp32;
  });
  auto *prev_out = pattern->NewNode(prev_out_repr())->AsOutput();

  auto *op = pattern->NewNode(op_repr())->assert_is_op();
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });
  auto *op_out = pattern->NewNode(op_out_repr())->AsOutput();

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();
  next_op->assert_more([&](Node *node) {
    bool data_type_is_missing = !node->Op()->HasAttr("mkldnn_data_type");
    bool data_type_is_fp32 = node->Op()->GetAttrIfExists<std::string>(
                                 "mkldnn_data_type") == "float32";
    return data_type_is_missing || data_type_is_fp32;
  });

  prev_op->LinksTo({prev_out});
  op->LinksFrom({prev_out}).LinksTo({op_out});
  next_op->LinksFrom({op_out});
  return next_op;
}

PDNode *patterns::UnsupportedBfloat16::operator()() {
  auto *prev_op = pattern->NewNode(prev_op_repr())->assert_is_op();
  prev_op->assert_more([&](Node *node) {
    return node->Op()->HasAttr("mkldnn_data_type") == false;
  });
  auto *prev_out = pattern->NewNode(prev_out_repr())->AsOutput();

  auto *op = pattern->NewNode(op_repr())->assert_is_op();
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });
  prev_op->LinksTo({prev_out});
  op->LinksFrom({prev_out});
  return op;
}

PDNode *patterns::Bloat16Ops::operator()() {
  auto op = pattern->NewNode(op_repr())->assert_is_op();
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });
  return op;
}

PDNode *patterns::MKLDNNInPlace::operator()() {
  const std::unordered_set<std::string> &supported_op_types = {
      "abs", "gelu", "leaky_relu", "relu", "softmax", "sqrt", "swish", "tanh"};

  auto possible_inplace_op = pattern->NewNode(inplace_to_be_op_repr())
                                 ->assert_is_ops(supported_op_types);

  auto input = pattern->NewNode(inplace_to_be_op_in_repr())
                   ->assert_is_ops_input(supported_op_types)
                   ->AsInput();
  auto output = pattern->NewNode(inplace_to_be_op_out_repr())
                    ->assert_is_ops_output(supported_op_types)
                    ->AsOutput();

  auto next_op = pattern->NewNode(next_op_repr())->assert_is_op();
  auto next_output = pattern->NewNode(next_op_out_repr())->AsOutput();

  // Check if op is MKL-DNN enabled
  possible_inplace_op->assert_op_attr("use_mkldnn", true);

  // linked structure
  possible_inplace_op->LinksTo({output});
  possible_inplace_op->LinksFrom({input});
  next_op->LinksFrom({output});
  next_op->LinksTo({next_output});

  return possible_inplace_op;
}

// a -> transpose_op(1) -> transpose_out_a -> flatten_op(1) -> flatten_out_a
// b -> transpose_op(2) -> transpose_out_b -> flatten_op(2) -> flatten_out_b
// ...
// z -> transpose_op(n) -> transpose_out_z -> flatten_op(n) -> flatten_out_z
// flatten_out_a -> concat_op  flatten_out_b -> concat_op ... flatten_out_z ->
// concat_op
PDNode *patterns::TransposeFlattenConcat::operator()(
    std::vector<PDNode *> conv_in, int times) {
  // The times represents the repeat times of the
  // {trans, trans_out, flatten, flatten_out}
  const int kNumFields = 4;
  const int kTransOutOffset = 1;
  const int kFlattenOffset = 2;
  const int kFlattenOutOffset = 3;

  std::vector<PDNode *> nodes;

  for (int i = 0; i < times; i++) {
    nodes.push_back(
        pattern->NewNode(GetNodeName("transpose" + std::to_string(i)))
            ->assert_is_op("transpose2"));
    nodes.push_back(
        pattern->NewNode(GetNodeName("transpose_out" + std::to_string(i)))
            ->assert_is_op_output("transpose2")
            ->assert_is_op_input("flatten2", "X")
            ->AsIntermediate());
    nodes.push_back(pattern->NewNode(GetNodeName("flatten" + std::to_string(i)))
                        ->assert_is_op("flatten2"));

    nodes.push_back(
        pattern->NewNode(GetNodeName("flatten_out" + std::to_string(i)))
            ->assert_is_op_output("flatten2")
            ->assert_is_op_nth_input("concat", "X", i)
            ->AsIntermediate());
  }

  auto concat_op = pattern->NewNode(GetNodeName("concat"))
                       ->assert_is_op("concat")
                       ->assert_op_has_n_inputs("concat", times);
  auto concat_out = pattern->NewNode(GetNodeName("concat_out"))
                        ->assert_is_op_output("concat")
                        ->AsOutput();

  std::vector<PDNode *> flatten_outs;
  for (int i = 0; i < times; i++) {
    conv_in[i]->AsInput();
    // trans
    nodes[i * kNumFields]->LinksFrom({conv_in[i]});
    // trans_out
    nodes[i * kNumFields + kTransOutOffset]->LinksFrom({nodes[i * kNumFields]});
    // flatten
    nodes[i * kNumFields + kFlattenOffset]->LinksFrom(
        {nodes[i * kNumFields + kTransOutOffset]});
    // flatten_out
    nodes[i * kNumFields + kFlattenOutOffset]->LinksFrom(
        {nodes[i * kNumFields + kFlattenOffset]});
    flatten_outs.push_back(nodes[i * kNumFields + kFlattenOutOffset]);
  }

  concat_op->LinksFrom(flatten_outs).LinksTo({concat_out});
  return concat_out;
}

void patterns::DeleteDropoutOpPattern::operator()(bool with_mask) {
  auto dropout_op_x = pattern->NewNode(dropout_op_x_repr())
                          ->assert_is_op_input("dropout", "X")
                          ->AsInput();
  auto dropout_op = pattern->NewNode(dropout_op_repr())
                        ->assert_is_op("dropout")
                        ->assert_op_attr("dropout_implementation",
                                         std::string("upscale_in_train"));
  auto dropout_op_out = pattern->NewNode(dropout_op_out_repr())
                            ->assert_is_op_output("dropout", "Out");
  if (with_mask) {
    auto dropout_op_mask = pattern->NewNode(dropout_op_mask_repr())
                               ->assert_is_op_output("dropout", "Mask");
    dropout_op->LinksFrom({dropout_op_x})
        .LinksTo({dropout_op_out, dropout_op_mask});
  } else {
    dropout_op->LinksFrom({dropout_op_x}).LinksTo({dropout_op_out});
  }
}

void patterns::DeleteQuantOpFuse::operator()(PDNode *input_act_node,
                                             const std::string &quant_type) {
  auto *input_scale_node = pattern->NewNode(GetNodeName("input_scale_node"))
                               ->assert_is_op_input(quant_type, "InScale")
                               ->AsInput();
  auto *quant_node =
      pattern->NewNode(GetNodeName("quant_node"))->assert_is_op(quant_type);
  auto *output_scale_node = pattern->NewNode(GetNodeName("output_scale_node"))
                                ->assert_is_op_output(quant_type, "OutScale")
                                ->AsOutput();
  auto *output_act_node = pattern->NewNode(GetNodeName("output_act_node"))
                              ->assert_is_op_output(quant_type, "Out")
                              ->AsOutput();
  quant_node->LinksFrom({input_scale_node, input_act_node});
  output_scale_node->LinksFrom({quant_node});
  output_act_node->LinksFrom({quant_node});
}

void patterns::DequantOpFuse::operator()(PDNode *quantized_op_input,
                                         const std::string &quantized_op_type,
                                         const std::string &dequant_type,
                                         const std::string &weight_name) {
  auto *quantized_op_weight =
      pattern->NewNode(GetNodeName("quantized_op_weight"))
          ->assert_is_op_input(quantized_op_type, weight_name)
          ->AsInput();
  auto *quantized_op = pattern->NewNode(GetNodeName("quantized_op"))
                           ->assert_is_op(quantized_op_type);
  auto *quantized_op_out = pattern->NewNode(GetNodeName("quantized_op_out"))
                               ->assert_is_op_output(quantized_op_type)
                               ->assert_is_op_input(dequant_type, "X");
  auto *dequant_op =
      pattern->NewNode(GetNodeName("dequant_op"))->assert_is_op(dequant_type);
  auto *dequant_op_out = pattern->NewNode(GetNodeName("dequant_op_out"))
                             ->assert_is_op_output(dequant_type, "Out")
                             ->AsOutput();
  PDNode *dequant_channel_scale = nullptr;
  if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
    dequant_channel_scale =
        pattern->NewNode(GetNodeName("dequant_channel_scale"))
            ->assert_is_op_nth_input(dequant_type, "Scales", 0)
            ->AsInput();
  }
  quantized_op->LinksFrom({quantized_op_input, quantized_op_weight});
  quantized_op_out->LinksFrom({quantized_op});

  if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
    dequant_op->LinksFrom({quantized_op_out, dequant_channel_scale});
  } else {
    dequant_op->LinksFrom({quantized_op_out});
  }
  dequant_op_out->LinksFrom({dequant_op});
}

void patterns::ShuffleChannelPattern::operator()(PDNode *reshape1_in) {
  auto reshape1_op =
      pattern->NewNode(reshape1_op_repr())->assert_is_op("reshape2");
  reshape1_op->assert_more([&](Node *x) {
    return PADDLE_GET_CONST(std::vector<int>, x->Op()->GetAttr("shape"))
               .size() == 5;
  });

  auto reshape1_out = pattern->NewNode(reshape1_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->assert_is_op_input("transpose2")
                          ->AsIntermediate();

  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");

  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->assert_is_op_output("transpose2", "Out")
                           ->assert_is_op_input("reshape2")
                           ->AsIntermediate();

  auto reshape2_op =
      pattern->NewNode(reshape2_op_repr())->assert_is_op("reshape2");
  auto reshape2_out = pattern->NewNode(reshape2_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->AsOutput();

  reshape1_op->LinksFrom({reshape1_in});
  reshape1_out->LinksFrom({reshape1_op});
  transpose_op->LinksFrom({reshape1_out});
  transpose_out->LinksFrom({transpose_op});
  reshape2_op->LinksFrom({transpose_out});
  reshape2_out->LinksFrom({reshape2_op});
}

void patterns::DeleteQuantDequantOpPattern::operator()(
    PDNode *input_node, const std::string &quant_dequant_types) {
  auto quant_dequant_op_inscale =
      pattern->NewNode(quant_dequant_op_inscale_repr())
          ->assert_is_op_input(quant_dequant_types, "InScale")
          ->AsInput();
  auto quant_dequant_op = pattern->NewNode(quant_dequant_op_repr())
                              ->assert_is_op(quant_dequant_types);

  auto quant_dequant_op_out =
      pattern->NewNode(quant_dequant_op_out_repr())
          ->assert_is_op_output(quant_dequant_types, "Out")
          ->AsOutput();

  auto quant_dequant_op_outscale =
      pattern->NewNode(quant_dequant_op_outscale_repr())
          ->assert_is_op_output(quant_dequant_types, "OutScale")
          ->AsOutput();

  quant_dequant_op->LinksFrom({quant_dequant_op_inscale, input_node});
  quant_dequant_op_outscale->LinksFrom({quant_dequant_op});
  quant_dequant_op_out->LinksFrom({quant_dequant_op});
}

void patterns::DeleteQuantDequantFilterOpPattern::operator()() {
  auto quant_dequant_op_x =
      pattern->NewNode(quant_dequant_op_x_repr())
          ->assert_is_ops_input(
              {"fake_channel_wise_quantize_dequantize_abs_max",
               "fake_quantize_dequantize_abs_max"},
              "X")
          ->AsInput();

  auto quant_dequant_op =
      pattern->NewNode(quant_dequant_op_repr())
          ->assert_is_ops({"fake_channel_wise_quantize_dequantize_abs_max",
                           "fake_quantize_dequantize_abs_max"});

  auto quant_dequant_out =
      pattern->NewNode(quant_dequant_op_out_repr())
          ->assert_is_ops_output(
              {"fake_channel_wise_quantize_dequantize_abs_max",
               "fake_quantize_dequantize_abs_max"},
              "Out")
          ->AsIntermediate();

  auto quant_dequant_op_outscale =
      pattern->NewNode(quant_dequant_op_outscale_repr())
          ->assert_is_ops_output(
              {"fake_channel_wise_quantize_dequantize_abs_max",
               "fake_quantize_dequantize_abs_max"},
              "OutScale")
          ->AsOutput();
  auto any_op2 = pattern->NewNode(any_op2_repr())->assert_is_op()->AsOutput();

  quant_dequant_op->LinksFrom({quant_dequant_op_x});
  quant_dequant_op_outscale->LinksFrom({quant_dequant_op});
  quant_dequant_out->LinksFrom({quant_dequant_op});
  any_op2->LinksFrom({quant_dequant_out});
}

void patterns::DeleteWeightQuantDequantLinearOpPattern::operator()() {
  auto weight_dequantize_linear_op_x =
      pattern->NewNode(weight_dequantize_linear_op_x_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "X")
          ->assert_is_persistable_var();

  auto weight_dequantize_linear_op_scale =
      pattern->NewNode(weight_dequantize_linear_op_scale_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "Scale")
          ->assert_is_persistable_var();

  auto weight_dequantize_linear_op =
      pattern->NewNode(weight_dequantize_linear_op_repr())
          ->assert_is_op("dequantize_linear");

  auto weight_dequantize_linear_op_out =
      pattern->NewNode(weight_dequantize_linear_op_out_repr())
          ->AsOutput()
          ->assert_is_op_output("dequantize_linear", "Y");

  weight_dequantize_linear_op
      ->LinksFrom(
          {weight_dequantize_linear_op_x, weight_dequantize_linear_op_scale})
      .LinksTo({weight_dequantize_linear_op_out});
}

void patterns::DeleteWeightDequantLinearOpEncoderPattern::operator()() {
  auto weight_dequantize_linear_op_x =
      pattern->NewNode(weight_dequantize_linear_op_x_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "X")
          ->assert_is_persistable_var();

  auto weight_dequantize_linear_op_scale =
      pattern->NewNode(weight_dequantize_linear_op_scale_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "Scale")
          ->assert_is_persistable_var();

  auto weight_dequantize_linear_op =
      pattern->NewNode(weight_dequantize_linear_op_repr())
          ->assert_is_op("dequantize_linear");

  auto weight_dequantize_linear_op_out =
      pattern->NewNode(weight_dequantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("dequantize_linear", "Y");

  auto any_op2 = pattern->NewNode(any_op2_repr())->assert_is_op()->AsOutput();

  // while loop
  auto *while0 =
      pattern->NewNode(while0_repr())->assert_is_op("while")->AsOutput();
  while0->LinksFrom({weight_dequantize_linear_op_out});

  weight_dequantize_linear_op
      ->LinksFrom(
          {weight_dequantize_linear_op_x, weight_dequantize_linear_op_scale})
      .LinksTo({weight_dequantize_linear_op_out});
  any_op2->LinksFrom({weight_dequantize_linear_op_out});
}

PDNode *patterns::QuantLinearFusePattern::operator()(bool with_bias,
                                                     bool with_relu) {
  auto *quantize_linear_op_x = pattern->NewNode(quantize_linear_op_x_repr())
                                   ->AsInput()
                                   ->assert_is_op_input("quantize_linear", "X");

  auto *quantize_linear_op_scale =
      pattern->NewNode(quantize_linear_op_scale_repr())
          ->AsInput()
          ->assert_is_op_input("quantize_linear", "Scale")
          ->assert_is_persistable_var();

  auto *quantize_linear_op = pattern->NewNode(quantize_linear_op_repr())
                                 ->assert_is_op("quantize_linear");

  auto *quantize_linear_op_out =
      pattern->NewNode(quantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("quantize_linear", "Y")
          ->assert_is_op_input("dequantize_linear", "X")
          ->assert_var_not_persistable();

  auto *dequantize_linear_op = pattern->NewNode(dequantize_linear_op_repr())
                                   ->assert_is_op("dequantize_linear");

  auto *dequantize_linear_op_out =
      pattern->NewNode(dequantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("dequantize_linear", "Y")
          ->AsOutput();
  // Add links.
  quantize_linear_op
      ->LinksFrom({quantize_linear_op_x, quantize_linear_op_scale})
      .LinksTo({quantize_linear_op_out});
  dequantize_linear_op->LinksFrom({quantize_linear_op_out})
      .LinksTo({dequantize_linear_op_out});

  auto *weight_dequantize_linear_op_x =
      pattern->NewNode(weight_dequantize_linear_op_x_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "X")
          ->assert_is_persistable_var();

  auto *weight_dequantize_linear_op_scale =
      pattern->NewNode(weight_dequantize_linear_op_scale_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "Scale")
          ->assert_is_persistable_var();

  auto *weight_dequantize_linear_op =
      pattern->NewNode(weight_dequantize_linear_op_repr())
          ->assert_is_op("dequantize_linear");

  auto *weight_dequantize_linear_op_out =
      pattern->NewNode(weight_dequantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("dequantize_linear", "Y")
          ->assert_is_op_input("matmul_v2", "Y");

  // Add links.
  weight_dequantize_linear_op
      ->LinksFrom(
          {weight_dequantize_linear_op_x, weight_dequantize_linear_op_scale})
      .LinksTo({weight_dequantize_linear_op_out});

  auto *mul = pattern->NewNode(mul_repr())->assert_is_op("matmul_v2");

  auto *mul_out =
      pattern->NewNode(mul_out_repr())->assert_is_op_output("matmul_v2");

  // Add links.
  mul->LinksFrom({dequantize_linear_op_out, weight_dequantize_linear_op_out})
      .LinksTo({mul_out});

  if (!with_bias) {  // not with bias
    return mul_out;
  } else {  // with bias
    mul_out->AsIntermediate()->assert_is_op_input("elementwise_add", "X");

    auto *elementwise_add = pattern->NewNode(elementwise_add_repr())
                                ->assert_is_op("elementwise_add");

    auto *bias = pattern->NewNode(bias_repr())
                     ->assert_is_op_input("elementwise_add", "Y")
                     ->assert_is_persistable_var();

    auto *elementwise_add_out =
        pattern->NewNode(elementwise_add_out_repr())
            ->AsOutput()
            ->assert_is_op_output("elementwise_add", "Out");

    elementwise_add->LinksFrom({mul_out, bias}).LinksTo({elementwise_add_out});

    if (!with_relu) {
      return elementwise_add_out;
    } else {
      elementwise_add_out->AsIntermediate()->assert_is_op_input("relu");
      // Create operators.
      auto *relu = pattern->NewNode(relu_repr())->assert_is_op("relu");
      auto *relu_out = pattern->NewNode(relu_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("relu", "Out");

      relu->LinksFrom({elementwise_add_out}).LinksTo({relu_out});
      return relu_out;
    }
  }
}

void patterns::DeleteWeightDequantLinearOpDecoderPattern::operator()() {
  auto weight_dequantize_linear_op_x =
      pattern->NewNode(weight_dequantize_linear_op_x_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "X")
          ->assert_is_persistable_var();

  auto weight_dequantize_linear_op_scale =
      pattern->NewNode(weight_dequantize_linear_op_scale_repr())
          ->AsInput()
          ->assert_is_op_input("dequantize_linear", "Scale")
          ->assert_is_persistable_var();

  auto weight_dequantize_linear_op =
      pattern->NewNode(weight_dequantize_linear_op_repr())
          ->assert_is_op("dequantize_linear");

  auto weight_dequantize_linear_op_out =
      pattern->NewNode(weight_dequantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("dequantize_linear", "Y");

  auto any_op2 = pattern->NewNode(any_op2_repr())->assert_is_op()->AsOutput();

  weight_dequantize_linear_op
      ->LinksFrom(
          {weight_dequantize_linear_op_x, weight_dequantize_linear_op_scale})
      .LinksTo({weight_dequantize_linear_op_out});
  any_op2->LinksFrom({weight_dequantize_linear_op_out});
}

void patterns::DeleteQuantDequantLinearOpPattern::operator()() {
  auto quantize_linear_op_x = pattern->NewNode(quantize_linear_op_x_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("quantize_linear", "X");

  auto quantize_linear_op_scale =
      pattern->NewNode(quantize_linear_op_scale_repr())
          ->AsInput()
          ->assert_is_op_input("quantize_linear", "Scale")
          ->assert_is_persistable_var();

  auto quantize_linear_op = pattern->NewNode(quantize_linear_op_repr())
                                ->assert_is_op("quantize_linear");

  auto quantize_linear_op_out =
      pattern->NewNode(quantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("quantize_linear", "Y")
          ->assert_is_op_input("dequantize_linear", "X")
          ->assert_var_not_persistable();

  // Can not add this node. Todo: Wangzheee
  /*
    auto dequantize_linear_op_scale =
        pattern->NewNode(dequantize_linear_op_scale_repr())
            ->assert_is_op_input("dequantize_linear", "Scale")
            ->AsIntermediate();
  */

  auto dequantize_linear_op = pattern->NewNode(dequantize_linear_op_repr())
                                  ->assert_is_op("dequantize_linear");

  auto dequantize_linear_op_out =
      pattern->NewNode(dequantize_linear_op_out_repr())
          ->AsIntermediate()
          ->assert_is_op_output("dequantize_linear", "Y")
          ->AsOutput();

  quantize_linear_op
      ->LinksFrom({quantize_linear_op_x, quantize_linear_op_scale})
      .LinksTo({quantize_linear_op_out});
  dequantize_linear_op->LinksFrom({quantize_linear_op_out})
      .LinksTo({dequantize_linear_op_out});
}

PDNode *patterns::ReshapeTransposeMatmulPattern::operator()(
    const std::string &op_name,
    bool with_reshape_xshape,
    bool with_transpose_xshape) {
  auto reshape_op =
      pattern->NewNode(reshape_op_repr())->assert_is_op("reshape2");
  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op(op_name);

  auto reshape_in = pattern->NewNode(reshape_in_repr())
                        ->AsInput()
                        ->assert_is_op_input("reshape2", "X");

  auto reshape_out = pattern->NewNode(reshape_out_repr())
                         ->AsIntermediate()
                         ->assert_is_op_input("transpose2", "X")
                         ->assert_is_op_output("reshape2", "Out");
  if (!with_reshape_xshape)
    reshape_out->assert_is_only_output_of_op("reshape2");

  auto reshape_xshape = with_reshape_xshape
                            ? pattern->NewNode(reshape_xshape_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("reshape2", "XShape")
                            : nullptr;

  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->AsIntermediate()
                           ->assert_is_op_input(op_name)
                           ->assert_is_op_output("transpose2", "Out");
  if (!with_transpose_xshape)
    transpose_out->assert_is_only_output_of_op("transpose2");

  auto transpose_xshape =
      with_transpose_xshape ? pattern->NewNode(transpose_xshape_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("transpose2", "XShape")
                            : nullptr;

  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output(op_name, "Out");

  reshape_op->LinksFrom({reshape_in}).LinksTo({reshape_out});
  if (with_reshape_xshape) reshape_op->LinksTo({reshape_xshape});
  transpose_op->LinksFrom({reshape_out}).LinksTo({transpose_out});
  if (with_transpose_xshape) transpose_op->LinksTo({transpose_xshape});
  matmul_op->LinksFrom({transpose_out}).LinksTo({matmul_out});
  return matmul_out;
}

// shared function for matmul and matmul_v2
PDNode *patterns::MatmulTransposeReshapePattern::operator()(
    const std::string &op_name) {
  auto reshape_op =
      pattern->NewNode(reshape_op_repr())->assert_is_op("reshape2");
  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op(op_name);

  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsInput()
                        ->assert_is_op_output(op_name, "Out")
                        ->assert_is_op_input("transpose2", "X");

  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->AsIntermediate()
                           ->assert_is_op_output("transpose2", "Out")
                           ->assert_is_op_input("reshape2", "X");

  auto transpose_out_xshape = pattern->NewNode(transpose_out_xshape_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("transpose2", "XShape");

  auto reshape_out = pattern->NewNode(reshape_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("reshape2");

  auto reshape_out_xshape = pattern->NewNode(reshape_out_xshape_repr())
                                ->AsIntermediate()
                                ->assert_is_op_output("reshape2", "XShape");

  matmul_op->LinksTo({matmul_out});
  transpose_op->LinksTo({transpose_out_xshape});
  reshape_op->LinksTo({reshape_out_xshape});
  transpose_op->LinksFrom({matmul_out}).LinksTo({transpose_out});
  reshape_op->LinksFrom({transpose_out}).LinksTo({reshape_out});
  return reshape_out;
}

PDNode *patterns::FusionGru::operator()() {
  auto op = pattern->NewNode(op_repr())->assert_is_op("fusion_gru");
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "X");
  auto weight_h = pattern->NewNode(weight_h_repr())
                      ->AsInput()
                      ->assert_is_op_input("fusion_gru", "WeightH");
  auto weight_x = pattern->NewNode(weight_x_repr())
                      ->AsInput()
                      ->assert_is_op_input("fusion_gru", "WeightX");
  auto out = pattern->NewNode(out_repr())
                 ->AsOutput()
                 ->assert_is_op_output("fusion_gru", "Hidden");
  op->LinksFrom({x, weight_h, weight_x}).LinksTo({out});
  return out;
}

PDNode *patterns::FusionLSTM::operator()() {
  auto op = pattern->NewNode(op_repr())->assert_is_op("fusion_lstm");
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "fusion_lstm", "X");
  auto weight_h = pattern->NewNode(weight_h_repr())
                      ->AsInput()
                      ->assert_is_op_input("fusion_lstm", "WeightH");
  auto weight_x = pattern->NewNode(weight_x_repr())
                      ->AsInput()
                      ->assert_is_op_input("fusion_lstm", "WeightX");
  auto hidden = pattern->NewNode(hidden_repr())
                    ->AsOutput()
                    ->assert_is_op_output("fusion_lstm", "Hidden");
  auto cell = pattern->NewNode(cell_repr())
                  ->AsOutput()
                  ->assert_is_op_output("fusion_lstm", "Cell");
  op->LinksFrom({x, weight_h, weight_x}).LinksTo({hidden, cell});
  return hidden;
}

PDNode *patterns::TwoFusionGruConcat::operator()() {
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "X");
  auto gru1 =
      pattern->NewNode(gru1_repr())
          ->assert_is_op("fusion_gru")
          ->assert_more([&](Node *node) {
            return node->Op()->GetAttrIfExists<bool>("is_reverse") == false;
          });
  auto gru2 =
      pattern->NewNode(gru2_repr())
          ->assert_is_op("fusion_gru")
          ->assert_more([&](Node *node) {
            return node->Op()->GetAttrIfExists<bool>("is_reverse") == true;
          });
  auto wh1 = pattern->NewNode(wh1_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightH");
  auto wh2 = pattern->NewNode(wh2_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightH");
  auto wx1 = pattern->NewNode(wx1_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightX");
  auto wx2 = pattern->NewNode(wx2_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightX");
  auto b1 = pattern->NewNode(b1_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "Bias");
  auto b2 = pattern->NewNode(b2_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "Bias");
  auto h1 = pattern->NewNode(h1_repr())
                ->AsOutput()
                ->assert_is_op_output("fusion_gru", "Hidden")
                ->assert_is_op_input("concat")
                ->AsIntermediate();
  auto h2 = pattern->NewNode(h2_repr())
                ->AsOutput()
                ->assert_is_op_output("fusion_gru", "Hidden")
                ->assert_is_op_input("concat")
                ->AsIntermediate();
  auto concat = pattern->NewNode(concat_repr())->assert_is_op("concat");
  auto out = pattern->NewNode(out_repr())
                 ->AsOutput()
                 ->assert_is_op_output("concat", "Out");
  gru1->LinksFrom({x, wh1, wx1, b1}).LinksTo({h1});
  gru2->LinksFrom({x, wh2, wx2, b2}).LinksTo({h2});
  concat->LinksFrom({h1, h2}).LinksTo({out});
  return out;
}

PDNode *patterns::MultiGruSeq::operator()() {
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "multi_gru", "X");
  auto gru1 = pattern->NewNode(gru1_repr())->assert_is_op("multi_gru");
  auto wx11 = pattern->NewNode(wx11_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 0);
  auto wx12 = pattern->NewNode(wx12_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 1);
  auto wh11 = pattern->NewNode(wh11_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 0);
  auto wh12 = pattern->NewNode(wh12_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 1);
  auto b11 = pattern->NewNode(b11_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 0);
  auto b12 = pattern->NewNode(b12_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 1);
  auto h1 = pattern->NewNode(h1_repr())
                ->AsOutput()
                ->assert_is_op_output("multi_gru", "Hidden")
                ->assert_is_op_input("multi_gru", "X")
                ->AsIntermediate();
  auto gru2 = pattern->NewNode(gru2_repr())->assert_is_op("multi_gru");
  auto wx21 = pattern->NewNode(wx21_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 0);
  auto wx22 = pattern->NewNode(wx22_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 1);
  auto wh21 = pattern->NewNode(wh21_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 0);
  auto wh22 = pattern->NewNode(wh22_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 1);
  auto b21 = pattern->NewNode(b21_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 0);
  auto b22 = pattern->NewNode(b22_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 1);
  auto h2 = pattern->NewNode(h2_repr())->AsOutput()->assert_is_op_output(
      "multi_gru", "Hidden");
  gru1->LinksFrom({x, wx11, wx12, wh11, wh12, b11, b12}).LinksTo({h1});
  gru2->LinksFrom({h1, wx21, wx22, wh21, wh22, b21, b22}).LinksTo({h2});
  return h2;
}

PDNode *patterns::MultiGru::operator()() {
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "multi_gru", "X");
  auto gru = pattern->NewNode(gru_repr())->assert_is_op("multi_gru");
  auto wx = pattern->NewNode(wx_repr())->AsInput()->assert_is_op_nth_input(
      "multi_gru", "WeightX", 0);
  auto wh = pattern->NewNode(wh_repr())->AsInput()->assert_is_op_nth_input(
      "multi_gru", "WeightH", 0);
  auto h = pattern->NewNode(h_repr())->AsOutput()->assert_is_op_output(
      "multi_gru", "Hidden");
  gru->LinksFrom({x, wx, wh}).LinksTo({h});
  return h;
}

PDNode *patterns::LayerNorm::operator()() {
  auto *x = pattern->NewNode(x_repr())->AsInput()->assert_is_ops_input(
      {"reduce_mean", "elementwise_sub"});
  auto *x_mean = pattern->NewNode(x_mean_repr())->assert_is_op("reduce_mean");
  auto *x_mean_out = pattern->NewNode(x_mean_out_repr())
                         ->assert_is_op_output("reduce_mean", "Out")
                         ->assert_is_op_input("elementwise_sub", "Y")
                         ->AsIntermediate();
  auto *x_sub_mean =
      pattern->NewNode(x_sub_mean_repr())->assert_is_op("elementwise_sub");
  auto *x_sub_mean_out =
      pattern->NewNode(x_sub_mean_out_repr())
          ->assert_is_op_output("elementwise_sub")
          ->assert_is_ops_input({"elementwise_pow", "elementwise_div"}, "X")
          ->AsIntermediate();
  auto *sqr_pow = pattern->NewNode(sqr_pow_repr())
                      ->assert_is_op_input("elementwise_pow", "Y")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto *x_sub_mean_sqr =
      pattern->NewNode(x_sub_mean_sqr_repr())->assert_is_op("elementwise_pow");
  auto *x_sub_mean_sqr_out = pattern->NewNode(x_sub_mean_sqr_out_repr())
                                 ->assert_is_op_output("elementwise_pow")
                                 ->assert_is_op_input("reduce_mean")
                                 ->AsIntermediate();
  auto *std_dev = pattern->NewNode(std_dev_repr())->assert_is_op("reduce_mean");
  auto *std_dev_out = pattern->NewNode(std_dev_out_repr())
                          ->assert_is_op_output("reduce_mean")
                          ->assert_is_op_input("elementwise_add")
                          ->AsIntermediate();
  auto *eps = pattern->NewNode(eps_repr())
                  ->assert_is_op_input("elementwise_add", "Y")
                  ->assert_is_persistable_var()
                  ->AsInput();
  auto *std_dev_eps =
      pattern->NewNode(std_dev_eps_repr())->assert_is_op("elementwise_add");
  auto *std_dev_eps_out = pattern->NewNode(std_dev_eps_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->assert_is_op_input("sqrt")
                              ->AsIntermediate();
  auto *std_dev_eps_sqrt =
      pattern->NewNode(std_dev_eps_sqrt_repr())->assert_is_op("sqrt");
  auto *std_dev_eps_sqrt_out = pattern->NewNode(std_dev_eps_sqrt_out_repr())
                                   ->assert_is_op_output("sqrt")
                                   ->assert_is_op_input("elementwise_div", "Y")
                                   ->AsIntermediate();
  auto *division =
      pattern->NewNode(division_repr())->assert_is_op("elementwise_div");
  auto *division_out = pattern->NewNode(division_out_repr())
                           ->assert_is_op_output("elementwise_div")
                           ->assert_is_op_input("elementwise_mul")
                           ->AsIntermediate();
  auto *gamma = pattern->NewNode(gamma_repr())
                    ->assert_is_op_input("elementwise_mul", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();
  auto *scale = pattern->NewNode(scale_repr())->assert_is_op("elementwise_mul");
  auto *scale_out = pattern->NewNode(scale_out_repr())
                        ->assert_is_op_output("elementwise_mul")
                        ->assert_is_op_input("elementwise_add")
                        ->AsIntermediate();
  auto *beta = pattern->NewNode(beta_repr())
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->assert_is_persistable_var()
                   ->AsInput();
  auto *shift = pattern->NewNode(shift_repr())->assert_is_op("elementwise_add");
  auto *shift_out = pattern->NewNode(shift_out_repr())
                        ->assert_is_op_output("elementwise_add")
                        ->AsOutput();

  /*
   *            X
   *           / \
   *          /   reduce_mean "u(x)"
   *          \   /
   *      elementwise_sub     "x - u(x)"
   *      /           \    2
   *      |            \  /
   *      |      elementwise_pow  "(x - u(x))^2"
   *      |             |
   *      |       reduce_mean     "sigma^2 = 1/C*Sum{(x - u(x))^2}"
   *      |             |     eps
   *      |             |     /
   *      |       elementwise_add "sigma^2 + epsilon"
   *      \             |
   *       \           sqrt       "sqrt(sigma^2 + epsilon)"
   *        \          /
   *         \        /
   *       elementwise_div        "lnorm = {x-u(x)}/{sqrt(sigma^2 + epsilon)}"
   *              |
   *       gamma  |
   *          \   |
   *       elementwise_mul        "scale: gamma(C) * lnorm"
   *              |
   *        beta  |
   *          \   |
   *       elementwise_add        "shift: gamma(C) * lnorm + beta(C)"
   */

  x_mean->LinksFrom({x}).LinksTo({x_mean_out});
  x_sub_mean->LinksFrom({x, x_mean_out}).LinksTo({x_sub_mean_out});
  x_sub_mean_sqr->LinksFrom({x_sub_mean_out, sqr_pow})
      .LinksTo({x_sub_mean_sqr_out});
  std_dev->LinksFrom({x_sub_mean_sqr_out}).LinksTo({std_dev_out});
  std_dev_eps->LinksFrom({std_dev_out, eps}).LinksTo({std_dev_eps_out});

  std_dev_eps_sqrt->LinksFrom({std_dev_eps_out})
      .LinksTo({std_dev_eps_sqrt_out});
  division->LinksFrom({x_sub_mean_out, std_dev_eps_sqrt_out})
      .LinksTo({division_out});
  scale->LinksFrom({division_out, gamma}).LinksTo({scale_out});
  shift->LinksFrom({scale_out, beta}).LinksTo({shift_out});

  return shift_out;
}

// Add support int8 flag and out_threshold
PDNode *patterns::AddSupportInt8::operator()() {
  auto quant_op = pattern->NewNode(quant_op_repr())->assert_is_op();
  auto quant_out =
      pattern->NewNode(quant_out_repr())
          ->assert_is_var()
          ->assert_more([&](Node *node) { return !node->outputs.empty(); })
          ->AsOutput();
  quant_op->LinksTo({quant_out});
  return quant_out;
}

PDNode *patterns::SplitLayerNorm::operator()() {
  auto layer_norm_op =
      pattern->NewNode(layer_norm_op_repr())->assert_is_op("layer_norm");
  auto layer_norm_in = pattern->NewNode(layer_norm_in_repr())
                           ->AsInput()
                           ->assert_is_op_input("layer_norm", "X");
  auto layer_norm_bias = pattern->NewNode(layer_norm_bias_repr())
                             ->AsInput()
                             ->assert_is_op_input("layer_norm", "Bias");
  auto layer_norm_scale = pattern->NewNode(layer_norm_scale_repr())
                              ->AsInput()
                              ->assert_is_op_input("layer_norm", "Scale");
  auto layer_norm_out = pattern->NewNode(layer_norm_out_repr())
                            ->assert_is_op_output("layer_norm", "Y");

  layer_norm_op->LinksFrom({layer_norm_in, layer_norm_bias, layer_norm_scale})
      .LinksTo({layer_norm_out});

  return layer_norm_out;
}

PDNode *patterns::LayernormShiftPartitionPattern::operator()() {
  auto layer_norm_op =
      pattern->NewNode(layer_norm_op_repr())
          ->assert_is_op("layer_norm")
          ->assert_more([&](Node *node) {
            return node->Op()->HasAttr("begin_norm_axis") &&
                   (PADDLE_GET_CONST(
                        int, node->Op()->GetAttr("begin_norm_axis")) == 2);
          });
  auto layer_norm_in = pattern->NewNode(layer_norm_in_repr())
                           ->AsInput()
                           ->assert_is_op_input("layer_norm", "X");
  auto layer_norm_bias = pattern->NewNode(layer_norm_bias_repr())
                             ->AsInput()
                             ->assert_is_op_input("layer_norm", "Bias");
  auto layer_norm_scale = pattern->NewNode(layer_norm_scale_repr())
                              ->AsInput()
                              ->assert_is_op_input("layer_norm", "Scale");
  auto layer_norm_out = pattern->NewNode(layer_norm_out_repr())
                            ->AsIntermediate()
                            ->assert_is_op_input("reshape2", "X")
                            ->assert_is_op_output("layer_norm", "Y");
  auto reshape1_op =
      pattern->NewNode(reshape1_op_repr())
          ->assert_is_op("reshape2")
          ->assert_more([&](Node *node) {
            return node->Op()->HasAttr("shape") &&
                   (PADDLE_GET_CONST(std::vector<int>,
                                     node->Op()->GetAttr("shape"))
                        .size() == 4);
          });
  auto reshape1_out = pattern->NewNode(reshape1_out_repr())
                          ->AsIntermediate()
                          ->assert_is_op_output("reshape2", "Out");
  PDNode *roll1_op = nullptr;
  PDNode *roll1_out = nullptr;

  if (!with_roll_) {
    reshape1_out->assert_is_op_input("reshape2", "X");
  } else {
    reshape1_out->assert_is_op_input("roll", "X");
    roll1_op = pattern->NewNode(roll1_op_repr())->assert_is_op("roll");
    roll1_out = pattern->NewNode(roll1_out_repr())
                    ->AsIntermediate()
                    ->assert_is_op_output("roll", "Out")
                    ->assert_is_op_input("reshape2", "X");
  }
  auto reshape2_op =
      pattern->NewNode(reshape2_op_repr())
          ->assert_is_op("reshape2")
          ->assert_more([&](Node *node) {
            return node->Op()->HasAttr("shape") &&
                   (PADDLE_GET_CONST(std::vector<int>,
                                     node->Op()->GetAttr("shape"))
                        .size() == 6);
          });

  auto reshape2_out = pattern->NewNode(reshape2_out_repr())
                          ->AsIntermediate()
                          ->assert_is_op_input("transpose2", "X")
                          ->assert_is_op_output("reshape2", "Out");
  auto transpose_op =
      pattern->NewNode(transpose_op_repr())
          ->assert_is_op("transpose2")
          ->assert_more([&](Node *node) {
            if (!node->Op()->HasAttr("axis")) return false;
            std::vector<int> axis =
                PADDLE_GET_CONST(std::vector<int>, node->Op()->GetAttr("axis"));
            if (axis.size() != 6) return false;
            const std::vector<int> axis_cmp{0, 1, 3, 2, 4, 5};
            return std::equal(axis.begin(), axis.end(), axis_cmp.begin());
          });
  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->AsIntermediate()
                           ->assert_is_op_input("reshape2", "X")
                           ->assert_is_op_output("transpose2", "Out");
  auto reshape3_op =
      pattern->NewNode(reshape3_op_repr())
          ->assert_is_op("reshape2")
          ->assert_more([&](Node *node) {
            return node->Op()->HasAttr("shape") &&
                   (PADDLE_GET_CONST(std::vector<int>,
                                     node->Op()->GetAttr("shape"))
                        .size() == 4);
          });
  auto reshape3_out = pattern->NewNode(reshape3_out_repr())
                          ->AsIntermediate()
                          ->assert_is_op_input("reshape2", "X")
                          ->assert_is_op_output("reshape2", "Out");
  auto reshape4_op =
      pattern->NewNode(reshape4_op_repr())
          ->assert_is_op("reshape2")
          ->assert_more([&](Node *node) {
            return node->Op()->HasAttr("shape") &&
                   (PADDLE_GET_CONST(std::vector<int>,
                                     node->Op()->GetAttr("shape"))
                        .size() == 3);
          });
  auto reshape4_out = pattern->NewNode(reshape4_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->AsOutput();

  layer_norm_op->LinksFrom({layer_norm_in, layer_norm_bias, layer_norm_scale})
      .LinksTo({layer_norm_out});
  reshape1_op->LinksFrom({layer_norm_out}).LinksTo({reshape1_out});
  if (!with_roll_) {
    reshape2_op->LinksFrom({reshape1_out}).LinksTo({reshape2_out});
  } else {
    roll1_op->LinksFrom({reshape1_out}).LinksTo({roll1_out});
    reshape2_op->LinksFrom({roll1_out}).LinksTo({reshape2_out});
  }
  transpose_op->LinksFrom({reshape2_out}).LinksTo({transpose_out});
  reshape3_op->LinksFrom({transpose_out}).LinksTo({reshape3_out});
  reshape4_op->LinksFrom({reshape3_out}).LinksTo({reshape4_out});

  return reshape4_out;
}

PDNode *patterns::ReverseRollPattern::operator()(PDNode *in) {
  in->AsInput();
  auto reshape2_00_op =
      pattern->NewNode(reshape2_00_op_repr())->assert_is_op("reshape2");

  auto reshape2_00_out = pattern->NewNode(reshape2_00_out_repr())
                             ->AsIntermediate()
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("reshape2", "X");

  auto reshape2_10_op =
      pattern->NewNode(reshape2_10_op_repr())->assert_is_op("reshape2");
  auto reshape2_10_out = pattern->NewNode(reshape2_10_out_repr())
                             ->AsIntermediate()
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("transpose2", "X");

  auto transpose2_20_op =
      pattern->NewNode(transpose2_20_op_repr())->assert_is_op("transpose2");
  auto transpose2_20_out = pattern->NewNode(transpose2_20_out_repr())
                               ->AsIntermediate()
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("reshape2", "X");

  auto reshape2_30_op =
      pattern->NewNode(reshape2_30_op_repr())->assert_is_op("reshape2");
  auto reshape2_30_out = pattern->NewNode(reshape2_30_out_repr())
                             ->AsIntermediate()
                             ->assert_is_op_output("reshape2", "Out");
  PDNode *roll_40_op = nullptr;
  PDNode *roll_40_out = nullptr;
  if (with_roll_) {
    reshape2_30_out->assert_is_op_input("roll", "X");
    roll_40_op = pattern->NewNode(roll_40_op_repr())->assert_is_op("roll");
    roll_40_out = pattern->NewNode(roll_40_out_repr())
                      ->AsIntermediate()
                      ->assert_is_op_output("roll", "Out")
                      ->assert_is_op_input("reshape2", "X");
  } else {
    reshape2_30_out->assert_is_op_input("reshape2", "X");
  }
  auto reshape2_50_op =
      pattern->NewNode(reshape2_50_op_repr())->assert_is_op("reshape2");
  auto reshape2_50_out = pattern->NewNode(reshape2_50_out_repr())
                             ->assert_is_op_output("reshape2", "Out")
                             ->AsOutput();
  reshape2_00_op->LinksFrom({in});
  reshape2_00_out->LinksFrom({reshape2_00_op});
  reshape2_10_op->LinksFrom({reshape2_00_out});
  reshape2_10_out->LinksFrom({reshape2_10_op});
  transpose2_20_op->LinksFrom({reshape2_10_out});
  transpose2_20_out->LinksFrom({transpose2_20_op});
  reshape2_30_op->LinksFrom({transpose2_20_out});
  reshape2_30_out->LinksFrom({reshape2_30_op});
  if (with_roll_) {
    roll_40_op->LinksFrom({reshape2_30_out});
    roll_40_out->LinksFrom({roll_40_op});
    reshape2_50_op->LinksFrom({roll_40_out});
  } else {
    reshape2_50_op->LinksFrom({reshape2_30_out});
  }
  reshape2_50_out->LinksFrom({reshape2_50_op});
  return reshape2_50_out;
}
PDNode *patterns::MergeLayernormPattern::operator()(PDNode *in) {
  in->AsInput();
  auto reshape2_00_op =
      pattern->NewNode(reshape2_00_op_repr())->assert_is_op("reshape2");
  auto reshape2_00_out = pattern->NewNode(reshape2_00_out_repr())
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("strided_slice", "Input")
                             ->AsIntermediate();
  auto strided_slice_10_op = pattern->NewNode(strided_slice_10_op_repr())
                                 ->assert_is_op("strided_slice");
  auto strided_slice_10_out = pattern->NewNode(strided_slice_10_out_repr())
                                  ->assert_is_op_output("strided_slice", "Out")
                                  ->assert_is_op_nth_input("concat", "X", 0)
                                  ->AsIntermediate();
  auto strided_slice_11_op = pattern->NewNode(strided_slice_11_op_repr())
                                 ->assert_is_op("strided_slice");
  auto strided_slice_11_out = pattern->NewNode(strided_slice_11_out_repr())
                                  ->assert_is_op_output("strided_slice", "Out")
                                  ->assert_is_op_nth_input("concat", "X", 1)
                                  ->AsIntermediate();
  auto strided_slice_12_op = pattern->NewNode(strided_slice_12_op_repr())
                                 ->assert_is_op("strided_slice");
  auto strided_slice_12_out = pattern->NewNode(strided_slice_12_out_repr())
                                  ->assert_is_op_output("strided_slice", "Out")
                                  ->assert_is_op_nth_input("concat", "X", 2)
                                  ->AsIntermediate();
  auto strided_slice_13_op = pattern->NewNode(strided_slice_13_op_repr())
                                 ->assert_is_op("strided_slice");
  auto strided_slice_13_out = pattern->NewNode(strided_slice_13_out_repr())
                                  ->assert_is_op_output("strided_slice", "Out")
                                  ->assert_is_op_nth_input("concat", "X", 3)
                                  ->AsIntermediate();
  auto concat_20_op = pattern->NewNode(concat_20_op_repr())
                          ->assert_is_op("concat")
                          ->assert_has_n_inputs(4);
  auto concat_20_out = pattern->NewNode(concat_20_out_repr())
                           ->assert_is_op_output("concat", "Out")
                           ->assert_is_op_input("reshape2", "X")
                           ->AsIntermediate();
  auto reshape2_30_op =
      pattern->NewNode(reshape2_30_op_repr())->assert_is_op("reshape2");
  auto reshape2_30_out = pattern->NewNode(reshape2_30_out_repr())
                             ->assert_is_op_output("reshape2", "Out")
                             ->assert_is_op_input("layer_norm", "X")
                             ->AsIntermediate();
  auto layernorm_40_op =
      pattern->NewNode(layernorm_40_op_repr())
          ->assert_is_op("layer_norm")
          ->assert_more([&](Node *node) {
            return node->Op()->HasAttr("begin_norm_axis") &&
                   (PADDLE_GET_CONST(
                        int, node->Op()->GetAttr("begin_norm_axis")) == 2);
          });
  auto layernorm_40_in_bias = pattern->NewNode(layernorm_40_in_bias_repr())
                                  ->assert_is_op_input("layer_norm", "Bias")
                                  ->AsInput();
  auto layernorm_40_in_scale = pattern->NewNode(layernorm_40_in_scale_repr())
                                   ->assert_is_op_input("layer_norm", "Scale")
                                   ->AsInput();
  auto layernorm_40_out = pattern->NewNode(layernorm_40_out_repr())
                              ->assert_is_op_output("layer_norm", "Y")
                              ->AsOutput();

  reshape2_00_op->LinksFrom({in});
  reshape2_00_out->LinksFrom({reshape2_00_op});
  strided_slice_10_op->LinksFrom({reshape2_00_out});
  strided_slice_10_out->LinksFrom({strided_slice_10_op});
  strided_slice_11_op->LinksFrom({reshape2_00_out});
  strided_slice_11_out->LinksFrom({strided_slice_11_op});
  strided_slice_12_op->LinksFrom({reshape2_00_out});
  strided_slice_12_out->LinksFrom({strided_slice_12_op});
  strided_slice_13_op->LinksFrom({reshape2_00_out});
  strided_slice_13_out->LinksFrom({strided_slice_13_op});
  concat_20_op->LinksFrom({strided_slice_10_out,
                           strided_slice_11_out,
                           strided_slice_12_out,
                           strided_slice_13_out});
  concat_20_out->LinksFrom({concat_20_op});
  reshape2_30_op->LinksFrom({concat_20_out});
  reshape2_30_out->LinksFrom({reshape2_30_op});
  layernorm_40_op->LinksFrom(
      {reshape2_30_out, layernorm_40_in_bias, layernorm_40_in_scale});
  layernorm_40_out->LinksFrom({layernorm_40_op});
  return layernorm_40_out;
}

PDNode *patterns::FusedFeedForwardFwd::operator()(
    paddle::framework::ir::PDNode *x_var,
    std::unordered_set<std::string> act_types,
    bool use_mp,
    bool pre_layer_norm,
    bool add_residual,
    bool use_dropout_1,
    bool use_dropout_2) {
  // Possible patterns
  // 1. layer_norm -> linear1 -> activation -> dropout1 -> linear2 -> dropout2
  // -> residual_add (pre_layer_norm)
  // 2. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> residual_add
  // -> layer_norm (pOST_layer_norm)
  // other cases: may delete residual_add, dropout1, dropout2 operators

  // intermediate input, and final pattern output
  PDNode *out_var = x_var;
  // LayerNorm
  auto *layer_norm_op =
      pattern->NewNode(layer_norm_op_repr())->assert_is_op("layer_norm");
  auto *layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto *layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_op_input("layer_norm", "Scale");
  auto *layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output("layer_norm", "Y");
  auto *layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto *layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  if (pre_layer_norm) {
    out_var->assert_is_op_input("layer_norm", "X");
    layer_norm_op
        ->LinksFrom({out_var, layer_norm_bias_var, layer_norm_scale_var})
        .LinksTo(
            {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});
    out_var = layer_norm_out_var;
  }

  // Model parallel, do nothing in forward.
  if (use_mp) {
    out_var->assert_is_op_input("c_identity", "X");
    auto *c_identity_op =
        pattern->NewNode(c_identity_op_repr())->assert_is_op("c_identity");
    auto *c_identity_out_var = pattern->NewNode(c_identity_out_repr())
                                   ->assert_is_op_output("c_identity", "Out");
    c_identity_op->LinksFrom({out_var}).LinksTo({c_identity_out_var});
    out_var = c_identity_out_var;
  }

  // Linear1
  out_var->assert_is_op_input("matmul_v2", "X");
  auto *matmul_op_1 =
      pattern->NewNode(matmul_op_1_repr())->assert_is_op("matmul_v2");
  auto *matmul_w_var_1 = pattern->NewNode(matmul_w_1_repr())
                             ->AsInput()
                             ->assert_is_op_input("matmul_v2", "Y");
  auto *matmul_out_var_1 = pattern->NewNode(matmul_out_1_repr())
                               ->assert_is_op_output("matmul_v2", "Out");
  matmul_op_1->LinksFrom({out_var, matmul_w_var_1}).LinksTo({matmul_out_var_1});
  out_var = matmul_out_var_1;

  out_var->assert_is_op_input("elementwise_add", "X");
  auto *ele_add_op_1 =
      pattern->NewNode(ele_add_op_1_repr())->assert_is_op("elementwise_add");
  auto *ele_add_bias_var_1 = pattern->NewNode(ele_add_bias_1_repr())
                                 ->AsInput()
                                 ->assert_is_op_input("elementwise_add", "Y");
  auto *ele_add_out_var_1 = pattern->NewNode(ele_add_out_1_repr())
                                ->assert_is_op_output("elementwise_add", "Out");
  ele_add_op_1->LinksFrom({out_var, ele_add_bias_var_1})
      .LinksTo({ele_add_out_var_1});
  out_var = ele_add_out_var_1;

  // Activation
  out_var->assert_is_ops_input(act_types);
  auto *act_op = pattern->NewNode(act_op_repr())->assert_is_ops(act_types);
  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");
  act_op->LinksFrom({out_var}).LinksTo({act_out_var});
  out_var = act_out_var;

  // Dropout1
  if (use_dropout_1) {
    out_var->assert_is_op_input("dropout", "X");
    auto *dropout_op_1 =
        pattern->NewNode(dropout_op_1_repr())->assert_is_op("dropout");
    auto *dropout_mask_var_1 = pattern->NewNode(dropout_mask_1_repr())
                                   ->assert_is_op_output("dropout", "Mask");
    auto *dropout_out_var_1 = pattern->NewNode(dropout_out_1_repr())
                                  ->assert_is_op_output("dropout", "Out");
    dropout_op_1->LinksFrom({out_var}).LinksTo(
        {dropout_mask_var_1, dropout_out_var_1});
    out_var = dropout_out_var_1;
  }

  // Linear2
  out_var->assert_is_op_input("matmul_v2", "X");
  auto *matmul_op_2 =
      pattern->NewNode(matmul_op_2_repr())->assert_is_op("matmul_v2");
  auto *matmul_w_var_2 =
      pattern->NewNode(matmul_w_2_repr())->assert_is_op_input("matmul_v2", "Y");
  auto *matmul_out_var_2 = pattern->NewNode(matmul_out_2_repr())
                               ->assert_is_op_output("matmul_v2", "Out");
  matmul_op_2->LinksFrom({out_var, matmul_w_var_2}).LinksTo({matmul_out_var_2});
  out_var = matmul_out_var_2;

  // Model parallel, do nothing in forward.
  if (use_mp) {
    out_var->assert_is_op_input("c_allreduce_sum", "X");
    auto *c_allreduce_sum_op = pattern->NewNode(c_allreduce_sum_op_repr())
                                   ->assert_is_op("c_allreduce_sum");
    auto *c_allreduce_sum_out_var =
        pattern->NewNode(c_allreduce_sum_out_repr())
            ->assert_is_op_output("c_allreduce_sum", "Out");
    c_allreduce_sum_op->LinksFrom({out_var}).LinksTo({c_allreduce_sum_out_var});
    out_var = c_allreduce_sum_out_var;
  }

  out_var->assert_is_op_input("elementwise_add", "X");
  auto *ele_add_op_2 =
      pattern->NewNode(ele_add_op_2_repr())->assert_is_op("elementwise_add");
  auto *ele_add_bias_var_2 = pattern->NewNode(ele_add_bias_2_repr())
                                 ->assert_is_op_input("elementwise_add", "Y");
  auto *ele_add_out_var_2 = pattern->NewNode(ele_add_out_2_repr())
                                ->assert_is_op_output("elementwise_add", "Out");
  ele_add_op_2->LinksFrom({out_var, ele_add_bias_var_2})
      .LinksTo({ele_add_out_var_2});
  out_var = ele_add_out_var_2;

  // Dropout 2
  if (use_dropout_2) {
    out_var->assert_is_op_input("dropout", "X");
    auto *dropout_op_2 =
        pattern->NewNode(dropout_op_2_repr())->assert_is_op("dropout");
    auto *dropout_mask_var_2 = pattern->NewNode(dropout_mask_2_repr())
                                   ->assert_is_op_output("dropout", "Mask");
    auto *dropout_out_var_2 = pattern->NewNode(dropout_out_2_repr())
                                  ->assert_is_op_output("dropout", "Out");
    dropout_op_2->LinksFrom({out_var}).LinksTo(
        {dropout_mask_var_2, dropout_out_var_2});
    out_var = dropout_out_var_2;
  }

  // Residual Add
  if (add_residual) {
    out_var->assert_is_op_input("elementwise_add", "X");
    x_var->assert_is_op_input("elementwise_add", "Y");
    auto *ele_add_op_3 =
        pattern->NewNode(ele_add_op_3_repr())->assert_is_op("elementwise_add");
    auto *ele_add_out_var_3 =
        pattern->NewNode(ele_add_out_3_repr())
            ->assert_is_op_output("elementwise_add", "Out");
    ele_add_op_3->LinksFrom({out_var, x_var}).LinksTo({ele_add_out_var_3});
    out_var = ele_add_out_var_3;
  }

  // Post LayerNorm
  if (!pre_layer_norm) {
    out_var->assert_is_op_input("layer_norm", "X");
    layer_norm_op
        ->LinksFrom({out_var, layer_norm_bias_var, layer_norm_scale_var})
        .LinksTo(
            {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});
    out_var = layer_norm_out_var;
  }
  return out_var;
}

PDNode *patterns::FusedFeedForwardBwd::operator()(
    paddle::framework::ir::PDNode *x_grad,
    std::unordered_set<std::string> act_grad_types,
    bool use_mp,
    bool pre_layer_norm,
    bool add_residual,
    bool use_dropout_1,
    bool use_dropout_2) {
  // Possible patterns
  // 1. residual_add_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
  // activation_grad -> linear1_grad -> layer_norm_grad
  // 2. layer_norm_grad -> residual_add_grad -> dropout2_grad -> linear2_grad ->
  // dropout1_grad -> activation_grad -> linear1_grad
  // other cases: may delete residual_add_grad, dropout1_grad, dropout2_grad
  // operators

  // intermediate input_grad, and final pattern output_grad
  PDNode *out_grad = x_grad;
  // LayerNorm: in["Mean", "Variance", "Scale", "Bias", "Y@GRAD"],
  // out["X@GRAD", "Scale@GRAD", "Bias@GRAD"]
  auto *layer_norm_op_grad = pattern->NewNode(layer_norm_op_grad_repr())
                                 ->assert_is_op("layer_norm_grad");
  auto *layer_norm_in_var = pattern->NewNode(layer_norm_in_repr())
                                ->assert_is_op_input("layer_norm_grad", "X");
  auto *layer_norm_mean_var =
      pattern->NewNode(layer_norm_mean_repr())
          ->assert_is_op_input("layer_norm_grad", "Mean");
  auto *layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->assert_is_op_input("layer_norm_grad", "Variance");
  auto *layer_norm_scale_var =
      pattern->NewNode(layer_norm_scale_repr())
          ->assert_is_op_input("layer_norm_grad", "Scale");
  auto *layer_norm_bias_var =
      pattern->NewNode(layer_norm_bias_repr())
          ->assert_is_op_input("layer_norm_grad", "Bias");
  auto *layer_norm_in_grad =
      pattern->NewNode(layer_norm_in_grad_repr())
          ->assert_is_op_output("layer_norm_grad", GradVarName("X"));
  auto *layer_norm_scale_grad =
      pattern->NewNode(layer_norm_scale_grad_repr())
          ->assert_is_op_output("layer_norm_grad", GradVarName("Scale"));
  auto *layer_norm_bias_grad =
      pattern->NewNode(layer_norm_bias_grad_repr())
          ->assert_is_op_output("layer_norm_grad", GradVarName("Bias"));
  // post_layer_norm
  if (!pre_layer_norm) {
    out_grad->assert_is_op_input("layer_norm_grad", GradVarName("Y"));
    layer_norm_op_grad
        ->LinksFrom({out_grad,
                     layer_norm_in_var,
                     layer_norm_mean_var,
                     layer_norm_variance_var,
                     layer_norm_scale_var,
                     layer_norm_bias_var})
        .LinksTo(
            {layer_norm_in_grad, layer_norm_scale_grad, layer_norm_bias_grad});
    out_grad = layer_norm_in_grad;
  }
  // partial input_grad of residual_add
  PDNode *tmp = nullptr;
  auto *matmul_in_var_1 = pattern->NewNode(matmul_in_1_repr())
                              ->assert_is_op_input("matmul_v2_grad", "X");
  if (add_residual) {
    // Residual Add: in["Out@GRAD", "X", "Y"], out["X@GRAD", "Y@GRAD"]
    out_grad->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));
    auto *ele_add_op_grad_3 = pattern->NewNode(ele_add_op_grad_3_repr())
                                  ->assert_is_op("elementwise_add_grad");
    auto *ele_add_in_var_3 =
        pattern->NewNode(ele_add_in_3_repr())
            ->assert_is_op_input("elementwise_add_grad", "X");
    auto *ele_add_in_grad_3 =
        pattern->NewNode(ele_add_in_grad_3_repr())
            ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));
    auto *ele_add_bias_grad_3 =
        pattern->NewNode(ele_add_bias_grad_3_repr())
            ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));
    tmp = ele_add_bias_grad_3;
    if (pre_layer_norm) {
      ele_add_op_grad_3
          ->LinksFrom({out_grad, ele_add_in_var_3, layer_norm_in_var})
          .LinksTo({ele_add_in_grad_3, ele_add_bias_grad_3});
    } else {
      ele_add_op_grad_3
          ->LinksFrom({out_grad, ele_add_in_var_3, matmul_in_var_1})
          .LinksTo({ele_add_in_grad_3, ele_add_bias_grad_3});
    }
    out_grad = ele_add_in_grad_3;
  }

  // Dropout 2: in["Out@GRAD", "Mask"], out["X@GRAD"]
  if (use_dropout_2) {
    out_grad->assert_is_op_input("dropout_grad", GradVarName("Out"));
    auto *dropout_op_grad_2 = pattern->NewNode(dropout_op_grad_2_repr())
                                  ->assert_is_op("dropout_grad");
    auto *dropout_mask_grad_2 =
        pattern->NewNode(dropout_mask_2_repr())
            ->assert_is_op_input("dropout_grad", "Mask");
    auto *dropout_in_grad_2 =
        pattern->NewNode(dropout_in_grad_2_repr())
            ->assert_is_op_output("dropout_grad", GradVarName("X"));
    dropout_op_grad_2->LinksFrom({out_grad, dropout_mask_grad_2})
        .LinksTo({dropout_in_grad_2});
    out_grad = dropout_in_grad_2;
  }

  // Linear 2:
  // elementwise_add: in["Out@GRAD", "X", "Y"], out["X@GRAD", "Y@GRAD"]
  out_grad->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));
  auto *ele_add_op_grad_2 = pattern->NewNode(ele_add_op_grad_2_repr())
                                ->assert_is_op("elementwise_add_grad");
  auto *ele_add_in_var_2 =
      pattern->NewNode(ele_add_in_2_repr())
          ->assert_is_op_input("elementwise_add_grad", "X");
  auto *ele_add_bias_var_2 =
      pattern->NewNode(ele_add_bias_2_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y");
  auto *ele_add_in_grad_2 =
      pattern->NewNode(ele_add_in_grad_2_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));
  auto *ele_add_bias_grad_2 =
      pattern->NewNode(ele_add_bias_grad_2_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));
  ele_add_op_grad_2->LinksFrom({out_grad, ele_add_in_var_2, ele_add_bias_var_2})
      .LinksTo({ele_add_in_grad_2, ele_add_bias_grad_2});
  out_grad = ele_add_in_grad_2;

  // Model parallel, do nothing in backward.
  if (use_mp) {
    out_grad->assert_is_op_input("c_identity", "X");
    auto *c_identity_op =
        pattern->NewNode(c_identity_op_repr())->assert_is_op("c_identity");
    auto *c_identity_out_grad = pattern->NewNode(c_identity_out_repr())
                                    ->assert_is_op_output("c_identity", "Out");
    c_identity_op->LinksFrom({out_grad}).LinksTo({c_identity_out_grad});
    out_grad = c_identity_out_grad;
  }

  // matmul_v2: in["Out@GRAD", "X", "Y"], out["X@GRAD", "Y@GRAD"]
  out_grad->assert_is_op_input("matmul_v2_grad", GradVarName("Out"));
  auto *matmul_op_grad_2 =
      pattern->NewNode(matmul_op_grad_2_repr())->assert_is_op("matmul_v2_grad");
  auto *matmul_in_var_2 = pattern->NewNode(matmul_in_2_repr())
                              ->assert_is_op_input("matmul_v2_grad", "X");
  auto *matmul_w_var_2 = pattern->NewNode(matmul_w_2_repr())
                             ->assert_is_op_input("matmul_v2_grad", "Y");
  auto *matmul_in_grad_2 =
      pattern->NewNode(matmul_in_grad_2_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("X"));
  auto *matmul_w_grad_2 =
      pattern->NewNode(matmul_w_grad_2_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("Y"));
  matmul_op_grad_2->LinksFrom({out_grad, matmul_in_var_2, matmul_w_var_2})
      .LinksTo({matmul_in_grad_2, matmul_w_grad_2});
  out_grad = matmul_in_grad_2;

  // Dropout 1: in["Out@GRAD", "Mask"], out["X@GRAD"]
  if (use_dropout_1) {
    out_grad->assert_is_op_input("dropout_grad", GradVarName("Out"));
    auto *dropout_op_grad_1 = pattern->NewNode(dropout_op_grad_1_repr())
                                  ->assert_is_op("dropout_grad");
    auto *dropout_mask_var_1 = pattern->NewNode(dropout_mask_1_repr())
                                   ->assert_is_op_input("dropout_grad", "Mask");
    auto *dropout_in_grad_1 =
        pattern->NewNode(dropout_in_grad_1_repr())
            ->assert_is_op_output("dropout_grad", GradVarName("X"));
    dropout_op_grad_1->LinksFrom({out_grad, dropout_mask_var_1})
        .LinksTo({dropout_in_grad_1});
    out_grad = dropout_in_grad_1;
  }

  // Activation: in["Out", "Out@GRAD"], out["X@GRAD"]
  out_grad->assert_is_ops_input(act_grad_types, GradVarName("Out"));
  auto *act_op_grad =
      pattern->NewNode(act_op_grad_repr())->assert_is_ops(act_grad_types);
  auto *act_in_var =
      pattern->NewNode(act_in_repr())->assert_is_ops_input(act_grad_types, "X");
  auto *act_in_grad =
      pattern->NewNode(act_in_grad_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"));
  act_op_grad->LinksFrom({out_grad, act_in_var}).LinksTo({act_in_grad});
  out_grad = act_in_grad;

  // Linear 1:
  // elementwise_add: in["Out@GRAD", "X", "Y"], out["X@GRAD", "Y@GRAD"]
  out_grad->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));
  auto *ele_add_op_grad_1 = pattern->NewNode(ele_add_op_grad_1_repr())
                                ->assert_is_op("elementwise_add_grad");
  auto *ele_add_in_var_1 =
      pattern->NewNode(ele_add_in_1_repr())
          ->assert_is_op_input("elementwise_add_grad", "X");
  auto *ele_add_bias_var_1 =
      pattern->NewNode(ele_add_bias_1_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y");
  auto *ele_add_in_grad_1 =
      pattern->NewNode(ele_add_in_grad_1_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));
  auto *ele_add_bias_grad_1 =
      pattern->NewNode(ele_add_bias_grad_1_repr())
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));
  ele_add_op_grad_1->LinksFrom({out_grad, ele_add_in_var_1, ele_add_bias_var_1})
      .LinksTo({ele_add_in_grad_1, ele_add_bias_grad_1});
  out_grad = ele_add_in_grad_1;
  // matmul_v2: in["Out@GRAD", "X", "Y"], out["X@GRAD", "Y@GRAD"]
  out_grad->assert_is_op_input("matmul_v2_grad", GradVarName("Out"));
  auto *matmul_op_grad_1 =
      pattern->NewNode(matmul_op_grad_1_repr())->assert_is_op("matmul_v2_grad");
  // auto *matmul_in_var_1 = pattern->NewNode(matmul_in_1_repr())
  //                                 ->assert_is_op_input("matmul_v2_grad",
  //                                 "X");
  auto *matmul_w_var_1 = pattern->NewNode(matmul_w_1_repr())
                             ->assert_is_op_input("matmul_v2_grad", "Y");
  auto *matmul_in_grad_1 =
      pattern->NewNode(matmul_in_grad_1_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("X"));
  auto *matmul_w_grad_1 =
      pattern->NewNode(matmul_w_grad_1_repr())
          ->assert_is_op_output("matmul_v2_grad", GradVarName("Y"));
  matmul_op_grad_1->LinksFrom({out_grad, matmul_in_var_1, matmul_w_var_1})
      .LinksTo({matmul_in_grad_1, matmul_w_grad_1});
  out_grad = matmul_in_grad_1;

  // Model parallel, all_reduce in backward.
  if (use_mp) {
    out_grad->assert_is_op_input("c_allreduce_sum", "X");
    auto *c_allreduce_sum_op = pattern->NewNode(c_allreduce_sum_op_repr())
                                   ->assert_is_op("c_allreduce_sum");
    auto *c_allreduce_sum_out_grad =
        pattern->NewNode(c_allreduce_sum_out_repr())
            ->assert_is_op_output("c_allreduce_sum", "Out");
    c_allreduce_sum_op->LinksFrom({out_grad})
        .LinksTo({c_allreduce_sum_out_grad});
    out_grad = c_allreduce_sum_out_grad;
  }

  // pre LayerNorm
  if (pre_layer_norm) {
    out_grad->assert_is_op_input("layer_norm_grad", GradVarName("Y"));
    layer_norm_op_grad
        ->LinksFrom({out_grad,
                     layer_norm_in_var,
                     layer_norm_mean_var,
                     layer_norm_variance_var,
                     layer_norm_scale_var,
                     layer_norm_bias_var})
        .LinksTo(
            {layer_norm_in_grad, layer_norm_scale_grad, layer_norm_bias_grad});
    out_grad = layer_norm_in_grad;
  }

  // sum for final gradient
  if (add_residual) {
    auto *sum_op = pattern->NewNode(sum_op_repr())->assert_is_op("sum");
    auto *sum_out =
        pattern->NewNode(sum_out_repr())->assert_is_op_output("sum", "Out");
    sum_op->LinksFrom({tmp, out_grad}).LinksTo({sum_out});
    out_grad = sum_out;
  }

  return out_grad;
}

void patterns::MulMatmulMatmulV2::operator()(
    const std::unordered_set<std::string> &ops_type) {
  auto ops = pattern->NewNode(ops_repr())->assert_is_ops(ops_type);
  auto ops_out = pattern->NewNode(ops_out_repr())
                     ->AsOutput()
                     ->assert_is_ops_output(ops_type, "Out");

  ops->LinksTo({ops_out});
}

// subgraph_edge_pattern
PDNode *patterns::SubgraphEdgePattern::operator()(
    const std::unordered_set<std::string> &ops_type) {
  auto ops = pattern->NewNode(ops_repr())->assert_is_ops(ops_type);
  return ops;
}

PDNode *patterns::ConvBNAddAct::operator()(
    const std::unordered_set<std::string> &act_types,
    bool shortcut,
    bool is_training) {
  // Conv1
  auto *x1 = pattern->NewNode(x1_repr())
                 ->assert_is_op_input("conv2d", "Input")
                 ->assert_var_dtype(proto::VarType::FP16);
  auto *conv1_w =
      pattern->NewNode(conv1_w_repr())->assert_is_op_input("conv2d", "Filter");
  auto *conv1_out = pattern->NewNode(conv1_out_repr())
                        ->assert_is_op_output("conv2d", "Output")
                        ->assert_is_op_input("batch_norm", "X");
  auto conv1_op = pattern->NewNode(conv1_op_repr())
                      ->assert_is_op("conv2d")
                      ->assert_op_attr<std::string>("data_format", "NHWC");
  // Conv2
  PDNode *x2 = nullptr;
  PDNode *conv2_w = nullptr;
  PDNode *conv2_out = nullptr;
  PDNode *conv2_op = nullptr;
  if (shortcut) {
    x2 = pattern->NewNode(x2_repr())
             ->assert_is_op_input("elementwise_add", "Y")
             ->assert_var_dtype(proto::VarType::FP16);
  } else {
    x2 = pattern->NewNode(x2_repr())->assert_is_op_input("conv2d", "Input");
    conv2_w = pattern->NewNode(conv2_w_repr())
                  ->assert_is_op_input("conv2d", "Filter");
    conv2_out = pattern->NewNode(conv2_out_repr())
                    ->assert_is_op_output("conv2d", "Output")
                    ->assert_is_op_input("batch_norm", "X");
    conv2_op = pattern->NewNode(conv2_op_repr())
                   ->assert_is_op("conv2d")
                   ->assert_op_attr<std::string>("data_format", "NHWC");
  }
  // BN1
  auto *bn1_scale_var = pattern->NewNode(bn1_scale_repr())
                            ->assert_is_op_input("batch_norm", "Scale");
  auto *bn1_bias_var = pattern->NewNode(bn1_bias_repr())
                           ->assert_is_op_input("batch_norm", "Bias");
  auto *bn1_variance_var = pattern->NewNode(bn1_variance_repr())
                               ->assert_is_op_input("batch_norm", "Variance");
  auto *bn1_mean_var = pattern->NewNode(bn1_mean_repr())
                           ->assert_is_op_input("batch_norm", "Mean");

  auto *bn1_op = pattern->NewNode(bn1_op_repr())
                     ->assert_is_op("batch_norm")
                     ->assert_is_not_op_input("MomentumTensor")
                     ->assert_op_attr<bool>("use_global_stats", false)
                     ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn1_mean_out_var = pattern->NewNode(bn1_mean_out_repr())
                               ->assert_is_op_output("batch_norm", "MeanOut");
  auto *bn1_variance_out_var =
      pattern->NewNode(bn1_variance_out_repr())
          ->assert_is_op_output("batch_norm", "VarianceOut");
  auto *bn1_saved_variance_var =
      pattern->NewNode(bn1_saved_variance_repr())
          ->assert_is_op_output("batch_norm", "SavedVariance");
  auto *bn1_saved_mean_var =
      pattern->NewNode(bn1_saved_mean_repr())
          ->assert_is_op_output("batch_norm", "SavedMean");
  auto *bn1_out_var =
      pattern->NewNode(bn1_out_repr())->assert_is_op_output("batch_norm", "Y");
  bn1_out_var->assert_is_op_input("elementwise_add", "X");

  // BN2
  PDNode *bn2_scale_var = nullptr;
  PDNode *bn2_bias_var = nullptr;
  PDNode *bn2_variance_var = nullptr;
  PDNode *bn2_mean_var = nullptr;
  PDNode *bn2_op = nullptr;
  PDNode *bn2_mean_out_var = nullptr;
  PDNode *bn2_variance_out_var = nullptr;
  PDNode *bn2_saved_variance_var = nullptr;
  PDNode *bn2_saved_mean_var = nullptr;
  PDNode *bn2_out_var = nullptr;

  if (!shortcut) {
    bn2_scale_var = pattern->NewNode(bn2_scale_repr())
                        ->assert_is_op_input("batch_norm", "Scale");
    bn2_bias_var = pattern->NewNode(bn2_bias_repr())
                       ->assert_is_op_input("batch_norm", "Bias");
    bn2_variance_var = pattern->NewNode(bn2_variance_repr())
                           ->assert_is_op_input("batch_norm", "Variance");
    bn2_mean_var = pattern->NewNode(bn2_mean_repr())
                       ->assert_is_op_input("batch_norm", "Mean");

    bn2_op = pattern->NewNode(bn2_op_repr())
                 ->assert_is_op("batch_norm")
                 ->assert_is_not_op_input("MomentumTensor")
                 ->assert_op_attr<bool>("use_global_stats", false)
                 ->assert_op_attr<std::string>("data_layout", "NHWC");

    bn2_mean_out_var = pattern->NewNode(bn2_mean_out_repr())
                           ->assert_is_op_output("batch_norm", "MeanOut");
    bn2_variance_out_var =
        pattern->NewNode(bn2_variance_out_repr())
            ->assert_is_op_output("batch_norm", "VarianceOut");
    bn2_saved_variance_var =
        pattern->NewNode(bn2_saved_variance_repr())
            ->assert_is_op_output("batch_norm", "SavedVariance");
    bn2_saved_mean_var = pattern->NewNode(bn2_saved_mean_repr())
                             ->assert_is_op_output("batch_norm", "SavedMean");
    bn2_out_var = pattern->NewNode(bn2_out_repr())
                      ->assert_is_op_output("batch_norm", "Y");
    bn2_out_var->assert_is_op_input("elementwise_add", "Y");
  }

  // Add
  auto *add_out = pattern->NewNode(add_out_repr())
                      ->assert_is_only_output_of_op("elementwise_add");
  auto *elewise_add_op =
      pattern->NewNode(elewise_add_op_repr())->assert_is_op("elementwise_add");
  // Act
  auto *act_op = pattern->NewNode(act_op_repr())->assert_is_ops(act_types);
  auto *act_out =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  // Links
  conv1_op->LinksFrom({x1, conv1_w}).LinksTo({conv1_out});
  bn1_op
      ->LinksFrom({conv1_out,
                   bn1_scale_var,
                   bn1_bias_var,
                   bn1_mean_var,
                   bn1_variance_var})
      .LinksTo({bn1_out_var,
                bn1_mean_out_var,
                bn1_variance_out_var,
                bn1_saved_mean_var,
                bn1_saved_variance_var});
  if (!shortcut) {
    conv2_op->LinksFrom({x2, conv2_w}).LinksTo({conv2_out});
    bn2_op
        ->LinksFrom({conv2_out,
                     bn2_scale_var,
                     bn2_bias_var,
                     bn2_mean_var,
                     bn2_variance_var})
        .LinksTo({bn2_out_var,
                  bn2_mean_out_var,
                  bn2_variance_out_var,
                  bn2_saved_mean_var,
                  bn2_saved_variance_var});
  }
  if (shortcut) {
    elewise_add_op->LinksFrom({bn1_out_var, x2}).LinksTo({add_out});
  } else {
    elewise_add_op->LinksFrom({bn1_out_var, bn2_out_var}).LinksTo({add_out});
  }
  act_op->LinksFrom({add_out}).LinksTo({act_out});

  // Note(tizheng): The backward fusion pattern is
  // dConv + Add + dReLU + dBN. Considering that forward
  // and backward fusion should be applied (or not applied)
  // simultaneously, we only fuse ConvBNAddAct with a following Conv2D.
  if (is_training) {
    act_out->assert_is_op_input("conv2d", "Input");
  } else {
    // For inference, avoid selecting this pattern in training programs
    conv1_out->AsIntermediate();
    bn1_out_var->AsIntermediate();
    if (!shortcut) {
      conv2_out->AsIntermediate();
      bn2_out_var->AsIntermediate();
    }
    add_out->AsIntermediate();
  }
  return act_out;
}

PDNode *patterns::ConvBNActConvBNStats::operator()(
    const std::unordered_set<std::string> &act_types, bool is_training) {
  // Conv
  auto *conv_x_var = pattern->NewNode(conv_x_repr())
                         ->assert_is_op_input("conv2d", "Input")
                         ->assert_var_dtype(proto::VarType::FP16);
  auto *conv_w_var =
      pattern->NewNode(conv_w_repr())->assert_is_op_input("conv2d", "Filter");
  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->assert_is_op_output("conv2d", "Output")
                           ->assert_is_op_input("batch_norm", "X");
  if (is_training) {
    // has link to bn_grad
    conv_out_var->assert_has_n_outputs(2);
  } else {
    conv_out_var->AsIntermediate();
  }
  auto conv_op = pattern->NewNode(conv_op_repr())
                     ->assert_is_op("conv2d")
                     ->assert_op_attr<std::string>("data_format", "NHWC");
  // BN
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm", "Bias");
  auto *bn_variance_var = pattern->NewNode(bn_variance_repr())
                              ->assert_is_op_input("batch_norm", "Variance");
  auto *bn_mean_var = pattern->NewNode(bn_mean_repr())
                          ->assert_is_op_input("batch_norm", "Mean");

  auto *bn_op = pattern->NewNode(bn_op_repr())
                    ->assert_is_op("batch_norm")
                    ->assert_is_not_op_input("MomentumTensor")
                    ->assert_op_attr<bool>("use_global_stats", false)
                    ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->assert_is_op_output("batch_norm", "MeanOut");
  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->assert_is_op_output("batch_norm", "VarianceOut");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_output("batch_norm", "SavedVariance");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_output("batch_norm", "SavedMean");
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->assert_is_op_output("batch_norm", "Y")
                         ->AsIntermediate();
  bn_out_var->assert_is_ops_input(act_types);
  // Act
  auto *act_op = pattern->NewNode(act_op_repr())->assert_is_ops(act_types);
  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");
  act_out_var->assert_is_op_input("fused_scale_bias_relu_conv_bn", "x");
  if (is_training) {
    // has link to conv_grad, relu_grad and fused_scale_bias_relu_conv_bn
    act_out_var->assert_has_n_outputs(3);
  } else {
    act_out_var->AsIntermediate();
  }
  // ConvBNStats
  auto *conv_bnstats_op = pattern->NewNode(conv_bnstats_op_repr())
                              ->assert_is_op("fused_scale_bias_relu_conv_bn")
                              ->assert_op_attr<bool>("fuse_prologue", false);

  // Links
  conv_op->LinksFrom({conv_x_var, conv_w_var}).LinksTo({conv_out_var});
  bn_op
      ->LinksFrom({conv_out_var,
                   bn_scale_var,
                   bn_bias_var,
                   bn_mean_var,
                   bn_variance_var})
      .LinksTo({bn_out_var,
                bn_mean_out_var,
                bn_variance_out_var,
                bn_saved_mean_var,
                bn_saved_variance_var});
  act_op->LinksFrom({bn_out_var}).LinksTo({act_out_var});
  conv_bnstats_op->LinksFrom({act_out_var});
  return act_out_var;
}

PDNode *patterns::BNActConvGrad::operator()(
    const std::unordered_set<std::string> &act_grad_types) {
  auto *d_conv_out_var =
      pattern->NewNode(d_conv_out_repr())
          ->assert_is_op_input("conv2d_grad", GradVarName("Output"));
  auto *conv_w_var = pattern->NewNode(conv_w_repr())
                         ->assert_is_op_input("conv2d_grad", "Filter");
  auto *d_conv_w_var =
      pattern->NewNode(d_conv_w_repr())
          ->assert_is_op_output("conv2d_grad", GradVarName("Filter"));
  auto *d_conv_x_var =
      pattern->NewNode(d_conv_x_repr())
          ->assert_is_op_output("conv2d_grad", GradVarName("Input"));
  d_conv_x_var->assert_is_ops_input(act_grad_types, GradVarName("Out"));
  // No conv_x because it is already deleted by forward pass
  auto *conv_grad = pattern->NewNode(conv_grad_repr())
                        ->assert_is_op("conv2d_grad")
                        ->assert_op_attr<std::string>("data_format", "NHWC")
                        ->assert_has_n_inputs(2);
  auto *act_grad =
      pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
  auto *bn_grad = pattern->NewNode(batch_norm_grad_repr())
                      ->assert_is_op("batch_norm_grad")
                      ->assert_op_attr<bool>("use_global_stats", false)
                      ->assert_op_attr<std::string>("data_layout", "NHWC");
  auto *d_act_x_var =
      pattern->NewNode(d_act_x_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"))
          ->assert_has_n_outputs(1)
          ->assert_var_dtype(proto::VarType::FP16);  // d_act_x

  d_act_x_var->AsIntermediate()->assert_is_op_input("batch_norm_grad",
                                                    GradVarName("Y"));

  auto *bn_x_var = pattern->NewNode(bn_x_repr())
                       ->assert_is_op_input("batch_norm_grad", "X")
                       ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm_grad", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm_grad", "Bias");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedMean");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedVariance");
  auto *d_bn_x_var =
      pattern->NewNode(d_bn_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("X"))
          ->assert_var_dtype(proto::VarType::FP16);
  auto *d_bn_scale_var =
      pattern->NewNode(d_bn_scale_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
  auto *d_bn_bias_var =
      pattern->NewNode(d_bn_bias_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));

  conv_grad->LinksFrom({d_conv_out_var, conv_w_var})
      .LinksTo({d_conv_w_var, d_conv_x_var});

  act_grad->LinksFrom({d_conv_x_var}).LinksTo({d_act_x_var});

  bn_grad
      ->LinksFrom({bn_x_var,
                   d_act_x_var,
                   bn_scale_var,
                   bn_bias_var,
                   bn_saved_mean_var,
                   bn_saved_variance_var})
      .LinksTo({d_bn_x_var, d_bn_scale_var, d_bn_bias_var});

  return bn_grad;
}

PDNode *patterns::BNAddActConvGrad::operator()(
    const std::unordered_set<std::string> &act_grad_types,
    bool shortcut,
    bool with_sum) {
  // dConv
  auto *d_conv_out_var =
      pattern->NewNode(d_conv_out_repr())
          ->assert_is_op_input("conv2d_grad", GradVarName("Output"));
  auto *conv_x_var = pattern->NewNode(conv_x_repr())
                         ->assert_is_op_input("conv2d_grad", "Input");
  conv_x_var->assert_is_ops_input(act_grad_types, "Out");
  auto *conv_w_var = pattern->NewNode(conv_w_repr())
                         ->assert_is_op_input("conv2d_grad", "Filter");
  auto *d_conv_w_var =
      pattern->NewNode(d_conv_w_repr())
          ->assert_is_op_output("conv2d_grad", GradVarName("Filter"));
  auto *d_conv_x_var =
      pattern->NewNode(d_conv_x_repr())
          ->assert_is_op_output("conv2d_grad", GradVarName("Input"));
  auto *conv_grad = pattern->NewNode(conv_grad_repr())
                        ->assert_is_op("conv2d_grad")
                        ->assert_op_attr<std::string>("data_format", "NHWC");
  // (optional) sum
  PDNode *sum_in_extra_var = nullptr;
  PDNode *sum_out_var = nullptr;
  PDNode *sum_op = nullptr;
  if (with_sum) {
    sum_op = pattern->NewNode(sum_repr())->assert_is_op("sum");
    d_conv_x_var->assert_is_op_input("sum");
    sum_in_extra_var =
        pattern->NewNode(sum_in_extra_repr())->assert_is_op_input("sum");
    sum_out_var =
        pattern->NewNode(sum_out_repr())->assert_is_only_output_of_op("sum");
    sum_out_var->assert_is_ops_input(act_grad_types, GradVarName("Out"));
  } else {
    d_conv_x_var->assert_is_ops_input(act_grad_types, GradVarName("Out"));
  }
  // dAct
  auto *act_grad =
      pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
  auto *d_act_x_var =
      pattern->NewNode(d_act_x_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"))
          ->assert_has_n_outputs(1)
          ->assert_var_dtype(proto::VarType::FP16);  // d_act_x

  d_act_x_var->AsIntermediate()->assert_is_op_input("elementwise_add_grad");

  // elementwise_add_grad
  auto *elewise_add_grad = pattern->NewNode(elewise_add_grad_repr())
                               ->assert_is_op("elementwise_add_grad");
  auto *d_elewise_add_x_var = pattern->NewNode(d_elewise_add_x_repr())
                                  ->assert_is_op_output("elementwise_add_grad");
  auto *d_elewise_add_y_var = pattern->NewNode(d_elewise_add_y_repr())
                                  ->assert_is_op_output("elementwise_add_grad");
  d_elewise_add_x_var->assert_is_op_input("batch_norm_grad", GradVarName("Y"));
  if (shortcut) {
    d_elewise_add_y_var->AsOutput();
  } else {
    d_elewise_add_y_var->assert_is_op_input("batch_norm_grad",
                                            GradVarName("Y"));
  }
  // dBN1
  auto *bn1_grad = pattern->NewNode(batch_norm1_grad_repr())
                       ->assert_is_op("batch_norm_grad")
                       ->assert_op_attr<bool>("use_global_stats", false)
                       ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn1_x_var = pattern->NewNode(bn1_x_repr())
                        ->assert_is_op_input("batch_norm_grad", "X")
                        ->assert_var_dtype(proto::VarType::FP16);
  auto *bn1_scale_var = pattern->NewNode(bn1_scale_repr())
                            ->assert_is_op_input("batch_norm_grad", "Scale");
  auto *bn1_bias_var = pattern->NewNode(bn1_bias_repr())
                           ->assert_is_op_input("batch_norm_grad", "Bias");
  auto *bn1_saved_mean_var =
      pattern->NewNode(bn1_saved_mean_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedMean");
  auto *bn1_saved_variance_var =
      pattern->NewNode(bn1_saved_variance_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedVariance");
  auto *d_bn1_x_var =
      pattern->NewNode(d_bn1_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("X"))
          ->assert_var_dtype(proto::VarType::FP16);
  auto *d_bn1_scale_var =
      pattern->NewNode(d_bn1_scale_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
  auto *d_bn1_bias_var =
      pattern->NewNode(d_bn1_bias_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));
  // dBN2
  PDNode *bn2_grad = nullptr;
  PDNode *bn2_x_var = nullptr;
  PDNode *bn2_scale_var = nullptr;
  PDNode *bn2_bias_var = nullptr;
  PDNode *bn2_saved_mean_var = nullptr;
  PDNode *bn2_saved_variance_var = nullptr;
  PDNode *d_bn2_x_var = nullptr;
  PDNode *d_bn2_scale_var = nullptr;
  PDNode *d_bn2_bias_var = nullptr;
  if (!shortcut) {
    bn2_grad = pattern->NewNode(batch_norm2_grad_repr())
                   ->assert_is_op("batch_norm_grad")
                   ->assert_op_attr<bool>("use_global_stats", false)
                   ->assert_op_attr<std::string>("data_layout", "NHWC");

    bn2_x_var = pattern->NewNode(bn2_x_repr())
                    ->assert_is_op_input("batch_norm_grad", "X")
                    ->assert_var_dtype(proto::VarType::FP16);
    bn2_scale_var = pattern->NewNode(bn2_scale_repr())
                        ->assert_is_op_input("batch_norm_grad", "Scale");
    bn2_bias_var = pattern->NewNode(bn2_bias_repr())
                       ->assert_is_op_input("batch_norm_grad", "Bias");
    bn2_saved_mean_var =
        pattern->NewNode(bn2_saved_mean_repr())
            ->assert_is_op_input("batch_norm_grad", "SavedMean");
    bn2_saved_variance_var =
        pattern->NewNode(bn2_saved_variance_repr())
            ->assert_is_op_input("batch_norm_grad", "SavedVariance");
    d_bn2_x_var = pattern->NewNode(d_bn2_x_repr())
                      ->assert_is_not_ctrl_var()
                      ->assert_is_op_output("batch_norm_grad", GradVarName("X"))
                      ->assert_var_dtype(proto::VarType::FP16);
    d_bn2_scale_var =
        pattern->NewNode(d_bn2_scale_repr())
            ->assert_is_not_ctrl_var()
            ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
    d_bn2_bias_var =
        pattern->NewNode(d_bn2_bias_repr())
            ->assert_is_not_ctrl_var()
            ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));
  }

  conv_grad->LinksFrom({d_conv_out_var, conv_x_var, conv_w_var})
      .LinksTo({d_conv_w_var, d_conv_x_var});
  if (with_sum) {
    sum_op->LinksFrom({d_conv_x_var, sum_in_extra_var}).LinksTo({sum_out_var});
    act_grad->LinksFrom({sum_out_var, conv_x_var}).LinksTo({d_act_x_var});
  } else {
    act_grad->LinksFrom({d_conv_x_var, conv_x_var}).LinksTo({d_act_x_var});
  }

  elewise_add_grad->LinksFrom({d_act_x_var})
      .LinksTo({d_elewise_add_x_var, d_elewise_add_y_var});

  bn1_grad
      ->LinksFrom({d_elewise_add_x_var,
                   bn1_x_var,
                   bn1_scale_var,
                   bn1_bias_var,
                   bn1_saved_mean_var,
                   bn1_saved_variance_var})
      .LinksTo({d_bn1_x_var, d_bn1_scale_var, d_bn1_bias_var});
  if (!shortcut) {
    bn2_grad
        ->LinksFrom({d_elewise_add_y_var,
                     bn2_x_var,
                     bn2_scale_var,
                     bn2_bias_var,
                     bn2_saved_mean_var,
                     bn2_saved_variance_var})
        .LinksTo({d_bn2_x_var, d_bn2_scale_var, d_bn2_bias_var});
  }
  return bn1_grad;
}

void patterns::SparseConvOptimPartern::operator()() {
  auto sp_conv3d_x = pattern->NewNode(sp_conv3d_x_repr())
                         ->AsInput()
                         ->assert_is_op_input("sparse_conv3d", "x");
  auto sp_conv3d_kernel = pattern->NewNode(sp_conv3d_kernel_repr())
                              ->AsInput()
                              ->assert_is_op_input("sparse_conv3d", "kernel");
  auto sp_conv3d_op =
      pattern->NewNode(sp_conv3d_op_repr())->assert_is_op("sparse_conv3d");
  auto sp_conv3d_out = pattern->NewNode(sp_conv3d_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("sparse_conv3d", "out");

  sp_conv3d_op->LinksFrom({sp_conv3d_x, sp_conv3d_kernel})
      .LinksTo({sp_conv3d_out});
}

}  // namespace paddle::framework::ir
