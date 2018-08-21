/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * Data flow graph is an pass that build the basic graph. It contains a graph
 * and the iterators that enable the iteration over the graph.
 */

#pragma once

#include <deque>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/inference/analysis/graph_traits.h"
#include "paddle/fluid/inference/analysis/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * DataFlowGraph - A container of Value and Function Nodes.
 *
 * This is the base graph for any other type of graphs, such as SSA or CFG.
 */
struct DataFlowGraph {
  NodeMap nodes;
  // inputs and outputs are deduced from the graph.
  // Used to interact with IR.
  const framework::ir::Graph *ir_graph{nullptr};

  // Extract inputs and outputs of the graph.
  void Build();

  void Build(const framework::proto::ProgramDesc &prog);

  // Build a graph from ir::Graph.
  void Build(const framework::ir::Graph &graph);

  bool IsFullyConnected() const;

  // Output a DOT graph file for debug.
  std::string DotString() const;

  std::string HumanReadableInfo(bool show_values = true,
                                bool show_functions = true) const;

  const std::vector<Node *> &inputs() const {
    PADDLE_ENFORCE(!inputs_.empty(),
                   "No inputs are deduced, need to Build() first.");
    return inputs_;
  }
  const std::vector<Node *> &outputs() const {
    PADDLE_ENFORCE(!outputs_.empty(),
                   "No outputs are deduced, need to Build() first.");
    return outputs_;
  }

 private:
  mutable std::vector<Node *> inputs_;
  mutable std::vector<Node *> outputs_;

  // Remove duplicate edges and so on.
  void Clean();
};

/*
 * An graph trait help to traverse the graph using BFS.
 * The BFS start from a graph's inputs, the graph should be fully-connected, so
 * that the iterator can reach the end.
 */
template <>
struct GraphTraits<DataFlowGraph> {
  // BFS iterator on nodes.
  struct NodesBFSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesBFSIterator() = default;
    explicit NodesBFSIterator(const std::vector<Node *> &source,
                              bool directive);
    NodesBFSIterator(NodesBFSIterator &&other) noexcept;
    // NOTE Heavy to use.
    NodesBFSIterator(const NodesBFSIterator &other);

    Node &operator*();
    NodesBFSIterator &operator++();
    Node *operator->();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesBFSIterator &operator=(const NodesBFSIterator &other);
    bool operator==(const NodesBFSIterator &other);
    bool operator!=(const NodesBFSIterator &other) { return !(*this == other); }

   private:
    bool directive_;
    std::deque<Node *> queue_;
    std::unordered_set<Node *> visited_;
  };

  // DFS iterator on nodes.
  struct NodesDFSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesDFSIterator() = default;
    NodesDFSIterator(const std::vector<Node *> &source, bool directive);
    NodesDFSIterator(NodesDFSIterator &&other) noexcept;
    NodesDFSIterator(const NodesDFSIterator &other);

    Node &operator*();
    NodesDFSIterator &operator++();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesDFSIterator &operator=(const NodesDFSIterator &other);
    bool operator==(const NodesDFSIterator &other);
    bool operator!=(const NodesDFSIterator &other) { return !(*this == other); }
    Node *operator->();

   private:
    bool directive_;
    std::stack<Node *> stack_;
    std::unordered_set<Node *> visited_;
  };

  // Topological sorting iterator on nodes.
  struct NodesTSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesTSIterator() = default;
    NodesTSIterator(const std::vector<Node *> &source, bool directive);
    NodesTSIterator(NodesTSIterator &&other)
        : directive_(other.directive_),
          sorted_(std::move(other.sorted_)),
          cursor_(other.cursor_) {
      other.cursor_ = 0;
    }
    NodesTSIterator(const NodesTSIterator &other);

    Node &operator*();
    NodesTSIterator &operator++();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesTSIterator &operator=(const NodesTSIterator &other);
    bool operator==(const NodesTSIterator &other);
    bool operator!=(const NodesTSIterator &other) { return !(*this == other); }
    Node *operator->();

   private:
    bool directive_;
    std::vector<Node *> sorted_;
    size_t cursor_{0};
  };

  explicit GraphTraits(const DataFlowGraph &graph, bool directive = true)
      : graph_(graph), directive_(directive) {}

  // default use BFS to visit the nodes.
  iterator_range<NodesBFSIterator> nodes() {
    return iterator_range<NodesBFSIterator>(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range<NodesBFSIterator> nodes_in_BFS() {
    return iterator_range<NodesBFSIterator>(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range<NodesDFSIterator> nodes_in_DFS() {
    return iterator_range<NodesDFSIterator>(nodes_dfs_begin(), nodes_dfs_end());
  }
  iterator_range<NodesTSIterator> nodes_in_TS() {
    return iterator_range<NodesTSIterator>(nodes_ts_begin(), nodes_ts_end());
  }

 private:
  NodesBFSIterator nodes_bfs_begin() {
    return NodesBFSIterator(graph_.inputs(), directive_);
  }
  NodesBFSIterator nodes_bfs_end() { return NodesBFSIterator(); }

  NodesDFSIterator nodes_dfs_begin() {
    return NodesDFSIterator(graph_.inputs(), directive_);
  }
  NodesDFSIterator nodes_dfs_end() { return NodesDFSIterator(); }

  NodesTSIterator nodes_ts_begin() {
    return NodesTSIterator(graph_.inputs(), directive_);
  }
  NodesTSIterator nodes_ts_end() { return NodesTSIterator(); }

 private:
  const DataFlowGraph &graph_;
  bool directive_;
};

// Extract the inputs and outputs of a graph. The inputs and outputs of a
// sub-graph is the inputs nodes and output nodes that doesn't inside the
// sub-graph.
std::pair<std::vector<Node *>, std::vector<Node *>>
ExtractInputAndOutputOfSubGraph(std::vector<Node *> &graph);  // NOLINT

void FilterRedundantOutputOfSubGraph(DataFlowGraph *graph);
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
