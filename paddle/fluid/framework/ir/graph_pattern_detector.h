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

#pragma once

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest_prod.h>
#endif

#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/inference/analysis/dot.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
class PDPattern;

// Some basic terminologies:
//   - PDPattern: a pattern defined as a data flow graph.
//   - PDNode: the node in the pattern, each PDNode represents an `ir::Node`
//     that meets some conditions defined in `PDNode.teller`.
//   - A pattern is defined with PDNodes with edges.

// Pattern detector node. This node helps to build a pattern.
struct PDNode {
  // tell whether an ir::Node* is a candidation for a PDNode.
  using teller_t = std::function<bool(Node*)>;
  enum class Type { kOp, kVar };
  enum class Role {
    kUnknown,      // No role,
    kInput,        // an input and will be retained,
    kOutput,       // an output and will be retained,
    kIntermediate  // will be removed after handler.
  };

  // this link to others
  PDNode& LinksTo(const std::vector<PDNode*>& others);
  PDNode& LinksFrom(const std::vector<PDNode*>& others);

  bool Tell(Node* node) const {
    if (teller_) return teller_(node);

    for (auto& asrt : asserts_) {
      if (!asrt(node)) return false;
    }
    return true;
  }

  bool IsOp() const { return type_ == Type::kOp; }
  bool IsVar() const { return type_ == Type::kVar; }

  const std::string& name() const { return name_; }
  const PDPattern* pdpattern() const { return pattern_; }

  PDNode& operator=(const PDNode&) = delete;
  PDNode(const PDNode&) = delete;

  // Mark this node is an Input of a subgraph and will be retained.
  PDNode* AsInput() {
    role_ = Role::kInput;
    return this;
  }
  // Mark this node is an Output of a subgraph and will be retained.
  PDNode* AsOutput() {
    role_ = Role::kOutput;
    return this;
  }
  // Mark this node will be removed, so all the links should be inside a matched
  // sub-graph.
  PDNode* AsIntermediate() {
    role_ = Role::kIntermediate;
    return this;
  }

  bool IsIntermediate() const { return role_ == Role::kIntermediate; }
  bool IsInput() const { return role_ == Role::kInput; }
  bool IsOutput() const { return role_ == Role::kOutput; }

  // Assertions, helper functions to simplify the pattern definition.
  PDNode* assert_is_op();
  PDNode* assert_is_op(const std::string& op_type);
  PDNode* assert_is_not_op_type(const std::string& op_type);
  PDNode* assert_is_var();
  PDNode* assert_var_dtype(proto::VarType::Type dtype);
  PDNode* assert_is_not_ctrl_var();
  PDNode* assert_var_not_persistable();
  PDNode* assert_is_persistable_var();
  PDNode* assert_is_op_output(const std::string& op_type);
  PDNode* assert_is_op_output(const std::string& op_type,
                              const std::string& argument);
  PDNode* assert_is_op_input(const std::string& op_type);
  PDNode* assert_is_op_input(const std::string& op_type,
                             const std::string& argument);
  PDNode* assert_is_op_nth_input(const std::string& op_type,
                                 const std::string& argument,
                                 int nth);
  PDNode* assert_is_not_op_input(const std::string& argument);
  PDNode* assert_is_op_nth_output(const std::string& op_type,
                                  const std::string& argument,
                                  int nth);
  PDNode* assert_is_only_input_of_op(const std::string& op_type);
  PDNode* assert_is_only_output_of_op(const std::string& op_type);
  PDNode* assert_op_has_n_inputs(const std::string& op_type, size_t n);
  PDNode* assert_op_has_n_outputs(const std::string& op_type, size_t n);
  PDNode* assert_more(teller_t&& teller);

  PDNode* assert_is_ops_output(const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_ops(const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_ops_output(const std::unordered_set<std::string>& op_types,
                               const std::string& argument);
  PDNode* assert_is_ops_nth_input(
      const std::unordered_set<std::string>& op_types,
      const std::string& argument,
      int nth);
  PDNode* assert_is_ops_input(const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_ops_input(const std::unordered_set<std::string>& op_types,
                              const std::string& argument);
  PDNode* assert_is_ops_nth_output(
      const std::unordered_set<std::string>& op_types,
      const std::string& argument,
      int nth);

  PDNode* assert_is_only_input_of_ops(
      const std::unordered_set<std::string>& op_types);
  PDNode* assert_is_only_output_of_ops(
      const std::unordered_set<std::string>& op_types);

  PDNode* assert_has_n_inputs(size_t n);
  PDNode* assert_has_n_outputs(size_t n);

  template <typename T>
  PDNode* assert_op_attr(const std::string& attr_name, const T& attr) {
    asserts_.emplace_back([=](Node* x) {
      return x && x->IsOp() && x->Op()->HasAttr(attr_name) &&
             PADDLE_GET_CONST(T, x->Op()->GetAttr(attr_name)) == attr;
    });
    return this;
  }

 private:
  PDNode(PDPattern* pattern,
         const std::string& name = "",
         Type type = Type::kVar)
      : pattern_(pattern), name_(name), type_(type) {}
  PDNode(teller_t&& teller,
         PDPattern* pattern,
         const std::string& name = "",
         Type type = Type::kVar)
      : teller_(std::move(teller)),
        pattern_(pattern),
        name_(name),
        type_(type) {
    PADDLE_ENFORCE_NOT_NULL(
        teller_,
        common::errors::NotFound("invalid teller is set, teller is null"));
  }

  PDNode(PDNode&& other) = default;

  friend class PDPattern;

  // Will removed latter.
  teller_t teller_;
  std::vector<teller_t> asserts_;
  PDPattern* pattern_;
  std::string name_;
  Type type_;
  Role role_{Role::kUnknown};
};

/*
 * A pattern in a graph, which defined with PDNode and edges. Most graph
 * patterns can be divided into PDNodes and link relations between them.
 *
 * For example, the FC fusion need to filter the MUL and ELEMENTWISE_ADD
 * operators from the computation graph, the MUL's output should have only one
 * consumer which is the ELEMENTWISE_ADD.
 * This pattern can be defined as with the following pseudo codes
 *
 *     // Create two operator PDNodes.
 *     MUL = PDPattern.NewNode().assert_is_op("mul");
 *     ELE = PDPattern.NewNode().assert_is_op("elementwise_add");
 *     // Create the variable PDNodes.
 *     MUL_out = PDPattern.NewNode().assert_is_op_output("mul") \
 *                                  .assert_is_op_input("elementwise_add") \
 *                                  .AsIntermediate();
 *     // Add relations.
 *     MUL->LinksTo({MUL_out});
 *     MUL_out->LinksTo({ELE});
 *
 * One can add more specific asserts for PDNodes or edges, both the Operator
 * and Variable Nodes can be ruled in PDNode.assert_more(...).
 *
 * PDPattern can record the general patterns, such as the pattern represents
 *   - Op in CPU -> Op in GPU -> Op in CPU, to find out the IO abnormal place.
 *   - Ops whose inputs and outputs share the same variables
 */
class PDPattern {
 public:
  using edge_t = std::pair<PDNode*, PDNode*>;

  void AddEdge(PDNode* a, PDNode* b);

  PDNode* NewNode(PDNode::teller_t&& teller, const std::string& name = NewID());
  PDNode* NewNode(const std::string& name = NewID());
  PDNode* NewNode(const std::string& prefix, const std::string& name) {
    return NewNode(prefix + "/" + name);
  }
  PDNode* RetrieveNode(const std::string& id) const;

  const std::vector<std::unique_ptr<PDNode>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

  std::string DotString() const;

 private:
#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(PDPattern, AddEdge);
  FRIEND_TEST(PDPattern, NewNode);
#endif

  static std::string NewID() { return "pdnode-" + std::to_string(id_++); }

  std::vector<std::unique_ptr<PDNode>> nodes_;
  std::vector<edge_t> edges_;
  std::map<std::string, PDNode*> node_map_;
  static size_t id_;
};

/*
 * GraphPatternDetector helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 *
 * The algorithm has three phases:
 *   1. Mark the nodes that match the defined PDNodes in a PDPattern,
 *   2. Extend a PDNode to subgraphs by deducing the connection relation defined
 *      in PAPattern(the edges),
 *   3. Get the filtered subgraphs and treat them with a pre-defined handler.
 *
 * Usage:
 *    // Create a detector
 *    GraphPatternDetector detector;
 *    // Define the detector's pattern, by adding PDNode and define the edges.
 *    auto* node0 = detector.mutable_pattern().AddNode(...)
 *    auto* node1 = detector.mutable_pattern().AddNode(...)
 *    node0->teller = some lambda.
 *    node1->teller = some lambda.
 *    detector.mutable_pattern().AddEdge(node0, node1);
 *    // Create an handler, to define the behavior of treating the filtered
 *    // subgraphs that comply with the patterns.
 *    GraphPatternDetector::handle_t handler = some lambda
 *    // Execute the detector.
 *    detector(&graph, handler);
 */
class GraphPatternDetector {
 public:
  struct NodeIdCompare {
    bool operator()(Node* node1, Node* node2) const {
      return node1->id() < node2->id();
    }
  };

  struct PDNodeCompare {
    bool operator()(const PDNode* node1, const PDNode* node2) const {
      auto& nodes1 = node1->pdpattern()->nodes();
      auto& nodes2 = node2->pdpattern()->nodes();
      if (nodes1.size() != nodes2.size()) {
        return nodes1.size() < nodes2.size();
      } else {
        std::string pdnode_hash_key1 = "";
        std::string pdnode_hash_key2 = "";
        for (auto& node : nodes1) {
          pdnode_hash_key1 += node.get()->name();
          pdnode_hash_key1 += "#";
        }
        pdnode_hash_key1 += node1->name();
        for (auto& node : nodes2) {
          pdnode_hash_key2 += node.get()->name();
          pdnode_hash_key2 += "#";
        }
        pdnode_hash_key2 += node2->name();

        auto pdnode_key1 =
            std::to_string(std::hash<std::string>()(pdnode_hash_key1));
        auto pdnode_key2 =
            std::to_string(std::hash<std::string>()(pdnode_hash_key2));

        return pdnode_key1 < pdnode_key2;
      }
      return false;
    }
  };

  using subgraph_t = std::map<PDNode*, Node*, PDNodeCompare>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, Graph*)>;

  void operator()(Graph* graph, handle_t handler);

  const PDPattern& pattern() const { return pattern_; }
  PDPattern* mutable_pattern() { return &pattern_; }

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPDNodesInGraph(const ir::Graph& graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns.
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

  // Sort subgraphs, sort subgraphs by the specified node so that
  // the removed forward and backward subgraphs are corresponding
  // when two subgraphs are overlapped. Note: this function is
  // currently only used for bn_add_act, refer to PR28196 for details.
  void SortSubgraphs(std::vector<subgraph_t>* subgraphs);

  // Remove overlapped match subgraphs, when overlapped, keep the previous one.
  // The intermediate PDNodes will be removed, so can't shared by multiple
  // patterns.
  void RemoveOverlappedMatch(std::vector<subgraph_t>* subgraphs);

  // Validate whether the intermediate nodes are linked by external nodes.
  void ValidateByNodeRole(std::vector<subgraph_t>* subgraphs);

#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(GraphPatternDetecter, MarkPDNodesInGraph);
  FRIEND_TEST(GraphPatternDetecter, DetectPatterns);
#endif

 private:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PDNode* /*node in pattern*/>;
  PDPattern pattern_;
  std::map<const PDNode*, std::set<Node*, NodeIdCompare>, PDNodeCompare>
      pdnodes2nodes_;
};

// some helper methods.

// Tell if a var links to an Op
bool VarLinksToOp(Node* node, const std::string& op_type);

// Tell if an op links to a var
bool VarLinksFromOp(Node* node, const std::string& op_type);

// Check whether a var node is a op node's nth input.
bool IsNthInput(Node* var, Node* op, const std::string& argument, size_t nth);

// Check whether the op node has input of given name.
bool HasInput(Node* op, const std::string& argument);

// Check whether the op node has output of given name.
bool HasOutput(Node* op, const std::string& argument);

// Tell whether a var node is a op node's nth output.
bool IsNthOutput(Node* var, Node* op, const std::string& argument, size_t nth);

// Graph safely remove some nodes, will automatically clean up the edges.
void GraphSafeRemoveNodes(
    Graph* graph,
    const std::unordered_set<const Node*>& nodes,
    std::unordered_set<std::shared_ptr<Node>>* saved_nodes = nullptr);

// Some pre-defined patterns those can be reused in multiple passes.
// The related Fluid Layer or Op should be one pattern here for better re-usage
// across different fusion.
namespace patterns {

struct KeyCounter {
  static KeyCounter& Instance() {
    static KeyCounter x;
    return x;
  }

#ifdef PADDLE_WITH_TENSORRT
  static int IncCounter(const std::string& key) { return dic_[key]++; }
  static void CleanCounter() { dic_.clear(); }

 private:
  static thread_local std::unordered_map<std::string, size_t> dic_;
#else
  int IncCounter(const std::string& key) { return dic_[key]++; }

 private:
  std::unordered_map<std::string, size_t> dic_;
#endif
};

// Generate a unique PDNode's name with name_scope and id.
// The format is {name_scope}/{repr}/{id}/{name}
static std::string PDNodeName(const std::string& name_scope,
                              const std::string& repr,
                              size_t id,
                              const std::string& name) {
  return string::Sprintf("%s/%s/%d/%s", name_scope, repr, id, name);
}
// Generate a unique PDNode's name.
// The format is {name_scope}/{repr}/{id}
static std::string PDNodeName(const std::string& name_scope,
                              const std::string& repr) {
  return string::Sprintf(
      "%s/%s/%d", name_scope, repr, KeyCounter::Instance().IncCounter(repr));
}
// Generate a unique key. It can be used for a universally unique temporary
// name.
// The format is {repr}/{id}
static std::string UniqueKey(const std::string& repr) {
  return string::Sprintf(
      "%s/%d", repr, KeyCounter::Instance().IncCounter(repr));
}

// Declare a PDNode in a pattern, will create two methods:
// std::string xxx_repr(); return this PDNode's string id.
// PDNode* xxx_n(); return the corresponding PDNode.
#define PATTERN_DECL_NODE(name__)                        \
  std::string name__##_repr() const {                    \
    return PDNodeName(name_scope_, repr_, id_, #name__); \
  }                                                      \
  PDNode* name__##_n() const { return pattern->RetrieveNode(name__##_repr()); }

// Get an ir::Node* from the matched subgraph.
// var: variable.
// arg: the argument declared by PATTERN_DECL_NODE in a pattern definition.
// pat: the pattern object.
#define GET_IR_NODE_FROM_SUBGRAPH(var, arg, pat)                             \
  PADDLE_ENFORCE_NE(subgraph.count(pat.arg##_n()),                           \
                    0UL,                                                     \
                    common::errors::NotFound("Node not found for PDNode %s", \
                                             pat.arg##_repr()));             \
  Node* var = subgraph.at(pat.arg##_n());                                    \
  PADDLE_ENFORCE_NOT_NULL(                                                   \
      var,                                                                   \
      common::errors::NotFound("node %s not exists in the sub-graph", #arg));

// The base class of all the patterns.
struct PatternBase {
  PatternBase(PDPattern* pattern,
              const std::string& name_scope,
              const std::string& repr)
      : pattern(pattern),
        name_scope_(name_scope),
        repr_(repr),
        id_(KeyCounter::Instance().IncCounter(repr)) {}

  PDPattern* pattern;

 protected:
  std::string name_scope_;
  std::string repr_;
  size_t id_;
};

// Conv with batch norm
// op: conv + (elementwise_add +) batch_norm
// named nodes:
// conv_weight, conv_out, conv,
// bn_x, bn_scale, bn_bias, bn_mean,  bn_variance,
// bn_batch_norm, bn_y, bn_mean_out, bn_variance_out,
// bn_saved_mean, bn_saved_variance
struct ConvBN : public PatternBase {
  ConvBN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_bn") {}

  PDNode* operator()(PDNode* conv_input,
                     const std::string& conv_type,
                     bool with_eltwise_add);

  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(eltwise);  // ELEMENTWISE_ADD
  // CONV inputs
  PATTERN_DECL_NODE(conv_weight);  // Filter
  // CONV outputs
  PATTERN_DECL_NODE(conv_out);  // tmp
  // ELTWISE inputs
  PATTERN_DECL_NODE(eltwise_y_in);
  // ELTWISE outputs
  PATTERN_DECL_NODE(eltwise_out);  // tmp
  // BN inputs
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_variance);
  // BN outputs
  PATTERN_DECL_NODE(bn_out);  // Out
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
};

struct OperatorActivation : public PatternBase {
  OperatorActivation(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "operator_activation") {}

  PDNode* operator()(const std::string& operator_type,
                     const std::string& activation_type);

  PATTERN_DECL_NODE(preceding_op);
  PATTERN_DECL_NODE(preceding_op_out);
  PATTERN_DECL_NODE(activation);
  PATTERN_DECL_NODE(activation_out);
};

struct QuantTranspose : public PatternBase {
  QuantTranspose(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quant_transpose") {}

  PDNode* operator()(const std::string& transpose_type);

  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
  PATTERN_DECL_NODE(transpose_op);
};

struct TransposeDequant : public PatternBase {
  TransposeDequant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "transpose_dequant") {}
  PDNode* operator()(const std::string& transpose_type);

  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
};

struct Squeeze2Transpose2 : public PatternBase {
  Squeeze2Transpose2(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "squeeze2_transpose2") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(squeeze2_op_in);
  PATTERN_DECL_NODE(squeeze2_op);
  PATTERN_DECL_NODE(squeeze2_op_out);
  PATTERN_DECL_NODE(transpose2_op);
};

struct OperatorUnsqueeze2 : public PatternBase {
  OperatorUnsqueeze2(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "operator_unsqueeze2") {}

  PDNode* operator()(const std::string& operator_type,
                     const int num_of_outputs);

  PATTERN_DECL_NODE(preceding_op);
  PATTERN_DECL_NODE(preceding_op_out);
  PATTERN_DECL_NODE(unsqueeze2_op);
  PATTERN_DECL_NODE(unsqueeze2_out);
};

struct OperatorReshape2 : public PatternBase {
  OperatorReshape2(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "operator_reshape2") {}

  PDNode* operator()(const std::string& operator_type,
                     const int num_of_outputs);

  PATTERN_DECL_NODE(preceding_op);
  PATTERN_DECL_NODE(preceding_op_out);
  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(reshape2_out);
};

// SEQCONV with Elementwise_Add ReLU
// op: seqconv + elementwise_add + relu
// named nodes:
// seqconv_input, seqconv_weight,
// seqconv_out, seqconv,
// elementwise_add_bias, elementwise_add_out, elementwise_add
// relu_out, relu
struct SeqConvEltAddRelu : public PatternBase {
  SeqConvEltAddRelu(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "seqconv_eltadd_relu") {}

  PDNode* operator()(PDNode* seqconv_input);

  // declare operator node's name
  PATTERN_DECL_NODE(seqconv);
  PATTERN_DECL_NODE(eltadd);
  PATTERN_DECL_NODE(relu);
  // declare variable node's name
  PATTERN_DECL_NODE(seqconv_weight);
  PATTERN_DECL_NODE(seqconv_out);
  PATTERN_DECL_NODE(eltadd_bias);
  PATTERN_DECL_NODE(eltadd_out);
  PATTERN_DECL_NODE(relu_out);
};

// FC with bias
// op: mul + elementwise_add
// named nodes:
// mul, elementwise_add
// w, mul_out, bias, fc_out
struct FC : public PatternBase {
  FC(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fc") {}

  PDNode* operator()(PDNode* x, bool with_bias, bool with_relu);

  // declare operator node's name
  PATTERN_DECL_NODE(fc);
  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(elementwise_add);
  PATTERN_DECL_NODE(relu);
  // declare variable node's name
  PATTERN_DECL_NODE(w);
  PATTERN_DECL_NODE(mul_out);  // (x,w) -> mul_out
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(elementwise_add_out);
  PATTERN_DECL_NODE(relu_out);
};

// MKL-DNN's FC with bias
// op: fc
// named node:
// fc
// w, bias, output, residual_data
struct FCMKLDNN : public PatternBase {
  FCMKLDNN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fc_mkldnn") {}

  PDNode* operator()(bool with_residual_data);

  // declare operator node's name
  PATTERN_DECL_NODE(fc);
  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(weights);
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(output);
  PATTERN_DECL_NODE(residual_data);
};

// Embedding
struct Embedding : public PatternBase {
  Embedding(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "embedding") {}

  PDNode* operator()(PDNode* x);

  // declare operator node's name
  PATTERN_DECL_NODE(lookup_table);
  // Inputs
  //
  PATTERN_DECL_NODE(Ids);
  PATTERN_DECL_NODE(W);  // embeddings
  // Outputs
  PATTERN_DECL_NODE(Out);
};

struct LSTM : public PatternBase {
  LSTM(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "lstm") {}

  PDNode* operator()(PDNode* x);

  // Operators
  PATTERN_DECL_NODE(lstm);

  // Inputs
  PATTERN_DECL_NODE(Input);
  PATTERN_DECL_NODE(H0);
  PATTERN_DECL_NODE(C0);
  PATTERN_DECL_NODE(Weight);
  PATTERN_DECL_NODE(Bias);

  // Outputs
  PATTERN_DECL_NODE(Hidden);
  PATTERN_DECL_NODE(Cell);
  PATTERN_DECL_NODE(BatchGate);
  PATTERN_DECL_NODE(BatchCellPreAct);
};

struct GRU : public PatternBase {
  GRU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "gru") {}

  PDNode* operator()(PDNode* x);

  // Operators
  PATTERN_DECL_NODE(gru);

  // Inputs
  PATTERN_DECL_NODE(Bias);
  PATTERN_DECL_NODE(Weight);

  // Outputs
  PATTERN_DECL_NODE(BatchGate);
  PATTERN_DECL_NODE(BatchResetHiddenPrev);
  PATTERN_DECL_NODE(BatchHidden);
  PATTERN_DECL_NODE(Hidden);
};

// The following pattern is used to fuse batch_norm and act
// formula: act(bn(x))
// op: batch_norm + act
struct BatchNormAct : public PatternBase {
  BatchNormAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  // BN inputs
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_variance);
  PATTERN_DECL_NODE(bn_mean);
  // BN outputs
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(bn_out);
  // ACT output
  PATTERN_DECL_NODE(act_out);
};

// the backward of act(bn(x))
// op: batch_norm_grad + act_grad
struct BatchNormActGrad : public PatternBase {
  BatchNormActGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act_grad") {}

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // bn_grad: in["X", "Y@GRAD", "Scale", "Bias", "SavedMean", "SavedVariance",
  // "ReserveSpace"],
  // out["X@GRAD", "Scale@GRAD", "Bias@GRAD"]
  PDNode* operator()(PDNode* x, std::unordered_set<std::string> act_grad_types);

  // declare operator node's name
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(batch_norm_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(d_intermediate_out);
  PATTERN_DECL_NODE(bn_x);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(d_bn_x);
  PATTERN_DECL_NODE(d_bn_scale);
  PATTERN_DECL_NODE(d_bn_bias);
};

//
// \brief   Pattern looking for batch_norm and a directly following activation
// operator.
//
// \note    Currently only ReLU is supported as an activation function.
//          Formula: act(bn(x))
//          Op: batch_norm + act
struct BatchNormActOneDNN : public PatternBase {
  BatchNormActOneDNN(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act_onednn") {}

  PDNode* operator()(const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(bn_in);
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(act_out);
};

// The following pattern is used to fuse batch_norm, elewise_add, and act
// formula: act(bn(x) + z)
// op: batch_norm + elewise_add + act
struct BatchNormAddAct : public PatternBase {
  BatchNormAddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_add_act") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(batch_norm);
  PATTERN_DECL_NODE(elewise_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  // BN inputs
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  // BN outputs
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(bn_out);
  // Elewise_Add input
  PATTERN_DECL_NODE(elewise_add_in);
  // Elewise_Add output
  PATTERN_DECL_NODE(elewise_add_out);
  // ACT output
  PATTERN_DECL_NODE(act_out);
};

// the backward of act(bn(x) + z)
// op: batch_norm_grad + elewise_add_grad + act_grad
struct BatchNormAddActGrad : public PatternBase {
  BatchNormAddActGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_add_act_grad") {}

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // elewise_add_grad: in["Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  // bn_grad: in["X", "Z", "Y@GRAD", "Scale", "Bias", "SavedMean",
  // "SavedVariance",
  // "ReserveSpace"],
  // out["X@GRAD", "Z@GRAD", "Scale@GRAD", "Bias@GRAD"]
  PDNode* operator()(PDNode* x, std::unordered_set<std::string> act_grad_types);

  // declare operator node's name
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(elewise_add_grad);
  PATTERN_DECL_NODE(batch_norm_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(d_act_x);
  PATTERN_DECL_NODE(d_elewise_add_in);
  PATTERN_DECL_NODE(d_bn_out);
  PATTERN_DECL_NODE(bn_x);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_reserve_space);
  PATTERN_DECL_NODE(d_bn_x);
  PATTERN_DECL_NODE(d_bn_scale);
  PATTERN_DECL_NODE(d_bn_bias);
};

// The following patterns are used to fuse elewise_add and act
// formula: act(ele_add(x, y))
// op: elementwise_add + act
// named nodes: elementwise_add, act
//              ele_x, ele_y, elewise_add_out, act_out
struct ElewiseAddAct : public PatternBase {
  ElewiseAddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elewise_add_act") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(ele_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(elewise_add_out);
  PATTERN_DECL_NODE(ele_y);
  PATTERN_DECL_NODE(act_out);
};

// formula: ele_add(x, act(y))
// op: elementwise_add + act
// named nodes: elementwise_add, act
//              act_in, act_out, ele_x, elewise_add_out
struct ActElewiseAdd : public PatternBase {
  ActElewiseAdd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "act_elewise_add") {}

  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(ele_add);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(ele_x);
  PATTERN_DECL_NODE(elewise_add_out);
};

// the backward of act(ele_add(x, y))
// the act is inplace.
// op: elementwise_add_grad + act_grad
// named nodes: elementwise_add_grad, act_grad
//              act_out, act_out_g, ele_y, d_intermediate_out, d_ele_x, d_ele_y
struct ElewiseAddActInplaceGrad : public PatternBase {
  ElewiseAddActInplaceGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elewise_add_act_grad1") {}

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  PDNode* operator()(PDNode* x, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(ele_add_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(d_intermediate_out);
  PATTERN_DECL_NODE(d_ele_x);
  PATTERN_DECL_NODE(d_ele_y);
  PATTERN_DECL_NODE(ele_y);
};

// the backward of ele_add(act(x), y)
// the act is inplace.
// op: elementwise_add_grad + act_grad
// named nodes: elementwise_add_grad, act_grad
//              ele_y, d_ele_y, d_intermeiate_out, intermediate_out, d_x
struct ActElewiseAddInplaceGrad : public PatternBase {
  ActElewiseAddInplaceGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "act_elewise_add_grad1") {}

  // ele_add_grad: in["Y", "Out@GRAD"], out["IntermediateOut@GRAD", "Y@GRAD"]
  // act_grad: in["IntermediateOut", "IntermediateOut@GRAD"], out["X@GRAD"]
  PDNode* operator()(PDNode* d_out_var, std::unordered_set<std::string> acts);

  // declare operator node's name
  PATTERN_DECL_NODE(ele_add_grad_op);
  PATTERN_DECL_NODE(act_grad_op);
  // // declare variable node's name
  PATTERN_DECL_NODE(intermediate_var);
  PATTERN_DECL_NODE(d_intermediate_var);
};

// The following patterns are used to fuse linear and act (ReLu or GeLU)
// formula: act(F.linear(x))
// op: matmul_v2 + elementwise_add + act
// named nodes: matmul, elementwise_add, act
//              matmul_w, matmul_out
//              ele_bias, elewise_add_out, act_out
struct LinearAct : public PatternBase {
  LinearAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "linear_act") {}

  PDNode* operator()(PDNode* x,
                     const std::unordered_set<std::string>& act_types,
                     bool with_grad_link,
                     bool is_act_grad_x_from_act);

  // declare operator node's name
  PATTERN_DECL_NODE(matmul);
  PATTERN_DECL_NODE(ele_add);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(act_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(matmul_w);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(elewise_add_out);
  PATTERN_DECL_NODE(ele_bias);
  PATTERN_DECL_NODE(act_out);
};

struct DotProductAttention : public PatternBase {
  DotProductAttention(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dot_product_attention_fwd") {}

  PDNode* operator()(bool with_dropout);
  // declare operator node's name for Attention Computing
  PATTERN_DECL_NODE(attn_q_transpose);
  PATTERN_DECL_NODE(attn_k_transpose);
  PATTERN_DECL_NODE(attn_v_transpose);
  PATTERN_DECL_NODE(attn_q_scale);
  PATTERN_DECL_NODE(attn_qk_matmul);
  PATTERN_DECL_NODE(attn_mask_eleadd);
  PATTERN_DECL_NODE(attn_softmax);
  PATTERN_DECL_NODE(attn_dropout);
  PATTERN_DECL_NODE(attn_context_matmul);
  PATTERN_DECL_NODE(attn_transpose);
  // declare variable node's name for Attention Computing

  PATTERN_DECL_NODE(attn_q);
  PATTERN_DECL_NODE(attn_k);
  PATTERN_DECL_NODE(attn_v);
  PATTERN_DECL_NODE(attn_q_transpose_out);
  PATTERN_DECL_NODE(attn_q_transpose_xshape);
  PATTERN_DECL_NODE(attn_k_transpose_out);
  PATTERN_DECL_NODE(attn_k_transpose_xshape);
  PATTERN_DECL_NODE(attn_v_transpose_out);
  PATTERN_DECL_NODE(attn_v_transpose_xshape);
  PATTERN_DECL_NODE(attn_q_scale_out);
  PATTERN_DECL_NODE(attn_qk_matmul_out);
  PATTERN_DECL_NODE(attn_mask);
  PATTERN_DECL_NODE(attn_mask_eleadd_out);
  PATTERN_DECL_NODE(attn_softmax_out);
  PATTERN_DECL_NODE(attn_dropout_out);
  PATTERN_DECL_NODE(attn_dropout_mask);
  PATTERN_DECL_NODE(attn_context_matmul_out);
  PATTERN_DECL_NODE(attn_transpose_out);
  PATTERN_DECL_NODE(attn_transpose_xshape);
};

struct DotProductAttentionGrad : public PatternBase {
  DotProductAttentionGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dot_product_attention_bwd") {}

  PDNode* operator()(bool with_dropout);

  // declare operator node's name for grad of Attention Computing
  PATTERN_DECL_NODE(attn_transpose_grad);
  PATTERN_DECL_NODE(attn_context_matmul_grad);
  PATTERN_DECL_NODE(attn_dropout_grad);
  PATTERN_DECL_NODE(attn_softmax_grad);
  PATTERN_DECL_NODE(attn_mask_eleadd_grad);
  PATTERN_DECL_NODE(attn_qk_matmul_grad);
  PATTERN_DECL_NODE(attn_scale_grad);
  PATTERN_DECL_NODE(attn_q_transpose_grad);
  PATTERN_DECL_NODE(attn_k_transpose_grad);
  PATTERN_DECL_NODE(attn_v_transpose_grad);
  // declare variable node's name for grad of Attention Computing
  PATTERN_DECL_NODE(attn_dout);
  PATTERN_DECL_NODE(attn_transpose_grad_out);
  PATTERN_DECL_NODE(attn_context_matmul_grad_x);
  PATTERN_DECL_NODE(attn_context_matmul_grad_y);
  PATTERN_DECL_NODE(attn_context_matmul_grad_dx);
  PATTERN_DECL_NODE(attn_context_matmul_grad_dy);
  PATTERN_DECL_NODE(attn_dropout_grad_out);
  PATTERN_DECL_NODE(attn_softmax_out);
  PATTERN_DECL_NODE(attn_softmax_grad_out);
  PATTERN_DECL_NODE(attn_mask_eleadd_grad_mask);
  PATTERN_DECL_NODE(attn_mask_eleadd_grad_dx);
  PATTERN_DECL_NODE(attn_qk_matmul_grad_x);
  PATTERN_DECL_NODE(attn_qk_matmul_grad_y);
  PATTERN_DECL_NODE(attn_qk_matmul_grad_dx);
  PATTERN_DECL_NODE(attn_qk_matmul_grad_dy);
  PATTERN_DECL_NODE(attn_scale_grad_out);
  PATTERN_DECL_NODE(attn_dq);
  PATTERN_DECL_NODE(attn_dk);
  PATTERN_DECL_NODE(attn_dv);
};

// The following patterns are used to fuse linear_grad and act_grad (ReLu or
// GeLU)
// formula: the backward of F.linear( act(x) )
// op: elementwise_add_grad + matmul_v2_grad + act_grad
// named nodes: ele_add_grad, matmul_grad, act_grad
//              ele_grad_bias, ele_grad_dx, ele_grad_dbias
//              matmul_grad_x, matmul_grad_dx, matmul_grad_dx
//              matmul_grad_dw, act_grad_dx
struct ElewiseAddMatmulAct : public PatternBase {
  ElewiseAddMatmulAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elewiseadd_matmul_act") {}

  PDNode* operator()(PDNode* x,
                     const std::unordered_set<std::string>& act_grad_types,
                     bool without_x_gradient,
                     bool is_act_grad_x_from_act);

  // declare operator node's name
  PATTERN_DECL_NODE(ele_add_grad);
  PATTERN_DECL_NODE(matmul_grad);
  PATTERN_DECL_NODE(act_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(ele_out);
  PATTERN_DECL_NODE(ele_grad_bias);
  PATTERN_DECL_NODE(ele_grad_dx);
  PATTERN_DECL_NODE(ele_grad_dbias);
  PATTERN_DECL_NODE(matmul_grad_x);
  PATTERN_DECL_NODE(matmul_grad_w);
  PATTERN_DECL_NODE(matmul_grad_dx);
  PATTERN_DECL_NODE(matmul_grad_dw);
  PATTERN_DECL_NODE(act_grad_dx);
};

// Conv with Elementwise_add as bias
// op: conv + elementwise_add
// named nodes:
// conv_input, conv_weight,
// conv_out, conv,
// eltwise_bias, eltwise_out,
// elementwise_add
struct ConvBias : public PatternBase {
  ConvBias(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_bias") {}
  PDNode* operator()(PDNode* conv_input, std::string conv_type = "conv2d");
  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(eltwise);
  // declare variable node's name
  PATTERN_DECL_NODE(conv_weight);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(eltwise_bias);
  PATTERN_DECL_NODE(eltwise_out);
};

// Convolution op
// Forward pass for convolution.
// conv_input, conv_bias and conv_filter are inputs.
// conv_output is a result of the operator.
// residual_data is data used by skip connection.
// If residual connection fusion is on, the formula is:
// conv_output = conv_op(conv_filter, conv_input, conv_bias)
//             + conv_residual_data
// If the fusion is off, conv_residual_data is not added.
struct Conv : public PatternBase {
  Conv(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "convolution") {}

  PDNode* operator()(const std::string& conv_type);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_output);
};

// Convolution op with residual data
struct ConvResidual : public PatternBase {
  ConvResidual(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_residual") {}

  PDNode* operator()(const std::string& conv_type, bool with_residual_data);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_residual_data);
  PATTERN_DECL_NODE(conv_output);
};

// Pool op
// Forward pass for pooling.
// pool_input is the input.
// pool_output is a result of the operator.
struct Pool : public PatternBase {
  Pool(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "pooling") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(pool_op);
  PATTERN_DECL_NODE(pool_input);
  PATTERN_DECL_NODE(pool_output);
};

// Elementwise ops
// Forward pass for element-wise operators
// elementwise_out is the result of the operator
struct Elementwise : public PatternBase {
  Elementwise(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elementwise") {}

  PDNode* operator()(PDNode* x_var,
                     PDNode* y_var,
                     const std::string& elementwise_type);

  PATTERN_DECL_NODE(elementwise_op);
  PATTERN_DECL_NODE(elementwise_x);
  PATTERN_DECL_NODE(elementwise_y);
  PATTERN_DECL_NODE(elementwise_out);
};

// Elementwise ops
// Forward pass for element-wise operators
// elementwise_out is the result of the operator
struct ElementwiseOp : public PatternBase {
  ElementwiseOp(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elementwise") {}

  PDNode* operator()(const std::string& elementwise_type);

  PATTERN_DECL_NODE(elementwise_op);
  PATTERN_DECL_NODE(elementwise_out);
};

struct MatmulElementwiseAdd : public PatternBase {
  MatmulElementwiseAdd(PDPattern* pattern UNUSED,
                       const std::string& name_scope UNUSED,
                       const std::string& matmul_type UNUSED,
                       bool as_x UNUSED)
      : PatternBase(pattern, name_scope, "matmul_elementwise_add") {}

  PDNode* operator()(const std::string& matmul_type, bool as_x);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(elementwise_addend);
  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_out);
};

// Residual Elementwise ops
// This pattern allows operator output to be X or Y
// and residual data Y or X, based on as_x flag
struct ResidualElementwise : public PatternBase {
  ResidualElementwise(PDPattern* pattern,
                      const std::string& name_scope,
                      bool as_x UNUSED)
      : PatternBase(pattern, name_scope, "residual_elementwise") {}
  PDNode* operator()(PDNode* op_var,
                     PDNode* residual_var,
                     const std::string& elementwise_type,
                     bool as_x);

  PATTERN_DECL_NODE(operator_output);
  PATTERN_DECL_NODE(residual_data);
  PATTERN_DECL_NODE(elementwise_op);
  PATTERN_DECL_NODE(elementwise_out);
};

// General struct for immutable ops:
// reshape, transpose, slice, shape, nearest-interp, split
// Forward pass for no weights-op.
// immutable_out is a result of the operator.
struct Immutable : public PatternBase {
  Immutable(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "immutable") {}

  PDNode* operator()(const std::string& immutable_type,
                     const std::string& input_name);
  PATTERN_DECL_NODE(prev_op);
  PATTERN_DECL_NODE(immutable_in);
  PATTERN_DECL_NODE(immutable_op);
  PATTERN_DECL_NODE(immutable_out);
};

// Matmul op
// Forward pass for matmul.
struct Matmul : public PatternBase {
  Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// MatmulV2: tensor * weight
struct MatmulV2Weight : public PatternBase {
  MatmulV2Weight(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_v2_weight") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(matmul_v2_in_x);
  PATTERN_DECL_NODE(matmul_v2_in_y);
  PATTERN_DECL_NODE(matmul_v2_op);
  PATTERN_DECL_NODE(matmul_v2_out);
};

// MatmulV2: tensor * tensor or tensor * weight
struct MatmulV2 : public PatternBase {
  MatmulV2(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_v2") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(matmul_v2_in_x);
  PATTERN_DECL_NODE(matmul_v2_in_y);
  PATTERN_DECL_NODE(matmul_v2_op);
  PATTERN_DECL_NODE(matmul_v2_out);
};

// Matmul + scale
// Forward pass.
struct MatmulScale : public PatternBase {
  MatmulScale(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_scale") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(scale_in_x);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);
};

// Matmul_v2 + scale
// Forward pass.
struct MatmulV2Scale : public PatternBase {
  MatmulV2Scale(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_v2_scale") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(matmul_v2_in_x);
  PATTERN_DECL_NODE(matmul_v2_in_y);
  PATTERN_DECL_NODE(matmul_v2_op);
  PATTERN_DECL_NODE(scale_in_x);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);
};

// Squeeze2 + Matmul
// Forward pass.
struct Squeeze2Matmul : public PatternBase {
  Squeeze2Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "squeeze2_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(squeeze2_in_x);
  PATTERN_DECL_NODE(squeeze2_op);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Reshape2 + Matmul
// Forward pass.
struct Reshape2Matmul : public PatternBase {
  Reshape2Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "reshape2_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(reshape2_in_x);
  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Forward pass for two input ops and fused_matmul op.
// matmul_out is a result of the operator.
struct FusedMatmul : public PatternBase {
  FusedMatmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fused_matmul") {}

  PDNode* operator()(bool with_residual);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_residual_data);
  PATTERN_DECL_NODE(matmul_out);
};

// Flatten2 + Matmul
// Forward pass.
struct Flatten2Matmul : public PatternBase {
  Flatten2Matmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "flatten2_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(flatten2_in_x);
  PATTERN_DECL_NODE(flatten2_op);
  PATTERN_DECL_NODE(matmul_in_x);
  PATTERN_DECL_NODE(matmul_in_y);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Concat op
// Forward pass for concat.
// concat_out is a result of the operator.
struct Concat : public PatternBase {
  Concat(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "concat") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(concat_op);
  PATTERN_DECL_NODE(concat_out);
};

// Op + Requant
// named nodes:
// any_op, any_out
// requant_op, requant_out
struct OpRequant : public PatternBase {
  OpRequant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "op_requant") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(requant_in);
  PATTERN_DECL_NODE(requant_op);
  PATTERN_DECL_NODE(requant_out);
};

// Requant + Op
// named nodes:
// requant_in, requant_op,
// requant_out, any_op
struct RequantOp : public PatternBase {
  RequantOp(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "requant_op") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(requant_in);
  PATTERN_DECL_NODE(requant_op);
  PATTERN_DECL_NODE(requant_out);
};

// Op + Dequant
// named nodes:
// any_op, dequant_in
// dequant_op, dequant_out
struct OpDequant : public PatternBase {
  OpDequant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "op_dequant") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
};

// Dequantize + Scale
struct DequantScale : public PatternBase {
  DequantScale(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_scale") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);
};

// Scale + Quantize
struct ScaleQuant : public PatternBase {
  ScaleQuant(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "scale_quant") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(scale_in);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
};

// Quantize + Conv2d
struct QuantConv : public PatternBase {
  QuantConv(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quant_conv") {}

  PDNode* operator()(const std::string& conv_type);

  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(conv_in);
  PATTERN_DECL_NODE(conv_op);
};

// Scale + Matmul
struct ScaleMatmul : public PatternBase {
  ScaleMatmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "scale_matmul") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(scale_in);
  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(matmul_op);
};

// PriorBox operator
// operator: prior_box_op
// inputs: prior_box_input, prior_box_image
// outputs: prior_box_boxes, prior_box_variances
struct PriorBox : public PatternBase {
  PriorBox(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "PriorBox") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(prior_box_op);
  PATTERN_DECL_NODE(prior_box_input);
  PATTERN_DECL_NODE(prior_box_image);
  PATTERN_DECL_NODE(prior_box_boxes);
  PATTERN_DECL_NODE(prior_box_variances);
};

// vit_attention
struct VitAttention : public PatternBase {
  VitAttention(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "vit_attention") {}

  PDNode* operator()(PDNode* in);

  PATTERN_DECL_NODE(matmul0_op);
  PATTERN_DECL_NODE(matmul0_in_y);
  PATTERN_DECL_NODE(matmul0_out);

  PATTERN_DECL_NODE(elementwise0_op);
  PATTERN_DECL_NODE(elementwise0_in_y);
  PATTERN_DECL_NODE(elementwise0_out);

  PATTERN_DECL_NODE(reshape1_op);
  PATTERN_DECL_NODE(reshape1_out);

  PATTERN_DECL_NODE(transpose1_op);
  PATTERN_DECL_NODE(transpose1_out);

  PATTERN_DECL_NODE(slice1_op);
  PATTERN_DECL_NODE(slice1_out);

  PATTERN_DECL_NODE(slice2_op);
  PATTERN_DECL_NODE(slice2_out);

  PATTERN_DECL_NODE(slice3_op);
  PATTERN_DECL_NODE(slice3_out);

  PATTERN_DECL_NODE(matmul2_op);
  PATTERN_DECL_NODE(matmul2_out);

  PATTERN_DECL_NODE(matmul1_op);
  PATTERN_DECL_NODE(matmul1_out);

  PATTERN_DECL_NODE(transpose2_op);
  PATTERN_DECL_NODE(transpose2_out);

  PATTERN_DECL_NODE(scale1_op);
  PATTERN_DECL_NODE(scale1_out);

  PATTERN_DECL_NODE(softmax1_op);
  PATTERN_DECL_NODE(softmax1_out);

  PATTERN_DECL_NODE(transpose3_op);
  PATTERN_DECL_NODE(transpose3_out);

  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(reshape2_out);
};

// self_attention in vit
struct SelfAttention : public PatternBase {
  SelfAttention(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "vit_block") {}

  PDNode* operator()(PDNode* in);

  PATTERN_DECL_NODE(transpose2_0_op);
  PATTERN_DECL_NODE(transpose2_0_out);
  PATTERN_DECL_NODE(transpose2_1_op);
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(transpose2_2_op);
  PATTERN_DECL_NODE(transpose2_2_out);
  PATTERN_DECL_NODE(matmul_0_op);
  PATTERN_DECL_NODE(matmul_0_out);
  PATTERN_DECL_NODE(matmul_1_op);
  PATTERN_DECL_NODE(matmul_1_out);
  PATTERN_DECL_NODE(slice_0_op);
  PATTERN_DECL_NODE(slice_0_out);
  PATTERN_DECL_NODE(slice_1_op);
  PATTERN_DECL_NODE(slice_1_out);
  PATTERN_DECL_NODE(slice_2_op);
  PATTERN_DECL_NODE(slice_2_out);
  PATTERN_DECL_NODE(softmax_op);
  PATTERN_DECL_NODE(softmax_out);
};

// Conv + ElementwiseAdd + an activation
// This pattern can further fuse the conv related ops after the conv+bn fusion.
struct ConvElementwiseaddAct : public PatternBase {
  ConvElementwiseaddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_elementwiseadd_act") {}

  PDNode* operator()(PDNode* conv_in,
                     const std::unordered_set<std::string>& conv_act_set);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(conv_filter);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_in_y);  // input
  PATTERN_DECL_NODE(elementwise_add_out);

  PATTERN_DECL_NODE(act_op);
  PATTERN_DECL_NODE(act_out);
};

// Conv + ElementwiseAdd + ElementwiseAdd + Activation
struct ConvElementwiseAdd2Act : public PatternBase {
  ConvElementwiseAdd2Act(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(
            pattern, name_scope, "conv_elementwiseadd2_elementwiseadd_act") {}

  PDNode* operator()(PDNode* conv_in,
                     const std::unordered_set<std::string>& conv_act_set);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_out);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_in_y);  // input
  PATTERN_DECL_NODE(elementwise_add_out);

  PATTERN_DECL_NODE(elementwise_add_op_1);
  PATTERN_DECL_NODE(elementwise_add_in_y_1);  // input
  PATTERN_DECL_NODE(elementwise_add_out_1);

  PATTERN_DECL_NODE(act_op);
  PATTERN_DECL_NODE(act_out);
};

// Conv + ElementwiseAdd
// This pattern should be used after ConvElementwiseAdd2Act or
// ConvElementwiseadd pass
struct ConvElementwiseadd : public PatternBase {
  ConvElementwiseadd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_elementwiseadd") {}

  PDNode* operator()(PDNode* conv_in);

  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(conv_filter);

  PATTERN_DECL_NODE(elementwise_add_op);
  PATTERN_DECL_NODE(elementwise_add_in_y);
  PATTERN_DECL_NODE(elementwise_add_out);
};

// Conv with affine_channel
// op: conv + (elementwise_add +) affine_channel
// named nodes:
// conv_weight, conv_out, conv,
// ac_x, ac_scale, ac_bias
// affine_channel, ac_out
struct ConvAffineChannel : public PatternBase {
  ConvAffineChannel(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_affine_channel") {}

  PDNode* operator()(PDNode* conv_input,
                     const std::string& conv_type,
                     bool with_eltwise_add);

  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(affine_channel);
  PATTERN_DECL_NODE(eltwise);  // ELEMENTWISE_ADD
  // CONV inputs
  PATTERN_DECL_NODE(conv_weight);  // Filter
  // CONV outputs
  PATTERN_DECL_NODE(conv_out);  // tmp
  // ELTWISE inputs
  PATTERN_DECL_NODE(eltwise_y_in);
  // ELTWISE outputs
  PATTERN_DECL_NODE(eltwise_out);  // tmp

  // AC(Affine_Channel) inputs
  PATTERN_DECL_NODE(ac_scale);
  PATTERN_DECL_NODE(ac_bias);
  // AC outputs
  PATTERN_DECL_NODE(ac_out);  // Out
};

// Dequantize + Quantize + anyOP
// This pattern is used for squashing the dequantize-quantize pairs.
struct DequantQuantAny : public PatternBase {
  DequantQuantAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_quant_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
  PATTERN_DECL_NODE(next_op);
};

// Dequantize + anyOP
// This quantize is used for getting number of ops the Dequantize's
// output is an input to.
struct DequantAny : public PatternBase {
  DequantAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(next_op);
};

// anyOp + more then one quantize op
// This pattern is used for squashing multiple quantize with the same scale.
struct MultipleQuantize : public PatternBase {
  MultipleQuantize(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multiple_quantize") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(prev_out);
};

struct QuantizePlacement : public PatternBase {
  QuantizePlacement(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quantize_placement") {}
  PDNode* operator()(
      const std::unordered_set<std::string>& quantize_enabled_op_types);

  PATTERN_DECL_NODE(op);
};

struct Bfloat16Placement : public PatternBase {
  Bfloat16Placement(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bfloat16_placement") {}
  PDNode* operator()(
      const std::unordered_set<std::string>& bfloat16_enabled_op_types);

  PATTERN_DECL_NODE(op_in);
  PATTERN_DECL_NODE(op);
};

struct OrphanedBfloat16 : public PatternBase {
  OrphanedBfloat16(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "orphaned_bfloat16") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(prev_op);
  PATTERN_DECL_NODE(prev_out);
  PATTERN_DECL_NODE(op);
  PATTERN_DECL_NODE(op_out);
  PATTERN_DECL_NODE(next_op);
};

struct UnsupportedBfloat16 : public PatternBase {
  UnsupportedBfloat16(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "unsupported_bfloat16") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(prev_op);
  PATTERN_DECL_NODE(prev_out);
  PATTERN_DECL_NODE(op);
};

struct Bloat16Ops : public PatternBase {
  Bloat16Ops(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "many_bfloat16_ops") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(op);
};

// Pattern used for enforcing inplace computation for in-place computation
// supporting DNNL ops. softmax, batch_norm and layer_norm
struct MKLDNNInPlace : public PatternBase {
  MKLDNNInPlace(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "mkldnn_inplace") {}
  PDNode* operator()();

  // MKL-DNN's in-place ops: BatchNorm, Softmax, Elementwise_add
  PATTERN_DECL_NODE(inplace_to_be_op);
  PATTERN_DECL_NODE(inplace_to_be_op_in);
  PATTERN_DECL_NODE(inplace_to_be_op_out);
  PATTERN_DECL_NODE(next_op);
  PATTERN_DECL_NODE(next_op_out);
};

struct TransposeFlattenConcat : public PatternBase {
  TransposeFlattenConcat(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "transpose_flatten_concat") {}

  PDNode* operator()(std::vector<PDNode*> conv_inputs, int times);

  std::string GetNodeName(const std::string& op_type) {
    return PDNodeName(name_scope_, repr_, id_, op_type);
  }

  PDNode* GetPDNode(const std::string& op_type) {
    return pattern->RetrieveNode(GetNodeName(op_type));
  }
};

struct DeleteQuantOpFuse : public PatternBase {
  DeleteQuantOpFuse(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_quant_fuse") {}

  void operator()(PDNode* input_act_node, const std::string& quant_type);

  std::string GetNodeName(const std::string& op_type) {
    return PDNodeName(name_scope_, repr_, id_, op_type);
  }

  PDNode* GetPDNode(const std::string& op_type) {
    return pattern->RetrieveNode(GetNodeName(op_type));
  }
};

struct DequantOpFuse : public PatternBase {
  DequantOpFuse(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_fuse") {}

  void operator()(PDNode* quant_op_input,
                  const std::string& quantized_op_type,
                  const std::string& dequant_type,
                  const std::string& weight_name);

  std::string GetNodeName(const std::string& op_type) {
    return PDNodeName(name_scope_, repr_, id_, op_type);
  }

  PDNode* GetPDNode(const std::string& op_type) {
    return pattern->RetrieveNode(GetNodeName(op_type));
  }
};

struct ShuffleChannelPattern : public PatternBase {
  ShuffleChannelPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "shufflechannel_pattern") {}

  void operator()(PDNode* reshape1_in);

  PATTERN_DECL_NODE(reshape1_op);
  PATTERN_DECL_NODE(reshape1_out);

  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(reshape2_out);
};

struct DeleteDropoutOpPattern : public PatternBase {
  DeleteDropoutOpPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_dropout_op_pattern") {}

  void operator()(bool with_mask);

  PATTERN_DECL_NODE(dropout_op_x);
  PATTERN_DECL_NODE(dropout_op);
  PATTERN_DECL_NODE(dropout_op_out);
  PATTERN_DECL_NODE(dropout_op_mask);
};

struct DeleteQuantDequantOpPattern : public PatternBase {
  DeleteQuantDequantOpPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_quant_dequant_op_pattern") {}

  void operator()(PDNode* input_node, const std::string& quant_dequant_types);

  PATTERN_DECL_NODE(quant_dequant_op_inscale);
  PATTERN_DECL_NODE(quant_dequant_op);
  PATTERN_DECL_NODE(quant_dequant_op_outscale);
  PATTERN_DECL_NODE(quant_dequant_op_out);
};

struct DeleteQuantDequantFilterOpPattern : public PatternBase {
  DeleteQuantDequantFilterOpPattern(PDPattern* pattern,
                                    const std::string& name_scope)
      : PatternBase(
            pattern, name_scope, "delete_quant_dequant_filter_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(quant_dequant_op_x);
  PATTERN_DECL_NODE(quant_dequant_op);
  PATTERN_DECL_NODE(quant_dequant_op_outscale);
  PATTERN_DECL_NODE(quant_dequant_op_out);
  PATTERN_DECL_NODE(any_op2);
};

struct DeleteWeightQuantDequantLinearOpPattern : public PatternBase {
  DeleteWeightQuantDequantLinearOpPattern(PDPattern* pattern,
                                          const std::string& name_scope)
      : PatternBase(pattern,
                    name_scope,
                    "delete_weight_quant_dequant_linear_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(weight_dequantize_linear_op_x);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_scale);
  PATTERN_DECL_NODE(weight_dequantize_linear_op);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_out);
};

struct DeleteWeightDequantLinearOpEncoderPattern : public PatternBase {
  DeleteWeightDequantLinearOpEncoderPattern(PDPattern* pattern,
                                            const std::string& name_scope)
      : PatternBase(pattern,
                    name_scope,
                    "delete_weight_quant_dequant_linear_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(weight_dequantize_linear_op_x);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_scale);
  PATTERN_DECL_NODE(while0);
  PATTERN_DECL_NODE(weight_dequantize_linear_op);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_out);
  PATTERN_DECL_NODE(any_op2);
};

struct QuantLinearFusePattern : public PatternBase {
  QuantLinearFusePattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quant_linear_fuse_pattern") {}

  PDNode* operator()(bool with_bias, bool with_relu);

  PATTERN_DECL_NODE(quantize_linear_op_x);
  PATTERN_DECL_NODE(quantize_linear_op_scale);
  PATTERN_DECL_NODE(quantize_linear_op);
  PATTERN_DECL_NODE(quantize_linear_op_out);

  PATTERN_DECL_NODE(dequantize_linear_op);
  PATTERN_DECL_NODE(dequantize_linear_op_out);

  PATTERN_DECL_NODE(weight_dequantize_linear_op_x);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_scale);
  PATTERN_DECL_NODE(weight_dequantize_linear_op);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_out);

  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(mul_out);

  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(elementwise_add);
  PATTERN_DECL_NODE(elementwise_add_out);

  PATTERN_DECL_NODE(relu);
  PATTERN_DECL_NODE(relu_out);
};

struct DeleteWeightDequantLinearOpDecoderPattern : public PatternBase {
  DeleteWeightDequantLinearOpDecoderPattern(PDPattern* pattern,
                                            const std::string& name_scope)
      : PatternBase(pattern,
                    name_scope,
                    "delete_weight_quant_dequant_linear_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(weight_dequantize_linear_op_x);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_scale);
  PATTERN_DECL_NODE(weight_dequantize_linear_op);
  PATTERN_DECL_NODE(weight_dequantize_linear_op_out);
  PATTERN_DECL_NODE(any_op2);
};

struct DeleteQuantDequantLinearOpPattern : public PatternBase {
  DeleteQuantDequantLinearOpPattern(PDPattern* pattern,
                                    const std::string& name_scope)
      : PatternBase(
            pattern, name_scope, "delete_quant_dequant_linear_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(quantize_linear_op_x);
  PATTERN_DECL_NODE(quantize_linear_op_scale);
  PATTERN_DECL_NODE(quantize_linear_op);
  PATTERN_DECL_NODE(quantize_linear_op_out);
  PATTERN_DECL_NODE(dequantize_linear_op);
  // PATTERN_DECL_NODE(dequantize_linear_op_scale);  // Can not add this node.
  // Todo: Wangzheee
  PATTERN_DECL_NODE(dequantize_linear_op_out);
};

// Reshape + Transpose + Matmul
// named nodes:
// reshape_op, reshape_out, reshape_xshape,
// transpose_op, transpose_out, transpose_xshape,
// matmul_op, matmul_out
struct ReshapeTransposeMatmulPattern : public PatternBase {
  ReshapeTransposeMatmulPattern(PDPattern* pattern,
                                const std::string& name_scope)
      : PatternBase(pattern, name_scope, "reshape_transpose_matmul") {}

  PDNode* operator()(const std::string& op_name,
                     bool with_reshape_xshape,
                     bool with_transpose_xshape);

  PATTERN_DECL_NODE(reshape_in);
  PATTERN_DECL_NODE(reshape_op);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(reshape_xshape);
  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(transpose_xshape);
  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
};

// Matmul + Transpose + Reshape
struct MatmulTransposeReshapePattern : public PatternBase {
  MatmulTransposeReshapePattern(PDPattern* pattern,
                                const std::string& name_scope)
      : PatternBase(pattern, name_scope, "matmul_transpose_reshape") {}

  PDNode* operator()(const std::string& op_name);

  PATTERN_DECL_NODE(matmul_op);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(transpose_out_xshape);
  PATTERN_DECL_NODE(reshape_op);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(reshape_out_xshape);
};

// fusion_gru op
// Forward pass for fusion_gru.
// fusion_gru out is a result of the operator.
struct FusionGru : public PatternBase {
  FusionGru(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fusion_gru") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(op);
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(weight_h);
  PATTERN_DECL_NODE(weight_x);
  PATTERN_DECL_NODE(out);
};

// fusion_lstm op
// Forward pass for fusion_lstm.
// fusion_lstm out is a result of the operator.
struct FusionLSTM : public PatternBase {
  FusionLSTM(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fusion_lstm") {}
  PDNode* operator()();

  // declare op
  PATTERN_DECL_NODE(op);

  // declare inputs
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(weight_h);
  PATTERN_DECL_NODE(weight_x);

  // declare outputs
  PATTERN_DECL_NODE(hidden);
  PATTERN_DECL_NODE(cell);
};

// two concatenated fusion_gru ops
// Forward pass for fusion of two concatenated fusion_gru ops.
// concat_out is a result of the operator().
struct TwoFusionGruConcat : public PatternBase {
  TwoFusionGruConcat(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bi_fusion_gru") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(gru1);
  PATTERN_DECL_NODE(gru2);
  PATTERN_DECL_NODE(wh1);
  PATTERN_DECL_NODE(wh2);
  PATTERN_DECL_NODE(wx1);
  PATTERN_DECL_NODE(wx2);
  PATTERN_DECL_NODE(b1);
  PATTERN_DECL_NODE(b2);
  PATTERN_DECL_NODE(h1);
  PATTERN_DECL_NODE(h2);
  PATTERN_DECL_NODE(concat);
  PATTERN_DECL_NODE(out);
};

// two subsequent bi_fusion_gru ops
// Forward pass for fusion of two subsequent fusion_gru ops.
// Hidden of the last fusion_gru op is a result of the operator().
struct MultiGruSeq : public PatternBase {
  MultiGruSeq(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multi_gru_seq") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(gru1);
  PATTERN_DECL_NODE(wx11);
  PATTERN_DECL_NODE(wx12);
  PATTERN_DECL_NODE(wh11);
  PATTERN_DECL_NODE(wh12);
  PATTERN_DECL_NODE(b11);
  PATTERN_DECL_NODE(b12);
  PATTERN_DECL_NODE(h1);
  PATTERN_DECL_NODE(gru2);
  PATTERN_DECL_NODE(wx21);
  PATTERN_DECL_NODE(wx22);
  PATTERN_DECL_NODE(wh21);
  PATTERN_DECL_NODE(wh22);
  PATTERN_DECL_NODE(b21);
  PATTERN_DECL_NODE(b22);
  PATTERN_DECL_NODE(h2);
};

// multi_gru op
// Quantization pass for multi_gru op.
// Hidden of the multi_gru op is a result of the operator().
struct MultiGru : public PatternBase {
  MultiGru(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multi_gru") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(gru);
  PATTERN_DECL_NODE(wx);
  PATTERN_DECL_NODE(wh);
  PATTERN_DECL_NODE(h);
};

//
// \brief   Pattern looking for subgraph representing layer normalization
//          operation.
//
struct LayerNorm : public PatternBase {
  LayerNorm(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "layer_norm") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(x_mean);
  PATTERN_DECL_NODE(x_mean_out);
  PATTERN_DECL_NODE(x_sub_mean);
  PATTERN_DECL_NODE(x_sub_mean_out);
  PATTERN_DECL_NODE(sqr_pow);
  PATTERN_DECL_NODE(x_sub_mean_sqr);
  PATTERN_DECL_NODE(x_sub_mean_sqr_out);
  PATTERN_DECL_NODE(std_dev);
  PATTERN_DECL_NODE(std_dev_out);
  PATTERN_DECL_NODE(eps);
  PATTERN_DECL_NODE(std_dev_eps);
  PATTERN_DECL_NODE(std_dev_eps_out);
  PATTERN_DECL_NODE(std_dev_eps_sqrt);
  PATTERN_DECL_NODE(std_dev_eps_sqrt_out);
  PATTERN_DECL_NODE(division);
  PATTERN_DECL_NODE(division_out);
  PATTERN_DECL_NODE(gamma);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(beta);
  PATTERN_DECL_NODE(shift);
  PATTERN_DECL_NODE(shift_out);
};

//
// \brief   Pattern looking for subgraph representing layer normalization
//          operation.
//
struct SplitLayerNorm : public PatternBase {
  SplitLayerNorm(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "split_layer_norm") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(layer_norm_in);
  PATTERN_DECL_NODE(layer_norm_op);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
};

//
// \brief   Pattern looking for subgraph representing layernorm_shift_partition
//          operation with shift_size = 0.
//
struct LayernormShiftPartitionPattern : public PatternBase {
  LayernormShiftPartitionPattern(PDPattern* pattern,
                                 const std::string& name_scope,
                                 bool with_roll)
      : PatternBase(pattern, name_scope, "layernorm_shift_partition"),
        with_roll_(with_roll) {}

  PDNode* operator()();
  bool with_roll_;
  PATTERN_DECL_NODE(layer_norm_in);
  PATTERN_DECL_NODE(layer_norm_op);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(reshape1_op);
  PATTERN_DECL_NODE(reshape1_out);
  // optional op roll
  PATTERN_DECL_NODE(roll1_op);
  PATTERN_DECL_NODE(roll1_out);

  PATTERN_DECL_NODE(reshape2_op);
  PATTERN_DECL_NODE(reshape2_out);
  PATTERN_DECL_NODE(transpose_op);
  PATTERN_DECL_NODE(transpose_out);
  PATTERN_DECL_NODE(reshape3_op);
  PATTERN_DECL_NODE(reshape3_out);
  PATTERN_DECL_NODE(reshape4_op);
  PATTERN_DECL_NODE(reshape4_out);
};

//
// \bref pattern looking for reverse circlic shift in window attention.
//       The reverse circlic shift based on roll op,
//       therefore, reverse_roll were adopted as pattern and fused op name.
//
struct ReverseRollPattern : public PatternBase {
  ReverseRollPattern(PDPattern* pattern,
                     const std::string& name_scope,
                     bool with_roll)
      : PatternBase(pattern, name_scope, "reverse_roll"),
        with_roll_(with_roll) {}

  PDNode* operator()(PDNode* in);
  bool with_roll_;
  PATTERN_DECL_NODE(reshape2_00_op);
  PATTERN_DECL_NODE(reshape2_00_out);
  PATTERN_DECL_NODE(reshape2_10_op);
  PATTERN_DECL_NODE(reshape2_10_out);
  PATTERN_DECL_NODE(transpose2_20_op);
  PATTERN_DECL_NODE(transpose2_20_out);
  PATTERN_DECL_NODE(reshape2_30_op);
  PATTERN_DECL_NODE(reshape2_30_out);
  PATTERN_DECL_NODE(roll_40_op);
  PATTERN_DECL_NODE(roll_40_out);
  PATTERN_DECL_NODE(reshape2_50_op);
  PATTERN_DECL_NODE(reshape2_50_out);
};

// pattern for merge_layernorm
struct MergeLayernormPattern : public PatternBase {
  MergeLayernormPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "merge_layernorm") {}

  PDNode* operator()(PDNode* reshape2_in);

  PATTERN_DECL_NODE(reshape2_00_op);
  PATTERN_DECL_NODE(reshape2_00_out);
  PATTERN_DECL_NODE(strided_slice_10_op);
  PATTERN_DECL_NODE(strided_slice_10_out);
  PATTERN_DECL_NODE(strided_slice_11_op);
  PATTERN_DECL_NODE(strided_slice_11_out);
  PATTERN_DECL_NODE(strided_slice_12_op);
  PATTERN_DECL_NODE(strided_slice_12_out);
  PATTERN_DECL_NODE(strided_slice_13_op);
  PATTERN_DECL_NODE(strided_slice_13_out);
  PATTERN_DECL_NODE(concat_20_op);
  PATTERN_DECL_NODE(concat_20_out);
  PATTERN_DECL_NODE(reshape2_30_op);
  PATTERN_DECL_NODE(reshape2_30_out);
  PATTERN_DECL_NODE(layernorm_40_op);
  PATTERN_DECL_NODE(layernorm_40_in_bias);
  PATTERN_DECL_NODE(layernorm_40_in_scale);
  PATTERN_DECL_NODE(layernorm_40_out);
};

// MulMatmulMatmulV2: ops(mul, matmul, matmul_v2)
// Forward pass for ops(mul, matmul, matmul_v2) convert to matrix_multiply.
struct MulMatmulMatmulV2 : public PatternBase {
  MulMatmulMatmulV2(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "mul_matmul_matmul_v2") {}

  void operator()(const std::unordered_set<std::string>& ops_type);
  PATTERN_DECL_NODE(ops);
  PATTERN_DECL_NODE(ops_out);
};

// Add support int8 flag
struct AddSupportInt8 : public PatternBase {
  AddSupportInt8(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "Add_support_int8") {}

  PDNode* operator()();
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
};

// subgraph_edge_pattern
struct SubgraphEdgePattern : public PatternBase {
  SubgraphEdgePattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "subgraph_edge_pattern") {}
  PDNode* operator()(const std::unordered_set<std::string>& ops_type);
  PATTERN_DECL_NODE(ops);
};

// The following patterns are used to fuse feedforward in forward
// 1. layer_norm -> linear1 -> activation -> dropout1 -> linear2 -> dropout2
// -> residual_add (pre_layer_norm)
// 2. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> residual_add
// -> layer_norm (pOST_layer_norm)
// other cases: may delete residual_add, dropout1, dropout2 operators
struct FusedFeedForwardFwd : public PatternBase {
  FusedFeedForwardFwd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fused_feedforward_fwd") {}

  PDNode* operator()(PDNode* x,
                     std::unordered_set<std::string> act_types,
                     bool use_mp,
                     bool pre_layer_norm,
                     bool add_residual,
                     bool use_dropout_1,
                     bool use_dropout_2);

#ifndef FEEDFORWARD_LINEAR_DROPOUT_NODE
#define FEEDFORWARD_LINEAR_DROPOUT_NODE(suffix__) \
  PATTERN_DECL_NODE(matmul_op_##suffix__);        \
  PATTERN_DECL_NODE(matmul_w_##suffix__);         \
  PATTERN_DECL_NODE(matmul_out_##suffix__);       \
  PATTERN_DECL_NODE(ele_add_op_##suffix__);       \
  PATTERN_DECL_NODE(ele_add_bias_##suffix__);     \
  PATTERN_DECL_NODE(ele_add_out_##suffix__);      \
  PATTERN_DECL_NODE(dropout_op_##suffix__);       \
  PATTERN_DECL_NODE(dropout_out_##suffix__);      \
  PATTERN_DECL_NODE(dropout_mask_##suffix__);

  // LayerNorm: layer_norm
  PATTERN_DECL_NODE(layer_norm_op);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
  // Mode parallelism
  PATTERN_DECL_NODE(c_identity_op);
  PATTERN_DECL_NODE(c_identity_out);
  PATTERN_DECL_NODE(c_allreduce_sum_op);
  PATTERN_DECL_NODE(c_allreduce_sum_out);
  // Linear 1 and Dropout 1: matmul_v2 + elementwise_add + dropout
  FEEDFORWARD_LINEAR_DROPOUT_NODE(1);
  // Activation Grad: gelu or relu
  PATTERN_DECL_NODE(act_op);
  PATTERN_DECL_NODE(act_out);
  // Linear 2 and Dropout 2: matmul_v2 + elementwise_add + dropout
  FEEDFORWARD_LINEAR_DROPOUT_NODE(2);
  // ResidualAdd: elementwise_add
  PATTERN_DECL_NODE(ele_add_op_3);
  PATTERN_DECL_NODE(ele_add_out_3);
#undef FEEDFORWARD_LINEAR_DROPOUT_NODE
#endif
};

// The following patterns are used to fuse feedforward in backward
// 1. residual_add_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
// activation_grad -> linear1_grad -> layer_norm_grad
// 2. layer_norm_grad -> residual_add_grad -> dropout2_grad -> linear2_grad ->
// dropout1_grad -> activation_grad -> linear1_grad
// other cases: may delete residual_add_grad, dropout1_grad, dropout2_grad
// operators
struct FusedFeedForwardBwd : public PatternBase {
  FusedFeedForwardBwd(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fused_feedforward_bwd") {}

  PDNode* operator()(PDNode* x,
                     std::unordered_set<std::string> act_grad_types,
                     bool use_mp,
                     bool pre_layer_norm,
                     bool add_residual,
                     bool use_dropout_1,
                     bool use_dropout_2);
#ifndef FEEDFORWARD_LINEAR_DROPOUT_GRAD_NODE
#define FEEDFORWARD_LINEAR_DROPOUT_GRAD_NODE(suffix__) \
  PATTERN_DECL_NODE(matmul_op_grad_##suffix__);        \
  PATTERN_DECL_NODE(matmul_in_##suffix__);             \
  PATTERN_DECL_NODE(matmul_w_##suffix__);              \
  PATTERN_DECL_NODE(matmul_in_grad_##suffix__);        \
  PATTERN_DECL_NODE(matmul_w_grad_##suffix__);         \
  PATTERN_DECL_NODE(ele_add_op_grad_##suffix__);       \
  PATTERN_DECL_NODE(ele_add_in_##suffix__);            \
  PATTERN_DECL_NODE(ele_add_bias_##suffix__);          \
  PATTERN_DECL_NODE(ele_add_in_grad_##suffix__);       \
  PATTERN_DECL_NODE(ele_add_bias_grad_##suffix__);     \
  PATTERN_DECL_NODE(dropout_op_grad_##suffix__);       \
  PATTERN_DECL_NODE(dropout_mask_##suffix__);          \
  PATTERN_DECL_NODE(dropout_in_grad_##suffix__);

  // LayerNorm Grad: layer_norm_grad
  PATTERN_DECL_NODE(layer_norm_op_grad);
  PATTERN_DECL_NODE(layer_norm_in);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_in_grad);
  PATTERN_DECL_NODE(layer_norm_scale_grad);
  PATTERN_DECL_NODE(layer_norm_bias_grad);
  // Mode parallelism
  PATTERN_DECL_NODE(c_identity_op);
  PATTERN_DECL_NODE(c_identity_out);
  PATTERN_DECL_NODE(c_allreduce_sum_op);
  PATTERN_DECL_NODE(c_allreduce_sum_out);
  // Linear 1 and Dropout 1: matmul_v2_grad + elementwise_add_grad +
  // dropout_grad
  FEEDFORWARD_LINEAR_DROPOUT_GRAD_NODE(1);
  // Activation Grad: gelu_grad or relu_add
  PATTERN_DECL_NODE(act_op_grad);
  PATTERN_DECL_NODE(act_in);
  PATTERN_DECL_NODE(act_in_grad);
  // Linear 2 and Dropout 2: matmul_v2_grad + elementwise_add_grad +
  // dropout_grad
  FEEDFORWARD_LINEAR_DROPOUT_GRAD_NODE(2);
  // Residual Add: elementwise_add
  PATTERN_DECL_NODE(ele_add_op_grad_3);
  PATTERN_DECL_NODE(ele_add_in_3);
  PATTERN_DECL_NODE(ele_add_bias_3);
  PATTERN_DECL_NODE(ele_add_in_grad_3);
  PATTERN_DECL_NODE(ele_add_bias_grad_3);
  PATTERN_DECL_NODE(sum_op);
  PATTERN_DECL_NODE(sum_out);

#undef FEEDFORWARD_LINEAR_DROPOUT_GRAD_NODE
#endif
};
// The following patterns are used to fuse Conv + BN + Add + Act
// pattern:
// (a) shortcut=true
//
//     |        |
//  [Conv]      |
//   [BN]       |
//     \       /
//       [Add]
//       [Act]
//          |
//
// (b) shortcut=false
//    |         |
// [Conv]     [Conv]
//  [BN]       [BN]
//     \       /
//       [Add]
//       [Act]
//         |
struct ConvBNAddAct : public PatternBase {
  ConvBNAddAct(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_bn_add_act") {}

  PDNode* operator()(const std::unordered_set<std::string>& act_types,
                     bool shortcut,
                     bool is_training);

  // declare operator node's name
  PATTERN_DECL_NODE(conv1_op);
  PATTERN_DECL_NODE(bn1_op);
  PATTERN_DECL_NODE(conv2_op);
  PATTERN_DECL_NODE(bn2_op);
  PATTERN_DECL_NODE(elewise_add_op);
  PATTERN_DECL_NODE(act_op);

  // declare variable node's name
  PATTERN_DECL_NODE(x1);
  PATTERN_DECL_NODE(conv1_w);
  PATTERN_DECL_NODE(conv1_out);
  PATTERN_DECL_NODE(bn1_scale);
  PATTERN_DECL_NODE(bn1_bias);
  PATTERN_DECL_NODE(bn1_variance);
  PATTERN_DECL_NODE(bn1_mean);
  PATTERN_DECL_NODE(bn1_mean_out);
  PATTERN_DECL_NODE(bn1_variance_out);
  PATTERN_DECL_NODE(bn1_saved_variance);
  PATTERN_DECL_NODE(bn1_saved_mean);
  PATTERN_DECL_NODE(bn1_out);
  PATTERN_DECL_NODE(x2);
  PATTERN_DECL_NODE(conv2_w);
  PATTERN_DECL_NODE(conv2_out);
  PATTERN_DECL_NODE(bn2_scale);
  PATTERN_DECL_NODE(bn2_bias);
  PATTERN_DECL_NODE(bn2_variance);
  PATTERN_DECL_NODE(bn2_mean);
  PATTERN_DECL_NODE(bn2_mean_out);
  PATTERN_DECL_NODE(bn2_variance_out);
  PATTERN_DECL_NODE(bn2_saved_variance);
  PATTERN_DECL_NODE(bn2_saved_mean);
  PATTERN_DECL_NODE(bn2_out);
  PATTERN_DECL_NODE(add_out);
  PATTERN_DECL_NODE(act_out);
};

// The following patterns are used to fuse Conv + BN + Act + ConvBNStats
// pattern:
struct ConvBNActConvBNStats : public PatternBase {
  ConvBNActConvBNStats(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "conv_bn_act_conv_bnstats") {}

  PDNode* operator()(const std::unordered_set<std::string>& act_types,
                     bool is_training);

  // declare operator node's name
  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(bn_op);
  PATTERN_DECL_NODE(act_op);
  PATTERN_DECL_NODE(conv_bnstats_op);

  // declare variable node's name
  PATTERN_DECL_NODE(conv_x);
  PATTERN_DECL_NODE(conv_w);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_variance);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_variance_out);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(act_out);
};

// The following patterns are used to fuse dConv + dAct + dBN
// pattern:
struct BNActConvGrad : public PatternBase {
  BNActConvGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_act_conv_grad") {}

  PDNode* operator()(const std::unordered_set<std::string>& act_grad_types);
  // declare operator node's name
  PATTERN_DECL_NODE(conv_grad);
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(batch_norm_grad);
  // declare variable node's name
  PATTERN_DECL_NODE(d_conv_out);
  PATTERN_DECL_NODE(conv_w);
  PATTERN_DECL_NODE(d_conv_x);
  PATTERN_DECL_NODE(d_conv_w);
  PATTERN_DECL_NODE(d_act_x);
  PATTERN_DECL_NODE(bn_x);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_variance);
  PATTERN_DECL_NODE(d_bn_x);
  PATTERN_DECL_NODE(d_bn_scale);
  PATTERN_DECL_NODE(d_bn_bias);
};

// The following patterns are used to fuse BN + Add + Act + Conv backward
// pattern, [sum] is optional, controlled by with_sum
// (a) shortcut=true
//     |      |
//  [dBN]     /
//     |     /
//  [dAdd]---
//     |
//  [dReLU]
//     |
//  [sum (optional)]
//     |       |
//  [dConv]    |
//     |       |
//
// (b) shortcut=false
//     |      |
//  [dBN]   [dBN]
//     |     /
//  [dAdd]---
//     |
//  [dReLU]
//     |
//  [sum (optional)]
//     |       |
//  [dConv]    |
//     |       |
struct BNAddActConvGrad : public PatternBase {
  BNAddActConvGrad(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "bn_add_act_conv_grad") {}

  PDNode* operator()(const std::unordered_set<std::string>& act_grad_types,
                     bool shortcut,
                     bool with_sum);
  // declare operator node's name
  PATTERN_DECL_NODE(conv_grad);
  PATTERN_DECL_NODE(act_grad);
  PATTERN_DECL_NODE(elewise_add_grad);
  PATTERN_DECL_NODE(sum);
  PATTERN_DECL_NODE(batch_norm1_grad);
  PATTERN_DECL_NODE(batch_norm2_grad);
  // declare variable node's name
  // dConv
  PATTERN_DECL_NODE(d_conv_out);
  PATTERN_DECL_NODE(conv_x);
  PATTERN_DECL_NODE(conv_w);
  PATTERN_DECL_NODE(d_conv_x);
  PATTERN_DECL_NODE(d_conv_w);
  // (optional) sum
  PATTERN_DECL_NODE(sum_in_extra);
  PATTERN_DECL_NODE(sum_out);
  // dAct
  PATTERN_DECL_NODE(d_act_x);
  // dAdd
  PATTERN_DECL_NODE(d_elewise_add_x);
  PATTERN_DECL_NODE(d_elewise_add_y);
  // BN 1
  PATTERN_DECL_NODE(bn1_x);
  PATTERN_DECL_NODE(bn1_scale);
  PATTERN_DECL_NODE(bn1_bias);
  PATTERN_DECL_NODE(bn1_saved_mean);
  PATTERN_DECL_NODE(bn1_saved_variance);
  PATTERN_DECL_NODE(d_bn1_x);
  PATTERN_DECL_NODE(d_bn1_scale);
  PATTERN_DECL_NODE(d_bn1_bias);
  // (optional) BN 2
  PATTERN_DECL_NODE(bn2_x);
  PATTERN_DECL_NODE(bn2_scale);
  PATTERN_DECL_NODE(bn2_bias);
  PATTERN_DECL_NODE(bn2_saved_mean);
  PATTERN_DECL_NODE(bn2_saved_variance);
  PATTERN_DECL_NODE(d_bn2_x);
  PATTERN_DECL_NODE(d_bn2_scale);
  PATTERN_DECL_NODE(d_bn2_bias);
};

struct SparseConvOptimPartern : public PatternBase {
  SparseConvOptimPartern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "sparse_conv_optim_partern") {}

  void operator()();
  PATTERN_DECL_NODE(sp_conv3d_x);
  PATTERN_DECL_NODE(sp_conv3d_kernel);
  PATTERN_DECL_NODE(sp_conv3d_op);
  PATTERN_DECL_NODE(sp_conv3d_out);
};

}  // namespace patterns

// Link two ir::Nodes from each other.
#define IR_NODE_LINK_TO(a, b) \
  a->outputs.push_back(b);    \
  b->inputs.push_back(a);

// UnLink 2 ir::Nodes from each other.
#define IR_NODE_UNLINK(a, b)                                                  \
  a->outputs.erase(                                                           \
      std::remove(std::begin(a->outputs), std::end(a->outputs), b),           \
      std::end(a->outputs));                                                  \
  b->inputs.erase(std::remove(std::begin(b->inputs), std::end(b->inputs), a), \
                  std::end(b->inputs));

// Set the out_var as the output of the op
#define IR_OP_VAR_LINK(op, out_var) \
  op->outputs.push_back(out_var);   \
  out_var->inputs.clear();          \
  out_var->inputs.push_back(op);

// Set the in_var as the input of the op
#define IR_VAR_OP_LINK(in_var, op) \
  in_var->outputs.clear();         \
  in_var->outputs.push_back(op);   \
  op->inputs.push_back(in_var);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
