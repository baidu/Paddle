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

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

static bool GetBoolFromEnv(const std::string& str, bool def = false) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  if (strcmp(variable, "false") == 0 || strcmp(variable, "0") == 0) {
    return false;
  } else {
    return true;
  }
}

namespace patterns {

struct EmbeddingWithEltwiseAddXPUPattern : public PatternBase {
  EmbeddingWithEltwiseAddXPUPattern(PDPattern* pattern,
                                    const std::string& name_scope,
                                    int n_embedding_,
                                    const std::string& op_type,
                                    const std::string& pre_op_type);

  // declare operator node's name
  PATTERN_DECL_NODE(embedding0);
  PATTERN_DECL_NODE(embedding1);
  PATTERN_DECL_NODE(ewadd01);
  // declare variable node's name
  PATTERN_DECL_NODE(x0);
  PATTERN_DECL_NODE(x1);
  PATTERN_DECL_NODE(table0);
  PATTERN_DECL_NODE(table1);
  PATTERN_DECL_NODE(embedding_out0);
  PATTERN_DECL_NODE(embedding_out1);
  PATTERN_DECL_NODE(ewadd01_out);

  std::unordered_map<std::string, std::string> node_reprs;

 private:
  int n_embedding_;
  std::string op_type_;
  std::string pre_op_type_;
};

EmbeddingWithEltwiseAddXPUPattern::EmbeddingWithEltwiseAddXPUPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    int n_embedding,
    const std::string& op_type,
    const std::string& pre_op_type)
    : PatternBase(pattern, name_scope, name_scope),
      n_embedding_(n_embedding),
      op_type_(op_type),
      pre_op_type_(pre_op_type) {
  for (int i = 0; i < n_embedding; i++) {
    node_reprs["x" + std::to_string(i)] =
        PDNodeName(name_scope_, repr_, id_, "x" + std::to_string(i));
    node_reprs["table" + std::to_string(i)] =
        PDNodeName(name_scope_, repr_, id_, "table" + std::to_string(i));
    node_reprs["embedding" + std::to_string(i)] =
        PDNodeName(name_scope_, repr_, id_, "embedding" + std::to_string(i));
    node_reprs["embedding_out" + std::to_string(i)] = PDNodeName(
        name_scope_, repr_, id_, "embedding_out" + std::to_string(i));
    if (i - 1 >= 0) {
      auto ewadd_name = string::Sprintf("ewadd%d%d", i - 1, i);
      node_reprs[ewadd_name] = PDNodeName(name_scope_, repr_, id_, ewadd_name);
      auto ewadd_out_name = string::Sprintf("ewadd%d%d_out", i - 1, i);
      node_reprs[ewadd_out_name] =
          PDNodeName(name_scope_, repr_, id_, ewadd_out_name);
    }
  }
  PDNode* x0 = pattern->NewNode(x0_repr())
                   ->assert_is_op_input(op_type_, "Ids")
                   ->assert_var_not_persistable()
                   ->AsInput();
  PDNode* x1 = pattern->NewNode(x1_repr())
                   ->assert_is_op_input(op_type_, "Ids")
                   ->assert_var_not_persistable()
                   ->AsInput();
  PDNode* embedding0 =
      pattern->NewNode(embedding0_repr())->assert_is_op(op_type_);
  auto* table0 = pattern->NewNode(table0_repr())
                     ->assert_is_op_input(op_type_, "W")
                     ->AsInput();
  auto* embedding_out0 = pattern->NewNode(embedding_out0_repr())
                             ->assert_is_op_output(op_type_, "Out")
                             ->assert_is_op_input("elementwise_add", "X")
                             ->assert_has_n_outputs(1);
  auto* table1 = pattern->NewNode(table1_repr())
                     ->assert_is_op_input(op_type_, "W")
                     ->AsInput();
  auto* embedding1 =
      pattern->NewNode(embedding1_repr())->assert_is_op(op_type_);

  auto* embedding_out1 = pattern->NewNode(embedding_out1_repr())
                             ->assert_is_op_output(op_type_, "Out")
                             ->assert_is_op_input("elementwise_add", "Y");
  auto* ewadd01 =
      pattern->NewNode(ewadd01_repr())->assert_is_op("elementwise_add");
  auto* ewadd01_out = pattern->NewNode(ewadd01_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");
  embedding0->LinksFrom({x0, table0});
  embedding1->LinksFrom({x1, table1});
  embedding0->LinksTo({embedding_out0});
  embedding1->LinksTo({embedding_out1});
  ewadd01->LinksFrom({embedding_out0, embedding_out1});
  ewadd01->LinksTo({ewadd01_out});

  auto* last_ewadd_out = ewadd01_out;
  for (int i = 2; i < n_embedding; ++i) {
    auto x_name = node_reprs["x" + std::to_string(i)];
    auto table_name = node_reprs["table" + std::to_string(i)];
    auto embedding_name = node_reprs["embedding" + std::to_string(i)];
    auto embedding_out_name = node_reprs["embedding_out" + std::to_string(i)];
    auto* new_table = pattern->NewNode(table_name)
                          ->assert_is_op_input(op_type_, "W")
                          ->AsInput();
    auto* new_embedding =
        pattern->NewNode(embedding_name)->assert_is_op(op_type_);
    auto* new_embedding_out = pattern->NewNode(embedding_out_name)
                                  ->assert_is_op_output(op_type_, "Out")
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->assert_has_n_outputs(1);
    auto* new_x = pattern->NewNode(x_name)
                      ->assert_is_op_input(op_type_, "Ids")
                      ->AsInput();
    new_embedding->LinksFrom({new_x, new_table});
    new_embedding->LinksTo({new_embedding_out});
    auto ewadd_name =
        node_reprs["ewadd" + std::to_string(i - 1) + std::to_string(i)];
    auto ewadd_out_name = node_reprs["ewadd" + std::to_string(i - 1) +
                                     std::to_string(i) + "_out"];
    auto* new_ewadd =
        pattern->NewNode(ewadd_name)->assert_is_op("elementwise_add");
    auto* new_ewadd_out = pattern->NewNode(ewadd_out_name)
                              ->assert_is_op_output("elementwise_add", "Out");
    new_ewadd->LinksFrom({last_ewadd_out, new_embedding_out});
    new_ewadd->LinksTo({new_ewadd_out});
    last_ewadd_out = new_ewadd_out;
  }
  last_ewadd_out->AsOutput();
}

}  // namespace patterns

class EmbeddingWithEltwiseAddXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyImpl(ir::Graph* graph,
                 int n_embedding,
                 const std::string op_type,
                 const std::string pre_op_type) const;

  const std::string name_scope_{"embedding_with_eltwise_add_xpu_fuse_pass"};
};

void EmbeddingWithEltwiseAddXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init(name_scope_, graph);
  std::vector<std::string> pre_op_types{"reshape2", "squeeze2", ""};
  std::vector<std::string> op_types{"lookup_table", "lookup_table_v2"};
  for (auto& pre_op_type : pre_op_types) {
    for (int n_embedding : {4, 3, 2}) {
      for (auto& op_type : op_types) {
        ApplyImpl(graph, n_embedding, op_type, pre_op_type);
      }
    }
  }
}

void EmbeddingWithEltwiseAddXPUFusePass::ApplyImpl(
    ir::Graph* graph,
    int n_embedding,
    const std::string op_type,
    const std::string pre_op_type) const {
  GraphPatternDetector gpd;
  patterns::EmbeddingWithEltwiseAddXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, n_embedding, op_type, pre_op_type);
  int found_subgraph_count = 0;
#define GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(name, rt_node, pat)                \
  PADDLE_ENFORCE_NE(                                                         \
      subgraph.count(pat.PatternBase::pattern->RetrieveNode(name)),          \
      0UL,                                                                   \
      common::errors::NotFound("Node not found for PDNode %s", name));       \
  Node* rt_node = subgraph.at(pat.PatternBase::pattern->RetrieveNode(name)); \
  PADDLE_ENFORCE_NOT_NULL(                                                   \
      rt_node,                                                               \
      common::errors::NotFound("node %s not exists in the sub-graph",        \
                               #rt_node));

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    std::vector<std::string> x_names;
    std::vector<std::string> table_names;
    std::vector<Node*> x_nodes;
    std::vector<Node*> table_nodes;
    std::vector<Node*> embedding_nodes;
    auto output_name = pattern.node_reprs[string::Sprintf(
        "ewadd%d%d_out", n_embedding - 2, n_embedding - 1)];
    GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(output_name, output_node, pattern)
    std::unordered_set<const Node*> delete_nodes;
    for (int i = 0; i < n_embedding; ++i) {
      // Ids
      auto x_name = pattern.node_reprs["x" + std::to_string(i)];
      GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(x_name, x_node, pattern)
      x_nodes.push_back(x_node);
      x_names.push_back(x_node->Name());
      // Tables
      auto table_name = pattern.node_reprs["table" + std::to_string(i)];
      GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(table_name, table_node, pattern)
      table_nodes.push_back(table_node);
      table_names.push_back(table_node->Name());
      // Embedding
      auto embedding_name = pattern.node_reprs["embedding" + std::to_string(i)];
      GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(embedding_name, embedding_node, pattern)
      embedding_nodes.push_back(embedding_node);
      delete_nodes.insert(embedding_node);
      auto embedding_out_name =
          pattern.node_reprs["embedding_out" + std::to_string(i)];
      GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(
          embedding_out_name, embedding_out_node, pattern)
      delete_nodes.insert(embedding_out_node);
      if (i - 1 >= 0) {
        auto ewadd_name =
            pattern.node_reprs[string::Sprintf("ewadd%d%d", i - 1, i)];
        GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(ewadd_name, ewadd_node, pattern)
        delete_nodes.insert(ewadd_node);
        auto ewadd_out_name =
            pattern.node_reprs[string::Sprintf("ewadd%d%d_out", i - 1, i)];
        GET_IR_NODE_FROM_SUBGRAPH_BY_NAME(
            ewadd_out_name, ewadd_out_node, pattern)
        if (i != n_embedding - 1) {
          delete_nodes.insert(ewadd_out_node);
        }
      }
    }
    // Generate embedding_with_eltwise_add_xpu op
    framework::OpDesc embedding_with_eltwise_add_xpu_op_desc;
    embedding_with_eltwise_add_xpu_op_desc.SetType(
        "embedding_with_eltwise_add_xpu");
    embedding_with_eltwise_add_xpu_op_desc.SetInput("ids", x_names);
    embedding_with_eltwise_add_xpu_op_desc.SetInput("tables", table_names);
    embedding_with_eltwise_add_xpu_op_desc.SetOutput("out",
                                                     {output_node->Name()});
    embedding_with_eltwise_add_xpu_op_desc.SetAttr("n_embedding", n_embedding);
    int64_t padding_idx = PADDLE_GET_CONST(
        int64_t, embedding_nodes[0]->Op()->GetAttr("padding_idx"));
    if (GetBoolFromEnv("XPU_PADDING_IDX", true)) {
      padding_idx = -1;
    }
    embedding_with_eltwise_add_xpu_op_desc.SetAttr(
        "padding_idx", static_cast<int64_t>(padding_idx));
    auto* embedding_with_eltwise_add_xpu_op =
        graph->CreateOpNode(&embedding_with_eltwise_add_xpu_op_desc);
    for (size_t i = 0; i < x_nodes.size(); i++) {
      SAFE_IR_NODE_LINK_TO(x_nodes[i], embedding_with_eltwise_add_xpu_op);
      SAFE_IR_NODE_LINK_TO(table_nodes[i], embedding_with_eltwise_add_xpu_op);
    }
    SAFE_IR_NODE_LINK_TO(embedding_with_eltwise_add_xpu_op, output_node);
    // delete useless node
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(embedding_with_eltwise_add_xpu_fuse_pass,
              paddle::framework::ir::EmbeddingWithEltwiseAddXPUFusePass);

REGISTER_PASS_CAPABILITY(embedding_with_eltwise_add_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "embedding_with_eltwise_add_xpu", 0));
