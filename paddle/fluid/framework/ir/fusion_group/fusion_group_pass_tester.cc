/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/fusion_group/fusion_group_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle::framework::ir {

void VisualizeGraph(std::unique_ptr<Graph>* graph, std::string graph_viz_path) {
  // Insert a graph_viz_pass to transform the graph to a .dot file.
  // It can be used for debug.
  auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  graph_viz_pass->Set("graph_viz_path", new std::string(graph_viz_path));
  graph->reset(graph_viz_pass->Apply(graph->release()));
}

std::unique_ptr<Graph> BuildElementwiseListGraph(bool backward = false) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (x, y)                     mul              -> tmp_0
  // (tmp_0, z)                 elementwise_add  -> tmp_1
  // tmp_1                      relu             -> tmp_2
  // (tmp_2, w)                 elementwise_add  -> tmp_3
  //
  // Expression: tmp_3 = relu(mul(x, y) + z) + w
  Layers layers;
  std::vector<int64_t> shape = {16, 32};
  auto* x = layers.data("x", {16, 16});
  auto* y = layers.data("y", {16, 32});
  auto* tmp_0 = layers.mul(x, y);
  auto* z = layers.data("z", shape);
  auto* tmp_1 = layers.elementwise_add(tmp_0, z);
  auto* tmp_2 = layers.relu(tmp_1);
  auto* w = layers.data("w", shape);
  auto* tmp_3 = layers.elementwise_add(tmp_2, w);
  std::vector<VarDesc*> elementwise_vars = {tmp_0, z, tmp_1, tmp_2, w, tmp_3};
  for (auto* var : elementwise_vars) {
    var->SetShape(shape);
  }

  if (backward) {
    layers.backward({tmp_3});
  }

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  for (auto* n : graph->Nodes()) {
    if (n && n->IsVar() && n->Var()) {
      n->Var()->SetDataType(proto::VarType::FP32);
    }
  }
  return graph;
}

std::unique_ptr<Graph> BuildElementwiseTreeGraph(bool backward = false) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (x0, y0)                   mul              -> tmp_0
  // x1                         sigmoid          -> tmp_1
  // (tmp_0, tmp_1)             elementwise_mul  -> tmp_2
  // x2                         sigmoid          -> tmp_3
  // x3                         tanh             -> tmp_4
  // (tmp_3, tmp_4)             elementwise_mul  -> tmp_5
  // (tmp_2, tmp_5)             elementwise_add  -> tmp_6
  // x4                         tanh             -> tmp_7
  // x5                         sigmoid          -> tmp_8
  // (tmp_7, tmp_8)             elementwise_mul  -> tmp_9
  // (tmp_6, tmp_9)             mul              -> tmp_10
  //
  // Expression: tmp_6 = mul(x0, y0) * sigmoid(x1) + sigmoid(x2) * tanh(x3)
  //             tmp_9 = tanh(x4) * sigmoid(x5)
  //             tmp_10 = mul(tmp_6, tmp_9)
  Layers layers;
  std::vector<int64_t> shape = {16, 32};
  auto* x0 = layers.data("x0", {16, 16});
  auto* y0 = layers.data("y0", {16, 32});
  auto* tmp_0 = layers.mul(x0, y0);
  auto* x1 = layers.data("x1", shape);
  auto* tmp_1 = layers.sigmoid(x1);
  auto* tmp_2 = layers.elementwise_mul(tmp_0, tmp_1);
  auto* x2 = layers.data("x2", shape);
  auto* tmp_3 = layers.sigmoid(x2);
  auto* x3 = layers.data("x3", shape);
  auto* tmp_4 = layers.tanh(x3);
  auto* tmp_5 = layers.elementwise_mul(tmp_3, tmp_4);
  auto* tmp_6 = layers.elementwise_add(tmp_2, tmp_5);
  auto* x4 = layers.data("x4", shape);
  auto* tmp_7 = layers.tanh(x4);
  auto* x5 = layers.data("x5", shape);
  auto* tmp_8 = layers.sigmoid(x5);
  auto* tmp_9 = layers.elementwise_mul(tmp_7, tmp_8);
  auto* tmp_10 = layers.mul(tmp_6, tmp_9);

  std::vector<VarDesc*> elementwise_vars = {
      tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9};
  for (auto* var : elementwise_vars) {
    var->SetShape(shape);
  }

  if (backward) {
    layers.backward({tmp_10});
  }

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  for (auto* n : graph->Nodes()) {
    if (n && n->IsVar() && n->Var()) {
      n->Var()->SetDataType(proto::VarType::FP32);
    }
  }
  return graph;
}

int TestMain(std::unique_ptr<Graph> graph, std::string prefix) {
  // VisualizeGraph(&graph, prefix + ".dot");
  auto pass = PassRegistry::Instance().Get("fusion_group_pass");
  pass->Set("use_gpu", new bool(true));
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  // VisualizeGraph(&graph, prefix + ".fusion_group.dot");
  int num_fusion_group_ops = GetNumOpNodes(graph, "fusion_group");
  VLOG(3) << DebugString(graph);

  return num_fusion_group_ops;
}

TEST(FusionGroupPass, elementwise_list) {
  std::unique_ptr<Graph> graph = BuildElementwiseListGraph(true);
  int num_fusion_group_ops = TestMain(std::move(graph), "elementwise_list");
  EXPECT_EQ(num_fusion_group_ops, 2);
}

TEST(FusionGroupPass, elementwise_tree) {
  std::unique_ptr<Graph> graph = BuildElementwiseTreeGraph(true);
  int num_fusion_group_ops = TestMain(std::move(graph), "elementwise_tree");
  EXPECT_EQ(num_fusion_group_ops, 4);
}

}  // namespace paddle::framework::ir

USE_PASS(fusion_group_pass);
USE_PASS(graph_viz_pass);
