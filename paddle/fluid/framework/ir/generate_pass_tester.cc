// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/generate_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

REGISTER_GENERATE_PASS(generate_fc_fuse) {
  paddle::framework::ir::PassPairs pass_pairs;
  for (bool with_relu : {true, false}) {
    // pattern
    SUBGRAPH_(pattern) = [subgraph = &pattern, with_relu](
                             VAR_(x), VAR_(y), VAR_(z)) {
      VLOG(3) << "exec lambda func.";
      auto mul = OP_(mul)({{"X", x}, {"Y", y}}).Out("Out");
      auto ewadd = OP_(elementwise_add)({{"X", mul}, {"Y", z}}).Out("Out");
      if (with_relu) {  // NOLINT
        return OP_(relu)({"X", ewadd}).Out("Out");
      } else {
        return ewadd;
      }
    };
    // replace
    SUBGRAPH_(replace) = [subgraph = &replace](VAR_(x), VAR_(y), VAR_(z)) {
      auto& fc = OP_(fc)({{"Input", x}, {"W", y}, {"Bias", z}});
      return fc.Out("Out");
    };
    pass_pairs.AddPassDesc(pattern, replace);
  }
  return pass_pairs;
}

REGISTER_GENERATE_PASS(generate_multi_add_to_addn) {
  // pattern
  SUBGRAPH_(pattern) = [subgraph = &pattern](VAR_(x), VAR_(y), VAR_(z)) {
    auto ewadd1 = OP_(elementwise_add)({{"X", x}, {"Y", y}}).Out("Out");
    auto ewadd2 = OP_(elementwise_add)({{"X", ewadd1}, {"Y", z}}).Out("Out");
    return ewadd2;
  };
  // replace
  SUBGRAPH_(replace) = [subgraph = &replace](VAR_(x), VAR_(y), VAR_(z)) {
    return OP_(sum)({"X", {x, y, z}}).Out("Out");
  };
  return {pattern, replace};
}

REGISTER_GENERATE_PASS(generate_combine_matmul) {
  // pattern
  SUBGRAPH_(pattern) = [subgraph = &pattern](VAR_(x), VAR_(y), VAR_(z)) {
    auto matmul1 = OP_(matmul)({{"X", x}, {"Y", y}}).Out("Out");
    auto matmul2 = OP_(matmul)({{"X", x}, {"Y", z}}).Out("Out");
    return std::make_tuple(matmul1, matmul2);
  };
  // replace
  SUBGRAPH_(replace) = [subgraph = &replace](VAR_(x), VAR_(y), VAR_(z)) {
    auto concat = OP_(concat)({"X", {y, z}}).Out("Out");
    auto matmul = OP_(matmul)({{"X", x}, {"Y", concat}}).Out("Out");
    auto slice1 = OP_(slice)({"X", matmul}).Out("Out");
    auto slice2 = OP_(slice)({"X", matmul}).Out("Out");
    return std::make_tuple(slice1, slice2);
  };
  return {pattern, replace};
}

namespace paddle::framework::ir {

TEST(GeneratePass, construct_with_string) {
  std::string binary_str;
  register_generate_fc_fuse().MultiPassDesc().SerializeToString(&binary_str);
  GeneratePass generate_pass(binary_str);
}

TEST(GeneratePass, generate_fc_fuse) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, filters_0 bias_0)      conv2d           -> conv2d_out
  // conv2d_out                 relu             -> relu_out_0
  // (relu_out_0, weights_0)    mul              -> mul_out_0
  // (mul_out_0, bias_1)        elementwise_add  -> add_out_0
  // add_out_0                  relu             -> relu_out_1
  // (relu_out_1, weights_1)    mul              -> mul_out_1
  // (mul_out_1, bias_2)        elementwise_add  -> add_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* filters_0 = layers.data("conv2d_filters_0", {}, true);
  auto* bias_0 = layers.data("conv2d_bias_0", {}, true);
  auto* conv2d_out = layers.conv2d(a, filters_0, bias_0, false);
  auto* relu_out_0 = layers.relu(conv2d_out);
  auto* weights_0 = layers.data("weights_0", {}, true);
  auto* mul_out_0 = layers.mul(relu_out_0, weights_0);
  auto* bias_1 = layers.data("bias_1", {}, true);
  auto* add_out_0 = layers.elementwise_add(mul_out_0, bias_1, nullptr, 1);
  auto* relu_out_1 = layers.relu(add_out_0);
  auto* weights_1 = layers.data("weights_1", {}, true);
  auto* mul_out_1 = layers.mul(relu_out_1, weights_1);
  auto* bias_2 = layers.data("bias_2", {}, true);
  auto* add_out_1 = layers.elementwise_add(mul_out_1, bias_2, nullptr, 1);
  VLOG(4) << add_out_1;

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("generate_fc_fuse");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  int num_mul_nodes_before = GetNumOpNodes(graph, "mul");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_fc_nodes_after = GetNumOpNodes(graph, "fc");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 6,
                    common::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before,
                        num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fc_nodes_after,
                    2,
                    common::errors::InvalidArgument("num_fc_nodes_after=%d.",
                                                    num_fc_nodes_after));
  PADDLE_ENFORCE_EQ(num_mul_nodes_before,
                    num_fc_nodes_after,
                    common::errors::InvalidArgument(
                        "num_mul_nodes_before=%d, num_fc_nodes_after=%d.",
                        num_mul_nodes_before,
                        num_fc_nodes_after));
}

TEST(GeneratePass, generate_multi_add_to_addn) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, b)                     elementwise_add  -> add_out_0
  // (add_out_0, c)             elementwise_add  -> add_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* b = layers.data("b");
  auto* c = layers.data("c");
  auto* add_out_0 = layers.elementwise_add(a, b);
  layers.elementwise_add(add_out_0, c);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("generate_multi_add_to_addn");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  int num_add_nodes_before = GetNumOpNodes(graph, "elementwise_add");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_addn_nodes_after = GetNumOpNodes(graph, "sum");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 2,
                    common::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before,
                        num_nodes_after));
  PADDLE_ENFORCE_EQ(num_addn_nodes_after,
                    1,
                    common::errors::InvalidArgument("num_addn_nodes_after=%d.",
                                                    num_addn_nodes_after));
  PADDLE_ENFORCE_EQ(num_add_nodes_before,
                    num_addn_nodes_after + 1,
                    common::errors::InvalidArgument(
                        "num_add_nodes_before=%d, num_addn_nodes_after=%d.",
                        num_add_nodes_before,
                        num_addn_nodes_after));
}

TEST(GeneratePass, generate_combine_matmul) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, b)                     matmul           -> matmul_out_0
  // (a, c)                     matmul           -> matmul_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* b = layers.data("b");
  auto* c = layers.data("c");
  layers.matmul(a, b);
  layers.matmul(a, c);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("generate_combine_matmul");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  int num_matmul_nodes_before = GetNumOpNodes(graph, "matmul");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_matmul_nodes_after = GetNumOpNodes(graph, "matmul");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after - 4,
                    common::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before,
                        num_nodes_after));
  PADDLE_ENFORCE_EQ(num_matmul_nodes_after,
                    1,
                    common::errors::InvalidArgument(
                        "num_matmul_nodes_after=%d.", num_matmul_nodes_after));
  PADDLE_ENFORCE_EQ(
      num_matmul_nodes_before,
      num_matmul_nodes_after + 1,
      common::errors::InvalidArgument(
          "num_matmul_nodes_before=%d, num_matmul_nodes_after=%d.",
          num_matmul_nodes_before,
          num_matmul_nodes_after));
}

}  // namespace paddle::framework::ir
