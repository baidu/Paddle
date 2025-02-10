/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/ir/delete_weight_dequant_linear_op_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle::framework::ir {

template <typename T>
void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  dev_ctx->HostAlloc<T>(tensor, tensor->numel() * sizeof(T));
}

template <typename T>
Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope<T>(param_scope, "scale", {1});

  return param_scope;
}

TEST(DeleteWeightDequantLinearOpPass, basic) {
  // inputs                    operator                   output
  // --------------------------------------------------------------------
  // (weight, scale)           dequantize_linear       -> dequantized_weight
  // (x, dequantized_weight)   matmul/fc/conv          -> matmul_out
  // (dequantized_weight)      while                   ->           [optional]

  Layers layers;

  auto* x = layers.data("x", {1, 128, 768});
  auto* weight = layers.data("weight", {768, 768}, true);
  auto* scale = layers.data("scale", {1}, true);
  auto* zero_point = layers.data("zero_point", {1}, true);
  auto* dequantized_weight =
      layers.dequantize_linear(weight, scale, zero_point);
  layers.matmul_v2(x, dequantized_weight);
  layers.while_loop({dequantized_weight});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));

  graph->Set("__param_scope__", CreateParamScope<float>());
  auto pass =
      PassRegistry::Instance().Get("delete_weight_dequant_linear_op_pass");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_dequant_nodes_after = GetNumOpNodes(graph, "dequantize_linear");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(
      num_nodes_before,
      num_nodes_after + 3,
      common::errors::InvalidArgument(
          "After pass, the number of nodes should be reduced by 3, but the "
          "number before pass is %d, after pass is %d.",
          num_nodes_before,
          num_nodes_after));
  PADDLE_ENFORCE_EQ(num_dequant_nodes_after,
                    0,
                    common::errors::InvalidArgument(
                        "After pass, the number of nodes of type "
                        "'dequantize_linear' should be 1, not %d.",
                        num_dequant_nodes_after));
}

TEST(DeleteWeightDequantLinearOpPass, basic_fp16) {
  // inputs                    operator                   output
  // --------------------------------------------------------------------
  // (weight, scale)           dequantize_linear       -> dequantized_weight
  // (x, dequantized_weight)   matmul/fc/conv          -> matmul_out
  // (dequantized_weight)      while                   ->           [optional]

  Layers layers;

  auto* x = layers.data("x", {1, 128, 768});
  auto* weight = layers.data("weight", {768, 768}, true);
  auto* scale = layers.data("scale", {1}, true);
  auto* zero_point = layers.data("zero_point", {1}, true);
  auto* dequantized_weight =
      layers.dequantize_linear(weight, scale, zero_point);
  layers.matmul_v2(x, dequantized_weight);
  layers.while_loop({dequantized_weight});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));

  graph->Set("__param_scope__", CreateParamScope<phi::dtype::float16>());
  auto pass =
      PassRegistry::Instance().Get("delete_weight_dequant_linear_op_pass");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_dequant_nodes_after = GetNumOpNodes(graph, "dequantize_linear");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(
      num_nodes_before,
      num_nodes_after + 3,
      common::errors::InvalidArgument(
          "After pass, the number of nodes should be reduced by 3, but the "
          "number before pass is %d, after pass is %d.",
          num_nodes_before,
          num_nodes_after));
  PADDLE_ENFORCE_EQ(num_dequant_nodes_after,
                    0,
                    common::errors::InvalidArgument(
                        "After pass, the number of nodes of type "
                        "'dequantize_linear' should be 1, not %d.",
                        num_dequant_nodes_after));
}

}  // namespace paddle::framework::ir

USE_PASS(delete_weight_dequant_linear_op_pass);
