/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/scope.h"

USE_NO_KERNEL_OP(cinn_launch);
USE_OP(elementwise_add);

namespace paddle {
namespace operators {

using framework::ir::Graph;
using framework::ir::Node;
using framework::paddle2cinn::CinnCompiler;

std::unique_ptr<Graph> CreateOnlyElementwiseAddGraph(
    const std::string& x_name, const std::string& y_name,
    const std::string& out_name) {
  auto g = std::make_unique<Graph>(framework::ProgramDesc());
  framework::OpDesc feed_op_x, feed_op_y;
  feed_op_x.SetType("feed");
  feed_op_x.SetOutput("Out", {x_name});
  feed_op_y.SetType("feed");
  feed_op_y.SetOutput("Out", {y_name});

  framework::VarDesc x_var(x_name);
  framework::VarDesc y_var(y_name);
  framework::VarDesc out_var(out_name);

  framework::OpDesc elementwise_add_op;
  elementwise_add_op.SetType("add");
  elementwise_add_op.SetInput("X", {x_name});
  elementwise_add_op.SetInput("Y", {y_name});
  elementwise_add_op.SetOutput("Out", {out_name});

  auto* feed_op_node_x = g->CreateOpNode(&feed_op_x);
  auto* feed_op_node_y = g->CreateOpNode(&feed_op_y);
  auto* elementwise_add_node = g->CreateOpNode(&elementwise_add_op);
  auto* x_node = g->CreateVarNode(&x_var);
  auto* y_node = g->CreateVarNode(&y_var);
  auto* out_node = g->CreateVarNode(&out_var);

  // fill op node
  feed_op_node_x->outputs = {x_node};
  feed_op_node_y->outputs = {y_node};
  elementwise_add_node->inputs = {x_node, y_node};
  elementwise_add_node->outputs = {out_node};

  // fill variable node
  x_node->inputs = {feed_op_node_x};
  x_node->outputs = {elementwise_add_node};
  y_node->inputs = {feed_op_node_y};
  y_node->outputs = {elementwise_add_node};
  out_node->inputs = {elementwise_add_node};
  return g;
}

TEST(CinnLaunchOpTest, TestElementwiseAddPass) {
  auto place = platform::CPUPlace();
  framework::Scope scope;

  // Step 1: Prepare test data
  const auto dimension_len = 5;
  const auto common_ddim = framework::make_ddim({dimension_len});
  auto* x_var = scope.Var("test_x");
  auto* x_tensor = x_var->GetMutable<framework::LoDTensor>();
  auto* x_data = x_tensor->mutable_data<float>(common_ddim, place);
  for (auto i = 0; i < dimension_len; ++i) {
    x_data[i] = i;
  }

  auto* y_var = scope.Var("test_y");
  auto* y_tensor = y_var->GetMutable<framework::LoDTensor>();
  auto* y_data = y_tensor->mutable_data<float>(common_ddim, place);
  for (auto i = 0; i < dimension_len; ++i) {
    y_data[i] = 2 * i;
  }

  auto* test_out_var = scope.Var("test_out");
  test_out_var->GetMutable<framework::LoDTensor>();
  auto* expected_out_var = scope.Var("expected_out");
  expected_out_var->GetMutable<framework::LoDTensor>();

  // Step 2: Cache test graph into CinnCompiler
  auto compilation_key = CinnCompiler::GetInstance()->AddGraph(
      CreateOnlyElementwiseAddGraph("test_x", "test_y", "test_out"));
  // Step 3: Create cinn_launch_op and elementwise_add op, then run ops
  auto cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
      "cinn_launch", {{"X", {"test_x", "test_y"}}}, {{"Out", {"test_out"}}},
      {{"compilation_key", compilation_key}});
  auto elementwise_add_op = paddle::framework::OpRegistry::CreateOp(
      "elementwise_add", {{"X", {"test_x"}}, {"Y", {"test_y"}}},
      {{"Out", {"expected_out"}}}, {{}});

  cinn_launch_op->Run(scope, place);
  elementwise_add_op->Run(scope, place);

  // Step 4. Compare computation results.
  const auto& test_out_tensor = test_out_var->Get<framework::LoDTensor>();
  const auto& expected_out_tensor =
      expected_out_var->Get<framework::LoDTensor>();
  ASSERT_TRUE(test_out_tensor.IsInitialized());
  ASSERT_TRUE(expected_out_tensor.IsInitialized());
  ASSERT_EQ(test_out_tensor.dims(), common_ddim);
  ASSERT_EQ(test_out_tensor.dims(), expected_out_tensor.dims());
  // TODO(CtfGo): remove comment to check accuracy
  // const auto* test_out_data = test_out_tensor.data<float>();
  // const auto* excepted_out_data = expected_out_tensor.data<float>();
  // for (auto i = 0; i < dimension_len; ++i) {
  //   EXPECT_FLOAT_EQ(test_out_data[i], excepted_out_data[i]);
  //   EXPECT_FLOAT_EQ(test_out_data[i], i * 3);
  // }
}

}  // namespace operators
}  // namespace paddle
