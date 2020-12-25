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

#include "paddle/fluid/framework/ir/map_matmul_to_mul_pass.h"

#include <cmath>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void MapMatmul2MulPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "map_matmul_to_mul_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Matmul matmul_pattern(gpd.mutable_pattern(), name_scope);
  matmul_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "map matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, matmul_pattern);

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    bool transpose_Y =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_Y"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    std::vector<int64_t> x_shape = matmul_in_x->Var()->GetShape();
    std::vector<int64_t> y_shape = matmul_in_y->Var()->GetShape();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();

    if (!transpose_X && !transpose_Y && std::abs(alpha - 1.0) < 1e-5 &&
        x_rank == 2 && y_rank == 2) {
      OpDesc desc;
      desc.SetType("mul");
      desc.SetInput("X", {matmul_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(matmul_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {matmul_op});
      ++found_count;
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void Squeeze2MatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "squeeze2_matmul_fuse_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Squeeze2Matmul fuse_pattern(gpd.mutable_pattern(), name_scope);
  fuse_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "fuse squeeze2+matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(squeeze2_in_x, squeeze2_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(squeeze2_op, squeeze2_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, fuse_pattern);

    size_t squeeze2_in_x_rank = (squeeze2_in_x->Var()->GetShape()).size();
    std::vector<int> squeeze2_op_axes =
        BOOST_GET_CONST(std::vector<int>, squeeze2_op->Op()->GetAttr("axes"));

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    bool transpose_Y =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_Y"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    size_t matmul_in_x_rank = (matmul_in_x->Var()->GetShape()).size();
    size_t matmul_in_y_rank = (matmul_in_y->Var()->GetShape()).size();

    if (squeeze2_in_x_rank == 4 && squeeze2_op_axes == std::vector<int>{2, 3} &&
        !transpose_X && !transpose_Y && std::abs(alpha - 1.0) < 1e-5 &&
        matmul_in_x_rank == 2 && matmul_in_y_rank == 2) {
      OpDesc desc;
      desc.SetType("mul");
      desc.SetInput("X", {squeeze2_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(squeeze2_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {squeeze2_op, matmul_in_x, matmul_op});
      ++found_count;
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void Reshape2MatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "reshape2_matmul_fuse_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Reshape2Matmul fuse_pattern(gpd.mutable_pattern(), name_scope);
  fuse_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "fuse reshape2+matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_in_x, reshape2_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_op, reshape2_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, fuse_pattern);

    size_t reshape2_in_x_rank = (reshape2_in_x->Var()->GetShape()).size();
    std::vector<int> reshape2_op_shape =
        BOOST_GET_CONST(std::vector<int>, reshape2_op->Op()->GetAttr("shape"));

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    bool transpose_Y =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_Y"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    size_t matmul_in_x_rank = (matmul_in_x->Var()->GetShape()).size();
    size_t matmul_in_y_rank = (matmul_in_y->Var()->GetShape()).size();

    if (reshape2_in_x_rank == 4 && reshape2_op_shape.size() == 2 &&
        !transpose_X && !transpose_Y && std::abs(alpha - 1.0) < 1e-5 &&
        matmul_in_x_rank == 2 && matmul_in_y_rank == 2) {
      OpDesc desc;
      desc.SetType("mul");
      desc.SetInput("X", {reshape2_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(reshape2_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {reshape2_op, matmul_in_x, matmul_op});
      ++found_count;
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(map_matmul_to_mul_pass, paddle::framework::ir::MapMatmul2MulPass);
REGISTER_PASS_CAPABILITY(map_matmul_to_mul_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "matmul", 0));

REGISTER_PASS(squeeze2_matmul_fuse_pass,
              paddle::framework::ir::Squeeze2MatmulFusePass);
REGISTER_PASS_CAPABILITY(squeeze2_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul", 0)
            .EQ("squeeze2", 0));

REGISTER_PASS(reshape2_matmul_fuse_pass,
              paddle::framework::ir::Reshape2MatmulFusePass);
REGISTER_PASS_CAPABILITY(reshape2_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul", 0)
            .EQ("reshape2", 0));
