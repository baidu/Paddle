/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/elementwiseadd_transpose_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {
class Node;
}  // namespace paddle::framework::ir

namespace paddle::framework::ir::patterns {
//  input_x  input_y, input_x and input_y are both (n,h*w,c)
//    |        |
//  elementwise_add (n,h*w,c)
//    |
//   reshape (n,h, w,c)
//    |
//   transpose ((n,c,h,w))
//    |
//
// fuse ->
//
//   |
//  elementwiseadd_transpose
//   |
struct ElementwiseAddTransposePattern : public PatternBase {
  ElementwiseAddTransposePattern(PDPattern *pattern,
                                 const std::string &name_scope)
      : PatternBase(pattern, name_scope, "elementwiseadd_transpose") {}
  void operator()(PDNode *x, PDNode *y);
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(elementwise_out);
  PATTERN_DECL_NODE(reshape);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(transpose);
  PATTERN_DECL_NODE(transpose_out);
};
void ElementwiseAddTransposePattern::operator()(PDNode *x, PDNode *y) {
  auto *elementwise = pattern->NewNode(elementwise_repr())
                          ->assert_is_op("elementwise_add")
                          ->assert_has_n_outputs(1);
  auto *elementwise_out = pattern->NewNode(elementwise_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->assert_is_op_input("reshape2")
                              ->AsIntermediate();
  elementwise->LinksFrom({x, y}).LinksTo({elementwise_out});
  auto *reshape = pattern->NewNode(reshape_repr())->assert_is_op("reshape2");
  auto *reshape_out = pattern->NewNode(reshape_out_repr())
                          ->assert_is_op_output("reshape2")
                          ->assert_is_op_input("transpose2")
                          ->AsIntermediate();
  reshape->LinksFrom({elementwise_out}).LinksTo({reshape_out});
  auto *transpose =
      pattern->NewNode(transpose_repr())->assert_is_op("transpose2");
  auto *transpose_out = pattern->NewNode(transpose_out_repr())
                            ->assert_is_op_output("transpose2")
                            ->AsOutput();
  transpose->LinksFrom({reshape_out}).LinksTo({transpose_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int ElementwiseAddTransposeFusePass::ApplyEleTransPattern(
    ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("eleadd_transpose_fuse", graph);
  int found_subgraph_count = 0;
  GraphPatternDetector gpd;
  PDNode *x = nullptr;
  PDNode *y = nullptr;
  x = gpd.mutable_pattern()
          ->NewNode("eleadd_transpose/x")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "X");

  y = gpd.mutable_pattern()
          ->NewNode("eleadd_transpose/y")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "Y");
  patterns::ElementwiseAddTransposePattern fused_pattern(
      gpd.mutable_pattern(), "eleadd_transpose_fuse");
  fused_pattern(x, y);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    VLOG(4) << "handle elementwiseadd transpose fuse";

    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape, reshape, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose, transpose, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, fused_pattern);

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "elementwiseadd transpose pass in op compat failed.";
      return;
    }
    // elementwiseadd_trans is suit for
    // nhwc-to-nchw transpose after elementwise add
    // check for it
    std::vector<int> trans_axis =
        PADDLE_GET_CONST(std::vector<int>, transpose->Op()->GetAttr("axis"));
    if (trans_axis != std::vector<int>{0, 3, 1, 2}) {
      VLOG(1)
          << "elementwiseadd transpose fuse pass, transpose axis check fail, "
             "stop fusion";
      return;
    }
    if (!reshape->Op()->HasAttr("shape")) {
      VLOG(1) << "reshape op in elementwise_add_transpose fusion do not found "
                 "shape attr, the fusion will be stopped.";
      return;
    }
    std::vector<int> shape_attr =
        PADDLE_GET_CONST(std::vector<int>, reshape->Op()->GetAttr("shape"));
    VLOG(4) << "Fuse elementwiseadd transpose, with reshape attr:"
            << shape_attr[0] << ", " << shape_attr[1] << ", " << shape_attr[2]
            << ", " << shape_attr[3];
    if (shape_attr[1] <= 0 || shape_attr[2] <= 0 || shape_attr.size() != 4) {
      VLOG(1) << "found that shape_attr[1] and shape_attr[2]<=0 for reshape op "
                 "in elementwise_add_transpose, "
                 "currently, the elementwiseadd transpose pass only support "
                 "reshape bay shape attr rather than shape tensor."
                 "Therefore, the fusion will be stopped.";
      return;
    }
    if (shape_attr[3] % 8 != 0) {
      VLOG(1)
          << "found that shape_attr[3](channel size) mod 8 !=0 for reshape op "
             "in elementwise_add_transpose, "
             "currently, the elementwiseadd transpose pass only support "
             "channel size mod 8 == 0 for khwc8 trt format"
             "Therefore, the fusion will be stopped.";
      return;
    }
    std::unordered_set<const Node *> del_node_set;
    OpDesc new_desc;
    new_desc.SetType("fuse_eleadd_transpose");
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetOutput("Out", {transpose_out->Name()});
    new_desc.SetAttr("axis", elementwise->Op()->GetAttr("axis"));
    new_desc.SetAttr("output_shape", shape_attr);
    new_desc.Flush();
    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.
    del_node_set.insert(elementwise);
    del_node_set.insert(elementwise_out);
    del_node_set.insert(reshape);
    del_node_set.insert(reshape_out);
    del_node_set.insert(transpose);
    GraphSafeRemoveNodes(graph, del_node_set);
    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    IR_NODE_LINK_TO(fused_node, transpose_out);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}
void ElementwiseAddTransposeFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("elementwiseadd_transpose_fuse_pass", graph);
  int found_subgraph_count = ApplyEleTransPattern(graph);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(elementwiseadd_transpose_pass,
              paddle::framework::ir::ElementwiseAddTransposeFusePass);
REGISTER_PASS_CAPABILITY(elementwiseadd_transpose_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0));
