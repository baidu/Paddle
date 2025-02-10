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
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::framework::ir::patterns {

struct ConcatPattern : public PatternBase {
  ConcatPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(concat);
  // declare variable node's name
  PATTERN_DECL_NODE(any_op_out);
  PATTERN_DECL_NODE(concat_out);
};

ConcatPattern::ConcatPattern(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* any_op = pattern->NewNode(any_op_repr())->assert_is_op();
  auto* any_op_out = pattern->NewNode(any_op_out_repr())
                         ->assert_is_op_input("concat", "X")
                         ->assert_has_n_inputs(1)
                         ->assert_has_n_outputs(1);
  auto* concat = pattern->NewNode(concat_repr())
                     ->assert_is_op("concat")
                     ->assert_has_n_inputs(1)
                     ->assert_more([](Node* node) {
                       return node->Op()->Input("X").size() == 1;
                     });
  auto* concat_out =
      pattern->NewNode(concat_out_repr())->assert_is_op_output("concat", "Out");
  any_op->LinksTo({any_op_out});
  concat->LinksFrom({any_op_out}).LinksTo({concat_out});
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

/*
Delete "concat" if only has one input.
*/
class DeleteConcatOpPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"delete_concat_op_pass"};
};

void DeleteConcatOpPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::ConcatPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteConcatOpPass fuse";
#define GET_IR_NODE(node_) GET_IR_NODE_FROM_SUBGRAPH(node_, node_, pattern)
    GET_IR_NODE(any_op);
    GET_IR_NODE(concat);
    GET_IR_NODE(any_op_out);
    GET_IR_NODE(concat_out);
#undef GET_IR_NODE

    any_op->Op()->RenameOutput(any_op_out->Name(), concat_out->Name());
    IR_NODE_LINK_TO(any_op, concat_out);

    std::unordered_set<const Node*> delete_nodes{any_op_out, concat};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_concat_op_pass, paddle::framework::ir::DeleteConcatOpPass);

REGISTER_PASS_CAPABILITY(delete_concat_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "concat", 0));
