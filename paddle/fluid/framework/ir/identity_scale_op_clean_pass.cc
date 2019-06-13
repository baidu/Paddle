// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/identity_scale_op_clean_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

void IdentityScaleOpCleanPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("identity_scale_op_clean", graph);

  // scale_in -> scale_op -> scale_out -> fetch_op
  // ==>
  // scale_in -> fetch_op
  GraphPatternDetector detector;
  auto scale_in = detector.mutable_pattern()
                      ->NewNode("scale_in")
                      ->assert_is_op_input("scale");
  auto scale_op = detector.mutable_pattern()
                      ->NewNode("scale_fuse")
                      ->assert_is_op("scale")
                      ->assert_op_attr<float>("scale", 1.)
                      ->assert_op_attr<float>("bias", 0.);
  auto scale_out =
      detector.mutable_pattern()
          ->NewNode("scale_out")
          ->assert_is_op_output("scale")
          ->AsIntermediate()
          // scale's output var should has only one consumer, or it can't be
          // removed.
          ->assert_more([](Node* x) { return x->outputs.size() == 1UL; });
  auto behind_op =
      detector.mutable_pattern()->NewNode("behind_op")->assert_is_op("fetch");

  scale_op->LinksFrom({scale_in}).LinksTo({scale_out});
  behind_op->LinksFrom({scale_out});

  GraphPatternDetector::handle_t handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    Node* scale_op_var = subgraph.at(scale_op);
    Node* scale_in_var = subgraph.at(scale_in);
    Node* scale_out_var = subgraph.at(scale_out);
    Node* behind_op_var = subgraph.at(behind_op);
    // Link pre_op directly to scale_out
    const std::string scale_in_name = scale_in_var->Name();
    const std::string scale_out_name = scale_out_var->Name();
    // Remove links in graph
    GraphSafeRemoveNodes(graph, {scale_op_var, scale_out_var});
    // Modify proto message
    auto* behind_op_desc = behind_op_var->Op();
    for (auto& parameter : *behind_op_desc->Proto()->mutable_inputs()) {
      auto* arguments = parameter.mutable_arguments();
      auto it = std::find(arguments->begin(), arguments->end(), scale_out_name);
      PADDLE_ENFORCE(it != arguments->end());
      *it = scale_in_name;
    }
    IR_NODE_LINK_TO(scale_in_var, behind_op_var);
  };

  detector(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(identity_scale_op_clean_pass,
              paddle::framework::ir::IdentityScaleOpCleanPass);
