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

#include "paddle/fluid/framework/ir/mkldnn/softplus_activation_mkldnn_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/onednn/activation_fuse_pass.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void SoftplusActivationOneDNNPass::ApplyImpl(Graph *graph) const {
  auto act_types = phi::funcs::GetSupportedActivations();

  // Currently softplus can't be fused with hard_sigmoid
  act_types.erase(
      std::remove(act_types.begin(), act_types.end(), "hard_sigmoid"),
      act_types.end());

  for (const auto &act_type : act_types) {
    FuseSoftplusActivation(graph, act_type);
  }
}

void SoftplusActivationOneDNNPass::FuseSoftplusActivation(
    Graph *graph, const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, phi::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("softplus_activation", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation softplus_activation_pattern(
      gpd.mutable_pattern(), "softplus_activation");
  softplus_activation_pattern("softplus", act_type);

  int found_softplus_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse softplus with activation op.";
    GET_IR_NODE_FROM_SUBGRAPH(
        softplus_out, preceding_op_out, softplus_activation_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        activation_out, activation_out, softplus_activation_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softplus, preceding_op, softplus_activation_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        activation, activation, softplus_activation_pattern);

    phi::funcs::SetActivationAttrs(softplus->Op(), activation->Op(), act_type);
    softplus->Op()->SetOutput("Out", {activation_out->Name()});

    IR_OP_VAR_LINK(softplus, activation_out);
    GraphSafeRemoveNodes(g, {activation, softplus_out});
    found_softplus_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_softplus_activation_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      (found_softplus_activation_count > 0))
    PrettyLogDetail("---    fused %d softplus with %s activation",
                    found_softplus_activation_count,
                    act_type);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(softplus_activation_mkldnn_fuse_pass,
              paddle::framework::ir::SoftplusActivationOneDNNPass);
REGISTER_PASS_CAPABILITY(softplus_activation_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("softplus", 1)
            .EQ("abs", 0)
            .LE("clip", 1)
            .EQ("gelu", 0)
            .LE("hard_swish", 0)
            .LE("leaky_relu", 1)
            .LE("mish", 1)
            .EQ("relu", 0)
            .EQ("relu6", 0)
            .EQ("sigmoid", 0)
            .EQ("sqrt", 0)
            .EQ("swish", 0)
            .EQ("tanh", 0));
