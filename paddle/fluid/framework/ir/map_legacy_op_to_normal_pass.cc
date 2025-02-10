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

#include "paddle/fluid/framework/ir/map_legacy_op_to_normal_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void MapLegacyOp2NormalPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("map_legacy_op_to_normal_pass", graph);

  int found_count = 0;
  std::unordered_map<std::string, std::string> replaced_map{
      {"lookup_table", "lookup_table_v2"},
      {"top_k", "top_k_v2"},
  };

  auto nodes = graph->Nodes();

  for (auto& node : nodes) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    std::string op_type = op_desc->Type();
    if (!replaced_map.count(op_type)) continue;
    op_desc->SetType(replaced_map[op_type]);
    if (op_type == "top_k") {
      int axis = op_desc->GetAttrIfExists<int>("axis");
      op_desc->SetAttr("axis", axis);
    }

    op_desc->Flush();
    ++found_count;
  }

  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(map_legacy_op_to_normal_pass,
              paddle::framework::ir::MapLegacyOp2NormalPass);
