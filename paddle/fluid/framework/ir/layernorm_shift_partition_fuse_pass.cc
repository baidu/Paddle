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

#include "paddle/fluid/framework/ir/layernorm_shift_partition_fuse_pass.h"

#include <cmath>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

class Node;

LayerNormShiftPartitionFusePass::LayerNormShiftPartitionFusePass() {
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumEQ(2)
      .End();
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End();
  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();
  AddOpCompat(OpCompat("roll"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int64_t>>()
      .End()
      .AddAttr("shifts")
      .IsType<std::vector<int64_t>>()
      .End();
}

int LayerNormShiftPartitionFusePass::ApplyPattern(ir::Graph* graph,
                                                  bool with_roll) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      common::errors::InvalidArgument(
          "The input graph of LayerNormShiftPartitionFusePass should not be "
          "nullptr."));
  FusePassBase::Init(scope_name_, graph);
  GraphPatternDetector gpd;
  patterns::LayernormShiftPartitionPattern shift_partition_pattern(
      gpd.mutable_pattern(), scope_name_, with_roll);
  shift_partition_pattern();
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "layernorm_shift_partition_fuse in op compat failed.";
      return;
    }
    if (with_roll) {
      VLOG(4)
          << "layernorm_shift_partition_fuse pass, shift_size>0, with roll op";
    } else {
      VLOG(4) << "layernorm_shift_partition_fuse pass, shift_size=0, without "
                 "roll op";
    }
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_in, layer_norm_in, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_op, layer_norm_op, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_out, layer_norm_out, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape1_op, reshape1_op, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape1_out, reshape1_out, shift_partition_pattern);
    Node* roll1_op = nullptr;
    Node* roll1_out = nullptr;
    if (with_roll) {
      GET_IR_NODE_FROM_SUBGRAPH(
          tmp_roll1_op, roll1_op, shift_partition_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(
          tmp_roll1_out, roll1_out, shift_partition_pattern);
      roll1_op = tmp_roll1_op;
      roll1_out = tmp_roll1_out;
    }
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_op, reshape2_op, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_out, reshape2_out, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose_op, transpose_op, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose_out, transpose_out, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape3_op, reshape3_op, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape3_out, reshape3_out, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape4_op, reshape4_op, shift_partition_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape4_out, reshape4_out, shift_partition_pattern);
    std::unordered_set<const Node*> del_node_set = {layer_norm_op,
                                                    layer_norm_out,
                                                    reshape1_op,
                                                    reshape1_out,
                                                    reshape2_op,
                                                    reshape2_out,
                                                    transpose_op,
                                                    transpose_out,
                                                    reshape3_op,
                                                    reshape3_out,
                                                    reshape4_op};
    if (with_roll) {
      del_node_set.insert(roll1_op);
      del_node_set.insert(roll1_out);
    }

    std::vector<int> shape_atr1 =
        PADDLE_GET_CONST(std::vector<int>, reshape1_op->Op()->GetAttr("shape"));
    std::vector<int> shape_atr2 =
        PADDLE_GET_CONST(std::vector<int>, reshape2_op->Op()->GetAttr("shape"));
    std::vector<int> shape_atr3 =
        PADDLE_GET_CONST(std::vector<int>, reshape3_op->Op()->GetAttr("shape"));
    std::vector<int> shape_atr4 =
        PADDLE_GET_CONST(std::vector<int>, reshape4_op->Op()->GetAttr("shape"));

    // emb dim should be same
    if (!((shape_atr1.back() == shape_atr2.back()) &&
          (shape_atr2.back() == shape_atr3.back()) &&
          (shape_atr3.back() == shape_atr4.back()))) {
      return;
    }

    if (shape_atr1[1] != shape_atr1[2]) {
      return;
    }
    int input_resolution = shape_atr1[1];

    if (shape_atr3[1] != shape_atr3[2]) {
      return;
    }
    int window_size = shape_atr2[2];
    if (window_size < 0 || input_resolution < 0) {
      return;
    }
    int shift_size = 0;
    if (with_roll) {
      std::vector<int64_t> roll_axis = PADDLE_GET_CONST(
          std::vector<int64_t>, roll1_op->Op()->GetAttr("axis"));
      std::vector<int64_t> roll_shifts = PADDLE_GET_CONST(
          std::vector<int64_t>, roll1_op->Op()->GetAttr("shifts"));
      if (roll_axis.size() != 2 || roll_axis[0] != 1 || roll_axis[1] != 2) {
        return;
      }
      if (roll_shifts.size() != 2 || roll_shifts[0] != roll_shifts[1]) {
        return;
      }
      shift_size = static_cast<int>(-roll_shifts[0]);
    }
    OpDesc new_op_desc;
    new_op_desc.SetType("layernorm_shift_partition");
    new_op_desc.SetInput("X", {layer_norm_in->Name()});
    new_op_desc.SetInput("Bias", {layer_norm_bias->Name()});
    new_op_desc.SetInput("Scale", {layer_norm_scale->Name()});
    new_op_desc.SetOutput("Y", {reshape4_out->Name()});
    new_op_desc.SetAttr("epsilon", layer_norm_op->Op()->GetAttr("epsilon"));
    new_op_desc.SetAttr("begin_norm_axis",
                        layer_norm_op->Op()->GetAttr("begin_norm_axis"));
    new_op_desc.SetAttr("window_size", window_size);
    new_op_desc.SetAttr("shift_size", shift_size);
    new_op_desc.SetAttr("input_resolution", input_resolution);
    new_op_desc.Flush();

    auto* layernorm_shift_partition = graph->CreateOpNode(&new_op_desc);

    IR_NODE_LINK_TO(layer_norm_in, layernorm_shift_partition);
    IR_NODE_LINK_TO(layer_norm_bias, layernorm_shift_partition);
    IR_NODE_LINK_TO(layer_norm_scale, layernorm_shift_partition);
    IR_NODE_LINK_TO(layernorm_shift_partition, reshape4_out);
    GraphSafeRemoveNodes(graph, del_node_set);
    ++found_count;
  };

  gpd(graph, handler);

  return found_count;
}

void LayerNormShiftPartitionFusePass::ApplyImpl(ir::Graph* graph) const {
  int found_count = 0;
  found_count += ApplyPattern(graph, true);
  found_count += ApplyPattern(graph, false);
  AddStatis(found_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(layernorm_shift_partition_fuse_pass,
              paddle::framework::ir::LayerNormShiftPartitionFusePass);
REGISTER_PASS_CAPABILITY(layernorm_shift_partition_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("transpose2", 0)
            .EQ("reshape2", 0));
