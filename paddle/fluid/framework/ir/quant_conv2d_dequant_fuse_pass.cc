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

#include "paddle/fluid/framework/ir/quant_conv2d_dequant_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {
QuantDequantFusePass::QuantDequantFusePass() {
  AddOpCompat(OpCompat("fake_quantize_range_abs_max"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("InScale")
      .IsTensor()
      .End()
      .AddInput("Iter")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("OutScale")
      .IsTensor()
      .End()
      .AddOutput("OutScales")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("window_size")
      .IsType<int>()
      .IsNumGT(0)
      .End()
      .AddAttr("bit_length")
      .IsIntIn({8, 16})
      .End()
      .AddAttr("round_type")
      .IsOptional()
      .IsIntIn({0, 1})
      .End();
  AddOpCompat(OpCompat("fake_quantize_moving_average_abs_max"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("InScale")
      .IsTensor()
      .End()
      .AddInput("InAccum")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("InState")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("OutScale")
      .IsTensor()
      .End()
      .AddOutput("OutState")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("OutAccum")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("moving_rate")
      .IsType<float>()
      .IsNumGT(0.0f)
      .End()
      .AddAttr("bit_length")
      .IsIntIn({8, 16})
      .End()
      .AddAttr("round_type")
      .IsOptional()
      .IsIntIn({0, 1})
      .End();
  AddOpCompat(OpCompat("fake_dequantize_max_abs"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("max_range")
      .IsType<float>()
      .IsNumGT(0.0f)
      .End();
  AddOpCompat(OpCompat("fake_channel_wise_dequantize_max_abs"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scales")  // "Scales" is a vector with at most two tensors
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("quant_bits")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("quant_axis")
      .IsIntIn({0, 1})
      .IsOptional()
      .End()
      .AddAttr("x_num_col_dims")
      .IsType<int>()
      .IsOptional()
      .End();
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
  AddOpCompat(OpCompat("depthwise_conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
  AddOpCompat(OpCompat("conv2d_transpose"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("output_padding")
      .IsType<std::vector<int>>()
      .IsOptional()
      .End()
      .AddAttr("output_size")
      .IsType<std::vector<int>>()
      .IsOptional()
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
}
// Delete quant op before quantized ops, and set input scale in the attr of
// quantized ops
void QuantDequantFusePass::DeleteQuant(ir::Graph* graph,
                                       Scope* scope,
                                       const std::string& quant_type) const {
  const std::string pattern_name = "delete_quant_fuse";
  GraphPatternDetector gpd;
  auto* input_act_node = gpd.mutable_pattern()
                             ->NewNode("input_act_node")
                             ->assert_is_op_input(quant_type, "X")
                             ->AsInput();

  // Create pattern
  patterns::DeleteQuantOpFuse pattern(gpd.mutable_pattern(), pattern_name);
  pattern(input_act_node, quant_type);

  // extract input scale from quant op input to set it in attr of all quantized
  // ops linked from it
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    PADDLE_ENFORCE_EQ(
        subgraph.count(input_act_node),
        true,
        common::errors::NotFound(
            "Input act node(%s) not found in QuantDequantFuse pass.",
            input_act_node->name()));
    Node* input_act = subgraph.at(input_act_node);
    Node* input_scale = subgraph.at(pattern.GetPDNode("input_scale_node"));
    Node* quant = subgraph.at(pattern.GetPDNode("quant_node"));
    Node* output_scale = subgraph.at(pattern.GetPDNode("output_scale_node"));
    Node* output_act = subgraph.at(pattern.GetPDNode("output_act_node"));
    int bit_length = PADDLE_GET_CONST(int, quant->Op()->GetAttr("bit_length"));

    // Get input scale from tensor
    std::string input_scale_var_name = quant->Op()->Input("InScale").front();
    PADDLE_ENFORCE_NOT_NULL(
        scope,
        common::errors::InvalidArgument(
            "Scope in QuantDequantFuse pass should not be null."));
    const phi::DenseTensor& input_scale_tensor =
        scope->FindVar(input_scale_var_name)->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::is_cpu_place(input_scale_tensor.place()),
                      true,
                      common::errors::InvalidArgument(
                          "Input scale tensor's place should be CPU."));
    const float* input_scale_data = input_scale_tensor.data<float>();
    float in_scale = input_scale_data[0];
    float scale_value = in_scale;

    // Set input scale in attr, and relink nodes
    std::string input_act_name = input_act->Var()->Name();
    std::string output_act_name = output_act->Var()->Name();
    auto outlinks = output_act->outputs;
    for (auto* quantized_node : outlinks) {
      auto op_desc = quantized_node->Op();
      std::string quantized_op_type = op_desc->Type();
      if (quantized_op_type == "conv2d" ||
          quantized_op_type == "fused_conv2d_add_act" ||
          quantized_op_type == "depthwise_conv2d" ||
          quantized_op_type == "conv2d_transpose" ||
          quantized_op_type == "matrix_multiply") {
        op_desc->SetAttr("Input_scale", scale_value);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported quantized op type %s.", quantized_op_type));
      }
      op_desc->SetAttr("bit_length", bit_length);
      op_desc->RenameInput(output_act_name, input_act_name);
      op_desc->Flush();
      IR_NODE_LINK_TO(input_act, quantized_node);
    }
    // Delete nodes and edges
    std::unordered_set<const Node*> nodes2rm = {
        input_scale, quant, output_scale, output_act};
    GraphSafeRemoveNodes(graph, nodes2rm);
  };
  gpd(graph, handler);
}

// Delete dequant op after quantized ops, and convert weight from fp32 range to
// int8 range
void QuantDequantFusePass::FuseDequant(ir::Graph* graph,
                                       Scope* scope,
                                       const std::string& quantized_op_type,
                                       const std::string& dequant_type) const {
  std::string weight_name = "";
  std::string input_name = "";
  if (quantized_op_type == "conv2d" ||
      quantized_op_type == "depthwise_conv2d" ||
      quantized_op_type == "fused_conv2d" ||
      quantized_op_type == "fused_conv2d_add_act" ||
      quantized_op_type == "conv2d_transpose") {
    weight_name = "Filter";
    input_name = "Input";
  } else if (quantized_op_type == "matrix_multiply") {
    weight_name = "Y";
    input_name = "X";
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "QuantDequantFuse: We only support conv2d, fused_conv2d_add_act, "
        "fused_conv2d,"
        "conv2d_transpose, matrix_multiply(mul/matmul/matmul_v2) for now, but "
        "received: "
        "%s.",
        quantized_op_type));
  }
  const std::string pattern_name = "dequant_fuse";
  GraphPatternDetector gpd;

  auto* quantized_op_input =
      gpd.mutable_pattern()
          ->NewNode("quantized_op_input")
          ->assert_is_op_input(quantized_op_type, input_name)
          ->AsInput();

  // Create pattern
  patterns::DequantOpFuse pattern(gpd.mutable_pattern(), pattern_name);
  pattern(quantized_op_input, quantized_op_type, dequant_type, weight_name);
  // Record whether quantized_op_weight_node has been dealt with
  std::unordered_set<Node*> quantized_op_weight_node_set;
  // Create new op desc
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    PADDLE_ENFORCE_EQ(
        subgraph.count(quantized_op_input),
        true,
        common::errors::NotFound("Quantized op input node(%s) did not find "
                                 "in QuantDequantFuse pass.",
                                 quantized_op_input->name()));
    Node* quantized_op_input_node = subgraph.at(quantized_op_input);
    Node* quantized_op_weight_node =
        subgraph.at(pattern.GetPDNode("quantized_op_weight"));
    Node* quantized_op_node = subgraph.at(pattern.GetPDNode("quantized_op"));
    Node* dequant_op_node = subgraph.at(pattern.GetPDNode("dequant_op"));
    Node* dequant_op_out_node =
        subgraph.at(pattern.GetPDNode("dequant_op_out"));

    std::unordered_set<const Node*> nodes2rm = {};
    int bit_length =
        PADDLE_GET_CONST(int, quantized_op_node->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);
    std::vector<float> weight_scale;
    int quant_axis = 0;
    if (dequant_op_node->Op()->HasAttr("quant_axis")) {
      quant_axis =
          PADDLE_GET_CONST(int, dequant_op_node->Op()->GetAttr("quant_axis"));
    }
    // Get weight scale
    if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
      Node* dequant_channel_scale_node =
          subgraph.at(pattern.GetPDNode("dequant_channel_scale"));
      auto scales_name = dequant_op_node->Op()->Input("Scales");
      PADDLE_ENFORCE_EQ(
          scales_name.size(),
          2,
          common::errors::InvalidArgument(
              "Scales size in channel-wise dequantize op should be 2, got %d.",
              scales_name.size()));
      const phi::DenseTensor& channel_scale_tensor =
          scope->FindVar(scales_name[0])->Get<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(phi::is_cpu_place(channel_scale_tensor.place()),
                        true,
                        common::errors::InvalidArgument(
                            "Channel scale tensor's place should be CPU."));
      const float* channel_scale_data = channel_scale_tensor.data<float>();
      for (int i = 0; i < channel_scale_tensor.numel(); i++) {
        weight_scale.push_back(channel_scale_data[i] /
                               static_cast<float>(range));
      }
      nodes2rm.insert(dequant_channel_scale_node);
    } else {
      float max_range =
          PADDLE_GET_CONST(float, dequant_op_node->Op()->GetAttr("max_range"));
      weight_scale.push_back(static_cast<float>(range * range) / max_range /
                             static_cast<float>(range));
    }

    // Convert weight to fp32 range
    auto* weight_tensor = scope->Var(quantized_op_weight_node->Name())
                              ->GetMutable<phi::DenseTensor>();
    const auto& w_dims = weight_tensor->dims();
    float* quantized_weight_data =
        weight_tensor->mutable_data<float>(phi::CPUPlace());
    //  Determine whether this weight tensor has been re-written, avoiding
    //  re-write it again when this weight tensor is shared among many ops.
    if (!quantized_op_weight_node_set.count(quantized_op_weight_node)) {
      quantized_op_weight_node_set.insert(quantized_op_weight_node);
      // If quantized op is matrix_multiply, weight scale size = 1;
      // If quantized op is conv2d, weight scale size = weight dims[0]
      // If quantized op is conv2d_transpose, weight scale size = weight dims[1]
      if (quantized_op_type == "matrix_multiply") {
        if (dequant_type == "fake_dequantize_max_abs") {
          PADDLE_ENFORCE_EQ(weight_scale.size(),
                            1,
                            common::errors::InvalidArgument(
                                "matrix_multiply(mul/matmul/matmul_v2) op "
                                "weight dequantized by "
                                "[fake_dequantize_max_abs] "
                                "requires weight scale size = 1, but got %d.",
                                weight_scale.size()));
          for (int j = 0; j < weight_tensor->numel(); j++) {
            quantized_weight_data[j] *= weight_scale[0];
          }
        }
        if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
          if (quant_axis == 0) {
          } else {
            PADDLE_ENFORCE_EQ(
                quant_axis == 1,
                true,
                common::errors::InvalidArgument(
                    "'quant_axis' of matrix_multiply(mul/matmul/matmul_v2) op "
                    "weight "
                    "dequantized by "
                    "[fake_channel_wise_dequantize_max_abs]should be 1, but "
                    "the received is %d",
                    quant_axis));
          }
          PADDLE_ENFORCE_EQ(
              weight_scale.size(),
              static_cast<size_t>(w_dims[1]),
              common::errors::InvalidArgument(
                  "matrix_multiply(mul/matmul/matmul_v2) op weight dequantized "
                  "by "
                  "[fake_channel_wise_dequantize_max_abs] "
                  "requires weight scale "
                  "size = 2nd dim of matrix_multiply(mul/matmul/matmul_v2)'s "
                  "weight, which is %d, "
                  "but got "
                  "%d.",
                  static_cast<size_t>(w_dims[1]),
                  weight_scale.size()));
          for (int j = 0; j < weight_tensor->numel(); j++) {
            quantized_weight_data[j] *= weight_scale[j % w_dims[1]];
          }
        }
      } else if (quantized_op_type == "conv2d" ||
                 quantized_op_type == "fusd_conv2d" ||
                 quantized_op_type == "depthwise_conv2d") {
        PADDLE_ENFORCE_EQ(
            dequant_type,
            "fake_channel_wise_dequantize_max_abs",
            common::errors::InvalidArgument(
                "conv2d op must be dequantized by "
                "[fake_channel_wise_dequantize_max_abs], but got %s. "
                "If you uses PaddleSlim to generate the quantized "
                "model, please set the 'weight_quantize_type' params as "
                "'channel_wise_abs_max' and generate the quantized model "
                "again.",
                dequant_type));
        if (quant_axis == 0) {
        } else {
          PADDLE_ENFORCE_EQ(
              quant_axis == 0,
              true,
              common::errors::InvalidArgument(
                  "'quant_axis' of conv2d/depthwise_conv2d op weight "
                  "dequantized "
                  "by [fake_channel_wise_dequantize_max_abs]should be 0, but "
                  "the received is %d",
                  quant_axis));
        }
        PADDLE_ENFORCE_EQ(
            weight_scale.size(),
            static_cast<size_t>(w_dims[0]),
            common::errors::InvalidArgument(
                "conv2d op requires weight scale size = channel size of the "
                "weight, which is %d, but got %d.",
                static_cast<size_t>(w_dims[0]),
                weight_scale.size()));
        for (int j = 0; j < weight_tensor->numel(); j++) {
          int inner_size = static_cast<int>(w_dims[1] * w_dims[2] * w_dims[3]);
          quantized_weight_data[j] *= weight_scale[j / inner_size];
        }
      } else if (quantized_op_type == "conv2d_transpose") {
        PADDLE_ENFORCE_EQ(
            dequant_type,
            "fake_channel_wise_dequantize_max_abs",
            common::errors::InvalidArgument(
                "conv2d_transpose must be dequantized by "
                "[fake_channel_wise_dequantize_max_abs], but got %s",
                dequant_type));
        if (quant_axis == 0) {
        } else {
          PADDLE_ENFORCE_EQ(
              quant_axis == 1,
              true,
              common::errors::InvalidArgument(
                  "'quant_axis' of conv2d_transpose op weight dequantized by "
                  "[fake_channel_wise_dequantize_max_abs]should be 1, but "
                  "the received is %d",
                  quant_axis));
        }
        PADDLE_ENFORCE_EQ(
            weight_scale.size(),
            static_cast<size_t>(w_dims[1]),
            common::errors::InvalidArgument(
                "conv2d_transpose op requires weight scale size = channel size "
                "of the weight, which is %d, but got %d.",
                static_cast<size_t>(w_dims[1]),
                weight_scale.size()));
        for (int j = 0; j < weight_tensor->numel(); j++) {
          int inner_size = static_cast<int>(w_dims[2] * w_dims[3]);
          quantized_weight_data[j] *=
              weight_scale[(j / inner_size) % w_dims[1]];
        }
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Unsupported quantized op type: %s", quantized_op_type));
      }
    }
    // create new op_desc
    auto base_op_desc = *quantized_op_node->Op()->Proto();
    std::string new_input = quantized_op_input_node->Name();
    std::string new_output = dequant_op_out_node->Name();

    framework::OpDesc new_op_desc(base_op_desc,
                                  quantized_op_node->Op()->Block());
    new_op_desc.SetType(quantized_op_type);
    new_op_desc.SetAttr("enable_int8", true);
    if (quantized_op_type == "conv2d" ||
        quantized_op_type == "fused_conv2d_add_act" ||
        quantized_op_type == "fused_conv2d" ||
        quantized_op_type == "depthwise_conv2d" ||
        quantized_op_type == "conv2d_transpose") {
      new_op_desc.SetInput("Input", {new_input});
      new_op_desc.SetOutput("Output", {new_output});
    } else if (quantized_op_type == "matrix_multiply") {
      new_op_desc.SetInput("X", {new_input});
      new_op_desc.SetOutput("Out", {new_output});
    }
    new_op_desc.Flush();
    auto* new_op = graph->CreateOpNode(&new_op_desc);
    IR_NODE_LINK_TO(quantized_op_input_node, new_op);
    IR_NODE_LINK_TO(quantized_op_weight_node, new_op);
    IR_NODE_LINK_TO(new_op, dequant_op_out_node);
    // Delete nodes and edges
    nodes2rm.insert(quantized_op_node);
    nodes2rm.insert(dequant_op_node);
    GraphSafeRemoveNodes(graph, nodes2rm);
  };
  gpd(graph, handler);
}

void QuantDequantFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "quant_dequant_fuse";
  FusePassBase::Init(pattern_name, graph);

  std::unordered_set<std::string> dequant_types = {
      "fake_channel_wise_dequantize_max_abs", "fake_dequantize_max_abs"};
  std::unordered_set<std::string> quant_types = {
      "fake_quantize_range_abs_max", "fake_quantize_moving_average_abs_max"};
  std::unordered_set<std::string> quantized_op_types = {
      "conv2d",
      "fused_conv2d",
      "matrix_multiply",
      "depthwise_conv2d",
      "conv2d_transpose",
  };
  auto* scope = param_scope();

  for (auto& quant_type : quant_types) {
    DeleteQuant(graph, scope, quant_type);
  }
  for (auto& dequant_type : dequant_types) {
    for (auto& quantized_op_type : quantized_op_types) {
      FuseDequant(graph, scope, quantized_op_type, dequant_type);
    }
  }
}

}  // namespace paddle::framework::ir

REGISTER_PASS(quant_conv2d_dequant_fuse_pass,
              paddle::framework::ir::QuantDequantFusePass);

REGISTER_PASS_CAPABILITY(quant_conv2d_dequant_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("conv2d_transpose", 2)
            .EQ("fake_quantize_abs_max", 0)
            .EQ("fake_quantize_range_abs_max", 0)
            .EQ("fake_quantize_moving_average_abs_max", 0)
            .LE("fake_channel_wise_quantize_abs_max", 1)
            .EQ("fake_dequantize_max_abs", 0));
