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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *conv2d_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto dilations_ =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto dilations = std::vector<int64_t>{dilations_.begin(), dilations_.end()};
  auto group_ = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto pads_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  if (pads_.size() == 2) {
    pads_.push_back(pads_[0]);
    pads_.push_back(pads_[1]);
  }
  auto pads = std::vector<int64_t>{pads_.begin(), pads_.end()};
  auto stride_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto stride = std::vector<int64_t>{stride_.begin(), stride_.end()};
  if (!op->Input("Bias").empty()) {
    return CreateConv(graph,
                      node,
                      {
                          GetInputVarNode("Input", node),
                          GetInputVarNode("Filter", node),
                          GetInputVarNode("Bias", node),
                      },
                      node->outputs,
                      dilations,
                      group_,
                      {},
                      pads,
                      stride);
  } else {
    return CreateConv(graph,
                      node,
                      {
                          GetInputVarNode("Input", node),
                          GetInputVarNode("Filter", node),
                      },
                      node->outputs,
                      dilations,
                      group_,
                      {},
                      pads,
                      stride);
  }
}

Node *batch_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  std::vector<Node *> inputs;
  inputs.push_back(GetInputVarNode("X", node));
  inputs.push_back(GetInputVarNode("Scale", node));
  inputs.push_back(GetInputVarNode("Bias", node));
  inputs.push_back(GetInputVarNode("Mean", node));
  inputs.push_back(GetInputVarNode("Variance", node));
  int64_t num_outputs = 1;
  std::vector<Node *> outputs;
  auto is_test_type = op->GetAttrType("is_test");
  bool is_test;
  if (is_test_type == 0) {
    // int
    is_test = PADDLE_GET_CONST(int, op->GetAttr("is_test"));
  } else {
    // bool
    is_test = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  }
  outputs.push_back(GetOutputVarNode("Y", node));
  if (!is_test) {
    outputs.push_back(GetOutputVarNode("MeanOut", node));
    outputs.push_back(GetOutputVarNode("VarianceOut", node));
    outputs.push_back(GetOutputVarNode("SavedMean", node));
    outputs.push_back(GetOutputVarNode("SavedVariance", node));
    num_outputs = 5;
  }
  // outputs.push_back(GetOutputVarNode("ReserveSpace", node));
  auto momentum = PADDLE_GET_CONST(float, op->GetAttr("momentum"));
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  // data_layout
  return CreateBaseOp(graph,
                      node,
                      "popart_batchnormalization",
                      inputs,
                      outputs,
                      {
                          {"momentum", momentum},
                          {"epsilon", epsilon},
                          {"num_outputs", num_outputs},
                      });
}

Node *pool2d_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto pooling_type =
      PADDLE_GET_CONST(std::string, op->GetAttr("pooling_type"));
  auto global_pooling = PADDLE_GET_CONST(bool, op->GetAttr("global_pooling"));
  if (op->HasAttr("adaptive")) {
    auto adaptive = PADDLE_GET_CONST(bool, op->GetAttr("adaptive"));
    if (adaptive) {
      auto ksize = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
      if (ksize[0] != 1 || ksize[1] != 1) {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Only support pool_size=1 with adaptive mode."));
      }
      // adaptive maxpool op is max_pool2d_with_index. Only process avgpool
      // here.
      return CreateBaseOp(
          graph, node, "popart_globalaveragepool", node->inputs, node->outputs);
    }
  }

  if (global_pooling) {
    if (pooling_type == "max") {
      return CreateBaseOp(
          graph, node, "popart_globalmaxpool", node->inputs, node->outputs);
    } else if (pooling_type == "avg") {
      return CreateBaseOp(
          graph, node, "popart_globalaveragepool", node->inputs, node->outputs);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "op pool2d with Unknown pooling_type: %s", pooling_type));
    }
  }
  if (op->HasAttr("padding_algorithm")) {
    auto padding_algorithm =
        PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
    if (padding_algorithm != "EXPLICIT") {
      PADDLE_THROW(common::errors::InvalidArgument(
          "op pool2d with Unknown padding_algorithm: %s", padding_algorithm));
    }
  }

  auto ksize = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
  auto kernel_shape = std::vector<int64_t>{ksize.begin(), ksize.end()};
  auto ceil_mode_ = PADDLE_GET_CONST(bool, op->GetAttr("ceil_mode"));
  auto ceil_mode = int64_t(ceil_mode_ ? 1 : 0);
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto pads = std::vector<int64_t>{paddings.begin(), paddings.end()};
  if (pads.size() == 2) {
    pads.push_back(paddings[0]);
    pads.push_back(paddings[1]);
  }
  auto strides_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto strides = std::vector<int64_t>{strides_.begin(), strides_.end()};
  if (pooling_type == "max") {
    int64_t num_outputs = 1;
    auto dilations = std::vector<int64_t>{};
    int64_t storage_order = 0;
    return CreateBaseOp(graph,
                        node,
                        "popart_maxpool",
                        node->inputs,
                        node->outputs,
                        {
                            {"num_outputs", num_outputs},
                            {"kernel_shape", kernel_shape},
                            {"ceil_mode", ceil_mode},
                            {"dilations", dilations},
                            {"pads", pads},
                            {"storage_order", storage_order},
                            {"strides", strides},
                        });
  } else if (pooling_type == "avg") {
    int64_t count_include_pad = 0;
    return CreateBaseOp(graph,
                        node,
                        "popart_averagepool",
                        node->inputs,
                        node->outputs,
                        {
                            {"kernel_shape", kernel_shape},
                            {"ceil_mode", ceil_mode},
                            {"count_include_pad", count_include_pad},
                            {"pads", pads},
                            {"strides", strides},
                        });
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "op pool2d with Unknown pooling_type: %s", pooling_type));
  }
}

Node *max_pool2d_with_index_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto ksize = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
  if (ksize[0] != 1 || ksize[1] != 1) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Only support pool_size=1 with adaptive mode."));
  }
  return CreateBaseOp(graph,
                      node,
                      "popart_globalmaxpool",
                      node->inputs,
                      {GetOutputVarNode("Out", node)});
}

Node *group_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto epsilon_ = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto groups_ = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto groups = int64_t{groups_};
  auto attrs_ = AttributeMap{{"epsilon", epsilon_}, {"num_groups", groups}};

  std::vector<Node *> inputs_ = {GetInputVarNode("X", node),
                                 GetInputVarNode("Scale", node),
                                 GetInputVarNode("Bias", node)};
  std::vector<Node *> outputs_ = {GetOutputVarNode("Y", node),
                                  GetOutputVarNode("Mean", node),
                                  GetOutputVarNode("Variance", node)};
  return CreateBaseOp(
      graph, node, "popart_groupnormalization_v2", inputs_, outputs_, attrs_);
}

Node *instance_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto epsilon_ = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto attrs_ = AttributeMap{{"epsilon", epsilon_}};

  std::vector<Node *> inputs_ = {GetInputVarNode("X", node),
                                 GetInputVarNode("Scale", node),
                                 GetInputVarNode("Bias", node)};
  std::vector<Node *> outputs_ = {GetOutputVarNode("Y", node)};
  return CreateBaseOp(
      graph, node, "popart_instancenormalization", inputs_, outputs_, attrs_);
}

Node *layer_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto begin_norm_axis_ = PADDLE_GET_CONST(int, op->GetAttr("begin_norm_axis"));
  auto input_shape_ = GetInputVarNode("X", node)->Var()->GetShape();
  auto epsilon_ = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  int64_t groups_ = 1;

  auto groupnorm_attrs_ =
      AttributeMap{{"epsilon", epsilon_}, {"num_groups", groups_}};

  if (input_shape_.size() == 2) {
    return CreateBaseOp(graph,
                        node,
                        "popart_groupnormalization_v2",
                        {GetInputVarNode("X", node),
                         GetInputVarNode("Scale", node),
                         GetInputVarNode("Bias", node)},
                        {GetOutputVarNode("Y", node),
                         GetOutputVarNode("Mean", node),
                         GetOutputVarNode("Variance", node)},
                        groupnorm_attrs_);
  }

  std::vector<int64_t> norm_shape_{1, 1};
  for (int i = 0; i < input_shape_.size(); i++) {
    if (i < begin_norm_axis_) {
      norm_shape_[0] *= input_shape_[i];
    } else {
      norm_shape_[1] *= input_shape_[i];
    }
  }

  auto attrs1 = AttributeMap{
      {"value", norm_shape_},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(norm_shape_.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto reshape1_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs1);
  auto new_node_reshape1 =
      CreateBaseOp(graph,
                   node,
                   "popart_reshape",
                   {GetInputVarNode("X", node), reshape1_const->outputs[0]},
                   {},
                   {});

  auto out_Y_ = MakeVarNode(graph, node);
  CreateBaseOp(graph,
               node,
               "popart_groupnormalization_v2",
               {new_node_reshape1->outputs[0],
                GetInputVarNode("Scale", node),
                GetInputVarNode("Bias", node)},
               {out_Y_,
                GetOutputVarNode("Mean", node),
                GetOutputVarNode("Variance", node)},
               groupnorm_attrs_);

  auto attrs2 = AttributeMap{
      {"value", input_shape_},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(input_shape_.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto reshape2_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs2);
  auto new_node_reshape2 = CreateBaseOp(graph,
                                        node,
                                        "popart_reshape",
                                        {out_Y_, reshape2_const->outputs[0]},
                                        {GetOutputVarNode("Y", node)},
                                        {});
  return new_node_reshape2;
}

Node *dropout_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto dropout_prob_ = PADDLE_GET_CONST(float, op->GetAttr("dropout_prob"));
  auto dropout_implementation_ =
      PADDLE_GET_CONST(std::string, op->GetAttr("dropout_implementation"));
  auto is_test_type_ = op->GetAttrType("is_test");
  bool is_test_;
  if (is_test_type_ == 0) {
    // int
    is_test_ = PADDLE_GET_CONST(int, op->GetAttr("is_test"));
  } else {
    // bool
    is_test_ = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  }

  if (is_test_) {
    if (dropout_implementation_ == "upscale_in_train") {
      return CreateBaseOp(graph,
                          node,
                          "popart_identity",
                          {GetInputVarNode("X", node)},
                          {GetOutputVarNode("Out", node)},
                          {});
    } else if (dropout_implementation_ == "downgrade_in_infer") {
      auto scale =
          CreateConst(graph,
                      node,
                      {},
                      {},
                      {{"value", std::vector<float>{1 - dropout_prob_}},
                       {"dims", std::vector<int64_t>{1}},
                       {"dtype", GetOutputVarDType(node)}});
      return CreateBaseOp(graph,
                          node,
                          "popart_mul",
                          {GetInputVarNode("X", node), scale->outputs[0]},
                          {GetOutputVarNode("Out", node)},
                          {});
    } else {
      PADDLE_THROW(
          common::errors::InvalidArgument("Invalid dropout_implementation"));
    }
  } else {
    if (dropout_implementation_ == "upscale_in_train") {
      auto attrs_ =
          AttributeMap{{"num_outputs", (int64_t)1}, {"ratio", dropout_prob_}};
      return CreateBaseOp(graph,
                          node,
                          "popart_dropout",
                          {GetInputVarNode("X", node)},
                          {GetOutputVarNode("Out", node)},
                          attrs_);
    } else if (dropout_implementation_ == "downgrade_in_infer") {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Do not support downgrade_in_infer with training"));
    } else {
      PADDLE_THROW(
          common::errors::InvalidArgument("Invalid dropout_implementation"));
    }
  }
}

Node *conv2d_transpose_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  if (data_format != "NCHW") {
    PADDLE_THROW(
        common::errors::InvalidArgument("Only support NCHW as data_format."));
  }

  auto *kernel_info = GetInputVarNode("Filter", node);
  auto kernel_shape = kernel_info->Var()->GetShape();

  auto dilations_ =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto dilations = std::vector<int64_t>{dilations_.begin(), dilations_.end()};
  auto strides_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto strides = std::vector<int64_t>{strides_.begin(), strides_.end()};
  auto output_padding_ =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("output_padding"));
  auto output_padding =
      std::vector<int64_t>{output_padding_.begin(), output_padding_.end()};
  auto group_ = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto group = int64_t(group_);

  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));

  auto paddings_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  if (paddings_.size() == 2) {
    paddings_.push_back(paddings_[0]);
    paddings_.push_back(paddings_[1]);
  } else if (paddings_.size() == 4) {
    std::swap(paddings_[1], paddings_[2]);
  }
  auto paddings = std::vector<int64_t>{paddings_.begin(), paddings_.end()};

  if (padding_algorithm == "SAME") {
    // Update paddings and dilations based on the sizes of H and W.
    auto input_shape = GetInputVarNode("Input", node)->Var()->GetShape();
    for (auto i = 0; i < 2; i++) {
      auto out_size = (input_shape[i + 2] + strides[i] - 1) / strides[i];
      auto pad_sum = std::max(
          (out_size - 1) * strides[i] + kernel_shape[i] - input_shape[i + 2],
          static_cast<int64_t>(0));
      auto pad_0 = pad_sum / 2;
      auto pad_1 = pad_sum - pad_0;
      paddings[i] = pad_0;
      paddings[i + 2] = pad_1;
    }
    for (auto i = 0; i < dilations.size(); i++) {
      dilations[i] = 1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto i = 0; i < paddings.size(); i++) {
      paddings[i] = 0;
    }
  }

  auto attrs = AttributeMap{{"dilations", dilations},
                            {"group", group},
                            {"kernel_shape", kernel_shape},
                            {"output_padding", output_padding},
                            {"pads", paddings},
                            {"strides", strides}};
  if (!op->Input("Bias").empty()) {
    return CreateBaseOp(graph,
                        node,
                        "popart_convtranspose",
                        {
                            GetInputVarNode("Input", node),
                            GetInputVarNode("Filter", node),
                            GetInputVarNode("Bias", node),
                        },
                        node->outputs,
                        attrs);
  } else {
    return CreateBaseOp(graph,
                        node,
                        "popart_convtranspose",
                        {
                            GetInputVarNode("Input", node),
                            GetInputVarNode("Filter", node),
                        },
                        node->outputs,
                        attrs);
  }
}

Node *affine_channel_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  if (data_layout != "NCHW") {
    PADDLE_THROW(
        common::errors::InvalidArgument("Only support NCHW as data_format."));
  }

  auto *scale = GetInputVarNode("Scale", node);
  auto *bias = GetInputVarNode("Bias", node);
  auto scale_shape = scale->Var()->GetShape();
  auto bias_shape = bias->Var()->GetShape();
  if (scale_shape.size() <= 1 || bias_shape.size() <= 1) {
    auto attrs = AttributeMap{{"value", std::vector<int64_t>{1, -1, 1, 1}},
                              {"dims", std::vector<int64_t>{4}},
                              {"dtype", ONNXDataType::INT64}};
    auto new_shape_const = CreateConst(graph, node, {}, {}, attrs);

    scale = CreateBaseOp(graph,
                         node,
                         "popart_reshape",
                         {scale, new_shape_const->outputs[0]},
                         {},
                         {})
                ->outputs[0];
    bias = CreateBaseOp(graph,
                        node,
                        "popart_reshape",
                        {bias, new_shape_const->outputs[0]},
                        {},
                        {})
               ->outputs[0];
  }
  auto *out = CreateBaseOp(
      graph, node, "popart_mul", {GetInputVarNode("X", node), scale}, {});
  return CreateBaseOp(graph,
                      node,
                      "popart_add",
                      {out->outputs[0], bias},
                      {GetOutputVarNode("Out", node)});
}

Node *interp_handler(Graph *graph, Node *node, const std::string &mode) {
  auto *op = node->Op();

  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  if (data_layout != "NCHW") {
    PADDLE_THROW(
        common::errors::InvalidArgument("Only support NCHW as data_format."));
  }

  auto align_corners = PADDLE_GET_CONST(bool, op->GetAttr("align_corners"));
  auto align_mode = PADDLE_GET_CONST(int, op->GetAttr("align_mode"));

  auto paddle_target_dtype = VarType::FP32;
  auto onnx_target_dtype = ONNXDataType::FLOAT;
  if (GetInputVarNode("X", node)->Var()->GetDataType() == VarType::FP16) {
    paddle_target_dtype = VarType::FP16;
    onnx_target_dtype = ONNXDataType::FLOAT16;
  }

  std::string coordinate_transformation_mode = "half_pixel";
  if (align_corners) {
    coordinate_transformation_mode = "align_corners";
  } else if (mode == "nearest") {
    coordinate_transformation_mode = "asymmetric";
  } else if (align_mode == 1 && mode == "cubic") {
    coordinate_transformation_mode = "asymmetric";
  }

  bool has_out_size = node->Op()->Input("OutSize").size() > 0;
  bool has_size_tensor = node->Op()->Input("SizeTensor").size() > 0;
  bool has_scale_tensor = node->Op()->Input("Scale").size() > 0;

  Node *size = nullptr;
  Node *scale = nullptr;
  // Input: Size and Scale
  if (has_out_size) {
    // Get 'size' from the tensor
    size = GetInputVarNode("OutSize", node);
    if (size->Var()->GetDataType() != VarType::INT64) {
      size = CreateCast(graph,
                        node,
                        {GetInputVarNode("OutSize", node)},
                        {},
                        VarType::INT64)
                 ->outputs[0];
    }
  } else if (has_size_tensor) {
    // Get 'size' from multi-tensors
    std::vector<Node *> size_nodes;
    for (auto var_name : node->Op()->Input("SizeTensor")) {
      Node *size_node = GetInputVarNodeByVarName(var_name, node);
      if (size_node->Var()->GetDataType() != VarType::INT64) {
        size_node = CreateCast(graph, node, {size_node}, {}, VarType::INT64)
                        ->outputs[0];
      }
      size_nodes.push_back(size_node);
    }
    size = CreateBaseOp(graph,
                        node,
                        "popart_concat",
                        size_nodes,
                        {},
                        {{"axis", int64_t(0)}})
               ->outputs[0];
  } else if (has_scale_tensor) {
    // Get 'scale' from tensor
    scale = GetInputVarNode("Scale", node);
    if (scale->Var()->GetDataType() != paddle_target_dtype) {
      scale =
          CreateCast(graph, node, {scale}, {}, paddle_target_dtype)->outputs[0];
    }
    auto *padding = CreateConst(graph,
                                node,
                                {},
                                {},
                                {{"value", std::vector<float>{1.0, 1.0}},
                                 {"dims", std::vector<int64_t>{2}},
                                 {"dtype", onnx_target_dtype}})
                        ->outputs[0];
    scale = CreateBaseOp(graph,
                         node,
                         "popart_concat",
                         {padding, scale},
                         {},
                         {{"axis", int64_t(0)}})
                ->outputs[0];
  } else {
    // Get 'size' or 'scale' from attribute
    auto out_d = PADDLE_GET_CONST(int, op->GetAttr("out_d"));
    auto out_h = PADDLE_GET_CONST(int, op->GetAttr("out_h"));
    auto out_w = PADDLE_GET_CONST(int, op->GetAttr("out_w"));
    if (out_d > 0 || out_w > 0 || out_h > 0) {
      std::vector<int64_t> out_size;
      if (GetInputVarNode("X", node)->Var()->GetShape().size() == 5) {
        out_size.push_back(int64_t(out_d));
        out_size.push_back(int64_t(out_h));
      } else if (GetInputVarNode("X", node)->Var()->GetShape().size() == 4) {
        out_size.push_back(int64_t(out_h));
      }
      out_size.push_back(int64_t(out_w));
      size =
          CreateConst(graph,
                      node,
                      {},
                      {},
                      {{"value", out_size},
                       {"dims", std::vector<int64_t>{int64_t(out_size.size())}},
                       {"dtype", ONNXDataType::INT64}})
              ->outputs[0];
    } else {
      auto scale_value =
          PADDLE_GET_CONST(std::vector<float>, op->GetAttr("scale"));
      float padding = 1.0;
      scale_value.insert(scale_value.begin(), padding);
      scale_value.insert(scale_value.begin(), padding);
      scale = CreateConst(
                  graph,
                  node,
                  {},
                  {},
                  {{"value", scale_value},
                   {"dims", std::vector<int64_t>{int64_t(scale_value.size())}},
                   {"dtype", onnx_target_dtype}})
                  ->outputs[0];
    }
  }

  Node *roi =
      CreateConst(
          graph,
          node,
          {},
          {},
          {{"value",
            std::vector<float>(
                GetInputVarNode("X", node)->Var()->GetShape().size() * 2, 1.0)},
           {"dims",
            std::vector<int64_t>{int64_t(
                GetInputVarNode("X", node)->Var()->GetShape().size() * 2)}},
           {"dtype", onnx_target_dtype}})
          ->outputs[0];

  if (size != nullptr) {
    Node *input_shape =
        CreateBaseOp(
            graph, node, "popart_shape", {GetInputVarNode("X", node)}, {})
            ->outputs[0];
    Node *nc = CreateSlice(graph,
                           node,
                           {input_shape},
                           {},
                           std::vector<int>{0},
                           std::vector<int>{2},
                           std::vector<int>{0},
                           std::vector<int>{1})
                   ->outputs[0];
    size = CreateBaseOp(graph,
                        node,
                        "popart_concat",
                        {nc, size},
                        {},
                        {{"axis", int64_t(0)}})
               ->outputs[0];
  }
  auto resize_attrs = AttributeMap{
      {"coordinate_transformation_mode", coordinate_transformation_mode},
      {"cubic_coeff_a", float{-0.75}},
      {"exclude_outside", int64_t{0}},
      {"extrapolation_value", float{0.0}},
      {"mode", mode},
      {"nearest_mode", std::string("round_prefer_floor")}};

  if (mode == "nearest" && coordinate_transformation_mode == "asymmetric") {
    resize_attrs.at("nearest_mode") = std::string("floor");
  }

  return CreateBaseOp(graph,
                      node,
                      "popart_resize",
                      {GetInputVarNode("X", node), roi, scale, size},
                      {GetOutputVarNode("Out", node)},
                      resize_attrs);
}

Node *bilinear_interp_v2_handler(Graph *graph, Node *node) {
  return interp_handler(graph, node, "linear");
}

Node *nearest_interp_v2_handler(Graph *graph, Node *node) {
  return interp_handler(graph, node, "nearest");
}

Node *bicubic_interp_v2_handler(Graph *graph, Node *node) {
  return interp_handler(graph, node, "cubic");
}

Node *linear_interp_v2_handler(Graph *graph, Node *node) {
  return interp_handler(graph, node, "linear");
}

Node *trilinear_interp_v2_handler(Graph *graph, Node *node) {
  return interp_handler(graph, node, "linear");
}

Node *data_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  int slot_dim = -1;
  if (op->HasAttr("slot_dim")) {
    slot_dim = PADDLE_GET_CONST(int, op->GetAttr("slot_dim"));
  }

  if (slot_dim > 0) {
    PADDLE_THROW(
        common::errors::InvalidArgument("slot_dim > 0 is not supported."));
  }

  bool enable_scale_and_shift = false;
  if (op->HasAttr("enable_scale_and_shift")) {
    enable_scale_and_shift =
        PADDLE_GET_CONST(bool, op->GetAttr("enable_scale_and_shift"));
  }

  auto *mean_arr = CreateBaseOp(graph,
                                node,
                                "popart_div",
                                {GetInputVarNode("BatchSum", node),
                                 GetInputVarNode("BatchSize", node)},
                                {})
                       ->outputs[0];
  auto *scale_arr = CreateBaseOp(graph,
                                 node,
                                 "popart_div",
                                 {GetInputVarNode("BatchSize", node),
                                  GetInputVarNode("BatchSquareSum", node)},
                                 {})
                        ->outputs[0];
  scale_arr =
      CreateBaseOp(graph, node, "popart_sqrt", {scale_arr}, {})->outputs[0];
  auto out =
      CreateBaseOp(
          graph, node, "popart_sub", {GetInputVarNode("X", node), mean_arr}, {})
          ->outputs[0];

  if (enable_scale_and_shift) {
    auto scale_res = CreateBaseOp(graph,
                                  node,
                                  "popart_mul",
                                  {out, GetInputVarNode("scale_w", node)},
                                  {})
                         ->outputs[0];
    return CreateBaseOp(graph,
                        node,
                        "popart_add",
                        {scale_res, GetInputVarNode("bias", node)},
                        {GetOutputVarNode("Y", node)});
  } else {
    return CreateBaseOp(graph,
                        node,
                        "popart_mul",
                        {out, scale_arr},
                        {GetOutputVarNode("Y", node)});
  }
}

Node *pad_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto mode = PADDLE_GET_CONST(std::string, op->GetAttr("mode"));
  auto value = PADDLE_GET_CONST(float, op->GetAttr("value"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));

  if (data_format == "NDHWC") {
    PADDLE_THROW(
        common::errors::Unimplemented("NDHWC format is not supported."));
  }
  if (mode == "replicate" || mode == "circular") {
    PADDLE_THROW(common::errors::Unimplemented(
        "circular and replicate modes are not supported."));
  }
  if (op->Input("Paddings").size()) {
    // Paddings -> input tensor
    // PopART Pad Op only support `pad` as a constant
    PADDLE_THROW(common::errors::Unimplemented(
        "Do not support Paddings as a inputs tensor"));
  }
  // Paddings -> Attr
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  std::vector<int64_t> new_paddings(10, 0);
  new_paddings[2] = paddings[4];
  new_paddings[3] = paddings[2];
  new_paddings[4] = paddings[0];
  new_paddings[7] = paddings[5];
  new_paddings[8] = paddings[3];
  new_paddings[9] = paddings[1];

  auto *paddings_node = CreateConst(graph,
                                    node,
                                    new_paddings,
                                    std::vector<int64_t>{10},
                                    ONNXDataType::INT64)
                            ->outputs[0];
  auto *value_node = CreateConst(graph,
                                 node,
                                 std::vector<float>{value},
                                 std::vector<int64_t>{1},
                                 ONNXDataType::FLOAT)
                         ->outputs[0];
  return CreateBaseOp(graph,
                      node,
                      "popart_pad",
                      {GetInputVarNode("X", node), paddings_node, value_node},
                      {GetOutputVarNode("Out", node)},
                      {{"mode", mode}});
}

Node *depthwise_conv2d_handler(Graph *graph, Node *node) {
  return conv2d_handler(graph, node);
}

Node *depthwise_conv2d_transpose_handler(Graph *graph, Node *node) {
  return conv2d_transpose_handler(graph, node);
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(affine_channel, affine_channel_handler);
REGISTER_HANDLER(pool2d, pool2d_handler);
REGISTER_HANDLER(max_pool2d_with_index, max_pool2d_with_index_handler);
REGISTER_HANDLER(batch_norm, batch_norm_handler);
REGISTER_HANDLER(group_norm, group_norm_handler);
REGISTER_HANDLER(instance_norm, instance_norm_handler);
REGISTER_HANDLER(layer_norm, layer_norm_handler);
REGISTER_HANDLER(conv2d, conv2d_handler);
REGISTER_HANDLER(conv2d_transpose, conv2d_transpose_handler);
REGISTER_HANDLER(dropout, dropout_handler);
REGISTER_HANDLER(bilinear_interp_v2, bilinear_interp_v2_handler);
REGISTER_HANDLER(nearest_interp_v2, nearest_interp_v2_handler);
REGISTER_HANDLER(bicubic_interp_v2, bicubic_interp_v2_handler);
REGISTER_HANDLER(linear_interp_v2, linear_interp_v2_handler);
REGISTER_HANDLER(trilinear_interp_v2, trilinear_interp_v2_handler);
REGISTER_HANDLER(data_norm, data_norm_handler);
REGISTER_HANDLER(pad3d, pad_handler);
REGISTER_HANDLER(depthwise_conv2d, depthwise_conv2d_handler);
REGISTER_HANDLER(depthwise_conv2d_transpose,
                 depthwise_conv2d_transpose_handler);
