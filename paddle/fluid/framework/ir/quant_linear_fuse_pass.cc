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

#include "paddle/fluid/framework/ir/quant_linear_fuse_pass.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

namespace {
template <typename T1, typename T2>
void ConvertTensorType(phi::DenseTensor* tensor) {
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  phi::DenseTensor tmp_tensor;
  tmp_tensor.set_type(phi::CppTypeToDataType<T2>::Type());
  tmp_tensor.Resize(tensor->dims());
  auto* tmp_data = dev_ctx->template HostAlloc<T2>(
      &tmp_tensor, sizeof(T2) * tmp_tensor.numel());
  auto* data = tensor->data<T1>();
  for (int i = 0; i < tensor->numel(); i++) {
    tmp_data[i] = static_cast<T2>(data[i]);
  }
  tensor->clear();
  paddle::framework::TensorCopySync(tmp_tensor, phi::CPUPlace(), tensor);
}
}  // namespace

namespace paddle::framework::ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                                 \
  GET_IR_NODE(quantize_linear_op_x);              \
  GET_IR_NODE(quantize_linear_op_scale);          \
  GET_IR_NODE(quantize_linear_op);                \
  GET_IR_NODE(quantize_linear_op_out);            \
  GET_IR_NODE(dequantize_linear_op);              \
  GET_IR_NODE(dequantize_linear_op_out);          \
  GET_IR_NODE(weight_dequantize_linear_op_x);     \
  GET_IR_NODE(weight_dequantize_linear_op_scale); \
  GET_IR_NODE(weight_dequantize_linear_op);       \
  GET_IR_NODE(weight_dequantize_linear_op_out);   \
  GET_IR_NODE(mul);                               \
  GET_IR_NODE(mul_out);                           \
  GET_IR_NODE(bias);                              \
  GET_IR_NODE(elementwise_add);                   \
  GET_IR_NODE(elementwise_add_out);

QuantLinearFusePass::QuantLinearFusePass() {
  AddOpCompat(OpCompat("quantize_linear"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("ZeroPoint")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("bit_length")
      .IsType<int>()
      .End()
      .AddAttr("quant_axis")
      .IsType<int>()
      .End()
      .AddAttr("round_type")
      .IsOptional()
      .IsType<int>()
      .End();
  AddOpCompat(OpCompat("dequantize_linear"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("ZeroPoint")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("bit_length")
      .IsType<int>()
      .End()
      .AddAttr("quant_axis")
      .IsType<int>()
      .End()
      .AddAttr("round_type")
      .IsOptional()
      .IsType<int>()
      .End();
  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumMatch<int>([](int axis) -> bool {
        if (axis == -1 || axis >= 1) {
          return true;
        }
        return false;
      })
      .End();
  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

// Delete the quant and dequant op and weight dequant op,
// then fuse the matmul_v2 and elementwise_add op to a quant_linear op,
// if have relu after elementwise_add, then fuse relu into quant_linear op.
void QuantLinearFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  FusePassBase::Init("quant_linear_fuse_pattern", graph);

  int found_count = 0;
  for (bool with_relu : {true, false}) {
    found_count += ApplyQuantLinearFusePattern(graph, with_relu);
  }
  AddStatis(found_count);

  if (!graph->Has("enable_int8")) graph->Set("enable_int8", new bool(true));
}

int QuantLinearFusePass::ApplyQuantLinearFusePattern(Graph* graph,
                                                     bool with_relu) const {
  GraphPatternDetector gpd;

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(scope,
                          common::errors::InvalidArgument(
                              "Scope in QuantLinearFusePass should not be "
                              "null."));

  patterns::QuantLinearFusePattern pattern(gpd.mutable_pattern(),
                                           "quant_linear_fuse_pattern");
  pattern(true /*with bias*/, with_relu);

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    // Get input scale from tensor
    const phi::DenseTensor& input_scale_tensor =
        scope->GetVar(quantize_linear_op_scale->Name())
            ->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::is_cpu_place(input_scale_tensor.place()),
                      true,
                      common::errors::InvalidArgument(
                          "Input scale tensor's place should be CPU."));

    float input_scale = NAN;
    if (input_scale_tensor.dtype() == phi::DataType::FLOAT32) {
      const float* input_scale_data = input_scale_tensor.data<float>();
      input_scale = input_scale_data[0];
    } else if (input_scale_tensor.dtype() == phi::DataType::FLOAT16) {
      const phi::dtype::float16* input_scale_data =
          input_scale_tensor.data<phi::dtype::float16>();
      input_scale = static_cast<float>(input_scale_data[0]);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported type. The type of 'Scale' in quantize_linear op is "
          "expected to be float32 or float16, but the current type is %d",
          input_scale_tensor.dtype()));
    }
    // Get in_num_col_dims
    int in_num_col_dims = quantize_linear_op_x->Var()->GetShape().size() - 1;

    // because quant_linear kernel need weight's type be int8
    // convert weight fp32 --> int8
    auto* weight_tensor = scope->FindVar(weight_dequantize_linear_op_x->Name())
                              ->GetMutable<phi::DenseTensor>();
    ConvertTensorType<float, int8_t>(weight_tensor);

    // Get scale_weights
    const phi::DenseTensor& weight_scale_tensor =
        scope->FindVar(weight_dequantize_linear_op_scale->Name())
            ->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::is_cpu_place(weight_scale_tensor.place()),
                      true,
                      common::errors::InvalidArgument(
                          "weight_scale tensor's place should be CPU."));
    const float* weight_scale_data = weight_scale_tensor.data<float>();

    std::vector<float> scale_weights(weight_tensor->dims()[1], 1.0f);

    for (int i = 0; i < weight_tensor->dims()[1]; ++i) {
      scale_weights[i] = 1.0f / weight_scale_data[i];
    }

    Node* relu = nullptr;
    Node* relu_out = nullptr;
    if (with_relu) {
      GET_IR_NODE_FROM_SUBGRAPH(tmp_relu, relu, pattern);
      GET_IR_NODE_FROM_SUBGRAPH(tmp_relu_out, relu_out, pattern);
      relu = tmp_relu;
      relu_out = tmp_relu_out;
    }

    // Create an quant_linear Node.
    OpDesc desc;
    desc.SetType("quant_linear");

    // Set inputs of quant_linear
    desc.SetInput("x", {quantize_linear_op_x->Name()});
    desc.SetInput("w", {weight_dequantize_linear_op_x->Name()});
    desc.SetInput("bias", {bias->Name()});

    // Set output of quant_linear
    std::string quant_linear_out_name =
        with_relu ? relu_out->Name() : elementwise_add_out->Name();
    desc.SetOutput("out", std::vector<std::string>({quant_linear_out_name}));

    // Set attributes of quant_linear
    desc.SetAttr("scale_in", input_scale);
    desc.SetAttr("scale_weights", scale_weights);
    desc.SetAttr("in_num_col_dims", in_num_col_dims);

    std::string activation_type = with_relu ? "relu" : "";
    desc.SetAttr("activation_type", activation_type);

    // link input to quant_linear
    desc.RenameInput(dequantize_linear_op_out->Var()->Name(),
                     quantize_linear_op_x->Var()->Name());
    desc.RenameInput(weight_dequantize_linear_op_out->Var()->Name(),
                     weight_dequantize_linear_op_x->Var()->Name());
    desc.Flush();

    auto quant_linear_node = g->CreateOpNode(&desc);
    std::unordered_set<const Node*> nodes2rm = {
        quantize_linear_op_scale,
        quantize_linear_op,
        quantize_linear_op_out,
        dequantize_linear_op,
        dequantize_linear_op_out,
        weight_dequantize_linear_op_scale,
        weight_dequantize_linear_op,
        weight_dequantize_linear_op_out,
        mul,
        mul_out,
        elementwise_add};

    if (with_relu) {
      nodes2rm.insert(relu);
      nodes2rm.insert(elementwise_add_out);
    }
    GraphSafeRemoveNodes(graph, nodes2rm);

    IR_NODE_LINK_TO(quantize_linear_op_x, quant_linear_node);
    IR_NODE_LINK_TO(weight_dequantize_linear_op_x, quant_linear_node);
    IR_NODE_LINK_TO(bias, quant_linear_node);

    if (with_relu) {
      IR_NODE_LINK_TO(quant_linear_node, relu_out);
    } else {
      IR_NODE_LINK_TO(quant_linear_node, elementwise_add_out);
    }

    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}

}  // namespace paddle::framework::ir

REGISTER_PASS(quant_linear_fuse_pass,
              paddle::framework::ir::QuantLinearFusePass);
REGISTER_PASS_CAPABILITY(quant_linear_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul_v2", 0)
            .LE("elementwise_add", 1)
            .EQ("relu", 0)
            .EQ("quantize_linear", 0)
            .EQ("dequantize_linear", 0)
            .EQ("quant_linear", 0));
