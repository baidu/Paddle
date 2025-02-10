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

#include "paddle/fluid/framework/ir/trt_qk_multihead_matmul_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle::framework::ir::patterns {

//       input_qk   input_v
//       |q     |k      v
//       |------|       |
//    matmul  matmul  matmul
//       |      |       |
//    reshape reshape reshape
//       |      |       |
//     trans   trans   trans
//       |(x)   |(x)    |
//        matmul        |
//          |           |
//        scale         |
//          |           |
//        softmax       |(y)
//          |------matmul
//                    |
//                  trans
//                    |
//                  reshape
//                    |
//                   output
//
// -> fused to
//
//   input_qk input_v
//           |
//     qk_multihead_matmul
//           |
//         output

PDNode* TrtQKMultiHeadMatmulPattern::operator()() {
  std::unordered_set<std::string> mul_ops{"mul", "matmul_v2"};
  std::unordered_set<std::string> matmul_ops{"matmul", "matmul_v2"};
  auto* input0 = pattern->NewNode(input0_repr());
  auto* input1 = pattern->NewNode(input1_repr());

  input0->assert_is_ops_input(mul_ops);
  input1->assert_is_ops_input(mul_ops);
  VLOG(5) << "Start match TrtQKMultiHeadMatmulPattern";

  // First path
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_ops(mul_ops);
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(mul_ops, "Y");
  auto* mul0_out_var = pattern->NewNode(mul0_out_repr())
                           ->assert_is_ops_output(mul_ops)
                           ->assert_is_op_input("elementwise_add", "X")
                           ->AsIntermediate();

  auto* elementwise0 =
      pattern->NewNode(elementwise0_repr())->assert_is_op("elementwise_add");
  auto* elementwise0_w = pattern->NewNode(elementwise0_w_repr())
                             ->AsInput()
                             ->assert_is_op_input("elementwise_add", "Y");
  auto* elementwise0_out = pattern->NewNode(elementwise0_out_repr())
                               ->assert_is_op_output("elementwise_add", "Out")
                               ->assert_is_op_input("reshape2", "X")
                               ->AsIntermediate();

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var = pattern->NewNode(reshape2_0_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->assert_is_op_input("transpose2")
                                 ->AsIntermediate();

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->assert_is_ops_input(matmul_ops, "X")
                                   ->AsIntermediate();

  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_ops(matmul_ops);
  auto* matmul_qk_out_var = pattern->NewNode(matmul_qk_out_repr())
                                ->assert_is_ops_output(matmul_ops)
                                ->assert_is_op_input("scale")
                                ->AsIntermediate();

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var = pattern->NewNode(scale_out_repr())
                            ->assert_is_op_output("scale")
                            ->assert_is_op_input("softmax")
                            ->AsIntermediate();

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var = pattern->NewNode(softmax_qk_out_repr())
                                 ->assert_is_op_output("softmax")
                                 ->assert_is_ops_input(matmul_ops)
                                 ->AsIntermediate();

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_ops(matmul_ops);
  auto* matmul_qkv_out_var = pattern->NewNode(matmul_qkv_out_repr())
                                 ->assert_is_ops_output(matmul_ops)
                                 ->assert_is_op_input("transpose2")
                                 ->AsIntermediate();

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2")
                                     ->assert_is_op_input("reshape2")
                                     ->AsIntermediate();

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var = pattern->NewNode(reshape2_qkv_out_repr())
                                   ->assert_is_op_output("reshape2")
                                   ->AsOutput();

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_ops(mul_ops);
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(mul_ops, "Y");
  auto* mul1_out_var = pattern->NewNode(mul1_out_repr())
                           ->assert_is_ops_output(mul_ops)
                           ->assert_is_op_input("elementwise_add", "X")
                           ->AsIntermediate();

  auto* elementwise1 =
      pattern->NewNode(elementwise1_repr())->assert_is_op("elementwise_add");
  auto* elementwise1_w = pattern->NewNode(elementwise1_w_repr())
                             ->AsInput()
                             ->assert_is_op_input("elementwise_add", "Y");
  auto* elementwise1_out = pattern->NewNode(elementwise1_out_repr())
                               ->assert_is_op_output("elementwise_add", "Out")
                               ->assert_is_op_input("reshape2", "X")
                               ->AsIntermediate();

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");

  auto* reshape2_1_out_var = pattern->NewNode(reshape2_1_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->assert_is_op_input("transpose2")
                                 ->AsIntermediate();

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->assert_is_ops_input(matmul_ops, "Y")
                                   ->AsIntermediate();  // link to matmul qk

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_ops(mul_ops);
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(mul_ops, "Y");
  auto* mul2_out_var = pattern->NewNode(mul2_out_repr())
                           ->assert_is_ops_output(mul_ops)
                           ->assert_is_op_input("elementwise_add", "X")
                           ->AsIntermediate();

  auto* elementwise2 =
      pattern->NewNode(elementwise2_repr())->assert_is_op("elementwise_add");
  auto* elementwise2_w = pattern->NewNode(elementwise2_w_repr())
                             ->AsInput()
                             ->assert_is_op_input("elementwise_add", "Y");
  auto* elementwise2_out = pattern->NewNode(elementwise2_out_repr())
                               ->assert_is_op_output("elementwise_add", "Out")
                               ->assert_is_op_input("reshape2", "X")
                               ->AsIntermediate();

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");

  auto* reshape2_2_out_var = pattern->NewNode(reshape2_2_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->assert_is_op_input("transpose2")
                                 ->AsIntermediate();

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->assert_is_ops_input(matmul_ops)
                                   ->AsIntermediate();  // link to matmul qkv

  // Q path
  mul0->LinksFrom({input0, mul0_w_var}).LinksTo({mul0_out_var});
  elementwise0->LinksFrom({mul0_out_var, elementwise0_w})
      .LinksTo({elementwise0_out});

  reshape2_0->LinksFrom({elementwise0_out}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  // K path
  mul1->LinksFrom({input0, mul1_w_var}).LinksTo({mul1_out_var});
  elementwise1->LinksFrom({mul1_out_var, elementwise1_w})
      .LinksTo({elementwise1_out});

  reshape2_1->LinksFrom({elementwise1_out}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({transpose2_0_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  scale->LinksFrom({matmul_qk_out_var}).LinksTo({scale_out_var});
  softmax_qk->LinksFrom({scale_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  mul2->LinksFrom({input1, mul2_w_var}).LinksTo({mul2_out_var});
  elementwise2->LinksFrom({mul2_out_var, elementwise2_w})
      .LinksTo({elementwise2_out});

  reshape2_2->LinksFrom({elementwise2_out}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return reshape2_qkv_out_var;
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int TrtQkMultiHeadMatmulFusePass::BuildQkFusion(Graph* graph,
                                                const std::string& name_scope,
                                                Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::TrtQKMultiHeadMatmulPattern multihead_pattern(pattern, name_scope);

  multihead_pattern();
  auto fuse_creator = [&](Node* input0,
                          Node* input1,
                          Node* mul0,
                          Node* mul1,
                          Node* mul2,
                          Node* mul0_out,
                          Node* mul1_out,
                          Node* mul2_out,
                          Node* mul0_w,
                          Node* mul1_w,
                          Node* mul2_w,
                          Node* elementwise0,
                          Node* elementwise0_w,
                          Node* elementwise1,
                          Node* elementwise1_w,
                          Node* elementwise2,
                          Node* elementwise2_w,
                          Node* reshape2,
                          Node* reshape2_qkv_out,
                          Node* scale,
                          Node* scale_out) {
    // get Device context
    auto* dev_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));

    auto scale_attr = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));

    // create multihead
    OpDesc multihead_op_desc(mul0->Op()->Block());
    auto reshape_desc = reshape2->Op();
    int head_number =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);
    multihead_op_desc.SetType("qk_multihead_matmul");
    multihead_op_desc.SetInput("Input_qk", {input0->Name()});
    multihead_op_desc.SetInput("Input_v", {input1->Name()});

    auto* wq_tensor =
        scope->FindVar(mul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wk_tensor =
        scope->FindVar(mul1_w->Name())->GetMutable<phi::DenseTensor>();
    auto* bq_tensor =
        scope->FindVar(elementwise0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* bk_tensor =
        scope->FindVar(elementwise1_w->Name())->GetMutable<phi::DenseTensor>();

    int hidden_out = wq_tensor->dims()[1];
    int head_size = hidden_out / head_number;
    if (abs(scale_attr - 1.0f / sqrt(static_cast<float>(head_size))) > 1e-5) {
      VLOG(3) << "scale of multi-head matmul do not fit the requirement of "
                 "qk attention plugin, Stop fusing.";
      return;
    }
    VLOG(3) << "trt qk attention get wq_tensor name = " << mul0_w->Name()
            << "trt qk attention get wk_tensor name = " << mul1_w->Name();

    auto* wq_data = wq_tensor->data<float>();
    auto* wk_data = wk_tensor->data<float>();
    auto* bq_data = bq_tensor->data<float>();
    auto* bk_data = bk_tensor->data<float>();

    // combined_w_dims = [in,2,out]
    auto combined_w_qk_dims =
        common::make_ddim({wq_tensor->dims()[0], 2, wq_tensor->dims()[1]});
    auto combined_bias_dims = common::make_ddim({2, bq_tensor->dims()[0]});

    VLOG(3) << "trt qk attention trt wq_dim in:" << wq_tensor->dims()[0]
            << "trt qk attention trt wk_dim out:" << wq_tensor->dims()[1];
    auto* combined_w_qk_desc = mul0_w->Var();
    combined_w_qk_desc->SetShape(
        {wq_tensor->dims()[0], 2, wq_tensor->dims()[1]});
    combined_w_qk_desc->SetPersistable(true);
    phi::DenseTensor tmp_combined_w_qk_tensor;
    tmp_combined_w_qk_tensor.Resize(combined_w_qk_dims);
    float* tmp_combined_w_qk_data =
        dev_ctx->template HostAlloc<float>(&tmp_combined_w_qk_tensor);

    std::vector<float*> w_vec = {wq_data, wk_data};
    int dims_h = combined_w_qk_dims[0], dims_w = combined_w_qk_dims[2];
    // dims_h=in_feature, dims_w=out_feature
    // Combine the two fc weights together.
    // weight [Hidden_in * 2 * N * H]
    for (int i = 0; i < dims_h; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < dims_w; k++) {
          int out_index = i * (2 * dims_w) + j * dims_w + k;
          int in_index = i * dims_w + k;
          tmp_combined_w_qk_data[out_index] = w_vec[j][in_index];
        }
      }
    }
    wq_tensor->clear();
    wq_tensor->Resize(combined_w_qk_dims);
    auto* new_combined_w_qk_data = dev_ctx->template HostAlloc<float>(
        wq_tensor, sizeof(float) * wq_tensor->numel());
    memcpy(new_combined_w_qk_data,
           tmp_combined_w_qk_data,
           sizeof(float) * wq_tensor->numel());

    scope->EraseVars({mul1_w->Name()});
    auto* combined_bias_desc = elementwise0_w->Var();
    combined_bias_desc->SetShape({2, bq_tensor->dims()[0]});
    combined_bias_desc->SetPersistable(true);

    phi::DenseTensor tmp_combined_bias_tensor;
    tmp_combined_bias_tensor.Resize(combined_bias_dims);
    float* tmp_combined_bias_data =
        dev_ctx->template HostAlloc<float>(&tmp_combined_bias_tensor);

    size_t bias_size = bq_tensor->numel();
    memcpy(tmp_combined_bias_data, bq_data, sizeof(float) * bias_size);
    memcpy(
        tmp_combined_bias_data + bias_size, bk_data, sizeof(float) * bias_size);

    bq_tensor->clear();
    bq_tensor->Resize(combined_bias_dims);
    auto* new_combined_bias_data = dev_ctx->template HostAlloc<float>(
        bq_tensor, sizeof(float) * bq_tensor->numel());

    memcpy(new_combined_bias_data,
           tmp_combined_bias_data,
           sizeof(float) * bq_tensor->numel());

    scope->EraseVars({elementwise1_w->Name()});

    multihead_op_desc.SetInput("W_qk", {mul0_w->Name()});
    multihead_op_desc.SetInput("W_v", {mul2_w->Name()});
    multihead_op_desc.SetInput("B_qk", {elementwise0_w->Name()});
    multihead_op_desc.SetInput("B_v", {elementwise2_w->Name()});
    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* multihead = graph->CreateOpNode(&multihead_op_desc);
    IR_NODE_LINK_TO(input0, multihead);
    IR_NODE_LINK_TO(input1, multihead);
    IR_NODE_LINK_TO(mul0_w, multihead);
    IR_NODE_LINK_TO(mul2_w, multihead);
    IR_NODE_LINK_TO(elementwise0_w, multihead);
    IR_NODE_LINK_TO(elementwise2_w, multihead);
    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };
  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input1, input1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise0, elementwise0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise0_w, elementwise0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise0_out, elementwise0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise1, elementwise1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise1_w, elementwise1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise1_out, elementwise1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise2, elementwise2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise2_w, elementwise2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise2_out, elementwise2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, multihead_pattern);

    fuse_creator(input0,
                 input1,
                 mul0,
                 mul1,
                 mul2,
                 mul0_out,
                 mul1_out,
                 mul2_out,
                 mul0_w,
                 mul1_w,
                 mul2_w,
                 elementwise0,
                 elementwise0_w,
                 elementwise1,
                 elementwise1_w,
                 elementwise2,
                 elementwise2_w,
                 reshape2_0,
                 reshape2_qkv_out,
                 scale,
                 scale_out);

    std::unordered_set<const Node*> marked_nodes({reshape2_0,
                                                  reshape2_1,
                                                  reshape2_2,
                                                  reshape2_0_out,
                                                  reshape2_1_out,
                                                  reshape2_2_out,
                                                  transpose2_0,
                                                  transpose2_1,
                                                  transpose2_2,
                                                  transpose2_0_out,
                                                  transpose2_1_out,
                                                  transpose2_2_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  mul0,
                                                  mul1,
                                                  mul2,
                                                  mul0_out,
                                                  mul1_out,
                                                  mul2_out,
                                                  elementwise0,
                                                  elementwise0_out,
                                                  elementwise1,
                                                  elementwise1_w,
                                                  elementwise1_out,
                                                  elementwise2,
                                                  elementwise2_out,
                                                  reshape2_qkv,
                                                  scale});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void TrtQkMultiHeadMatmulFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
#ifdef PADDLE_WITH_TENSORRT
  auto trt_version = paddle::inference::tensorrt::GetTrtRuntimeVersion();
  if (std::get<0>(trt_version) * 1000 + std::get<1>(trt_version) * 100 +
          std::get<2>(trt_version) * 10 <
      8520) {
    VLOG(3) << "Qk attention oss plugin only available for trt version >= "
               "8.5.2.2. Stop this pass";
    return;
  }
#else
  // if no tensorrt, early stop
  return;
#endif
  bool with_dynamic_shape = Get<bool>("with_dynamic_shape");
  if (!with_dynamic_shape) {
    VLOG(3) << "Qk attention oss plugin need trt "
               "with_dynamic_shape. Stop this pass";
    return;
  }
  auto* scope = param_scope();
  int fusion_count = BuildQkFusion(graph, name_scope_, scope);
  AddStatis(fusion_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(trt_qk_multihead_matmul_fuse_pass,
              paddle::framework::ir::TrtQkMultiHeadMatmulFusePass);
REGISTER_PASS_CAPABILITY(trt_qk_multihead_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0)
            .EQ("matmul_v2", 0));
