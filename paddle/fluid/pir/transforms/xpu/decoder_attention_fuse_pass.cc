// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/xpu/decoder_attention_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

/*
This pass is used to fuse the QKV attention subgraph into one op in decoder
module of visual models .

For example:
Origin subgraph:

                  v             q             k
                  |             |             |
                  |             |             |
            full_int_array full_int_array  full_int_array
                  |             |             |
                  |             |             |
               reshape       reshape       reshape
                  |             |             |
                  |             |             |
                  |             |             |
              transpose     transpose   transpose
                  |             |           |
                  |              \         /
                  |                \     /
                  |              qk_matmul
                  |                   |
                  |                   |
                  |                 full
                  |                   |
                  |                   |
                  |                 scale
                  |                   |
                  |                   |
                   \              qk_softmax
                    \                 |
                     \                |
                       \             /
                         \          /
                           qkv_matmul
                              |
                              |
                              |
                          transpose
                              |
                              |
                              |
                        full_int_array
                              |
                              |
                              |
                           reshape
                              |
                              |
                              |
                            output

-------------------------------------------------------
Fused subgraph:
                 q          k          v
                  \         |         /
                    \       |       /
                      \     |     /
                   qkv_attention_xpu
                           |
                           |
                           |
                         output

*/

namespace {
class DecoderAttentionFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "DecoderAttentionFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &full_int_array_1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"dtype", pat.Attr("full_dtype1")}});
    pat.Tensor("shape_1") = full_int_array_1();
    const auto &reshape_1 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape_1({&pat.Tensor("input_q"), &pat.Tensor("shape_1")},
              {&pat.Tensor("reshape2_1_out"), &pat.Tensor("x_shape_1")});
    const auto &transpose_1 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_1")}});
    transpose_1({&pat.Tensor("reshape2_1_out")},
                {&pat.Tensor("transpose2_1_out")});

    const auto &full_int_array_2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"dtype", pat.Attr("full_dtype2")}});
    pat.Tensor("shape_2") = full_int_array_2();
    const auto &reshape_2 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape_2({&pat.Tensor("input_k"), &pat.Tensor("shape_2")},
              {&pat.Tensor("reshape2_2_out"), &pat.Tensor("x_shape_2")});
    const auto &transpose_2 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_2")}});
    transpose_2({&pat.Tensor("reshape2_2_out")},
                {&pat.Tensor("transpose2_2_out")});

    const auto &full_int_array_3 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"dtype", pat.Attr("full_dtype3")}});
    pat.Tensor("shape_3") = full_int_array_3();
    const auto &reshape_3 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape_3({&pat.Tensor("input_v"), &pat.Tensor("shape_3")},
              {&pat.Tensor("reshape2_3_out"), &pat.Tensor("x_shape_3")});
    const auto &transpose_3 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_3")}});
    transpose_3({&pat.Tensor("reshape2_3_out")},
                {&pat.Tensor("transpose2_3_out")});

    const auto &qk_matmul = pat.Op(paddle::dialect::MatmulOp::name());
    qk_matmul(
        {&pat.Tensor("transpose2_1_out"), &pat.Tensor("transpose2_2_out")},
        {&pat.Tensor("qk_matmul_out")});
    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    scale({&pat.Tensor("qk_matmul_out"), &full_op()},
          {&pat.Tensor("scale_out")});
    const auto &qk_softmax = pat.Op(paddle::dialect::SoftmaxOp::name(),
                                    {{"axis", pat.Attr("axis")}});
    qk_softmax({&pat.Tensor("scale_out")}, {&pat.Tensor("qk_softmax_out")});

    const auto &qkv_matmul = pat.Op(paddle::dialect::MatmulOp::name());
    qkv_matmul({&pat.Tensor("qk_softmax_out"), &pat.Tensor("transpose2_3_out")},
               {&pat.Tensor("qkv_matmul_out")});
    const auto &transpose_4 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_4")}});
    transpose_4({&pat.Tensor("qkv_matmul_out")},
                {&pat.Tensor("transpose2_4_out")});
    const auto &full_int_array_4 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"dtype", pat.Attr("full_dtype4")}});
    pat.Tensor("shape_4") = full_int_array_4();
    const auto &reshape_4 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape_4({&pat.Tensor("transpose2_4_out"), &pat.Tensor("shape_4")},
              {&pat.Tensor("qkv_output"), &pat.Tensor("x_shape_4")});

    // assert more
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &axis = match_ctx.Attr<std::vector<int>>("perm_1");
      size_t axis_rank = axis.size();
      if (axis_rank == 4 && axis[0] == 0 && axis[1] == 2 && axis[2] == 1 &&
          axis[3] == 3) {
        return true;
      } else {
        return false;
      }
    });
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &axis = match_ctx.Attr<std::vector<int>>("perm_2");
      size_t axis_rank = axis.size();
      if (axis_rank == 4 && axis[0] == 0 && axis[1] == 2 && axis[2] == 1 &&
          axis[3] == 3) {
        return true;
      } else {
        return false;
      }
    });
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &axis = match_ctx.Attr<std::vector<int>>("perm_3");
      size_t axis_rank = axis.size();
      if (axis_rank == 4 && axis[0] == 0 && axis[1] == 2 && axis[2] == 1 &&
          axis[3] == 3) {
        return true;
      } else {
        return false;
      }
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    // head_num
    const auto &head_num =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto shape =
              pir::GetShapeFromValue(match_ctx.Tensor("transpose2_1_out"));
          return shape[1];
        });
    const auto &head_dim =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto shape =
              pir::GetShapeFromValue(match_ctx.Tensor("transpose2_1_out"));
          return shape[3];
        });

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("input_q"));
          if (x_dtype.isa<pir::UInt8Type>()) {
            return phi::DataType::UINT8;
          } else if (x_dtype.isa<pir::Int8Type>()) {
            return phi::DataType::INT8;
          } else if (x_dtype.isa<pir::Int16Type>()) {
            return phi::DataType::INT16;
          } else if (x_dtype.isa<pir::Int32Type>()) {
            return phi::DataType::INT32;
          } else if (x_dtype.isa<pir::Int64Type>()) {
            return phi::DataType::INT64;
          } else if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    const auto &qkv_attention_xpu =
        res.Op(paddle::dialect::QkvAttentionXpuOp::name(),
               {{{"alpha", pat.Attr("value")},
                 {"head_num", head_num},
                 {"head_dim", head_dim},
                 {"qkv_fc_fusion", res.BoolAttr(false)},
                 {"out_dtype", out_dtype_attr}}});
    qkv_attention_xpu({&res.Tensor("input_q"),
                       &res.Tensor("input_k"),
                       &res.Tensor("input_v"),
                       &res.InputNoneTensor(),
                       &res.InputNoneTensor(),
                       &res.InputNoneTensor(),
                       &res.InputNoneTensor(),
                       &res.InputNoneTensor()},
                      {&res.Tensor("qkv_output")});
  }
};

class DecoderAttentionXpuFusePass : public pir::PatternRewritePass {
 public:
  DecoderAttentionXpuFusePass()
      : pir::PatternRewritePass("decoder_attention_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DecoderAttentionFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateDecoderAttentionXpuFusePass() {
  return std::make_unique<DecoderAttentionXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(decoder_attention_xpu_fuse_pass, DecoderAttentionXpuFusePass);
