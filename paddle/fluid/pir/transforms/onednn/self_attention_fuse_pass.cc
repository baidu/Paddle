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

#include "paddle/fluid/pir/transforms/onednn/self_attention_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class SelfAttentionFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string self_attn_name_;
  uint32_t benefit_;

 public:
  SelfAttentionFusePattern(std::string self_attn_name, uint32_t benefit)
      : self_attn_name_(std::move(self_attn_name)), benefit_(benefit) {}

  std::string name() const override { return "SelfAttentionFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &transpose_0 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_0")}});
    pat.Tensor("transpose_0_out") = transpose_0(pat.Tensor("input"));

    const auto &full_int_array_0 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_0")}});
    const auto &full_int_array_1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_1")}});
    const auto &full_int_array_2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_2")}});
    const auto &full_int_array_3 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_3")}});
    const auto &full_int_array_4 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_4")}});
    const auto &full_int_array_5 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_5")}});

    const auto &slice_0 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_0")},
                {"infer_flags", pat.Attr("infer_flags_0")},
                {"decrease_axis", pat.Attr("decrease_axis_0")}});
    pat.Tensor("slice_0_out") = slice_0(
        pat.Tensor("transpose_0_out"), full_int_array_0(), full_int_array_1());

    const auto &slice_1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_1")},
                {"infer_flags", pat.Attr("infer_flags_1")},
                {"decrease_axis", pat.Attr("decrease_axis_1")}});
    pat.Tensor("slice_1_out") = slice_1(
        pat.Tensor("transpose_0_out"), full_int_array_2(), full_int_array_3());

    const auto &slice_2 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_2")},
                {"infer_flags", pat.Attr("infer_flags_2")},
                {"decrease_axis", pat.Attr("decrease_axis_2")}});
    pat.Tensor("slice_2_out") = slice_2(
        pat.Tensor("transpose_0_out"), full_int_array_4(), full_int_array_5());

    const auto &transpose_1 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_1")}});
    pat.Tensor("transpose_1_out") = transpose_1(pat.Tensor("slice_1_out"));

    const auto &matmul_1 = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x_1")},
                                   {"transpose_y", pat.Attr("transpose_y_1")}});
    pat.Tensor("matmul_1_out") =
        matmul_1(pat.Tensor("slice_0_out"), pat.Tensor("transpose_1_out"));

    const auto &softmax = pat.Op(paddle::dialect::SoftmaxOp::name(),
                                 {{"axis", pat.Attr("axis")}});
    pat.Tensor("softmax_out") = softmax(pat.Tensor("matmul_1_out"));

    const auto &matmul_0 = pat.Op(paddle::dialect::MatmulOp::name(),
                                  {{"transpose_x", pat.Attr("transpose_x_0")},
                                   {"transpose_y", pat.Attr("transpose_y_0")}});
    pat.Tensor("matmul_0_out") =
        matmul_0(pat.Tensor("softmax_out"), pat.Tensor("slice_2_out"));

    const auto &transpose_2 = pat.Op(paddle::dialect::TransposeOp::name(),
                                     {{"perm", pat.Attr("perm_2")}});
    pat.Tensor("transpose_2_out") = transpose_2(pat.Tensor("matmul_0_out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto input_shape = pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto transpose_0_outshape =
          pir::GetShapeFromValue(match_ctx.Tensor("transpose_0_out"));
      // input_shape should be [batch_size, seq_len, 3, num_heads, head_size]
      if (input_shape.size() != 5 || input_shape[2] != 3 ||
          transpose_0_outshape.size() != 5 || transpose_0_outshape[0] != 3 ||
          transpose_0_outshape[2] != input_shape[3]) {
        return false;
      }
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      bool result_x = match_ctx.Attr<bool>("transpose_x_0");
      bool result_y = match_ctx.Attr<bool>("transpose_y_0");
      if (result_x || result_y) {
        return false;
      }
      result_x = match_ctx.Attr<bool>("transpose_x_1");
      result_y = match_ctx.Attr<bool>("transpose_y_1");
      if (result_x || result_y) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &head_number_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          auto shape =
              pir::GetShapeFromValue(match_ctx.Tensor("transpose_0_out"));
          return shape[2];
        });

    const auto &self_dp_attention = res.Op(
        self_attn_name_,
        {{"alpha", res.Float32Attr(1.0f)}, {"head_number", head_number_attr}});

    self_dp_attention({&res.Tensor("input")}, {&res.Tensor("transpose_2_out")});
  }
};

class SelfAttentionFusePass : public pir::PatternRewritePass {
 public:
  SelfAttentionFusePass()
      : pir::PatternRewritePass("self_attention_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    ps.Add(paddle::drr::Create<SelfAttentionFusePattern>(
        context, paddle::dialect::SelfDpAttentionOp::name(), 1));

    return ps;
  }

  bool CanApplyOn(pir::Operation *op) const override {
#if !defined(PADDLE_WITH_AVX512F) || !defined(PADDLE_WITH_MKLML) || \
    !defined(PADDLE_WITH_DNNL)
    LOG(WARNING) << "No-avx512 or MKL or oneDNN supported!";
    return false;
#endif
    if (!phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512f)) {
      return false;
    }

    return op->num_regions() > 0;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateSelfAttentionFusePass() {
  // specific fusion for self_attention
  return std::make_unique<SelfAttentionFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(self_attention_fuse_pass, SelfAttentionFusePass);
