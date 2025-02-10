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

#include "paddle/fluid/pir/transforms/onednn/scale_matmul_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class ScaleMatmulFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool as_x_;  // decide if the output of scale is for input_x of matmul

 public:
  ScaleMatmulFusePattern(std::string matmul_name,
                         std::string fused_matmul_name,
                         uint32_t benefit,
                         bool as_x)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit),
        as_x_(as_x) {}

  std::string name() const override { return "ScaleMatmulFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"value", pat.Attr("scale_")}});
    pat.Tensor("scale") = full();

    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    scale({&pat.Tensor("scale_in"), &pat.Tensor("scale")},
          {&pat.Tensor("scale_out")});

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    if (as_x_) {
      matmul({&pat.Tensor("scale_out"), &pat.Tensor("other")},
             {&pat.Tensor("Out")});
    } else {
      matmul({&pat.Tensor("other"), &pat.Tensor("scale_out")},
             {&pat.Tensor("Out")});
    }

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto scale = match_ctx.Attr<double>("scale_");
      auto bias = match_ctx.Attr<float>("bias");
      // conditions align with fluid pass
      if (bias != 0.0f) return false;
      if (scale <= 0.0f) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"fuse_activation", res.StrAttr("")},
        {"fuse_alpha", res.Float32Attr(0.0f)},
        {"fuse_beta", res.Float32Attr(0.0f)},
        {"fused_reshape_x", res.VectorInt32Attr({})},
        {"fused_transpose_x", res.VectorInt32Attr({})},
        {"fused_reshape_y", res.VectorInt32Attr({})},
        {"fused_transpose_y", res.VectorInt32Attr({})},
        {"fused_output_scale", res.Float32Attr(1.0f)},
        {"fused_reshape_out", res.VectorInt32Attr({})},
        {"fused_transpose_out", res.VectorInt32Attr({})},
        {"mkldnn_data_type", res.StrAttr("float32")},
        {"scale_x", res.Float32Attr(1.0f)},
        {"scale_y", res.Float32Attr(1.0f)},
        {"scale_in_eltwise", res.Float32Attr(0.0f)},
        {"scale_out", res.Float32Attr(1.0f)},
        {"force_fp32_output", res.BoolAttr(false)}};

    const auto &matmul_alpha_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          auto scale = match_ctx.Attr<double>("scale_");
          return scale;
        });

    fused_attrs.emplace("matmul_alpha", matmul_alpha_attr);

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    if (as_x_) {
      fused_matmul({&res.Tensor("scale_in"),
                    &res.Tensor("other"),
                    &res.InputNoneTensor()},
                   {&res.Tensor("Out")});
    } else {
      fused_matmul({&res.Tensor("other"),
                    &res.Tensor("scale_in"),
                    &res.InputNoneTensor()},
                   {&res.Tensor("Out")});
    }
  }
};

class ScaleFusedMatmulFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool as_x_;  // decide if the output of transpose is for input_x of matmul

 public:
  ScaleFusedMatmulFusePattern(std::string matmul_name,
                              std::string fused_matmul_name,
                              uint32_t benefit,
                              bool as_x)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit),
        as_x_(as_x) {}

  std::string name() const override { return "ScaleFusedMatmulFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"value", pat.Attr("scale_")}});
    pat.Tensor("scale") = full();

    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    scale({&pat.Tensor("scale_in"), &pat.Tensor("scale")},
          {&pat.Tensor("scale_out")});

    const auto &matmul =
        pat.Op(matmul_name_,
               {{"trans_x", pat.Attr("transpose_x")},
                {"trans_y", pat.Attr("transpose_y")},
                {"matmul_alpha", pat.Attr("matmul_alpha")},
                {"fuse_activation", pat.Attr("fuse_activation")},
                {"fuse_alpha", pat.Attr("fuse_alpha")},
                {"fuse_beta", pat.Attr("fuse_beta")},
                {"fused_output_scale", pat.Attr("fused_output_scale")},
                {"fused_reshape_x", pat.Attr("fused_reshape_x")},
                {"fused_transpose_x", pat.Attr("fused_transpose_x")},
                {"fused_reshape_y", pat.Attr("fused_reshape_y")},
                {"fused_transpose_y", pat.Attr("fused_transpose_y")},
                {"fused_reshape_out", pat.Attr("fused_reshape_out")},
                {"fused_transpose_out", pat.Attr("fused_transpose_out")},
                {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
                {"scale_x", pat.Attr("scale_x")},
                {"scale_y", pat.Attr("scale_y")},
                {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
                {"scale_out", pat.Attr("scale_out")},
                {"force_fp32_output", pat.Attr("force_fp32_output")}});
    if (as_x_) {
      matmul({&pat.Tensor("scale_out"),
              &pat.Tensor("other"),
              &pat.Tensor("residual")},
             {&pat.Tensor("Out")});
    } else {
      matmul({&pat.Tensor("other"),
              &pat.Tensor("scale_out"),
              &pat.Tensor("residual")},
             {&pat.Tensor("Out")});
    }

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto matmul_alpha = match_ctx.Attr<float>("matmul_alpha");
      auto scale = match_ctx.Attr<double>("scale_");
      auto bias = match_ctx.Attr<float>("bias");
      // conditions align with fluid pass
      if (matmul_alpha == 0.0f) return false;
      if (bias != 0.0f) return false;
      if (scale <= 0.0f) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"fuse_activation", pat.Attr("fuse_activation")},
        {"fuse_alpha", pat.Attr("fuse_alpha")},
        {"fuse_beta", pat.Attr("fuse_beta")},
        {"fused_reshape_x", pat.Attr("fused_reshape_x")},
        {"fused_transpose_x", pat.Attr("fused_transpose_x")},
        {"fused_reshape_y", pat.Attr("fused_reshape_y")},
        {"fused_transpose_y", pat.Attr("fused_transpose_y")},
        {"fused_output_scale", pat.Attr("fused_output_scale")},
        {"fused_reshape_out", pat.Attr("fused_reshape_out")},
        {"fused_transpose_out", pat.Attr("fused_transpose_out")},
        {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
        {"scale_x", pat.Attr("scale_x")},
        {"scale_y", pat.Attr("scale_y")},
        {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
        {"scale_out", pat.Attr("scale_out")},
        {"force_fp32_output", pat.Attr("force_fp32_output")}};

    const auto &matmul_alpha_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          auto scale = match_ctx.Attr<double>("scale_");
          auto matmul_alpha = match_ctx.Attr<float>("matmul_alpha");
          return scale * matmul_alpha;
        });

    fused_attrs.emplace("matmul_alpha", matmul_alpha_attr);

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    if (as_x_) {
      fused_matmul({&res.Tensor("scale_in"),
                    &res.Tensor("other"),
                    &res.Tensor("residual")},
                   {&res.Tensor("Out")});
    } else {
      fused_matmul({&res.Tensor("other"),
                    &res.Tensor("scale_in"),
                    &res.Tensor("residual")},
                   {&res.Tensor("Out")});
    }
  }
};

class ScaleMatmulFusePass : public pir::PatternRewritePass {
 public:
  ScaleMatmulFusePass()
      : pir::PatternRewritePass("scale_matmul_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    std::vector<bool> bool_set = {false, true};
    int benefit_idx = 5;
    for (auto as_x : bool_set) {
      ps.Add(paddle::drr::Create<ScaleMatmulFusePattern>(
          context,
          paddle::dialect::MatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          as_x));
      benefit_idx--;
    }

    for (auto as_x : bool_set) {
      ps.Add(paddle::drr::Create<ScaleFusedMatmulFusePattern>(
          context,
          paddle::onednn::dialect::FusedMatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          as_x));
      benefit_idx--;
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateScaleMatmulFusePass() {
  // pd_op.scale + pd_op.matmul -> onednn_op.fused_matmul
  // pd_op.scale + onednn_op.fused_matmul -> onednn_op.fused_matmul
  return std::make_unique<ScaleMatmulFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(scale_matmul_fuse_pass, ScaleMatmulFusePass);
