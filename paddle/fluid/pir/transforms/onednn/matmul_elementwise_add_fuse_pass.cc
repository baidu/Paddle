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

#include "paddle/fluid/pir/transforms/onednn/matmul_elementwise_add_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class MatmulElementwiseAddFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool as_x_;  // Decide input direction of add

 public:
  MatmulElementwiseAddFusePattern(std::string matmul_name,
                                  std::string fused_matmul_name,
                                  uint32_t benefit,
                                  bool as_x)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit),
        as_x_(as_x) {}

  std::string name() const override {
    return "MatmulElementwiseAddFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

    pat.Tensor("add_out") =
        as_x_ ? add(pat.Tensor("Out"), pat.Tensor("residual"))
              : add(pat.Tensor("residual"), pat.Tensor("Out"));

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_matmul =
        res.Op(fused_matmul_name_,
               {{
                   {"trans_x", pat.Attr("transpose_x")},
                   {"trans_y", pat.Attr("transpose_y")},
                   {"matmul_alpha", res.Float32Attr(1.0f)},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"fused_output_scale", res.Float32Attr(1.0f)},
                   {"fused_reshape_x", res.VectorInt32Attr({})},
                   {"fused_transpose_x", res.VectorInt32Attr({})},
                   {"fused_reshape_y", res.VectorInt32Attr({})},
                   {"fused_transpose_y", res.VectorInt32Attr({})},
                   {"fused_reshape_out", res.VectorInt32Attr({})},
                   {"fused_transpose_out", res.VectorInt32Attr({})},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"scale_x", res.Float32Attr(1.0f)},
                   {"scale_y", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(0.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"force_fp32_output", res.BoolAttr(false)},
               }});

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("add_out")});
  }
};

class FusedMatmulElementwiseAddFusePattern
    : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool as_x_;  // Decide input direction of add

 public:
  FusedMatmulElementwiseAddFusePattern(std::string matmul_name,
                                       std::string fused_matmul_name,
                                       uint32_t benefit,
                                       bool as_x)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit),
        as_x_(as_x) {}

  std::string name() const override {
    return "FusedMatmulElementwiseAddFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

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

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    matmul({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("none")},
           {&pat.Tensor("Out")});

    pat.Tensor("add_out") =
        as_x_ ? add(pat.Tensor("Out"), pat.Tensor("residual"))
              : add(pat.Tensor("residual"), pat.Tensor("Out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto none_tensor = match_ctx.Tensor("none");
      if (none_tensor.impl() != nullptr) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_matmul =
        res.Op(fused_matmul_name_,
               {{
                   {"trans_x", pat.Attr("transpose_x")},
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
                   {"force_fp32_output", pat.Attr("force_fp32_output")},
               }});

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("add_out")});
  }
};

class MatmulElementwiseAddFusePass : public pir::PatternRewritePass {
 public:
  MatmulElementwiseAddFusePass()
      : pir::PatternRewritePass("matmul_elementwise_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    std::vector<bool> bool_set = {false, true};
    int benefit_idx = 1;
    for (auto as_x : bool_set) {
      ps.Add(paddle::drr::Create<MatmulElementwiseAddFusePattern>(
          context,
          paddle::dialect::MatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          as_x));
      benefit_idx++;
    }

    for (auto as_x : bool_set) {
      ps.Add(paddle::drr::Create<FusedMatmulElementwiseAddFusePattern>(
          context,
          paddle::onednn::dialect::FusedMatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          as_x));
      benefit_idx++;
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulElementwiseAddFusePass() {
  // pd_op.matmul + pd_op.add -> onednn_op.fused_matmul
  // onednn_op.fused_matmul + pd_op.add -> onednn_op.fused_matmul
  return std::make_unique<MatmulElementwiseAddFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_elementwise_add_fuse_pass,
                 MatmulElementwiseAddFusePass);
