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

#include "paddle/fluid/pir/transforms/onednn/matmul_transpose_reshape_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class MatmulTransposeReshapeFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;

 public:
  MatmulTransposeReshapeFusePattern(std::string matmul_name,
                                    std::string fused_matmul_name,
                                    uint32_t benefit)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit) {}

  std::string name() const override {
    return "MatmulTransposeReshapeFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

    const auto &transpose = pat.Op(paddle::dialect::TransposeOp::name(),
                                   {{"perm", pat.Attr("perm")}});
    pat.Tensor("transpose_out") = transpose(pat.Tensor("Out"));

    const auto &full_int_array = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                        {{"value", pat.Attr("int_array")}});
    pat.Tensor("shape") = full_int_array();

    const auto &reshape = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape({&pat.Tensor("transpose_out"), &pat.Tensor("shape")},
            {&pat.Tensor("reshape_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto shape = match_ctx.Attr<std::vector<int64_t>>("int_array");
      auto perm = match_ctx.Attr<std::vector<int>>("perm");
      const std::vector<int> supported_axis{0, 2, 1, 3};
      if (perm != supported_axis) return false;
      if (shape.size() != 3) return false;
      if (std::count(shape.begin(), shape.end(), -1) > 1) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
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
        {"mkldnn_data_type", res.StrAttr("float32")},
        {"scale_x", res.Float32Attr(1.0f)},
        {"scale_y", res.Float32Attr(1.0f)},
        {"scale_in_eltwise", res.Float32Attr(0.0f)},
        {"scale_out", res.Float32Attr(1.0f)},
        {"force_fp32_output", res.BoolAttr(false)}};

    const auto &fused_reshape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("int_array");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    fused_attrs.emplace("fused_reshape_out", fused_reshape_attr);
    fused_attrs.emplace("fused_transpose_out", pat.Attr("perm"));

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.InputNoneTensor()},
                 {&res.Tensor("reshape_out")});
  }
};

class FusedMatmulTransposeReshapeFusePattern
    : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;

 public:
  FusedMatmulTransposeReshapeFusePattern(std::string matmul_name,
                                         std::string fused_matmul_name,
                                         uint32_t benefit)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit) {}

  std::string name() const override {
    return "FusedMatmulTransposeReshapeFusePattern";
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

    matmul({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("residual")},
           {&pat.Tensor("Out")});

    const auto &transpose = pat.Op(paddle::dialect::TransposeOp::name(),
                                   {{"perm", pat.Attr("perm")}});
    pat.Tensor("transpose_out") = transpose(pat.Tensor("Out"));

    const auto &full_int_array = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                        {{"value", pat.Attr("int_array")}});
    pat.Tensor("shape") = full_int_array();

    const auto &reshape = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape({&pat.Tensor("transpose_out"), &pat.Tensor("shape")},
            {&pat.Tensor("reshape_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto shape = match_ctx.Attr<std::vector<int64_t>>("int_array");
      auto perm = match_ctx.Attr<std::vector<int>>("perm");
      const std::vector<int> supported_axis{0, 2, 1, 3};
      if (perm != supported_axis) return false;
      if (shape.size() != 3) return false;
      if (std::count(shape.begin(), shape.end(), -1) > 1) return false;

      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (!(match_ctx.Attr<std::vector<int>>("fused_reshape_out").empty()))
        return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
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
        {"mkldnn_data_type", pat.Attr("mkldnn_data_type")},
        {"scale_x", pat.Attr("scale_x")},
        {"scale_y", pat.Attr("scale_y")},
        {"scale_in_eltwise", pat.Attr("scale_in_eltwise")},
        {"scale_out", pat.Attr("scale_out")},
        {"force_fp32_output", pat.Attr("force_fp32_output")}};

    const auto &fused_reshape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("int_array");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    fused_attrs.emplace("fused_reshape_out", fused_reshape_attr);
    fused_attrs.emplace("fused_transpose_out", pat.Attr("perm"));

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("reshape_out")});
  }
};

class MatmulTransposeReshapeFusePass : public pir::PatternRewritePass {
 public:
  MatmulTransposeReshapeFusePass()
      : pir::PatternRewritePass("matmul_transpose_reshape_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    int benefit_idx = 1;
    ps.Add(paddle::drr::Create<MatmulTransposeReshapeFusePattern>(
        context,
        paddle::dialect::MatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++));

    ps.Add(paddle::drr::Create<FusedMatmulTransposeReshapeFusePattern>(
        context,
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulTransposeReshapeFusePass() {
  // pd_op.matmul + pd_op.transpose + pd_op.reshape -> onednn_op.fused_matmul
  // pd_op.fused_matmul + pd_op.transpose + pd_op.reshape ->
  // onednn_op.fused_matmul
  return std::make_unique<MatmulTransposeReshapeFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_transpose_reshape_fuse_pass,
                 MatmulTransposeReshapeFusePass);
