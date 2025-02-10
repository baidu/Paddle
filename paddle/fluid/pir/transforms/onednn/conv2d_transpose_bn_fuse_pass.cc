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

#include "paddle/fluid/pir/transforms/onednn/conv2d_transpose_bn_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class Conv2dTransposeBnOneDNNFusePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "Conv2dTransposeBnOneDNNFusePattern";
  }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv =
        pat.Op(paddle::dialect::Conv2dTransposeOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"output_padding", pat.Attr("output_padding")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    conv({&pat.Tensor("input"),
          &pat.Tensor("filter"),
          &pat.Tensor("output_size")},
         {&pat.Tensor("conv2d_out")});

    const auto &bn =
        pat.Op(paddle::dialect::BatchNorm_Op::name(),
               {{"is_test", pat.Attr("is_test")},
                {"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")}});

    bn({&pat.Tensor("conv2d_out"),
        &pat.Tensor("bn_mean"),
        &pat.Tensor("bn_var"),
        &pat.Tensor("bn_scale"),
        &pat.Tensor("bn_bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("var_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      std::vector<int64_t> conv_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto paddings_size = match_ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> bn_bias_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("bn_bias"));
      std::vector<int64_t> filter_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("filter"));

      auto conv2d_filter_dtype =
          pir::GetDataTypeFromValue(match_ctx.Tensor("filter"));
      if (conv2d_filter_dtype.isa<pir::Float16Type>()) {
        return false;
      }

      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      if (conv_input_shape.size() != 4) {
        return false;
      }
      float epsilon = !match_ctx.Attr<float>("epsilon");
      if (epsilon < 0.0f || epsilon > 0.001f) {
        return false;
      }

      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    // bn_var shape
    const auto &bn_var_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto bn_var_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bn_var"));
          return bn_var_shape;
        });

    // reshape scale shape
    const auto &scale_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto bn_scale_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bn_scale"));
          return {bn_scale_shape[0], 1, 1, 1};
        });

    const auto &full1 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", bn_var_shape_attr},
                                {"value", pat.Attr("epsilon")},
                                {"dtype", res.DataTypeAttr("float32")},
                                {"place", res.PlaceAttr("cpu")}});
    const auto &var_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("var_add_out") = var_add(res.Tensor("bn_var"), full1());
    const auto &sqrt = res.Op(paddle::dialect::SqrtOp::name());
    res.Tensor("sqrt_out") = sqrt(res.Tensor("var_add_out"));
    const auto &div = res.Op(paddle::dialect::DivideOp::name());
    res.Tensor("new_scale") =
        div(res.Tensor("bn_scale"), res.Tensor("sqrt_out"));

    const auto &reshape_scale = res.Op(paddle::dialect::ReshapeOp::name(),
                                       {{"shape", scale_shape_attr}});
    res.Tensor("res_scale") = reshape_scale(res.Tensor("new_scale"));

    //--- deal with filter ---

    // ConvTranspose weight is gIOHW, conv is gOIHW
    // We transpose IOHW to IOHW first, then multiply scale, and transpose it to
    // IOHW again
    const auto &new_conv2d_filter_shape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int64_t> filter_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("filter"));
          std::vector<int> new_conv2d_filter_shape;
          int size = filter_shape.size();
          for (int i = 0; i < size; i++) {
            new_conv2d_filter_shape.emplace_back(i);
          }
          auto groups = static_cast<int>(match_ctx.Attr<int>("groups"));
          int g_dim = (groups > 1) ? 1 : 0;
          std::swap(new_conv2d_filter_shape[g_dim + 0],
                    new_conv2d_filter_shape[g_dim + 1]);

          return new_conv2d_filter_shape;
        });

    const auto &transpose_filter_op =
        res.Op(paddle::dialect::TransposeOp::name(),
               {{"perm", new_conv2d_filter_shape}});

    res.Tensor("transpose_filter") = transpose_filter_op(res.Tensor("filter"));

    const auto &mul_filter_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("res_filter") =
        mul_filter_op(res.Tensor("transpose_filter"), res.Tensor("res_scale"));

    const auto &res_filter_op = res.Op(paddle::dialect::TransposeOp::name(),
                                       {{"perm", new_conv2d_filter_shape}});
    res.Tensor("res_transpose_filter") =
        res_filter_op(res.Tensor("res_filter"));

    // --- deal with bias ---
    // new bias: bn_bias - (bn_mean * scale)
    const auto &bn_mean_mul_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("bn_mean_mul_out") =
        bn_mean_mul_op(res.Tensor("bn_mean"), res.Tensor("new_scale"));
    const auto &sub_bias_op = res.Op(paddle::dialect::SubtractOp::name());
    res.Tensor("res_bias") =
        sub_bias_op(res.Tensor("bn_bias"), res.Tensor("bn_mean_mul_out"));

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::Conv2dTransposeBiasOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"output_padding", pat.Attr("output_padding")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_relu", res.BoolAttr(false)},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"is_test", res.BoolAttr(true)},
               }});
    fused_conv({&res.Tensor("input"),
                &res.Tensor("res_transpose_filter"),
                &res.Tensor("res_bias"),
                &res.Tensor("output_size")},
               {&res.Tensor("bn_out")});
  }
};

class Conv2dTransposeEltwiseBnOneDNNFusePattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "Conv2dTransposeEltwiseBnOneDNNFusePattern";
  }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv =
        pat.Op(paddle::dialect::Conv2dTransposeOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"output_padding", pat.Attr("output_padding")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    conv({&pat.Tensor("input"),
          &pat.Tensor("filter"),
          &pat.Tensor("output_size")},
         {&pat.Tensor("conv2d_out")});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    pat.Tensor("add_out") =
        add(pat.Tensor("conv2d_out"), pat.Tensor("residual_param"));

    const auto &bn =
        pat.Op(paddle::dialect::BatchNorm_Op::name(),
               {{"is_test", pat.Attr("is_test")},
                {"momentum", pat.Attr("momentum")},
                {"epsilon", pat.Attr("epsilon")},
                {"data_format", pat.Attr("data_format")},
                {"use_global_stats", pat.Attr("use_global_stats")},
                {"trainable_statistics", pat.Attr("trainable_statistics")}});

    bn({&pat.Tensor("add_out"),
        &pat.Tensor("bn_mean"),
        &pat.Tensor("bn_var"),
        &pat.Tensor("bn_scale"),
        &pat.Tensor("bn_bias")},
       {&pat.Tensor("bn_out"),
        &pat.Tensor("mean_out"),
        &pat.Tensor("var_out"),
        &pat.Tensor("saved_mean"),
        &pat.Tensor("saved_variance"),
        &pat.Tensor("reserve_space")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (!pir::ValueIsPersistable(match_ctx.Tensor("residual_param"))) {
        return false;
      }
      std::vector<int64_t> conv_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto paddings_size = match_ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> bn_bias_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("bn_bias"));
      std::vector<int64_t> filter_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("filter"));
      std::vector<int64_t> residual_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("residual_param"));
      if (residual_shape.size() != 1) {
        return false;
      }
      if (residual_shape.at(0) != 1) {
        return false;
      }

      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }
      if (conv_input_shape.size() != 4) {
        return false;
      }
      float epsilon = !match_ctx.Attr<float>("epsilon");
      if (epsilon < 0.0f || epsilon > 0.001f) {
        return false;
      }

      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    // bn_var shape
    const auto &bn_var_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto bn_var_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bn_var"));
          return bn_var_shape;
        });

    // reshape scale shape
    const auto &scale_shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto bn_scale_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("bn_scale"));
          return {bn_scale_shape[0], 1, 1, 1};
        });

    const auto &full1 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", bn_var_shape_attr},
                                {"value", pat.Attr("epsilon")},
                                {"dtype", res.DataTypeAttr("float32")},
                                {"place", res.PlaceAttr("cpu")}});
    const auto &var_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("var_add_out") = var_add(res.Tensor("bn_var"), full1());
    const auto &sqrt = res.Op(paddle::dialect::SqrtOp::name());
    res.Tensor("sqrt_out") = sqrt(res.Tensor("var_add_out"));
    const auto &div = res.Op(paddle::dialect::DivideOp::name());
    res.Tensor("new_scale") =
        div(res.Tensor("bn_scale"), res.Tensor("sqrt_out"));

    const auto &reshape_scale = res.Op(paddle::dialect::ReshapeOp::name(),
                                       {{"shape", scale_shape_attr}});
    res.Tensor("res_scale") = reshape_scale(res.Tensor("new_scale"));

    //--- deal with filter ---

    // ConvTranspose weight is gIOHW, conv is gOIHW
    // We transpose IOHW to IOHW first, then multiply scale, and transpose it to
    // IOHW again
    const auto &new_conv2d_filter_shape = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int64_t> filter_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("filter"));
          std::vector<int> new_conv2d_filter_shape;
          int size = filter_shape.size();
          for (int i = 0; i < size; i++) {
            new_conv2d_filter_shape.emplace_back(i);
          }
          auto groups = static_cast<int>(match_ctx.Attr<int>("groups"));
          int g_dim = (groups > 1) ? 1 : 0;
          std::swap(new_conv2d_filter_shape[g_dim + 0],
                    new_conv2d_filter_shape[g_dim + 1]);

          return new_conv2d_filter_shape;
        });

    const auto &transpose_filter_op =
        res.Op(paddle::dialect::TransposeOp::name(),
               {{"perm", new_conv2d_filter_shape}});

    res.Tensor("transpose_filter") = transpose_filter_op(res.Tensor("filter"));

    const auto &mul_filter_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("res_filter") =
        mul_filter_op(res.Tensor("transpose_filter"), res.Tensor("res_scale"));

    const auto &res_filter_op = res.Op(paddle::dialect::TransposeOp::name(),
                                       {{"perm", new_conv2d_filter_shape}});
    res.Tensor("res_transpose_filter") =
        res_filter_op(res.Tensor("res_filter"));

    // --- deal with bias ---
    // new bias: (elwise_add_y - bn_mean) * scale + bn_bias
    const auto &sub_bias_op = res.Op(paddle::dialect::SubtractOp::name());
    res.Tensor("add_bn") =
        sub_bias_op(res.Tensor("residual_param"), res.Tensor("bn_mean"));

    const auto &bn_mean_mul_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("bn_mean_mul_out") =
        bn_mean_mul_op(res.Tensor("add_bn"), res.Tensor("new_scale"));

    const auto &bn_mean_add = res.Op(paddle::dialect::AddOp::name());
    res.Tensor("add_bn_bias") =
        bn_mean_add(res.Tensor("bn_mean_mul_out"), res.Tensor("bn_bias"));

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::Conv2dTransposeBiasOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"output_padding", pat.Attr("output_padding")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_relu", res.BoolAttr(false)},
                   {"fuse_activation", res.StrAttr("")},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"is_test", res.BoolAttr(true)},
               }});
    fused_conv({&res.Tensor("input"),
                &res.Tensor("res_transpose_filter"),
                &res.Tensor("add_bn_bias"),
                &res.Tensor("output_size")},
               {&res.Tensor("bn_out")});
  }
};

class ConvTransposeEltwiseaddBnOneDNNFusePass : public pir::PatternRewritePass {
 public:
  ConvTransposeEltwiseaddBnOneDNNFusePass()
      : pir::PatternRewritePass("conv2d_transpose_bias_bn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<Conv2dTransposeEltwiseBnOneDNNFusePattern>(
        context));
    return ps;
  }
};

class ConvTransposeBnOneDNNFusePass : public pir::PatternRewritePass {
 public:
  ConvTransposeBnOneDNNFusePass()
      : pir::PatternRewritePass("conv2d_transpose_bn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<Conv2dTransposeBnOneDNNFusePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dTransposeBnOneDNNFusePass() {
  return std::make_unique<ConvTransposeBnOneDNNFusePass>();
}

std::unique_ptr<Pass> CreateConv2dTransposeEltwiseaddBnOneDNNFusePass() {
  return std::make_unique<ConvTransposeEltwiseaddBnOneDNNFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_transpose_bn_fuse_pass, ConvTransposeBnOneDNNFusePass);
REGISTER_IR_PASS(conv2d_transpose_bias_bn_fuse_pass,
                 ConvTransposeEltwiseaddBnOneDNNFusePass);
