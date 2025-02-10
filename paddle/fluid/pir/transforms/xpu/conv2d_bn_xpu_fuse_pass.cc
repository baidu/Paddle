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

#include "paddle/fluid/pir/transforms/xpu/conv2d_bn_xpu_fuse_pass.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/backends/xpu/xpu_info.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class Conv2dBnFusePattern : public paddle::drr::DrrPatternBase {
 private:
  bool bn_inplace_;

 public:
  explicit Conv2dBnFusePattern(bool bn_inplace) : bn_inplace_(bn_inplace) {}

  std::string name() const override { return "Conv2dBnFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv2d =
        pat.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &bn = pat.Op(bn_inplace_ ? paddle::dialect::BatchNorm_Op::name()
                                        : paddle::dialect::BatchNormOp::name(),
                            {
                                {"epsilon", pat.Attr("epsilon")},
                            });

    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
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
      if (conv_input_shape.size() != 4) {
        return false;
      }
      if (!pir::ValueIsPersistable(match_ctx.Tensor("bn_mean")) ||
          !pir::ValueIsPersistable(match_ctx.Tensor("bn_var")) ||
          !pir::ValueIsPersistable(match_ctx.Tensor("bn_scale")) ||
          !pir::ValueIsPersistable(match_ctx.Tensor("bn_bias"))) {
        return false;
      }
      if (!(paddings_size.size() == 2 || paddings_size.size() == 4)) {
        return false;
      }
      if (bn_bias_shape.at(0) != filter_shape.at(0)) {
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

    // reshape scale shape
    const auto &expand_1_shape =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx)
                            -> std::vector<int64_t> {
          return {static_cast<int64_t>(
              phi::backends::xpu::get_xpu_max_ptr_size(-1))};
        });

    // paddings
    const auto &paddings_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          auto paddings = match_ctx.Attr<std::vector<int>>("paddings");
          if (paddings.size() == 2) {
            return {paddings[0], paddings[0], paddings[1], paddings[1]};
          } else {
            return paddings;
          }
        });

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("input"));
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    // make new scale:  bn_scale/sqrt(bn_var+epsilon)
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
    const auto &mul_filter_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("res_filter") =
        mul_filter_op(res.Tensor("filter"), res.Tensor("res_scale"));

    // --- deal with bias ---
    // new bias: bn_bias - (bn_mean * scale)
    const auto &bn_mean_mul_op = res.Op(paddle::dialect::MultiplyOp::name());
    res.Tensor("bn_mean_mul_out") =
        bn_mean_mul_op(res.Tensor("bn_mean"), res.Tensor("new_scale"));
    const auto &sub_bias_op = res.Op(paddle::dialect::SubtractOp::name());
    res.Tensor("res_bias") =
        sub_bias_op(res.Tensor("bn_bias"), res.Tensor("bn_mean_mul_out"));

    // get max filter and max x
    const auto &max_op1 =
        res.Op(paddle::dialect::MaxOp::name(),
               {{"axis", res.VectorInt64Attr(std::vector<int64_t>{})},
                {"keepdim", res.BoolAttr(false)}});
    res.Tensor("filter_max") = max_op1(res.Tensor("filter"));
    const auto &expand =
        res.Op(paddle::dialect::ExpandOp::name(), {{"shape", expand_1_shape}});
    res.Tensor("res_filter_max") = expand(res.Tensor("filter_max"));

    const auto &conv2d_xpu =
        res.Op(paddle::dialect::Conv2dXpuOp::name(),
               {{
                   {"paddings", paddings_attr},
                   {"dilations", pat.Attr("dilations")},
                   {"strides", pat.Attr("strides")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"groups", pat.Attr("groups")},
                   {"act_type",
                    res.Int32Attr(static_cast<int>(xpu::Activation_t::LINEAR))},
                   {"act_param", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    conv2d_xpu(
        {
            &res.Tensor("input"),
            &res.InputNoneTensor(),
            &res.Tensor("res_filter"),
            &res.Tensor("res_filter_max"),
            &res.Tensor("res_bias"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("bn_out"), &res.Tensor("out_max")});
  }
};

class Conv2dBnFuseXpuPass : public pir::PatternRewritePass {
 public:
  Conv2dBnFuseXpuPass()
      : pir::PatternRewritePass("conv2d_bn_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    bool bn_inplace = true;
    ps.Add(paddle::drr::Create<Conv2dBnFusePattern>(context, bn_inplace));
    ps.Add(paddle::drr::Create<Conv2dBnFusePattern>(context, !bn_inplace));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dBnFuseXpuPass() {
  return std::make_unique<Conv2dBnFuseXpuPass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv2d_bn_xpu_fuse_pass, Conv2dBnFuseXpuPass);
