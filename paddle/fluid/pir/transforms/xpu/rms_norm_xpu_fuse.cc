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

#include "paddle/fluid/pir/transforms/xpu/rms_norm_xpu_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

/*
For example:
graph:

              x                  w
     _ _ _ _ _| _ _ _ _ _        |
     |                  |        |
    cast               cast      |
     |                   |       |
     |                   |       |
    pow                  |       |
     |                   |       |
    mean     epilson     |       |
       \     /           |       |
        rsqrt            |       |
          |              |       |
            \          /         |
              multiply           |
                 |               |
                cast             |
                    \          /
                      multiply
                          |
                        output
------------------------------------------------------
After the pass is applied:
                   x      w
                     \  /
                      |
                   rms_norm
                      |
                      |
                    Output
*/

namespace {

class RmsNormFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const bool is_half_weight_;

 public:
  explicit RmsNormFusePattern(bool is_half_weight)
      : is_half_weight_(is_half_weight) {}

  std::string name() const override { return "RmsNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &pow = pat.Op(paddle::dialect::PowOp::name());
    const auto &full_int_array = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                        {{"value", pat.Attr("value")}});
    const auto &mean = pat.Op(paddle::dialect::MeanOp::name());
    const auto &full = pat.Op(paddle::dialect::FullOp::name());
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(), {{"bias", pat.Attr("bias")}});
    const auto &rsqrt = pat.Op(paddle::dialect::RsqrtOp::name());
    const auto &multiply1 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply2 = pat.Op(paddle::dialect::MultiplyOp::name());
    if (is_half_weight_) {
      const auto &cast1 = pat.Op(paddle::dialect::CastOp::name(),
                                 {{"dtype", pat.Attr("cast_type_1")}});
      const auto &cast3 = pat.Op(paddle::dialect::CastOp::name(),
                                 {{"dtype", pat.Attr("cast_type_1")}});
      pat.Tensor("cast_1_out") = cast1(pat.Tensor("x"));
      pat.Tensor("cast_3_out") = cast3(pat.Tensor("x"));
      pat.Tensor("pow_out") = pow(pat.Tensor("cast_1_out"));
      pat.Tensor("full_int_array_out") = full_int_array();
      pat.Tensor("mean_out") =
          mean(pat.Tensor("pow_out"), pat.Tensor("full_int_array_out"));
      pat.Tensor("scale_out") = scale(pat.Tensor("mean_out"), full());
      pat.Tensor("rsqrt_out") = rsqrt(pat.Tensor("scale_out"));
      pat.Tensor("multiply_out1") =
          multiply1(pat.Tensor("rsqrt_out"), pat.Tensor("cast_3_out"));
      const auto &cast2 = pat.Op(paddle::dialect::CastOp::name(),
                                 {{"dtype", pat.Attr("cast_type_2")}});
      pat.Tensor("cast_2_out") = cast2(pat.Tensor("multiply_out1"));
      pat.Tensor("multiply_out2") =
          multiply2(pat.Tensor("cast_2_out"), pat.Tensor("w"));
    } else {
      pat.Tensor("pow_out") = pow(pat.Tensor("x"));
      pat.Tensor("full_int_array_out") = full_int_array();
      pat.Tensor("mean_out") =
          mean(pat.Tensor("pow_out"), pat.Tensor("full_int_array_out"));
      pat.Tensor("scale_out") = scale(pat.Tensor("mean_out"), full());
      pat.Tensor("rsqrt_out") = rsqrt(pat.Tensor("scale_out"));
      pat.Tensor("multiply_out1") =
          multiply1(pat.Tensor("rsqrt_out"), pat.Tensor("x"));
      pat.Tensor("multiply_out2") =
          multiply2(pat.Tensor("multiply_out1"), pat.Tensor("w"));
    }

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto axis = match_ctx.Attr<std::vector<int64_t>>("value");
      if (axis.size() > 1) {
        return false;
      }
      if (this->is_half_weight_) {
        auto w_type = pir::GetDataTypeFromValue(match_ctx.Tensor("w"));
        if (!(w_type.isa<pir::Float16Type>() ||
              w_type.isa<pir::BFloat16Type>())) {
          return false;
        }

        auto cast_type_1 = match_ctx.Attr<phi::DataType>("cast_type_1");
        auto cast_type_2 = match_ctx.Attr<phi::DataType>("cast_type_2");
        if (cast_type_1 != phi::DataType::FLOAT32) {
          return false;
        }
        if (w_type.isa<pir::Float16Type>() &&
            cast_type_2 != phi::DataType::FLOAT16) {
          return false;
        }
        if (w_type.isa<pir::BFloat16Type>() &&
            cast_type_2 != phi::DataType::BFLOAT16) {
          return false;
        }
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &begin_norm_axis =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> int {
          const auto &axis = match_ctx.Attr<std::vector<int64_t>>("value");
          auto pow_out_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("pow_out"));
          return axis[0] == -1 ? static_cast<int>(pow_out_shape.size()) - 1
                               : axis[0];
        });

    const auto &rms_norm = res.Op(paddle::dialect::RmsNormOp::name(),
                                  {{
                                      {"epsilon", pat.Attr("bias")},
                                      {"begin_norm_axis", begin_norm_axis},
                                      {"quant_scale", res.Float32Attr(-1.0)},
                                      {"quant_round_type", res.Int32Attr(0)},
                                      {"quant_max_bound", res.Float32Attr(0.0)},
                                      {"quant_min_bound", res.Float32Attr(0.0)},
                                  }});

    rms_norm(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.Tensor("w"),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("multiply_out2"),
         &res.Tensor("residual_out"),
         &res.Tensor("inv_var")});
  }
};

class RmsNormXpuFusePass : public pir::PatternRewritePass {
 public:
  RmsNormXpuFusePass() : pir::PatternRewritePass("rms_norm_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<RmsNormFusePattern>(context, false));
    ps.Add(paddle::drr::Create<RmsNormFusePattern>(context, true));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateRmsNormXpuFusePass() {
  return std::make_unique<RmsNormXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(rms_norm_xpu_fuse_pass, RmsNormXpuFusePass);
