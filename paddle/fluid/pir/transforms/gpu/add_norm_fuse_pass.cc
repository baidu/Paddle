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

#include "paddle/fluid/pir/transforms/gpu/add_norm_fuse_pass.h"

#include <string>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class RmsNormFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const bool is_half_weight_;

 public:
  explicit RmsNormFusePattern(bool is_half_weight)
      : is_half_weight_(is_half_weight) {}

  std::string name() const override { return "RmsNormFusePattern"; }

  uint32_t benefit() const override { return 3; }

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
      pat.Tensor("cast_1_out") = cast1(pat.Tensor("x"));
      pat.Tensor("pow_out") = pow(pat.Tensor("cast_1_out"));
      pat.Tensor("full_int_array_out") = full_int_array();
      pat.Tensor("mean_out") =
          mean(pat.Tensor("pow_out"), pat.Tensor("full_int_array_out"));
      pat.Tensor("scale_out") = scale(pat.Tensor("mean_out"), full());
      pat.Tensor("rsqrt_out") = rsqrt(pat.Tensor("scale_out"));
      pat.Tensor("multiply_out1") =
          multiply1(pat.Tensor("rsqrt_out"), pat.Tensor("cast_1_out"));
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

class AddRmsNormFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const bool extra_add_;
  const bool trans_extra_add_;

 public:
  AddRmsNormFusePattern(bool extra_add, bool trans_extra_add)
      : extra_add_(extra_add), trans_extra_add_{trans_extra_add} {}

  uint32_t benefit() const override { return extra_add_ ? 4 : 3; }

  std::string name() const override { return "AddRmsNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &pat_rms_norm =
        pat.Op(paddle::dialect::RmsNormOp::name(),
               {
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"quant_scale", pat.Attr("quant_scale")},
                   {"quant_round_type", pat.Attr("quant_round_type")},
                   {"quant_max_bound", pat.Attr("quant_max_bound")},
                   {"quant_min_bound", pat.Attr("quant_min_bound")},
               });
    pat.Tensor("add_out") = add(pat.Tensor("x"), pat.Tensor("residual"));
    pat_rms_norm({&pat.Tensor("add_out"),
                  &pat.Tensor("bias"),
                  &pat.InputNoneTensor(),
                  &pat.Tensor("w"),
                  &pat.InputNoneTensor()},
                 {&pat.Tensor("rms_norm_out"),
                  &pat.Tensor("residual_out_0"),
                  &pat.Tensor("inv_var_0")});
    // TODO(bukejiyu) :DRR support matching placeholder op,
    // the following needs to be deleted
    if (extra_add_) {
      const auto &add1 = pat.Op(paddle::dialect::AddOp::name());
      pat.Tensor("add_out1") =
          trans_extra_add_
              ? add1(pat.Tensor("any_tensor"), pat.Tensor("add_out"))
              : add1(pat.Tensor("add_out"), pat.Tensor("any_tensor"));
    }
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &res_rms_norm =
        res.Op(paddle::dialect::RmsNormOp::name(),
               {
                   {"epsilon", pat.Attr("epsilon")},
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"quant_scale", pat.Attr("quant_scale")},
                   {"quant_round_type", pat.Attr("quant_round_type")},
                   {"quant_max_bound", pat.Attr("quant_max_bound")},
                   {"quant_min_bound", pat.Attr("quant_min_bound")},
               });

    res_rms_norm(
        {
            &res.Tensor("x"),
            &res.Tensor("bias"),
            &res.Tensor("residual"),
            &res.Tensor("w"),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("rms_norm_out"),
         &res.Tensor("add_out"),
         &res.Tensor("inv_var")});
  }
};

class AddLayerNormFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const bool extra_add_;
  const bool trans_extra_add_;

 public:
  AddLayerNormFusePattern(bool extra_add, bool trans_extra_add)
      : extra_add_(extra_add), trans_extra_add_{trans_extra_add} {}

  uint32_t benefit() const override { return extra_add_ ? 4 : 3; }
  std::string name() const override { return "AddLayerNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layer_norm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});
    pat.Tensor("add_out") = add(pat.Tensor("x"), pat.Tensor("residual"));
    layer_norm({&pat.Tensor("add_out"), &pat.Tensor("w"), &pat.Tensor("bias")},
               {&pat.Tensor("layer_norm_out"),
                &pat.Tensor("mean_out_0"),
                &pat.Tensor("variance_out_0")});
    // TODO(bukejiyu) :DRR support matching placeholder op,
    // the following needs to be deleted
    if (extra_add_) {
      const auto &add1 = pat.Op(paddle::dialect::AddOp::name());
      pat.Tensor("add_out1") =
          trans_extra_add_
              ? add1(pat.Tensor("any_tensor"), pat.Tensor("add_out"))
              : add1(pat.Tensor("add_out"), pat.Tensor("any_tensor"));
    }
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto r_shape = pir::GetShapeFromValue(match_ctx.Tensor("residual"));
      if (x_shape[0] != r_shape[0]) {
        return false;
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &cast_op_dtype = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          return phi::DataType::FLOAT32;
        });
    const auto cast_1_op =
        res.Op(paddle::dialect::CastOp::name(), {{"dtype", cast_op_dtype}});
    const auto &fuse_layer_norm =
        res.Op(paddle::dialect::FusedBiasResidualLayernormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"residual_alpha", res.Float32Attr(1.0)},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                {"quant_scale", res.Float32Attr(-1.0)},
                {"quant_round_type", res.Int32Attr(0)},
                {"quant_max_bound", res.Float32Attr(0.0)},
                {"quant_min_bound", res.Float32Attr(0.0)}});
    res.Tensor("w_cast") = cast_1_op(res.Tensor("w"));
    res.Tensor("bias_cast") = cast_1_op(res.Tensor("bias"));
    fuse_layer_norm(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            &res.Tensor("residual"),
            &res.Tensor("w_cast"),
            &res.Tensor("bias_cast"),
        },
        {&res.Tensor("layer_norm_out"),
         &res.Tensor("add_out"),
         &res.Tensor("mean_out"),
         &res.Tensor("variance_out")});
  }
};

class AddGroupNormFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const bool extra_add_;
  const bool trans_extra_add_;
  const bool enable_gpu_mixed_;

 public:
  AddGroupNormFusePattern(bool extra_add,
                          bool trans_extra_add,
                          bool enable_gpu_mixed)
      : extra_add_(extra_add),
        trans_extra_add_{trans_extra_add},
        enable_gpu_mixed_(enable_gpu_mixed) {}

  uint32_t benefit() const override { return extra_add_ ? 4 : 3; }
  std::string name() const override { return "AddGroupNormFusePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &group_norm = pat.Op(paddle::dialect::GroupNormOp::name(),
                                    {{"epsilon", pat.Attr("epsilon")},
                                     {"groups", pat.Attr("groups")},
                                     {"data_format", pat.Attr("data_format")}});
    pat.Tensor("add_out") = add(pat.Tensor("x"), pat.Tensor("residual"));
    group_norm(
        {&pat.Tensor("add_out"), &pat.Tensor("scale"), &pat.Tensor("bias")},
        {&pat.Tensor("group_out"),
         &pat.Tensor("mean_out_0"),
         &pat.Tensor("variance_out_0")});
    // TODO(bukejiyu) :DRR support matching placeholder op,
    // the following needs to be deleted
    if (extra_add_) {
      const auto &add1 = pat.Op(paddle::dialect::AddOp::name());
      pat.Tensor("add_out1") =
          trans_extra_add_
              ? add1(pat.Tensor("any_tensor"), pat.Tensor("add_out"))
              : add1(pat.Tensor("add_out"), pat.Tensor("any_tensor"));
    }
    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
      if (!this->enable_gpu_mixed_ && !x_dtype.isa<pir::Float16Type>() &&
          !x_dtype.isa<pir::BFloat16Type>()) {
        return false;
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &add_group_norm_silu_op =
        res.Op(paddle::dialect::AddGroupNormSiluOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")},
                {"activation", res.StrAttr("")}});

    add_group_norm_silu_op({&res.Tensor("x"),
                            &res.Tensor("residual"),
                            &res.Tensor("scale"),
                            &res.Tensor("bias")},
                           {&res.Tensor("group_out"),
                            &res.Tensor("add_out"),
                            &res.Tensor("mean_out"),
                            &res.Tensor("variance_out")});
  }
};

class AddGroupNormWithActPattern : public paddle::drr::DrrPatternBase {
 private:
  const bool enable_gpu_mixed_;

 public:
  explicit AddGroupNormWithActPattern(bool enable_gpu_mixed)
      : enable_gpu_mixed_(enable_gpu_mixed) {}

  uint32_t benefit() const override { return 2; }
  std::string name() const override { return "AddGroupNormWithActPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add_group_norm_silu_op =
        pat.Op(paddle::dialect::AddGroupNormSiluOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")},
                {"activation", pat.Attr("activation")}});
    const auto &silu = pat.Op(paddle::dialect::SiluOp::name());
    add_group_norm_silu_op({&pat.Tensor("x"),
                            &pat.Tensor("residual"),
                            &pat.Tensor("scale"),
                            &pat.Tensor("bias")},
                           {&pat.Tensor("group_out"),
                            &pat.Tensor("add_out"),
                            &pat.Tensor("mean_out_0"),
                            &pat.Tensor("variance_out_0")});
    pat.Tensor("silu_out") = silu(pat.Tensor("group_out"));
    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
      if (!this->enable_gpu_mixed_ && !x_dtype.isa<pir::Float16Type>() &&
          !x_dtype.isa<pir::BFloat16Type>()) {
        return false;
      }
      auto activation = match_ctx.Attr<std::string>("activation");
      if (activation != "") {
        return false;
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &res_add_group_norm_silu_op =
        res.Op(paddle::dialect::AddGroupNormSiluOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")},
                {"activation", res.StrAttr("silu")}});
    res_add_group_norm_silu_op({&res.Tensor("x"),
                                &res.Tensor("residual"),
                                &res.Tensor("scale"),
                                &res.Tensor("bias")},
                               {&res.Tensor("silu_out"),
                                &res.Tensor("add_out"),
                                &res.Tensor("mean_out"),
                                &res.Tensor("variance_out")});
  }
};

class AddNormFusePass : public pir::PatternRewritePass {
 public:
  AddNormFusePass() : pir::PatternRewritePass("add_norm_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    bool enable_gpu_mixed = false;
    if (Has("enable_gpu_mixed")) {
      enable_gpu_mixed = Get<bool>("enable_gpu_mixed");
    }

    // x-pow-mean-scale->rsqrt-
    //                          mul--
    // x-----------------------
    //                                mul --->rms_norm
    // w-----------------------------
    bool is_half_weight = true;
    bool extra_add = true;
    ps.Add(paddle::drr::Create<RmsNormFusePattern>(context, !is_half_weight));
    ps.Add(paddle::drr::Create<RmsNormFusePattern>(context, is_half_weight));
    // x--------
    //           add-rms_norm ---> rms_norm
    // residual-
    ps.Add(
        paddle::drr::Create<AddRmsNormFusePattern>(context, !extra_add, false));
    ps.Add(
        paddle::drr::Create<AddRmsNormFusePattern>(context, extra_add, true));
    ps.Add(
        paddle::drr::Create<AddRmsNormFusePattern>(context, extra_add, false));

    // x--------
    //           add-layer_norm ----> fused_bias_residual_layernorm
    // residual-
    ps.Add(paddle::drr::Create<AddLayerNormFusePattern>(
        context, !extra_add, false));
    ps.Add(
        paddle::drr::Create<AddLayerNormFusePattern>(context, extra_add, true));
    ps.Add(paddle::drr::Create<AddLayerNormFusePattern>(
        context, extra_add, false));

    // x--------
    //           add-group_norm ----> add_group_norm_silu
    // residual-
    ps.Add(paddle::drr::Create<AddGroupNormFusePattern>(
        context, !extra_add, true, enable_gpu_mixed));
    ps.Add(paddle::drr::Create<AddGroupNormFusePattern>(
        context, extra_add, true, enable_gpu_mixed));
    ps.Add(paddle::drr::Create<AddGroupNormFusePattern>(
        context, extra_add, false, enable_gpu_mixed));

    // add_group_norm_silu-silu --->add_group_norm_silu
    ps.Add(paddle::drr::Create<AddGroupNormWithActPattern>(context,
                                                           enable_gpu_mixed));
    // group-silu->add_group_norm_silu moved to group_norm_silu_fuse_pass
    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateAddNormFusePass() {
  return std::make_unique<AddNormFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(add_norm_fuse_pass, AddNormFusePass);
