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

#include "paddle/fluid/pir/transforms/onednn/elementwise_act_onednn_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
std::string GetFusedElement(const std::string &elementwise_type) {
  const std::map<std::string, std::string> fused_ops = {
      {"pd_op.add", "onednn_op.fused_elementwise_add"},
      {"pd_op.subtract", "onednn_op.fused_elementwise_sub"},
      {"pd_op.multiply", "onednn_op.fused_elementwise_mul"}};
  auto it = fused_ops.find(elementwise_type);
  if (it != fused_ops.end()) {
    return it->second;
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("The op type is not supported."));
  }
}
class ElementwiseActivationFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string elementwise_type_;
  std::string activation_name_;
  const int level_;

 public:
  ElementwiseActivationFusePattern(std::string elementwise_type,
                                   std::string activation_name,
                                   int level)
      : elementwise_type_(std::move(elementwise_type)),
        activation_name_(std::move(activation_name)),
        level_(level) {}

  std::string name() const override {
    return elementwise_type_ + activation_name_ + "FusePattern";
  }

  uint32_t benefit() const override { return level_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &elementwise = pat.Op(elementwise_type_);

    std::string activation_name_op = "pd_op." + activation_name_;
    if (activation_name_ == "hard_swish") {
      // oneDNN use hard_swish, paddle use hardswish
      activation_name_op = "pd_op.hardswish";
    } else if (activation_name_ == "hard_sigmoid") {
      activation_name_op = "pd_op.hardsigmoid";
    }

    std::unordered_map<std::string, paddle::drr::Attribute> act_attrs;
    if (activation_name_op == paddle::dialect::HardsigmoidOp::name()) {
      act_attrs.emplace("slope", pat.Attr("slope"));
      act_attrs.emplace("offset", pat.Attr("offset"));
    } else if (activation_name_op == paddle::dialect::LeakyReluOp::name()) {
      act_attrs.emplace("negative_slope", pat.Attr("negative_slope"));
    }
    const auto &activation = pat.Op(activation_name_op, act_attrs);
    elementwise({&pat.Tensor("x"), &pat.Tensor("y")},
                {&pat.Tensor("elementwise_out")});

    pat.Tensor("act_out") = activation(pat.Tensor("elementwise_out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      if (activation_name_ == "leaky_relu") {
        float negative_slope = match_ctx.Attr<float>("negative_slope");
        // leaky relu alpha is a positive number
        if (negative_slope <= 0.0) {
          return false;
        }
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto fuse_beta = res.Float32Attr(0.0f);
    auto fuse_alpha = res.Float32Attr(0.0f);
    if (activation_name_ == "relu6") {
      fuse_beta = res.Float32Attr(6.0f);
    } else if (activation_name_ == "hard_swish") {
      fuse_beta = res.Float32Attr(1.f / 2.f);
      fuse_alpha = res.Float32Attr(1.f / 6.f);
    } else if (activation_name_ == "swish") {
      fuse_alpha = res.Float32Attr(1.0f);
    } else if (activation_name_ == "leaky_relu") {
      fuse_alpha = pat.Attr("negative_slope");
    } else if (activation_name_ == "hard_sigmoid") {
      fuse_alpha = pat.Attr("slope");
      fuse_beta = pat.Attr("offset");
    }

    std::string fused_elementwise_type = GetFusedElement(elementwise_type_);

    const auto &fused_elementwise =
        res.Op(fused_elementwise_type,
               {{
                   {"axis", res.Int32Attr(-1)},
                   {"fuse_activation", res.StrAttr(activation_name_)},
                   {"fuse_alpha", fuse_alpha},
                   {"fuse_beta", fuse_beta},
                   {"fused_output_scale", res.Float32Attr(1.0f)},
                   {"fused_unsqueeze2_axes", res.VectorInt32Attr({})},
                   {"scale_x", res.Float32Attr(1.0f)},
                   {"scale_y", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
               }});

    fused_elementwise({&res.Tensor("x"), &res.Tensor("y")},
                      {&res.Tensor("act_out")});
  }
};

class ElementwiseGeluFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string elementwise_type_;
  std::string activation_name_;
  const int level_;

 public:
  ElementwiseGeluFusePattern(const std::string elementwise_type,
                             std::string activation_name,
                             int level)
      : elementwise_type_(elementwise_type),
        activation_name_(std::move(activation_name)),
        level_(level) {}

  std::string name() const override {
    return elementwise_type_ + "GeluFusePattern";
  }

  uint32_t benefit() const override { return level_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &elementwise = pat.Op(elementwise_type_);

    const auto &activation =
        pat.Op(activation_name_, {{"approximate", pat.Attr("approximate")}});
    elementwise({&pat.Tensor("x"), &pat.Tensor("y")},
                {&pat.Tensor("elementwise_out")});

    pat.Tensor("act_out") = activation(pat.Tensor("elementwise_out"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &gelu = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::string {
          bool approximate = match_ctx.Attr<bool>("approximate");
          if (approximate) return "gelu_tanh";
          return "gelu_erf";
        });
    std::string fused_elementwise_type = GetFusedElement(elementwise_type_);
    const auto &fused_elementwise =
        res.Op(fused_elementwise_type,
               {{
                   {"axis", res.Int32Attr(-1)},
                   {"fuse_activation", gelu},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"fused_output_scale", res.Float32Attr(1.0f)},
                   {"fused_unsqueeze2_axes", res.VectorInt32Attr({})},
                   {"scale_x", res.Float32Attr(1.0f)},
                   {"scale_y", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
               }});

    fused_elementwise({&res.Tensor("x"), &res.Tensor("y")},
                      {&res.Tensor("act_out")});
  }
};

class ElementwiseClipFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string elementwise_type_;
  std::string activation_name_;
  const int level_;

 public:
  ElementwiseClipFusePattern(std::string elementwise_type,
                             std::string activation_name,
                             int level)
      : elementwise_type_(std::move(elementwise_type)),
        activation_name_(std::move(activation_name)),
        level_(level) {}

  std::string name() const override {
    return elementwise_type_ + "ClipFusePattern";
  }

  uint32_t benefit() const override { return level_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &elementwise = pat.Op(elementwise_type_);

    const auto &full_1 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    const auto &full_2 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_2_value")}});
    pat.Tensor("min") = full_1();
    pat.Tensor("max") = full_2();

    const auto &activation = pat.Op(activation_name_);
    elementwise({&pat.Tensor("x"), &pat.Tensor("y")},
                {&pat.Tensor("elementwise_out")});

    pat.Tensor("act_out") = activation(
        pat.Tensor("elementwise_out"), pat.Tensor("min"), pat.Tensor("max"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::string fused_elementwise_type = GetFusedElement(elementwise_type_);

    const auto &fuse_alpha = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_1_value");
        });
    const auto &fuse_beta = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_2_value");
        });

    const auto &fused_elementwise =
        res.Op(fused_elementwise_type,
               {{
                   {"axis", res.Int32Attr(-1)},
                   {"fuse_activation", res.StrAttr("clip")},
                   {"fuse_alpha", fuse_alpha},
                   {"fuse_beta", fuse_beta},
                   {"fused_output_scale", res.Float32Attr(1.0f)},
                   {"fused_unsqueeze2_axes", res.VectorInt32Attr({})},
                   {"scale_x", res.Float32Attr(1.0f)},
                   {"scale_y", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
               }});

    fused_elementwise({&res.Tensor("x"), &res.Tensor("y")},
                      {&res.Tensor("act_out")});
  }
};

class ElementwiseActFusePass : public pir::PatternRewritePass {
 public:
  ElementwiseActFusePass()
      : pir::PatternRewritePass("elementwise_act_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    // This ten activations have no extra attribute, can use the same pattern
    std::vector<std::string> supported_activations_name = {"abs",
                                                           "sqrt",
                                                           "mish",
                                                           "relu",
                                                           "sigmoid",
                                                           "tanh",
                                                           "relu6",
                                                           "hard_swish",
                                                           "swish",
                                                           "leaky_relu",
                                                           "hard_sigmoid"};
    size_t pattern_num = 1;
    for (const auto &activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ElementwiseActivationFusePattern>(
          context, paddle::dialect::AddOp::name(), activation, pattern_num));
      pattern_num++;
    }

    for (const auto &activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ElementwiseActivationFusePattern>(
          context,
          paddle::dialect::SubtractOp::name(),
          activation,
          pattern_num));
      pattern_num++;
    }

    for (auto activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ElementwiseActivationFusePattern>(
          context,
          paddle::dialect::MultiplyOp::name(),
          activation,
          pattern_num));
      pattern_num++;
    }

    ps.Add(paddle::drr::Create<ElementwiseGeluFusePattern>(
        context,
        paddle::dialect::AddOp::name(),
        paddle::dialect::GeluOp::name(),
        1));
    ps.Add(paddle::drr::Create<ElementwiseGeluFusePattern>(
        context,
        paddle::dialect::SubtractOp::name(),
        paddle::dialect::GeluOp::name(),
        2));
    ps.Add(paddle::drr::Create<ElementwiseGeluFusePattern>(
        context,
        paddle::dialect::MultiplyOp::name(),
        paddle::dialect::GeluOp::name(),
        3));

    ps.Add(paddle::drr::Create<ElementwiseClipFusePattern>(
        context,
        paddle::dialect::AddOp::name(),
        paddle::dialect::ClipOp::name(),
        1));
    ps.Add(paddle::drr::Create<ElementwiseClipFusePattern>(
        context,
        paddle::dialect::SubtractOp::name(),
        paddle::dialect::ClipOp::name(),
        2));
    ps.Add(paddle::drr::Create<ElementwiseClipFusePattern>(
        context,
        paddle::dialect::MultiplyOp::name(),
        paddle::dialect::ClipOp::name(),
        3));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateElementwiseActivationFusePass() {
  /**
   *  elementxx
   *    |     ->  fused_elementxx
   * activation
   */
  return std::make_unique<ElementwiseActFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(elementwise_act_onednn_fuse_pass, ElementwiseActFusePass);
