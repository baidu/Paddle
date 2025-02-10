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

#include "paddle/fluid/pir/transforms/onednn/softplus_activation_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
std::set<std::string> act_ops = {paddle::dialect::AbsOp::name(),          //
                                 paddle::dialect::Abs_Op::name(),         //
                                 paddle::dialect::GeluOp::name(),         //
                                 paddle::dialect::HardsigmoidOp::name(),  //
                                 paddle::dialect::HardswishOp::name(),    //
                                 paddle::dialect::LeakyReluOp::name(),    //
                                 paddle::dialect::LeakyRelu_Op::name(),   //
                                 paddle::dialect::MishOp::name(),         //
                                 paddle::dialect::ReluOp::name(),         //
                                 paddle::dialect::Relu_Op::name(),        //
                                 paddle::dialect::Relu6Op::name(),        //
                                 paddle::dialect::SigmoidOp::name(),      //
                                 paddle::dialect::Sigmoid_Op::name(),     //
                                 paddle::dialect::SqrtOp::name(),         //
                                 paddle::dialect::Sqrt_Op::name(),        //
                                 paddle::dialect::SwishOp::name(),        //
                                 paddle::dialect::TanhOp::name(),         //
                                 paddle::dialect::Tanh_Op::name()};

std::unordered_map<std::string, std::string> activation_type = {
    {paddle::dialect::AbsOp::name(), "abs"},
    {paddle::dialect::Abs_Op::name(), "abs"},
    {paddle::dialect::GeluOp::name(), "gelu"},
    {paddle::dialect::HardsigmoidOp::name(), "hard_sigmoid"},
    {paddle::dialect::HardswishOp::name(), "hard_swish"},
    {paddle::dialect::LeakyReluOp::name(), "leaky_relu"},
    {paddle::dialect::LeakyRelu_Op::name(), "leaky_relu"},
    {paddle::dialect::MishOp::name(), "mish"},
    {paddle::dialect::ReluOp::name(), "relu"},
    {paddle::dialect::Relu_Op::name(), "relu"},
    {paddle::dialect::Relu6Op::name(), "relu6"},
    {paddle::dialect::SigmoidOp::name(), "sigmoid"},
    {paddle::dialect::Sigmoid_Op::name(), "sigmoid"},
    {paddle::dialect::SqrtOp::name(), "sqrt"},
    {paddle::dialect::Sqrt_Op::name(), "sqrt"},
    {paddle::dialect::SwishOp::name(), "swish"},
    {paddle::dialect::TanhOp::name(), "tanh"},
    {paddle::dialect::Tanh_Op::name(), "tanh"}};

class SoftplusActivationFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string softplus_name_;
  std::string fused_softplus_name_;
  uint32_t benefit_;
  std::string act_type_;

 public:
  SoftplusActivationFusePattern(std::string softplus_name,
                                std::string fused_softplus_name,
                                uint32_t benefit,
                                std::string act_type)
      : softplus_name_(std::move(softplus_name)),
        fused_softplus_name_(std::move(fused_softplus_name)),
        benefit_(benefit),
        act_type_(std::move(act_type)) {}

  std::string name() const override { return "SoftplusActivationFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &softplus = pat.Op(
        softplus_name_,
        {{"beta", pat.Attr("beta")}, {"threshold", pat.Attr("threshold")}});

    std::unordered_map<std::string, paddle::drr::Attribute> act_attrs;
    if (act_type_ == paddle::dialect::HardsigmoidOp::name()) {
      act_attrs.emplace("slope", pat.Attr("fuse_alpha"));
      act_attrs.emplace("offset", pat.Attr("fuse_beta"));
    } else if (act_type_ == paddle::dialect::LeakyRelu_Op::name() ||
               act_type_ == paddle::dialect::LeakyReluOp::name()) {
      act_attrs.emplace("negative_slope", pat.Attr("fuse_alpha"));
    } else if (act_type_ == paddle::dialect::GeluOp::name()) {
      act_attrs.emplace("approximate", pat.Attr("approximate"));
    }

    const auto &act = pat.Op(act_type_, act_attrs);
    softplus({&pat.Tensor("x")}, {&pat.Tensor("Out")});

    pat.Tensor("act_out") = act(pat.Tensor("Out"));

    if (act_type_ == paddle::dialect::GeluOp::name()) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto result_gelu = match_ctx.Attr<bool>("approximate");
        if (result_gelu) return false;
        return true;
      });
    }

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"beta", pat.Attr("beta")}, {"threshold", pat.Attr("threshold")}};

    if (act_type_ == paddle::dialect::HardswishOp::name()) {
      fused_attrs.emplace("fuse_alpha", res.Float32Attr(1.0f / 6.0f));
      fused_attrs.emplace("fuse_beta", res.Float32Attr(1.0f / 2.0f));
    } else if (act_type_ == paddle::dialect::HardsigmoidOp::name()) {
      fused_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      fused_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
    } else if (act_type_ == paddle::dialect::LeakyRelu_Op::name() ||
               act_type_ == paddle::dialect::LeakyReluOp::name()) {
      fused_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
    } else if (act_type_ == paddle::dialect::SwishOp::name()) {
      fused_attrs.emplace("fuse_alpha", res.Float32Attr(1.0f));
    } else if (act_type_ == paddle::dialect::Relu6Op::name()) {
      fused_attrs.emplace("fuse_beta", res.Float32Attr(6.0f));
    }

    fused_attrs.insert(std::make_pair("fuse_activation",
                                      res.StrAttr(activation_type[act_type_])));
    fused_attrs.insert(std::make_pair("fuse_alpha", res.Float32Attr(0.0f)));
    fused_attrs.insert(std::make_pair("fuse_beta", res.Float32Attr(0.0f)));

    const auto &fused_softplus = res.Op(fused_softplus_name_, fused_attrs);

    fused_softplus({&res.Tensor("x")}, {&res.Tensor("act_out")});
  }
};

class SoftplusGeluTanhFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string softplus_name_;
  std::string fused_softplus_name_;
  uint32_t benefit_;

 public:
  SoftplusGeluTanhFusePattern(std::string softplus_name,
                              std::string fused_softplus_name,
                              uint32_t benefit)
      : softplus_name_(std::move(softplus_name)),
        fused_softplus_name_(std::move(fused_softplus_name)),
        benefit_(benefit) {}

  std::string name() const override { return "SoftplusActivationFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &softplus = pat.Op(
        softplus_name_,
        {{"beta", pat.Attr("beta")}, {"threshold", pat.Attr("threshold")}});

    const auto &act = pat.Op(paddle::dialect::GeluOp::name(),
                             {{"approximate", pat.Attr("approximate")}});
    softplus({&pat.Tensor("x")}, {&pat.Tensor("Out")});

    pat.Tensor("act_out") = act(pat.Tensor("Out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto result_gelu = match_ctx.Attr<bool>("approximate");
      if (!result_gelu) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"beta", pat.Attr("beta")},
        {"threshold", pat.Attr("threshold")},
        {"fuse_activation", res.StrAttr("gelu_tanh")},
        {"fuse_alpha", res.Float32Attr(0.0f)},
        {"fuse_beta", res.Float32Attr(0.0f)}};

    const auto &fused_softplus = res.Op(fused_softplus_name_, fused_attrs);

    fused_softplus({&res.Tensor("x")}, {&res.Tensor("act_out")});
  }
};

class SoftplusClipFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string softplus_name_;
  std::string fused_softplus_name_;
  uint32_t benefit_;
  bool inplace_;

 public:
<<<<<<< HEAD
  SoftplusClipFusePattern(std::string softplus_name,
                          std::string fused_softplus_name,
                          uint32_t benefit)
      : softplus_name_(std::move(softplus_name)),
        fused_softplus_name_(std::move(fused_softplus_name)),
        benefit_(benefit) {}
=======
  SoftplusClipFusePattern(const std::string &softplus_name,
                          const std::string &fused_softplus_name,
                          uint32_t benefit,
                          bool inplace)
      : softplus_name_(softplus_name),
        fused_softplus_name_(fused_softplus_name),
        benefit_(benefit),
        inplace_(inplace) {}
>>>>>>> 337f0d225e59843df33c4ca189ae312b8c36a4f3

  std::string name() const override { return "SoftplusActivationFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &softplus = pat.Op(
        softplus_name_,
        {{"beta", pat.Attr("beta")}, {"threshold", pat.Attr("threshold")}});

    const auto &full1 =
        pat.Op(paddle::dialect::FullOp::name(),
               {{"shape", pat.Attr("shape1")}, {"value", pat.Attr("value1")}});
    const auto &full2 =
        pat.Op(paddle::dialect::FullOp::name(),
               {{"shape", pat.Attr("shape2")}, {"value", pat.Attr("value2")}});
    pat.Tensor("min") = full1();
    pat.Tensor("max") = full2();

    const auto &act = inplace_ ? pat.Op(paddle::dialect::Clip_Op::name())
                               : pat.Op(paddle::dialect::ClipOp::name());
    softplus({&pat.Tensor("x")}, {&pat.Tensor("Out")});

    pat.Tensor("act_out") =
        act(pat.Tensor("Out"), pat.Tensor("min"), pat.Tensor("max"));

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fuse_alpha = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("value1");
        });
    const auto &fuse_beta = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("value2");
        });

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"beta", pat.Attr("beta")},
        {"threshold", pat.Attr("threshold")},
        {"fuse_activation", res.StrAttr("clip")},
        {"fuse_alpha", fuse_alpha},
        {"fuse_beta", fuse_beta}};

    const auto &fused_softplus = res.Op(fused_softplus_name_, fused_attrs);

    fused_softplus({&res.Tensor("x")}, {&res.Tensor("act_out")});
  }
};

class SoftplusActivationFusePass : public pir::PatternRewritePass {
 public:
  SoftplusActivationFusePass()
      : pir::PatternRewritePass("softplus_activation_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    int benefit_idx = 1;
    // There is no pattern for "fused_softplus + activation" since currently no
    // pass will output fused_softplus. We will add fused patterns when such
    // pass exists.
    for (auto act_op : act_ops) {
      ps.Add(paddle::drr::Create<SoftplusActivationFusePattern>(
          context,
          paddle::dialect::SoftplusOp::name(),
          paddle::onednn::dialect::FusedSoftplusOp::name(),
          benefit_idx,
          act_op));
      benefit_idx++;
    }
    ps.Add(paddle::drr::Create<SoftplusGeluTanhFusePattern>(
        context,
        paddle::dialect::SoftplusOp::name(),
        paddle::onednn::dialect::FusedSoftplusOp::name(),
        benefit_idx++));
    ps.Add(paddle::drr::Create<SoftplusClipFusePattern>(
        context,
        paddle::dialect::SoftplusOp::name(),
        paddle::onednn::dialect::FusedSoftplusOp::name(),
        benefit_idx++,
        0));
    ps.Add(paddle::drr::Create<SoftplusClipFusePattern>(
        context,
        paddle::dialect::SoftplusOp::name(),
        paddle::onednn::dialect::FusedSoftplusOp::name(),
        benefit_idx++,
        1));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateSoftplusActivationFusePass() {
  // pd_op.softplus + pd_op.relu(act) -> onednn_op.softplus
  return std::make_unique<SoftplusActivationFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(softplus_activation_fuse_pass, SoftplusActivationFusePass);
