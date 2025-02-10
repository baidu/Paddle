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

#include "paddle/fluid/pir/transforms/onednn/matmul_activation_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

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

class MatmulActivationFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  std::string act_type_;

 public:
  MatmulActivationFusePattern(std::string matmul_name,
                              std::string fused_matmul_name,
                              uint32_t benefit,
                              std::string act_type)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit),
        act_type_(std::move(act_type)) {}

  std::string name() const override { return "MatmulActivationFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

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
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

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
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"matmul_alpha", res.Float32Attr(1.0f)},
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
        {"force_fp32_output", res.BoolAttr(false)}};

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

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
  }
};

class MatmulGeluTanhFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;

 public:
  MatmulGeluTanhFusePattern(std::string matmul_name,
                            std::string fused_matmul_name,
                            uint32_t benefit)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit) {}

  std::string name() const override { return "MatmulActivationFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

    const auto &act = pat.Op(paddle::dialect::GeluOp::name(),
                             {{"approximate", pat.Attr("approximate")}});
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

    pat.Tensor("act_out") = act(pat.Tensor("Out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto result_gelu = match_ctx.Attr<bool>("approximate");
      if (!result_gelu) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"matmul_alpha", res.Float32Attr(1.0f)},
        {"fuse_activation", res.StrAttr("gelu_tanh")},
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
        {"force_fp32_output", res.BoolAttr(false)}};

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
  }
};

class MatmulClipFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool inplace_;

 public:
<<<<<<< HEAD
  MatmulClipFusePattern(std::string matmul_name,
                        std::string fused_matmul_name,
                        uint32_t benefit)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit) {}
=======
  MatmulClipFusePattern(const std::string &matmul_name,
                        const std::string &fused_matmul_name,
                        uint32_t benefit,
                        bool inplace)
      : matmul_name_(matmul_name),
        fused_matmul_name_(fused_matmul_name),
        benefit_(benefit),
        inplace_(inplace) {}
>>>>>>> 337f0d225e59843df33c4ca189ae312b8c36a4f3

  std::string name() const override { return "MatmulActivationFusePattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(matmul_name_,
                                {{"transpose_x", pat.Attr("transpose_x")},
                                 {"transpose_y", pat.Attr("transpose_y")}});

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
    matmul({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});

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
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"matmul_alpha", res.Float32Attr(1.0f)},
        {"fuse_activation", res.StrAttr("clip")},
        {"fuse_alpha", fuse_alpha},
        {"fuse_beta", fuse_beta},
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
        {"force_fp32_output", res.BoolAttr(false)}};

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
  }
};

class FusedMatmulActivationFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  std::string act_type_;

 public:
  FusedMatmulActivationFusePattern(std::string matmul_name,
                                   std::string fused_matmul_name,
                                   uint32_t benefit,
                                   std::string act_type)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit),
        act_type_(std::move(act_type)) {}

  std::string name() const override {
    return "FusedMatmulActivationFusePattern";
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
    matmul({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("residual")},
           {&pat.Tensor("Out")});

    pat.Tensor("act_out") = act(pat.Tensor("Out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto act_type = match_ctx.Attr<std::string>("fuse_activation");
      if (act_type != "") return false;
      return true;
    });

    if (act_type_ == paddle::dialect::GeluOp::name()) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto result_gelu = match_ctx.Attr<bool>("approximate");
        if (result_gelu) return false;
        return true;
      });
    }

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"matmul_alpha", pat.Attr("matmul_alpha")},
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
        {"force_fp32_output", pat.Attr("force_fp32_output")}};

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

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("act_out")});
  }
};

class FusedMatmulGeluTanhFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;

 public:
  FusedMatmulGeluTanhFusePattern(std::string matmul_name,
                                 std::string fused_matmul_name,
                                 uint32_t benefit)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit) {}

  std::string name() const override {
    return "FusedMatmulActivationFusePattern";
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

    const auto &act = pat.Op(paddle::dialect::GeluOp::name(),
                             {{"approximate", pat.Attr("approximate")}});
    matmul({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("residual")},
           {&pat.Tensor("Out")});

    pat.Tensor("act_out") = act(pat.Tensor("Out"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto act_type = match_ctx.Attr<std::string>("fuse_activation");
      if (act_type != "") return false;
      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto result_gelu = match_ctx.Attr<bool>("approximate");
      if (!result_gelu) return false;
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"matmul_alpha", pat.Attr("matmul_alpha")},
        {"fuse_activation", res.StrAttr("gelu_tanh")},
        {"fuse_alpha", res.Float32Attr(0.0f)},
        {"fuse_beta", res.Float32Attr(0.0f)},
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
        {"force_fp32_output", pat.Attr("force_fp32_output")}};

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("act_out")});
  }
};

class FusedMatmulClipFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string matmul_name_;
  std::string fused_matmul_name_;
  uint32_t benefit_;
  bool inplace_;

 public:
<<<<<<< HEAD
  FusedMatmulClipFusePattern(std::string matmul_name,
                             std::string fused_matmul_name,
                             uint32_t benefit)
      : matmul_name_(std::move(matmul_name)),
        fused_matmul_name_(std::move(fused_matmul_name)),
        benefit_(benefit) {}
=======
  FusedMatmulClipFusePattern(const std::string &matmul_name,
                             const std::string &fused_matmul_name,
                             uint32_t benefit,
                             bool inplace)
      : matmul_name_(matmul_name),
        fused_matmul_name_(fused_matmul_name),
        benefit_(benefit),
        inplace_(inplace) {}
>>>>>>> 337f0d225e59843df33c4ca189ae312b8c36a4f3

  std::string name() const override {
    return "FusedMatmulActivationFusePattern";
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
    matmul({&pat.Tensor("X"), &pat.Tensor("Y"), &pat.Tensor("residual")},
           {&pat.Tensor("Out")});

    pat.Tensor("act_out") =
        act(pat.Tensor("Out"), pat.Tensor("min"), pat.Tensor("max"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto act_type = match_ctx.Attr<std::string>("fuse_activation");
      if (act_type != "") return false;
      return true;
    });

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
        {"trans_x", pat.Attr("transpose_x")},
        {"trans_y", pat.Attr("transpose_y")},
        {"matmul_alpha", pat.Attr("matmul_alpha")},
        {"fuse_activation", res.StrAttr("clip")},
        {"fuse_alpha", fuse_alpha},
        {"fuse_beta", fuse_beta},
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
        {"force_fp32_output", pat.Attr("force_fp32_output")}};

    const auto &fused_matmul = res.Op(fused_matmul_name_, fused_attrs);

    fused_matmul({&res.Tensor("X"), &res.Tensor("Y"), &res.Tensor("residual")},
                 {&res.Tensor("act_out")});
  }
};

class MatmulActivationFusePass : public pir::PatternRewritePass {
 public:
  MatmulActivationFusePass()
      : pir::PatternRewritePass("matmul_activation_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    int benefit_idx = 1;
    for (auto act_op : act_ops) {
      ps.Add(paddle::drr::Create<MatmulActivationFusePattern>(
          context,
          paddle::dialect::MatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          act_op));
      benefit_idx++;
    }
    ps.Add(paddle::drr::Create<MatmulGeluTanhFusePattern>(
        context,
        paddle::dialect::MatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++));
    ps.Add(paddle::drr::Create<MatmulClipFusePattern>(
        context,
        paddle::dialect::MatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++,
        0));
    ps.Add(paddle::drr::Create<MatmulClipFusePattern>(
        context,
        paddle::dialect::MatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++,
        1));
    for (auto act_op : act_ops) {
      ps.Add(paddle::drr::Create<FusedMatmulActivationFusePattern>(
          context,
          paddle::onednn::dialect::FusedMatmulOp::name(),
          paddle::onednn::dialect::FusedMatmulOp::name(),
          benefit_idx,
          act_op));
      benefit_idx++;
    }
    ps.Add(paddle::drr::Create<FusedMatmulGeluTanhFusePattern>(
        context,
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++));
    ps.Add(paddle::drr::Create<FusedMatmulClipFusePattern>(
        context,
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++,
        0));
    ps.Add(paddle::drr::Create<FusedMatmulClipFusePattern>(
        context,
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        benefit_idx++,
        1));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMatmulActivationFusePass() {
  // pd_op.matmul + pd_op.relu -> onednn_op.fused_matmul
  // pd_op.matmul + pd_op.add + pd_op.relu(act) ->  onednn_op.fused_matmul +
  // pd_op.relu(act) -> onednn_op.fused_matmul
  return std::make_unique<MatmulActivationFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(matmul_activation_fuse_pass, MatmulActivationFusePass);
