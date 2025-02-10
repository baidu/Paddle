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

#include "paddle/fluid/pir/transforms/onednn/depthwise_conv_onednn_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DepthwiseConvPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string depthwise_conv_name_;

 public:
  explicit DepthwiseConvPattern(std::string conv_name)
      : depthwise_conv_name_(std::move(conv_name)) {}

  std::string name() const override { return "DepthwiseConvPattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &depthwise_conv =
        pat.Op(depthwise_conv_name_,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    depthwise_conv({&pat.Tensor("input"), &pat.Tensor("filter")},
                   {&pat.Tensor("conv_out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      std::set<std::string> padding_algorithm = {"EXPLICIT", "SAME", "VALID"};
      std::set<std::string> data_format = {"NCHW", "NHWC", "AnyLayout"};
      if (padding_algorithm.count(
              match_ctx.Attr<std::string>("padding_algorithm")) == 0 ||
          data_format.count(match_ctx.Attr<std::string>("data_format")) == 0 ||
          match_ctx.Attr<int>("groups") < 1) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &conv2d =
        res.Op(paddle::dialect::Conv2dOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
               }});

    conv2d({&res.Tensor("input"), &res.Tensor("filter")},
           {&res.Tensor("conv_out")});
  }
};

class DepthwiseConvMKLDNNPass : public pir::PatternRewritePass {
 public:
  DepthwiseConvMKLDNNPass()
      : pir::PatternRewritePass("depthwise_conv_onednn_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DepthwiseConvPattern>(
        context, paddle::dialect::DepthwiseConv2dOp::name()));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDepthwiseConvMKLDNNPass() {
  // pd_op.depthwise_conv  -> pd_op.conv2d
  return std::make_unique<DepthwiseConvMKLDNNPass>();
}

}  // namespace pir

REGISTER_IR_PASS(depthwise_conv_onednn_pass, DepthwiseConvMKLDNNPass);
