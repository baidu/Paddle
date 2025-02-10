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

#include "paddle/fluid/pir/transforms/general/map_op_to_another_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class DepthWiseConv2d2Conv2dPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "DepthWiseConv2d2Conv2dPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &depthwise_conv2d_op =
        pat.Op(paddle::dialect::DepthwiseConv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    depthwise_conv2d_op({&pat.Tensor("input"), &pat.Tensor("filter")},
                        {&pat.Tensor("depthwise_conv2d_out")});
    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) -> bool {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 8100
      auto groups = match_ctx.Attr<int>("groups");
      return groups > 1;
#else
      return false;
#endif
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &conv2d =
        res.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    conv2d({&res.Tensor("input"), &res.Tensor("filter")},
           {&res.Tensor("depthwise_conv2d_out")});
  }
};

// flatten_contiguous to reshape
class FlattenContiguousRange2ReshapePattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FlattenContiguousRange2ReshapePattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &flatten_op = pat.Op(paddle::dialect::FlattenOp::name(),
                                    {{"start_axis", pat.Attr("start_axis")},
                                     {"stop_axis", pat.Attr("stop_axis")}});
    flatten_op({&pat.Tensor("flatten_in")},
               {&pat.Tensor("flatten_out"), &pat.Tensor("flatten_xshape")});
    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto shape = pir::GetShapeFromValue(match_ctx.Tensor("flatten_in"));
      size_t shape_size = shape.size();
      int start_axis = match_ctx.Attr<int>("start_axis");
      int stop_axis = match_ctx.Attr<int>("stop_axis");

      if (start_axis == 1 && stop_axis == 3 && shape_size == 4 &&
          shape[2] == 1 && shape[3] == 1) {
        return true;
      } else if (start_axis == 2 && stop_axis == 3 && shape_size == 4 &&
                 shape[2] == 1) {
        return true;
      }
      return false;
    });

    const auto &shape_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int64_t> {
          auto shape = pir::GetShapeFromValue(match_ctx.Tensor("flatten_in"));
          size_t shape_size = shape.size();
          int start_axis = match_ctx.Attr<int>("start_axis");
          int stop_axis = match_ctx.Attr<int>("stop_axis");

          if (start_axis == 1 && stop_axis == 3 && shape_size == 4 &&
              shape[2] == 1 && shape[3] == 1) {
            return {0, -1};
          } else if (start_axis == 2 && stop_axis == 3 && shape_size == 4 &&
                     shape[2] == 1) {
            return {0, 0, -1};
          }
          return shape;
        });
    const auto &reshape_op =
        res.Op(paddle::dialect::ReshapeOp::name(), {{"shape", shape_attr}});
    reshape_op({&res.Tensor("flatten_in")},
               {&res.Tensor("flatten_out"), &res.OutputNoneTensor()});
  }
};

class MapOpToAnotherPass : public pir::PatternRewritePass {
 public:
  MapOpToAnotherPass() : pir::PatternRewritePass("map_op_to_another_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<DepthWiseConv2d2Conv2dPattern>(context));
    ps.Add(paddle::drr::Create<FlattenContiguousRange2ReshapePattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateMapOpToAnotherPass() {
  return std::make_unique<MapOpToAnotherPass>();
}
}  // namespace pir

REGISTER_IR_PASS(map_op_to_another_pass, MapOpToAnotherPass);
