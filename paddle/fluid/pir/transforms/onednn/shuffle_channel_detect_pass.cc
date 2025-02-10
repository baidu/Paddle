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

#include "paddle/fluid/pir/transforms/onednn/shuffle_channel_detect_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class ShuffleChannelDetectPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fused_name_;
  uint32_t benefit_;

 public:
  ShuffleChannelDetectPattern(std::string fused_name, uint32_t benefit)
      : fused_name_(fused_name), benefit_(benefit) {}

  std::string name() const override { return "ShuffleChannelDetectPattern"; }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &full_int_array_0 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_0")}});
    pat.Tensor("shape_0") = full_int_array_0();

    const auto &reshape_0 = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape_0({&pat.Tensor("x"), &pat.Tensor("shape_0")},
              {&pat.Tensor("reshape_0_out")});

    const auto &transpose = pat.Op(paddle::dialect::TransposeOp::name(),
                                   {{"perm", pat.Attr("perm")}});
    pat.Tensor("transpose_out") = transpose(pat.Tensor("reshape_0_out"));

    const auto &full_int_array_1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("int_array_1")}});
    pat.Tensor("shape_1") = full_int_array_1();

    const auto &reshape = pat.Op(paddle::dialect::ReshapeOp::name());
    reshape({&pat.Tensor("transpose_out"), &pat.Tensor("shape_1")},
            {&pat.Tensor("out")});

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto shape_0 = match_ctx.Attr<std::vector<int64_t>>("int_array_0");
      auto trans_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("transpose_out"));
      auto shape_1 = match_ctx.Attr<std::vector<int64_t>>("int_array_1");
      auto perm = match_ctx.Attr<std::vector<int>>("perm");
      // Currently only support 4D shuffle_channel
      if (x_shape.size() != 4 || shape_0.size() != 5 || shape_1.size() != 4 ||
          perm.size() != 5) {
        return false;
      }
      if (perm[0] != 0 || perm[3] != 3 || perm[4] != 4) return false;

      int64_t unk_dim = -1;
      bool unk_flag = false;
      bool all_positive = std::all_of(
          x_shape.cbegin(), x_shape.cend(), [](int64_t i) { return i > 0; });
      // There couldn't be more than 1 unknown dim in "shape" attr of reshape.
      // Besides, when unknown dim is not on idx_0(BS) & not all positive dim in
      // input shape, there is no enough info to calculate full dims of reshape
      for (size_t i = 0; i < shape_0.size(); i++) {
        if (!unk_flag) {
          if (shape_0[i] == unk_dim) {
            if (i != 0 && !all_positive) return false;
            unk_flag = true;
          }
        } else {
          if (shape_0[i] == unk_dim) return false;
        }
      }
      unk_flag = false;
      all_positive = std::all_of(trans_shape.cbegin(),
                                 trans_shape.cend(),
                                 [](int64_t i) { return i > 0; });
      for (size_t j = 0; j < shape_1.size(); j++) {
        if (!unk_flag) {
          if (shape_1[j] == unk_dim) {
            if (j != 0 && !all_positive) return false;
            unk_flag = true;
          }
        } else {
          if (shape_1[j] == unk_dim) return false;
        }
      }

      return true;
    });

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto trans_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("transpose_out"));
      auto shape_0 = match_ctx.Attr<std::vector<int64_t>>("int_array_0");
      auto shape_1 = match_ctx.Attr<std::vector<int64_t>>("int_array_1");
      auto perm = match_ctx.Attr<std::vector<int>>("perm");
      int64_t unk_dim = -1;
      int64_t copy_dim = 0;
      for (size_t i = 0; i < shape_0.size(); i++) {
        if (shape_0[i] == copy_dim) {
          shape_0[i] = x_shape[i];
        }
        if (shape_0[i] == unk_dim && i != 0) {
          shape_0[i] = std::accumulate(x_shape.begin(),
                                       x_shape.end(),
                                       static_cast<int64_t>(1),
                                       std::multiplies<int64_t>()) /  // NOLINT
                       std::accumulate(shape_0.begin(),
                                       shape_0.end(),
                                       static_cast<int64_t>(-1),
                                       std::multiplies<int64_t>());
        }
      }

      for (size_t j = 0; j < shape_1.size(); j++) {
        if (shape_1[j] == copy_dim) {
          shape_1[j] = trans_shape[j];
        }
        if (shape_1[j] == unk_dim && j != 0) {
          shape_1[j] = std::accumulate(trans_shape.begin(),
                                       trans_shape.end(),
                                       static_cast<int64_t>(1),
                                       std::multiplies<int64_t>()) /  // NOLINT
                       std::accumulate(shape_1.begin(),
                                       shape_1.end(),
                                       static_cast<int64_t>(-1),
                                       std::multiplies<int64_t>());
        }
      }

      if (shape_1[1] != shape_0[2] * shape_0[1]) return false;
      if (!(shape_0[1] == 1 || shape_0[2] == 1)) {
        if (perm[1] != 2 || perm[2] != 1) return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    std::unordered_map<std::string, paddle::drr::Attribute> fused_attrs{};

    const auto &group_attr =
        res.ComputeAttr([=](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          auto trans_shape =
              pir::GetShapeFromValue(match_ctx.Tensor("transpose_out"));
          auto shape_0 = match_ctx.Attr<std::vector<int64_t>>("int_array_0");
          auto shape_1 = match_ctx.Attr<std::vector<int64_t>>("int_array_1");
          auto perm = match_ctx.Attr<std::vector<int>>("perm");
          int64_t unk_dim = -1;
          int64_t copy_dim = 0;
          for (size_t i = 0; i < shape_0.size(); i++) {
            if (shape_0[i] == copy_dim) {
              shape_0[i] = x_shape[i];
            }
            if (shape_0[i] == unk_dim && i != 0) {
              shape_0[i] =
                  std::accumulate(x_shape.begin(),
                                  x_shape.end(),
                                  static_cast<int64_t>(1),
                                  std::multiplies<int64_t>()) /  // NOLINT
                  std::accumulate(shape_0.begin(),
                                  shape_0.end(),
                                  static_cast<int64_t>(-1),
                                  std::multiplies<int64_t>());
            }
          }

          for (size_t j = 0; j < shape_1.size(); j++) {
            if (shape_1[j] == copy_dim) {
              shape_1[j] = trans_shape[j];
            }
            if (shape_1[j] == unk_dim && j != 0) {
              shape_1[j] =
                  std::accumulate(trans_shape.begin(),
                                  trans_shape.end(),
                                  static_cast<int64_t>(1),
                                  std::multiplies<int64_t>()) /  // NOLINT
                  std::accumulate(shape_1.begin(),
                                  shape_1.end(),
                                  static_cast<int64_t>(-1),
                                  std::multiplies<int64_t>());
            }
          }

          auto group = shape_1[1] / shape_0[2];
          return group;
        });

    fused_attrs.emplace("group", group_attr);

    const auto &shuffle_channel = res.Op(fused_name_, fused_attrs);

    shuffle_channel({&res.Tensor("x")}, {&res.Tensor("out")});
  }
};

class ShuffleChannelDetectPass : public pir::PatternRewritePass {
 public:
  ShuffleChannelDetectPass()
      : pir::PatternRewritePass("shuffle_channel_detect_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ShuffleChannelDetectPattern>(
        context, paddle::onednn::dialect::ShuffleChannelOp::name(), 1));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateShuffleChannelDetectPass() {
  // pd_op.reshape + pd_op.transpose + pd_op.reshape ->
  // onednn_op.shuffle_channel
  return std::make_unique<ShuffleChannelDetectPass>();
}
}  // namespace pir

REGISTER_IR_PASS(shuffle_channel_detect_pass, ShuffleChannelDetectPass);
