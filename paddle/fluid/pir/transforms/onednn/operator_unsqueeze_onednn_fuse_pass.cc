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

#include "paddle/fluid/pir/transforms/onednn/operator_unsqueeze_onednn_fuse_pass.h"

#include <utility>

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class OperatorUnsqueezeFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string fusable_ops_;
  std::string fused_ops_name_;
  uint32_t benefit_;

 public:
  OperatorUnsqueezeFusePattern(std::string fusable_ops,
                               std::string fused_ops_name,
                               uint32_t benefit)
      : fusable_ops_(std::move(fusable_ops)),
        fused_ops_name_(std::move(fused_ops_name)),
        benefit_(benefit) {}

  std::string name() const override {
    return fusable_ops_ + "UnsqueezeFusePattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (fusable_ops_ == paddle::onednn::dialect::FusedTransposeOp::name()) {
      op_attrs.emplace("axis", pat.Attr("axis"));
      op_attrs.emplace("fused_squeeze2_axes", pat.Attr("fused_squeeze2_axes"));
      op_attrs.emplace("fused_unsqueeze2_axes",
                       pat.Attr("fused_unsqueeze2_axes"));
      op_attrs.emplace("fused_reshape2_shape",
                       pat.Attr("fused_reshape2_shape"));
      op_attrs.emplace("scale", pat.Attr("scale"));
      op_attrs.emplace("shift", pat.Attr("shift"));
      op_attrs.emplace("output_data_type", pat.Attr("output_data_type"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (fusable_ops_ == paddle::dialect::TransposeOp::name()) {
      op_attrs.emplace("perm", pat.Attr("perm"));
    } else if (fusable_ops_ ==
               paddle::onednn::dialect::FusedElementwiseMulOp::name()) {
      op_attrs.emplace("axis", pat.Attr("axis"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_unsqueeze2_axes",
                       pat.Attr("fused_unsqueeze2_axes"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    }

    const auto &op = pat.Op(fusable_ops_, op_attrs);

    if (fusable_ops_ == paddle::dialect::TransposeOp::name() ||
        fusable_ops_ == paddle::onednn::dialect::FusedTransposeOp::name()) {
      op({&pat.Tensor("X")}, {&pat.Tensor("Out")});
    } else {
      op({&pat.Tensor("X"), &pat.Tensor("Y")}, {&pat.Tensor("Out")});
    }
    const auto &unsqueeze = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});

    unsqueeze({&pat.Tensor("Out"), &full_1()}, {&pat.Tensor("Unsqueeze_out")});

    if (fusable_ops_ == paddle::onednn::dialect::FusedTransposeOp::name() ||
        fusable_ops_ ==
            paddle::onednn::dialect::FusedElementwiseMulOp::name()) {
      pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
        auto fused_unsqueeze2_axes =
            match_ctx.Attr<std::vector<int>>("fused_unsqueeze2_axes");
        if (fused_unsqueeze2_axes.size() > 0) {
          // It means that it has been fused and has a value.
          return false;
        }
        return true;
      });
    }

    paddle::drr::ResultPattern res = pat.ResultPattern();
    std::unordered_map<std::string, paddle::drr::Attribute> fused_op_attrs{};
    const auto &fused_unsqueeze2_axes = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("full_1_value");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    if (fusable_ops_ == paddle::onednn::dialect::FusedTransposeOp::name()) {
      fused_op_attrs.emplace("axis", pat.Attr("axis"));
      fused_op_attrs.emplace("fused_squeeze2_axes",
                             pat.Attr("fused_squeeze2_axes"));
      fused_op_attrs.emplace("fused_unsqueeze2_axes", fused_unsqueeze2_axes);
      fused_op_attrs.emplace("fused_reshape2_shape",
                             pat.Attr("fused_reshape2_shape"));
      fused_op_attrs.emplace("scale", pat.Attr("scale"));
      fused_op_attrs.emplace("shift", pat.Attr("shift"));
      fused_op_attrs.emplace("output_data_type", pat.Attr("output_data_type"));
      fused_op_attrs.emplace("data_format", pat.Attr("data_format"));
      fused_op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (fusable_ops_ == paddle::dialect::TransposeOp::name()) {
      fused_op_attrs.emplace("axis", pat.Attr("perm"));
      fused_op_attrs.emplace("fused_squeeze2_axes", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("fused_unsqueeze2_axes", fused_unsqueeze2_axes);
      fused_op_attrs.emplace("fused_reshape2_shape", res.VectorInt32Attr({}));
      fused_op_attrs.emplace("scale", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("shift", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("output_data_type", res.StrAttr("fp32"));
      fused_op_attrs.emplace("data_format", res.StrAttr("AnyLayout"));
      fused_op_attrs.emplace("mkldnn_data_type", res.StrAttr("float32"));

    } else if (fusable_ops_ ==
               paddle::onednn::dialect::FusedElementwiseMulOp::name()) {
      fused_op_attrs.emplace("axis", pat.Attr("axis"));
      fused_op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      fused_op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      fused_op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      fused_op_attrs.emplace("fused_output_scale",
                             pat.Attr("fused_output_scale"));
      fused_op_attrs.emplace("fused_unsqueeze2_axes", fused_unsqueeze2_axes);
      fused_op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      fused_op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      fused_op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    } else {
      // Mul
      fused_op_attrs.emplace("axis", res.Int32Attr(-1));
      fused_op_attrs.emplace("fuse_activation", res.StrAttr(""));
      fused_op_attrs.emplace("fuse_alpha", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("fuse_beta", res.Float32Attr(0.0f));
      fused_op_attrs.emplace("fused_output_scale", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("fused_unsqueeze2_axes", fused_unsqueeze2_axes);
      fused_op_attrs.emplace("scale_x", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("scale_y", res.Float32Attr(1.0f));
      fused_op_attrs.emplace("scale_out", res.Float32Attr(1.0f));
    }

    const auto &fused_op = res.Op(fused_ops_name_, fused_op_attrs);
    if (fusable_ops_ == paddle::dialect::TransposeOp::name() ||
        fusable_ops_ == paddle::onednn::dialect::FusedTransposeOp::name()) {
      fused_op({&res.Tensor("X")}, {&res.Tensor("Unsqueeze_out")});
    } else {
      fused_op({&res.Tensor("X"), &res.Tensor("Y")},
               {&res.Tensor("Unsqueeze_out")});
    }
  }
};

class OperatorUnsqueezeFusePass : public pir::PatternRewritePass {
 public:
  OperatorUnsqueezeFusePass()
      : pir::PatternRewritePass("operator_unsqueeze_onednn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    const std::vector<std::string> fusable_ops{
        paddle::onednn::dialect::FusedTransposeOp::name(),
        paddle::dialect::TransposeOp::name(),
        paddle::onednn::dialect::FusedElementwiseMulOp::name(),
        paddle::dialect::MultiplyOp::name(),
    };

    const std::vector<std::string> fused_ops{
        paddle::onednn::dialect::FusedTransposeOp::name(),
        paddle::onednn::dialect::FusedTransposeOp::name(),
        paddle::onednn::dialect::FusedElementwiseMulOp::name(),
        paddle::onednn::dialect::FusedElementwiseMulOp::name(),
    };
    int benefit_idx = 1;
    int fused = 0;
    for (auto op : fusable_ops) {
      ps.Add(paddle::drr::Create<OperatorUnsqueezeFusePattern>(
          context, op, fused_ops[fused++], benefit_idx));
      benefit_idx++;
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateOperatorUnsqueezeFusePass() {
  return std::make_unique<OperatorUnsqueezeFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(operator_unsqueeze_onednn_fuse_pass,
                 OperatorUnsqueezeFusePass);
