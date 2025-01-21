// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/gpu/conv2d_add_fuse_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class Conv2dAddFusePattern : public paddle::drr::DrrPatternBase {
 private:
  bool cutlass_pattern_;
  int sm_version_;

 public:
  static const int CUTLASS_NHWC_ALIGNMENT = 8;
  std::string name() const override { return "Conv2dAddFusePattern"; }
  uint32_t benefit() const override { return cutlass_pattern_ ? 2 : 1; }
  Conv2dAddFusePattern(bool cutlass_pattern, int sm_version)
      : cutlass_pattern_(cutlass_pattern), sm_version_(sm_version) {}
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
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    pat.Tensor("add_out") = add(pat.Tensor("conv2d_out"), pat.Tensor("bias"));
    pat.AddConstraint([this](
                          const paddle::drr::MatchContext &match_ctx) -> bool {
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      auto add_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto bias_ = 1;
      auto add_input_ = 1;
      for (auto &bias_dim : bias_shape) {
        bias_ *= bias_dim;
      }
      for (auto &add_input_dim : add_input_shape) {
        add_input_ *= add_input_dim;
      }
      if (bias_ == add_input_) {
        return false;
      }
      if (!pir::ValueIsPersistable(match_ctx.Tensor("bias"))) {
        return false;
      }
      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      auto groups = match_ctx.Attr<int>("groups");
      auto filter_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("filter"));
      if (!cutlass_pattern_) {
        if (!(filter_dtype.isa<pir::Float16Type>() ||
              filter_dtype.isa<pir::Float32Type>() ||
              filter_dtype.isa<pir::Float64Type>())) {
          return false;
        }
        if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
            padding_algorithm != "VALID") {
          return false;
        }
        if (groups < 1) {
          return false;
        }
        auto data_format = match_ctx.Attr<std::string>("data_format");
        if (data_format != "NCHW" && data_format != "AnyLayout") {
          return false;
        }
      } else {
        auto filter_shape = pir::GetShapeFromValue(match_ctx.Tensor("filter"));
        auto strides_shape = match_ctx.Attr<std::vector<int>>("strides");
        auto dilations_shape = match_ctx.Attr<std::vector<int>>("dilations");
        int stride_h = strides_shape[0];
        int stride_w = strides_shape[1];
        int dilation_h = dilations_shape[0];
        int dilation_w = dilations_shape[1];
        int oc = filter_shape[0];
        int kc = filter_shape[1];
        int kh = filter_shape[2];
        int kw = filter_shape[3];
        if (sm_version_ < 80 && !filter_dtype.isa<pir::Float16Type>()) {
          return false;
        }
        if (!(filter_dtype.isa<pir::Float16Type>() ||
              filter_dtype.isa<pir::Float32Type>() ||
              filter_dtype.isa<pir::BFloat16Type>())) {
          return false;
        }
        if (padding_algorithm != "EXPLICIT") {
          return false;
        }
        int ic = kc * groups;
        if (groups == 1) {
          if (oc % CUTLASS_NHWC_ALIGNMENT != 0 ||
              ic % CUTLASS_NHWC_ALIGNMENT != 0) {
            return false;
          }
        } else if (groups == ic && ic == oc) {
          if (!(kh == 3 && kw == 3) || (kh == 5 && kw == 5)) {
            return false;
          }
          if (!(stride_h == 1 || stride_h == 2)) {
            return false;
          }
          if (stride_h != stride_w) {
            return false;
          }
          if (dilation_h != 1) {
            return false;
          }
          if (dilation_w != 1) {
            return false;
          }
          if (ic % 8 != 0) {
            return false;
          }
        } else {
          return false;
        }
      }
      return true;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &force_backend_runtime_attr = res.ComputeAttr(
        [this](const paddle::drr::MatchContext &match_ctx) -> std::string {
          return cutlass_pattern_ ? "gpu" : "gpudnn";
        });
    const auto &fused_conv2d_add_act = res.Op(
        paddle::dialect::FusedConv2dAddActOp::name(),
        {{
            {"strides", pat.Attr("strides")},
            {"paddings", pat.Attr("paddings")},
            {"padding_algorithm", pat.Attr("padding_algorithm")},
            {"dilations", pat.Attr("dilations")},
            {"groups", pat.Attr("groups")},
            {"data_format", pat.Attr("data_format")},
            {"activation", res.StrAttr("identity")},
            {"split_channels", res.VectorInt32Attr({})},
            {"exhaustive_search", res.BoolAttr(false)},
            {"workspace_size_MB", res.Int32Attr(32)},
            {"fuse_alpha", res.Float32Attr(0.0f)},
        }},
        {{{paddle::dialect::kForceBackendAttr, force_backend_runtime_attr}}});

    fused_conv2d_add_act({&res.Tensor("input"),
                          &res.Tensor("filter"),
                          &res.Tensor("bias"),
                          &res.InputNoneTensor()},
                         {&res.Tensor("add_out"), &res.OutputNoneTensor()});
  }
};

class Conv2dAddFusePass : public pir::PatternRewritePass {
 public:
  Conv2dAddFusePass() : pir::PatternRewritePass("conv2d_add_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    // cutlass related
    const std::unordered_set<int> cutlass_sm = {
        75,
        80,
        85,
        86,
    };
    bool use_cutlass = false;
    if (Has(std::string("use_cutlass"))) {
      use_cutlass = Get<bool>(std::string("use_cutlass"));
    }

    int sm_version = 0;
#ifdef PADDLE_WITH_CUDA
    sm_version = paddle::platform::GetGPUComputeCapability(
        paddle::platform::GetCurrentDeviceId());
#endif

    bool cutlass_pattern = false;
    if (use_cutlass && cutlass_sm.count(sm_version)) {
#if defined(PADDLE_WITH_CUTLASS)
      cutlass_pattern = true;
#endif
    }
    ps.Add(paddle::drr::Create<Conv2dAddFusePattern>(context, false, 0));
    if (cutlass_pattern) {
      ps.Add(
          paddle::drr::Create<Conv2dAddFusePattern>(context, true, sm_version));
    }
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateConv2dAddFusePass() {
  return std::make_unique<Conv2dAddFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(conv2d_add_fuse_pass, Conv2dAddFusePass);
