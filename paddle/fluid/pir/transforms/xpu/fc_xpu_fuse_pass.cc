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

#include "paddle/fluid/pir/transforms/xpu/fc_xpu_fuse_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

int ConvertActivationType(const std::string &act_type) {
  if (act_type == "") {
    return static_cast<int>(xpu::Activation_t::LINEAR);
  } else if (act_type == "relu") {
    return static_cast<int>(xpu::Activation_t::RELU);
  } else if (act_type == "sigmoid") {
    return static_cast<int>(xpu::Activation_t::SIGMOID);
  } else if (act_type == "tanh") {
    return static_cast<int>(xpu::Activation_t::TANH);
  } else if (act_type == "gelu") {
    return static_cast<int>(xpu::Activation_t::GELU);
  } else if (act_type == "leaky_relu") {
    return static_cast<int>(xpu::Activation_t::LEAKY_RELU);
  } else if (act_type == "exp") {
    return static_cast<int>(xpu::Activation_t::EXP);
  } else if (act_type == "hard_swish") {
    return static_cast<int>(xpu::Activation_t::HARD_SWISH);
  } else if (act_type == "hard_sigmoid") {
    return static_cast<int>(xpu::Activation_t::HARD_SIGMOID);
  } else if (act_type == "swish") {
    return static_cast<int>(xpu::Activation_t::SWISH);
  } else if (act_type == "relu6") {
    return static_cast<int>(xpu::Activation_t::RELU6);
  } else if (act_type == "swish_glu") {
    return static_cast<int>(xpu::Activation_t::SWISH_GLU);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Not support convert activation_type(%s).", act_type));
  }
  return -1;
}

/*
fuse malmul + add + act to fc_xpu
For example:
graph:

                  x       w
                    \   /
                      |
                     mul
                      |
                      |
           bias ---  add
                      |
                      |
                     act
                      |
                      |
                    output
------------------------------------------------------
After the pass is applied:
                   x      w
                     \  /
                      |
            bias--- fc_xpu
                      |
                      |
                    Output
*/
class FcXpuFuseAddActPattern : public paddle::drr::DrrPatternBase {
 private:
  bool transpose_w_;

 public:
  explicit FcXpuFuseAddActPattern(bool transpose_w)
      : transpose_w_(transpose_w) {}
  std::string name() const override { return "FcXpuFuseAddActPattern"; }
  uint32_t benefit() const override { return 4; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &mul = pat.Op(paddle::dialect::MatmulOp::name(),
                             {{"transpose_x", pat.Attr("transpose_x")},
                              {"transpose_y", pat.Attr("transpose_y")}});
    mul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("mul_out")});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    add({&pat.Tensor("mul_out"), &pat.Tensor("bias")},
        {&pat.Tensor("add_out")});
    const auto &swiglu = pat.Op(paddle::dialect::SwigluOp::name());
    swiglu({&pat.Tensor("add_out"), &pat.InputNoneTensor()},
           {&pat.Tensor("act_out")});

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      if (transpose_w_ != match_ctx.Attr<bool>("transpose_y")) {
        return false;
      }
      return (w_shape.size() == 2 && x_shape.size() >= 2 &&
              bias_shape.size() == 1);
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return x_shape.size() - 1;
        });

    if (!transpose_w_) {
      // prepare weight, transpose it if necessary
      const auto &perm_attr = res.ComputeAttr(
          [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
            auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
            if (w_shape.size() == 2) {
              return {1, 0};
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "Not support convert w_shape.size()(%d).", w_shape.size()));
            }
          });
      const auto &transpose_op =
          res.Op(paddle::dialect::TransposeOp::name(), {{"perm", perm_attr}});
      res.Tensor("w_trans") = transpose_op(res.Tensor("w"));
      VLOG(3) << "transpose weight for fc_xpu op";
    }

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
          // 目前仅支持以下几种非量化的情况
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else if (x_dtype.isa<pir::Float16Type>()) {
            return phi::DataType::FLOAT16;
          } else if (x_dtype.isa<pir::BFloat16Type>()) {
            return phi::DataType::BFLOAT16;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });
    // only support float32 bias now
    const auto &cast_op = res.Op(paddle::dialect::CastOp::name(),
                                 {{"dtype", res.DataTypeAttr("float32")}});
    res.Tensor("bias_fp32") = cast_op(res.Tensor("bias"));

    const auto &fc_xpu = res.Op(
        paddle::dialect::FcXpuOp::name(),
        {{
            {"in_num_col_dims", in_num_col_dims_attr},
            {"transpose_x", pat.Attr("transpose_x")},
            {"alpha", res.Float32Attr(1.0f)},
            {"beta", res.Float32Attr(0.f)},
            {"act_type", res.Int32Attr(ConvertActivationType("swish_glu"))},
            {"act_alpha", res.Float32Attr(0.0f)},
            {"out_dtype", out_dtype_attr},
        }});
    fc_xpu(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            transpose_w_ ? &res.Tensor("w") : &res.Tensor("w_trans"),
            &res.InputNoneTensor(),
            &res.Tensor("bias_fp32"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("act_out"), &res.Tensor("out_max")});
  }
};

/*
fuse malmul + act to fc_xpu
For example:
graph:

                  x       w
                    \   /
                      |
                     mul
                      |
                      |
                     act
                      |
                      |
                    output
------------------------------------------------------
After the pass is applied:
                   x      w
                     \  /
                      |
            bias--- fc_xpu
                      |
                      |
                    Output
*/

class FcXpuFuseActPattern : public paddle::drr::DrrPatternBase {
 private:
  bool transpose_w_;

 public:
  explicit FcXpuFuseActPattern(bool transpose_w) : transpose_w_(transpose_w) {}
  std::string name() const override { return "FcXpuFuseActPattern"; }
  uint32_t benefit() const override { return 3; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &mul = pat.Op(paddle::dialect::MatmulOp::name(),
                             {{"transpose_x", pat.Attr("transpose_x")},
                              {"transpose_y", pat.Attr("transpose_y")}});
    mul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("mul_out")});

    const auto &swiglu = pat.Op(paddle::dialect::SwigluOp::name());
    swiglu({&pat.Tensor("mul_out"), &pat.InputNoneTensor()},
           {&pat.Tensor("act_out")});

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      if (transpose_w_ != match_ctx.Attr<bool>("transpose_y")) {
        return false;
      }
      return (w_shape.size() == 2 && x_shape.size() >= 2);
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return x_shape.size() - 1;
        });

    if (!transpose_w_) {
      // prepare weight, transpose it if necessary
      const auto &perm_attr = res.ComputeAttr(
          [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
            auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
            if (w_shape.size() == 2) {
              return {1, 0};
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "Not support convert w_shape.size()(%d).", w_shape.size()));
            }
          });
      const auto &transpose_op =
          res.Op(paddle::dialect::TransposeOp::name(), {{"perm", perm_attr}});
      res.Tensor("w_trans") = transpose_op(res.Tensor("w"));
      VLOG(3) << "transpose weight for fc_xpu op";
    }

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
          // 目前仅支持以下几种非量化的情况
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else if (x_dtype.isa<pir::Float16Type>()) {
            return phi::DataType::FLOAT16;
          } else if (x_dtype.isa<pir::BFloat16Type>()) {
            return phi::DataType::BFLOAT16;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    const auto &fc_xpu = res.Op(
        paddle::dialect::FcXpuOp::name(),
        {{
            {"in_num_col_dims", in_num_col_dims_attr},
            {"transpose_x", pat.Attr("transpose_x")},
            {"alpha", res.Float32Attr(1.0f)},
            {"beta", res.Float32Attr(0.f)},
            {"act_type", res.Int32Attr(ConvertActivationType("swish_glu"))},
            {"act_alpha", res.Float32Attr(0.0f)},
            {"out_dtype", out_dtype_attr},
        }});
    fc_xpu(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            transpose_w_ ? &res.Tensor("w") : &res.Tensor("w_trans"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("act_out"), &res.Tensor("out_max")});
  }
};

/*
fuse malmul + add to fc_xpu
For example:
graph:

                  x       w
                    \   /
                      |
                     mul
                      |
                      |
           bias ---  add
                      |
                      |
                    output
------------------------------------------------------
After the pass is applied:
                   x      w
                     \  /
                      |
            bias--- fc_xpu
                      |
                      |
                    Output
*/
class FcXpuFuseAddPattern : public paddle::drr::DrrPatternBase {
 private:
  bool transpose_w_;

 public:
  explicit FcXpuFuseAddPattern(bool transpose_w) : transpose_w_(transpose_w) {}
  std::string name() const override { return "FcXpuFuseAddPattern"; }
  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &mul = pat.Op(paddle::dialect::MatmulOp::name(),
                             {{"transpose_x", pat.Attr("transpose_x")},
                              {"transpose_y", pat.Attr("transpose_y")}});
    mul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("mul_out")});

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    add({&pat.Tensor("mul_out"), &pat.Tensor("bias")},
        {&pat.Tensor("add_out")});

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      if (transpose_w_ != match_ctx.Attr<bool>("transpose_y")) {
        return false;
      }
      return (w_shape.size() == 2 && x_shape.size() >= 2 &&
              bias_shape.size() == 1);
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return x_shape.size() - 1;
        });

    if (!transpose_w_) {
      // prepare weight, transpose it if necessary
      const auto &perm_attr = res.ComputeAttr(
          [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
            auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
            if (w_shape.size() == 2) {
              return {1, 0};
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "Not support convert w_shape.size()(%d).", w_shape.size()));
            }
          });
      const auto &transpose_op =
          res.Op(paddle::dialect::TransposeOp::name(), {{"perm", perm_attr}});
      res.Tensor("w_trans") = transpose_op(res.Tensor("w"));
      VLOG(3) << "transpose weight for fc_xpu op";
    }

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
          // 目前仅支持以下几种非量化的情况
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else if (x_dtype.isa<pir::Float16Type>()) {
            return phi::DataType::FLOAT16;
          } else if (x_dtype.isa<pir::BFloat16Type>()) {
            return phi::DataType::BFLOAT16;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });
    // only support float32 bias now
    const auto &cast_op = res.Op(paddle::dialect::CastOp::name(),
                                 {{"dtype", res.DataTypeAttr("float32")}});
    res.Tensor("bias_fp32") = cast_op(res.Tensor("bias"));

    const auto &fc_xpu =
        res.Op(paddle::dialect::FcXpuOp::name(),
               {{
                   {"in_num_col_dims", in_num_col_dims_attr},
                   {"transpose_x", pat.Attr("transpose_x")},
                   {"alpha", res.Float32Attr(1.0f)},
                   {"beta", res.Float32Attr(0.f)},
                   {"act_type", res.Int32Attr(ConvertActivationType(""))},
                   {"act_alpha", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    fc_xpu(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            transpose_w_ ? &res.Tensor("w") : &res.Tensor("w_trans"),
            &res.InputNoneTensor(),
            &res.Tensor("bias_fp32"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("add_out"), &res.Tensor("out_max")});
  }
};

/*
fuse malmul + add to fc_xpu
For example:
graph:

                  x       w
                    \   /
                      |
                     mul
                      |
                      |
                    output
------------------------------------------------------
After the pass is applied:
                   x      w
                     \  /
                      |
                    fc_xpu
                      |
                      |
                    Output
*/
class FcXpuOnlyMapPattern : public paddle::drr::DrrPatternBase {
 private:
  bool transpose_w_;

 public:
  explicit FcXpuOnlyMapPattern(bool transpose_w) : transpose_w_(transpose_w) {}
  std::string name() const override { return "FcXpuOnlyMapPattern"; }
  uint32_t benefit() const override { return 1; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &mul = pat.Op(paddle::dialect::MatmulOp::name(),
                             {{"transpose_x", pat.Attr("transpose_x")},
                              {"transpose_y", pat.Attr("transpose_y")}});
    mul({&pat.Tensor("x"), &pat.Tensor("w")}, {&pat.Tensor("mul_out")});

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      if (transpose_w_ != match_ctx.Attr<bool>("transpose_y")) {
        return false;
      }
      return (w_shape.size() == 2 && x_shape.size() >= 2);
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &in_num_col_dims_attr =
        res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx) -> int {
          auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
          return x_shape.size() - 1;
        });

    if (!transpose_w_) {
      // prepare weight, transpose it if necessary
      const auto &perm_attr = res.ComputeAttr(
          [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
            auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
            if (w_shape.size() == 2) {
              return {1, 0};
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "Not support convert w_shape.size()(%d).", w_shape.size()));
            }
          });
      const auto &transpose_op =
          res.Op(paddle::dialect::TransposeOp::name(), {{"perm", perm_attr}});
      res.Tensor("w_trans") = transpose_op(res.Tensor("w"));
      VLOG(3) << "transpose weight for fc_xpu op";
    }

    const auto &out_dtype_attr = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
          // 目前仅支持以下几种非量化的情况
          if (x_dtype.isa<pir::Float32Type>()) {
            return phi::DataType::FLOAT32;
          } else if (x_dtype.isa<pir::Float16Type>()) {
            return phi::DataType::FLOAT16;
          } else if (x_dtype.isa<pir::BFloat16Type>()) {
            return phi::DataType::BFLOAT16;
          } else {
            return phi::DataType::UNDEFINED;
          }
        });

    const auto &fc_xpu =
        res.Op(paddle::dialect::FcXpuOp::name(),
               {{
                   {"in_num_col_dims", in_num_col_dims_attr},
                   {"transpose_x", pat.Attr("transpose_x")},
                   {"alpha", res.Float32Attr(1.0f)},
                   {"beta", res.Float32Attr(0.f)},
                   {"act_type", res.Int32Attr(ConvertActivationType(""))},
                   {"act_alpha", res.Float32Attr(0.0f)},
                   {"out_dtype", out_dtype_attr},
               }});
    fc_xpu(
        {
            &res.Tensor("x"),
            &res.InputNoneTensor(),
            transpose_w_ ? &res.Tensor("w") : &res.Tensor("w_trans"),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
            &res.InputNoneTensor(),
        },
        {&res.Tensor("mul_out"), &res.Tensor("out_max")});
  }
};

class FCXpuFusePass : public pir::PatternRewritePass {
 public:
  FCXpuFusePass() : pir::PatternRewritePass("fc_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    unsigned char enable_fc_xpu_pattern = 0xff;
    const char *value = std::getenv("XPU_PADDLE_FC_PATTERN");
    if (value != nullptr) {
      enable_fc_xpu_pattern = std::stoi(value);
      std::cout << "XPU_PADDLE_FC_PATTERN: " << enable_fc_xpu_pattern
                << std::endl;
    }
    if (enable_fc_xpu_pattern & 0x08) {
      ps.Add(paddle::drr::Create<FcXpuFuseAddActPattern>(context,
                                                         false));  // benefit 4
      ps.Add(paddle::drr::Create<FcXpuFuseAddActPattern>(context, true));
    }
    if (enable_fc_xpu_pattern & 0x04) {
      ps.Add(paddle::drr::Create<FcXpuFuseActPattern>(context,
                                                      false));  // benefit 3
      ps.Add(paddle::drr::Create<FcXpuFuseActPattern>(context, true));
    }
    if (enable_fc_xpu_pattern & 0x02) {
      ps.Add(paddle::drr::Create<FcXpuFuseAddPattern>(context,
                                                      false));  // benefit 2
      ps.Add(paddle::drr::Create<FcXpuFuseAddPattern>(context, true));
    }
    if (enable_fc_xpu_pattern & 0x01) {
      ps.Add(paddle::drr::Create<FcXpuOnlyMapPattern>(context,
                                                      false));  // benefit 1
      ps.Add(paddle::drr::Create<FcXpuOnlyMapPattern>(context, true));
    }
    return ps;
  }

  pir::GreedyRewriteConfig InitializeConfig() override {
    pir::GreedyRewriteConfig config;

    config.use_top_down_traversal = false;

    config.max_iterations = 10;
    return config;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFCXpuFusePass() {
  return std::make_unique<FCXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(fc_xpu_fuse_pass, FCXpuFusePass);
