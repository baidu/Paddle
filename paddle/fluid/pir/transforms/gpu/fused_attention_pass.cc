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

#include "paddle/fluid/pir/transforms/gpu/fused_attention_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// cast + mask = SDPAttn
// mask = FlashAttn

// FlashAttn
// 1. scale before matmul
// 2. cast before and after softmax
class FlashAttnPatternQscaleCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FlashAttnPatternQscaleCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // scale before matmul
    const auto &scale_q = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_q_value")}});
    src.Tensor("q_scale_out") =
        scale_q(src.Tensor("q_transpose_out"), full_scale());
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose2_out"));

    // mask
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask"));

    // cast + softmax + cast
    const auto &softmax_cast1 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast1_out") = softmax_cast1(src.Tensor("mask_add_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("softmax_cast1_out"));
    const auto &softmax_cast2 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast2_out") = softmax_cast2(src.Tensor("softmax_out"));
    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.flash_attn",
                                    {{{"dropout", res.Float32Attr(0.0)},
                                      {"causal", res.BoolAttr(false)},
                                      {"return_softmax", res.BoolAttr(false)},
                                      {"is_test", res.BoolAttr(false)},
                                      {"rng_name", res.StrAttr("")}}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                &res.InputNoneTensor(),
                &res.Tensor("mask")},
               {&res.Tensor("out"),
                &res.Tensor("softmax"),
                &res.Tensor("softmax_lse"),
                &res.Tensor("seed_offset")});
  }
};

// FlashAttn
// 1. scale before matmul
// 2. no cast before and after softmax
class FlashAttnPatternQscaleNoCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FlashAttnPatternQscaleNoCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose,
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // scale before matmul
    const auto &scale_q = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_q_value")}});
    src.Tensor("q_scale_out") =
        scale_q(src.Tensor("q_transpose_out"), full_scale());
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose2_out"));

    // mask
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask"));

    // softmax
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.flash_attn",
                                    {{{"dropout", res.Float32Attr(0.0)},
                                      {"causal", res.BoolAttr(false)},
                                      {"return_softmax", res.BoolAttr(false)},
                                      {"is_test", res.BoolAttr(false)},
                                      {"rng_name", res.StrAttr("")}}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                &res.InputNoneTensor(),
                &res.Tensor("mask")},
               {&res.Tensor("out"),
                &res.Tensor("softmax"),
                &res.Tensor("softmax_lse"),
                &res.Tensor("seed_offset")});
  }
};

// FlashAttn
// 1. scale after matmul
// 2. cast before and after softmax
class FlashAttnPatternOutscaleCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FlashAttnPatternOutscaleCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose,
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = qk_matmul(src.Tensor("q_transpose_out"),
                                     src.Tensor("k_transpose2_out"));
    const auto &scale_out = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_out_value")}});
    src.Tensor("qk_scale_out") = scale_out(src.Tensor("qk_out"), full_scale());

    // mask
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_scale_out"), src.Tensor("mask"));

    // cast + softmax + cast
    const auto &softmax_cast1 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast1_out") = softmax_cast1(src.Tensor("mask_add_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("softmax_cast1_out"));
    const auto &softmax_cast2 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast2_out") = softmax_cast2(src.Tensor("softmax_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.flash_attn",
                                    {{{"dropout", res.Float32Attr(0.0)},
                                      {"causal", res.BoolAttr(false)},
                                      {"return_softmax", res.BoolAttr(false)},
                                      {"is_test", res.BoolAttr(false)},
                                      {"rng_name", res.StrAttr("")}}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                &res.InputNoneTensor(),
                &res.Tensor("mask")},
               {&res.Tensor("out"),
                &res.Tensor("softmax"),
                &res.Tensor("softmax_lse"),
                &res.Tensor("seed_offset")});
  }
};

// FlashAttn
// 1. scale after matmul
// 2. no cast before and after softmax
class FlashAttnPatternOutscaleNoCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FlashAttnPatternOutscaleNoCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose,
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = qk_matmul(src.Tensor("q_transpose_out"),
                                     src.Tensor("k_transpose2_out"));
    // scale
    const auto &scale_out = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_out_value")}});
    src.Tensor("qk_scale_out") = scale_out(src.Tensor("qk_out"), full_scale());

    // mask
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_scale_out"), src.Tensor("mask"));

    // softmax
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.flash_attn",
                                    {{{"dropout", res.Float32Attr(0.0)},
                                      {"causal", res.BoolAttr(false)},
                                      {"return_softmax", res.BoolAttr(false)},
                                      {"is_test", res.BoolAttr(false)},
                                      {"rng_name", res.StrAttr("")}}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                &res.InputNoneTensor(),
                &res.Tensor("mask")},
               {&res.Tensor("out"),
                &res.Tensor("softmax"),
                &res.Tensor("softmax_lse"),
                &res.Tensor("seed_offset")});
  }
};

// Scale Dot Product Attention
// 1. scale before matmul
// 2. cast before and after softmax
class SDPAttnPatternQscaleCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SDPAttnPatternQscaleCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // scale before matmul
    const auto &scale_q = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_q_value")}});
    src.Tensor("q_scale_out") =
        scale_q(src.Tensor("q_transpose_out"), full_scale());
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose2_out"));

    // cast + mask
    const auto &mask_cast = src.Op("pd_op.cast");
    src.Tensor("mask_cast_out") = mask_cast(src.Tensor("mask"));
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask_cast_out"));

    // cast + softmax + cast
    const auto &softmax_cast1 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast1_out") = softmax_cast1(src.Tensor("mask_add_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("softmax_cast1_out"));
    const auto &softmax_cast2 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast2_out") = softmax_cast2(src.Tensor("softmax_out"));
    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // scale_dot_product_attn impl
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("scale_q_value");
        });
    const auto &scale_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(true)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    scale_dot_product_attention({&res.Tensor("q"),
                                 &res.Tensor("k"),
                                 &res.Tensor("v"),
                                 &res.Tensor("mask")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

// Scale Dot Product Attention
// 1. scale before matmul
// 2. no cast before and after softmax
class SDPAttnPatternQscaleNoCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SDPAttnPatternQscaleNoCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose,
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // scale before matmul
    const auto &scale_q = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_q_value")}});
    src.Tensor("q_scale_out") =
        scale_q(src.Tensor("q_transpose_out"), full_scale());
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") =
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose2_out"));

    // cast + mask
    const auto &mask_cast = src.Op("pd_op.cast");
    src.Tensor("mask_cast_out") = mask_cast(src.Tensor("mask"));
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask_cast_out"));

    // softmax
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // scale_dot_product_attn impl
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("scale_q_value");
        });
    const auto &scale_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(true)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    scale_dot_product_attention({&res.Tensor("q"),
                                 &res.Tensor("k"),
                                 &res.Tensor("v"),
                                 &res.Tensor("mask")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

// Scale Dot Product Attention
// 1. scale after matmul
// 2. cast before and after softmax
class SDPAttnPatternOutscaleCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SDPAttnPatternOutscaleCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose,
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = qk_matmul(src.Tensor("q_transpose_out"),
                                     src.Tensor("k_transpose2_out"));
    const auto &scale_out = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_out_value")}});
    src.Tensor("qk_scale_out") = scale_out(src.Tensor("qk_out"), full_scale());

    // cast + mask
    const auto &mask_cast = src.Op("pd_op.cast");
    src.Tensor("mask_cast_out") = mask_cast(src.Tensor("mask"));
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask_cast_out"));

    // cast + softmax + cast
    const auto &softmax_cast1 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast1_out") = softmax_cast1(src.Tensor("mask_add_out"));
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("softmax_cast1_out"));
    const auto &softmax_cast2 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast2_out") = softmax_cast2(src.Tensor("softmax_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // scale_dot_product_attn impl
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("scale_out_value");
        });
    const auto &scale_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(true)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    scale_dot_product_attention({&res.Tensor("q"),
                                 &res.Tensor("k"),
                                 &res.Tensor("v"),
                                 &res.Tensor("mask")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

// Scale Dot Product Attention
// 1. scale after matmul
// 2. no cast before and after softmax
class SDPAttnPatternOutscaleNoCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SDPAttnPatternOutscaleNoCast"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose,
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") =
        transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = qk_matmul(src.Tensor("q_transpose_out"),
                                     src.Tensor("k_transpose2_out"));
    // scale
    const auto &scale_out = src.Op("pd_op.scale");
    const auto &full_scale =
        src.Op("pd_op.full", {{"value", src.Attr("scale_out_value")}});
    src.Tensor("qk_scale_out") = scale_out(src.Tensor("qk_out"), full_scale());

    // cast + mask
    const auto &mask_cast = src.Op("pd_op.cast");
    src.Tensor("mask_cast_out") = mask_cast(src.Tensor("mask"));
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask_cast_out"));

    // softmax
    const auto &softmax =
        src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          // softmax
          const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
          if (softmax_axis != -1 && softmax_axis != 3) return false;
          // matmul transpose
          bool matmul_qk_transpose_x =
              match_ctx.Attr<bool>("matmul_qk_transpose_x");
          bool matmul_qk_transpose_y =
              match_ctx.Attr<bool>("matmul_qk_transpose_y");
          if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

          bool matmul_o_transpose_x =
              match_ctx.Attr<bool>("context_matmul_transpose_x");
          bool matmul_o_transpose_y =
              match_ctx.Attr<bool>("context_matmul_transpose_y");
          if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
          // tensor shape
          auto q_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
          auto k_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
          auto v_transpose_out =
              pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
          if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 ||
              v_transpose_out.size() != 4 ||
              !(q_transpose_out.at(0) == k_transpose_out.at(0) &&
                k_transpose_out.at(0) == v_transpose_out.at(0)) ||
              !(q_transpose_out.at(1) == k_transpose_out.at(1) &&
                k_transpose_out.at(1) == v_transpose_out.at(1)) ||
              !(q_transpose_out.at(3) == k_transpose_out.at(3) &&
                k_transpose_out.at(3) == v_transpose_out.at(3))) {
            return false;
          }
          // add shape
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // scale_dot_product_attn impl
    const auto &scaling_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<float>("scale_out_value");
        });
    const auto &scale_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", scaling_factor},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(true)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    scale_dot_product_attention({&res.Tensor("q"),
                                 &res.Tensor("k"),
                                 &res.Tensor("v"),
                                 &res.Tensor("mask")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

class AttnFusePass : public pir::PatternRewritePass {
 public:
  AttnFusePass() : pir::PatternRewritePass("attn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    // FlashAttn
    ps.Add(paddle::drr::Create<FlashAttnPatternQscaleCast>(context));
    ps.Add(paddle::drr::Create<FlashAttnPatternQscaleNoCast>(context));
    ps.Add(paddle::drr::Create<FlashAttnPatternOutscaleCast>(context));
    ps.Add(paddle::drr::Create<FlashAttnPatternOutscaleNoCast>(context));
    // Scale Dot Product Attetion
    ps.Add(paddle::drr::Create<SDPAttnPatternQscaleCast>(context));
    ps.Add(paddle::drr::Create<SDPAttnPatternQscaleNoCast>(context));
    ps.Add(paddle::drr::Create<SDPAttnPatternOutscaleCast>(context));
    ps.Add(paddle::drr::Create<SDPAttnPatternOutscaleNoCast>(context));

    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateAttnFusePass() {
  return std::make_unique<AttnFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(attn_fuse_pass, AttnFusePass);
