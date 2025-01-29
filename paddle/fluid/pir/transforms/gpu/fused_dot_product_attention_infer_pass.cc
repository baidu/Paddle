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

#include "paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_infer_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// scale before matmul
class FusedDotProductAttentionPatternQscale
    : public paddle::drr::DrrPatternBase {
 private:
  bool softmax_with_cast_;

 public:
  explicit FusedDotProductAttentionPatternQscale(bool softmax_with_cast)
      : softmax_with_cast_(softmax_with_cast) {}

 public:
  std::string name() const override {
    return "FusedDotProductAttentionPatternQscale";
  }

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

    if (softmax_with_cast_) {
      // cast + softmax + cast
      const auto &softmax_cast1 = src.Op("pd_op.cast");
      src.Tensor("softmax_cast1_out") =
          softmax_cast1(src.Tensor("mask_add_out"));
      const auto &softmax =
          src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
      src.Tensor("softmax_cast2_in") = softmax(src.Tensor("softmax_cast1_out"));
      const auto &softmax_cast2 = src.Op("pd_op.cast");
      src.Tensor("softmax_out") = softmax_cast2(src.Tensor("softmax_cast2_in"));
    } else {
      // softmax
      const auto &softmax =
          src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
      src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    }

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
          auto q_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("q"));
          if (!q_dtype.isa<pir::Float16Type>() &&
              !q_dtype.isa<pir::BFloat16Type>()) {
            return false;
          }
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
          auto q = pir::GetShapeFromValue(match_ctx.Tensor("q"));
          auto k = pir::GetShapeFromValue(match_ctx.Tensor("k"));
          auto v = pir::GetShapeFromValue(match_ctx.Tensor("v"));
          if (q.size() != 4 || k.size() != 4 || v.size() != 4 ||
              !(q.at(0) == k.at(0) && k.at(0) == v.at(0)) ||
              !(q.at(1) == k.at(1) && k.at(1) == v.at(1)) ||
              !(q.at(2) == k.at(2) && k.at(2) == v.at(2)) ||
              !(q.at(3) == k.at(3) && k.at(3) == v.at(3))) {
            return false;
          }
          // seq_len % 64 == 0
          if (q.at(1) != -1 && q.at(1) % 64 != 0) {
            return false;
          }

          // mask's shape [bs, 1, seq_len, seq_len]
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4 || mask_add.at(1) != 1) {
            return false;
          }
          auto mask_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("mask"));
          if (!mask_dtype.isa<pir::Int32Type>()) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // cast the mask
    const auto &mask_cast =
        res.Op("pd_op.cast", {{"dtype", res.DataTypeAttr("int32")}});
    res.Tensor("mask_cast_out") = mask_cast(res.Tensor("mask"));

    const auto &fused_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", src.Attr("scale_q_value")},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(false)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    fused_dot_product_attention({&res.Tensor("q"),
                                 &res.Tensor("k"),
                                 &res.Tensor("v"),
                                 &res.Tensor("mask_cast_out")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

// scale after matmul
class FusedDotProductAttentionPatternOutscale
    : public paddle::drr::DrrPatternBase {
 private:
  bool softmax_with_cast_;

 public:
  explicit FusedDotProductAttentionPatternOutscale(bool softmax_with_cast)
      : softmax_with_cast_(softmax_with_cast) {}

 public:
  std::string name() const override {
    return "FusedDotProductAttentionPatternOutscale";
  }

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
        mask_add(src.Tensor("qk_out"), src.Tensor("mask"));

    if (softmax_with_cast_) {
      // cast + softmax + cast
      const auto &softmax_cast1 = src.Op("pd_op.cast");
      src.Tensor("softmax_cast1_out") =
          softmax_cast1(src.Tensor("mask_add_out"));
      const auto &softmax =
          src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
      src.Tensor("softmax_cast2_in") = softmax(src.Tensor("softmax_cast1_out"));
      const auto &softmax_cast2 = src.Op("pd_op.cast");
      src.Tensor("softmax_out") = softmax_cast2(src.Tensor("softmax_cast2_in"));
    } else {
      // softmax
      const auto &softmax =
          src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
      src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));
    }

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
          auto q_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("q"));
          if (!q_dtype.isa<pir::Float16Type>() &&
              !q_dtype.isa<pir::BFloat16Type>()) {
            return false;
          }
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
          auto q = pir::GetShapeFromValue(match_ctx.Tensor("q"));
          auto k = pir::GetShapeFromValue(match_ctx.Tensor("k"));
          auto v = pir::GetShapeFromValue(match_ctx.Tensor("v"));
          if (q.size() != 4 || k.size() != 4 || v.size() != 4 ||
              !(q.at(0) == k.at(0) && k.at(0) == v.at(0)) ||
              !(q.at(1) == k.at(1) && k.at(1) == v.at(1)) ||
              !(q.at(2) == k.at(2) && k.at(2) == v.at(2)) ||
              !(q.at(3) == k.at(3) && k.at(3) == v.at(3))) {
            return false;
          }
          // seq_len % 64 == 0
          if (q.at(1) != -1 && q.at(1) % 64 != 0) {
            return false;
          }

          // mask's shape [bs, 1, seq_len, seq_len]
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4 || mask_add.at(1) != 1) {
            return false;
          }
          auto mask_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("mask"));
          if (!mask_dtype.isa<pir::Int32Type>()) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // cast the mask
    const auto &mask_cast =
        res.Op("pd_op.cast", {{"dtype", res.DataTypeAttr("int32")}});
    res.Tensor("mask_cast_out") = mask_cast(res.Tensor("mask"));

    const auto &fused_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", src.Attr("scale_out_value")},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(false)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    fused_dot_product_attention({&res.Tensor("q"),
                                 &res.Tensor("k"),
                                 &res.Tensor("v"),
                                 &res.Tensor("mask_cast_out")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

// slice qkv
class TransposeSliceFusedDotProductAttentionPattern
    : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "TransposeSliceFusedDotProductAttentionPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // transpose
    const auto &transpose_qkv =
        src.Op("pd_op.transpose", {{"perm", src.Attr("perm")}});
    src.Tensor("qkv_transpose") = transpose_qkv(src.Tensor("qkv"));
    // slice q -> [b, head, s, head_dim]
    const auto &slice_q =
        src.Op(paddle::dialect::SliceOp::name(),
               {{"axes", src.Attr("axes_q")},
                {"infer_flags", src.Attr("infer_flags_q")},
                {"decrease_axis", src.Attr("decrease_axis_q")}});
    const auto &full_int_array_q1 = src.Op("pd_op.full_int_array");
    const auto &full_int_array_q2 = src.Op("pd_op.full_int_array");
    src.Tensor("q") = slice_q(
        src.Tensor("qkv_transpose"), full_int_array_q1(), full_int_array_q2());
    // slice k -> [b, head, s, head_dim]
    const auto &slice_k =
        src.Op(paddle::dialect::SliceOp::name(),
               {{"axes", src.Attr("axes_k")},
                {"infer_flags", src.Attr("infer_flags_k")},
                {"decrease_axis", src.Attr("decrease_axis_k")}});
    const auto &full_int_array_k1 = src.Op("pd_op.full_int_array");
    const auto &full_int_array_k2 = src.Op("pd_op.full_int_array");
    src.Tensor("k") = slice_k(
        src.Tensor("qkv_transpose"), full_int_array_k1(), full_int_array_k2());
    // slice v -> [b, head, s, head_dim]
    const auto &slice_v =
        src.Op(paddle::dialect::SliceOp::name(),
               {{"axes", src.Attr("axes_v")},
                {"infer_flags", src.Attr("infer_flags_v")},
                {"decrease_axis", src.Attr("decrease_axis_v")}});
    const auto &full_int_array_v1 = src.Op("pd_op.full_int_array");
    const auto &full_int_array_v2 = src.Op("pd_op.full_int_array");
    src.Tensor("v") = slice_v(
        src.Tensor("qkv_transpose"), full_int_array_v1(), full_int_array_v2());

    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") =
        qk_matmul(src.Tensor("q"), src.Tensor("k_transpose_out"));
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
    src.Tensor("context_matmul_out") =
        context_matmul(src.Tensor("softmax_out"), src.Tensor("v"));
    // [b, head, s, head_dim] -> [b, s, head, head_dim]
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    // Constraints
    src.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
          auto q_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("q"));
          if (!q_dtype.isa<pir::Float16Type>() &&
              !q_dtype.isa<pir::BFloat16Type>()) {
            return false;
          }
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
          auto q = pir::GetShapeFromValue(match_ctx.Tensor("q"));
          auto k = pir::GetShapeFromValue(match_ctx.Tensor("k"));
          auto v = pir::GetShapeFromValue(match_ctx.Tensor("v"));
          if (q.size() != 4 || k.size() != 4 || v.size() != 4 ||
              !(q.at(0) == k.at(0) && k.at(0) == v.at(0)) ||
              !(q.at(1) == k.at(1) && k.at(1) == v.at(1)) ||
              !(q.at(2) == k.at(2) && k.at(2) == v.at(2)) ||
              !(q.at(3) == k.at(3) && k.at(3) == v.at(3))) {
            return false;
          }
          // seq_len % 64 == 0
          if (q.at(2) != -1 && q.at(2) % 64 != 0) {
            return false;
          }

          // mask's shape [bs, 1, seq_len, seq_len]
          auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask"));
          if (mask_add.size() != 4 || mask_add.at(1) != 1) {
            return false;
          }
          auto mask_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("mask"));
          if (!mask_dtype.isa<pir::Int32Type>()) {
            return false;
          }

          return true;
        });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // cast the mask
    const auto &mask_cast =
        res.Op("pd_op.cast", {{"dtype", res.DataTypeAttr("int32")}});
    res.Tensor("mask_cast_out") = mask_cast(res.Tensor("mask"));

    const auto &fused_dot_product_attention =
        res.Op("pd_op.fused_dot_product_attention",
               {{{"scaling_factor", src.Attr("scale_out_value")},
                 {"dropout_probability", res.Float32Attr(0.0)},
                 {"is_training", res.BoolAttr(false)},
                 {"is_causal_masking", res.BoolAttr(false)}}});

    // [b, head, seq_len, head_dim] -> [b, seq_len, head, head_dim]
    const auto &q_transpose = res.Op(
        "pd_op.transpose", {{"perm", res.VectorInt32Attr({0, 2, 1, 3})}});
    res.Tensor("q_transpose") = q_transpose(res.Tensor("q"));
    const auto &k_transpose = res.Op(
        "pd_op.transpose", {{"perm", res.VectorInt32Attr({0, 2, 1, 3})}});
    res.Tensor("k_transpose") = k_transpose(res.Tensor("k"));
    const auto &v_transpose = res.Op(
        "pd_op.transpose", {{"perm", res.VectorInt32Attr({0, 2, 1, 3})}});
    res.Tensor("v_transpose") = v_transpose(res.Tensor("v"));

    fused_dot_product_attention({&res.Tensor("q_transpose"),
                                 &res.Tensor("k_transpose"),
                                 &res.Tensor("v_transpose"),
                                 &res.Tensor("mask_cast_out")},
                                {&res.Tensor("out"),
                                 &res.Tensor("softmax_aux"),
                                 &res.Tensor("rng_state")});
  }
};

class FusedDotProductAttentionInferPass : public pir::PatternRewritePass {
 public:
  FusedDotProductAttentionInferPass()
      : pir::PatternRewritePass("fused_dot_product_attention_infer_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedDotProductAttentionPatternQscale>(context,
                                                                      true));
    ps.Add(paddle::drr::Create<FusedDotProductAttentionPatternQscale>(context,
                                                                      false));
    ps.Add(paddle::drr::Create<FusedDotProductAttentionPatternOutscale>(context,
                                                                        true));
    ps.Add(paddle::drr::Create<FusedDotProductAttentionPatternOutscale>(context,
                                                                        false));
    ps.Add(paddle::drr::Create<TransposeSliceFusedDotProductAttentionPattern>(
        context));

    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateFusedDotProductAttentionInferPass() {
  return std::make_unique<FusedDotProductAttentionInferPass>();
}
}  // namespace pir

REGISTER_IR_PASS(fused_dot_product_attention_infer_pass,
                 FusedDotProductAttentionInferPass);
