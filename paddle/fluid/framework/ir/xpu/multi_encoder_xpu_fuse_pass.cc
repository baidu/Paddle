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

#include "paddle/fluid/framework/ir/xpu/multi_encoder_xpu_fuse_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/kernels/concat_kernel.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct SingleEncoderXPUPattern : public PatternBase {
  SingleEncoderXPUPattern(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& act_type,
                          const std::string& matmul_type_0,
                          const std::string& matmul_type_1,
                          const std::string& matmul_type_2,
                          bool norm_before,
                          bool with_q_scale,
                          bool with_mask,
                          bool is_smooth_quant,
                          const std::string& relative_type);

  // declare operator node's name
  // If norm_before, use ln_0 & ln_1.
  // If not norm_before, use ln_1 & ln_2.
  PATTERN_DECL_NODE(ln_0);
  PATTERN_DECL_NODE(ln_1);
  PATTERN_DECL_NODE(ln_2);
  PATTERN_DECL_NODE(q_matmul);
  PATTERN_DECL_NODE(q_add);
  PATTERN_DECL_NODE(q_reshape);
  PATTERN_DECL_NODE(q_transpose);
  PATTERN_DECL_NODE(q_scale);
  PATTERN_DECL_NODE(k_matmul);
  PATTERN_DECL_NODE(k_add);
  PATTERN_DECL_NODE(k_reshape);
  PATTERN_DECL_NODE(k_transpose);
  PATTERN_DECL_NODE(v_matmul);
  PATTERN_DECL_NODE(v_add);
  PATTERN_DECL_NODE(v_reshape);
  PATTERN_DECL_NODE(v_transpose);
  PATTERN_DECL_NODE(qk_matmul);
  PATTERN_DECL_NODE(qk_add);
  PATTERN_DECL_NODE(qk_softmax);
  PATTERN_DECL_NODE(qkv_matmul_0);
  PATTERN_DECL_NODE(qkv_transpose);
  PATTERN_DECL_NODE(qkv_reshape);
  PATTERN_DECL_NODE(qkv_matmul_1);
  PATTERN_DECL_NODE(qkv_add_0);
  PATTERN_DECL_NODE(qkv_add_1);
  PATTERN_DECL_NODE(qkv_matmul_2);
  PATTERN_DECL_NODE(qkv_add_2);
  PATTERN_DECL_NODE(qkv_act);
  PATTERN_DECL_NODE(qkv_matmul_3);
  PATTERN_DECL_NODE(qkv_add_3);
  PATTERN_DECL_NODE(qkv_add_4);
  // declare variable node's name
  PATTERN_DECL_NODE(ln_0_x);
  PATTERN_DECL_NODE(ln_0_bias);
  PATTERN_DECL_NODE(ln_0_scale);
  PATTERN_DECL_NODE(ln_0_out);
  PATTERN_DECL_NODE(ln_0_mean);
  PATTERN_DECL_NODE(ln_0_variance);
  PATTERN_DECL_NODE(q_matmul_w);
  PATTERN_DECL_NODE(q_matmul_out);
  PATTERN_DECL_NODE(q_add_bias);
  PATTERN_DECL_NODE(q_add_out);
  PATTERN_DECL_NODE(q_reshape_out);
  PATTERN_DECL_NODE(q_transpose_out);
  PATTERN_DECL_NODE(q_scale_out);
  PATTERN_DECL_NODE(k_matmul_w);
  PATTERN_DECL_NODE(k_matmul_out);
  PATTERN_DECL_NODE(k_add_bias);
  PATTERN_DECL_NODE(k_add_out);
  PATTERN_DECL_NODE(k_reshape_out);
  PATTERN_DECL_NODE(k_transpose_out);
  PATTERN_DECL_NODE(v_matmul_w);
  PATTERN_DECL_NODE(v_matmul_out);
  PATTERN_DECL_NODE(v_add_bias);
  PATTERN_DECL_NODE(v_add_out);
  PATTERN_DECL_NODE(v_reshape_out);
  PATTERN_DECL_NODE(v_transpose_out);
  PATTERN_DECL_NODE(qk_matmul_out);
  PATTERN_DECL_NODE(qk_add_mask);
  PATTERN_DECL_NODE(qk_add_out);
  PATTERN_DECL_NODE(qk_softmax_out);
  PATTERN_DECL_NODE(qkv_matmul_0_out);
  PATTERN_DECL_NODE(qkv_transpose_out);
  PATTERN_DECL_NODE(qkv_reshape_out);
  PATTERN_DECL_NODE(qkv_matmul_1_w);
  PATTERN_DECL_NODE(qkv_matmul_1_out);
  PATTERN_DECL_NODE(qkv_add_0_bias);
  PATTERN_DECL_NODE(qkv_add_0_out);
  PATTERN_DECL_NODE(qkv_add_1_out);
  PATTERN_DECL_NODE(ln_1_bias);
  PATTERN_DECL_NODE(ln_1_scale);
  PATTERN_DECL_NODE(ln_1_out);
  PATTERN_DECL_NODE(ln_1_mean);
  PATTERN_DECL_NODE(ln_1_variance);
  PATTERN_DECL_NODE(qkv_matmul_2_w);
  PATTERN_DECL_NODE(qkv_matmul_2_out);
  PATTERN_DECL_NODE(qkv_add_2_bias);
  PATTERN_DECL_NODE(qkv_add_2_out);
  PATTERN_DECL_NODE(qkv_act_out);
  PATTERN_DECL_NODE(qkv_matmul_3_w);
  PATTERN_DECL_NODE(qkv_matmul_3_out);
  PATTERN_DECL_NODE(qkv_add_3_bias);
  PATTERN_DECL_NODE(qkv_add_3_out);
  PATTERN_DECL_NODE(qkv_add_4_out);
  PATTERN_DECL_NODE(ln_2_x);
  PATTERN_DECL_NODE(ln_2_bias);
  PATTERN_DECL_NODE(ln_2_scale);
  PATTERN_DECL_NODE(ln_2_out);
  PATTERN_DECL_NODE(ln_2_mean);
  PATTERN_DECL_NODE(ln_2_variance);
  // smooth quant
  PATTERN_DECL_NODE(smooth_scale_1);
  PATTERN_DECL_NODE(smooth_scale_2);
  PATTERN_DECL_NODE(smooth_scale_1_weight);
  PATTERN_DECL_NODE(smooth_scale_2_weight);
  PATTERN_DECL_NODE(smooth_scale_1_out);
  PATTERN_DECL_NODE(smooth_scale_2_out);

  // roformer_relative_embedding_xpu
  PATTERN_DECL_NODE(q_relative_emb);
  PATTERN_DECL_NODE(q_cos_embedding);
  PATTERN_DECL_NODE(q_sin_embedding);
  PATTERN_DECL_NODE(q_relative_emb_out);
  PATTERN_DECL_NODE(k_relative_emb);
  PATTERN_DECL_NODE(k_cos_embedding);
  PATTERN_DECL_NODE(k_sin_embedding);
  PATTERN_DECL_NODE(k_relative_emb_out);

 private:
  std::string act_type_;
  std::string matmul_type_0_;
  std::string matmul_type_1_;
  std::string matmul_type_2_;
  bool norm_before_{false};
  bool with_q_scale_{false};
  bool with_mask_{true};
  bool is_smooth_quant_{false};
  std::string relative_type_ = "";
};

SingleEncoderXPUPattern::SingleEncoderXPUPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& act_type,
    const std::string& matmul_type_0,
    const std::string& matmul_type_1,
    const std::string& matmul_type_2,
    bool norm_before,
    bool with_q_scale,
    bool with_mask,
    bool is_smooth_quant,
    const std::string& relative_type)
    : PatternBase(pattern, name_scope, name_scope),
      act_type_(act_type),
      matmul_type_0_(matmul_type_0),
      matmul_type_1_(matmul_type_1),
      matmul_type_2_(matmul_type_2),
      norm_before_(norm_before),
      with_q_scale_(with_q_scale),
      with_mask_(with_mask),
      is_smooth_quant_(is_smooth_quant),
      relative_type_(relative_type) {
  // layer_norm 0
  PDNode* ln_0_x = pattern->NewNode(ln_0_x_repr());
  PDNode* ln_0_bias = nullptr;
  PDNode* ln_0_scale = nullptr;
  PDNode* ln_0 = nullptr;
  PDNode* ln_0_out = nullptr;
  PDNode* ln_0_mean = nullptr;
  PDNode* ln_0_variance = nullptr;
  PDNode* smooth_scale_1_weight = nullptr;
  PDNode* smooth_scale_1 = nullptr;
  PDNode* smooth_scale_1_out = nullptr;
  if (norm_before_) {
    ln_0_x->assert_is_op_input("layer_norm", "X")->assert_var_not_persistable();
    ln_0_bias = pattern->NewNode(ln_0_bias_repr())
                    ->assert_is_op_input("layer_norm", "Bias")
                    ->assert_is_persistable_var();
    ln_0_scale = pattern->NewNode(ln_0_scale_repr())
                     ->assert_is_op_input("layer_norm", "Scale")
                     ->assert_is_persistable_var();
    ln_0 = pattern->NewNode(ln_0_repr())->assert_is_op("layer_norm");
    ln_0_out = pattern->NewNode(ln_0_out_repr())
                   ->assert_is_op_output("layer_norm", "Y")
                   ->assert_var_not_persistable();
    ln_0_mean = pattern->NewNode(ln_0_mean_repr())
                    ->assert_is_op_output("layer_norm", "Mean")
                    ->assert_var_not_persistable();
    ln_0_variance = pattern->NewNode(ln_0_variance_repr())
                        ->assert_is_op_output("layer_norm", "Variance")
                        ->assert_var_not_persistable();
  }
  if (!norm_before_ && is_smooth_quant_) {
    VLOG(3) << "build first smooth_quant_scale";
    ln_0_x->assert_is_op_input("elementwise_mul", "X")
        ->assert_var_not_persistable();
    smooth_scale_1_weight = pattern->NewNode(smooth_scale_1_weight_repr())
                                ->assert_is_op_input("elementwise_mul", "Y")
                                ->assert_is_persistable_var();
    smooth_scale_1 = pattern->NewNode(smooth_scale_1_repr())
                         ->assert_is_op("elementwise_mul");
    smooth_scale_1_out = pattern->NewNode(smooth_scale_1_out_repr())
                             ->assert_is_op_output("elementwise_mul", "Out")
                             ->assert_var_not_persistable();
  }

  // q: matmul + add + reshape + transpose
  auto q_matmul_w = pattern->NewNode(q_matmul_w_repr())
                        ->assert_is_op_input(matmul_type_0_, "Y")
                        ->assert_is_persistable_var();
  auto* q_matmul =
      pattern->NewNode(q_matmul_repr())->assert_is_op(matmul_type_0_);
  auto* q_matmul_out = pattern->NewNode(q_matmul_out_repr())
                           ->assert_is_op_output(matmul_type_0_, "Out")
                           ->assert_var_not_persistable();
  auto q_add_bias = pattern->NewNode(q_add_bias_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var();
  auto* q_add = pattern->NewNode(q_add_repr())->assert_is_op("elementwise_add");
  auto* q_add_out = pattern->NewNode(q_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_var_not_persistable();
  auto* q_reshape =
      pattern->NewNode(q_reshape_repr())->assert_is_op("reshape2");
  auto* q_reshape_out = pattern->NewNode(q_reshape_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_var_not_persistable();
  auto* q_transpose =
      pattern->NewNode(q_transpose_repr())->assert_is_op("transpose2");
  auto* q_transpose_out = pattern->NewNode(q_transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_var_not_persistable();
  PDNode* q_scale = nullptr;
  PDNode* q_scale_out = nullptr;
  std::string target_op_type = matmul_type_1_;
  if (with_q_scale_) {
    q_scale = pattern->NewNode(q_scale_repr())->assert_is_op("scale");
    q_scale_out = pattern->NewNode(q_scale_out_repr())
                      ->assert_is_op_output("scale", "Out")
                      ->assert_is_op_input(matmul_type_1_, "X")
                      ->assert_var_not_persistable();
    target_op_type = "scale";
  } else {
    if (relative_type_.empty()) {
      q_transpose_out->assert_is_op_input(target_op_type, "X");
    } else {
      q_transpose_out->assert_is_op_input(relative_type_, "x");
    }
  }
  PDNode* q_relative_emb = nullptr;
  PDNode* q_cos_embedding = nullptr;
  PDNode* q_sin_embedding = nullptr;
  PDNode* q_relative_emb_out = nullptr;
  if (relative_type_ == "roformer_relative_embedding_xpu") {
    VLOG(3) << "build q_relative_emb";
    q_relative_emb =
        pattern->NewNode(q_relative_emb_repr())->assert_is_op(relative_type_);
    q_sin_embedding = pattern->NewNode(q_sin_embedding_repr())
                          ->assert_is_op_input(relative_type_, "sin_emb")
                          ->AsInput();
    q_cos_embedding = pattern->NewNode(q_cos_embedding_repr())
                          ->assert_is_op_input(relative_type_, "cos_emb")
                          ->AsInput();
    q_relative_emb_out = pattern->NewNode(q_relative_emb_out_repr())
                             ->assert_is_op_output(relative_type_, "out")
                             ->assert_is_op_input(target_op_type, "X");
  }

  // k: matmul + add + reshape + transpose
  auto k_matmul_w = pattern->NewNode(k_matmul_w_repr())
                        ->assert_is_op_input(matmul_type_0_, "Y")
                        ->assert_is_persistable_var();
  auto* k_matmul =
      pattern->NewNode(k_matmul_repr())->assert_is_op(matmul_type_0_);
  auto* k_matmul_out = pattern->NewNode(k_matmul_out_repr())
                           ->assert_is_op_output(matmul_type_0_, "Out")
                           ->assert_var_not_persistable();
  auto k_add_bias = pattern->NewNode(k_add_bias_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var();
  auto* k_add = pattern->NewNode(k_add_repr())->assert_is_op("elementwise_add");
  auto* k_add_out = pattern->NewNode(k_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_var_not_persistable();
  auto* k_reshape =
      pattern->NewNode(k_reshape_repr())->assert_is_op("reshape2");
  auto* k_reshape_out = pattern->NewNode(k_reshape_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_var_not_persistable();
  auto* k_transpose =
      pattern->NewNode(k_transpose_repr())->assert_is_op("transpose2");
  auto* k_transpose_out = pattern->NewNode(k_transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_var_not_persistable();

  PDNode* k_relative_emb = nullptr;
  PDNode* k_sin_embedding = q_sin_embedding;
  PDNode* k_cos_embedding = q_cos_embedding;
  PDNode* k_relative_emb_out = nullptr;
  if (relative_type_.empty()) {
    k_transpose_out->assert_is_op_input(matmul_type_1_, "Y");
  } else if (relative_type_ == "roformer_relative_embedding_xpu") {
    VLOG(3) << "build k_relative_emb";
    k_transpose_out->assert_is_op_input(relative_type_, "x");
    k_relative_emb =
        pattern->NewNode(k_relative_emb_repr())->assert_is_op(relative_type_);
    k_relative_emb_out = pattern->NewNode(k_relative_emb_out_repr())
                             ->assert_is_op_output(relative_type_, "out")
                             ->assert_is_op_input(matmul_type_1_, "Y");
  }
  // qk: matmul + add + softmax
  auto* qk_matmul =
      pattern->NewNode(qk_matmul_repr())->assert_is_op(matmul_type_1_);
  auto* qk_matmul_out = pattern->NewNode(qk_matmul_out_repr())
                            ->assert_is_op_output(matmul_type_1_, "Out")
                            ->assert_var_not_persistable();
  PDNode* qk_add_mask = nullptr;
  PDNode* qk_add = nullptr;
  PDNode* qk_add_out = nullptr;
  if (with_mask_) {
    qk_add_mask = pattern->NewNode(qk_add_mask_repr())
                      ->assert_is_op_input("elementwise_add", "Y")
                      ->assert_var_not_persistable();
    qk_add = pattern->NewNode(qk_add_repr())->assert_is_op("elementwise_add");
    qk_add_out = pattern->NewNode(qk_add_out_repr())
                     ->assert_is_op_output("elementwise_add", "Out")
                     ->assert_var_not_persistable();
  }
  auto* qk_softmax =
      pattern->NewNode(qk_softmax_repr())->assert_is_op("softmax");
  auto* qk_softmax_out = pattern->NewNode(qk_softmax_out_repr())
                             ->assert_is_op_output("softmax", "Out")
                             ->assert_is_op_input(matmul_type_2_, "X")
                             ->assert_var_not_persistable();

  // v: matmul + add + reshape + transpose
  auto v_matmul_w = pattern->NewNode(v_matmul_w_repr())
                        ->assert_is_op_input(matmul_type_0_, "Y")
                        ->assert_is_persistable_var();
  auto* v_matmul =
      pattern->NewNode(v_matmul_repr())->assert_is_op(matmul_type_0_);
  auto* v_matmul_out = pattern->NewNode(v_matmul_out_repr())
                           ->assert_is_op_output(matmul_type_0_, "Out")
                           ->assert_var_not_persistable();
  auto v_add_bias = pattern->NewNode(v_add_bias_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var();
  auto* v_add = pattern->NewNode(v_add_repr())->assert_is_op("elementwise_add");
  auto* v_add_out = pattern->NewNode(v_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_var_not_persistable();
  auto* v_reshape =
      pattern->NewNode(v_reshape_repr())->assert_is_op("reshape2");
  auto* v_reshape_out = pattern->NewNode(v_reshape_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_var_not_persistable();
  auto* v_transpose =
      pattern->NewNode(v_transpose_repr())->assert_is_op("transpose2");
  auto* v_transpose_out = pattern->NewNode(v_transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input(matmul_type_2_, "Y")
                              ->assert_var_not_persistable();

  // qkv
  auto* qkv_matmul_0 =
      pattern->NewNode(qkv_matmul_0_repr())->assert_is_op(matmul_type_2_);
  auto* qkv_matmul_0_out = pattern->NewNode(qkv_matmul_0_out_repr())
                               ->assert_is_op_output(matmul_type_2_, "Out")
                               ->assert_var_not_persistable();
  auto* qkv_transpose =
      pattern->NewNode(qkv_transpose_repr())->assert_is_op("transpose2");
  auto* qkv_transpose_out = pattern->NewNode(qkv_transpose_out_repr())
                                ->assert_is_op_output("transpose2", "Out")
                                ->assert_var_not_persistable();
  auto* qkv_reshape =
      pattern->NewNode(qkv_reshape_repr())->assert_is_op("reshape2");
  auto* qkv_reshape_out = pattern->NewNode(qkv_reshape_out_repr())
                              ->assert_is_op_output("reshape2", "Out")
                              ->assert_var_not_persistable();
  auto qkv_matmul_1_w = pattern->NewNode(qkv_matmul_1_w_repr())
                            ->assert_is_op_input(matmul_type_0_, "Y")
                            ->assert_is_persistable_var();
  auto* qkv_matmul_1 =
      pattern->NewNode(qkv_matmul_1_repr())->assert_is_op(matmul_type_0_);
  auto* qkv_matmul_1_out = pattern->NewNode(qkv_matmul_1_out_repr())
                               ->assert_is_op_output(matmul_type_0_, "Out")
                               ->assert_var_not_persistable();
  auto qkv_add_0_bias = pattern->NewNode(qkv_add_0_bias_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var();
  auto* qkv_add_0 =
      pattern->NewNode(qkv_add_0_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_0_out = pattern->NewNode(qkv_add_0_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  auto* qkv_add_1 =
      pattern->NewNode(qkv_add_1_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_1_out = pattern->NewNode(qkv_add_1_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_is_op_input("layer_norm", "X")
                            ->assert_var_not_persistable();
  auto* ln_1_bias = pattern->NewNode(ln_1_bias_repr())
                        ->assert_is_op_input("layer_norm", "Bias")
                        ->assert_is_persistable_var();
  auto* ln_1_scale = pattern->NewNode(ln_1_scale_repr())
                         ->assert_is_op_input("layer_norm", "Scale")
                         ->assert_is_persistable_var();
  auto* ln_1 = pattern->NewNode(ln_1_repr())->assert_is_op("layer_norm");
  auto* ln_1_out = pattern->NewNode(ln_1_out_repr())
                       ->assert_is_op_output("layer_norm", "Y")
                       ->assert_var_not_persistable();
  auto* ln_1_mean = pattern->NewNode(ln_1_mean_repr())
                        ->assert_is_op_output("layer_norm", "Mean")
                        ->assert_var_not_persistable();
  auto* ln_1_variance = pattern->NewNode(ln_1_variance_repr())
                            ->assert_is_op_output("layer_norm", "Variance")
                            ->assert_var_not_persistable();
  PDNode* smooth_scale_2_weight = nullptr;
  PDNode* smooth_scale_2 = nullptr;
  PDNode* smooth_scale_2_out = nullptr;
  if (!norm_before_ && is_smooth_quant_) {
    VLOG(3) << "build second smooth_quant_scale";
    ln_1_out->assert_is_op_input("elementwise_mul", "X")
        ->assert_var_not_persistable();
    smooth_scale_2_weight = pattern->NewNode(smooth_scale_2_weight_repr())
                                ->assert_is_op_input("elementwise_mul", "Y")
                                ->assert_is_persistable_var();
    smooth_scale_2 = pattern->NewNode(smooth_scale_2_repr())
                         ->assert_is_op("elementwise_mul");
    smooth_scale_2_out = pattern->NewNode(smooth_scale_2_out_repr())
                             ->assert_is_op_output("elementwise_mul", "Out")
                             ->assert_var_not_persistable();
  }
  auto qkv_matmul_2_w = pattern->NewNode(qkv_matmul_2_w_repr())
                            ->assert_is_op_input(matmul_type_0_, "Y")
                            ->assert_is_persistable_var();
  auto* qkv_matmul_2 =
      pattern->NewNode(qkv_matmul_2_repr())->assert_is_op(matmul_type_0_);
  auto* qkv_matmul_2_out = pattern->NewNode(qkv_matmul_2_out_repr())
                               ->assert_is_op_output(matmul_type_0_, "Out")
                               ->assert_var_not_persistable();
  auto qkv_add_2_bias = pattern->NewNode(qkv_add_2_bias_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var();
  auto* qkv_add_2 =
      pattern->NewNode(qkv_add_2_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_2_out = pattern->NewNode(qkv_add_2_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  auto* qkv_act = pattern->NewNode(qkv_act_repr())->assert_is_op(act_type_);
  auto* qkv_act_out = pattern->NewNode(qkv_act_out_repr())
                          ->assert_is_op_output(act_type_, "Out")
                          ->assert_var_not_persistable();
  auto qkv_matmul_3_w = pattern->NewNode(qkv_matmul_3_w_repr())
                            ->assert_is_op_input(matmul_type_0_, "Y")
                            ->assert_is_persistable_var();
  auto* qkv_matmul_3 =
      pattern->NewNode(qkv_matmul_3_repr())->assert_is_op(matmul_type_0_);
  auto* qkv_matmul_3_out = pattern->NewNode(qkv_matmul_3_out_repr())
                               ->assert_is_op_output(matmul_type_0_, "Out")
                               ->assert_var_not_persistable();
  auto qkv_add_3_bias = pattern->NewNode(qkv_add_3_bias_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var();
  auto* qkv_add_3 =
      pattern->NewNode(qkv_add_3_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_3_out = pattern->NewNode(qkv_add_3_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  auto* qkv_add_4 =
      pattern->NewNode(qkv_add_4_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_4_out = pattern->NewNode(qkv_add_4_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  PDNode* ln_2_bias = nullptr;
  PDNode* ln_2_scale = nullptr;
  PDNode* ln_2 = nullptr;
  PDNode* ln_2_out = nullptr;
  PDNode* ln_2_mean = nullptr;
  PDNode* ln_2_variance = nullptr;
  if (!norm_before_) {
    ln_2_bias = pattern->NewNode(ln_2_bias_repr())
                    ->assert_is_op_input("layer_norm", "Bias")
                    ->assert_is_persistable_var();
    ln_2_scale = pattern->NewNode(ln_2_scale_repr())
                     ->assert_is_op_input("layer_norm", "Scale")
                     ->assert_is_persistable_var();
    ln_2 = pattern->NewNode(ln_2_repr())->assert_is_op("layer_norm");
    ln_2_out = pattern->NewNode(ln_2_out_repr())
                   ->assert_is_op_output("layer_norm", "Y")
                   ->assert_var_not_persistable();
    ln_2_mean = pattern->NewNode(ln_2_mean_repr())
                    ->assert_is_op_output("layer_norm", "Mean")
                    ->assert_var_not_persistable();
    ln_2_variance = pattern->NewNode(ln_2_variance_repr())
                        ->assert_is_op_output("layer_norm", "Variance")
                        ->assert_var_not_persistable();
  }

  // link nodes
  PDNode* q_matmul_x = ln_0_x;
  if (norm_before_) {
    ln_0->LinksFrom({ln_0_x, ln_0_bias, ln_0_scale})
        .LinksTo({ln_0_out, ln_0_mean, ln_0_variance});
    q_matmul_x = ln_0_out;
  }
  q_matmul->LinksFrom({q_matmul_x, q_matmul_w}).LinksTo({q_matmul_out});
  q_add->LinksFrom({q_matmul_out, q_add_bias}).LinksTo({q_add_out});
  q_reshape->LinksFrom({q_add_out}).LinksTo({q_reshape_out});
  q_transpose->LinksFrom({q_reshape_out}).LinksTo({q_transpose_out});
  PDNode* last_node = q_transpose_out;
  if (relative_type_ == "roformer_relative_embedding_xpu") {
    VLOG(3) << "build q_relative_emb link";
    q_relative_emb->LinksFrom({last_node, q_sin_embedding, q_cos_embedding})
        .LinksTo({q_relative_emb_out});
    last_node = q_relative_emb_out;
  }
  if (with_q_scale_) {
    q_scale->LinksFrom({last_node}).LinksTo({q_scale_out});
    last_node = q_scale_out;
  }
  PDNode* qk_matmul_x = last_node;

  k_matmul->LinksFrom({q_matmul_x, k_matmul_w}).LinksTo({k_matmul_out});
  k_add->LinksFrom({k_matmul_out, k_add_bias}).LinksTo({k_add_out});
  k_reshape->LinksFrom({k_add_out}).LinksTo({k_reshape_out});
  k_transpose->LinksFrom({k_reshape_out}).LinksTo({k_transpose_out});
  last_node = k_transpose_out;
  if (relative_type_ == "roformer_relative_embedding_xpu") {
    VLOG(3) << "build k_relative_emb link";
    k_relative_emb->LinksFrom({last_node, k_sin_embedding, k_cos_embedding})
        .LinksTo({k_relative_emb_out});
    last_node = k_relative_emb_out;
  }
  qk_matmul->LinksFrom({qk_matmul_x, last_node}).LinksTo({qk_matmul_out});
  PDNode* qk_softmax_x = qk_matmul_out;
  if (with_mask_) {
    qk_add->LinksFrom({qk_matmul_out, qk_add_mask}).LinksTo({qk_add_out});
    qk_softmax_x = qk_add_out;
  }
  qk_softmax->LinksFrom({qk_softmax_x}).LinksTo({qk_softmax_out});

  v_matmul->LinksFrom({q_matmul_x, v_matmul_w}).LinksTo({v_matmul_out});
  v_add->LinksFrom({v_matmul_out, v_add_bias}).LinksTo({v_add_out});
  v_reshape->LinksFrom({v_add_out}).LinksTo({v_reshape_out});
  v_transpose->LinksFrom({v_reshape_out}).LinksTo({v_transpose_out});

  qkv_matmul_0->LinksFrom({qk_softmax_out, v_transpose_out})
      .LinksTo({qkv_matmul_0_out});
  qkv_transpose->LinksFrom({qkv_matmul_0_out}).LinksTo({qkv_transpose_out});
  qkv_reshape->LinksFrom({qkv_transpose_out}).LinksTo({qkv_reshape_out});
  qkv_matmul_1->LinksFrom({qkv_reshape_out, qkv_matmul_1_w})
      .LinksTo({qkv_matmul_1_out});
  qkv_add_0->LinksFrom({qkv_matmul_1_out, qkv_add_0_bias})
      .LinksTo({qkv_add_0_out});
  if (!norm_before_ && is_smooth_quant_) {
    smooth_scale_1->LinksFrom({q_matmul_x, smooth_scale_1_weight})
        .LinksTo({smooth_scale_1_out});
    qkv_add_1->LinksFrom({qkv_add_0_out, smooth_scale_1_out})
        .LinksTo({qkv_add_1_out});
  } else {
    qkv_add_1->LinksFrom({qkv_add_0_out, q_matmul_x}).LinksTo({qkv_add_1_out});
  }

  ln_1->LinksFrom({qkv_add_1_out, ln_1_bias, ln_1_scale})
      .LinksTo({ln_1_out, ln_1_mean, ln_1_variance});
  qkv_matmul_2->LinksFrom({ln_1_out, qkv_matmul_2_w})
      .LinksTo({qkv_matmul_2_out});
  qkv_add_2->LinksFrom({qkv_matmul_2_out, qkv_add_2_bias})
      .LinksTo({qkv_add_2_out});
  qkv_act->LinksFrom({qkv_add_2_out}).LinksTo({qkv_act_out});
  qkv_matmul_3->LinksFrom({qkv_act_out, qkv_matmul_3_w})
      .LinksTo({qkv_matmul_3_out});
  qkv_add_3->LinksFrom({qkv_matmul_3_out, qkv_add_3_bias})
      .LinksTo({qkv_add_3_out});
  if (norm_before_) {
    qkv_add_4->LinksFrom({qkv_add_3_out, qkv_add_1_out})
        .LinksTo({qkv_add_4_out});
  } else {
    if (is_smooth_quant_) {
      smooth_scale_2->LinksFrom({ln_1_out, smooth_scale_2_weight})
          .LinksTo({smooth_scale_2_out});
      qkv_add_4->LinksFrom({qkv_add_3_out, smooth_scale_2_out})
          .LinksTo({qkv_add_4_out});
    } else {
      qkv_add_4->LinksFrom({qkv_add_3_out, ln_1_out}).LinksTo({qkv_add_4_out});
    }
    ln_2->LinksFrom({qkv_add_4_out, ln_2_bias, ln_2_scale})
        .LinksTo({ln_2_out, ln_2_mean, ln_2_variance});
  }
}

}  // namespace patterns

void MultiEncoderXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int single_encoder_fused_counts = 0;
  int multi_encoder_fused_counts = 0;
  auto pattern_params = GeneratePatternParams();
  for (auto pattern_param : pattern_params) {
    single_encoder_fused_counts +=
        ApplySingleEncoderXPUFuse(graph,
                                  pattern_param.act_type,
                                  pattern_param.matmul_type_0,
                                  pattern_param.matmul_type_1,
                                  pattern_param.matmul_type_2,
                                  pattern_param.norm_before,
                                  pattern_param.with_q_scale,
                                  pattern_param.with_mask,
                                  pattern_param.is_smooth_quant,
                                  pattern_param.relative_type);
    while (ApplyMultiEncoderXPUFuse(graph)) {
      multi_encoder_fused_counts++;
    }
  }
  int cast_mask_counts = CastMask(graph);

  AddStatis(single_encoder_fused_counts);
  AddStatis(multi_encoder_fused_counts);
  AddStatis(cast_mask_counts);
}

void MultiEncoderXPUFusePass::PrepareInputMax(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    std::unordered_map<std::string, std::vector<Node*>>* node_maps,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    std::vector<Node*>* input_max_nodes,
    std::vector<std::string>* quant_types,
    const std::string* act_type) const {
  // mul input_max, output_max * 6 + matmul x_max,y_max,output_max * 2
  auto quant_mul_ops = node_maps->find("quant_mul_ops")->second;
  auto matmul_ops = node_maps->find("matmul_ops")->second;
  auto mul_add_ops = node_maps->find("mul_add_ops")->second;
  auto softmax_ops = node_maps->find("softmax_ops")->second;
  std::vector<float> input_max(quant_mul_ops.size() * 2 + matmul_ops.size() * 3,
                               0);
  for (size_t i = 0; i < quant_mul_ops.size(); ++i) {
    auto input_name = quant_mul_ops[i]->Op()->Input("X")[0];
    auto output_name = mul_add_ops[i]->Op()->Output("Out")[0];
    if (quant_mul_ops[i]->Op()->HasAttr("enable_int8") &&
        PADDLE_GET_CONST(bool,
                         quant_mul_ops[i]->Op()->GetAttr("enable_int8")) &&
        var_quant_scales->find(input_name) != var_quant_scales->end() &&
        var_quant_scales->find(output_name) != var_quant_scales->end()) {
      input_max[i * 2] = var_quant_scales->at(input_name)[0];
      input_max[i * 2 + 1] = var_quant_scales->at(output_name)[0];
      VLOG(3) << quant_mul_ops[i] << " input_max: " << input_max[i * 2]
              << ", output_max(ew_add): " << input_max[i * 2 + 1];
      quant_types->push_back("enable_int8");
    } else {
      quant_types->push_back("not_quantized");
    }
  }
  float max_qkv_input = std::max(input_max[0], input_max[2]);
  max_qkv_input = std::max(max_qkv_input, input_max[4]);
  input_max[0] = max_qkv_input;
  input_max[2] = max_qkv_input;
  input_max[4] = max_qkv_input;
  float max_qkv_output = std::max(input_max[1], input_max[3]);
  max_qkv_output = std::max(max_qkv_output, input_max[5]);
  input_max[1] = max_qkv_output;
  input_max[3] = max_qkv_output;
  input_max[5] = max_qkv_output;
  VLOG(3) << "max_qkv_input: " << max_qkv_input
          << ", max_qkv_output: " << max_qkv_output;

  if (*act_type == "gelu") {
    // use gelu10 according to whitepaper http://arxiv.org/abs/2004.09602
    float gelu_out_threshold = 10.f;
    if (std::getenv("QUANT_GELU_OUT_THRESHOLD")) {
      gelu_out_threshold = atof(std::getenv("QUANT_GELU_OUT_THRESHOLD"));
      PADDLE_ENFORCE_GT(
          gelu_out_threshold,
          0.f,
          common::errors::InvalidArgument(
              "QUANT_GELU_OUT_THRESHOLD should be an positive float value: %f",
              gelu_out_threshold));
    }
    input_max[9] = std::min(gelu_out_threshold, input_max[9]);
    input_max[10] = std::min(gelu_out_threshold, input_max[10]);
  }

  auto input_x_name = matmul_ops[0]->Op()->Input("X")[0];
  auto input_y_name = matmul_ops[0]->Op()->Input("Y")[0];
  auto output_name = softmax_ops[0]->Op()->Output("Out")[0];
  auto matmul_offset = quant_mul_ops.size();
  std::vector<bool> matmul_quants(2, false);
  if (matmul_ops[0]->Op()->HasAttr("enable_int8") &&
      PADDLE_GET_CONST(bool, matmul_ops[0]->Op()->GetAttr("enable_int8")) &&
      var_quant_scales->find(input_x_name) != var_quant_scales->end() &&
      var_quant_scales->find(input_y_name) != var_quant_scales->end() &&
      var_quant_scales->find(output_name) != var_quant_scales->end()) {
    input_max[matmul_offset * 2 + 0] =
        max_qkv_output != 0 ? max_qkv_output
                            : var_quant_scales->at(input_x_name)[0];
    input_max[matmul_offset * 2 + 1] =
        max_qkv_output != 0 ? max_qkv_output
                            : var_quant_scales->at(input_y_name)[0];
    input_max[matmul_offset * 2 + 2] = var_quant_scales->at(output_name)[0];
    VLOG(3) << "qk_matmul X_max: " << input_max[matmul_offset * 2 + 0]
            << "          Y_max: " << input_max[matmul_offset * 2 + 1]
            << "        Out_max: " << input_max[matmul_offset * 2 + 2];
    matmul_quants[0] = true;
    quant_types->push_back("enable_int8");
  } else {
    quant_types->push_back("not_quantized");
  }
  input_x_name = matmul_ops[1]->Op()->Input("X")[0];
  input_y_name = matmul_ops[1]->Op()->Input("Y")[0];
  output_name = matmul_ops[1]->Op()->Output("Out")[0];
  if (matmul_ops[1]->Op()->HasAttr("enable_int8") &&
      PADDLE_GET_CONST(bool, matmul_ops[1]->Op()->GetAttr("enable_int8")) &&
      var_quant_scales->find(input_x_name) != var_quant_scales->end() &&
      var_quant_scales->find(input_y_name) != var_quant_scales->end() &&
      var_quant_scales->find(output_name) != var_quant_scales->end()) {
    input_max[matmul_offset * 2 + 3] = var_quant_scales->at(input_x_name)[0];
    input_max[matmul_offset * 2 + 4] = var_quant_scales->at(input_y_name)[0];
    input_max[matmul_offset * 2 + 5] = var_quant_scales->at(output_name)[0];
    VLOG(3) << "qk_matmul X_max: " << input_max[matmul_offset * 2 + 3]
            << "          Y_max: " << input_max[matmul_offset * 2 + 4]
            << "        Out_max: " << input_max[matmul_offset * 2 + 5];
    matmul_quants[1] = true;
    quant_types->push_back("enable_int8");
  } else {
    quant_types->push_back("not_quantized");
  }
  if (matmul_quants[0] == false && matmul_quants[1] == false) {
    // For backward compatible, API uses the size of input_max vector to
    // check whether it is mul quant or mul+matmul quant.
    input_max.resize(quant_mul_ops.size() * 2);
    input_max_nodes->resize(quant_mul_ops.size() * 2, nullptr);
  }
  std::string base_name = quant_mul_ops[0]->Op()->Input("X")[0];
  for (size_t i = 0; i < input_max.size(); i++) {
    std::string fc_input_max_name =
        base_name + "_" + std::to_string(i) + "_max_in";
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    VarDesc fc_input_max_desc(fc_input_max_name);
    fc_input_max_desc.SetPersistable(true);
    fc_input_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
    fc_input_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    Node* fc_input_max_in = graph->CreateVarNode(&fc_input_max_desc);
    auto* block_input_max_in_desc = block->Var(fc_input_max_name);
    block_input_max_in_desc->SetPersistable(fc_input_max_desc.Persistable());
    block_input_max_in_desc->SetShape(fc_input_max_desc.GetShape());
    block_input_max_in_desc->SetDataType(fc_input_max_desc.GetDataType());
    (*input_max_nodes)[i] = fc_input_max_in;

    phi::DenseTensor fc_max_in_cpu_tensor;
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    fc_max_in_cpu_tensor.set_type(phi::DataType::FLOAT32);
    fc_max_in_cpu_tensor.Resize({max_ptr_size});
    std::vector<float> output_scales(max_ptr_size, input_max[i]);
    memcpy(cpu_ctx->Alloc<float>(&fc_max_in_cpu_tensor),
           output_scales.data(),
           max_ptr_size * sizeof(float));
    Assign(fc_max_in_cpu_tensor,
           scope->Var(fc_input_max_name)->GetMutable<phi::DenseTensor>());
  }
}

void MultiEncoderXPUFusePass::PrepareQKVWeight(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    Node* q_w,
    Node* k_w,
    Node* v_w,
    bool enable_int8,
    bool local_quant,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    Node** qkv_w_intx,
    Node** qkv_w_max,
    Node** qkv_scale_max) const {
  phi::DenseTensor qkv_w_intx_t;
  phi::DenseTensor qkv_w_max_t;
  phi::DenseTensor qkv_scale_max_t;
  size_t qkv_w_intx_hash;
  phi::DenseTensor q_w_t;
  phi::DenseTensor k_w_t;
  phi::DenseTensor v_w_t;
  Assign(scope->Var(q_w->Name())->Get<phi::DenseTensor>(), &q_w_t);
  Assign(scope->Var(k_w->Name())->Get<phi::DenseTensor>(), &k_w_t);
  Assign(scope->Var(v_w->Name())->Get<phi::DenseTensor>(), &v_w_t);
  Transpose2D(&q_w_t);
  Transpose2D(&k_w_t);
  Transpose2D(&v_w_t);
  qkv_w_intx_t.Resize(DDim(
      {q_w_t.dims()[0] + k_w_t.dims()[0] + v_w_t.dims()[0], q_w_t.dims()[1]}));
  qkv_w_intx_t.set_type(q_w_t.type());
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  paddle::experimental::CheckAndTrans2Contiguous(&q_w_t);
  paddle::experimental::CheckAndTrans2Contiguous(&k_w_t);
  paddle::experimental::CheckAndTrans2Contiguous(&v_w_t);
  std::vector<const phi::DenseTensor*> in_tensors{&q_w_t, &k_w_t, &v_w_t};
  bool is_per_channel = false;
  if (!enable_int8) {
    CastToFp32(&q_w_t);
    CastToFp32(&k_w_t);
    CastToFp32(&v_w_t);
    phi::ConcatKernel<float>(*cpu_ctx, in_tensors, 0, &qkv_w_intx_t);
    if (local_quant) {
      qkv_w_intx_hash = HashTensor<float>(qkv_w_intx_t);
      ConvertFromFp32ToFp16(&qkv_w_intx_t, &qkv_w_max_t, false);
    } else {
      ConvertWithQuant<float, int16_t>(
          &qkv_w_intx_t, &qkv_w_max_t, &qkv_scale_max_t, false);
      qkv_w_intx_hash = HashTensor<int16_t>(qkv_w_intx_t);
    }
  } else {
    phi::ConcatKernel<int8_t>(*cpu_ctx, in_tensors, 0, &qkv_w_intx_t);
    std::unordered_map<std::string, std::vector<float>> var_quant_scales =
        GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
    std::vector<float> q_scales = var_quant_scales.at(q_w->Name());
    if (q_scales.size() != 1) {
      is_per_channel = true;
      std::vector<float> k_scales = var_quant_scales.at(k_w->Name());
      std::vector<float> v_scales = var_quant_scales.at(v_w->Name());
      q_scales.insert(q_scales.end(), k_scales.begin(), k_scales.end());
      q_scales.insert(q_scales.end(), v_scales.begin(), v_scales.end());
    }
    ConvertWithoutQuant<int8_t>(
        &qkv_w_intx_t, &qkv_w_max_t, &qkv_scale_max_t, false, q_scales);
    qkv_w_intx_hash = HashTensor<int8_t>(qkv_w_intx_t);
  }
  size_t qkv_w_max_hash = HashTensor<float>(qkv_w_max_t);
  std::string qkv_w_intx_name = std::to_string(qkv_w_intx_hash);
  std::string qkv_w_max_name = std::to_string(qkv_w_max_hash);
  std::string qkv_scale_max_name;

  *qkv_w_intx = FindNodeWithName(graph, qkv_w_intx_name);
  if (*qkv_w_intx == nullptr) {
    // Create qkv_w_intx node
    // Update qkv_w_intx var_desc in block
    VarDesc qkv_w_intx_desc(qkv_w_intx_name);
    qkv_w_intx_desc.SetPersistable(true);
    qkv_w_intx_desc.SetShape(common::vectorize(qkv_w_intx_t.dims()));
    qkv_w_intx_desc.SetDataType(
        framework::TransToProtoVarType(qkv_w_intx_t.dtype()));
    *qkv_w_intx = graph->CreateVarNode(&qkv_w_intx_desc);
    auto* block_qkv_w_intx_desc = block->Var(qkv_w_intx_name);
    block_qkv_w_intx_desc->SetPersistable(qkv_w_intx_desc.Persistable());
    block_qkv_w_intx_desc->SetShape(qkv_w_intx_desc.GetShape());
    block_qkv_w_intx_desc->SetDataType(qkv_w_intx_desc.GetDataType());
    // Create qkv_w_max node
    // Update qkv_w_max var_desc in block
    VarDesc qkv_w_max_desc(qkv_w_max_name);
    qkv_w_max_desc.SetPersistable(true);
    qkv_w_max_desc.SetShape(common::vectorize(qkv_w_max_t.dims()));
    qkv_w_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *qkv_w_max = graph->CreateVarNode(&qkv_w_max_desc);
    auto* block_qkv_w_max_desc = block->Var(qkv_w_max_name);
    block_qkv_w_max_desc->SetPersistable(qkv_w_max_desc.Persistable());
    block_qkv_w_max_desc->SetShape(qkv_w_max_desc.GetShape());
    block_qkv_w_max_desc->SetDataType(qkv_w_max_desc.GetDataType());

    if (is_per_channel) {
      size_t qkv_scale_max_hash = HashTensor<float>(qkv_scale_max_t);
      qkv_scale_max_name = std::to_string(qkv_scale_max_hash);
      // Create qkv_scale_max_t node
      // Update qkv_scale_max_t var_desc in block
      VarDesc qkv_scale_max_desc(qkv_scale_max_name);
      qkv_scale_max_desc.SetPersistable(true);
      qkv_scale_max_desc.SetShape(common::vectorize(qkv_scale_max_t.dims()));
      qkv_scale_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
      *qkv_scale_max = graph->CreateVarNode(&qkv_scale_max_desc);
      auto* block_qkv_scale_max_desc = block->Var(qkv_scale_max_name);
      block_qkv_scale_max_desc->SetPersistable(
          qkv_scale_max_desc.Persistable());
      block_qkv_scale_max_desc->SetShape(qkv_scale_max_desc.GetShape());
      block_qkv_scale_max_desc->SetDataType(qkv_scale_max_desc.GetDataType());
    }

    // Find qkv_w_intx/qkv_w_max variable in scope
    auto* qkv_w_intx_var = scope->FindVar(qkv_w_intx_name);
    if (qkv_w_intx_var == nullptr) {
      // Create qkv_w_intx/qkv_w_max variable/tensor
      Assign(qkv_w_intx_t,
             scope->Var(qkv_w_intx_name)->GetMutable<phi::DenseTensor>());
      Assign(qkv_w_max_t,
             scope->Var(qkv_w_max_name)->GetMutable<phi::DenseTensor>());
      if (is_per_channel) {
        Assign(qkv_scale_max_t,
               scope->Var(qkv_scale_max_name)->GetMutable<phi::DenseTensor>());
      }
    } else {
      // Share the same variable
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(qkv_w_max_name),
          common::errors::Fatal(
              "qkv_w_max(%s) variable should not be nullptr if qkv_w_intx(%s) "
              "variable is exist.",
              qkv_w_max_name,
              qkv_w_intx_name));
      if (is_per_channel) {
        PADDLE_ENFORCE_NOT_NULL(
            scope->FindVar(qkv_scale_max_name),
            common::errors::Fatal("qkv_scale_max(%s) variable should not be "
                                  "nullptr if qkv_w_intx(%s) "
                                  "variable is exist.",
                                  qkv_scale_max_name,
                                  qkv_w_intx_name));
      }
    }
  } else {
    *qkv_w_max = FindNodeWithName(graph, qkv_w_max_name);
    PADDLE_ENFORCE_NOT_NULL(
        *qkv_w_max,
        common::errors::Fatal(
            "qkv_w_max(%s) variable should not be nullptr if qkv_w_intx(%s) "
            "variable is exist.",
            qkv_w_max_name,
            qkv_w_intx_name));
  }
}

void MultiEncoderXPUFusePass::PrepareQKVBias(Graph* graph,
                                             Scope* scope,
                                             BlockDesc* block,
                                             Node* q_bias,
                                             Node* k_bias,
                                             Node* v_bias,
                                             Node** qkv_bias) const {
  auto* q_bias_tensor =
      scope->Var(q_bias->Name())->GetMutable<phi::DenseTensor>();
  auto* k_bias_tensor =
      scope->Var(k_bias->Name())->GetMutable<phi::DenseTensor>();
  auto* v_bias_tensor =
      scope->Var(v_bias->Name())->GetMutable<phi::DenseTensor>();
  phi::DenseTensor q_bias_fp32_tensor;
  phi::DenseTensor k_bias_fp32_tensor;
  phi::DenseTensor v_bias_fp32_tensor;
  CastToFp32(q_bias_tensor, &q_bias_fp32_tensor);
  CastToFp32(k_bias_tensor, &k_bias_fp32_tensor);
  CastToFp32(v_bias_tensor, &v_bias_fp32_tensor);

  phi::DenseTensor qkv_bias_tensor;
  int q_bias_fp32_size = q_bias_fp32_tensor.numel();
  qkv_bias_tensor.Resize(DDim({q_bias_fp32_size * 3}));
  qkv_bias_tensor.set_type(phi::DataType::FLOAT32);
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  auto* qkv_bias_data = cpu_ctx->Alloc<float>(&qkv_bias_tensor);
  memcpy(qkv_bias_data,
         q_bias_fp32_tensor.data(),
         q_bias_fp32_size * sizeof(float));
  qkv_bias_data += q_bias_fp32_size;
  memcpy(qkv_bias_data,
         k_bias_fp32_tensor.data(),
         q_bias_fp32_size * sizeof(float));
  qkv_bias_data += q_bias_fp32_size;
  memcpy(qkv_bias_data,
         v_bias_fp32_tensor.data(),
         q_bias_fp32_size * sizeof(float));

  size_t qkv_bias_hash = HashTensor<float>(qkv_bias_tensor);
  std::string qkv_bias_name = std::to_string(qkv_bias_hash);
  *qkv_bias = FindNodeWithName(graph, qkv_bias_name);
  if (*qkv_bias == nullptr) {
    // Create qkv_bias node
    // Update qkv_bias var_desc in block
    VarDesc qkv_bias_desc(qkv_bias_name);
    qkv_bias_desc.SetPersistable(true);
    qkv_bias_desc.SetShape(common::vectorize(qkv_bias_tensor.dims()));
    qkv_bias_desc.SetDataType(
        framework::TransToProtoVarType(qkv_bias_tensor.dtype()));
    *qkv_bias = graph->CreateVarNode(&qkv_bias_desc);
    auto* block_qkv_bias_desc = block->Var(qkv_bias_name);
    block_qkv_bias_desc->SetPersistable(qkv_bias_desc.Persistable());
    block_qkv_bias_desc->SetShape(qkv_bias_desc.GetShape());
    block_qkv_bias_desc->SetDataType(qkv_bias_desc.GetDataType());
    Assign(qkv_bias_tensor,
           scope->Var(qkv_bias_name)->GetMutable<phi::DenseTensor>());
  }
}

int MultiEncoderXPUFusePass::ApplySingleEncoderXPUFuse(
    ir::Graph* graph,
    const std::string& act_type,
    const std::string& matmul_type_0,
    const std::string& matmul_type_1,
    const std::string& matmul_type_2,
    bool norm_before,
    bool with_q_scale,
    bool with_mask,
    bool is_smooth_quant,
    const std::string& relative_type) const {
  bool local_quant = false;
  if (std::getenv("XPU_LOCAL_QUANT")) {
    local_quant = atoi(std::getenv("XPU_LOCAL_QUANT"));
  }
  GraphPatternDetector gpd;
  patterns::SingleEncoderXPUPattern pattern(gpd.mutable_pattern(),
                                            name_scope_,
                                            act_type,
                                            matmul_type_0,
                                            matmul_type_1,
                                            matmul_type_2,
                                            norm_before,
                                            with_q_scale,
                                            with_mask,
                                            is_smooth_quant,
                                            relative_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    // VLOG(4) << "handle MultiEncoderXPUFusePass fuse, step1";
    GET_IR_NODE(ln_0);
    GET_IR_NODE(ln_1);
    GET_IR_NODE(ln_2);
    GET_IR_NODE(q_matmul);
    GET_IR_NODE(q_add);
    GET_IR_NODE(q_reshape);
    GET_IR_NODE(q_transpose);
    GET_IR_NODE(q_scale);
    GET_IR_NODE(k_matmul);
    GET_IR_NODE(k_add);
    GET_IR_NODE(k_reshape);
    GET_IR_NODE(k_transpose);
    GET_IR_NODE(v_matmul);
    GET_IR_NODE(v_add);
    GET_IR_NODE(v_reshape);
    GET_IR_NODE(v_transpose);
    GET_IR_NODE(qk_matmul);
    GET_IR_NODE(qk_add);
    GET_IR_NODE(qk_softmax);
    GET_IR_NODE(qkv_matmul_0);
    GET_IR_NODE(qkv_transpose);
    GET_IR_NODE(qkv_reshape);
    GET_IR_NODE(qkv_matmul_1);
    GET_IR_NODE(qkv_add_0);
    GET_IR_NODE(qkv_add_1);
    GET_IR_NODE(qkv_matmul_2);
    GET_IR_NODE(qkv_add_2);
    GET_IR_NODE(qkv_act);
    GET_IR_NODE(qkv_matmul_3);
    GET_IR_NODE(qkv_add_3);
    GET_IR_NODE(qkv_add_4);
    GET_IR_NODE(ln_0_x);
    GET_IR_NODE(ln_0_bias);
    GET_IR_NODE(ln_0_scale);
    GET_IR_NODE(ln_0_out);
    GET_IR_NODE(ln_0_mean);
    GET_IR_NODE(ln_0_variance);
    GET_IR_NODE(q_matmul_w);
    GET_IR_NODE(q_matmul_out);
    GET_IR_NODE(q_add_bias);
    GET_IR_NODE(q_add_out);
    GET_IR_NODE(q_reshape_out);
    GET_IR_NODE(q_transpose_out);
    GET_IR_NODE(q_scale_out);
    GET_IR_NODE(k_matmul_w);
    GET_IR_NODE(k_matmul_out);
    GET_IR_NODE(k_add_bias);
    GET_IR_NODE(k_add_out);
    GET_IR_NODE(k_reshape_out);
    GET_IR_NODE(k_transpose_out);
    GET_IR_NODE(v_matmul_w);
    GET_IR_NODE(v_matmul_out);
    GET_IR_NODE(v_add_bias);
    GET_IR_NODE(v_add_out);
    GET_IR_NODE(v_reshape_out);
    GET_IR_NODE(v_transpose_out);
    GET_IR_NODE(qk_matmul_out);
    GET_IR_NODE(qk_add_mask);
    GET_IR_NODE(qk_add_out);
    GET_IR_NODE(qk_softmax_out);
    GET_IR_NODE(qkv_matmul_0_out);
    GET_IR_NODE(qkv_transpose_out);
    GET_IR_NODE(qkv_reshape_out);
    GET_IR_NODE(qkv_matmul_1_w);
    GET_IR_NODE(qkv_matmul_1_out);
    GET_IR_NODE(qkv_add_0_bias);
    GET_IR_NODE(qkv_add_0_out);
    GET_IR_NODE(qkv_add_1_out);
    GET_IR_NODE(ln_1_bias);
    GET_IR_NODE(ln_1_scale);
    GET_IR_NODE(ln_1_out);
    GET_IR_NODE(ln_1_mean);
    GET_IR_NODE(ln_1_variance);
    GET_IR_NODE(qkv_matmul_2_w);
    GET_IR_NODE(qkv_matmul_2_out);
    GET_IR_NODE(qkv_add_2_bias);
    GET_IR_NODE(qkv_add_2_out);
    GET_IR_NODE(qkv_act_out);
    GET_IR_NODE(qkv_matmul_3_w);
    GET_IR_NODE(qkv_matmul_3_out);
    GET_IR_NODE(qkv_add_3_bias);
    GET_IR_NODE(qkv_add_3_out);
    GET_IR_NODE(qkv_add_4_out);
    GET_IR_NODE(ln_2_x);
    GET_IR_NODE(ln_2_bias);
    GET_IR_NODE(ln_2_scale);
    GET_IR_NODE(ln_2_out);
    GET_IR_NODE(ln_2_mean);
    GET_IR_NODE(ln_2_variance);
    // smooth quant
    GET_IR_NODE(smooth_scale_1);
    GET_IR_NODE(smooth_scale_2);
    GET_IR_NODE(smooth_scale_1_weight);
    GET_IR_NODE(smooth_scale_2_weight);
    GET_IR_NODE(smooth_scale_1_out);
    GET_IR_NODE(smooth_scale_2_out);

    // roformer_relative_embedding_xpu
    GET_IR_NODE(q_relative_emb);
    GET_IR_NODE(q_cos_embedding);
    GET_IR_NODE(q_sin_embedding);
    GET_IR_NODE(q_relative_emb_out);
    GET_IR_NODE(k_relative_emb);
    GET_IR_NODE(k_cos_embedding);
    GET_IR_NODE(k_sin_embedding);
    GET_IR_NODE(k_relative_emb_out);

    auto* block = q_matmul->Op()->Block();
    auto* scope = param_scope();
    auto weight_dtype =
        scope->FindVar(q_matmul_w->Name())->Get<phi::DenseTensor>().dtype();
    std::string use_precision = "float32";
    if (weight_dtype == phi::DataType::INT8) {
      use_precision = "int8";
    } else if (weight_dtype == phi::DataType::FLOAT16) {
      use_precision = "float16";
    }
    bool is_per_channel = false;
    std::vector<Node*> input_max;
    std::unordered_map<std::string, std::vector<float>> var_quant_scales =
        GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
    std::vector<std::string> quant_types;
    std::vector<Node*> new_add_nodes;
    if (use_precision == "int8") {
      std::unordered_map<std::string, std::vector<Node*>> node_maps;
      std::vector<Node*> quant_mul_ops = {q_matmul,
                                          k_matmul,
                                          v_matmul,
                                          qkv_matmul_1,
                                          qkv_matmul_2,
                                          qkv_matmul_3};
      node_maps.insert(std::make_pair("quant_mul_ops", quant_mul_ops));
      std::vector<Node*> mul_add_ops = {
          q_add, k_add, v_add, qkv_add_0, qkv_act, qkv_add_3};
      node_maps.insert(std::make_pair("mul_add_ops", mul_add_ops));
      std::vector<Node*> matmul_ops = {qk_matmul, qkv_matmul_0};
      node_maps.insert(std::make_pair("matmul_ops", matmul_ops));
      std::vector<Node*> softmax_ops = {qk_softmax};
      node_maps.insert(std::make_pair("softmax_ops", softmax_ops));
      input_max.resize((quant_mul_ops.size() * 2 + quant_mul_ops.size() * 3),
                       nullptr);
      PrepareInputMax(graph,
                      scope,
                      block,
                      &node_maps,
                      &var_quant_scales,
                      &input_max,
                      &quant_types,
                      &act_type);
    }
    Node* qkv_w_intx = nullptr;
    Node* qkv_w_max = nullptr;
    Node* qkv_scale_max = nullptr;
    PrepareQKVWeight(graph,
                     scope,
                     block,
                     q_matmul_w,
                     k_matmul_w,
                     v_matmul_w,
                     (use_precision == "int8"),
                     local_quant,
                     &var_quant_scales,
                     &qkv_w_intx,
                     &qkv_w_max,
                     &qkv_scale_max);
    new_add_nodes.push_back(qkv_w_intx);
    new_add_nodes.push_back(qkv_w_max);
    new_add_nodes.push_back(qkv_scale_max);

#define PREPARE_QKV_MATMUL_W(idx_)                                  \
  Node* qkv_matmul_##idx_##_w_intx = nullptr;                       \
  Node* qkv_matmul_##idx_##_w_max = nullptr;                        \
  Node* qkv_matmul_##idx_##_scale_max = nullptr;                    \
  if (var_quant_scales.find(qkv_matmul_##idx_##_w->Name()) ==       \
      var_quant_scales.end()) {                                     \
    if (local_quant) {                                              \
      PrepareWeight<float, float16>(graph,                          \
                                    scope,                          \
                                    block,                          \
                                    qkv_matmul_##idx_##_w,          \
                                    &qkv_matmul_##idx_##_w_intx,    \
                                    &qkv_matmul_##idx_##_w_max,     \
                                    &qkv_matmul_##idx_##_scale_max, \
                                    true,                           \
                                    std::vector<float>({}));        \
    } else {                                                        \
      PrepareWeight<float, int16_t>(graph,                          \
                                    scope,                          \
                                    block,                          \
                                    qkv_matmul_##idx_##_w,          \
                                    &qkv_matmul_##idx_##_w_intx,    \
                                    &qkv_matmul_##idx_##_w_max,     \
                                    &qkv_matmul_##idx_##_scale_max, \
                                    true,                           \
                                    std::vector<float>({}));        \
    }                                                               \
  } else {                                                          \
    std::vector<float> weight_scales =                              \
        var_quant_scales.at(qkv_matmul_##idx_##_w->Name());         \
    is_per_channel = (weight_scales.size() != 1);                   \
    PrepareWeight<int8_t, int8_t>(graph,                            \
                                  scope,                            \
                                  block,                            \
                                  qkv_matmul_##idx_##_w,            \
                                  &qkv_matmul_##idx_##_w_intx,      \
                                  &qkv_matmul_##idx_##_w_max,       \
                                  &qkv_matmul_##idx_##_scale_max,   \
                                  true,                             \
                                  weight_scales);                   \
  }                                                                 \
  new_add_nodes.push_back(qkv_matmul_##idx_##_w_intx);              \
  new_add_nodes.push_back(qkv_matmul_##idx_##_w_max);               \
  new_add_nodes.push_back(qkv_matmul_##idx_##_scale_max);

    PREPARE_QKV_MATMUL_W(1);
    PREPARE_QKV_MATMUL_W(2);
    PREPARE_QKV_MATMUL_W(3);
#undef PREPARE_QKV_MATMUL_W

    Node* qkv_add_bias_fp32 = nullptr;
    PrepareQKVBias(graph,
                   scope,
                   block,
                   q_add_bias,
                   k_add_bias,
                   v_add_bias,
                   &qkv_add_bias_fp32);

    Node* qkv_add_0_bias_fp32 = nullptr;
    Node* qkv_add_2_bias_fp32 = nullptr;
    Node* qkv_add_3_bias_fp32 = nullptr;
    PrepareBias(graph, scope, block, qkv_add_0_bias, &qkv_add_0_bias_fp32);
    PrepareBias(graph, scope, block, qkv_add_2_bias, &qkv_add_2_bias_fp32);
    PrepareBias(graph, scope, block, qkv_add_3_bias, &qkv_add_3_bias_fp32);

    // Generate single_encoder_xpu op
    framework::OpDesc op_desc(block);
    op_desc.SetType("single_encoder_xpu");
    op_desc.SetInput("x", {ln_0_x->Name()});
    std::vector<std::string> fc_input_max_names;
    for (auto* node : input_max) {
      if (node) {
        fc_input_max_names.push_back(node->Name());
      }
    }
    op_desc.SetInput("fc_input_max", fc_input_max_names);
    std::vector<Node*> fc_weight_nodes = {
        qkv_matmul_1_w_intx, qkv_matmul_2_w_intx, qkv_matmul_3_w_intx};
    std::vector<std::string> fc_weight_names;
    fc_weight_names.push_back(qkv_w_intx->Name());
    for (size_t i = 0; i < fc_weight_nodes.size(); i++) {
      if (fc_weight_nodes[i]) {
        fc_weight_names.push_back(fc_weight_nodes[i]->Name());
      }
    }
    op_desc.SetInput("fc_weight", fc_weight_names);
    std::vector<std::string> fc_weight_max_names;

    std::vector<Node*> fc_weight_max_nodes = {
        qkv_w_max, qkv_matmul_1_w_max, qkv_matmul_2_w_max, qkv_matmul_3_w_max};
    std::vector<Node*> fc_weight_scale_nodes = {qkv_scale_max,
                                                qkv_matmul_1_scale_max,
                                                qkv_matmul_2_scale_max,
                                                qkv_matmul_3_scale_max};

    for (size_t i = 0; i < fc_weight_scale_nodes.size(); i++) {
      if (fc_weight_scale_nodes[i]) {
        fc_weight_max_names.push_back(fc_weight_scale_nodes[i]->Name());
      } else {
        fc_weight_max_names.push_back(fc_weight_max_nodes[i]->Name());
      }
    }
    op_desc.SetInput("fc_weight_max", fc_weight_max_names);
    op_desc.SetInput("fc_bias",
                     {qkv_add_bias_fp32->Name(),
                      qkv_add_0_bias_fp32->Name(),
                      qkv_add_2_bias_fp32->Name(),
                      qkv_add_3_bias_fp32->Name()});
    if (norm_before) {
      op_desc.SetInput("ln_scale", {ln_0_scale->Name(), ln_1_scale->Name()});
      op_desc.SetInput("ln_bias", {ln_0_bias->Name(), ln_1_bias->Name()});
    } else {
      op_desc.SetInput("ln_scale", {ln_1_scale->Name(), ln_2_scale->Name()});
      op_desc.SetInput("ln_bias", {ln_1_bias->Name(), ln_2_bias->Name()});
    }
    if (with_mask) {
      op_desc.SetInput("mask", {qk_add_mask->Name()});
    }
    std::vector<std::string> smooth_scale_names;
    if (is_smooth_quant) {
      smooth_scale_names.push_back(smooth_scale_1_weight->Name());
      smooth_scale_names.push_back(smooth_scale_2_weight->Name());
      for (auto smooth_scale_name : smooth_scale_names) {
        auto* in =
            scope->FindVar(smooth_scale_name)->GetMutable<phi::DenseTensor>();
        CastToFp16(in);
      }
    }
    op_desc.SetInput("smooth_scale_weight", {smooth_scale_names});
    op_desc.SetAttr("norm_before", norm_before);
    op_desc.SetAttr("hidden_dim",
                    static_cast<int>(q_matmul_w->Var()->GetShape()[0]));
    auto q_reshape_shape =
        PADDLE_GET_CONST(std::vector<int>, q_reshape->Op()->GetAttr("shape"));
    op_desc.SetAttr("head_num", q_reshape_shape[2]);
    op_desc.SetAttr("size_per_head", q_reshape_shape[3]);
    auto qkv_matmul_2_w_shape = qkv_matmul_2_w->Var()->GetShape();
    op_desc.SetAttr(
        "ffn_hidden_dim_scale",
        static_cast<int>(qkv_matmul_2_w_shape[1] / qkv_matmul_2_w_shape[0]));
    op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    op_desc.SetAttr("relative_type", static_cast<int>(0));
    op_desc.SetAttr("use_precision", use_precision);
    op_desc.SetAttr("is_per_channel", is_per_channel);
    if (relative_type == "roformer_relative_embedding_xpu") {
      // q/k share the rotary embedding
      op_desc.SetInput("roformer_embedding",
                       {q_cos_embedding->Name(), q_sin_embedding->Name()});
      op_desc.SetAttr("relative_type", 1);
      auto q_cos_emb_shape = q_cos_embedding->Var()->GetShape();
      PADDLE_ENFORCE_GE(static_cast<int>(q_cos_emb_shape.size()),
                        2,
                        common::errors::InvalidArgument(
                            "The rank of q_cos_embedding should be greater "
                            "than or equal to 2"));

      auto size_per_head = q_reshape_out->Var()->GetShape()[3];
      PADDLE_ENFORCE_EQ(
          size_per_head,
          q_cos_emb_shape[q_cos_emb_shape.size() - 1],
          common::errors::InvalidArgument(
              "The last dimension of q_cos_embedding should be %d "
              "equal to size_per_head, but received %d.",
              size_per_head,
              q_cos_emb_shape[q_cos_emb_shape.size() - 1]));
      int max_pos_len = q_cos_emb_shape[q_cos_emb_shape.size() - 2];
      VLOG(3) << "relative embedding max sequence len: " << max_pos_len;
      op_desc.SetAttr("max_pos_len", max_pos_len);
    } else {
      op_desc.SetInput("roformer_embedding", {});
      op_desc.SetAttr("max_pos_len", 0);
    }

    // if quant,skip softmax,and use qk_matmul out_threshold as softmax_max
    auto softmax_max_name = qk_matmul->Op()->Output("Out")[0];
    if (var_quant_scales.find(softmax_max_name) != var_quant_scales.end()) {
      op_desc.SetAttr("softmax_max_value",
                      var_quant_scales.at(softmax_max_name)[0]);
    }
    op_desc.SetAttr("quant_types", quant_types);
    if (norm_before) {
      op_desc.SetOutput("out", {qkv_add_4_out->Name()});
    } else {
      op_desc.SetOutput("out", {ln_2_out->Name()});
    }
    auto* single_encoder_xpu = graph->CreateOpNode(&op_desc);
    // Link nodes
    IR_NODE_LINK_TO(ln_0_x, single_encoder_xpu);
    for (auto* node : input_max) {
      if (node) {
        IR_NODE_LINK_TO(node, single_encoder_xpu);
      }
    }
    for (auto* node : new_add_nodes) {
      if (node) {
        IR_NODE_LINK_TO(node, single_encoder_xpu);
      }
    }
    IR_NODE_LINK_TO(qkv_add_bias_fp32, single_encoder_xpu);
    IR_NODE_LINK_TO(qkv_add_0_bias_fp32, single_encoder_xpu);
    IR_NODE_LINK_TO(qkv_add_2_bias_fp32, single_encoder_xpu);
    IR_NODE_LINK_TO(qkv_add_3_bias_fp32, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_0_scale, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_0_bias, single_encoder_xpu);
    IR_NODE_LINK_TO(ln_1_scale, single_encoder_xpu);
    IR_NODE_LINK_TO(ln_1_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_2_scale, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_2_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qk_add_mask, single_encoder_xpu);
    if (norm_before) {
      IR_NODE_LINK_TO(single_encoder_xpu, qkv_add_4_out);
    } else {
      IR_NODE_LINK_TO(single_encoder_xpu, ln_2_out);
    }
    if (is_smooth_quant) {
      IR_NODE_LINK_TO(smooth_scale_1_weight, single_encoder_xpu);
      IR_NODE_LINK_TO(smooth_scale_2_weight, single_encoder_xpu);
    }
    if (relative_type == "roformer_relative_embedding_xpu") {
      IR_NODE_LINK_TO(q_cos_embedding, single_encoder_xpu);
      IR_NODE_LINK_TO(q_sin_embedding, single_encoder_xpu);
    }

    // Delete nodes
    std::unordered_set<const Node*> delete_nodes{ln_1,
                                                 q_matmul,
                                                 q_add,
                                                 q_reshape,
                                                 q_transpose,
                                                 k_matmul,
                                                 k_add,
                                                 k_reshape,
                                                 k_transpose,
                                                 v_matmul,
                                                 v_add,
                                                 v_reshape,
                                                 v_transpose,
                                                 qk_matmul,
                                                 qk_softmax,
                                                 qkv_matmul_0,
                                                 qkv_transpose,
                                                 qkv_reshape,
                                                 qkv_matmul_1,
                                                 qkv_add_0,
                                                 qkv_add_1,
                                                 qkv_matmul_2,
                                                 qkv_add_2,
                                                 qkv_act,
                                                 qkv_matmul_3,
                                                 qkv_add_3,
                                                 qkv_add_4,
                                                 q_matmul_w,
                                                 q_matmul_out,
                                                 q_add_out,
                                                 q_reshape_out,
                                                 q_transpose_out,
                                                 k_matmul_w,
                                                 k_matmul_out,
                                                 k_add_out,
                                                 k_reshape_out,
                                                 k_transpose_out,
                                                 v_matmul_w,
                                                 v_matmul_out,
                                                 v_add_out,
                                                 v_reshape_out,
                                                 v_transpose_out,
                                                 qk_matmul_out,
                                                 qk_softmax_out,
                                                 qkv_matmul_0_out,
                                                 qkv_transpose_out,
                                                 qkv_reshape_out,
                                                 qkv_matmul_1_out,
                                                 qkv_add_0_out,
                                                 qkv_add_1_out,
                                                 ln_1_out,
                                                 ln_1_mean,
                                                 ln_1_variance,
                                                 qkv_matmul_2_out,
                                                 qkv_add_2_out,
                                                 qkv_act_out,
                                                 qkv_matmul_3_out,
                                                 qkv_add_3_out};
    if (norm_before) {
      delete_nodes.insert(ln_0);
      delete_nodes.insert(ln_0_mean);
      delete_nodes.insert(ln_0_variance);
      delete_nodes.insert(ln_0_out);
    } else {
      delete_nodes.insert(qkv_add_4_out);
      delete_nodes.insert(ln_2);
      delete_nodes.insert(ln_2_mean);
      delete_nodes.insert(ln_2_variance);
    }
    if (with_q_scale) {
      delete_nodes.insert(q_scale);
      delete_nodes.insert(q_scale_out);
    }
    if (with_mask) {
      delete_nodes.insert(qk_add);
      delete_nodes.insert(qk_add_out);
    }
    if (is_smooth_quant) {
      delete_nodes.insert(smooth_scale_1);
      delete_nodes.insert(smooth_scale_2);
      delete_nodes.insert(smooth_scale_1_out);
      delete_nodes.insert(smooth_scale_2_out);
    }
    if (relative_type == "roformer_relative_embedding_xpu") {
      delete_nodes.insert(q_relative_emb);
      delete_nodes.insert(q_relative_emb_out);
      delete_nodes.insert(k_relative_emb);
      delete_nodes.insert(k_relative_emb_out);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

static std::vector<Node*> GetSingleEncoders(ir::Graph* graph) {
  std::vector<Node*> single_encoders;
  for (auto* node : graph->Nodes()) {
    // Find first single_encoder_xpu
    if (node->IsVar() || node->Op()->Type() != "single_encoder_xpu") continue;
    bool is_first_encoder = true;
    for (auto* in_node : node->inputs) {
      if (in_node->Var()->Persistable()) continue;
      if (in_node->inputs[0]->Op()->Type() == "single_encoder_xpu") {
        is_first_encoder = false;
        break;
      }
    }
    if (!is_first_encoder) continue;
    // Add continuous single_encoder_xpu
    single_encoders.push_back(node);
    while (true) {
      auto next_ops = single_encoders.back()->outputs[0]->outputs;
      if (next_ops.empty()) break;
      auto next_op_type = next_ops[0]->Op()->Type();
      if (next_op_type != "single_encoder_xpu") break;
      single_encoders.push_back(next_ops[0]);
    }
    break;
  }
  return single_encoders;
}

bool MultiEncoderXPUFusePass::ApplyMultiEncoderXPUFuse(ir::Graph* graph) const {
  auto single_encoders = GetSingleEncoders(graph);
  if (single_encoders.empty()) return false;

  // Prepare inputs/outputs names/nodes
  std::string x_name = single_encoders[0]->Op()->Inputs().at("x")[0];
  std::vector<std::string> arg_names{"fc_input_max",
                                     "fc_weight",
                                     "fc_weight_max",
                                     "fc_bias",
                                     "ln_scale",
                                     "ln_bias",
                                     "smooth_scale_weight",
                                     "roformer_embedding"};
  std::map<std::string, std::vector<std::string>> arg_names_map;
  std::string mask_name = single_encoders[0]->Op()->Inputs().count("mask") > 0
                              ? single_encoders[0]->Op()->Inputs().at("mask")[0]
                              : "";
  std::string out_name = single_encoders.back()->Op()->Outputs().at("out")[0];

  std::vector<Node*> in_nodes;
  for (auto* in_node : single_encoders[0]->inputs) {
    if (in_node->Var()->Name() == x_name ||
        in_node->Var()->Name() == mask_name) {
      in_nodes.push_back(in_node);
    }
  }
  for (auto* single_encoder : single_encoders) {
    auto single_encoder_in_nodes = single_encoder->inputs;
    for (auto arg_name : arg_names) {
      auto var_names = single_encoder->Op()->Inputs().at(arg_name);
      for (auto var_name : var_names) {
        arg_names_map[arg_name].push_back(var_name);
        for (auto in_node : single_encoder_in_nodes) {
          if (in_node->Var()->Name() == var_name) {
            in_nodes.push_back(in_node);
          }
        }
      }
    }
  }

  std::vector<Node*> out_nodes;
  for (auto* out_node : single_encoders.back()->outputs) {
    if (out_node->Var()->Name() == out_name) {
      out_nodes.push_back(out_node);
      break;
    }
  }

  auto* block = single_encoders[0]->Op()->Block();
  auto* scope = param_scope();
  // Create x_fp16 variable/mode/tensor
  std::string x_fp16_name = x_name + "_fp16";
  VarDesc x_fp16_desc(x_fp16_name);
  auto* x_fp16 = graph->CreateVarNode(&x_fp16_desc);
  block->Var(x_fp16_name);
  scope->Var(x_fp16_name)->GetMutable<phi::DenseTensor>();
  out_nodes.push_back(x_fp16);
  // Create out_fp16 variable/mode/tensor
  std::string out_fp16_name = out_name + "_fp16";
  VarDesc out_fp16_desc(out_fp16_name);
  auto* out_fp16 = graph->CreateVarNode(&out_fp16_desc);
  block->Var(out_fp16_name);
  scope->Var(out_fp16_name)->GetMutable<phi::DenseTensor>();
  out_nodes.push_back(out_fp16);

  // Generate multi_encoder_xpu op
  framework::OpDesc op_desc(block);
  op_desc.SetType("multi_encoder_xpu");
  op_desc.SetInput("x", {x_name});
  for (auto arg_name : arg_names) {
    op_desc.SetInput(arg_name, arg_names_map[arg_name]);
  }
  if (!mask_name.empty()) {
    op_desc.SetInput("mask", {mask_name});
  }
  op_desc.SetAttr("layer_num", static_cast<int>(single_encoders.size()));
  op_desc.SetAttr(
      "norm_before",
      PADDLE_GET_CONST(bool, single_encoders[0]->Op()->GetAttr("norm_before")));
  for (auto attr_name : {"hidden_dim",
                         "head_num",
                         "size_per_head",
                         "ffn_hidden_dim_scale",
                         "act_type",
                         "relative_type"}) {
    op_desc.SetAttr(
        attr_name,
        PADDLE_GET_CONST(int, single_encoders[0]->Op()->GetAttr(attr_name)));
  }
  op_desc.SetAttr("slice_idx", static_cast<int>(-1));
  op_desc.SetAttr(
      "use_precision",
      PADDLE_GET_CONST(std::string,
                       single_encoders[0]->Op()->GetAttr("use_precision")));
  op_desc.SetAttr(
      "is_per_channel",
      PADDLE_GET_CONST(bool,
                       single_encoders[0]->Op()->GetAttr("is_per_channel")));
  std::vector<float> softmax_max_values;
  for (auto* single_encoder : single_encoders) {
    if (single_encoder->Op()->HasAttr("softmax_max_value")) {
      softmax_max_values.push_back(PADDLE_GET_CONST(
          float, single_encoder->Op()->GetAttr("softmax_max_value")));
    }
  }
  op_desc.SetAttr("softmax_max_value", softmax_max_values);
  std::vector<std::string> quant_types;
  for (auto* single_encoder : single_encoders) {
    auto per_quant_types = PADDLE_GET_CONST(
        std::vector<std::string>, single_encoder->Op()->GetAttr("quant_types"));
    quant_types.insert(
        quant_types.end(), per_quant_types.begin(), per_quant_types.end());
  }
  op_desc.SetAttr("quant_types", quant_types);
  if (single_encoders[0]->Op()->HasAttr("max_pos_len")) {
    op_desc.SetAttr("max_pos_len",
                    PADDLE_GET_CONST(
                        int, single_encoders[0]->Op()->GetAttr("max_pos_len")));
  }
  op_desc.SetOutput("out", {out_name});
  op_desc.SetOutput("x_fp16", {x_fp16_name});
  op_desc.SetOutput("out_fp16", {out_fp16_name});
  auto* multi_encoder_xpu = graph->CreateOpNode(&op_desc);
  for (auto* in_node : in_nodes) {
    IR_NODE_LINK_TO(in_node, multi_encoder_xpu);
  }
  for (auto* out_node : out_nodes) {
    IR_NODE_LINK_TO(multi_encoder_xpu, out_node);
  }

  // delete useless node
  std::unordered_set<const Node*> delete_nodes(single_encoders.begin(),
                                               single_encoders.end());
  for (int i = 0; i < static_cast<int>(single_encoders.size()) - 1; i++) {
    std::string out_name = single_encoders[i]->Op()->Outputs().at("out")[0];
    for (auto* out_node : single_encoders[i]->outputs) {
      if (out_node->Var()->Name() != out_name) {
        delete_nodes.insert(out_node);
      }
    }
  }
  GraphSafeRemoveNodes(graph, delete_nodes);

  return true;
}

int MultiEncoderXPUFusePass::CastMask(ir::Graph* graph) const {
  int cast_counts = 0;
  auto nodes = graph->Nodes();
  for (auto node : nodes) {
    if (node->IsVar()) continue;
    auto op_desc = node->Op();
    auto use_precision = op_desc->GetAttrIfExists<std::string>("use_precision");
    if (node->IsVar() ||  //
        op_desc->Type() != "multi_encoder_xpu" ||
        (use_precision != "float16") || op_desc->Inputs().count("mask") == 0)
      continue;

    auto* block = op_desc->Block();
    auto* scope = param_scope();

    // Find mask node
    std::string mask_name = op_desc->Inputs().at("mask")[0];
    Node* mask = nullptr;
    for (auto* in_node : node->inputs) {
      if (in_node->Var()->Name() == mask_name) {
        mask = in_node;
        break;
      }
    }

    // Create new_mask node/var/tensor
    std::string new_mask_name = mask_name + "_fp32";
    VarDesc new_mask_desc(new_mask_name);
    auto* new_mask = graph->CreateVarNode(&new_mask_desc);
    block->Var(new_mask_name);
    scope->Var(new_mask_name)->GetMutable<phi::DenseTensor>();

    // Create cast op
    framework::OpDesc cast_op_desc(block);
    cast_op_desc.SetType("cast");
    cast_op_desc.SetInput("X", {mask_name});
    cast_op_desc.SetAttr("in_dtype",
                         static_cast<int>(framework::proto::VarType::FP16));
    cast_op_desc.SetAttr("out_dtype",
                         static_cast<int>(framework::proto::VarType::FP32));
    cast_op_desc.SetOutput("Out", {new_mask_name});
    auto* cast = graph->CreateOpNode(&cast_op_desc);
    IR_NODE_LINK_TO(mask, cast);
    IR_NODE_LINK_TO(cast, new_mask);

    // Update encoder
    op_desc->SetInput("mask", {new_mask_name});
    IR_NODE_LINK_TO(new_mask, node);
    IR_NODE_UNLINK(node, mask);

    cast_counts++;
  }
  return cast_counts;
}

std::vector<PatternParam> MultiEncoderXPUFusePass::GeneratePatternParams()
    const {
  return std::vector<PatternParam>{
      // Params are arranged in alphabetic order
      {"gelu",
       "matmul_v2",
       "matmul",
       "matmul_v2",
       false,
       false,
       true,
       false,
       ""},
      {"gelu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       false,
       ""},
      {"gelu", "mul", "matmul", "matmul", false, true, true, false, ""},
      {"relu", "mul", "matmul", "matmul", false, true, true, false, ""},
      {"relu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       false,
       ""},

      {"gelu",
       "matmul_v2",
       "matmul",
       "matmul_v2",
       false,
       false,
       true,
       true,
       ""},
      {"gelu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       true,
       ""},
      {"gelu", "mul", "matmul", "matmul", false, true, true, true, ""},
      {"relu", "mul", "matmul", "matmul", false, true, true, true, ""},
      {"relu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       true,
       ""},

      {"gelu",
       "matmul_v2",
       "matmul",
       "matmul_v2",
       false,
       false,
       true,
       false,
       "roformer_relative_embedding_xpu"},
      {"gelu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       false,
       "roformer_relative_embedding_xpu"},
      {"gelu",
       "mul",
       "matmul",
       "matmul",
       false,
       true,
       true,
       false,
       "roformer_relative_embedding_xpu"},
      {"relu",
       "mul",
       "matmul",
       "matmul",
       false,
       true,
       true,
       false,
       "roformer_relative_embedding_xpu"},
      {"relu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       false,
       "roformer_relative_embedding_xpu"},

      {"gelu",
       "matmul_v2",
       "matmul",
       "matmul_v2",
       false,
       false,
       true,
       true,
       "roformer_relative_embedding_xpu"},
      {"gelu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       true,
       "roformer_relative_embedding_xpu"},
      {"gelu",
       "mul",
       "matmul",
       "matmul",
       false,
       true,
       true,
       true,
       "roformer_relative_embedding_xpu"},
      {"relu",
       "mul",
       "matmul",
       "matmul",
       false,
       true,
       true,
       true,
       "roformer_relative_embedding_xpu"},
      {"relu",
       "matmul_v2",
       "matmul_v2",
       "matmul_v2",
       false,
       true,
       true,
       true,
       "roformer_relative_embedding_xpu"},
  };
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_encoder_xpu_fuse_pass,
              paddle::framework::ir::MultiEncoderXPUFusePass);

REGISTER_PASS_CAPABILITY(multi_encoder_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "multi_encoder_xpu", 0));
