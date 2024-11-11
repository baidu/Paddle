/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for multiary operators, The format like:
//
//   1. The number of input MetaTensor is more than 3:
//      void [FunctionDesc|OpName]InferMeta(const MetaTensor& x,
//                                          const MetaTensor& y,
//                                          const MetaTensor& z,
//                                          const MetaTensor& w,
//                                          ...,
//                                          MetaTensor* out) {}
//
//   2. There are `const vector<MetaTensor*>&` in params:
//      void [FunctionDesc|OpName]InferMeta(const vector<MetaTensor*>& x,
//                                          ...,
//                                          MetaTensor* out) {}
//
// NOTE: The InferMeta Functions in this file are arranged in alphabetic order.

std::vector<DDim> GetMetaTensorsDim(
    const std::vector<const MetaTensor*>& tensors);

void AdadeltaInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& avg_squared_grad,
                       const MetaTensor& avg_squared_update,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float rho,
                       float epsilon,
                       bool multi_precision,
                       MetaTensor* param_out,
                       MetaTensor* avg_squared_grad_out,
                       MetaTensor* avg_squared_update_out,
                       MetaTensor* master_param_outs);

void AdagradInferMeta(const MetaTensor& param,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& master_param,
                      float epsilon,
                      bool multi_precision,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* master_param_out);

void AdamaxInferMeta(const MetaTensor& param,
                     const MetaTensor& grad,
                     const MetaTensor& learning_rate,
                     const MetaTensor& moment,
                     const MetaTensor& inf_norm,
                     const MetaTensor& beta1_pow,
                     const MetaTensor& master_param,
                     float beta1,
                     float beta2,
                     float epsilon,
                     bool multi_precision,
                     MetaTensor* param_out,
                     MetaTensor* moment_out,
                     MetaTensor* inf_norm_out,
                     MetaTensor* master_param_outs);

void AdamInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   const Scalar& beta1,
                   const Scalar& beta2,
                   const Scalar& epsilon,
                   bool lazy_mode,
                   int64_t min_row_size_to_use_multithread,
                   bool multi_precision,
                   bool use_global_beta_pow,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs);

void AdamwInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& beta1_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& master_param,
                    const MetaTensor& skip_update,
                    const Scalar& beta1,
                    const Scalar& beta2,
                    const Scalar& epsilon,
                    float lr_ratio,
                    float coeff,
                    bool with_decay,
                    bool lazy_mode,
                    int64_t min_row_size_to_use_multithread,
                    bool multi_precision,
                    bool use_global_beta_pow,
                    MetaTensor* param_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* beta1_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* master_param_outs);

void AddNInferMeta(const std::vector<const MetaTensor*>& x,
                   MetaTensor* out,
                   MetaConfig config = MetaConfig());

void AddNTensorArrayInferMeta(const std::vector<const MetaTensor*>& x,
                              MetaTensor* out,
                              MetaConfig config);

void ASGDInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& d,
                   const MetaTensor& y,
                   const MetaTensor& n,
                   const MetaTensor& master_param,
                   bool multi_precision,
                   MetaTensor* param_out,
                   MetaTensor* d_out,
                   MetaTensor* y_out,
                   MetaTensor* master_param_out);

void AttentionLstmInferMeta(const MetaTensor& x,
                            const MetaTensor& c0,
                            const MetaTensor& h0,
                            const MetaTensor& attention_weight,
                            const MetaTensor& attention_bias,
                            const MetaTensor& attention_scalar,
                            const MetaTensor& attention_scalar_bias,
                            const MetaTensor& lstm_weight,
                            const MetaTensor& lstm_bias,
                            const std::string& gate_activation,
                            const std::string& cell_activation,
                            const std::string& candidate_activation,
                            MetaTensor* hidden,
                            MetaTensor* cell,
                            MetaTensor* attentioned_x,
                            MetaTensor* attention_fc_out,
                            MetaTensor* lstm_x,
                            MetaTensor* lstm_out,
                            MetaConfig config = MetaConfig());

void AucInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& stat_pos,
                  const MetaTensor& stat_neg,
                  const MetaTensor& ins_tag_weight,
                  const std::string& curve,
                  int num_thresholds,
                  int slide_steps,
                  MetaTensor* auc,
                  MetaTensor* stat_pos_out,
                  MetaTensor* stat_neg_out,
                  MetaConfig config = MetaConfig());

void AverageAccumulatesInferMeta(const MetaTensor& param,
                                 const MetaTensor& in_sum_1,
                                 const MetaTensor& in_sum_2,
                                 const MetaTensor& in_sum_3,
                                 const MetaTensor& in_num_accumulates,
                                 const MetaTensor& in_old_num_accumulates,
                                 const MetaTensor& in_num_updates,
                                 float average_window,
                                 int64_t max_average_window,
                                 int64_t min_average_window,
                                 MetaTensor* out_sum_1,
                                 MetaTensor* out_sum_2,
                                 MetaTensor* out_sum_3,
                                 MetaTensor* out_num_accumulates,
                                 MetaTensor* out_old_num_accumulates,
                                 MetaTensor* out_num_updates);

void BatchNormInferMeta(const MetaTensor& x,
                        const MetaTensor& mean,
                        const MetaTensor& variance,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        bool is_test,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout,
                        bool use_global_stats,
                        bool trainable_statistics,
                        MetaTensor* y,
                        MetaTensor* mean_out,
                        MetaTensor* variance_out,
                        MetaTensor* saved_mean,
                        MetaTensor* saved_variance,
                        MetaTensor* reserve_space,
                        MetaConfig config = MetaConfig());

void BatchNormInferInferMeta(const MetaTensor& x,
                             const MetaTensor& mean,
                             const MetaTensor& variance,
                             const MetaTensor& scale,
                             const MetaTensor& bias,
                             float momentum,
                             float epsilon,
                             const std::string& data_layout,
                             MetaTensor* y,
                             MetaTensor* mean_out,
                             MetaTensor* variance_out,
                             MetaConfig config = MetaConfig());

void BeamSearchInferMeta(const MetaTensor& pre_ids,
                         const MetaTensor& pre_scores,
                         const MetaTensor& ids,
                         const MetaTensor& scores,
                         int level,
                         int beam_size,
                         int end_id,
                         bool is_accumulated,
                         MetaTensor* selected_ids,
                         MetaTensor* selected_scores,
                         MetaTensor* parent_idx);

void BilinearInferMeta(const MetaTensor& x,
                       const MetaTensor& y,
                       const MetaTensor& weight,
                       const MetaTensor& bias,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void BroadcastTensorsInferMeta(const std::vector<const MetaTensor*>& x,
                               std::vector<MetaTensor*> out);

void CheckFiniteAndUnscaleInferMeta(const std::vector<const MetaTensor*>& xs,
                                    const MetaTensor& scale,
                                    std::vector<MetaTensor*> outs,
                                    MetaTensor* found_infinite);

void CoalesceTensorInferMeta(const std::vector<const MetaTensor*>& input,
                             DataType dtype,
                             bool copy_data,
                             bool set_constant,
                             bool persist_output,
                             float constant,
                             bool use_align,
                             int align_size,
                             int size_of_dtype,
                             const std::vector<int64_t>& concated_shapes,
                             const std::vector<int64_t>& concated_ranks,
                             std::vector<MetaTensor*> output,
                             MetaTensor* fused_output,
                             MetaConfig config = MetaConfig());

void CheckMemoryContinueInferMeta(const std::vector<const MetaTensor*>& input,
                                  MetaTensor* output,
                                  std::vector<MetaTensor*> xout,
                                  MetaConfig config = MetaConfig());

void ConcatInferMeta(const std::vector<const MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void ChunkEvalInferMeta(const MetaTensor& inference,
                        const MetaTensor& label,
                        const MetaTensor& seq_length,
                        int num_chunk_types,
                        const std::string& chunk_scheme,
                        const std::vector<int>& excluded_chunk_types,
                        MetaTensor* precision,
                        MetaTensor* recall,
                        MetaTensor* f1_score,
                        MetaTensor* num_infer_chunks,
                        MetaTensor* num_label_chunks,
                        MetaTensor* num_correct_chunks);

void CrfDecodingInferMeta(const MetaTensor& emission,
                          const MetaTensor& transition,
                          const MetaTensor& label,
                          const MetaTensor& length,
                          MetaTensor* viterbi_path,
                          MetaConfig config = MetaConfig());

void CudnnLSTMInferMeta(
    const MetaTensor& x,
    const MetaTensor& init_h,
    const MetaTensor& init_c,
    const MetaTensor& w,
    const paddle::optional<std::vector<const MetaTensor*>>& weight_list,
    const MetaTensor& sequence_length,
    float dropout_prob,
    bool is_bidirec,
    int hidden_size,
    int num_layers,
    bool is_test,
    int seed,
    MetaTensor* out,
    MetaTensor* last_h,
    MetaTensor* last_c,
    MetaTensor* reserve,
    MetaTensor* state_out);

void LSTMInferMeta(const MetaTensor& input,
                   const MetaTensor& h0,
                   const MetaTensor& c0,
                   const MetaTensor& weight,
                   const MetaTensor& bias,
                   bool use_peepholes,
                   bool is_reverse,
                   bool is_test,
                   const std::string& gate_activation,
                   const std::string& cell_activation,
                   const std::string& candidate_activation,
                   MetaTensor* hidden,
                   MetaTensor* cell,
                   MetaTensor* batch_gate,
                   MetaTensor* batch_cell_pre_act,
                   MetaConfig config = MetaConfig());

void DecayedAdagradInferMeta(const MetaTensor& param,
                             const MetaTensor& grad,
                             const MetaTensor& moment,
                             const MetaTensor& learning_rate,
                             float decay,
                             float epsilon,
                             MetaTensor* param_out,
                             MetaTensor* moment_out);

void DeformableConvInferMeta(const MetaTensor& x,
                             const MetaTensor& offset,
                             const MetaTensor& filter,
                             const MetaTensor& mask,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& dilations,
                             int deformable_groups,
                             int groups,
                             int im2col_step,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

void DetectionMapInferMeta(const MetaTensor& detect_res,
                           const MetaTensor& label,
                           const MetaTensor& has_state,
                           const MetaTensor& pos_count,
                           const MetaTensor& true_pos,
                           const MetaTensor& false_pos,
                           int class_num,
                           int background_label,
                           float overlap_threshold,
                           bool evaluate_difficult,
                           const std::string& ap_type,
                           MetaTensor* accum_pos_count,
                           MetaTensor* accum_true_pos,
                           MetaTensor* accum_false_pos,
                           MetaTensor* m_ap,
                           MetaConfig config = MetaConfig());

void DgcInferMeta(const MetaTensor& u,
                  const MetaTensor& v,
                  const MetaTensor& grad,
                  const MetaTensor& param,
                  const MetaTensor& current_step_tensor,
                  const MetaTensor& nranks_tensor,
                  MetaTensor* u_out,
                  MetaTensor* v_out,
                  MetaTensor* encode_grad_out,
                  MetaTensor* grad_out,
                  MetaTensor* k_out,
                  MetaTensor* gather_buff);

void DGCMomentumInferMeta(const MetaTensor& param,
                          const MetaTensor& grad,
                          const MetaTensor& velocity,
                          const MetaTensor& learning_rate,
                          const MetaTensor& master_param,
                          const MetaTensor& current_step_tensor,
                          const MetaTensor& nranks_tensor,
                          float mu,
                          bool use_nesterov,
                          const std::string& regularization_method,
                          float regularization_coeff,
                          bool multi_precision,
                          float rescale_grad,
                          float rampup_begin_step,
                          MetaTensor* param_out,
                          MetaTensor* velocity_out,
                          MetaTensor* master_param_out,
                          MetaTensor* grad_out);

void EditDistanceInferMeta(const MetaTensor& hyps,
                           const MetaTensor& refs,
                           const MetaTensor& hypslength,
                           const MetaTensor& refslength,
                           bool normalized,
                           MetaTensor* sequencenum,
                           MetaTensor* out);

void FakeChannelWiseDequantizeMaxAbsInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& scales,
    const std::vector<int>& quant_bits,
    int quant_axis,
    int x_num_col_dims,
    MetaTensor* out);

void FakeQuantOrWithDequantMovingAverageAbsMaxInferMeta(
    const MetaTensor& x,
    const MetaTensor& in_scale,
    const MetaTensor& in_accum,
    const MetaTensor& in_state,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    MetaTensor* out,
    MetaTensor* out_scale,
    MetaTensor* out_state,
    MetaTensor* out_accum);

void FtrlInferMeta(const MetaTensor& param,
                   const MetaTensor& squared_accumulator,
                   const MetaTensor& linear_accumulator,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   float l1,
                   float l2,
                   float lr_power,
                   MetaTensor* param_out,
                   MetaTensor* squared_accum_out,
                   MetaTensor* linear_accum_out);

void FusedBatchNormActInferMeta(const MetaTensor& x,
                                const MetaTensor& scale,
                                const MetaTensor& bias,
                                const MetaTensor& mean,
                                const MetaTensor& variance,
                                MetaTensor* y,
                                MetaTensor* mean_out,
                                MetaTensor* variance_out,
                                MetaTensor* saved_mean,
                                MetaTensor* saved_variance,
                                MetaTensor* reserve_space);

void FusedBiasActInferMeta(const MetaTensor& x,
                           const MetaTensor& bias,
                           const MetaTensor& dequant_scales,
                           const MetaTensor& shift,
                           const MetaTensor& smooth,
                           const std::string& act_method,
                           const std::string& compute_dtype,
                           float quant_scale,
                           int quant_round_type,
                           float quant_max_bound,
                           float quant_min_bound,
                           MetaTensor* out,
                           MetaConfig config = MetaConfig());

void FusedLayerNormInferMeta(const MetaTensor& x,
                             const MetaTensor& bias,
                             const MetaTensor& residual,
                             const MetaTensor& norm_weight,
                             const MetaTensor& norm_bias,
                             const float epsilon,
                             const float residual_alpha,
                             const int begin_norm_axis,
                             const float quant_scale,
                             const int quant_round_type,
                             const float quant_max_bound,
                             const float quant_min_bound,
                             MetaTensor* out,
                             MetaTensor* residual_out,
                             MetaTensor* mean,
                             MetaTensor* variance,
                             MetaConfig config = MetaConfig());

void FusedLinearParamGradAddInferMeta(const MetaTensor& x,
                                      const MetaTensor& dout,
                                      const MetaTensor& dweight,
                                      const MetaTensor& dbias,
                                      bool multi_precision,
                                      bool has_bias,
                                      MetaTensor* dweight_out,
                                      MetaTensor* dbias_out);

void FusionGroupInferMeta(const std::vector<const MetaTensor*>& ins,
                          const std::vector<int>& outs_dtype,
                          const std::vector<int>& inputs_dtype,
                          const std::string& func_name,
                          int type,
                          std::vector<MetaTensor*> outs);

void GenerateProposalsV2InferMeta(const MetaTensor& scores,
                                  const MetaTensor& bbox_deltas,
                                  const MetaTensor& im_shape,
                                  const MetaTensor& anchors,
                                  const MetaTensor& variances,
                                  int pre_nms_top_n,
                                  int post_nms_top_n,
                                  float nms_thresh,
                                  float min_size,
                                  float eta,
                                  bool pixel_offset,
                                  MetaTensor* rpn_rois,
                                  MetaTensor* rpn_roi_probs,
                                  MetaTensor* rpn_rois_num);

void LegacyGenerateProposalsInferMeta(const MetaTensor& scores,
                                      const MetaTensor& bbox_deltas,
                                      const MetaTensor& im_info,
                                      const MetaTensor& anchors,
                                      const MetaTensor& variances,
                                      int pre_nms_top_n,
                                      int post_nms_top_n,
                                      float nms_thresh,
                                      float min_size,
                                      float eta,
                                      MetaTensor* rpn_rois,
                                      MetaTensor* rpn_roi_probs,
                                      MetaTensor* rpn_rois_num);

void GraphKhopSamplerInferMeta(const MetaTensor& row,
                               const MetaTensor& col_ptr,
                               const MetaTensor& x,
                               const MetaTensor& eids,
                               const std::vector<int>& sample_sizes,
                               bool return_eids,
                               MetaTensor* out_src,
                               MetaTensor* out_dst,
                               MetaTensor* sample_index,
                               MetaTensor* reindex_x,
                               MetaTensor* out_eids);

void GraphReindexInferMeta(const MetaTensor& x,
                           const MetaTensor& neighbors,
                           const MetaTensor& count,
                           const MetaTensor& hashtable_value,
                           const MetaTensor& hashtable_index,
                           MetaTensor* reindex_src,
                           MetaTensor* reindex_dst,
                           MetaTensor* out_nodes);

void GruInferMeta(const MetaTensor& input,
                  const MetaTensor& h0,
                  const MetaTensor& weight,
                  const MetaTensor& bias,
                  const std::string& activation,
                  const std::string& gate_activation,
                  bool is_reverse,
                  bool origin_mode,
                  bool is_test,
                  MetaTensor* batch_gate,
                  MetaTensor* batch_reset_hidden_prev,
                  MetaTensor* batch_hidden,
                  MetaTensor* hidden,
                  MetaConfig config = MetaConfig());

void GruUnitInferMeta(const MetaTensor& input,
                      const MetaTensor& hidden_prev,
                      const MetaTensor& weight,
                      const MetaTensor& bias,
                      int activation,
                      int gate_activation,
                      bool origin_mode,
                      MetaTensor* gate,
                      MetaTensor* reset_hidden_prev,
                      MetaTensor* hidden,
                      MetaConfig config = MetaConfig());

void GraphSampleNeighborsInferMeta(const MetaTensor& row,
                                   const MetaTensor& col_ptr,
                                   const MetaTensor& x,
                                   const MetaTensor& eids,
                                   const MetaTensor& perm_buffer,
                                   int sample_size,
                                   bool return_eids,
                                   bool flag_perm_buffer,
                                   MetaTensor* out,
                                   MetaTensor* out_count,
                                   MetaTensor* out_eids);

void HSigmoidLossInferMeta(const MetaTensor& x,
                           const MetaTensor& label,
                           const MetaTensor& w,
                           const MetaTensor& bias,
                           const MetaTensor& path,
                           const MetaTensor& code,
                           int num_classes,
                           bool is_sparse,
                           MetaTensor* out,
                           MetaTensor* pre_out,
                           MetaTensor* w_out);

void InterpolateInferMeta(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config = MetaConfig());

void LegacyInterpolateInferMeta(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config = MetaConfig());

void IndexPutInferMeta(const MetaTensor& x,
                       const std::vector<const MetaTensor*>& indices,
                       const MetaTensor& value,
                       bool accumulate,
                       MetaTensor* out);

void LambInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   float weight_decay,
                   float beta1,
                   float beta2,
                   float epsilon,
                   bool always_adapt,
                   bool multi_precision,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs);

void LarsMomentumInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& velocity,
    const std::vector<const MetaTensor*>& learning_rate,
    const std::vector<const MetaTensor*>& grad,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    const std::vector<float>& lars_weight_decay,
    float mu,
    float lars_coeff,
    float epsilon,
    bool multi_precision,
    float rescale_grad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> velocity_out,
    std::vector<MetaTensor*> master_param_out);

void LLMInt8LinearInferMeta(const MetaTensor& x,
                            const MetaTensor& weight,
                            const MetaTensor& bias,
                            const MetaTensor& weight_scale,
                            const float threshold,
                            MetaTensor* out);

void LogspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       const MetaTensor& base,
                       DataType dtype,
                       MetaTensor* out);

void MergedAdamInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& learning_rate,
    const std::vector<const MetaTensor*>& moment1,
    const std::vector<const MetaTensor*>& moment2,
    const std::vector<const MetaTensor*>& beta1_pow,
    const std::vector<const MetaTensor*>& beta2_pow,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> moment1_out,
    std::vector<MetaTensor*> moment2_out,
    std::vector<MetaTensor*> beta1_pow_out,
    std::vector<MetaTensor*> beta2_pow_out,
    std::vector<MetaTensor*> master_param_out);

void MergedMomentumInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& velocity,
    const std::vector<const MetaTensor*>& learning_rate,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> velocity_out,
    std::vector<MetaTensor*> master_param_out);

void MemoryEfficientAttentionInferMeta(const MetaTensor& query,
                                       const MetaTensor& key,
                                       const MetaTensor& value,
                                       const MetaTensor& bias,
                                       const MetaTensor& cu_seqlens_q,
                                       const MetaTensor& cu_seqlens_k,
                                       const MetaTensor& causal_diagonal,
                                       const MetaTensor& seqlen_k,
                                       const Scalar& max_seqlen_q,
                                       const Scalar& max_seqlen_k,
                                       const bool causal,
                                       const double dropout_p,
                                       const float scale,
                                       const bool is_test,
                                       MetaTensor* output,
                                       MetaTensor* logsumexp,
                                       MetaTensor* seed_and_offset);

void MeshgridInferMeta(const std::vector<const MetaTensor*>& inputs,
                       std::vector<MetaTensor*> outputs);

void MomentumInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& velocity,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float mu,
                       bool use_nesterov,
                       const std::string& regularization_method,
                       float regularization_coeff,
                       bool multi_precision,
                       float rescale_grad,
                       MetaTensor* param_out,
                       MetaTensor* velocity_out,
                       MetaTensor* master_param_out);

void MultiDotInferMeta(const std::vector<const MetaTensor*>& x,
                       MetaTensor* out);

void MultiplexInferMeta(const std::vector<const MetaTensor*>& ins,
                        const MetaTensor& ids,
                        MetaTensor* out);

void NAdamInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& momentum_decay_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& mu_product,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& master_param,
                    float beta1,
                    float beta2,
                    float epsilon,
                    float momentum_decay,
                    bool multi_precision,
                    MetaTensor* param_out,
                    MetaTensor* momentum_decay_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* mu_product_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* master_param_outs);

void NceInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& weight,
                  const MetaTensor& bias,
                  const MetaTensor& sample_weight,
                  const MetaTensor& custom_dist_probs,
                  const MetaTensor& custom_dist_alias,
                  const MetaTensor& custom_dist_alias_probs,
                  int num_total_classes,
                  const std::vector<int>& custom_neg_classes,
                  int num_neg_samples,
                  int sampler,
                  int seed,
                  bool is_sparse,
                  bool remote_prefetch,
                  bool is_test,
                  MetaTensor* cost,
                  MetaTensor* sample_logits,
                  MetaTensor* sample_labels,
                  MetaConfig config = MetaConfig());

void PsroiPoolInferMeta(const MetaTensor& x,
                        const MetaTensor& rois,
                        const MetaTensor& rois_num,
                        int pooled_height,
                        int pooled_width,
                        int output_channels,
                        float spatial_scale,
                        MetaTensor* out);

void PyramidHashInferMeta(const MetaTensor& x,
                          const MetaTensor& w,
                          const MetaTensor& white_list,
                          const MetaTensor& black_list,
                          int num_emb,
                          int space_len,
                          int pyramid_layer,
                          int rand_len,
                          float drop_out_percent,
                          int is_training,
                          bool use_filter,
                          int white_list_len,
                          int black_list_len,
                          int seed,
                          float lr,
                          const std::string& distribute_update_vars,
                          MetaTensor* out,
                          MetaTensor* drop_pos,
                          MetaTensor* x_temp_out,
                          MetaConfig config = MetaConfig());

void QuantizeLinearInferMeta(const MetaTensor& x,
                             const MetaTensor& scale,
                             const MetaTensor& zero_point,
                             const MetaTensor& in_accum,
                             const MetaTensor& in_state,
                             int quant_axis,
                             int bit_length,
                             int round_type,
                             bool is_test,
                             bool only_observer,
                             MetaTensor* y,
                             MetaTensor* out_state,
                             MetaTensor* out_accum,
                             MetaTensor* out_scale);

void RAdamInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& beta1_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& rho,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& master_param,
                    float beta1,
                    float beta2,
                    float epsilon,
                    bool multi_precision,
                    MetaTensor* param_out,
                    MetaTensor* beta1_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* rho_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* master_param_outs);

void RmsNormInferMeta(const MetaTensor& x,
                      const MetaTensor& bias,
                      const MetaTensor& residual,
                      const MetaTensor& norm_weight,
                      const MetaTensor& norm_bias,
                      const float epsilon,
                      const int begin_norm_axis,
                      const float quant_scale,
                      const int quant_round_type,
                      const float quant_max_bound,
                      const float quant_min_bound,
                      MetaTensor* out,
                      MetaTensor* residual_out,
                      MetaTensor* inv_var);

void RmspropInferMeta(const MetaTensor& param,
                      const MetaTensor& mean_square,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& mean_grad,
                      const MetaTensor& master_param,
                      float epsilon,
                      float decay,
                      float momentum,
                      bool centered,
                      bool multi_precision,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* mean_square_out,
                      MetaTensor* mean_grad_out,
                      MetaTensor* master_param_outs);

void RnnInferMeta(const MetaTensor& x,
                  const std::vector<const MetaTensor*>& pre_state,
                  const std::vector<const MetaTensor*>& weight_list,
                  const MetaTensor& sequence_length,
                  float dropout_prob,
                  bool is_bidirec,
                  int input_size,
                  int hidden_size,
                  int num_layers,
                  const std::string& mode,
                  int seed,
                  bool is_test,
                  MetaTensor* out,
                  MetaTensor* dropout_state,
                  std::vector<MetaTensor*> state,
                  MetaTensor* reserve);

void RpropInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& prev,
                    const MetaTensor& learning_rate,
                    const MetaTensor& master_param,
                    const MetaTensor& learning_rate_range,
                    const MetaTensor& etas,
                    bool multi_precision,
                    MetaTensor* param_out,
                    MetaTensor* prev_out,
                    MetaTensor* learning_rate_out,
                    MetaTensor* master_param_out);

void SendUERecvInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& src_index,
                         const MetaTensor& dst_index,
                         const std::string& message_op,
                         const std::string& reduce_op,
                         const IntArray& out_size,
                         MetaTensor* out,
                         MetaTensor* dst_count);

void SendUVInferMeta(const MetaTensor& x,
                     const MetaTensor& y,
                     const MetaTensor& src_index,
                     const MetaTensor& dst_index,
                     const std::string& message_op,
                     MetaTensor* out);

void SgdInferMeta(const MetaTensor& param,
                  const MetaTensor& learning_rate,
                  const MetaTensor& grad,
                  const MetaTensor& master_param,
                  bool multi_precision,
                  MetaTensor* param_out,
                  MetaTensor* master_param_out);

void SigmoidCrossEntropyWithLogitsInferMeta(const MetaTensor& x,
                                            const MetaTensor& label,
                                            const MetaTensor& pos_weight,
                                            bool normalize,
                                            int ignore_index,
                                            MetaTensor* out,
                                            MetaConfig config = MetaConfig());

void SparseAttentionInferMeta(const MetaTensor& q,
                              const MetaTensor& k,
                              const MetaTensor& v,
                              const MetaTensor& offset,
                              const MetaTensor& columns,
                              const MetaTensor& key_padding_mask,
                              const MetaTensor& attn_mask,
                              MetaTensor* out,
                              MetaTensor* sparse_dot_sdd,
                              MetaTensor* softmax);

void SparseMomentumInferMeta(const MetaTensor& param,
                             const MetaTensor& grad,
                             const MetaTensor& velocity,
                             const MetaTensor& index,
                             const MetaTensor& learning_rate,
                             MetaTensor* param_out,
                             MetaTensor* velocity_out,
                             MetaTensor* master_param_out);

void StackInferMeta(const std::vector<const MetaTensor*>& x,
                    int axis,
                    MetaTensor* out,
                    MetaConfig config = MetaConfig());

void UnchangedMultiInferMeta(const std::vector<const MetaTensor*>& x,
                             std::vector<MetaTensor*> out);

void ShareBufferInferMeta(const std::vector<const MetaTensor*>& x,
                          const std::vector<bool>& share_dims_and_dtype,
                          std::vector<MetaTensor*> out,
                          std::vector<MetaTensor*> xout);

void UpdateLossScalingInferMeta(const std::vector<const MetaTensor*>& xs,
                                const MetaTensor& found_infinite,
                                const MetaTensor& prev_loss_scaling,
                                const MetaTensor& in_good_steps,
                                const MetaTensor& in_bad_steps,
                                std::vector<MetaTensor*> outs,
                                MetaTensor* loss_scaling,
                                MetaTensor* out_good_steps,
                                MetaTensor* out_bad_steps);

void WarpctcInferMeta(const MetaTensor& logits,
                      const MetaTensor& label,
                      const MetaTensor& logits_length,
                      const MetaTensor& labels_length,
                      int blank,
                      bool norm_by_times,
                      MetaTensor* loss,
                      MetaTensor* warpctcgrad);

void WarprnntInferMeta(const MetaTensor& input,
                       const MetaTensor& label,
                       const MetaTensor& input_lengths,
                       const MetaTensor& label_lengths,
                       int blank,
                       float fastemit_lambda,
                       MetaTensor* loss,
                       MetaTensor* warpctcgrad);

void WeightOnlyLinearInferMeta(const MetaTensor& x,
                               const MetaTensor& weight,
                               const MetaTensor& bias,
                               const MetaTensor& weight_scale,
                               const std::string& weight_dtype,
                               const int32_t arch,
                               const int32_t group_size,
                               MetaTensor* out);

void WeightedSampleNeighborsInferMeta(const MetaTensor& row,
                                      const MetaTensor& col_ptr,
                                      const MetaTensor& edge_weight,
                                      const MetaTensor& x,
                                      const MetaTensor& eids,
                                      int sample_size,
                                      bool return_eids,
                                      MetaTensor* out,
                                      MetaTensor* out_count,
                                      MetaTensor* out_eids);

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out);

void YoloBoxPostInferMeta(const MetaTensor& boxes0,
                          const MetaTensor& boxes1,
                          const MetaTensor& boxes2,
                          const MetaTensor& image_shape,
                          const MetaTensor& image_scale,
                          const std::vector<int>& anchors0,
                          const std::vector<int>& anchors1,
                          const std::vector<int>& anchors2,
                          int class_num,
                          float conf_thresh,
                          int downsample_ratio0,
                          int downsample_ratio1,
                          int downsample_ratio2,
                          bool clip_bbox,
                          float scale_x_y,
                          float nms_threshold,
                          MetaTensor* out,
                          MetaTensor* nms_rois_num,
                          MetaConfig config = MetaConfig());

void YoloLossInferMeta(const MetaTensor& x,
                       const MetaTensor& gt_box,
                       const MetaTensor& gt_label,
                       const MetaTensor& gt_score,
                       const std::vector<int>& anchors,
                       const std::vector<int>& anchor_mask,
                       int class_num,
                       float ignore_thresh,
                       int downsample_ratio,
                       bool use_label_smooth,
                       float scale_x_y,
                       MetaTensor* loss,
                       MetaTensor* objectness_mask,
                       MetaTensor* gt_match_mask);

void FusedAdamInferMeta(
    const std::vector<const MetaTensor*>& params,
    const std::vector<const MetaTensor*>& grads,
    const MetaTensor& learning_rate,
    const std::vector<const MetaTensor*>& moments1,
    const std::vector<const MetaTensor*>& moments2,
    const std::vector<const MetaTensor*>& beta1_pows,
    const std::vector<const MetaTensor*>& beta2_pows,
    const paddle::optional<std::vector<const MetaTensor*>>& master_params,
    const MetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    int chunk_size,
    float weight_decay,
    bool use_adamw,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<MetaTensor*> params_out,
    std::vector<MetaTensor*> moments1_out,
    std::vector<MetaTensor*> moments2_out,
    std::vector<MetaTensor*> beta1_pows_out,
    std::vector<MetaTensor*> beta2_pows_out,
    std::vector<MetaTensor*> master_params_out);

void FusedConvInferMeta(const MetaTensor& input,
                        const MetaTensor& filter,
                        const MetaTensor& bias,
                        const MetaTensor& residual_param,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        const std::vector<int>& dilations,
                        int groups,
                        const std::string& data_format,
                        const std::string& mkldnn_data_type,
                        const std::string& fuse_activation,
                        bool fuse_residual_conn,
                        bool force_fp32_output,
                        MetaTensor* out,
                        MetaConfig config = MetaConfig());

void FusedMoeInferMeta(const MetaTensor& X,
                       const MetaTensor& gate_weight,
                       const MetaTensor& ffn1_weight,
                       const MetaTensor& ffn1_scale,
                       const MetaTensor& ffn1_bias,
                       const MetaTensor& ffn2_weight,
                       const MetaTensor& ffn2_scale,
                       const MetaTensor& ffn2_bias,
                       const std::string& quant_method,
                       const int moe_topk,
                       const bool norm_topk_prob,
                       const bool group_moe,
                       MetaTensor* out);




void moe_dispatchInferMeta(
                    const MetaTensor& X,
                    const MetaTensor& gating_output,
                    const int moe_topk,
                    MetaTensor* out,
                    MetaTensor* token_nums_per_expert,
                    MetaTensor* scatter_index,
                    MetaTensor* expert_scales_float,
                    MetaTensor* expert_for_source_row_tensor);




void moe_ffnInferMeta(const MetaTensor& X,
                    const MetaTensor& rows_per_expert,
                    const MetaTensor& ffn1_weight,
                    const MetaTensor& ffn1_scale,
                    const MetaTensor& ffn1_bias,
                    const MetaTensor& ffn2_weight,
                    const MetaTensor& ffn2_scale,
                    const std::string& quant_method,
                    MetaTensor* ffn_out);




void moe_reduceInferMeta(
                    const MetaTensor& fc2_result,  // ffn output [num_rows * topk, hidden_dim]
                    const MetaTensor& fc2_expert_biases, 
                    const MetaTensor& expert_scales_float, // 对于每个token来说，不同专家对他的weight
                    const MetaTensor& expanded_source_row_to_expanded_dest_row,
                    const MetaTensor& topk_indices,
                    const bool norm_topk_prob,
                    MetaTensor* output) ;


void FusedMultiHeadAttentionInferMeta(const MetaTensor& query,
                                      const MetaTensor& key,
                                      const MetaTensor& value,
                                      const MetaTensor& mask,
                                      float scale,
                                      bool causal,
                                      MetaTensor* out);

void FusedMultiHeadAttentionVariableInferMeta(const MetaTensor& query,
                                              const MetaTensor& key,
                                              const MetaTensor& value,
                                              const MetaTensor& seq_lens,
                                              const MetaTensor& mask,
                                              float scale,
                                              bool causal,
                                              MetaTensor* out);

void FusedRopeInferMeta(const MetaTensor& q,
                        const MetaTensor& k,
                        const MetaTensor& v,
                        const MetaTensor& sin,
                        const MetaTensor& cos,
                        const MetaTensor& position_ids,
                        bool use_neox_rotary_style,
                        bool time_major,
                        float rotary_emb_base,
                        MetaTensor* out_q,
                        MetaTensor* out_k,
                        MetaTensor* out_v);

void FusedTokenPruneInferMeta(const MetaTensor& attn,
                              const MetaTensor& x,
                              const MetaTensor& mask,
                              const MetaTensor& new_mask,
                              bool keep_first_token,
                              bool keep_order,
                              MetaTensor* slimmed_x,
                              MetaTensor* cls_inds);

void MultiheadMatmulInferMeta(const MetaTensor& input,
                              const MetaTensor& w,
                              const MetaTensor& bias,
                              const MetaTensor& bias_qk,
                              const bool transpose_q,
                              const bool transpose_k,
                              const bool transpose_v,
                              const float alpha,
                              const int head_number,
                              MetaTensor* out);

void MaskedMultiheadAttentionInferMeta(const MetaTensor& x,
                                       const MetaTensor& cache_kv,
                                       const MetaTensor& bias,
                                       const MetaTensor& src_mask,
                                       const MetaTensor& cum_offsets,
                                       const MetaTensor& sequence_lengths,
                                       const MetaTensor& rotary_tensor,
                                       const MetaTensor& beam_cache_offset,
                                       const MetaTensor& qkv_out_scale,
                                       const MetaTensor& out_shift,
                                       const MetaTensor& out_smooth,
                                       int seq_len,
                                       int rotary_emb_dims,
                                       const bool use_neox_rotary_style,
                                       const std::string& compute_dtype,
                                       const float out_scale,
                                       const int quant_round_type,
                                       const float quant_max_bound,
                                       const float quant_min_bound,
                                       MetaTensor* out,
                                       MetaTensor* cache_kv_out,
                                       MetaTensor* beam_cache_offset_out);

void FullWithTensorInferMeta(const IntArray& shape,
                             DataType dtype,
                             MetaTensor* out);

void TopPSamplingInferMeta(const MetaTensor& x,
                           const MetaTensor& ps,
                           const MetaTensor& threshold,
                           const MetaTensor& topp_seed,
                           int seed,
                           int k,
                           const std::string& mode,
                           MetaTensor* out,
                           MetaTensor* ids,
                           MetaTensor* topk_scores,
                           MetaTensor* topk_ids);

}  // namespace phi
