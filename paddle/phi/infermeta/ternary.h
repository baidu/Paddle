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
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for ternary operators, The format like:
//
//   1. void [FunctionDesc|OpName]InferMeta(const MetaTensor& x,
//                                          const MetaTensor& y,
//                                          const MetaTensor& z,
//                                          ...,
//                                          MetaTensor* out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
//   Because functions in this file not only can infer shape, but also need
//   infer lod or other useful data.
//
// The InferMeta Functions in this file are arranged in alphabetic order.

void AccuracyInferMeta(const MetaTensor& out,
                       const MetaTensor& indice,
                       const MetaTensor& label,
                       MetaTensor* accuracy,
                       MetaTensor* correct,
                       MetaTensor* total,
                       MetaConfig config = MetaConfig());

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float beta,
                    float alpha,
                    MetaTensor* out);

void ArangeTensorInferMeta(const MetaTensor& start,
                           const MetaTensor& end,
                           const MetaTensor& step,
                           MetaTensor* out);

void AssignPosInferMeta(const MetaTensor& x,
                        const MetaTensor& cum_count,
                        const MetaTensor& eff_num_len,
                        MetaTensor* out);

void BatchFCInferMeta(const MetaTensor& input,
                      const MetaTensor& w,
                      const MetaTensor& bias,
                      MetaTensor* out);

void BoxCoderInferMeta(const MetaTensor& prior_box,
                       const MetaTensor& prior_box_var,
                       const MetaTensor& target_box,
                       const std::string& code_type,
                       bool box_normalized,
                       int axis,
                       const std::vector<float>& variance,
                       MetaTensor* output_box,
                       MetaConfig config = MetaConfig());

void CSoftmaxWithMultiLabelCrossEntropyInferMeta(
    const MetaTensor& logits,
    const MetaTensor& label,
    const MetaTensor& smooth_weight,
    int64_t ignore_index,
    bool sum_multi_label_loss,
    int ring_id,
    int rank,
    int nranks,
    MetaTensor* softmax,
    MetaTensor* loss,
    MetaConfig config = MetaConfig());

void DistributedPushSparseInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& shows,
    const std::vector<const MetaTensor*>& clicks,
    int table_id,
    int size,
    bool is_distributed,
    const std::string& push_sparse_version,
    int64_t padding_idx,
    DataType dtype,
    bool is_test,
    bool use_cvm_op,
    std::vector<MetaTensor*> output);

void DpsgdInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    float clip,
                    float batch_size,
                    float sigma,
                    int size,
                    MetaTensor* param_out);

void FakeQuantizeRangeAbsMaxInferMeta(const MetaTensor& x,
                                      const MetaTensor& in_scale,
                                      const MetaTensor& iter,
                                      int window_size,
                                      int bit_length,
                                      bool is_test,
                                      int round_type,
                                      MetaTensor* out,
                                      MetaTensor* out_scale,
                                      MetaTensor* out_scales);

void FlashAttnInferMeta(const MetaTensor& q,
                        const MetaTensor& k,
                        const MetaTensor& v,
                        MetaTensor* out,
                        MetaTensor* softmax,
                        MetaTensor* softmax_lse,
                        MetaTensor* seed_offset);

void FlashAttnQKVPackedInferMeta(const MetaTensor& qkv,
                                 MetaTensor* out,
                                 MetaTensor* softmax,
                                 MetaTensor* softmax_lse,
                                 MetaTensor* seed_offset);

void InstanceNormInferMeta(const MetaTensor& x,
                           const MetaTensor& scale,
                           const MetaTensor& bias,
                           float epsilon,
                           MetaTensor* y,
                           MetaTensor* saved_mean,
                           MetaTensor* saved_variance,
                           MetaConfig config = MetaConfig());

void GlobalGatherInferMeta(const MetaTensor& x,
                           const MetaTensor& local_count,
                           const MetaTensor& global_count,
                           int ring_id,
                           bool use_calc_stream,
                           MetaTensor* out);

void GlobalScatterInferMeta(const MetaTensor& x,
                            const MetaTensor& local_count,
                            const MetaTensor& global_count,
                            int ring_id,
                            bool use_calc_stream,
                            MetaTensor* out);

void AddGroupNormSiluInferMeta(const MetaTensor& x,
                               const MetaTensor& residual,
                               const MetaTensor& scale,
                               const MetaTensor& bias,
                               float epsilon,
                               int groups,
                               const std::string& data_layout,
                               const std::string& activation,
                               MetaTensor* y,
                               MetaTensor* residual_out,
                               MetaTensor* mean,
                               MetaTensor* variance);

void GroupNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        float epsilon,
                        int groups,
                        const std::string& data_layout,
                        MetaTensor* y,
                        MetaTensor* mean,
                        MetaTensor* variance);

void LayerNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        float epsilon,
                        int begin_norm_axis,
                        MetaTensor* out,
                        MetaTensor* mean,
                        MetaTensor* variance,
                        MetaConfig config = MetaConfig());

void LayerNormGradInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            const MetaTensor& z,
                            MetaTensor* dx,
                            MetaTensor* dy,
                            MetaTensor* dz);

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out);

void LinspaceRawInferMeta(const MetaTensor& start,
                          const MetaTensor& stop,
                          const MetaTensor& number,
                          MetaTensor* out);

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       DataType dtype,
                       MetaTensor* out);

void MatchMatrixTensorInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                const MetaTensor& w,
                                int dim_t,
                                MetaTensor* out,
                                MetaTensor* tmp,
                                MetaConfig config = MetaConfig());

void MovingAverageAbsMaxScaleInferMeta(const MetaTensor& x,
                                       const MetaTensor& in_accum,
                                       const MetaTensor& in_state,
                                       MetaTensor* out,
                                       MetaTensor* out_scale,
                                       MetaTensor* out_state,
                                       MetaTensor* out_accum);

void MultiClassNMSInferMeta(const MetaTensor& bboxes,
                            const MetaTensor& scores,
                            const MetaTensor& rois_num,
                            float score_threshold,
                            int nms_top_k,
                            int keep_top_k,
                            float nms_threshold,
                            bool normalized,
                            float nms_eta,
                            int background_label,
                            MetaTensor* out,
                            MetaTensor* index,
                            MetaTensor* nms_rois_num,
                            MetaConfig config = MetaConfig());

void NllLossRawInferMeta(const MetaTensor& input,
                         const MetaTensor& label,
                         const MetaTensor& weight,
                         int64_t ignore_index,
                         const std::string& reduction,
                         MetaTensor* out,
                         MetaTensor* total_weight,
                         MetaConfig config = MetaConfig());

void PushGpupsSparseInferMeta(const std::vector<const MetaTensor*>& ids,
                              const std::vector<const MetaTensor*>& out,
                              const std::vector<int>& size,
                              bool is_sparse,
                              bool is_distributed,
                              std::vector<MetaTensor*> out_grad);

void PutAlongAxisInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& value,
                           int axis,
                           const std::string& reduce,
                           MetaTensor* out);

void RandomRoutingInferMeta(const MetaTensor& prob,
                            const MetaTensor& topk_value,
                            const MetaTensor& topk_idx,
                            MetaTensor* out);

void RankAttentionInferMeta(const MetaTensor& x,
                            const MetaTensor& rank_offset,
                            const MetaTensor& rank_param,
                            int max_rank,
                            int max_size,
                            MetaTensor* input_help,
                            MetaTensor* out,
                            MetaTensor* ins_rank);

void RoiAlignInferMeta(const MetaTensor& x,
                       const MetaTensor& boxes,
                       const MetaTensor& boxes_num,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int sampling_ratio,
                       bool aligned,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void RoiPoolInferMeta(const MetaTensor& x,
                      const MetaTensor& boxes,
                      const MetaTensor& boxes_num,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      MetaTensor* out,
                      MetaTensor* arg_max);

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out);

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out);

void SendURecvInferMeta(const MetaTensor& x,
                        const MetaTensor& src_index,
                        const MetaTensor& dst_index,
                        const std::string& reduce_op,
                        const IntArray& out_size,
                        MetaTensor* out,
                        MetaTensor* dst_count);

void SequenceConvInferMeta(const MetaTensor& x,
                           const MetaTensor& padding_data,
                           const MetaTensor& filter,
                           int context_length,
                           bool padding_trainable,
                           int context_start,
                           int context_stride,
                           MetaTensor* out);

void SparseMomentumInferMeta(const MetaTensor& param,
                             const MetaTensor& learning_rate,
                             const MetaTensor& velocity,
                             MetaTensor* param_out,
                             MetaTensor* velocity_out,
                             MetaTensor* master_param_out);
void SpectralNormInferMeta(const MetaTensor& weight,
                           const MetaTensor& u,
                           const MetaTensor& v,
                           int dim,
                           int power_iters,
                           float eps,
                           MetaTensor* out,
                           MetaConfig config = MetaConfig());

void ViterbiDecodeInferMeta(const MetaTensor& input,
                            const MetaTensor& transition,
                            const MetaTensor& length,
                            bool include_bos_eos_tag,
                            MetaTensor* scores,
                            MetaTensor* path,
                            MetaConfig config = MetaConfig());

void QuantLinearInferMeta(const MetaTensor& x,
                          const MetaTensor& w,
                          const MetaTensor& bias,
                          int in_num_col_dims,
                          const std::string& activation_type,
                          bool padding_weights,
                          float scale_in,
                          const std::vector<float>& scale_weights,
                          int quant_round_type,
                          float quant_max_bound,
                          float quant_min_bound,
                          MetaTensor* y);

void TdmSamplerInferMeta(const MetaTensor& x,
                         const MetaTensor& travel,
                         const MetaTensor& layer,
                         bool output_positive,
                         const std::vector<int>& neg_samples_num_list,
                         const std::vector<int>& layer_offset_lod,
                         int seed,
                         int dtype,
                         MetaTensor* out,
                         MetaTensor* labels,
                         MetaTensor* mask,
                         MetaConfig config = MetaConfig());

}  // namespace phi
