/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

struct MetaConfig;

// Common InferMeta Functions for unary operators, The format like:
//
//   void [FunctionDesc|OpName]InferMeta(const MetaTensor& x, ..., MetaTensor*
//   out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
// Because functions in this file not only can infer shape, but also need
// infer lod or other useful data.
//
// The InferMeta Functions in this file are arranged in alphabetic order.

void AddPositionEncodingInferMeta(const MetaTensor& x,
                                  float alpha,
                                  float beta,
                                  MetaTensor* out);

void AffineGridInferMeta(const MetaTensor& input,
                         const IntArray& outputShape,
                         bool align_corners,
                         MetaTensor* output);

void AllGatherInferMeta(const MetaTensor& x, int nranks, MetaTensor* out);

void AllReduceInferMeta(const MetaTensor& x, MetaTensor* out);

void AllToAllInferMeta(const MetaTensor& x, MetaTensor* out);

void AnchorGeneratorInferMeta(const MetaTensor& input,
                              const std::vector<float>& anchor_sizes,
                              const std::vector<float>& aspect_ratios,
                              const std::vector<float>& variances,
                              const std::vector<float>& stride,
                              float offset,
                              MetaTensor* anchors,
                              MetaTensor* variances_out);

void ArgMinMaxInferMeta(const MetaTensor& x,
                        const Scalar& axis,
                        bool keepdims,
                        bool flatten,
                        DataType dtype,
                        MetaTensor* out,
                        MetaConfig config = MetaConfig());

void ArgsortInferMeta(const MetaTensor& input,
                      int axis,
                      bool descending,
                      bool stable,
                      MetaTensor* output,
                      MetaTensor* indices);

void ArrayLengthInferMeta(const MetaTensor& x, MetaTensor* out);

void ArrayToTensorInferMeta(const MetaTensor& x,
                            int axis,
                            bool use_stack,
                            MetaTensor* out,
                            MetaTensor* out_index,
                            MetaConfig config = MetaConfig());

void BipartiteMatchInferMeta(const MetaTensor& dist_mat,
                             const std::string& match_type,
                             float dist_threshold,
                             MetaTensor* col_to_row_match_indices,
                             MetaTensor* col_to_row_match_dist);

void TensorToArrayInferMeta(const MetaTensor& x,
                            const MetaTensor& out_grad,
                            int axis,
                            bool use_stack,
                            MetaTensor* x_grad);

void AsRealInferMeta(const MetaTensor& input, MetaTensor* output);

void AsComplexInferMeta(const MetaTensor& input, MetaTensor* output);

void BarrierInferMeta(const MetaTensor& x, MetaTensor* out);

void BatchSizeLikeInferMeta(const MetaTensor& x,
                            const std::vector<int>& shape,
                            int x_batch_size_dim,
                            int out_batch_size_dim,
                            MetaTensor* out);

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out);

void CConcatInferMeta(const MetaTensor& x, int nranks, MetaTensor* out);

void ChannelShuffleInferMeta(const MetaTensor& x,
                             int groups,
                             const std::string& data_format,
                             MetaTensor* out);

void CheckNumericsInferMeta(const MetaTensor& tensor,
                            const std::string& op_type,
                            const std::string& var_name,
                            const int check_nan_inf_level,
                            const int stack_height_limit,
                            const std::string& output_dir,
                            MetaTensor* stats,
                            MetaTensor* values);

void CholeskyInferMeta(const MetaTensor& x, bool upper, MetaTensor* out);

void CINNBroadcastInferMeta(const MetaTensor& x,
                            const std::vector<int64_t>& axes,
                            const std::vector<int64_t>& out_shape,
                            MetaTensor* output);

void ClassCenterSampleInferMeta(const MetaTensor& label,
                                int num_classes,
                                int num_samples,
                                int ring_id,
                                int rank,
                                int nranks,
                                bool fix_seed,
                                int seed,
                                MetaTensor* remapped_label,
                                MetaTensor* sampled_local_class_center);

void ClipByNormInferMeta(const MetaTensor& x, float max_norm, MetaTensor* out);

void CIdentityInferMeta(const MetaTensor& x,
                        int ring_id,
                        bool use_calc_stream,
                        bool use_model_parallel,
                        MetaTensor* out);

void CreateLikeInferMeta(const MetaTensor& x, DataType dtype, MetaTensor* out);

void CreateArrayLikeInferMeta(const MetaTensor& x, MetaTensor* out);

void CropInferMeta(const MetaTensor& x,
                   const IntArray& shape,
                   const IntArray& offsets,
                   MetaTensor* out,
                   MetaConfig config = MetaConfig());

void CScatterInferMeta(
    const MetaTensor& x, int ring_id, int root, int nranks, MetaTensor* out);

void CSplitInferMeta(const MetaTensor& x, int nranks, MetaTensor* out);

void CumInferMeta(const MetaTensor& x,
                  int axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  MetaTensor* out);

void CumScalarAxisInferMeta(const MetaTensor& x,
                            const Scalar& axis,
                            bool flatten,
                            bool exclusive,
                            bool reverse,
                            MetaTensor* out);

void CumWithIndicesInferMeta(const MetaTensor& x,
                             int axis,
                             DataType dtype,
                             MetaTensor* out,
                             MetaTensor* indices);

void DecodeJpegInferMeta(const MetaTensor& x,
                         const std::string& mode,
                         MetaTensor* out);

void DeQuantizeXPUInferMeta(const MetaTensor& x,
                            DataType out_dtype,
                            float scale,
                            MetaTensor* y);

void DiagEmbedInferMeta(
    const MetaTensor& x, int offset, int dim1, int dim2, MetaTensor* out);

void DiagInferMeta(const MetaTensor& x,
                   int offset,
                   float padding_value,
                   MetaTensor* out);

void DiagonalInferMeta(
    const MetaTensor& input, int offset, int axis1, int axis2, MetaTensor* out);

void DirichletInferMeta(const MetaTensor& alpha, MetaTensor* out);

void DistBroadcastInferMeta(const MetaTensor& x, MetaTensor* out);

void DistConcatInferMeta(const MetaTensor& x, int nranks, MetaTensor* out);

void DistReduceInferMeta(const MetaTensor& x, MetaTensor* out);

void EmbeddingGradSparseInferMeta(const MetaTensor& x,
                                  const MetaTensor& weight,
                                  MetaTensor* out);

void EigInferMeta(const MetaTensor& x, MetaTensor* out_w, MetaTensor* out_v);

void EighInferMeta(const MetaTensor& x,
                   const std::string& uplo,
                   MetaTensor* out_w,
                   MetaTensor* out_v);

void EigvalsInferMeta(const MetaTensor& x,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void EigvalshInferMeta(const MetaTensor& x,
                       const std::string& uplo,
                       bool is_test,
                       MetaTensor* out_w,
                       MetaTensor* out_v);

void EinsumInferMeta(const std::vector<const MetaTensor*>& inputs,
                     const std::string& equation,
                     MetaTensor* out);

void EinsumRawInferMeta(const std::vector<const MetaTensor*>& inputs,
                        const std::string& equation,
                        MetaTensor* out,
                        std::vector<MetaTensor*> inner_cache,
                        std::vector<MetaTensor*> xshape);

void ExpandInferMeta(const MetaTensor& x,
                     const IntArray& shape,
                     MetaTensor* out);

void FakeChannelWiseQuantizeAbsMaxInferMeta(const MetaTensor& x,
                                            int bit_length,
                                            int round_type,
                                            int quant_axis,
                                            bool is_test,
                                            MetaTensor* out,
                                            MetaTensor* out_scale);

void FakeChannelWiseQuantizeDequantizeAbsMaxInferMeta(const MetaTensor& x,
                                                      int bit_length,
                                                      int round_type,
                                                      int quant_axis,
                                                      MetaTensor* out,
                                                      MetaTensor* out_scale);

void FakeQuantizeAbsMaxInferMeta(const MetaTensor& x,
                                 int bit_length,
                                 int round_type,
                                 MetaTensor* out,
                                 MetaTensor* out_scale);

void FetchBarrierInferMeta(const std::vector<const MetaTensor*>& x,
                           int trainer_id,
                           const std::vector<std::string>& endpoints,
                           std::vector<MetaTensor*> out);

void FillAnyLikeInferMeta(const MetaTensor& x,
                          const Scalar& value,
                          DataType dtype,
                          MetaTensor* out);

void FillDiagonalInferMeta(
    const MetaTensor& x, float value, int offset, bool wrap, MetaTensor* out);

void FFTC2CInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::string& normalization,
                     bool forward,
                     MetaTensor* out,
                     MetaConfig = MetaConfig());

void FFTC2RInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::string& normalization,
                     bool forward,
                     int64_t last_dim_size,
                     MetaTensor* out,
                     MetaConfig = MetaConfig());

void FFTR2CInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::string& normalization,
                     bool forward,
                     bool onesided,
                     MetaTensor* out,
                     MetaConfig = MetaConfig());

void FlattenInferMeta(const MetaTensor& x,
                      int start_axis,
                      int stop_axis,
                      MetaTensor* out);

void Flatten2InferMeta(const MetaTensor& x,
                       int axis,
                       MetaTensor* out,
                       MetaTensor* x_shape);

void FlattenWithXShapeInferMeta(const MetaTensor& x,
                                int start_axis,
                                int stop_axis,
                                MetaTensor* out,
                                MetaTensor* xshape);

void FlipInferMeta(const MetaTensor& x,
                   const std::vector<int>& axis,
                   MetaTensor* out);

void FoldInferMeta(const MetaTensor& x,
                   const std::vector<int>& output_sizes,
                   const std::vector<int>& kernel_sizes,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::vector<int>& dilations,
                   MetaTensor* out);

void FractionalMaxPoolInferMeta(const MetaTensor& x,
                                const std::vector<int>& output_size,
                                const std::vector<int>& kernel_size,
                                float random_u,
                                bool return_mask,
                                MetaTensor* out,
                                MetaTensor* mask,
                                MetaConfig config = MetaConfig());

void FrameInferMeta(const MetaTensor& x,
                    int frame_length,
                    int hop_length,
                    int axis,
                    MetaTensor* out,
                    MetaConfig = MetaConfig());

void FullBatchSizeLikeInferMeta(const MetaTensor& x,
                                const std::vector<int>& shape,
                                const Scalar& val,
                                DataType dtype,
                                int x_batch_size_dim,
                                int out_batch_size_dim,
                                MetaTensor* out);

void GumbelSoftmaxInferMeta(const MetaTensor& x,
                            float temperature,
                            bool hard,
                            int axis,
                            MetaTensor* out);

void HashInferMeta(const MetaTensor& x,
                   int num_hash,
                   int64_t mod_by,
                   MetaTensor* out);

void IdentityLossInferMeta(const MetaTensor& x, int reduction, MetaTensor* out);

void IncrementInferMeta(const MetaTensor& x, float value, MetaTensor* out);

void InferMetaFromVecValue(const MetaTensor& x,
                           const std::vector<int64_t>& shape,
                           MetaTensor* out);

void InverseInferMeta(const MetaTensor& x, MetaTensor* out);

void IsEmptyInferMeta(const MetaTensor& x, MetaTensor* out);

void IsfiniteInferMeta(const MetaTensor& input, MetaTensor* out);

void KthvalueInferMeta(const MetaTensor& x,
                       int k,
                       int axis,
                       bool keepdim,
                       MetaTensor* out,
                       MetaTensor* indices,
                       MetaConfig = MetaConfig());

void LogicalNotInferMeta(const MetaTensor& x, MetaTensor* out);

void LogsumexpInferMeta(const MetaTensor& input,
                        const std::vector<int>& axis,
                        bool keepdim,
                        bool reduce_all,
                        MetaTensor* out);

void LUInferMeta(const MetaTensor& x,
                 bool pivot,
                 MetaTensor* out,
                 MetaTensor* pivots,
                 MetaTensor* infos);

void MatrixPowerInferMeta(const MetaTensor& x, int n, MetaTensor* out);

void MatrixRankInferMeta(const MetaTensor& x,
                         bool use_default_tol,
                         bool hermitian,
                         MetaTensor* out);

void MaxOutInferMeta(const MetaTensor& x,
                     int groups,
                     int axis,
                     MetaTensor* out);

void MaxPoolWithIndexInferMeta(const MetaTensor& x,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               bool global_pooling,
                               bool adaptive,
                               bool ceil_mode,
                               MetaTensor* out,
                               MetaTensor* mask,
                               MetaConfig config = MetaConfig());

void MaxPoolV2InferMeta(const MetaTensor& x,
                        const std::vector<int>& kernel_size,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& data_format,
                        bool global_pooling,
                        bool adaptive,
                        MetaTensor* out,
                        MetaTensor* saved_idx,
                        MetaConfig config = MetaConfig());

void MeanAllInferMeta(const MetaTensor& x, MetaTensor* out);

void ModeInferMeta(const MetaTensor& x,
                   int axis,
                   bool keepdim,
                   MetaTensor* out,
                   MetaTensor* indices);

void MultinomialInferMeta(const MetaTensor& x,
                          const Scalar& num_samples,
                          bool replacement,
                          MetaTensor* out,
                          MetaConfig config = MetaConfig());

void NanmedianInferMeta(const MetaTensor& x,
                        const IntArray& axes,
                        bool keep_dim,
                        const std::string& mode,
                        MetaTensor* out,
                        MetaTensor* median_index);

void NonZeroInferMeta(const MetaTensor& condition, MetaTensor* out);

void NMSInferMeta(const MetaTensor& x, float threshold, MetaTensor* out);

void NormInferMeta(const MetaTensor& x,
                   int axis,
                   float epsilon,
                   bool is_test,
                   MetaTensor* out,
                   MetaTensor* norm);

void OneHotRawInferMeta(const MetaTensor& x,
                        const Scalar& depth,
                        DataType dtype,
                        bool allow_out_of_range,
                        MetaTensor* out);

void OneHotInferMeta(const MetaTensor& x, const Scalar& depth, MetaTensor* out);

void OverlapAddInferMeta(const MetaTensor& x,
                         int hop_length,
                         int axis,
                         MetaTensor* out,
                         MetaConfig config = MetaConfig());

void PadInferMeta(const MetaTensor& input,
                  const std::vector<int>& paddings,
                  const Scalar& padding_value,
                  MetaTensor* out,
                  MetaConfig config = MetaConfig());

void Pad3dInferMeta(const MetaTensor& x,
                    const IntArray& paddings,
                    const std::string& mode,
                    float value,
                    const std::string& data_format,
                    MetaTensor* out,
                    MetaConfig config = MetaConfig());

void PartialAllgatherInferMeta(const MetaTensor& x,
                               int nranks,
                               int rank,
                               MetaTensor* out);

void PartialSendInferMeta(const MetaTensor& x, int peer, int num, int id);

void PixelShuffleInferMeta(const MetaTensor& x,
                           int upscale_factor,
                           const std::string& data_format,
                           MetaTensor* out);

void PixelShuffleGradInferMeta(const MetaTensor& out_grad,
                               int upscale_factor,
                               const std::string& data_format,
                               MetaTensor* x_grad);

void PixelUnshuffleInferMeta(const MetaTensor& x,
                             int downscale_factor,
                             const std::string& data_format,
                             MetaTensor* out);

void PNormInferMeta(const MetaTensor& x,
                    float porder,
                    int axis,
                    float epsilon,
                    bool keepdim,
                    bool asvector,
                    MetaTensor* out);

void PoolInferMeta(const MetaTensor& x,
                   const std::vector<int>& kernel_size,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool ceil_mode,
                   bool exclusive,
                   const std::string& data_format,
                   const std::string& pooling_type,
                   bool global_pooling,
                   bool adaptive,
                   const std::string& padding_algorithm,
                   MetaTensor* out,
                   MetaConfig config = MetaConfig());

void Pool2DInferMeta(const MetaTensor& x,
                     const IntArray& kernel_size,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     bool ceil_mode,
                     bool exclusive,
                     const std::string& data_format,
                     const std::string& pooling_type,
                     bool global_pooling,
                     bool adaptive,
                     const std::string& padding_algorithm,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void PSendInferMeta(const MetaTensor& x, int peer);

void PSendArrayInferMeta(const MetaTensor& x, int peer);

void PushDenseInferMeta(const std::vector<const MetaTensor*>& ids,
                        int table_id,
                        float scale_data_norm,
                        const std::vector<std::string>& input_names);

void SendV2InferMeta(const int peer, const int ring_id);

void QrInferMeta(const MetaTensor& x,
                 const std::string& mode,
                 MetaTensor* q,
                 MetaTensor* r);

void QuantizeXPUInferMeta(const MetaTensor& x,
                          DataType out_dtype,
                          float scale,
                          MetaTensor* y);

void WeightQuantizeInferMeta(const MetaTensor& x,
                             const std::string& algo,
                             const int32_t arch,
                             const int32_t group_size,
                             MetaTensor* out,
                             MetaTensor* scale);

void PowInferMeta(const MetaTensor& x, const Scalar& y, MetaTensor* out);

void RealAndImagInferMeta(const MetaTensor& x, MetaTensor* out);

void ReduceSumInferMeta(const MetaTensor& x,
                        const std::vector<int64_t>& axis,
                        bool keep_dim,
                        DataType dtype,
                        MetaTensor* out);

void ReduceInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keep_dim,
                     MetaTensor* out);

void ReduceInferMetaBase(const MetaTensor& x,
                         const std::vector<int64_t>& axis,
                         bool keep_dim,
                         bool reduce_all,
                         MetaTensor* out);

void ReduceIntArrayAxisInferMetaBase(const MetaTensor& x,
                                     const IntArray& axis,
                                     bool keep_dim,
                                     bool reduce_all,
                                     MetaTensor* out,
                                     MetaConfig config = MetaConfig());

void ReduceIntArrayAxisInferMeta(const MetaTensor& x,
                                 const IntArray& axis,
                                 bool keep_dim,
                                 MetaTensor* out,
                                 MetaConfig config = MetaConfig());

void ReduceScatterInferMeta(const MetaTensor& x, int nranks, MetaTensor* out);

void RepeatInterleaveInferMeta(const MetaTensor& x,
                               int repeats,
                               int dim,
                               MetaTensor* out);

void ReshapeInferMeta(const MetaTensor& x,
                      const IntArray& shape,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());
void ViewShapeInferMeta(const MetaTensor& input,
                        const std::vector<int64_t>& shape,
                        MetaTensor* out);

void ReshapeWithXShapeInferMeta(const MetaTensor& x,
                                const IntArray& shape,
                                MetaTensor* out,
                                MetaTensor* xshape,
                                MetaConfig config = MetaConfig());

void ReverseInferMeta(const MetaTensor& x,
                      const IntArray& axis,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void ReverseArrayInferMeta(const std::vector<const phi::MetaTensor*>& x,
                           const IntArray& axis,
                           std::vector<phi::MetaTensor*> out,
                           MetaConfig config = MetaConfig());

void RollInferMeta(const MetaTensor& x,
                   const IntArray& shifts,
                   const std::vector<int64_t>& axis,
                   MetaTensor* out);

void RReluInferMeta(const MetaTensor& x,
                    float lower,
                    float upper,
                    bool is_test,
                    MetaTensor* out,
                    MetaTensor* noise);

void RReluGradInferMeta(const MetaTensor& out_grad,
                        const MetaTensor& noise,
                        MetaTensor* x_grad);

void SequenceMaskScalarInferMeta(const MetaTensor& x,
                                 const Scalar& max_len,
                                 DataType out_dtype,
                                 MetaTensor* y);

void SequencePoolInferMeta(const MetaTensor& x,
                           bool is_test,
                           const std::string& pooltype,
                           float pad_value,
                           MetaTensor* out,
                           MetaTensor* max_index,
                           MetaConfig config = MetaConfig());

void SetValueInferMeta(const MetaTensor& x, MetaTensor* out);

void ShareDataInferMeta(const MetaTensor& x, MetaTensor* out);

void ShapeInferMeta(const MetaTensor& input, MetaTensor* out);

void Shape64InferMeta(const MetaTensor& input,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void ShardIndexInferMeta(const MetaTensor& in,
                         int index_num,
                         int nshards,
                         int shard_id,
                         int ignore_value,
                         MetaTensor* out,
                         MetaConfig config = MetaConfig());

void NumelInferMeta(const MetaTensor& input, MetaTensor* out);

void ShuffleChannelInferMeta(const MetaTensor& x, int group, MetaTensor* out);

void SliceArrayInferMeta(const MetaTensor& input,
                         const IntArray& starts,
                         const IntArray& ends,
                         MetaTensor* out,
                         MetaConfig config = MetaConfig());

void SliceArrayDenseInferMeta(const MetaTensor& input,
                              const IntArray& starts,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

void SliceRawInferMeta(const MetaTensor& input,
                       const std::vector<int64_t>& axes,
                       const IntArray& starts,
                       const IntArray& ends,
                       const std::vector<int64_t>& infer_flags,
                       const std::vector<int64_t>& decrease_axis,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void ViewSliceInferMeta(const MetaTensor& input,
                        int64_t begin_idx,
                        int64_t end_idx,
                        MetaTensor* out);

void SoftmaxInferMeta(const MetaTensor& x, int axis, MetaTensor* out);

int GetSplitAxisValue(const MetaTensor& x,
                      const Scalar& axis,
                      MetaConfig config);

void FillSplitOutDims(const MetaTensor& x,
                      const int axis_value,
                      const std::vector<int64_t>& sections_vec,
                      std::vector<MetaTensor*>* out);

void SetInferMeta(const MetaTensor& x,
                  const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& stride,
                  MetaTensor* out);

void SequenceSoftmaxInferMeta(const MetaTensor& x, MetaTensor* out);

void SplitInferMeta(const MetaTensor& x_meta,
                    const IntArray& sections,
                    const Scalar& axis,
                    std::vector<MetaTensor*> out,
                    MetaConfig config = MetaConfig());

void SplitWithNumInferMeta(const MetaTensor& x_meta,
                           int num,
                           const Scalar& axis,
                           std::vector<MetaTensor*> out,
                           MetaConfig config = MetaConfig());

void SquaredL2NormInferMeta(const MetaTensor& x, MetaTensor* out);

void L1NormInferMeta(const MetaTensor& x, MetaTensor* out);

void SqueezeInferMeta(const MetaTensor& x,
                      const IntArray& axes,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void SqueezeWithXShapeInferMeta(const MetaTensor& x,
                                const IntArray& axes,
                                MetaTensor* out,
                                MetaTensor* xshape,
                                MetaConfig config = MetaConfig());

void StridedSliceRawInferMeta(const MetaTensor& x,
                              const std::vector<int>& axes,
                              const IntArray& starts,
                              const IntArray& ends,
                              const IntArray& strides,
                              const std::vector<int>& infer_flags,
                              const std::vector<int>& decrease_axis,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

void StridedSliceInferMeta(const MetaTensor& x,
                           const std::vector<int>& axes,
                           const IntArray& starts,
                           const IntArray& ends,
                           const IntArray& strides,
                           MetaTensor* out,
                           MetaConfig config = MetaConfig());

void SumInferMeta(const MetaTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  MetaTensor* out,
                  MetaConfig config = MetaConfig());

void DetInferMeta(const MetaTensor& x,
                  MetaTensor* out,
                  MetaConfig config = MetaConfig());

void SumRawInferMeta(const MetaTensor& x,
                     const IntArray& axis,
                     bool keep_dim,
                     bool reduce_all,
                     DataType dtype,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void PartialConcatInferMeta(const std::vector<const MetaTensor*>& xs,
                            int start_index,
                            int length,
                            MetaTensor* out,
                            MetaConfig config = MetaConfig());

void PartialSumInferMeta(const std::vector<const MetaTensor*>& xs,
                         int start_index,
                         int length,
                         MetaTensor* out,
                         MetaConfig config = MetaConfig());

void SvdvalsInferMeta(const MetaTensor& x, MetaTensor* s);

void SvdInferMeta(const MetaTensor& x,
                  bool full_matrices,
                  MetaTensor* u,
                  MetaTensor* s,
                  MetaTensor* vh);

void TemporalShiftInferMeta(const MetaTensor& x,
                            int seg_num,
                            float shift_ratio,
                            const std::string& data_format,
                            MetaTensor* out,
                            MetaConfig config = MetaConfig());

void TileInferMeta(const MetaTensor& x,
                   const IntArray& repeat_times,
                   MetaTensor* out,
                   MetaConfig config = MetaConfig());

void TopKInferMeta(const MetaTensor& x,
                   const Scalar& k_scalar,
                   int axis,
                   bool largest,
                   bool sorted,
                   MetaTensor* out,
                   MetaTensor* indices,
                   MetaConfig config = MetaConfig());

void TopkV1InferMeta(const MetaTensor& x,
                     const Scalar& k_scalar,
                     MetaTensor* out,
                     MetaTensor* indices,
                     MetaConfig config = MetaConfig());

void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out);

void TransferLayoutInferMeta(const MetaTensor& x,
                             int src_layout,
                             int dst_layout,
                             MetaTensor* out);

void TransposeInferMeta(const MetaTensor& x,
                        const std::vector<int>& axis,
                        MetaTensor* out);

void TransposeGradInferMeta(const MetaTensor& x,
                            const std::vector<int>& axis,
                            MetaTensor* out);

void TrilInferMeta(const MetaTensor& x, int diagonal, MetaTensor* out);

void TriuInferMeta(const MetaTensor& x, int diagonal, MetaTensor* out);

void TrilTriuInferMeta(const MetaTensor& x,
                       int diagonal,
                       bool lower,
                       MetaTensor* out);

void UnbindInferMeta(const MetaTensor& x,
                     int axis,
                     std::vector<MetaTensor*> outs);

void UnchangedExceptLayoutInferMeta(const MetaTensor& x, MetaTensor* out);
void UnchangedExceptDtypeInferMeta(const MetaTensor& x, MetaTensor* out);
void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out);
void UnchangedArrayInferMeta(const MetaTensor& x, MetaTensor* out);
void UnchangedInferMetaIncludingTensorArray(const MetaTensor& x,
                                            MetaTensor* out);
void UnchangedVectorInferMeta(const std::vector<const MetaTensor*>& xs,
                              std::vector<MetaTensor*> outs);

// meta x -> out without change, check if axis in range [-Rank(x), Rank(x)-1]
void UnchangedInferMetaCheckAxis(const MetaTensor& x,
                                 int axis,
                                 MetaTensor* out);

void UnfoldInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void UniformRandomInplaceInferMeta(const MetaTensor& x,
                                   float min,
                                   float max,
                                   int seed,
                                   int diag_num,
                                   int diag_step,
                                   float diag_val,
                                   MetaTensor* out);

void UniformRandomBatchSizeLikeInferMeta(const MetaTensor& input,
                                         const std::vector<int>& shape,
                                         int input_dim_idx,
                                         int output_dim_idx,
                                         float min,
                                         float max,
                                         int seed,
                                         int diag_num,
                                         int diag_step,
                                         float diag_val,
                                         DataType dtype,
                                         MetaTensor* out,
                                         MetaConfig config = MetaConfig());

void UniqueConsecutiveInferMeta(const MetaTensor& x,
                                bool return_inverse,
                                bool return_counts,
                                const std::vector<int>& axis,
                                DataType dtype,
                                MetaTensor* out,
                                MetaTensor* index,
                                MetaTensor* counts);

void UniqueInferMeta(const MetaTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     MetaTensor* out,
                     MetaTensor* indices,
                     MetaTensor* index,
                     MetaTensor* counts);

void UniqueRawInferMeta(const MetaTensor& x,
                        bool return_index,
                        bool return_inverse,
                        bool return_counts,
                        const std::vector<int>& axis,
                        DataType dtype,
                        bool is_sorted,
                        MetaTensor* out,
                        MetaTensor* indices,
                        MetaTensor* index,
                        MetaTensor* counts);

void UnsqueezeInferMeta(const MetaTensor& x,
                        const IntArray& axes,
                        MetaTensor* out,
                        MetaConfig config = MetaConfig());

void UnsqueezeWithXShapeInferMeta(const MetaTensor& x,
                                  const IntArray& axes,
                                  MetaTensor* out,
                                  MetaTensor* xshape,
                                  MetaConfig config = MetaConfig());

void UnStackInferMeta(const MetaTensor& x,
                      int axis,
                      int num,
                      std::vector<MetaTensor*> outs);

void NumberCountInferMeta(const MetaTensor& x,
                          int upper_range,
                          MetaTensor* out);

void StridedUnChangedInferMeta(const MetaTensor& x, MetaTensor* out);

void StraightThroughEstimatorInferMeta(const MetaTensor& out_grad,
                                       MetaTensor* x_grad);

void LrnInferMeta(const MetaTensor& x,
                  int n,
                  MetaTensor* out,
                  MetaTensor* mid_out);

void ArrayPopInferMeta(const MetaTensor& array,
                       int index,
                       MetaTensor* array_out,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

}  // namespace phi
