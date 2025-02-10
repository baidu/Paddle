// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#ifdef PADDLE_WITH_CUDA
#include <cudnn.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <miopen/miopen.h>
#endif
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

#include <glog/logging.h>

#include <algorithm>
#include <sstream>

namespace paddle {

void PaddlePassBuilder::AppendPass(const std::string &pass_type) {
  passes_.push_back(pass_type);
}

void PaddlePassBuilder::TurnOnDebug() {
  std::vector<std::string> passes;
  auto it = std::begin(passes_);
  while (it != std::end(passes_)) {
    if (*it != "graph_viz_pass") {
      it = passes_.insert(it + 1, "graph_viz_pass");
    } else {
      ++it;
    }
  }
}

std::string PaddlePassBuilder::DebugString() {
  std::stringstream ss;
  ss << "Passes to apply:\n";
  for (auto &pass : passes_) {
    ss << "  - " << pass << '\n';
  }
  return ss.str();
}

void PaddlePassBuilder::DeletePass(const std::string &pass_type) {
  deleted_passes_.insert(pass_type);
  auto it = std::begin(passes_);
  while (it != std::end(passes_)) {
    if (*it == pass_type) {
      it = passes_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t PaddlePassBuilder::GetPassIndex(const std::string &pass_type) {
  auto iter = std::find(std::begin(passes_), std::end(passes_), pass_type);
  if (iter == std::end(passes_)) return -1;
  return std::distance(std::begin(passes_), iter);
}

void PaddlePassBuilder::InsertPass(size_t idx, const std::string &pass_type) {
  passes_.insert(std::begin(passes_) + idx, pass_type);  // NOLINT
}

void PaddlePassBuilder::DeletePass(size_t idx) {
  passes_.erase(std::begin(passes_) + idx);  // NOLINT
}

void PaddlePassBuilder::AppendAnalysisPass(const std::string &pass) {
  analysis_passes_.push_back(pass);
}

void PaddlePassBuilder::ClearPasses() { passes_.clear(); }
#ifdef PADDLE_WITH_OPENVINO
const std::vector<std::string> kOVSubgraphPasses({"openvino_subgraph_pass"});
#endif
const std::vector<std::string> kTRTSubgraphPasses({
  "set_subgraph_edge_pass",                                       //
      "trt_remove_amp_strategy_op_pass",                          //
      "trt_support_nhwc_pass",                                    //
      "adaptive_pool2d_convert_global_pass",                      //
      "trt_map_ops_to_matrix_multiply_pass",                      //
      "shuffle_channel_detect_pass",                              //
      "quant_conv2d_dequant_fuse_pass",                           //
      "delete_quant_dequant_op_pass",                             //
      "delete_quant_dequant_filter_op_pass",                      //
      "trt_delete_weight_dequant_linear_op_pass",                 //
      "delete_quant_dequant_linear_op_pass",                      //
      "identity_op_clean_pass",                                   //
      "add_support_int8_pass",                                    //
      "simplify_with_basic_ops_pass",                             //
      "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",  //
      "trt_embedding_eltwise_layernorm_fuse_pass",                //
      "preln_embedding_eltwise_layernorm_fuse_pass",              //
      "trt_multihead_matmul_fuse_pass_v2",                        //
      "trt_multihead_matmul_fuse_pass_v3",                        //
      "multihead_matmul_roformer_fuse_pass",                      //
#if defined _WIN32  // Windows does not support sparse_conv3d_implicit_gemm
#else
      "sparse_conv_optim_pass",                //
#endif
      "constant_folding_pass",  //
#ifdef PADDLE_WITH_TENSORRT
#if !IS_TRT_VERSION_GE(8610)
      "trt_flash_multihead_matmul_fuse_pass",  //
      "trt_cross_multihead_matmul_fuse_pass",  //
#endif
#endif
      "vit_attention_fuse_pass",              //
      "trt_qk_multihead_matmul_fuse_pass",    //
      "layernorm_shift_partition_fuse_pass",  //
      "merge_layernorm_fuse_pass",            //
#if !defined _WIN32
      "split_layernorm_to_math_ops_pass",  //
#endif
#if defined _WIN32  // Windows CI is TensorRT7.0. Remove this after upgrading.
#else
      "trt_skip_layernorm_fuse_pass",          //
      "preln_skip_layernorm_fuse_pass",        //
#endif
      "preln_residual_bias_fuse_pass",   //
      "preln_layernorm_x_fuse_pass",     //
      "reverse_roll_fuse_pass",          //
      "conv_bn_fuse_pass",               //
      "conv_elementwise_add_fuse_pass",  //
#if defined _WIN32  // Windows CI is TensorRT7.0. Remove this after upgrading.
#else
      "trans_layernorm_fuse_pass",             //
#endif
      "remove_padding_recover_padding_pass",         //
      "delete_remove_padding_recover_padding_pass",  //
      // "yolo_box_fuse_pass",      //
      "dense_fc_to_sparse_pass",                //
      "dense_multihead_matmul_to_sparse_pass",  //
#if defined _WIN32  // Windows CI is TensorRT7.0. Remove this after upgrading.
#else
      "elementwise_groupnorm_act_pass",        //
      "preln_elementwise_groupnorm_act_pass",  //
      "groupnorm_act_pass",                    //
      "elementwiseadd_transpose_pass",         //
#endif
      "tensorrt_subgraph_pass",  //
      "conv_bn_fuse_pass",       //
#if CUDNN_VERSION >= 7100  // To run conv_fusion, the version of cudnn must be
                           // guaranteed at least v7
// cudnn8.0 has memory leak problem in conv + eltwise + act, so we
// disable the pass.
#if !(CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8100)
      "conv_elementwise_add_act_fuse_pass",   //
      "conv_elementwise_add2_act_fuse_pass",  //
#endif
#endif
      "transpose_flatten_concat_fuse_pass",  //
      "auto_mixed_precision_pass",
});

// TODO(inference): Most of the existing pass fusion operators do not
// support fp16/bf16 precision, temporarily use low precision pass to prevent
// running errors. After fusion operator supports low precision, delete this.
const std::vector<std::string> kGpuLowerPrecisionPasses{
    "map_op_to_another_pass",
    "identity_op_clean_pass",
    "simplify_with_basic_ops_pass",
    "silu_fuse_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass",
    "conv_bn_fuse_pass",
    "conv_eltwiseadd_bn_fuse_pass",
    "conv_elementwise_add_act_fuse_pass",
    "conv_elementwise_add2_act_fuse_pass",
    "conv_elementwise_add_fuse_pass",
    "transfer_layout_pass",
    "multihead_matmul_fuse_pass_v2",
    "fused_multi_transformer_encoder_pass",
    "fused_multi_transformer_decoder_pass",
    "fused_multi_transformer_encoder_fuse_qkv_pass",
    "fused_multi_transformer_decoder_fuse_qkv_pass",
    "multi_devices_fused_multi_transformer_encoder_pass",
    "multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass",
    "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass",
    "fuse_multi_transformer_layer_pass",
    "gpu_cpu_map_matmul_v2_to_mul_pass",
    "gpu_cpu_map_matmul_v2_to_matmul_pass",
    "gpu_cpu_map_matmul_to_mul_pass",
    "fc_fuse_pass",
    // "fc_elementwise_layernorm_fuse_pass",
    "embedding_eltwise_layernorm_fuse_pass",
    "inplace_op_var_pass"};

const std::vector<std::string> kTrtLowerPrecisionPasses{
    "trt_remove_amp_strategy_op_pass",
    "trt_support_nhwc_pass",
    "trt_map_ops_to_matrix_multiply_pass",
    "simplify_with_basic_ops_pass",
    // "conv_bn_fuse_pass",
    // "conv_eltwiseadd_bn_fuse_pass",
    "trt_embedding_eltwise_layernorm_fuse_pass",
    "trt_skip_layernorm_fuse_pass",
    "tensorrt_subgraph_pass",
};

const std::vector<std::string> kCINNCompilerPasses{
    "gpu_cpu_map_matmul_v2_to_mul_pass",
    "gpu_cpu_map_matmul_v2_to_matmul_pass",
    "gpu_cpu_map_matmul_to_mul_pass",
};

const std::vector<std::string> CpuBasicPasses{
    "simplify_with_basic_ops_pass",  //
    "layer_norm_fuse_pass",
    "attention_lstm_fuse_pass",       //
    "seqconv_eltadd_relu_fuse_pass",  //
    // "seqpool_concat_fuse_pass",    //
    "seqpool_cvm_concat_fuse_pass",  //
    // "embedding_fc_lstm_fuse_pass", //
    // TODO(wilber): fix correctness problem.
    // "fc_lstm_fuse_pass",                    //
    "mul_lstm_fuse_pass",                      //
    "fc_gru_fuse_pass",                        //
    "mul_gru_fuse_pass",                       //
    "seq_concat_fc_fuse_pass",                 //
    "gpu_cpu_squeeze2_matmul_fuse_pass",       //
    "gpu_cpu_reshape2_matmul_fuse_pass",       //
    "gpu_cpu_flatten2_matmul_fuse_pass",       //
    "matmul_v2_scale_fuse_pass",               //
    "gpu_cpu_map_matmul_v2_to_mul_pass",       //
    "gpu_cpu_map_matmul_v2_to_matmul_pass",    //
    "matmul_scale_fuse_pass",                  //
    "gpu_cpu_map_matmul_to_mul_pass",          //
    "fc_fuse_pass",                            //
    "repeated_fc_relu_fuse_pass",              //
    "squared_mat_sub_fuse_pass",               //
    "conv_bn_fuse_pass",                       //
    "conv_eltwiseadd_bn_fuse_pass",            //
    "conv_transpose_bn_fuse_pass",             //
    "conv_transpose_eltwiseadd_bn_fuse_pass",  //
    "is_test_pass",                            //
    "constant_folding_pass",
};

GpuPassStrategy::GpuPassStrategy() : PassStrategy({}) {
  passes_.assign({
    "map_op_to_another_pass",                    //
        "is_test_pass",                          //
        "simplify_with_basic_ops_pass",          //
        "delete_quant_dequant_linear_op_pass",   //
        "delete_weight_dequant_linear_op_pass",  //
#if defined _WIN32  // Windows does not support sparse_conv3d_implicit_gemm
#else
        "sparse_conv_optim_pass",              //
#endif
        "constant_folding_pass",                                        //
        "silu_fuse_pass",                                               //
        "conv_bn_fuse_pass",                                            //
        "conv_eltwiseadd_bn_fuse_pass",                                 //
        "embedding_eltwise_layernorm_fuse_pass",                        //
        "multihead_matmul_fuse_pass_v2",                                //
        "vit_attention_fuse_pass",                                      //
        "fused_multi_transformer_encoder_pass",                         //
        "fused_multi_transformer_decoder_pass",                         //
        "fused_multi_transformer_encoder_fuse_qkv_pass",                //
        "fused_multi_transformer_decoder_fuse_qkv_pass",                //
        "multi_devices_fused_multi_transformer_encoder_pass",           //
        "multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass",  //
        "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass",  //
        "fuse_multi_transformer_layer_pass",                            //
        "gpu_cpu_squeeze2_matmul_fuse_pass",                            //
        "gpu_cpu_reshape2_matmul_fuse_pass",                            //
        "gpu_cpu_flatten2_matmul_fuse_pass",                            //
        "gpu_cpu_map_matmul_v2_to_mul_pass",                            //
        "gpu_cpu_map_matmul_v2_to_matmul_pass",                         //
        "matmul_scale_fuse_pass",                                       //
        "multihead_matmul_fuse_pass_v3",                                //
        "gpu_cpu_map_matmul_to_mul_pass",                               //
        "fc_fuse_pass",                                                 //
        "fc_elementwise_layernorm_fuse_pass",                           //
#if CUDNN_VERSION >= 7100  // To run conv_fusion, the version of cudnn must be
                           // guaranteed at least v7
// cudnn8.0 has memory leak problem in conv + eltwise + act, so we
// disable the pass.
#if !(CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8100)
        "conv_elementwise_add_act_fuse_pass",   //
        "conv_elementwise_add2_act_fuse_pass",  //
#endif
        "conv_elementwise_add_fuse_pass",      //
#endif                                         //
        "transpose_flatten_concat_fuse_pass",  //
        "transfer_layout_pass",                //
        "transfer_layout_elim_pass",
        "auto_mixed_precision_pass",  //
        "identity_op_clean_pass",  // should be after auto_mixed_precision_pass.
        "inplace_op_var_pass",     // should be the last pass.
  });

  use_gpu_ = true;
}

void GpuPassStrategy::EnableCUDNN() {
  if (!use_cudnn_) {
    passes_.insert(passes_.begin(), "cudnn_placement_pass");
  }
  use_cudnn_ = true;
}

void GpuPassStrategy::EnableMKLDNN() {
  LOG(ERROR) << "GPU not support MKLDNN yet";
}

void GpuPassStrategy::EnableMkldnnBfloat16() {
  LOG(ERROR) << "GPU not support MKL-DNN bfloat16";
}

void GpuPassStrategy::EnableMkldnnInt8() {
  LOG(ERROR) << "GPU not support MKL-DNN int8";
}

void GpuPassStrategy::DisableMkldnnFcPasses() {
  LOG(ERROR) << "GPU not support MKL-DNN fc";
}

CpuPassStrategy::CpuPassStrategy() : PassStrategy({}) {
  // NOTE the large fusions should be located in the front, so that they will
  // not be damaged by smaller ones.
  passes_.assign(CpuBasicPasses.begin(), CpuBasicPasses.end());

  use_gpu_ = false;
}

void CpuPassStrategy::EnableCUDNN() { LOG(ERROR) << "CPU not support cuDNN"; }

void CpuPassStrategy::EnableMKLDNN() {
// TODO(Superjomn) Consider the way to mix CPU with GPU.
#ifdef PADDLE_WITH_DNNL
  if (!use_mkldnn_) {
    passes_.insert(passes_.begin(), "onednn_placement_pass");

    for (auto &pass : std::vector<std::string>({
             "squeeze2_transpose2_onednn_fuse_pass",
             "depthwise_conv_onednn_pass",    //
             "conv_bn_fuse_pass",             // Execute BN passes again to
             "conv_eltwiseadd_bn_fuse_pass",  // preserve correct pass order
             "conv_affine_channel_onednn_fuse_pass",    //
             "conv_transpose_bn_fuse_pass",             //
             "conv_transpose_eltwiseadd_bn_fuse_pass",  //
             "conv_bias_onednn_fuse_pass",              //
             "conv_transpose_bias_onednn_fuse_pass",
             // TODO(baoachun): Need to support 5-dimensional input.
             // "conv3d_bias_onednn_fuse_pass",  //
             "conv_elementwise_add_onednn_fuse_pass",
             "conv_activation_onednn_fuse_pass",           //
             "scale_matmul_fuse_pass",                     //
             "reshape_transpose_matmul_onednn_fuse_pass",  //
             "matmul_transpose_reshape_onednn_fuse_pass",  //
             "matmul_elementwise_add_onednn_fuse_pass",    //
             "matmul_activation_onednn_fuse_pass",         //
             // Disabled due to topology-dependent speed-up
             "fc_onednn_pass",
             "fc_act_onednn_fuse_pass",
             "self_attention_fuse_pass",              //
             "batch_norm_act_fuse_pass",              //
             "softplus_activation_onednn_fuse_pass",  //
             "shuffle_channel_onednn_detect_pass",    //
             "elementwise_act_onednn_fuse_pass",      //
             "operator_scale_onednn_fuse_pass",       //
             "operator_unsqueeze2_onednn_fuse_pass",  //
             "operator_reshape2_onednn_fuse_pass",    //
         })) {
      passes_.push_back(pass);
    }
  }
  use_mkldnn_ = true;
#else
  use_mkldnn_ = false;
#endif
}

void CpuPassStrategy::DisableMKLDNN() {
  ClearPasses();
  passes_.assign(CpuBasicPasses.begin(), CpuBasicPasses.end());
}

void CpuPassStrategy::EnableMkldnnBfloat16() {
#ifdef PADDLE_WITH_DNNL
  if (!use_mkldnn_bfloat16_) {
    passes_.emplace_back("fc_onednn_pass");
    passes_.emplace_back("fc_act_onednn_fuse_pass");

    passes_.emplace_back("cpu_bfloat16_placement_pass");
    passes_.emplace_back("cpu_bfloat16_pass");
    passes_.emplace_back("cpu_quantize_squash_pass");
  }
  use_mkldnn_bfloat16_ = true;
#else
  use_mkldnn_bfloat16_ = false;
#endif
}

void CpuPassStrategy::EnableMkldnnInt8() {
#ifdef PADDLE_WITH_DNNL
  if (!use_mkldnn_int8_) {
    passes_.clear();
    passes_.emplace_back("simplify_with_basic_ops_pass");
    passes_.emplace_back("quant_dequant_onednn_pass");
    passes_.emplace_back("onednn_placement_pass");
    passes_.emplace_back("constant_folding_pass");
    passes_.emplace_back("squeeze2_transpose2_onednn_fuse_pass");
    passes_.emplace_back("layer_norm_fuse_pass");
    passes_.emplace_back("attention_lstm_fuse_pass");
    passes_.emplace_back("seqconv_eltadd_relu_fuse_pass");
    passes_.emplace_back("fc_lstm_fuse_pass");
    passes_.emplace_back("mul_lstm_fuse_pass");
    passes_.emplace_back("fc_gru_fuse_pass");
    passes_.emplace_back("mul_gru_fuse_pass");
    passes_.emplace_back("multi_gru_fuse_pass");
    passes_.emplace_back("multi_gru_seq_fuse_pass");
    passes_.emplace_back("seq_concat_fc_fuse_pass");
    passes_.emplace_back("gpu_cpu_squeeze2_matmul_fuse_pass");
    passes_.emplace_back("gpu_cpu_reshape2_matmul_fuse_pass");
    passes_.emplace_back("gpu_cpu_flatten2_matmul_fuse_pass");
    passes_.emplace_back("matmul_v2_scale_fuse_pass");
    passes_.emplace_back("squared_mat_sub_fuse_pass");
    passes_.emplace_back("is_test_pass");
    passes_.emplace_back("gpu_cpu_map_matmul_v2_to_mul_pass");
    passes_.emplace_back("gpu_cpu_map_matmul_v2_to_matmul_pass");
    passes_.emplace_back("matmul_scale_fuse_pass");
    passes_.emplace_back("gpu_cpu_map_matmul_to_mul_pass");
    passes_.emplace_back("repeated_fc_relu_fuse_pass");
    passes_.emplace_back("depthwise_conv_onednn_pass");
    passes_.emplace_back("conv_bn_fuse_pass");
    passes_.emplace_back("conv_eltwiseadd_bn_fuse_pass");
    passes_.emplace_back("conv_affine_channel_onednn_fuse_pass");
    passes_.emplace_back("conv_transpose_bn_fuse_pass");
    passes_.emplace_back("conv_transpose_eltwiseadd_bn_fuse_pass");
    passes_.emplace_back("conv_bias_onednn_fuse_pass");
    passes_.emplace_back("conv_transpose_bias_onednn_fuse_pass");
    passes_.emplace_back("conv_elementwise_add_onednn_fuse_pass");
    passes_.emplace_back("conv_activation_onednn_fuse_pass");
    passes_.emplace_back("fc_fuse_pass");
    passes_.emplace_back("repeated_fc_relu_fuse_pass");
    passes_.emplace_back("fc_onednn_pass");
    passes_.emplace_back("fc_act_onednn_fuse_pass");
    passes_.emplace_back("matmul_transpose_reshape_onednn_fuse_pass");
    passes_.emplace_back("batch_norm_act_fuse_pass");
    passes_.emplace_back("softplus_activation_onednn_fuse_pass");
    passes_.emplace_back("compute_propagate_scales_onednn_pass");
    passes_.emplace_back("scale_matmul_fuse_pass");
    passes_.emplace_back("reshape_transpose_matmul_onednn_fuse_pass");
    passes_.emplace_back("matmul_elementwise_add_onednn_fuse_pass");
    passes_.emplace_back("operator_scale_onednn_fuse_pass");
    passes_.emplace_back("operator_unsqueeze2_onednn_fuse_pass");
    passes_.emplace_back("operator_reshape2_onednn_fuse_pass");
    passes_.emplace_back("cpu_quantize_placement_pass");
    passes_.emplace_back("cpu_quantize_pass");
    passes_.emplace_back("cpu_quantize_squash_pass");
    passes_.emplace_back("quant_transpose2_dequant_onednn_fuse_pass");
  }
  use_mkldnn_int8_ = true;
#else
  use_mkldnn_int8_ = false;
#endif
}

void CpuPassStrategy::DisableMkldnnFcPasses() {
#ifdef PADDLE_WITH_DNNL
  if (!disable_mkldnn_fc_passes_) {
    EraseFcMkldnnPasses();
  }
  disable_mkldnn_fc_passes_ = true;
#else
  disable_mkldnn_fc_passes_ = false;
#endif
}

void CpuPassStrategy::EraseFcMkldnnPasses() {
  std::vector<std::string> fc_passes_to_erase(
      {"fc_onednn_pass", "fc_act_onednn_fuse_pass"});
  for (const auto &pass : fc_passes_to_erase) {
    int idx = static_cast<int>(GetPassIndex(pass));
    if (idx != -1) {
      passes_.erase(std::begin(passes_) + idx);
    }
  }
}

XpuPassStrategy::XpuPassStrategy() : PassStrategy({}) {
  passes_.assign({
      "map_op_to_another_pass",
      // "quant_dequant_xpu_pass", open this pass when use old int8 model
      "delete_quant_dequant_linear_op_pass",
      "delete_weight_dequant_linear_op_pass",
      "delete_assign_op_pass",
      "delete_dropout_op_pass",
      "delete_concat_op_pass",
      "gather_squeeze_pass",
      "roformer_relative_pos_fuse_pass",
      "delete_repeated_ops_pass",
      "identity_op_clean_pass",
      "fused_continuous_same_ops_pass",
      "reshape_unstack_concat_fuse_pass",
      "delete_op_device_pass",
      "constant_folding_pass",
      "cast_embedding_trans_ids_to_int32_pass",
      "delete_elementwise_mul_op_pass",
      "generate_sequence_xpu_fuse_pass",
      "group_norm_silu_xpu_fuse_pass",
      "layer_norm_relu_xpu_fuse_pass",
      "embedding_with_eltwise_add_xpu_fuse_pass",
      "qk_qkv_attention_xpu_fuse_pass",
      "block_multihead_attention_xpu_pass",
      "multi_encoder_xpu_fuse_pass",
      "multi_encoder_xpu_adaptive_seqlen_fuse_pass",
      "multi_encoder_xpu_slice_fuse_pass",
      "weight_only_linear_xpu_pass",
      "fused_multi_transformer_cachekv_layout_trans_pass",
      "fused_multi_transformer_int8_cachekv_layout_trans_pass",
      "cross_attention_xpu_fuse_pass",
      "decoder_attention_xpu_fuse_pass",
      "one_beam_size_fuse_pass",
      "fold_interp_outsize_fuse_pass",
      "fold_two_squeeze2_fuse_pass",
      // "conv1d_xpu_fuse_pass",
      "duplicated_transpose_fuse_pass",
      "conv2d_bias_fuse_pass",
      "redundant_unsqueeze_squeeze_elimination_pass",
      "reduce_ops_fuse_pass",
      "delete_cast_op_pass",
      "xpu_delete_cast_op_pass",
      "conv2d_trans_filter_dilations_nxn_to_1x1_pass",
      "stack_fuse_pass",
      "fused_multi_transformer_xpu_pass",
      "fused_multi_transformer_int8_xpu_quant_pass",
      "relu6_fuse_pass",
      "sigmoid_elementmul_fuse_pass",
      "layer_norm_fuse_pass",
      "matmul_weight_trans_pass",
      "map_matmulv2_to_matmul_xpu_pass",
      "reshape2_matmul_xpu_fuse_pass",
      "squeeze2_matmul_xpu_fuse_pass",
      "redundant_squeeze_unsqueeze_elimination_pass",
      "fc_xpu_fuse_pass",
      "conv2d_xpu_fuse_pass",
      "conv2d_transpose_xpu_fuse_pass",
      "squeeze_excitation_fuse_pass",
      "add_activation_xpu_fuse_pass",
      "add_layernorm_xpu_fuse_pass",
      "layer_norm_act_xpu_fuse_pass",
      "fast_layernorm_xpu_fuse_pass",
      "bn_act_xpu_fuse_pass",
      "yolo_box_xpu_fuse_pass",
      "fast_where_xpu_fuse_pass",
      "elementwise_mul_add_fuse_pass",
      "sine_pos_fuse_pass",
      "pad2d_xpu_fuse_pass",
      // "auto_mixed_precision_pass",
      "cast_mixed_precision_op_fuse_pass",
      "xpu_quantize_op_pass",
      "xpu_quantize_squash_pass",
      "link_xpu_op_max_pass",
      "spatial_transformer_resblock_xpu_fuse_pass",
      "delete_isolated_node_pass",
      "inplace_op_var_pass",
  });
  use_xpu_ = true;
}

IpuPassStrategy::IpuPassStrategy() : PassStrategy({}) {
  passes_.assign({"inference_process_pass"});
}

const std::vector<std::string> kPirGpuPasses{
    // Functional pass
    "add_shadow_output_after_dead_parameter_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass",
    "map_op_to_another_pass",
    "identity_op_clean_pass",
    // Operator fusion pass
    "silu_fuse_pass",
    "conv2d_bn_fuse_pass",
    "conv2d_add_act_fuse_pass",
    "conv2d_add_fuse_pass",
    "embedding_eltwise_layernorm_fuse_pass",
    "fused_rotary_position_embedding_pass",
    "fused_flash_attn_pass",
    "multihead_matmul_fuse_pass",
    "fused_weight_only_linear_pass",
    "matmul_add_act_fuse_pass",
    "fc_elementwise_layernorm_fuse_pass",
    "add_norm_fuse_pass",
    "group_norm_silu_fuse_pass",
    "matmul_scale_fuse_pass",
    "matmul_transpose_fuse_pass",
    "transpose_flatten_concat_fuse_pass",
    "remove_redundant_transpose_pass",
    "horizontal_fuse_pass",
};

const std::vector<std::string> kPirXpuPasses{
    // Functional pass
    "add_shadow_output_after_dead_parameter_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass",
    "map_op_to_another_pass",
    "identity_op_clean_pass",
    // Operator fusion pass
    "add_activation_xpu_fuse_pass",
    "add_layernorm_xpu_fuse_pass",
    "rms_norm_xpu_fuse_pass",
    "elementwise_mul_add_xpu_fuse_pass",
    "conv2d_bn_xpu_fuse_pass",
    "conv2d_bn_add_xpu_fuse_pass",
    "conv2d_add_xpu_fuse_pass",
    "group_norm_silu_fuse_pass",
    "fc_xpu_fuse_pass"};

const std::vector<std::string> kPirMkldnnPasses {
  "add_shadow_output_after_dead_parameter_pass",
      "delete_quant_dequant_linear_op_pass",      //
      "delete_weight_dequant_linear_op_pass",     //
      "depthwise_conv_onednn_pass",               //
      "squeeze_transpose_onednn_fuse_pass",       //
      "conv2d_bn_onednn_fuse_pass",               //
      "conv2d_bias_bn_onednn_fuse_pass",          //
      "conv2d_bias_fuse_pass",                    //
      "conv2d_transpose_bn_fuse_pass",            //
      "conv2d_transpose_bias_bn_fuse_pass",       //
      "conv2d_transpose_bias_fuse_pass",          //
      "conv3d_bias_fuse_pass",                    //
      "conv_elementwise_add_onednn_fuse_pass",    //
      "conv_activation_onednn_fuse_pass",         //
      "conv_concat_activation_onednn_fuse_pass",  //
      "matmul_scale_fuse_pass",                   //
      "scale_matmul_fuse_pass",                   //
      "reshape_transpose_matmul_fuse_pass",       //
      "matmul_transpose_reshape_fuse_pass",       //
      "matmul_add_act_fuse_pass",                 //
      "matmul_reshape_add_fuse_pass",             //
      "fc_onednn_enable_pass",                    //
      "matmul_elementwise_add_fuse_pass",         //
      "matmul_activation_fuse_pass",              //
      "matmul_add_act_fuse_pass",                 //
      "fc_onednn_enable_pass",                    //
      "fc_activation_fuse_pass",                  //
#if defined(PADDLE_WITH_AVX512F) && defined(PADDLE_WITH_MKLML) && \
    defined(PADDLE_WITH_DNNL)
      "self_attention_fuse_pass",  //
#endif
      "batch_norm_act_fuse_pass",             //
      "softplus_activation_fuse_pass",        //
      "shuffle_channel_detect_pass",          //
      "elementwise_act_onednn_fuse_pass",     //
      "operator_scale_onednn_fuse_pass",      //
      "operator_unsqueeze_onednn_fuse_pass",  //
      "operator_reshape_onednn_fuse_pass",    //
      "onednn_placement_pass",                //
};

const std::vector<std::string> kPirMkldnnBf16Passes{
    "add_shadow_output_after_dead_parameter_pass",
    "cpu_bfloat16_placement_pass",
    "cpu_bfloat16_pass",
    "cpu_bfloat16_type_placement_pass",
    "cpu_special_ops_bf16_pass",
    "cpu_bf16_quantize_squash_pass",
};

const std::vector<std::string> kPirCpuPasses{
    "add_shadow_output_after_dead_parameter_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass"};

}  // namespace paddle
