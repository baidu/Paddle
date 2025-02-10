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

///
/// \file paddle_analysis_config.h
///
/// \brief Paddle Analysis Config API信息
///
/// \author paddle-infer@baidu.com
/// \date 2020-03-20
/// \since 1.7
///

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle_infer_declare.h"  // NOLINT
/*! \file */
// Here we include some header files with relative paths, for that in deploy,
// the abstract path of this header file will be changed.
#include "paddle_api.h"           // NOLINT
#include "paddle_pass_builder.h"  // NOLINT

namespace paddle {

class AnalysisPredictor;

struct PD_INFER_DECL XpuConfig {
  // Select which xpu device to run model.
  int device_id{0};

  // Available l3 size (Byte)
  // For kunlun1, max l3_size is 16773120 Byte
  // For kunlun2, max l3_size is 67104768 Byte
  size_t l3_size{0};
  // If l3_ptr is not nullptr, it is used as l3 buffer.
  // If l3_ptr is nullptr, new l3 buffer will be created.
  void* l3_ptr{nullptr};
  // Available l3 size for autotune.
  // If l3_autotune_size is 0, autotune is closed.
  // Note: The remaining l3 size (l3_size - l3_autotune_size) is for
  // kernels (both paddle/xdnn kernels)
  size_t l3_autotune_size{0};

  // Reserved xpu global memory size for xpu_context;
  // If not set(-1), default memory size for xpu_context is 128MB in XPU2 or
  // 64MB in XPU1. If set 1*1024*1024, memory size for xpu_context will be 1MB;
  int context_gm_size{-1};
  // xpu_context(from baidu::xpu::api::create_context) for execution.
  // If context is nullptr, new context will be created by default.
  void* context{nullptr};
  // Stream for execution.
  // If stream is nullptr, default stream will be used.
  void* stream{nullptr};

  // Conv autotune level. Default 0 means no autotune.
  int conv_autotune_level{0};
  // Base conv autotune info is read from conv_autotune_file.
  std::string conv_autotune_file;
  // Whether write new conv autotune info to conv_autotune_file.
  bool conv_autotune_file_writeback{false};

  // Fc autotune level. The Optional values are 0-9. Default 0 means no
  // autotune.
  int fc_autotune_level{0};
  // Base fc autotune info is read from fc_autotune_file.
  std::string fc_autotune_file;
  // Whether write new fc autotune info to fc_autotune_file.
  bool fc_autotune_file_writeback{false};

  // Gemm compute precision. Optional values are 0(int8),1(int16),2(int31).
  // Note: "gemm_compute_precision" has no effect on quanted ops of quant model
  // Note: Paddle-Lite only.
  int gemm_compute_precision{1};
  // Which method to optimize softmax in transformer structure. Optional values
  // are 0,1,2. Note: Paddle-Lite only.
  int transformer_softmax_optimize_level{0};
  // Whether enable adaptive_seqlen optimize on transformer encoder.
  // Note: Paddle-Lite only.
  bool transformer_encoder_adaptive_seqlen{true};

  // Gelu out max threshold is limited to quant_post_static_gelu_out_threshold
  // if use static post-quantization.
  // Note: Paddle-Lite only.
  float quant_post_static_gelu_out_threshold{10.f};
  // Activation method if use dynamic post-quantization.
  // For kunlun1, optional values are 0(per_tensor),1(per_batch),2(per_head).
  // For kunlun2, optional values are 0(per_tensor) or non-zero(every_16).
  // Note: Paddle-Lite only.
  int quant_post_dynamic_activation_method{0};
  // Preprocess weight to quant_post_dynamic_weight_precision if use dynamic
  // post-quantization. Optional values is 0,1,2.
  // * If 0, preprocess weight to int8.
  // * If 1, preprocess weight to int16.
  // * If 2, preprocess weight to float.
  // Note: PaddleInference only.
  int quant_post_dynamic_weight_precision{1};
  std::vector<std::string> quant_post_dynamic_op_types;
  // fc, conv2d
  // 0: int8 per tensor, 1: int8 per-channel, 2: int16 per-tensor(default), 3:
  // int16 per-channel, 4: int31 per-tensor. Note: PaddleInference only.
  std::map<std::string, int> quant_post_dynamic_weight_methods;
};

///
/// \brief configuration manager for AnalysisPredictor.
/// \since 1.7.0
///
/// AnalysisConfig manages configurations of AnalysisPredictor.
/// During inference procedure, there are many parameters(model/params path,
/// place of inference, etc.)
/// to be specified, and various optimizations(subgraph fusion, memory
/// optimization, TensorRT engine, etc.)
/// to be done. Users can manage these settings by creating and modifying an
/// AnalysisConfig,
/// and loading it into AnalysisPredictor.
///
struct PD_INFER_DECL AnalysisConfig {
  AnalysisConfig();
  ///
  /// \brief Construct a new AnalysisConfig from another
  /// AnalysisConfig.
  ///
  /// \param[in] other another AnalysisConfig
  ///
  AnalysisConfig(const AnalysisConfig& other);
  ///
  /// \brief Construct a new AnalysisConfig from a no-combined model.
  ///
  /// \param[in] model_dir model directory of the no-combined model.
  ///
  explicit AnalysisConfig(const std::string& model_dir);
  ///
  /// \brief Construct a new AnalysisConfig from a combined model.
  ///
  /// \param[in] prog_file_or_model_dir model file path of the combined model or
  /// the directory path containing the model. \param[in]
  /// params_file_or_model_prefix params file path of the combined model or the
  /// model prefix.
  ///
  explicit AnalysisConfig(const std::string& prog_file_or_model_dir,
                          const std::string& params_file_or_model_prefix);
  ///
  /// \brief Precision of inference.
  ///
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
    kBf16,         ///< bf16
  };

  ///
  /// \brief Set the no-combined model dir path.
  ///
  /// \param model_dir model dir path.
  ///
  void SetModel(const std::string& model_dir) { model_dir_ = model_dir; }

  ///
  /// \brief Set the combined model with two specific paths for program and
  /// parameters.
  ///
  /// \param prog_file_path_or_model_dir_path model file path of the combined
  /// model or the directory path containing the model. \param
  /// params_file_path_or_model_prefix params file path of the combined model or
  /// the model prefix.
  ///
  void SetModel(const std::string& prog_file_path_or_model_dir_path,
                const std::string& params_file_path_or_model_prefix);
  ///
  /// \brief Set the model file path of a combined model.
  ///
  /// \param x model file path.
  ///
  void SetProgFile(const std::string& x) { prog_file_ = x; }
  ///
  /// \brief Set the params file path of a combined model.
  ///
  /// \param x params file path.
  ///
  void SetParamsFile(const std::string& x) { params_file_ = x; }

  ///
  /// \brief Save optimized model.
  ///
  /// \param save_optimized_model whether to enable save optimized model.
  ///
  void EnableSaveOptimModel(bool save_optimized_model) {
    save_optimized_model_ = save_optimized_model;
  }
  ///
  /// \brief Set the path of optimization cache directory.
  ///
  /// \param opt_cache_dir the path of optimization cache directory.
  ///
  void SetOptimCacheDir(const std::string& opt_cache_dir) {
    opt_cache_dir_ = opt_cache_dir;
  }
  ///
  /// \brief Get the model directory path.
  ///
  /// \return const std::string& The model directory path.
  ///
  const std::string& model_dir() const { return model_dir_; }
  ///
  /// \brief Get the program file path.
  ///
  /// \return const std::string& The program file path.
  ///
  const std::string& prog_file() const { return prog_file_; }
  ///
  /// \brief Get the combined parameters file.
  ///
  /// \return const std::string& The combined parameters file.
  ///
  const std::string& params_file() const { return params_file_; }

  // Padding related.

  ///
  /// \brief Turn off FC Padding.
  ///
  ///
  void DisableFCPadding();
  ///
  /// \brief A boolean state telling whether fc padding is used.
  ///
  /// \return bool Whether fc padding is used.
  ///
  bool use_fc_padding() const { return use_fc_padding_; }

  // GPU related.

  ///
  /// \brief Turn on GPU.
  ///
  /// \param memory_pool_init_size_mb initial size of the GPU memory pool in MB.
  /// \param device_id device_id the GPU card to use (default is 0).
  /// \param precision the precision used in Paddle-GPU inference.
  ///
  void EnableUseGpu(uint64_t memory_pool_init_size_mb,
                    int device_id = 0,
                    Precision precision_mode = Precision::kFloat32);

  ///
  /// \brief Turn off GPU.
  ///
  ///
  void DisableGpu();

  ///
  /// \brief Turn on XPU.
  ///
  /// \param l3_workspace_size The size of the video memory allocated by the l3
  ///       cache, the maximum is 16M.
  /// \param l3_locked Whether the allocated L3 cache can be locked. If false,
  ///       it means that the L3 cache is not locked, and the allocated L3
  ///       cache can be shared by multiple models, and multiple models
  ///       sharing the L3 cache will be executed sequentially on the card.
  /// \param conv_autotune Whether to autotune the conv operator in the model.
  ///       If true, when the conv operator of a certain dimension is executed
  ///       for the first time, it will automatically search for a better
  ///       algorithm to improve the performance of subsequent conv operators
  ///       of the same dimension.
  /// \param conv_autotune_file Specify the path of the autotune file. If
  ///       autotune_file is specified, the algorithm specified in the
  ///       file will be used and autotune will not be performed again.
  /// \param transformer_encoder_precision Calculation accuracy of multi_encoder
  /// \param transformer_encoder_adaptive_seqlen Is the input of multi_encoder
  ///       variable length
  /// \param enable_multi_stream Whether to enable the multi
  ///       stream of xpu.
  ///
  void EnableXpu(int l3_size = 0xfffc00,
                 bool l3_locked = false,
                 bool conv_autotune = false,
                 const std::string& conv_autotune_file = "",
                 const std::string& transformer_encoder_precision = "int16",
                 bool transformer_encoder_adaptive_seqlen = false,
                 bool enable_multi_stream = false);

  ///
  /// \brief configs of XPU
  ///
  /// \param config Configs for xpu. See XpuConfig for more details.
  ///
  void SetXpuConfig(const XpuConfig& config);

  ///
  /// \brief Get configs of xpu
  ///
  /// \return XpuConfig The configs of xpu.
  ///
  XpuConfig xpu_config() { return xpu_config_; }

  ///
  /// \brief configs of IPU
  ///
  enum class ipu_config_code {
    ipu_device_num,
    ipu_micro_batch_size,
    ipu_enable_pipelining,
    ipu_batches_per_step,
    ipu_enable_fp16,
    ipu_replica_num,
    ipu_available_memory_proportion,
    ipu_enable_half_partial,
    ipu_custom_ops_info,
    ipu_custom_patterns,
    ipu_enable_model_runtime_executor,
  };

  ///
  /// \brief Turn on IPU.
  ///
  /// \param ipu_device_num the number of IPUs.
  /// \param ipu_micro_batch_size the batch size in the graph, only work with
  /// mutable input shapes.
  /// \param ipu_enable_pipelining enable pipelining.
  /// \param ipu_batches_per_step the number of batches per run in pipelining.
  ///
  void EnableIpu(int ipu_device_num = 1,
                 int ipu_micro_batch_size = 1,
                 bool ipu_enable_pipelining = false,
                 int ipu_batches_per_step = 1);

  ///
  /// \brief Set IPU config.
  ///
  /// \param ipu_enable_fp16 enable fp16.
  /// \param ipu_replica_num the number of graph replication.
  /// \param ipu_available_memory_proportion the available memory proportion for
  /// matmul/conv.
  /// \param ipu_enable_half_partial enable fp16 partial for matmul, only work
  /// with fp16.
  /// \param ipu_enable_model_runtime_executor whether to use model_runtime
  /// executor.
  ///
  void SetIpuConfig(bool ipu_enable_fp16 = false,
                    int ipu_replica_num = 1,
                    float ipu_available_memory_proportion = 1.0,
                    bool ipu_enable_half_partial = false,
                    bool ipu_enable_model_runtime_executor = false);

  ///
  /// \brief Set IPU custom ops and patterns.
  ///
  /// \param custom_ops_info the mapper of paddle custom ops and popart ops.
  /// e.g. {{paddle_op_name, popart_op_name, op_domain, op_version}}.
  /// \param custom_patterns the names of popart patterns. e.g. {{pattern_name,
  /// enable_pattern}}}
  ///
  void SetIpuCustomInfo(
      const std::vector<std::vector<std::string>>& ipu_custom_ops_info = {},
      const std::map<std::string, bool>& ipu_custom_patterns = {});

  ///
  /// \brief Load IPU config from configuration file.
  ///
  /// \param config_path configure file path for ipu.
  ///
  void LoadIpuConfig(const std::string& config_path);

  ///
  /// \brief Set XPU device id.
  ///
  /// \param device_id the XPU card to use (default is 0).
  ///
  void SetXpuDeviceId(int device_id = 0);
  ///
  /// \brief Turn on CustomDevice.
  ///
  /// \param device_type device_type the custom device to use.
  ///
  /// \param device_id device_id the custom device to use (default is 0).
  ///
  void EnableCustomDevice(const std::string& device_type,
                          int device_id = 0,
                          Precision precision_mode = Precision::kFloat32);
  ///
  /// \brief Turn on ONNXRuntime.
  ///
  void EnableONNXRuntime();
  ///
  /// \brief Turn off ONNXRuntime.
  ///
  void DisableONNXRuntime();
  ///
  /// \brief Turn on ONNXRuntime Optimization.
  ///
  void EnableORTOptimization();
  ///
  /// \brief A boolean state telling whether the GPU is turned on.
  ///
  /// \return bool Whether the GPU is turned on.
  ///
  bool use_gpu() const { return use_gpu_; }
  ///
  /// \brief When running the fp16 model on Nvidia GPU, you can also try running
  /// your model on cutlass.
  ///
  void Exp_EnableUseCutlass();
  ///
  ///
  /// \brief A boolean state telling whether the XPU is turned on.
  ///
  /// \return bool Whether the XPU is turned on.
  ///
  bool use_xpu() const { return use_xpu_; }
  /// \brief A boolean state telling whether the IPU is turned on.
  ///
  /// \return bool Whether the IPU is turned on.
  ///
  bool use_ipu() const { return use_ipu_; }
  /// \brief A boolean state telling whether the CustomDevice is turned on.
  ///
  /// \return bool Whether the CustomDevice is turned on.
  ///
  bool use_custom_device() const { return use_custom_device_; }
  ///
  /// \brief A boolean state telling whether the ONNXRuntime is turned on.
  ///
  /// \return bool Whether the ONNXRuntime is turned on.
  ///
  bool use_onnxruntime() const { return use_onnxruntime_; }
  ///
  /// \brief A boolean state telling whether the ONNXRuntime Optimization is
  /// turned on.
  ///
  /// \return bool Whether the ONNXRuntime Optimization is turned on.
  ///
  bool ort_optimization_enabled() const { return enable_ort_optimization_; }
  ///
  /// \brief Get the GPU device id.
  ///
  /// \return int The GPU device id.
  ///
  int gpu_device_id() const { return gpu_device_id_; }
  ///
  /// \brief Get the XPU device id.
  ///
  /// \return int The XPU device id.
  ///
  int xpu_device_id() const { return xpu_config_.device_id; }
  /// \brief Get the number of IPU device .
  ///
  /// \return int The number of IPU device.
  ///
  int ipu_device_num() const { return ipu_device_num_; }
  ///
  /// \brief Get the custom device id.
  ///
  /// \return int The custom device id.
  ///
  int custom_device_id() const { return custom_device_id_; }
  /// \brief Get the custom device type.
  ///
  /// \return string The custom device type.
  ///
  std::string custom_device_type() const { return custom_device_type_; }
  /// \brief Get whether the custom device mixed precision is enabled.
  ///
  /// \return bool custom device mixed is enabled.
  ///
  bool enable_custom_device_mixed() const {
    return enable_custom_device_mixed_;
  }
  ///
  /// \brief Get the initial size in MB of the GPU memory pool.
  ///
  /// \return int The initial size in MB of the GPU memory pool.
  ///
  int memory_pool_init_size_mb() const { return memory_pool_init_size_mb_; }
  ///
  /// \brief Get the proportion of the initial memory pool size compared to the
  /// device.
  ///
  /// \return float The proportion of the initial memory pool size.
  ///
  float fraction_of_gpu_memory_for_pool() const;

  // CUDNN related.
  ///
  /// \brief Turn on CUDNN.
  ///
  ///
  void EnableCUDNN();
  ///
  /// \brief A boolean state telling whether to use CUDNN.
  ///
  /// \return bool Whether to use CUDNN.
  ///
  bool cudnn_enabled() const { return use_cudnn_; }

  ///
  /// \brief Control whether to perform IR graph optimization.
  /// If turned off, the AnalysisConfig will act just like a NativeConfig.
  ///
  /// \param x Whether the ir graph optimization is activated.
  ///
  void SwitchIrOptim(int x = true) { enable_ir_optim_ = x; }
  ///
  /// \brief A boolean state telling whether the ir graph optimization is
  /// activated.
  ///
  /// \return bool Whether to use ir graph optimization.
  ///
  bool ir_optim() const { return enable_ir_optim_; }
  ///
  /// \brief INTERNAL Determine whether to use the feed and fetch operators.
  /// Just for internal development, not stable yet.
  /// When ZeroCopyTensor is used, this should be turned off.
  ///
  /// \param x Whether to use the feed and fetch operators.
  ///
  void SwitchUseFeedFetchOps(int x = true) {}
  ///
  /// \brief A boolean state telling whether to use the feed and fetch
  /// operators.
  ///
  /// \return bool Whether to use the feed and fetch operators.
  ///
  bool use_feed_fetch_ops_enabled() const { return false; }

  ///
  /// \brief Turn on the feed and fetch data with low precision.
  ///
  /// \param x Whether to enable feed and fetch data with low precision.
  ///
  void EnableLowPrecisionIO(bool x = true);

  ///
  /// \brief Control whether to specify the inputs' names.
  /// The ZeroCopyTensor type has a name member, assign it with the
  /// corresponding
  /// variable name. This is used only when the input ZeroCopyTensors passed to
  /// the
  /// AnalysisPredictor.ZeroCopyRun() cannot follow the order in the training
  /// phase.
  ///
  /// \param x Whether to specify the inputs' names.
  ///
  void SwitchSpecifyInputNames(bool x = true) { specify_input_name_ = x; }
  ///
  /// \brief A boolean state tell whether the input ZeroCopyTensor names
  /// specified should
  /// be used to reorder the inputs in AnalysisPredictor.ZeroCopyRun().
  ///
  /// \return bool Whether to specify the inputs' names.
  ///
  bool specify_input_name() const { return specify_input_name_; }

  ///
  /// \brief Turn on the OpenVINO engine.
  /// The OpenVINO engine will accelerate some subgraphs in the original Fluid
  /// computation graph. In some models such as resnet50, GoogleNet and so on,
  /// it gains significant performance acceleration.
  ///
  void EnableOpenVINOEngine(Precision inference_precision);

  ///
  /// \brief A boolean state telling whether the OpenVINO engine is used.
  ///
  /// \return bool Whether the OpenVINO engine is used.
  ///
  bool openvino_engine_enabled() const;

  ///
  /// \brief Turn on the TensorRT engine.
  /// The TensorRT engine will accelerate some subgraphs in the original Fluid
  /// computation graph. In some models such as resnet50, GoogleNet and so on,
  /// it gains significant performance acceleration.
  ///
  /// \param workspace_size The memory size(in byte) used for TensorRT
  /// workspace.
  /// \param max_batch_size The maximum batch size of this prediction task,
  /// better set as small as possible for less performance loss.
  /// \param min_subgraph_size The minimum TensorRT subgraph size needed, if a
  /// subgraph is smaller than this, it will not be transferred to TensorRT
  /// engine.
  /// \param precision The precision used in TensorRT.
  /// \param use_static Serialize optimization information to disk for reusing.
  /// \param use_calib_mode Use TRT int8 calibration(post training
  /// quantization).
  /// \param use_cuda_graph Use CudaGraph to reduce the time consumption of
  /// enqueue. Note that this option can only be enabled when your input is
  /// constant (including the batch dimension).
  ///
  ///
  void EnableTensorRtEngine(int64_t workspace_size = 1 << 30,
                            int max_batch_size = 1,
                            int min_subgraph_size = 3,
                            Precision precision = Precision::kFloat32,
                            bool use_static = false,
                            bool use_calib_mode = true,
                            bool use_cuda_graph = false);
  ///
  /// \brief A boolean state telling whether the TensorRT engine is used.
  ///
  /// \return bool Whether the TensorRT engine is used.
  ///
  bool tensorrt_engine_enabled() const { return use_tensorrt_; }
  ///
  /// \brief Whether to get the intermediate output of TensorRT Engine.
  ///
  /// \param output_tensor_names The name of the Tensor that needs to be marked
  ///
  void MarkTrtEngineOutputs(
      const std::vector<std::string>& output_tensor_names = {});
  ///
  /// \brief Turn on the TensorRT memory optimization.
  ///
  /// \param engine_memory_sharing Whether to enable TensorRT memory
  /// optimization.
  /// \param sharing_identifier This parameter can be set if TensorRT memory
  /// optimization is enabled, and the value must be greater than 0. If you have
  /// multiple predictors that want to share memory, you can specify a
  /// same value for these predictors. NOTE: The predictors specified with the
  /// same value must be guaranteed to be executed serially, otherwise undefined
  /// behavior will occur.
  ///
  void EnableTensorRTMemoryOptim(bool engine_memory_sharing = true,
                                 int sharing_identifier = 0);
  ///
  /// \brief A boolean state telling whether the tensorrt engine memory sharing
  /// is activated.
  ///
  /// \return bool Whether the tensorrt engine memory sharing is activated.
  ///
  bool trt_engine_memory_sharing() const;
  ///
  /// \brief  Get the TensorRT engine precision.
  ///
  /// \return Precision Get the TensorRT engine precision.
  ///
  Precision tensorrt_precision_mode() const { return tensorrt_precision_mode_; }
  ///
  /// \brief Set min, max, opt shape for TensorRT Dynamic shape mode.
  /// \param min_input_shape The min input shape of the subgraph input.
  /// \param max_input_shape The max input shape of the subgraph input.
  /// \param opt_input_shape The opt input shape of the subgraph input.
  /// \param disable_trt_plugin_fp16 Setting this parameter to true means that
  /// TRT plugin will not run fp16.
  ///
  void SetTRTDynamicShapeInfo(
      std::map<std::string, std::vector<int>> min_input_shape,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::map<std::string, std::vector<int>> optim_input_shape,
      bool disable_trt_plugin_fp16 = false);
  ///
  /// \brief A boolean state telling whether the trt dynamic_shape is used.
  ///
  /// \return bool Whether the trt dynamic_shape is used.
  ///
  bool tensorrt_dynamic_shape_enabled() const {
    return !min_input_shape_.empty();
  }
  ///
  /// \brief Enable tuned tensorrt dynamic shape.
  ///
  /// \param shape_range_info_path the path to shape_info file got in
  /// CollectShapeInfo
  /// mode.
  /// \param allow_build_at_runtime allow build trt engine at runtime.
  ///
  void EnableTunedTensorRtDynamicShape(
      const std::string& shape_range_info_path = "",
      bool allow_build_at_runtime = true);

  ///
  /// \brief A boolean state telling whether to use tuned tensorrt dynamic
  /// shape.
  ///
  bool tuned_tensorrt_dynamic_shape() const;

  ///
  /// \brief A boolean state telling whether to allow building trt engine at
  /// runtime.
  ///
  bool trt_allow_build_at_runtime() const;

  ///
  /// \brief Set execution stream. If not set a stream will be created
  /// internally.
  ///
  void SetExecStream(void* stream);

  ///
  /// \brief Get execution stream. The user needs to explicitly cast into a
  /// stream type such as cudaStream_t, hipStream_t, etc.
  ///
  void* GetExecStream() const;

  ///
  /// \brief Whether the external stream is used, if True, the predictor clone
  /// operation must use the external stream, otherwise the framework manages
  /// the stream internally.
  ///
  bool external_stream_enabled() const;

  ///
  /// \brief Collect shape info of all tensors in compute graph.
  ///
  /// \param shape_range_info_path the path to save shape info.
  ///
  void CollectShapeRangeInfo(const std::string& shape_range_info_path);

  ///
  /// \brief the shape info path in CollectShapeInfo mode.
  ///
  /// \return the shape info path.
  ///
  const std::string& shape_range_info_path() const;

  ///
  /// \brief A boolean state telling whether to collect shape info.
  ///
  /// \return bool Whether to collect shape info.
  ///
  bool shape_range_info_collected() const;

  ///
  /// \brief Prevent ops running in Paddle-TRT
  /// NOTE: just experimental, not an official stable API, easy to be broken.
  ///
  void Exp_DisableTensorRtOPs(const std::vector<std::string>& ops);

  void TensorRTRefitParamsPath(const std::string& refit_params_path);

  const std::string& refit_params_path() const { return refit_params_path_; }

  ///
  /// \brief Prevent TensorRtSubgraph running in Paddle-TRT
  /// NOTE: just experimental, not an official stable API, easy to be broken.
  ///
  void Exp_DisableTensorRtSubgraph(
      const std::vector<std::string>& var_name_not_trt);

  ///
  /// \brief Specify TensorRT subgraph precision,fp16, int8 or bfp16(TensorRT
  /// Version>=9.0) NOTE: just experimental, not an official stable API, easy to
  /// be broken.
  ///
  void Exp_SpecifyTensorRTSubgraphPrecision(
      const std::vector<std::string>& trt_parameters_fp16,
      const std::vector<std::string>& trt_parameters_int8,
      const std::vector<std::string>& trt_parameters_bfp16);

  ///
  /// \brief Prevent DynamicShape OPs running in Paddle-TRT
  /// NOTE: just experimental, not an official stable API, easy to be broken.
  ///
  void Exp_DisableTensorRTDynamicShapeOPs(bool trt_forbid_dynamic_op);

  ///
  /// \brief Replace some TensorRT plugins to TensorRT OSS(
  /// https://github.com/NVIDIA/TensorRT), with which some models's inference
  /// may be more high-performance. Libnvinfer_plugin.so greater than
  /// V7.2.1 is needed.
  ///
  void EnableVarseqlen();

  ///
  /// \brief A boolean state telling whether to use the TensorRT OSS.
  ///
  /// \return bool Whether to use the TensorRT OSS.
  ///
  bool tensorrt_varseqlen_enabled() { return trt_use_varseqlen_; }

  ///
  /// \brief Enable TensorRT DLA
  /// \param dla_core ID of DLACore, which should be 0, 1,
  ///        ..., IBuilder.getNbDLACores() - 1
  ///
  void EnableTensorRtDLA(int dla_core = 0);

  ///
  /// \brief A boolean state telling whether to use the TensorRT DLA.
  ///
  /// \return bool Whether to use the TensorRT DLA.
  ///
  bool tensorrt_dla_enabled() { return trt_use_dla_; }

  ///
  /// \brief A boolean state telling whether to show TensorRT inspector
  /// information.
  ///
  /// \return bool Whether to show TensorRT inspector information.
  ///
  void EnableTensorRtInspector(bool inspector_serialize = false);
  bool tensorrt_inspector_enabled() { return trt_use_inspector_; }

  ///
  /// \brief A boolean state telling whether to use TensorRT explicit
  /// quantization.
  ///
  /// \return bool Whether to use TensorRT explicit quantization.
  ///
  void EnableTensorRtExplicitQuantization();
  bool tensorrt_explicit_quantization_enabled() {
    return trt_use_explicit_quantization_;
  }

  ///
  /// \brief Set the optimization level of TensorRT
  /// \param level The optimization level
  /// The API accepts level in range [0, 5].
  /// Higher optimization level allows the optimizer to spend more time
  /// searching for optimization opportunities. The API supports TRT version
  /// >= 8.6, and takes no effect instead.
  ///
  void SetTensorRtOptimizationLevel(int level);

  ///
  /// \brief An integer telling the TRT optimization level.
  ///
  /// \return integer The TRT optimization level.
  ///
  int tensorrt_optimization_level() { return trt_optimization_level_; }

  /// \brief A boolean state telling whether to use new executor.
  ///
  /// \return bool whether to use new executor.
  ///
  void EnableNewExecutor(bool x = true) { use_new_executor_ = x; }

  bool new_executor_enabled() const { return use_new_executor_; }

  /// \brief A boolean state telling whether to use new IR.
  ///
  /// \return bool whether to use new IR.
  ///
  void EnableNewIR(bool x = true) { use_pir_ = x; }

  bool new_ir_enabled() const { return use_pir_; }

  ///
  /// \brief Control whether to use optimized model to inference.
  ///
  /// \param x whether to use optimized model.
  ///
  void UseOptimizedModel(bool x = true) { use_optimized_model_ = x; }

  ///
  /// \brief Control whether to debug IR graph analysis phase.
  /// This will generate DOT files for visualizing the computation graph after
  /// each analysis pass applied.
  ///
  /// \param x whether to debug IR graph analysis phase.
  ///
  void SwitchIrDebug(int x = true, const std::vector<std::string>& passes = {});

  ///
  /// \brief Turn on OneDNN.
  ///
  ///
  void EnableMKLDNN();

  ///
  /// \brief Turn down OneDNN.
  ///
  ///
  void DisableMKLDNN();

  ///
  /// \brief Set the cache capacity of different input shapes for OneDNN.
  /// Default value 0 means not caching any shape.
  /// Please see MKL-DNN Data Caching Design Document:
  /// https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md
  ///
  /// \param capacity The cache capacity.
  ///
  void SetMkldnnCacheCapacity(int capacity);
  ///
  /// \brief A boolean state telling whether to use the OneDNN.
  ///
  /// \return bool Whether to use the OneDNN.
  ///
  bool mkldnn_enabled() const { return use_mkldnn_; }

  ///
  /// \brief Set the number of cpu math library threads.
  ///
  /// \param cpu_math_library_num_threads The number of cpu math library
  /// threads.
  ///
  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
  ///
  /// \brief An int state telling how many threads are used in the CPU math
  /// library.
  ///
  /// \return int The number of threads used in the CPU math library.
  ///
  int cpu_math_library_num_threads() const {
    return cpu_math_library_num_threads_;
  }

  ///
  /// \brief Transform the AnalysisConfig to NativeConfig.
  ///
  /// \return NativeConfig The NativeConfig transformed.
  ///
  NativeConfig ToNativeConfig() const;
  ///
  /// \brief Specify the operator type list to use OneDNN acceleration.
  ///
  /// \param op_list The operator type list.
  ///
  void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
    mkldnn_enabled_op_types_ = op_list;
  }

  ///
  /// \brief Turn on OneDNN int8.
  ///
  /// \param op_list The operator type list.
  ///
  void EnableMkldnnInt8(const std::unordered_set<std::string>& op_list = {});

  ///
  /// \brief A boolean state telling whether to use the OneDNN Int8.
  ///
  /// \return bool Whether to use the OneDNN Int8.
  ///
  bool mkldnn_int8_enabled() const { return use_mkldnn_int8_; }

  ///
  /// \brief Turn on OneDNN bfloat16.
  ///
  ///
  void EnableMkldnnBfloat16();

  ///
  /// \brief Turn off OneDNN fc passes.
  ///
  void DisableMkldnnFcPasses();

  ///
  /// \brief A boolean state telling whether to disable the OneDNN Fc passes.
  ///
  /// \return bool Whether to disable the OneDNN Fc passes.
  ///
  bool mkldnn_fc_passes_disabled() const { return disable_mkldnn_fc_passes_; }

  ///
  /// \brief A boolean state telling whether to use the OneDNN Bfloat16.
  ///
  /// \return bool Whether to use the OneDNN Bfloat16.
  ///
  bool mkldnn_bfloat16_enabled() const { return use_mkldnn_bfloat16_; }

  /// \brief Specify the operator type list to use Bfloat16 acceleration.
  ///
  /// \param op_list The operator type list.
  ///
  void SetBfloat16Op(std::unordered_set<std::string> op_list) {
    bfloat16_enabled_op_types_ = op_list;
  }

  ///
  /// \brief A boolean state telling whether the thread local CUDA stream is
  /// enabled.
  ///
  /// \return bool Whether the thread local CUDA stream is enabled.
  ///
  bool thread_local_stream_enabled() const { return thread_local_stream_; }

  ///
  /// \brief Specify the memory buffer of program and parameter.
  /// Used when model and params are loaded directly from memory.
  ///
  /// \param prog_buffer The memory buffer of program.
  /// \param prog_buffer_size The size of the model data.
  /// \param params_buffer The memory buffer of the combined parameters file.
  /// \param params_buffer_size The size of the combined parameters data.
  ///
  void SetModelBuffer(const char* prog_buffer,
                      size_t prog_buffer_size,
                      const char* params_buffer,
                      size_t params_buffer_size);
  ///
  /// \brief A boolean state telling whether the model is set from the CPU
  /// memory.
  ///
  /// \return bool Whether model and params are loaded directly from memory.
  ///
  bool model_from_memory() const { return model_from_memory_; }

  ///
  /// \brief Turn on memory optimize
  /// NOTE still in development.
  ///
  /// \param x Whether to enable memory optimize.
  ///
  void EnableMemoryOptim(bool x = true);
  ///
  /// \brief A boolean state telling whether the memory optimization is
  /// activated.
  ///
  /// \return bool Whether the memory optimization is activated.
  ///
  bool enable_memory_optim() const;

  ///
  /// \brief Turn on profiling report.
  /// If not turned on, no profiling report will be generated.
  ///
  void EnableProfile();
  ///
  /// \brief A boolean state telling whether the profiler is activated.
  ///
  /// \return bool Whether the profiler is activated.
  ///
  bool profile_enabled() const { return with_profile_; }

  ///
  /// \brief Mute all logs in Paddle inference.
  ///
  void DisableGlogInfo();
  ///
  /// \brief A boolean state telling whether logs in Paddle inference are muted.
  ///
  /// \return bool Whether logs in Paddle inference are muted.
  ///
  bool glog_info_disabled() const { return !with_glog_info_; }

  ///
  /// \brief Set the AnalysisConfig to be invalid.
  /// This is to ensure that an AnalysisConfig can only be used in one
  /// AnalysisPredictor.
  ///
  void SetInValid() const { is_valid_ = false; }
  ///
  /// \brief A boolean state telling whether the AnalysisConfig is valid.
  ///
  /// \return bool Whether the AnalysisConfig is valid.
  ///
  bool is_valid() const { return is_valid_; }

  friend class ::paddle::AnalysisPredictor;

  ///
  /// \brief Get a pass builder for customize the passes in IR analysis phase.
  /// NOTE: Just for developer, not an official API, easy to be broken.
  ///
  ///
  PassStrategy* pass_builder() const;

  ///
  /// \brief Enable the GPU multi-computing stream feature.
  /// NOTE: The current behavior of this interface is to bind the computation
  /// stream to the thread, and this behavior may be changed in the future.
  ///
  void EnableGpuMultiStream();

  ///
  /// \brief Print the summary of config.
  ///
  std::string Summary();

  ///
  /// \brief Set a list of operators that do not support mixed precision. This
  /// interface is in the experimental stage and may change in the future. Note
  /// that the blacklist must be the same as the model conversion blacklist.
  ///
  void Exp_DisableMixedPrecisionOps(
      const std::unordered_set<std::string>& black_list);

  ///
  /// \brief Set a list of operators that do support mixed precision. This
  /// interface is in the experimental stage and may change in the future. Note
  /// that the whitelist must be the same as the model conversion whitelist.
  ///
  void Exp_EnableMixedPrecisionOps(
      const std::unordered_set<std::string>& white_list);

  void SetApplyOptim(bool value) { apply_optim_ = value; }

  void SetSkipLoadParams(bool value) { skip_load_params_ = value; }

  ///
  /// \brief Enable use cinn compiler optimization.
  ///
  void EnableCINN();

  ///
  /// \brief A boolean state telling whether the CINN compiler optimization is
  /// turned on.
  ///
  /// \return bool Whether the CINN compiler optimization is turned on.
  ///
  bool cinn_enabled() const;

  ///
  /// \brief Set the custom passes list .
  ///
  /// \param passes The custom passes list.
  /// \param custom_pass_only Custom pass run mode. The default is false,
  /// which means that paddle pass will run after custom pass.
  ///
  void EnableCustomPasses(const std::vector<std::string>& passes,
                          bool custom_pass_only = false);

  ///
  /// \brief Delete a pass to prevent it to optimizing the model.
  ///
  /// \param pass_name The pass's name to be deleted.
  ///
  void DeletePass(const std::string& pass_name);

  ///
  /// \brief Set pir Optimization level.
  /// \param opt_level The optimization level
  /// The optimization Level in range [0,4], Default 2.
  /// Higher optimization level allows the predictor to apply more passes.
  /// If 0, Only basic pass support.
  /// If 1, Additional support for functional pass.
  /// If 2, Additional support the fusion logical pass,maybe affect precision
  /// and speed.
  /// If 3, support layout pass, etc.
  /// If 4, add the radicaloptimization, maybe affect precision, etc.
  ///
  void SetOptimizationLevel(int opt_level);

 protected:
  // Update the config.
  void Update();

  std::string SerializeInfoCache();

 protected:
  // Model paths.
  std::string model_dir_;
  std::string refit_params_path_;
  mutable std::string prog_file_;
  mutable std::string params_file_;

  // Mixed precision related.
  Precision mixed_precision_mode_{Precision::kFloat32};
  std::unordered_set<std::string> mixed_black_list_;
  std::unordered_set<std::string> mixed_white_list_;
  bool enable_low_precision_io_{false};

  // GPU related.
  bool use_gpu_{false};
  bool use_cutlass_{false};
  int gpu_device_id_{0};
  uint64_t memory_pool_init_size_mb_{100};  // initial size is 100MB.
  bool enable_gpu_mixed_{false};
  bool thread_local_stream_{false};

  bool use_cudnn_{false};
  bool use_external_stream_{false};
  void* exec_stream_{nullptr};

  // CustomDevice related
  bool use_custom_device_{false};
  int custom_device_id_{0};
  std::string custom_device_type_;
  bool enable_custom_device_mixed_{false};

  // ONNXRuntime related
  bool use_onnxruntime_{false};
  bool enable_ort_optimization_{false};

  // Padding related
  bool use_fc_padding_{true};

  // OpenVINO related.
  bool use_openvino_{false};
  Precision openvino_inference_precision_{Precision::kFloat32};

  // TensorRT related.
  bool use_tensorrt_{false};
  // For workspace_size, refer it from here:
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#troubleshooting
  int64_t tensorrt_workspace_size_{1 << 30};
  // While TensorRT allows an engine optimized for a given max batch size
  // to run at any smaller size, the performance for those smaller
  // sizes may not be as well-optimized. Therefore, Max batch is best
  // equivalent to the runtime batch size.
  int tensorrt_max_batchsize_{1};
  //  We transform the Ops that can be converted into TRT layer in the model,
  //  and aggregate these Ops into subgraphs for TRT execution.
  //  We set this variable to control the minimum number of nodes in the
  //  subgraph, 3 as default value.
  int tensorrt_min_subgraph_size_{3};
  Precision tensorrt_precision_mode_{Precision::kFloat32};
  bool trt_use_static_engine_{false};
  bool trt_use_calib_mode_{true};
  bool trt_use_cuda_graph_{false};
  bool trt_use_varseqlen_{false};
  bool trt_with_interleaved_{false};
  bool trt_mark_output_{false};
  bool trt_forbid_dynamic_op_{false};

  std::vector<std::string> trt_output_tensor_names_{};
  std::vector<std::string> trt_exclude_var_names_{};
  std::vector<std::string> trt_parameters_run_fp16_{};
  std::vector<std::string> trt_parameters_run_int8_{};
  std::vector<std::string> trt_parameters_run_bfp16_{};

  std::string tensorrt_transformer_posid_{""};
  std::string tensorrt_transformer_maskid_{""};
  bool trt_use_dla_{false};
  int trt_dla_core_{0};
  std::map<std::string, std::vector<int>> min_input_shape_{};
  std::map<std::string, std::vector<int>> max_input_shape_{};
  std::map<std::string, std::vector<int>> optim_input_shape_{};
  std::vector<std::string> trt_disabled_ops_{};
  bool disable_trt_plugin_fp16_{false};
  bool trt_allow_build_at_runtime_{false};
  // tune to get dynamic_shape info.
  bool trt_tuned_dynamic_shape_{false};
  bool trt_use_inspector_{false};
  bool trt_inspector_serialize_{false};
  bool trt_use_explicit_quantization_{false};
  int trt_optimization_level_{3};

  // In CollectShapeInfo mode, we will collect the shape information of
  // all intermediate tensors in the compute graph and calculate the
  // min_shape, max_shape and opt_shape and save in shape_range_info_path_;
  bool collect_shape_range_info_{false};
  std::string shape_range_info_path_;

  // memory reuse related.
  bool enable_memory_optim_{false};
  bool trt_engine_memory_sharing_{true};
  int trt_engine_memory_sharing_identifier_{0};

  std::unordered_set<std::string> trt_ops_run_float_;

#ifdef PADDLE_WITH_DNNL
  bool use_mkldnn_{true};
#else
  bool use_mkldnn_{false};
#endif
  std::unordered_set<std::string> mkldnn_enabled_op_types_;

  bool model_from_memory_{false};

  bool enable_ir_optim_{true};
  bool ir_debug_{false};

  bool use_optimized_model_{false};

  bool use_new_executor_{false};

  bool specify_input_name_{false};

  int cpu_math_library_num_threads_{1};

  bool with_profile_{false};

  bool with_glog_info_{true};

  // A runtime cache, shouldn't be transferred to others.
  std::string serialized_info_cache_;

  mutable std::unique_ptr<PassStrategy> pass_builder_;

  // CINN compiler related.
  bool use_cinn_{false};

  // XPU related.
  bool use_xpu_{false};
  XpuConfig xpu_config_;

  // onednn related.
  int mkldnn_cache_capacity_{10};
  bool use_mkldnn_bfloat16_{false};
  std::unordered_set<std::string> bfloat16_enabled_op_types_;
  bool use_mkldnn_int8_{false};
  std::unordered_set<int> quantize_excluded_op_ids_{};
  std::unordered_set<std::string> quantize_enabled_op_types_{};

  bool disable_mkldnn_fc_passes_{false};

  // ipu related.
  bool use_ipu_{false};
  int ipu_device_num_{1};
  int ipu_micro_batch_size_{1};
  bool ipu_enable_pipelining_{false};
  int ipu_batches_per_step_{1};

  bool ipu_enable_fp16_{false};
  int ipu_replica_num_{1};
  float ipu_available_memory_proportion_{1.0};
  bool ipu_enable_half_partial_{false};
  bool ipu_enable_model_runtime_executor_{false};

  std::vector<std::vector<std::string>> ipu_custom_ops_info_;
  std::vector<std::vector<std::string>> ipu_custom_patterns_;

  const std::unordered_map<std::string, ipu_config_code> ipu_config_mapper_ = {
      {"ipu_device_num", ipu_config_code::ipu_device_num},
      {"ipu_micro_batch_size", ipu_config_code::ipu_micro_batch_size},
      {"ipu_enable_pipelining", ipu_config_code::ipu_enable_pipelining},
      {"ipu_batches_per_step", ipu_config_code::ipu_batches_per_step},
      {"ipu_enable_fp16", ipu_config_code::ipu_enable_fp16},
      {"ipu_replica_num", ipu_config_code::ipu_replica_num},
      {"ipu_available_memory_proportion",
       ipu_config_code::ipu_available_memory_proportion},
      {"ipu_enable_half_partial", ipu_config_code::ipu_enable_half_partial},
      {"ipu_enable_model_runtime_executor",
       ipu_config_code::ipu_enable_model_runtime_executor},
      {"ipu_custom_ops_info", ipu_config_code::ipu_custom_ops_info},
      {"ipu_custom_patterns", ipu_config_code::ipu_custom_patterns}};

  // If the config is already used on a predictor, it becomes invalid.
  // Any config can only be used with one predictor.
  // Variables held by config can take up a lot of memory in some cases.
  // So we release the memory when the predictor is set up.
  mutable bool is_valid_{true};
  bool save_optimized_model_{false};
  std::string opt_cache_dir_;
  friend class paddle_infer::experimental::InternalUtils;

  // jit engine related
  // NOTE(Aureliue84): In case of Predictor in JITLayer, program is from outer
  // which means Predictor should apply optimization by calling
  // PrepareProgram(). So we add this flag to control the process.
  bool apply_optim_{false};
  bool skip_load_params_{false};

  bool use_pir_{false};
  std::vector<std::string> custom_passes_;
  bool custom_pass_only_{false};
  int pm_opt_level_{2};
  std::vector<std::string> ir_debug_passes_;
  std::vector<std::string> deleted_passes_;
};

}  // namespace paddle
