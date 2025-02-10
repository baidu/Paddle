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

/*
 * This file defines the class Argument, which is the input and output of the
 * analysis module. All the fields that needed either by Passes or PassManagers
 * are contained in Argument.
 *
 * TODO(Superjomn) Find some way better to contain the fields when it grow too
 * big.
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {

#ifdef PADDLE_WITH_DNNL
using VarQuantScale =
    std::unordered_map<std::string, std::pair<bool, phi::DenseTensor>>;
#endif

/*
 * The argument definition of both Pass and PassManagers.
 *
 * All the fields should be registered here for clearness.
 */
struct Argument {
  Argument() = default;
  explicit Argument(const std::string& model_dir) { SetModelDir(model_dir); }

  using unique_ptr_t = std::unique_ptr<void, std::function<void(void*)>>;
  using fusion_statis_t = std::unordered_map<std::string, int>;
  using input_shape_t = std::map<std::string, std::vector<int>>;

  bool Has(const std::string& key) const { return valid_fields_.count(key); }
  // If we set the model using config.SetModelBuffer,
  // the model and parameter will occupy additional CPU resources.
  // Use this interface to release these resources.
  void PartiallyRelease() {
    if (Has("model_program_path")) {
      if (Has("model_from_memory") && model_from_memory()) {
        model_program_path().clear();
        model_program_path().shrink_to_fit();
        model_params_path().clear();
        model_params_path().shrink_to_fit();
      }
    }
  }

#define DECL_ARGUMENT_FIELD(field__, Field, type__)                    \
 public:                                                               \
  type__& field__() {                                                  \
    PADDLE_ENFORCE_EQ(                                                 \
        Has(#field__),                                                 \
        true,                                                          \
        common::errors::PreconditionNotMet("There is no such field")); \
    return field__##_;                                                 \
  }                                                                    \
  void Set##Field(const type__& x) {                                   \
    field__##_ = x;                                                    \
    valid_fields_.insert(#field__);                                    \
  }                                                                    \
  DECL_ARGUMENT_FIELD_VALID(field__);                                  \
  type__* field__##_ptr() { return &field__##_; }                      \
                                                                       \
 private:                                                              \
  type__ field__##_;

#define DECL_POINTER_ARGUMENT_FIELD(field__, Field, type__)            \
 public:                                                               \
  type__& field__() {                                                  \
    PADDLE_ENFORCE_EQ(                                                 \
        Has(#field__),                                                 \
        true,                                                          \
        common::errors::PreconditionNotMet("There is no such field")); \
    return field__##_;                                                 \
  }                                                                    \
  void Set##Field(type__ x) {                                          \
    field__##_ = x;                                                    \
    valid_fields_.insert(#field__);                                    \
  }                                                                    \
  DECL_ARGUMENT_FIELD_VALID(field__);                                  \
  type__* field__##_ptr() { return &field__##_; }                      \
                                                                       \
 private:                                                              \
  type__ field__##_;

#define DECL_ARGUMENT_FIELD_VALID(field__) \
  bool field__##_valid() { return Has(#field__); }

#define DECL_ARGUMENT_UNIQUE_FIELD(field__, Field, type__)                \
 public:                                                                  \
  type__& field__() {                                                     \
    PADDLE_ENFORCE_NOT_NULL(                                              \
        field__##_,                                                       \
        common::errors::PreconditionNotMet("filed should not be null.")); \
    PADDLE_ENFORCE_EQ(                                                    \
        Has(#field__),                                                    \
        true,                                                             \
        common::errors::PreconditionNotMet("There is no such field"));    \
    return *static_cast<type__*>(field__##_.get());                       \
  }                                                                       \
  void Set##Field(type__* x) {                                            \
    field__##_ =                                                          \
        unique_ptr_t(x, [](void* x) { delete static_cast<type__*>(x); }); \
    valid_fields_.insert(#field__);                                       \
  }                                                                       \
  void Set##Field##NotOwned(type__* x) {                                  \
    valid_fields_.insert(#field__);                                       \
    field__##_ = unique_ptr_t(x, [](void* x UNUSED) {});                  \
  }                                                                       \
  DECL_ARGUMENT_FIELD_VALID(field__);                                     \
  type__* field__##_ptr() {                                               \
    PADDLE_ENFORCE_EQ(                                                    \
        Has(#field__),                                                    \
        true,                                                             \
        common::errors::PreconditionNotMet("There is no such field"));    \
    return static_cast<type__*>(field__##_.get());                        \
  }                                                                       \
  type__* Release##Field() {                                              \
    PADDLE_ENFORCE_EQ(                                                    \
        Has(#field__),                                                    \
        true,                                                             \
        common::errors::PreconditionNotMet("There is no such field"));    \
    valid_fields_.erase(#field__);                                        \
    return static_cast<type__*>(field__##_.release());                    \
  }                                                                       \
                                                                          \
 private:                                                                 \
  unique_ptr_t field__##_;

  DECL_ARGUMENT_FIELD(predictor_id, PredictorID, int);
  DECL_ARGUMENT_FIELD(root_predictor_id, RootPredictorID, int);
  // Model path
  DECL_ARGUMENT_FIELD(model_dir, ModelDir, std::string);
  // Model specified with program and parameters files.
  DECL_ARGUMENT_FIELD(model_program_path, ModelProgramPath, std::string);
  DECL_ARGUMENT_FIELD(model_params_path, ModelParamsPath, std::string);
  DECL_ARGUMENT_FIELD(refit_params_path, RefitParamsPath, std::string);
  DECL_ARGUMENT_FIELD(model_from_memory, ModelFromMemory, bool);
  DECL_ARGUMENT_FIELD(save_optimized_model, SaveOptimizedModel, bool);
  DECL_ARGUMENT_FIELD(optimized_model_save_path,
                      OptimizedModelSavePath,
                      std::string);
  DECL_ARGUMENT_FIELD(optim_cache_dir, OptimCacheDir, std::string);
  DECL_ARGUMENT_FIELD(enable_ir_optim, EnableIrOptim, bool);

  // For JITLayer
  DECL_ARGUMENT_FIELD(skip_load_params, SkipLoadParams, bool);

  // The overall graph to work on.
  DECL_ARGUMENT_UNIQUE_FIELD(main_graph, MainGraph, framework::ir::Graph);
  // The overall Scope to work on.
  DECL_ARGUMENT_UNIQUE_FIELD(scope, Scope, framework::Scope);

  // The default program, loaded from disk.
  DECL_ARGUMENT_UNIQUE_FIELD(main_program, MainProgram, framework::ProgramDesc);

  // The ir passes to perform in analysis phase.
  DECL_ARGUMENT_FIELD(ir_analysis_passes,
                      IrAnalysisPasses,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(analysis_passes,
                      AnalysisPasses,
                      std::vector<std::string>);

  // whether to mute all logs in inference.
  DECL_ARGUMENT_FIELD(disable_logs, DisableLogs, bool);

  // Pass a set of op types to enable its mkldnn kernel
  DECL_ARGUMENT_FIELD(mkldnn_enabled_op_types,
                      MKLDNNEnabledOpTypes,
                      std::unordered_set<std::string>);
  // The cache capacity of different input shapes for mkldnn.
  DECL_ARGUMENT_FIELD(mkldnn_cache_capacity, MkldnnCacheCapacity, int);

#ifdef PADDLE_WITH_DNNL
  // A set of op types to enable their quantized kernels
  DECL_ARGUMENT_FIELD(quantize_enabled_op_types,
                      QuantizeEnabledOpTypes,
                      std::unordered_set<std::string>);

  // A set of op IDs to exclude from enabling their quantized kernels
  DECL_ARGUMENT_FIELD(quantize_excluded_op_ids,
                      QuantizeExcludedOpIds,
                      std::unordered_set<int>);

  // Scales for variables to be quantized
  DECL_ARGUMENT_FIELD(quant_var_scales, QuantVarScales, VarQuantScale);

  // A set of op types to enable their bfloat16 kernels
  DECL_ARGUMENT_FIELD(bfloat16_enabled_op_types,
                      Bfloat16EnabledOpTypes,
                      std::unordered_set<std::string>);

  DECL_ARGUMENT_FIELD(use_mkldnn_int8, UseMkldnnInt8, bool);
#endif

  // Passed from config.
  DECL_ARGUMENT_FIELD(use_gpu, UseGPU, bool);
  DECL_ARGUMENT_FIELD(use_cutlass, UseCutlass, bool);
  DECL_ARGUMENT_FIELD(use_fc_padding, UseFcPadding, bool);
  DECL_ARGUMENT_FIELD(gpu_device_id, GPUDeviceId, int);
  DECL_ARGUMENT_FIELD(use_pir, UsePIR, bool);

  // Usually use for trt dynamic shape.
  // TRT will select the best kernel according to opt shape
  // Setting the disable_trt_plugin_fp16 to true means that TRT plugin will not
  // run fp16.
  DECL_ARGUMENT_FIELD(min_input_shape, MinInputShape, input_shape_t);
  DECL_ARGUMENT_FIELD(max_input_shape, MaxInputShape, input_shape_t);
  DECL_ARGUMENT_FIELD(optim_input_shape, OptimInputShape, input_shape_t);
  DECL_ARGUMENT_FIELD(disable_trt_plugin_fp16, CloseTrtPluginFp16, bool);

  DECL_ARGUMENT_FIELD(use_tensorrt, UseTensorRT, bool);
  DECL_ARGUMENT_FIELD(tensorrt_use_dla, TensorRtUseDLA, bool);
  DECL_ARGUMENT_FIELD(tensorrt_dla_core, TensorRtDLACore, int);
  DECL_ARGUMENT_FIELD(tensorrt_max_batch_size, TensorRtMaxBatchSize, int);
  DECL_ARGUMENT_FIELD(tensorrt_workspace_size, TensorRtWorkspaceSize, int64_t);
  DECL_ARGUMENT_FIELD(tensorrt_min_subgraph_size, TensorRtMinSubgraphSize, int);
  DECL_ARGUMENT_FIELD(trt_mark_output, TRTMarkOutput, bool);
  DECL_ARGUMENT_FIELD(trt_output_tensor_names,
                      TRTOutputTensorNames,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(trt_exclude_var_names,
                      TRTExcludeVarNames,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(trt_forbid_dynamic_op, TRTForbidDynamicOp, bool);

  DECL_ARGUMENT_FIELD(tensorrt_disabled_ops,
                      TensorRtDisabledOPs,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(trt_parameter_run_fp16,
                      TRTParameterRunFp16,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(trt_parameter_run_int8,
                      TRTParameterRunInt8,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(trt_parameter_run_bfp16,
                      TRTParameterRunBfp16,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(tensorrt_precision_mode, TensorRtPrecisionMode, int);
  DECL_ARGUMENT_FIELD(tensorrt_use_static_engine,
                      TensorRtUseStaticEngine,
                      bool);
  DECL_ARGUMENT_FIELD(tensorrt_use_calib_mode, TensorRtUseCalibMode, bool);
  DECL_ARGUMENT_FIELD(tensorrt_use_cuda_graph, TensorRtUseCudaGraph, bool);
  DECL_ARGUMENT_FIELD(tensorrt_use_varseqlen, TensorRtUseOSS, bool);
  DECL_ARGUMENT_FIELD(tensorrt_with_interleaved, TensorRtWithInterleaved, bool);
  DECL_ARGUMENT_FIELD(tensorrt_transformer_posid,
                      TensorRtTransformerPosid,
                      std::string);
  DECL_ARGUMENT_FIELD(tensorrt_transformer_maskid,
                      TensorRtTransformerMaskid,
                      std::string);
  DECL_ARGUMENT_FIELD(tensorrt_shape_range_info_path,
                      TensorRtShapeRangeInfoPath,
                      std::string);
  DECL_ARGUMENT_FIELD(tensorrt_tuned_dynamic_shape,
                      TensorRtTunedDynamicShape,
                      bool);
  DECL_ARGUMENT_FIELD(tensorrt_allow_build_at_runtime,
                      TensorRtAllowBuildAtRuntime,
                      bool);
  DECL_ARGUMENT_FIELD(tensorrt_use_inspector, TensorRtUseInspector, bool);
  DECL_ARGUMENT_FIELD(tensorrt_inspector_serialize,
                      TensorRtInspectorSerialize,
                      bool);
  DECL_ARGUMENT_FIELD(tensorrt_use_explicit_quantization,
                      TensorRtUseExplicitQuantization,
                      bool);
  DECL_ARGUMENT_FIELD(tensorrt_optimization_level,
                      TensorRtOptimizationLevel,
                      int);
  DECL_ARGUMENT_FIELD(tensorrt_ops_run_float,
                      TensorRtOpsRunFloat,
                      std::unordered_set<std::string>);
#ifdef PADDLE_WITH_OPENVINO
  DECL_ARGUMENT_FIELD(use_openvino, UseOpenVINO, bool);
  DECL_ARGUMENT_FIELD(openvino_inference_precision,
                      OpenvinoInferencePrecision,
                      int);
#endif
  DECL_ARGUMENT_FIELD(use_xpu, UseXpu, bool);
  DECL_ARGUMENT_FIELD(xpu_locked, XpuLocked, bool);
  DECL_ARGUMENT_FIELD(xpu_precision, XpuPrecision, std::string);
  DECL_ARGUMENT_FIELD(xpu_enable_multi_stream, XpuEnableMultiStream, bool);
  // XpuConfig
  DECL_ARGUMENT_FIELD(xpu_device_id, XpuDeviceId, int);
  DECL_ARGUMENT_FIELD(xpu_l3_size, XpuL3Size, size_t);
  DECL_POINTER_ARGUMENT_FIELD(xpu_l3_ptr, XpuL3Ptr, void*);
  DECL_ARGUMENT_FIELD(xpu_l3_autotune_size, XpuL3AutotuneSize, size_t);
  DECL_ARGUMENT_FIELD(xpu_context_gm_size, XpuContextGmSize, int);
  DECL_POINTER_ARGUMENT_FIELD(xpu_context, XpuContext, void*);
  DECL_POINTER_ARGUMENT_FIELD(xpu_stream, XpuStream, void*);
  DECL_ARGUMENT_FIELD(xpu_conv_autotune_level, XpuConvAutotuneLevel, int);
  DECL_ARGUMENT_FIELD(xpu_conv_autotune_file, XpuConvAutotuneFile, std::string);
  DECL_ARGUMENT_FIELD(xpu_conv_autotune_file_writeback,
                      XpuConvAutotuneFileWriteback,
                      bool);
  DECL_ARGUMENT_FIELD(xpu_fc_autotune_level, XpuFcAutotuneLevel, int);
  DECL_ARGUMENT_FIELD(xpu_fc_autotune_file, XpuFcAutotuneFile, std::string);
  DECL_ARGUMENT_FIELD(xpu_fc_autotune_file_writeback,
                      XpuFcAutotuneFileWriteback,
                      bool);
  DECL_ARGUMENT_FIELD(xpu_gemm_compute_precision, XpuGemmComputePrecision, int);
  using quant_post_type = std::map<std::string, int>;
  DECL_ARGUMENT_FIELD(xpu_quant_post_dynamic_weight_methods,
                      XpuQuantPostDynamicWeightMethods,
                      quant_post_type);
  DECL_ARGUMENT_FIELD(xpu_transformer_softmax_optimize_level,
                      XpuTransformerSoftmaxOptimizeLevel,
                      int);
  DECL_ARGUMENT_FIELD(xpu_transformer_encoder_adaptive_seqlen,
                      XpuTransformerEncoderAdaptiveSeqlen,
                      bool);
  DECL_ARGUMENT_FIELD(xpu_quant_post_static_gelu_out_threshold,
                      XpuQuantPostStaticGeluOutThreshold,
                      float);
  DECL_ARGUMENT_FIELD(xpu_quant_post_dynamic_activation_method,
                      XpuQuantPostDynamicActivationMethod,
                      int);
  DECL_ARGUMENT_FIELD(xpu_quant_post_dynamic_weight_precision,
                      XpuQuantPostDynamicWeightPrecision,
                      int);
  DECL_ARGUMENT_FIELD(xpu_quant_post_dynamic_op_types,
                      XpuQuantPostDynamicOpTypes,
                      std::vector<std::string>);

  // Memory optimized related.
  DECL_ARGUMENT_FIELD(enable_memory_optim, EnableMemoryOptim, bool);
  DECL_ARGUMENT_FIELD(trt_engine_memory_sharing, TrtEngineMemorySharing, bool);

  // Indicate which kind of sort algorithm is used for operators, the memory
  // optimization relays on the sort algorithm.
  DECL_ARGUMENT_FIELD(memory_optim_sort_kind, MemoryOptimSortKind, int);

  // The program transformed by IR analysis phase.
  DECL_ARGUMENT_UNIQUE_FIELD(ir_analyzed_program,
                             IrAnalyzedProgram,
                             framework::proto::ProgramDesc);

  DECL_ARGUMENT_FIELD(fusion_statis, FusionStatis, fusion_statis_t);

  // Only used in paddle-lite subgraph.
  DECL_ARGUMENT_FIELD(cpu_math_library_num_threads,
                      CpuMathLibraryNumThreads,
                      int);

  // ipu related
  DECL_ARGUMENT_FIELD(use_ipu, UseIpu, bool);
  DECL_ARGUMENT_FIELD(ipu_device_num, IpuDeviceNum, int);
  DECL_ARGUMENT_FIELD(ipu_micro_batch_size, IpuMicroBatchSize, int);
  DECL_ARGUMENT_FIELD(ipu_enable_pipelining, IpuEnablePipelining, bool);
  DECL_ARGUMENT_FIELD(ipu_batches_per_step, IpuBatchesPerStep, int);
  DECL_ARGUMENT_FIELD(ipu_enable_fp16, IpuEnableFp16, bool);
  DECL_ARGUMENT_FIELD(ipu_replica_num, IpuReplicaNum, int);
  DECL_ARGUMENT_FIELD(ipu_available_memory_proportion,
                      IpuAvailableMemoryProportion,
                      float);
  DECL_ARGUMENT_FIELD(ipu_enable_half_partial, IpuEnableHalfPartial, bool);
  DECL_ARGUMENT_FIELD(ipu_custom_ops_info,
                      IpuCustomOpsInfo,
                      std::vector<std::vector<std::string>>);
  DECL_ARGUMENT_FIELD(ipu_custom_patterns,
                      IpuCustomPatterns,
                      std::vector<std::vector<std::string>>);
  DECL_ARGUMENT_FIELD(ipu_enable_model_runtime_executor,
                      IpuEnableModelRuntimeExecutor,
                      bool);

  // mixed precision related
  DECL_ARGUMENT_FIELD(model_precision, ModelPrecision, int);
  DECL_ARGUMENT_FIELD(mixed_black_list,
                      MixedBlackList,
                      std::unordered_set<std::string>);
  DECL_ARGUMENT_FIELD(mixed_white_list,
                      MixedWhiteList,
                      std::unordered_set<std::string>);
  DECL_ARGUMENT_FIELD(enable_gpu_mixed, EnableGPUMixed, bool);
  DECL_ARGUMENT_FIELD(mixed_precision_mode, MixedPrecisionMode, int);
  DECL_ARGUMENT_FIELD(enable_low_precision_io, EnableLowPrecisionIO, bool);

  // cinn compiler related
  DECL_ARGUMENT_FIELD(use_cinn_compiler, UseCinnCompiler, bool);

  // custom device
  DECL_ARGUMENT_FIELD(use_custom_device, UseCustomDevice, bool);
  DECL_ARGUMENT_FIELD(custom_device_type, CustomDeviceType, std::string);
  DECL_ARGUMENT_FIELD(custom_device_id, CustomDeviceId, int);
  DECL_ARGUMENT_FIELD(enable_custom_device_mixed,
                      EnableCustomDeviceMixed,
                      bool);

 private:
  std::unordered_set<std::string> valid_fields_;
};

#define ARGUMENT_CHECK_FIELD(argument__, fieldname__) \
  PADDLE_ENFORCE_EQ(                                  \
      argument__->Has(#fieldname__),                  \
      true,                                           \
      common::errors::PreconditionNotMet(             \
          "the argument field [%s] should be set", #fieldname__));

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
