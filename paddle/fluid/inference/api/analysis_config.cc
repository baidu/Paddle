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

#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/inference/utils/table_printer.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/utils/string/split.h"

#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
COMMON_DECLARE_uint64(initial_gpu_memory_in_mb);
#endif

#ifdef PADDLE_WITH_CINN
COMMON_DECLARE_bool(use_cinn);
#endif

#ifdef PADDLE_WITH_OPENVINO
#include "oneapi/tbb.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/openvino.hpp"
#endif

COMMON_DECLARE_bool(enable_pir_api);
namespace paddle {

extern const std::vector<std::string> kTRTSubgraphPasses;

AnalysisConfig::AnalysisConfig() {
  // NOTE(liuyuanle): Why put the following code here?
  // ref to https://github.com/PaddlePaddle/Paddle/pull/50864
  inference::InitGflagsFromEnv();
}

PassStrategy *AnalysisConfig::pass_builder() const {
  if (!pass_builder_) {
    if (use_gpu_) {
      LOG(INFO) << "Create GPU IR passes";
      pass_builder_ = std::make_unique<GpuPassStrategy>();
    } else if (use_xpu_) {
      pass_builder_ = std::make_unique<XpuPassStrategy>();
    } else if (use_ipu_) {
      LOG(INFO) << "Create IPU IR passes";
      pass_builder_ = std::make_unique<IpuPassStrategy>();
    } else if (use_custom_device_) {
      LOG(INFO) << "Create CUSTOM DEVICE IR passes";
      pass_builder_ = std::make_unique<CustomDevicePassStrategy>();
    } else {
      LOG(INFO) << "Create CPU IR passes";
      pass_builder_ = std::make_unique<CpuPassStrategy>();
    }
  } else if (pass_builder_->use_gpu() ^ use_gpu()) {
    LOG(WARNING) << "The use_gpu flag is not compatible between Config and "
                    "PassBuilder, the flags are "
                 << use_gpu() << " " << pass_builder_->use_gpu();
    LOG(WARNING) << "Please make them compatible, still use the existing "
                    "PassBuilder.";
  }

  return pass_builder_.get();
}

AnalysisConfig::AnalysisConfig(const std::string &model_dir) {
  model_dir_ = model_dir;

  Update();
}

AnalysisConfig::AnalysisConfig(const std::string &prog_file_or_model_dir,
                               const std::string &params_file_or_model_prefix) {
  if (paddle::inference::IsDirectory(prog_file_or_model_dir)) {
    if (FLAGS_enable_pir_api) {
      prog_file_ =
          prog_file_or_model_dir + "/" + params_file_or_model_prefix + ".json";
    } else {
      prog_file_ = prog_file_or_model_dir + "/" + params_file_or_model_prefix +
                   ".pdmodel";
    }
    params_file_ = prog_file_or_model_dir + "/" + params_file_or_model_prefix +
                   ".pdiparams";
  } else {
    prog_file_ = prog_file_or_model_dir;
    params_file_ = params_file_or_model_prefix;
  }

  PADDLE_ENFORCE_EQ(
      paddle::inference::IsFileExists(prog_file_),
      true,
      common::errors::NotFound(
          "Cannot open file %s, please confirm whether the file is normal.",
          prog_file_));

  Update();
}

void AnalysisConfig::SetModel(
    const std::string &prog_file_path_or_model_dir_path,
    const std::string &params_file_path_or_model_prefix) {
  if (paddle::inference::IsDirectory(prog_file_path_or_model_dir_path)) {
    if (FLAGS_enable_pir_api) {
      prog_file_ = prog_file_path_or_model_dir_path + "/" +
                   params_file_path_or_model_prefix + ".json";
    } else {
      prog_file_ = prog_file_path_or_model_dir_path + "/" +
                   params_file_path_or_model_prefix + ".pdmodel";
    }
    params_file_ = prog_file_path_or_model_dir_path + "/" +
                   params_file_path_or_model_prefix + ".pdiparams";
  } else {
    prog_file_ = prog_file_path_or_model_dir_path;
    params_file_ = params_file_path_or_model_prefix;
  }

  Update();
}

void AnalysisConfig::EnableUseGpu(uint64_t memory_pool_init_size_mb,
                                  int device_id,
                                  Precision precision_mode) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  use_gpu_ = true;
  use_new_executor_ = true;
  memory_pool_init_size_mb_ = memory_pool_init_size_mb;
  FLAGS_initial_gpu_memory_in_mb = memory_pool_init_size_mb_;
  gpu_device_id_ = device_id;
  if (precision_mode == Precision::kFloat32) {
    mixed_precision_mode_ = precision_mode;
  } else if (precision_mode == Precision::kHalf ||
             precision_mode == Precision::kBf16) {
    if (precision_mode == Precision::kBf16) {
      LOG(WARNING) << "Some op (matmul, conv, etc.) run at bfloat16 precision "
                      "requires GPU compute capability >= 80.";
    }
    enable_gpu_mixed_ = true;
    mixed_precision_mode_ = precision_mode;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The GPU inference currently only supports float32/float16/bfloat16 "
        "precision. Please check the parameters you specified in EnableUseGpu "
        "or enable_use_gpu function."));
  }
#else
  LOG(ERROR) << "Please use PaddlePaddle with GPU version.";
  use_gpu_ = false;
#endif

  Update();
}

void AnalysisConfig::Exp_EnableUseCutlass() {
#if defined(PADDLE_WITH_CUTLASS)
  use_cutlass_ = true;
#else
  LOG(ERROR) << "Please compile with cutlass to EnableUseCutlass()";
  use_cutlass_ = false;
#endif

  Update();
}

void AnalysisConfig::SetExecStream(void *stream) {
  PADDLE_ENFORCE_NOT_NULL(
      stream,
      common::errors::InvalidArgument("`stream` should not be nullptr"));
  exec_stream_ = stream;
  use_external_stream_ = true;
  Update();
}

void *AnalysisConfig::GetExecStream() const {
  PADDLE_ENFORCE_NOT_NULL(
      exec_stream_,
      common::errors::InvalidArgument("`stream` should not be nullptr"));
  return exec_stream_;
}

bool AnalysisConfig::external_stream_enabled() const {
  return use_external_stream_;
}

void AnalysisConfig::DisableGpu() {
  use_gpu_ = false;

  Update();
}

void AnalysisConfig::DisableFCPadding() {
  use_fc_padding_ = false;

  Update();
}

void AnalysisConfig::EnableXpu(int l3_size,
                               bool l3_locked,
                               bool conv_autotune,
                               const std::string &conv_autotune_file,
                               const std::string &transformer_encoder_precision,
                               bool transformer_encoder_adaptive_seqlen,
                               bool enable_multi_stream) {
#if defined(PADDLE_WITH_XPU)
  LOG_FIRST_N(WARNING, 1)
      << "Parameters in EnableXpu/enable_xpu is deprecated since version "
         "2.6.1, and will be removed in version 3.0! Please use "
         "EnableXpu/enable_xpu without parameters, and use "
         "SetXpuConfig/set_xpu_config to set options.";
  use_xpu_ = true;
  xpu_config_.l3_size = l3_size;
  xpu_config_.conv_autotune_level = conv_autotune;
  xpu_config_.conv_autotune_file = conv_autotune_file;
  if (transformer_encoder_precision == "int8") {
    xpu_config_.gemm_compute_precision = 0;
  } else if (transformer_encoder_precision == "int16") {
    xpu_config_.gemm_compute_precision = 1;
  } else if (transformer_encoder_precision == "int31") {
    xpu_config_.gemm_compute_precision = 2;
  }
  xpu_config_.transformer_encoder_adaptive_seqlen =
      transformer_encoder_adaptive_seqlen;
  Update();
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "To use XPU inference, please compile with option 'WITH_XPU' or "
      "'LITE_WITH_XPU' first."));
#endif
}

void AnalysisConfig::SetXpuDeviceId(int device_id) {
  PADDLE_ENFORCE_EQ(use_xpu_,
                    true,
                    common::errors::PreconditionNotMet(
                        "Should call EnableXpu before SetXpuDeviceId."));
  xpu_config_.device_id = device_id;
  Update();
}

void AnalysisConfig::SetXpuConfig(const XpuConfig &config) {
  PADDLE_ENFORCE(use_xpu_,
                 common::errors::PreconditionNotMet(
                     "Should call EnableXpu before SetXpuConfig."));
  PADDLE_ENFORCE_LE(
      config.l3_autotune_size,
      config.l3_size,
      common::errors::InvalidArgument(
          "l3_autotune_size(%zu) should be less than or equal to l3_size(%zu).",
          config.l3_autotune_size,
          config.l3_size));
  xpu_config_ = config;
  Update();
}

void AnalysisConfig::EnableCustomDevice(const std::string &device_type,
                                        int device_id,
                                        Precision precision_mode) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  use_custom_device_ = true;
  custom_device_id_ = device_id;
  custom_device_type_ = device_type;
  mixed_precision_mode_ = precision_mode;
  if (precision_mode == Precision::kFloat32) {
    // default
  } else if (precision_mode == Precision::kHalf ||
             precision_mode == Precision::kBf16) {
    enable_custom_device_mixed_ = true;
    LOG(INFO) << "enable_custom_device_mixed_";
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The Paddle-CustomDevice inference currently only supports "
        "float32/float16/bfloat16 precision. Please check the parameters "
        "you specified in EnableCustomDevice function."));
  }
#else
  LOG(ERROR) << "Please compile with CustomDevice to EnableCustomDevice()";
  use_custom_device_ = false;
#endif
  Update();
}

void AnalysisConfig::EnableIpu(int ipu_device_num,
                               int ipu_micro_batch_size,
                               bool ipu_enable_pipelining,
                               int ipu_batches_per_step) {
  enable_ir_optim_ = true;

  use_ipu_ = true;
  ipu_device_num_ = ipu_device_num;
  ipu_micro_batch_size_ = ipu_micro_batch_size;
  ipu_enable_pipelining_ = ipu_enable_pipelining;
  ipu_batches_per_step_ = ipu_batches_per_step;

  Update();
}

void AnalysisConfig::SetIpuConfig(bool ipu_enable_fp16,
                                  int ipu_replica_num,
                                  float ipu_available_memory_proportion,
                                  bool ipu_enable_half_partial,
                                  bool ipu_enable_model_runtime_executor) {
  ipu_enable_fp16_ = ipu_enable_fp16;
  ipu_replica_num_ = ipu_replica_num;
  ipu_available_memory_proportion_ = ipu_available_memory_proportion;
  ipu_enable_half_partial_ = ipu_enable_half_partial;
  ipu_enable_model_runtime_executor_ = ipu_enable_model_runtime_executor;

  Update();
}

void AnalysisConfig::SetIpuCustomInfo(
    const std::vector<std::vector<std::string>> &ipu_custom_ops_info,
    const std::map<std::string, bool> &ipu_custom_patterns) {
  ipu_custom_ops_info_ = ipu_custom_ops_info;
  for (const auto &ipu_custom_pattern : ipu_custom_patterns) {
    if (ipu_custom_pattern.second == true) {
      ipu_custom_patterns_.push_back(
          std::vector<std::string>{ipu_custom_pattern.first, "True"});
    } else if (ipu_custom_pattern.second == false) {
      ipu_custom_patterns_.push_back(
          std::vector<std::string>{ipu_custom_pattern.first, "False"});
    }
  }

  Update();
}

void AnalysisConfig::LoadIpuConfig(const std::string &config_path) {
  std::ifstream fin(config_path, std::ios::in);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin.is_open()),
      true,
      common::errors::NotFound(
          "Cannot open file %s, please confirm whether the file is normal.",
          config_path));
  std::string line;
  while (std::getline(fin, line)) {
    // remove all space
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

    std::string key;
    std::string value;
    std::istringstream stream(line);
    // Split string to key and value based on the first `,`
    std::getline(stream, key, ',');
    std::getline(stream, value);

    auto string2bool = [](std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return ::tolower(c);
      });
      return s == "true" || s == "1";
    };

    // ipu_custom_ops_info:
    // [[paddle_op_name, popart_op_name, domain, version], [paddle_op_name,
    // popart_op_name, domain, version]...]
    // ipu_custom_patterns:
    // [[paddle_op_name, enable_pattern], [paddle_op_name, enable_pattern]...]
    auto string2vector = [](std::string s) {
      std::vector<std::vector<std::string>> custom_info;
      s.erase(0, 1);
      s.pop_back();

      std::string one;
      std::istringstream s_stream(s);
      while (std::getline(s_stream, one, ']')) {
        if (!one.empty()) {
          // remove `[`
          one.erase(0, 1);
          custom_info.push_back(paddle::string::Split(one, ','));
        }
      }
      return custom_info;
    };

    if (ipu_config_mapper_.find(key) == ipu_config_mapper_.end()) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "invalid key %s in IPU config: ", key));
    }
    switch (ipu_config_mapper_.at(key)) {
      case ipu_config_code::ipu_device_num:
        ipu_device_num_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_micro_batch_size:
        ipu_micro_batch_size_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_enable_pipelining:
        ipu_enable_pipelining_ = string2bool(value);
        break;
      case ipu_config_code::ipu_batches_per_step:
        ipu_batches_per_step_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_enable_fp16:
        ipu_enable_fp16_ = string2bool(value);
        break;
      case ipu_config_code::ipu_replica_num:
        ipu_replica_num_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_available_memory_proportion:
        ipu_available_memory_proportion_ = std::stof(value);
        break;
      case ipu_config_code::ipu_enable_half_partial:
        ipu_enable_half_partial_ = string2bool(value);
        break;
      case ipu_config_code::ipu_custom_ops_info:
        ipu_custom_ops_info_ = string2vector(value);
        break;
      case ipu_config_code::ipu_custom_patterns:
        ipu_custom_patterns_ = string2vector(value);
        break;
      case ipu_config_code::ipu_enable_model_runtime_executor:
        ipu_enable_model_runtime_executor_ = string2bool(value);
        break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "invalid key %s in IPU config", key));
        break;
    }
  }

  Update();
}

void AnalysisConfig::EnableONNXRuntime() {
#ifdef PADDLE_WITH_ONNXRUNTIME
  use_onnxruntime_ = true;
#else
  LOG(ERROR) << "Please compile with onnxruntime to EnableONNXRuntime()";
  use_onnxruntime_ = false;
#endif

  Update();
}

void AnalysisConfig::DisableONNXRuntime() {
  use_onnxruntime_ = false;
  Update();
}

void AnalysisConfig::EnableORTOptimization() {
#ifdef PADDLE_WITH_ONNXRUNTIME
  enable_ort_optimization_ = true;
#else
  LOG(ERROR) << "Please compile with onnxruntime to EnableORTOptimization()";
  enable_ort_optimization_ = false;
#endif

  Update();
}

AnalysisConfig::AnalysisConfig(const AnalysisConfig &other) {
#define CP_MEMBER(member__) member__ = other.member__;

  // Model related.
  CP_MEMBER(model_dir_);
  CP_MEMBER(model_from_memory_);  // the memory model reuses prog_file_ and
                                  // params_file_ fields.
  CP_MEMBER(save_optimized_model_);
  CP_MEMBER(opt_cache_dir_);
  CP_MEMBER(prog_file_);
  CP_MEMBER(params_file_);

  CP_MEMBER(use_fc_padding_);
  // GPU related.
  CP_MEMBER(use_gpu_);
  CP_MEMBER(use_cutlass_);
  CP_MEMBER(use_external_stream_);
  CP_MEMBER(exec_stream_);
  CP_MEMBER(use_cudnn_);
  CP_MEMBER(gpu_device_id_);
  CP_MEMBER(memory_pool_init_size_mb_);

  // Mixed precision related.
  CP_MEMBER(mixed_black_list_);
  CP_MEMBER(mixed_white_list_);
  CP_MEMBER(enable_gpu_mixed_);
  CP_MEMBER(mixed_precision_mode_);
  CP_MEMBER(enable_low_precision_io_);

  CP_MEMBER(enable_memory_optim_);
#ifdef PADDLE_WITH_OPENVINO
  // Openvino related.
  CP_MEMBER(use_openvino_);
  CP_MEMBER(openvino_inference_precision_);
#endif
  // TensorRT related.
  CP_MEMBER(use_tensorrt_);
  CP_MEMBER(tensorrt_workspace_size_);
  CP_MEMBER(tensorrt_max_batchsize_);
  CP_MEMBER(tensorrt_min_subgraph_size_);
  CP_MEMBER(tensorrt_precision_mode_);
  CP_MEMBER(trt_mark_output_);
  CP_MEMBER(trt_parameters_run_fp16_);
  CP_MEMBER(trt_parameters_run_int8_);
  CP_MEMBER(trt_parameters_run_bfp16_);
  CP_MEMBER(trt_forbid_dynamic_op_);
  CP_MEMBER(refit_params_path_);
  CP_MEMBER(trt_output_tensor_names_);
  CP_MEMBER(trt_disabled_ops_);
  CP_MEMBER(trt_use_dla_);
  CP_MEMBER(trt_dla_core_);
  CP_MEMBER(trt_use_static_engine_);
  CP_MEMBER(trt_use_calib_mode_);
  CP_MEMBER(trt_use_cuda_graph_);
  CP_MEMBER(trt_use_varseqlen_);
  CP_MEMBER(trt_with_interleaved_);
  CP_MEMBER(tensorrt_transformer_posid_);
  CP_MEMBER(tensorrt_transformer_maskid_);
  CP_MEMBER(trt_tuned_dynamic_shape_);
  CP_MEMBER(trt_allow_build_at_runtime_);
  CP_MEMBER(collect_shape_range_info_);
  CP_MEMBER(shape_range_info_path_);
  CP_MEMBER(trt_use_inspector_);
  CP_MEMBER(trt_inspector_serialize_);
  CP_MEMBER(trt_use_explicit_quantization_);
  CP_MEMBER(trt_engine_memory_sharing_);
  CP_MEMBER(trt_engine_memory_sharing_identifier_);
  CP_MEMBER(trt_optimization_level_);
  CP_MEMBER(trt_ops_run_float_);
  CP_MEMBER(trt_exclude_var_names_);
  // OneDNN related.
  CP_MEMBER(use_mkldnn_);
  CP_MEMBER(mkldnn_enabled_op_types_);
  CP_MEMBER(mkldnn_cache_capacity_);
  // Bfloat16 related.
  CP_MEMBER(use_mkldnn_bfloat16_);
  CP_MEMBER(bfloat16_enabled_op_types_);
  // Quantization related.
  CP_MEMBER(use_mkldnn_int8_);
  CP_MEMBER(quantize_enabled_op_types_);
  CP_MEMBER(quantize_excluded_op_ids_);
  CP_MEMBER(min_input_shape_);
  CP_MEMBER(max_input_shape_);
  CP_MEMBER(optim_input_shape_);
  CP_MEMBER(disable_trt_plugin_fp16_);

  // XPU related.
  CP_MEMBER(use_xpu_);
  CP_MEMBER(xpu_config_);

  // profile related.
  CP_MEMBER(with_profile_);

  // cinn compiler related.
  CP_MEMBER(use_cinn_);

  // glog related.
  CP_MEMBER(with_glog_info_);

  // Ir related.
  CP_MEMBER(enable_ir_optim_);
  CP_MEMBER(ir_debug_);
  CP_MEMBER(specify_input_name_);

  CP_MEMBER(use_optimized_model_);

  CP_MEMBER(cpu_math_library_num_threads_);

  CP_MEMBER(serialized_info_cache_);

  CP_MEMBER(thread_local_stream_);

  // ipu related
  CP_MEMBER(use_ipu_);
  CP_MEMBER(ipu_device_num_);
  CP_MEMBER(ipu_micro_batch_size_);
  CP_MEMBER(ipu_enable_pipelining_);
  CP_MEMBER(ipu_batches_per_step_);
  CP_MEMBER(ipu_enable_fp16_);
  CP_MEMBER(ipu_replica_num_);
  CP_MEMBER(ipu_available_memory_proportion_);
  CP_MEMBER(ipu_enable_half_partial_);
  CP_MEMBER(ipu_enable_model_runtime_executor_);
  CP_MEMBER(ipu_custom_ops_info_);
  CP_MEMBER(ipu_custom_patterns_);

  // custom device related.
  CP_MEMBER(use_custom_device_);
  CP_MEMBER(custom_device_type_);
  CP_MEMBER(custom_device_id_);
  CP_MEMBER(enable_custom_device_mixed_);

  // JITLayer relate
  CP_MEMBER(apply_optim_);
  CP_MEMBER(skip_load_params_);

  CP_MEMBER(use_new_executor_);
  CP_MEMBER(use_pir_);
  CP_MEMBER(custom_passes_);
  CP_MEMBER(custom_pass_only_);
  CP_MEMBER(pm_opt_level_);
  CP_MEMBER(ir_debug_passes_);
  CP_MEMBER(deleted_passes_);

  if (use_gpu_) {
    PADDLE_ENFORCE_EQ(use_xpu_,
                      false,
                      common::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    pass_builder_ = std::make_unique<GpuPassStrategy>(
        *static_cast<GpuPassStrategy *>(other.pass_builder()));
  } else if (use_ipu_) {
    pass_builder_ = std::make_unique<IpuPassStrategy>(
        *static_cast<IpuPassStrategy *>(other.pass_builder()));
  } else if (use_xpu_) {
    pass_builder_ = std::make_unique<XpuPassStrategy>(
        *static_cast<XpuPassStrategy *>(other.pass_builder()));
  } else if (use_custom_device_) {
    pass_builder_ = std::make_unique<CustomDevicePassStrategy>(
        *static_cast<CustomDevicePassStrategy *>(other.pass_builder()));
  } else {
    pass_builder_ = std::make_unique<CpuPassStrategy>(
        *static_cast<CpuPassStrategy *>(other.pass_builder()));
  }

#undef CP_MEMBER

  Update();
  if (use_tensorrt_ || use_cinn_) {
    // Update() will reset all the passes, when some tensorRT pass is deleted in
    // other.pass_builder(), it will set again, so we just remove the
    // deleted_pass.
    pass_builder_->ClearPasses();
    auto other_passes = other.pass_builder()->AllPasses();
    for (auto const &pass : other_passes) {
      pass_builder_->AppendPass(pass);
    }
  }

  for (auto &delete_pass : other.pass_builder()->GetAllDeletedPasses()) {
    pass_builder_->DeletePass(delete_pass);
  }
}

void AnalysisConfig::EnableCUDNN() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  use_cudnn_ = use_gpu_;
#else
  LOG(ERROR) << "Please compile with CUDA first to use cuDNN";
  use_cudnn_ = false;
#endif

  Update();
}

void AnalysisConfig::EnableMKLDNN() {
#ifdef PADDLE_WITH_DNNL
  use_mkldnn_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
  use_mkldnn_ = false;
#endif

  Update();
}

void AnalysisConfig::DisableMKLDNN() {
  use_mkldnn_ = false;
  Update();
}

void AnalysisConfig::SetMkldnnCacheCapacity(int capacity) {
#ifdef PADDLE_WITH_DNNL
  mkldnn_cache_capacity_ = capacity;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to set MKLDNN Thread Id";
  mkldnn_cache_capacity_ = 0;
#endif
}

void AnalysisConfig::EnableMkldnnBfloat16() {
#ifdef PADDLE_WITH_DNNL
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512_core)) {
    use_mkldnn_bfloat16_ = true;
    LOG(INFO) << "Hardware support for BFLOAT16"
              << (phi::backends::cpu::MayIUse(
                      phi::backends::cpu::cpu_isa_t::avx512_bf16)
                      ? " is enabled"
                      : " is disabled. Simulation will be used");
  } else {
    LOG(INFO) << "CPU does not support BFLOAT16 calculations";
    use_mkldnn_bfloat16_ = false;
  }
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnBfloat16";
  use_mkldnn_bfloat16_ = false;
#endif

  Update();
}

void AnalysisConfig::DisableMkldnnFcPasses() {
#ifdef PADDLE_WITH_DNNL
  disable_mkldnn_fc_passes_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use DisableMkldnnFcPasses";
  disable_mkldnn_fc_passes_ = false;
#endif
  Update();
}

void AnalysisConfig::EnableMkldnnInt8(
    const std::unordered_set<std::string> &op_list) {
#ifdef PADDLE_WITH_DNNL
  use_mkldnn_int8_ = true;
  use_fc_padding_ = false;
  if (!op_list.empty())
    quantize_enabled_op_types_.insert(op_list.begin(), op_list.end());
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnInt8";
  use_mkldnn_int8_ = false;
#endif

  Update();
}

void AnalysisConfig::EnableOpenVINOEngine(Precision inference_precision) {
#ifdef PADDLE_WITH_OPENVINO
  use_openvino_ = true;
  openvino_inference_precision_ = inference_precision;
  Update();
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "To use Paddle-OpenVINO, please compile with OpenVINO first."));
#endif
}

bool AnalysisConfig::openvino_engine_enabled() const {
#ifdef PADDLE_WITH_OPENVINO
  return use_openvino_;
#else
  return false;
#endif
}

void AnalysisConfig::EnableTensorRtEngine(int64_t workspace_size,
                                          int max_batch_size,
                                          int min_subgraph_size,
                                          Precision precision_mode,
                                          bool use_static,
                                          bool use_calib_mode,
                                          bool use_cuda_graph) {
#ifdef PADDLE_WITH_TENSORRT
  if (!use_gpu()) {
    LOG(ERROR) << "To use TensorRT engine, please call EnableUseGpu() first";
    return;
  }

  use_tensorrt_ = true;
  tensorrt_workspace_size_ = workspace_size;
  tensorrt_max_batchsize_ = max_batch_size;
  tensorrt_min_subgraph_size_ = min_subgraph_size;
  tensorrt_precision_mode_ = precision_mode;
  trt_use_static_engine_ = use_static;
  trt_use_calib_mode_ = use_calib_mode;
  trt_use_cuda_graph_ = use_cuda_graph;
  if (use_cuda_graph) {
    LOG_FIRST_N(INFO, 1) << "You have enabled Trt Cuda Graph, you must ensure "
                            "that the input Shape remains unchanged.";
  }

  Update();
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "To use Paddle-TensorRT, please compile with TENSORRT first."));
#endif
}

void AnalysisConfig::MarkTrtEngineOutputs(
    const std::vector<std::string> &output_tensor_names) {
  trt_mark_output_ = true;
  trt_output_tensor_names_ = output_tensor_names;
}

void AnalysisConfig::Exp_DisableTensorRTDynamicShapeOPs(
    bool trt_forbid_dynamic_op) {
  trt_forbid_dynamic_op_ = trt_forbid_dynamic_op;
}

void AnalysisConfig::TensorRTRefitParamsPath(
    const std::string &refit_params_path) {
  refit_params_path_ = refit_params_path;
}

void AnalysisConfig::EnableTensorRTMemoryOptim(bool engine_memory_sharing,
                                               int sharing_identifier) {
  PADDLE_ENFORCE_EQ(
      use_tensorrt_,
      true,
      common::errors::InvalidArgument(
          "To enable TensorRT memory optim, please call "
          "EnableTensorRtEngine or enable_tensorrt_engine first."));
  PADDLE_ENFORCE_GE(sharing_identifier,
                    0,
                    common::errors::InvalidArgument(
                        "The value of sharing_identifier must be greater "
                        "than or equal to 0."));
  if (!engine_memory_sharing) {
    PADDLE_ENFORCE_EQ(sharing_identifier,
                      0,
                      common::errors::InvalidArgument(
                          "The value of sharing_identifier must be equal to 0 "
                          "when engine_memory_sharing is false."));
  }
  trt_engine_memory_sharing_ = engine_memory_sharing;
  trt_engine_memory_sharing_identifier_ = sharing_identifier;
}

void AnalysisConfig::EnableLowPrecisionIO(bool x) {
  PADDLE_ENFORCE_EQ(
      enable_gpu_mixed_ || !x,
      true,
      common::errors::InvalidArgument(
          "To enable low precision io, please call EnableUseGPU() to specify "
          "precision mode as low precision."));
  enable_low_precision_io_ = x;
}

void AnalysisConfig::SetTRTDynamicShapeInfo(
    std::map<std::string, std::vector<int>> min_input_shape,
    std::map<std::string, std::vector<int>> max_input_shape,
    std::map<std::string, std::vector<int>> optim_input_shape,
    bool disable_trt_plugin_fp16) {
  min_input_shape_ = min_input_shape;
  max_input_shape_ = max_input_shape;
  optim_input_shape_ = optim_input_shape;
  disable_trt_plugin_fp16_ = disable_trt_plugin_fp16;
}

void AnalysisConfig::EnableTensorRtDLA(int dla_core) {
  trt_use_dla_ = true;
  trt_dla_core_ = dla_core;
}

void AnalysisConfig::EnableTensorRtInspector(bool inspector_serialize) {
  trt_use_inspector_ = true;
  trt_inspector_serialize_ = inspector_serialize;
}

void AnalysisConfig::EnableTensorRtExplicitQuantization() {
  trt_use_explicit_quantization_ = true;
  Update();
}

void AnalysisConfig::Exp_DisableTensorRtOPs(
    const std::vector<std::string> &ops) {
  trt_disabled_ops_.insert(trt_disabled_ops_.end(), ops.begin(), ops.end());
}

void AnalysisConfig::Exp_DisableTensorRtSubgraph(
    const std::vector<std::string> &var_name_not_trt) {
  trt_exclude_var_names_.insert(trt_exclude_var_names_.end(),
                                var_name_not_trt.begin(),
                                var_name_not_trt.end());
}

void AnalysisConfig::Exp_SpecifyTensorRTSubgraphPrecision(
    const std::vector<std::string> &trt_parameters_run_fp16,
    const std::vector<std::string> &trt_parameters_run_int8,
    const std::vector<std::string> &trt_parameters_run_bfp16) {
  trt_parameters_run_fp16_.insert(trt_parameters_run_fp16_.end(),
                                  trt_parameters_run_fp16.begin(),
                                  trt_parameters_run_fp16.end());
  trt_parameters_run_int8_.insert(trt_parameters_run_int8_.end(),
                                  trt_parameters_run_int8.begin(),
                                  trt_parameters_run_int8.end());
  trt_parameters_run_bfp16_.insert(trt_parameters_run_bfp16_.end(),
                                   trt_parameters_run_bfp16.begin(),
                                   trt_parameters_run_bfp16.end());
}

void AnalysisConfig::EnableVarseqlen() { trt_use_varseqlen_ = true; }

void AnalysisConfig::SetTensorRtOptimizationLevel(int level) {
  PADDLE_ENFORCE(
      level >= 0 && level <= 5,
      common::errors::InvalidArgument(
          "The input level in SetTRTOptimizationLevel is invalid. The "
          "level must be in range [0, 5], but received level = %d (default "
          "level is 3).",
          level));
  trt_optimization_level_ = level;
}

// TODO(Superjomn) refactor this, buggy.
void AnalysisConfig::Update() {
  auto &&info = SerializeInfoCache();
  if (info == serialized_info_cache_) return;

  std::unordered_set<std::string> deleted_passes;
  if (pass_builder_) {
    deleted_passes = pass_builder_->GetAllDeletedPasses();
  }

  // Transfer pass_builder and copy the existing compatible passes.
  if (!pass_builder_ || ((use_gpu() ^ pass_builder_->use_gpu())) ||
      ((use_xpu() ^ pass_builder_->use_xpu())) ||
      ((use_ipu() ^ pass_builder_->use_ipu())) ||
      ((use_custom_device() ^ pass_builder_->use_custom_device()))) {
    if (use_gpu()) {
      pass_builder_ = std::make_unique<GpuPassStrategy>();
    } else if (use_ipu()) {
      pass_builder_ = std::make_unique<IpuPassStrategy>();
    } else if (use_xpu()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          common::errors::InvalidArgument(
              "Only one choice can be made between CPU and XPU."));
      pass_builder_ = std::make_unique<XpuPassStrategy>();
    } else if (use_custom_device()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          common::errors::InvalidArgument(
              "Only one choice can be made between GPU and CustomDevice."));
      pass_builder_ = std::make_unique<CustomDevicePassStrategy>();
    } else {
      pass_builder_ = std::make_unique<CpuPassStrategy>();
    }

  } else {
    if (use_gpu()) {
      pass_builder_ = std::make_unique<GpuPassStrategy>(
          *static_cast<GpuPassStrategy *>(pass_builder_.get()));
    } else if (use_ipu()) {
      VLOG(1) << "IpuPassStrategy has been used.";
      pass_builder_ = std::make_unique<IpuPassStrategy>(
          *static_cast<IpuPassStrategy *>(pass_builder_.get()));
    } else if (use_xpu()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          common::errors::InvalidArgument(
              "Only one choice can be made between CPU and XPU."));
      pass_builder_ = std::make_unique<XpuPassStrategy>(
          *static_cast<XpuPassStrategy *>(pass_builder_.get()));
    } else if (use_custom_device()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          common::errors::InvalidArgument(
              "Only one choice can be made between GPU and CustomDevice."));
      pass_builder_ = std::make_unique<CustomDevicePassStrategy>(
          *static_cast<CustomDevicePassStrategy *>(pass_builder_.get()));
    } else {
      pass_builder_ = std::make_unique<CpuPassStrategy>(
          *static_cast<CpuPassStrategy *>(pass_builder_.get()));
    }
  }

#ifdef PADDLE_WITH_DNNL
  // Since EnableMKLDNN is default, the pass_builder has created in the first
  // time.
  // Case1: User manually disable onednn after pass_builder
  // create.(config.disable_mkldnn())
  // Case2: User device is gpu/ipu/xpu, use
  // EnableXpu(), EnableCUDNN(), PassStrategy has been reset in the above code
  // block
  //  Case3: pass_builder_ has been created and belongs to
  // GpuPassStrategy(or IpuPassStrategy), neither enable onednn and
  // disable onednn will be executed
  if ((!use_gpu() && !use_xpu() && !use_ipu() && !use_mkldnn_) ||
      (use_mkldnn_ &&
       !phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx2))) {
    // User manually disable onednn or disable when not support AVX2
    use_mkldnn_ = false;
    pass_builder()->DisableMKLDNN();
  }
#endif
#ifdef PADDLE_WITH_OPENVINO
  if (use_openvino_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kOVSubgraphPasses) {
      pass_builder()->AppendPass(pass);
    }
  }
#endif
  if (use_tensorrt_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kTRTSubgraphPasses) {
      if (tensorrt_precision_mode_ == Precision::kInt8 &&
          (pass == "conv_bn_fuse_pass")) {
        continue;
      }
      // The following two IR pass will remove QDQ nodes. For explicit
      // quantization, they are unnecessary.
      if (trt_use_explicit_quantization_ &&
          (pass == "trt_delete_weight_dequant_linear_op_pass" ||
           pass == "delete_quant_dequant_linear_op_pass")) {
        continue;
      }
      pass_builder()->AppendPass(pass);
    }
  }

  // TODO(wilber): An ugly method to update pass, need to be fixed.
  if (use_cinn_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kCINNCompilerPasses) {
      pass_builder()->AppendPass(pass);
    }
  }

  if (use_gpu() && use_cudnn_) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableCUDNN() only works when IR optimization is enabled.";
    } else {
      pass_builder()->EnableCUDNN();
    }
#endif
  }

  if (!use_gpu() && !use_xpu() && !use_ipu()) {
    if (use_mkldnn_ && enable_ir_optim_) {
#ifdef PADDLE_WITH_DNNL
      // default enable onednn when device is cpu and enable_ir_optim
      pass_builder()->EnableMKLDNN();
#endif
    }
  }

  if (use_mkldnn_bfloat16_) {
#ifdef PADDLE_WITH_DNNL
    pass_builder()->EnableMkldnnBfloat16();
#endif
  }

  if (use_mkldnn_int8_) {
#ifdef PADDLE_WITH_DNNL
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableMkldnnInt8() only works when IR optimization "
                    "is enabled.";
    } else if (!use_mkldnn_) {
      LOG(ERROR) << "EnableMkldnnInt8() only works when MKLDNN "
                    "is enabled.";
    } else {
      pass_builder()->EnableMkldnnInt8();
    }
#endif
  }

  if (disable_mkldnn_fc_passes_) {
#ifdef PADDLE_WITH_DNNL
    pass_builder()->DisableMkldnnFcPasses();
#endif
  }

  if (enable_memory_optim_) {
    pass_builder()->AppendAnalysisPass("memory_optimize_pass");
  }

  if (use_xpu_) {
#if (defined PADDLE_WITH_XPU)
    PADDLE_ENFORCE_EQ(use_gpu_,
                      false,
                      common::errors::Unavailable(
                          "Currently, XPU and GPU cannot be enabled in the "
                          "same analysis configuration."));
#else
    PADDLE_THROW(common::errors::Unavailable(
        "You tried to use an XPU device, but Paddle was not compiled "
        "with XPU-runtime."));
#endif
  }
  if (use_ipu_) {
#ifndef PADDLE_WITH_IPU
    PADDLE_THROW(common::errors::Unavailable(
        "You tried to enable the ipu "
        "but did not have the option -DWITH_IPU compiled."));
#endif
  }
  if (use_custom_device_) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
    PADDLE_THROW(common::errors::Unavailable(
        "You tried to enable the custom device "
        "but did not have the option -DWITH_CUSTOM_DEVICE compiled."));
#endif
  }
  for (const auto &delete_pass : deleted_passes) {
    pass_builder_->DeletePass(delete_pass);
  }
}

std::string AnalysisConfig::SerializeInfoCache() {
  std::stringstream ss;
  ss << model_dir_;
  ss << prog_file_;
  ss << params_file_;
  ss << save_optimized_model_;

  ss << use_gpu_;
  ss << enable_gpu_mixed_;
  ss << use_external_stream_;
  ss << exec_stream_;
  ss << use_fc_padding_;
  ss << gpu_device_id_;
  ss << memory_pool_init_size_mb_;

  ss << use_tensorrt_;
  ss << tensorrt_workspace_size_;
  ss << tensorrt_max_batchsize_;
  ss << tensorrt_min_subgraph_size_;
  ss << trt_mark_output_;
  for (auto &name : trt_parameters_run_fp16_) ss << name.c_str();
  ss << ";";
  for (auto &name : trt_parameters_run_int8_) ss << name.c_str();
  ss << ";";
  for (auto &name : trt_parameters_run_bfp16_) ss << name.c_str();
  ss << ";";
  ss << trt_forbid_dynamic_op_;
  ss << refit_params_path_;

  for (auto &op : trt_disabled_ops_) ss << op.c_str();
  ss << ";";

  for (auto &name : trt_exclude_var_names_) ss << name.c_str();
  ss << ";";

  ss << trt_use_dla_;
  ss << trt_dla_core_;

  ss << enable_memory_optim_;
  ss << trt_engine_memory_sharing_;

  ss << use_mkldnn_;
  ss << mkldnn_cache_capacity_;
  for (auto &item : mkldnn_enabled_op_types_) ss << item;
  ss << ";";

  ss << use_mkldnn_bfloat16_;
  for (auto &item : bfloat16_enabled_op_types_) ss << item;
  ss << use_mkldnn_int8_;
  for (auto &item : quantize_enabled_op_types_) ss << item;
  for (auto &item : quantize_excluded_op_ids_) ss << item;
  ss << ";";
  ss << model_from_memory_;

  ss << with_profile_;

  ss << with_glog_info_;

  ss << enable_ir_optim_;
  ss << ir_debug_;

  ss << use_optimized_model_;

  ss << specify_input_name_;
  ss << cpu_math_library_num_threads_;

  ss << use_xpu_;
  ss << xpu_config_.device_id;
  ss << xpu_config_.l3_size;
  ss << xpu_config_.l3_ptr;
  ss << xpu_config_.l3_autotune_size;
  ss << xpu_config_.context_gm_size;
  ss << xpu_config_.context;
  ss << xpu_config_.stream;
  ss << xpu_config_.conv_autotune_level;
  ss << xpu_config_.conv_autotune_file;
  ss << xpu_config_.conv_autotune_file_writeback;
  ss << xpu_config_.fc_autotune_level;
  ss << xpu_config_.fc_autotune_file;
  ss << xpu_config_.fc_autotune_file_writeback;
  ss << xpu_config_.gemm_compute_precision;
  ss << xpu_config_.transformer_softmax_optimize_level;
  ss << xpu_config_.transformer_encoder_adaptive_seqlen;
  ss << xpu_config_.quant_post_static_gelu_out_threshold;
  ss << xpu_config_.quant_post_dynamic_activation_method;
  ss << xpu_config_.quant_post_dynamic_weight_precision;
  for (auto const &type : xpu_config_.quant_post_dynamic_op_types) ss << type;

  ss << thread_local_stream_;

  ss << use_ipu_;
  ss << ipu_device_num_;
  ss << ipu_micro_batch_size_;
  ss << ipu_enable_pipelining_;
  ss << ipu_batches_per_step_;
  ss << ipu_enable_fp16_;
  ss << ipu_replica_num_;
  ss << ipu_available_memory_proportion_;
  ss << ipu_enable_half_partial_;
  ss << ipu_enable_model_runtime_executor_;
  for (auto const &custom_op : ipu_custom_ops_info_)
    for (auto const &attr : custom_op) ss << attr;
  ss << ";";
  for (auto const &pattern : ipu_custom_patterns_)
    for (auto const &attr : pattern) ss << attr;
  ss << ";";
  for (auto &op : mixed_black_list_) ss << op.c_str();
  for (auto &op : mixed_white_list_) ss << op.c_str();
  return ss.str();
}

void AnalysisConfig::SetCpuMathLibraryNumThreads(
    int cpu_math_library_num_threads) {
  cpu_math_library_num_threads_ = cpu_math_library_num_threads;

  Update();
}

float AnalysisConfig::fraction_of_gpu_memory_for_pool() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Get the GPU memory details and calculate the fraction of memory for the
  // GPU memory pool.
  size_t gpu_total, gpu_available;
  platform::SetDeviceId(gpu_device_id_);
  platform::GpuMemoryUsage(&gpu_available, &gpu_total);
  double total_gpu_memory = static_cast<double>(gpu_total) / 1024. / 1024.;
  float fraction_of_gpu_memory =
      static_cast<float>(memory_pool_init_size_mb()) /
      static_cast<float>(total_gpu_memory);
  VLOG(3) << "total_gpu_memory is " << total_gpu_memory
          << "M, gpu_available is "
          << static_cast<double>(gpu_available) / 1024. / 1024.
          << "M, memory_pool_init_size is " << memory_pool_init_size_mb()
          << "M.";
  return fraction_of_gpu_memory;
#else
  return 0.;
#endif
}

void AnalysisConfig::EnableMemoryOptim(bool x) {
  enable_memory_optim_ = x;
  Update();
}

bool AnalysisConfig::enable_memory_optim() const {
  return enable_memory_optim_;
}

bool AnalysisConfig::trt_engine_memory_sharing() const {
  return trt_engine_memory_sharing_;
}

void AnalysisConfig::SetModelBuffer(const char *prog_buffer,
                                    size_t prog_buffer_size,
                                    const char *param_buffer,
                                    size_t param_buffer_size) {
  prog_file_ = std::string(prog_buffer, prog_buffer + prog_buffer_size);
  params_file_ = std::string(param_buffer, param_buffer + param_buffer_size);
  model_from_memory_ = true;
}

NativeConfig AnalysisConfig::ToNativeConfig() const {
  NativeConfig config;
  config.model_dir = model_dir_;
  config.prog_file = prog_file_;
  config.param_file = params_file_;
  config.use_gpu = use_gpu_;
  config.device = gpu_device_id_;
  config.fraction_of_gpu_memory = fraction_of_gpu_memory_for_pool();
  config.specify_input_name = specify_input_name_;
  return config;
}

void AnalysisConfig::SwitchIrDebug(int x,
                                   const std::vector<std::string> &passes) {
  ir_debug_ = x;
  ir_debug_passes_ = passes;
  Update();
}

void AnalysisConfig::EnableProfile() {
  with_profile_ = true;
  Update();
}

void AnalysisConfig::DisableGlogInfo() {
  with_glog_info_ = false;
  Update();
}

void AnalysisConfig::EnableGpuMultiStream() { thread_local_stream_ = true; }

std::string AnalysisConfig::Summary() {
  const std::vector<std::string> header{"Option", "Value"};
  paddle::inference::TablePrinter os(header);

  if (!model_dir_.empty()) {
    os.InsertRow({"model_dir", model_dir_});
  }
  if (!(prog_file_.empty() && params_file_.empty())) {
    os.InsertRow({"model_file", prog_file_});
    os.InsertRow({"params_file", params_file_});
  }

  if (model_from_memory_) {
    os.InsertRow({"model_from_memory", params_file_});
  }
  os.InsetDivider();
  // cpu info
  os.InsertRow(
      {"cpu_math_thread", std::to_string(cpu_math_library_num_threads_)});
  os.InsertRow({"enable_mkldnn", use_mkldnn_ ? "true" : "false"});
  os.InsertRow(
      {"mkldnn_cache_capacity", std::to_string(mkldnn_cache_capacity_)});
#ifdef PADDLE_WITH_OPENVINO
  os.InsertRow({"use_openvino", use_openvino_ ? "true" : "false"});
  os.InsertRow({"openvino_inference_precision",
                inference::Precision2String(openvino_inference_precision_)});
#endif
  os.InsetDivider();

  // gpu info
  os.InsertRow({"use_gpu", use_gpu_ ? "true" : "false"});
  if (use_gpu_) {
    os.InsertRow({"use_cutlass", use_cutlass_ ? "true" : "false"});
    os.InsertRow({"gpu_device_id", std::to_string(gpu_device_id_)});
    os.InsertRow({"enable_gpu_mixed", std::to_string(enable_gpu_mixed_)});
    os.InsertRow({"mixed_precision_mode",
                  inference::Precision2String(mixed_precision_mode_)});
    os.InsertRow({"memory_pool_init_size",
                  std::to_string(memory_pool_init_size_mb_) + "MB"});
    os.InsertRow(
        {"use_external_stream", use_external_stream_ ? "true" : "false"});
    os.InsertRow(
        {"thread_local_stream", thread_local_stream_ ? "true" : "false"});

    os.InsertRow({"use_tensorrt", use_tensorrt_ ? "true" : "false"});
    if (use_tensorrt_) {
#ifdef PADDLE_WITH_TENSORRT
      auto version2string =
          [](const std::tuple<int, int, int> &ver) -> std::string {
        std::ostringstream os;
        int major = std::get<0>(ver);
        int minor = std::get<1>(ver);
        int patch = std::get<2>(ver);
        os << major << "." << minor << "." << patch;
        return os.str();
      };
      os.InsertRow({"trt_refit_params_path", refit_params_path_});
      os.InsertRow(
          {"trt_compile_version",
           version2string(inference::tensorrt::GetTrtCompileVersion())});
      os.InsertRow(
          {"trt_runtime_version",
           version2string(inference::tensorrt::GetTrtRuntimeVersion())});
      os.InsertRow({"tensorrt_precision_mode",
                    inference::Precision2String(tensorrt_precision_mode_)});
      os.InsertRow({"tensorrt_workspace_size",
                    std::to_string(tensorrt_workspace_size_)});
      os.InsertRow(
          {"tensorrt_max_batch_size", std::to_string(tensorrt_max_batchsize_)});
      os.InsertRow({"tensorrt_min_subgraph_size",
                    std::to_string(tensorrt_min_subgraph_size_)});
      os.InsertRow({"tensorrt_use_static_engine",
                    trt_use_static_engine_ ? "true" : "false"});
      os.InsertRow(
          {"tensorrt_use_calib_mode", trt_use_calib_mode_ ? "true" : "false"});
      os.InsertRow(
          {"tensorrt_use_cuda_graph", trt_use_cuda_graph_ ? "true" : "false"});

      // dynamic_shape
      os.InsertRow({"tensorrt_enable_dynamic_shape",
                    min_input_shape_.empty() ? "false" : "true"});
      os.InsertRow(
          {"tensorrt_tuned_dynamic_shape",
           trt_tuned_dynamic_shape_ ? shape_range_info_path_ : "false"});

      os.InsertRow(
          {"tensorrt_use_varseqlen", trt_use_varseqlen_ ? "true" : "false"});
      os.InsertRow({"tensorrt_with_interleaved",
                    trt_with_interleaved_ ? "true" : "false"});
      os.InsertRow({"tensorrt_transformer_posid", tensorrt_transformer_posid_});
      os.InsertRow(
          {"tensorrt_transformer_maskid", tensorrt_transformer_maskid_});
      os.InsertRow({"tensorrt_use_dla", trt_use_dla_ ? "true" : "false"});
      if (trt_use_dla_) {
        os.InsertRow({"tensorrt_dla_core", std::to_string(trt_dla_core_)});
      }
      os.InsertRow({"trt_engine_memory_sharing",
                    trt_engine_memory_sharing_ ? "true" : "false"});
      os.InsertRow({"trt_mark_output", trt_mark_output_ ? "true" : "false"});
      os.InsertRow(
          {"trt_forbid_dynamic_op", trt_forbid_dynamic_op_ ? "true" : "false"});

#endif
    }
  }
  os.InsetDivider();

  // xpu info
  os.InsertRow({"use_xpu", use_xpu_ ? "true" : "false"});
  if (use_xpu_) {
    os.InsertRow({"xpu_device_id", std::to_string(xpu_config_.device_id)});
    os.InsertRow({"xpu_l3_size", std::to_string(xpu_config_.l3_size)});
    os.InsertRow(
        {"xpu_l3_ptr",
         std::to_string(reinterpret_cast<int64_t>(xpu_config_.l3_ptr))});
    os.InsertRow(
        {"xpu_l3_autotune_size", std::to_string(xpu_config_.l3_autotune_size)});
    os.InsertRow(
        {"xpu_context_gm_size", std::to_string(xpu_config_.context_gm_size)});
    os.InsertRow(
        {"xpu_context",
         std::to_string(reinterpret_cast<int64_t>(xpu_config_.context))});
    os.InsertRow(
        {"xpu_stream",
         std::to_string(reinterpret_cast<int64_t>(xpu_config_.stream))});
    os.InsertRow({"xpu_conv_autotune_level",
                  std::to_string(xpu_config_.conv_autotune_level)});
    os.InsertRow({"xpu_conv_autotune_file", xpu_config_.conv_autotune_file});
    os.InsertRow({"xpu_conv_autotune_file_writeback",
                  std::to_string(xpu_config_.conv_autotune_file_writeback)});
    os.InsertRow({"xpu_fc_autotune_level",
                  std::to_string(xpu_config_.fc_autotune_level)});
    os.InsertRow({"xpu_fc_autotune_file", xpu_config_.fc_autotune_file});
    os.InsertRow({"xpu_fc_autotune_file_writeback",
                  std::to_string(xpu_config_.fc_autotune_file_writeback)});
    os.InsertRow({"xpu_gemm_compute_precision",
                  std::to_string(xpu_config_.gemm_compute_precision)});
    os.InsertRow(
        {"xpu_transformer_softmax_optimize_level",
         std::to_string(xpu_config_.transformer_softmax_optimize_level)});
    os.InsertRow(
        {"xpu_transformer_encoder_adaptive_seqlen",
         std::to_string(xpu_config_.transformer_encoder_adaptive_seqlen)});
    os.InsertRow(
        {"xpu_quant_post_static_gelu_out_threshold",
         std::to_string(xpu_config_.quant_post_static_gelu_out_threshold)});
    os.InsertRow(
        {"xpu_quant_post_dynamic_activation_method",
         std::to_string(xpu_config_.quant_post_dynamic_activation_method)});
    os.InsertRow(
        {"xpu_quant_post_dynamic_weight_precision ",
         std::to_string(xpu_config_.quant_post_dynamic_weight_precision)});
    std::vector<std::string> quant_post_dynamic_op_types_info =
        xpu_config_.quant_post_dynamic_op_types;
    quant_post_dynamic_op_types_info.insert(
        quant_post_dynamic_op_types_info.begin(),
        "xpu_quant_post_dynamic_op_types");
    os.InsertRow(quant_post_dynamic_op_types_info);
  }
  os.InsetDivider();

  // cinn compiler
  os.InsertRow({"use_cinn_compiler", use_cinn_ ? "true" : "false"});

  // ir info
  os.InsertRow(
      {"save_optimized_model", save_optimized_model_ ? "true" : "false"});
  os.InsertRow({"ir_optim", enable_ir_optim_ ? "true" : "false"});
  os.InsertRow({"ir_debug", ir_debug_ ? "true" : "false"});
  os.InsertRow(
      {"use_optimized_model", use_optimized_model_ ? "true" : "false"});
  os.InsertRow({"memory_optim", enable_memory_optim_ ? "true" : "false"});
  os.InsertRow({"enable_profile", with_profile_ ? "true" : "false"});
  os.InsertRow({"enable_log", with_glog_info_ ? "true" : "false"});
  os.InsertRow({"collect_shape_range_info",
                collect_shape_range_info_ ? shape_range_info_path_ : "false"});

  return os.PrintTable();
}

void AnalysisConfig::CollectShapeRangeInfo(
    const std::string &shape_range_info_path) {
  LOG(INFO) << "In CollectShapeInfo mode, we will disable optimizations and "
               "collect the shape information of "
            << "all intermediate tensors in the compute graph and calculate "
               "the min_shape, max_shape and opt_shape.";
  collect_shape_range_info_ = true;
  PADDLE_ENFORCE_EQ(shape_range_info_path.empty(),
                    false,
                    common::errors::InvalidArgument(
                        "The shape_range_info_path should not be empty, please "
                        "re-check the argument."));
  shape_range_info_path_ = shape_range_info_path;
}

const std::string &AnalysisConfig::shape_range_info_path() const {
  return shape_range_info_path_;
}

bool AnalysisConfig::shape_range_info_collected() const {
  return collect_shape_range_info_;
}

void AnalysisConfig::EnableTunedTensorRtDynamicShape(
    const std::string &shape_range_info_path, bool allow_build_at_runtime) {
  shape_range_info_path_ = shape_range_info_path;
  trt_allow_build_at_runtime_ = allow_build_at_runtime;
  trt_tuned_dynamic_shape_ = true;
}

bool AnalysisConfig::tuned_tensorrt_dynamic_shape() const {
  return trt_tuned_dynamic_shape_;
}

bool AnalysisConfig::trt_allow_build_at_runtime() const {
  return trt_allow_build_at_runtime_;
}

void AnalysisConfig::Exp_DisableMixedPrecisionOps(
    const std::unordered_set<std::string> &black_list) {
  mixed_black_list_ = black_list;
}

void AnalysisConfig::Exp_EnableMixedPrecisionOps(
    const std::unordered_set<std::string> &white_list) {
  mixed_white_list_ = white_list;
}

void AnalysisConfig::EnableCINN() {
#ifdef PADDLE_WITH_CINN
  use_cinn_ = true;
  Update();
#else
  PADDLE_THROW(common::errors::Unavailable(
      "You tried to use CINN compiler, but Paddle was not compiled "
      "with CINN."));
#endif
}

bool AnalysisConfig::cinn_enabled() const {
  bool is_enabled = use_cinn_;
#ifdef PADDLE_WITH_CINN
  is_enabled = is_enabled || FLAGS_use_cinn;
#endif
  return is_enabled;
}

void AnalysisConfig::EnableCustomPasses(const std::vector<std::string> &passes,
                                        bool custom_pass_only) {
  custom_passes_ = passes;
  custom_pass_only_ = custom_pass_only;
}

void AnalysisConfig::DeletePass(const std::string &pass_name) {
  deleted_passes_.push_back(pass_name);
}

void AnalysisConfig::SetOptimizationLevel(int opt_level) {
  pm_opt_level_ = opt_level;
}
}  // namespace paddle
