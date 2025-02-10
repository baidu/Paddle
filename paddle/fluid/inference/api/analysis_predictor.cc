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

#include "paddle/fluid/inference/api/analysis_predictor.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/feed_hook.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/pass_result_info.h"
#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/infer_context.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/api/resource_manager.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/inference/utils/model_utils.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/fluid/primitive/base/decomp_trans.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/profiler.h"

#include "paddle/phi/core/generator.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/utils/string/split.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/phi/backends/dynload/mklml.h"
#endif

#ifdef PADDLE_WITH_ONNXRUNTIME
#include "paddle/fluid/inference/api/onnxruntime_predictor.h"
#endif

#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/paddle_ipu_handler.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_info.h"
#endif

#ifdef PADDLE_WITH_NVTX
#include "paddle/phi/core/platform/device/gpu/cuda/cuda_profiler.h"
#endif

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#endif

#include "paddle/common/flags.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/fluid/pir/transforms/general/auto_mixed_precision_pass.h"
#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include "paddle/fluid/pir/transforms/general/constant_folding_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/general/delete_assert_op_pass.h"
#include "paddle/fluid/pir/transforms/general/inplace_pass.h"
#include "paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.h"
#include "paddle/fluid/pir/transforms/general/remove_shadow_feed_pass.h"
#include "paddle/fluid/pir/transforms/general/replace_fetch_with_shadow_output_pass.h"
#include "paddle/fluid/pir/transforms/general/transfer_layout_pass.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block_argument.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_bool(pir_apply_inplace_pass);
COMMON_DECLARE_bool(enable_auto_layout_pass);
namespace paddle {
namespace {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void UpdatePrivateDeviceContext(InferGPUContext *gpu_context,
                                GPUContextResource *gpu_resource,
                                Place place_) {
  gpu_context->SetAllocator(memory::allocation::AllocatorFacade::Instance()
                                .GetAllocator(place_, gpu_resource->GetStream())
                                .get());
  gpu_context->SetPinnedAllocator(
      memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::GPUPinnedPlace())
          .get());
  gpu_context->SetHostAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetAllocator(phi::CPUPlace())
                                    .get());
  gpu_context->SetZeroAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetZeroAllocator(place_)
                                    .get());
  gpu_context->SetHostZeroAllocator(
      memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(phi::CPUPlace())
          .get());
  gpu_context->SetGenerator(
      phi::DefaultCUDAGenerator(place_.GetDeviceId()).get());
  gpu_context->SetHostGenerator(phi::DefaultCPUGenerator().get());

  gpu_context->SetStream(gpu_resource->GetStream());
  gpu_context->SetBlasHandle(gpu_resource->GetBlasHandleCreator());
  gpu_context->SetBlasTensorCoreHandle(
      gpu_resource->GetBlasTensorCoreHandleCreator());
  gpu_context->SetBlasTF32Handle(
      gpu_resource->GetBlasTF32TensorCoreHandleCreator());
  gpu_context->SetDnnHandle(gpu_resource->GetDnnHandleCreator());
  gpu_context->SetSolverHandle(gpu_resource->GetSolverDnHandleCreator());
  gpu_context->SetSparseHandle(gpu_resource->GetSparseHandleCreator());
  gpu_context->SetEigenDevice(gpu_resource->GetGpuEigenDevice());

  gpu_context->SetComputeCapability(gpu_resource->GetGpuComputeCapability());
  gpu_context->SetMaxThreadsPerBlock(gpu_resource->GetGpuMaxThreadsPerBlock());
  gpu_context->SetMaxThreadsPerMultiProcessor(
      gpu_resource->GetGpuMaxThreadsPerMp());
  gpu_context->SetMaxGridDimSize(gpu_resource->GetGpuMaxGridDimSize());
  gpu_context->SetMultiProcessors(gpu_resource->GetGPUMultiProcessors());
  gpu_context->SetDriverVersion(gpu_resource->GetGpuDriverVersion());
  gpu_context->SetRuntimeVersion(gpu_resource->GetGpuRuntimeVersion());
  VLOG(1) << "thread id is " << std::this_thread::get_id() << ", stream id is "
          << reinterpret_cast<void *>(gpu_resource->GetStream())
          << ", allotor ptr is "
          << reinterpret_cast<void *>(
                 memory::allocation::AllocatorFacade::Instance()
                     .GetAllocator(place_, gpu_resource->GetStream())
                     .get());
}
#endif
}  // namespace

#ifdef PADDLE_WITH_TENSORRT
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;
using inference::tensorrt::TRTInt8Calibrator;
#endif

int AnalysisPredictor::clone_num_ = 1;

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}

phi::DataType ConvertPrecision(AnalysisConfig::Precision precision) {
  switch (precision) {
    case AnalysisConfig::Precision::kFloat32:
      return phi::DataType::FLOAT32;
    case AnalysisConfig::Precision::kHalf:
      return phi::DataType::FLOAT16;
    case AnalysisConfig::Precision::kBf16:
      return phi::DataType::BFLOAT16;
    case AnalysisConfig::Precision::kInt8:
      return phi::DataType::INT8;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Paddle Inference not support precision. We now only support "
          "Float32, Half, Bfloat16 and Int8"));
      return phi::DataType::FLOAT32;
  }
}

phi::Backend ConvertBackend(paddle_infer::PlaceType backend) {
  switch (backend) {
    case paddle_infer::PlaceType::kGPU:
      // NOTE: phi also support phi::Backend::GPUDNN.
      return phi::Backend::GPU;
    case paddle_infer::PlaceType::kXPU:
      return phi::Backend::XPU;
    case paddle_infer::PlaceType::kCPU:
      return phi::Backend::CPU;
    case paddle_infer::PlaceType::kIPU:
      return phi::Backend::IPU;
    case paddle_infer::PlaceType::kCUSTOM:
      return phi::Backend::CUSTOM;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Paddle Inference not support backend, we now only support GPU, XPU "
          "and CPU."));
      return phi::Backend::CPU;
  }
}

bool PaddleTensorToDenseTensor(const PaddleTensor &pt,
                               phi::DenseTensor *t,
                               const phi::Place &place) {
  phi::DDim ddim = common::make_ddim(pt.shape);
  void *input_ptr = nullptr;
  if (pt.dtype == PaddleDType::INT64) {
    input_ptr = t->mutable_data<int64_t>(ddim, place);
  } else if (pt.dtype == PaddleDType::FLOAT32) {
    input_ptr = t->mutable_data<float>(ddim, place);
  } else if (pt.dtype == PaddleDType::INT32) {
    input_ptr = t->mutable_data<int32_t>(ddim, place);
  } else if (pt.dtype == PaddleDType::FLOAT16) {
    input_ptr = t->mutable_data<float16>(ddim, place);
  } else if (pt.dtype == PaddleDType::BFLOAT16) {
    input_ptr = t->mutable_data<bfloat16>(ddim, place);
  } else {
    LOG(ERROR) << "unsupported feed type " << pt.dtype;
    return false;
  }
  // NOTE(Aurelius84): Some kernels support zero shape input
  // without memory holder, we should skip enforce logic.
  bool has_zero_dim = (common::product(ddim) == 0);
  VLOG(3) << "Found zero dim: " << has_zero_dim
          << " from input with ddim: " << ddim;
  if (!has_zero_dim) {
    PADDLE_ENFORCE_NOT_NULL(
        input_ptr,
        common::errors::Fatal("Cannot convert to DenseTensor because "
                              "DenseTensor creation failed."));
    PADDLE_ENFORCE_NOT_NULL(
        pt.data.data(),
        common::errors::InvalidArgument(
            "The data contained in the input PaddleTensor is illegal."));
    PADDLE_ENFORCE_EQ(
        pt.data.length(),
        t->numel() * phi::SizeOf(t->dtype()),
        common::errors::InvalidArgument(
            "The data contained in the input PaddleTensor had wrong length."));
  }

  if (phi::is_cpu_place(place)) {
    // TODO(panyx0718): Init DenseTensor from existing memcpy to save a copy.
    if (input_ptr != nullptr) {
      std::memcpy(
          static_cast<void *>(input_ptr), pt.data.data(), pt.data.length());
    }
  } else if (phi::is_ipu_place(place)) {
#ifdef PADDLE_WITH_IPU
    std::memcpy(
        static_cast<void *>(input_ptr), pt.data.data(), pt.data.length());
#else
    PADDLE_THROW(common::errors::Fatal(
        "Not compile with WITH_IPU, should not reach here."));
#endif
  } else if (phi::is_gpu_place(place)) {
    PADDLE_ENFORCE_EQ(phi::is_xpu_place(place),
                      false,
                      common::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    auto *dev_ctx = static_cast<const phi::GPUContext *>(pool.Get(place));
    auto dst_gpu_place = place;
    memory::Copy(dst_gpu_place,
                 static_cast<void *>(input_ptr),
                 phi::CPUPlace(),
                 pt.data.data(),
                 pt.data.length(),
                 dev_ctx->stream());
#else
    PADDLE_THROW(
        common::errors::Fatal("Not compile with CUDA, should not reach here."));
#endif
  } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    auto dst_xpu_place = place;
    memory::Copy(dst_xpu_place,
                 static_cast<void *>(input_ptr),
                 phi::CPUPlace(),
                 pt.data.data(),
                 pt.data.length());
#else
    PADDLE_THROW(
        common::errors::Fatal("Not compile with XPU, should not reach here."));
#endif
  } else if (phi::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    auto custom_place = place;
    auto *dev_ctx =
        static_cast<const phi::CustomContext *>(pool.Get(custom_place));
    memory::Copy(custom_place,
                 static_cast<void *>(input_ptr),
                 phi::CPUPlace(),
                 pt.data.data(),
                 pt.data.length(),
                 dev_ctx->stream());
#else
    PADDLE_THROW(common::errors::Fatal(
        "Not compile with CUSTOM_DEVICE, should not reach here."));
#endif
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The analysis predictor supports CPU, GPU, XPU and CUSTOM_DEVICE "
        "now."));
  }
  // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
  phi::LegacyLoD lod;
  for (auto &level : pt.lod) {
    lod.emplace_back(level);
  }
  t->set_lod(lod);
  return true;
}
}  // namespace

AnalysisPredictor::AnalysisPredictor(const AnalysisConfig &config)
    : config_(config),
      fusion_statis_(),
      executor_(nullptr),
      feeds_(),
      feed_names_(),
      idx2feeds_(),
      fetches_(),
      idx2fetches_(),
      feed_tensors_(),
      output_hookfuncs_(),
      input_hookfuncs_(),
      shape_info_(),
      shape_tensor_value_(),
      device_contexts_() {
  if (config_.shape_range_info_collected()) {
    config_.SwitchIrOptim(false);
  }
  if (config_.new_executor_enabled()) {
    config_.EnableMemoryOptim(false);
    if (config_.new_ir_enabled()) {
      config_.SwitchIrOptim(false);
    }
  }
  if (!config_.new_ir_enabled()) {
    for (const auto &pass_name : config_.deleted_passes_) {
      config_.pass_builder()->DeletePass(pass_name);
    }
  }
  int trt_identifier = config_.trt_engine_memory_sharing_identifier_;
  if (trt_identifier > 0) {
    // NOTE(liuyuanle): For convenience, we set the id of the predictor to
    // negative sharing_identifier directly. In the future, this may affect
    // the meaning of negative predictor id.
    predictor_id_ = -trt_identifier;
    LOG(WARNING)
        << "Since the engine context memory of multiple predictors "
           "is enabled in Paddle-TRT, we set the id of these predictors to "
           "negative sharing_identifier you specified : "
        << predictor_id_;
    PADDLE_ENFORCE_EQ(
        config_.new_executor_enabled(),
        true,
        common::errors::InvalidArgument(
            "Please call the config.enable_new_executor() in python or "
            "config.EnableNewExecutor() in c++ when you want share the engine "
            "context memory of multiple predictors."));
  } else {
    predictor_id_ = inference::GetUniqueId();
  }
}

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
  VLOG(3) << "Predictor::init()";

  if (config_.with_profile_) {
    LOG(WARNING) << "Profiler is activated, which might affect the performance";
#ifdef PADDLE_WITH_NVTX
    platform::CudaProfilerStart();
    platform::NvprofEnableRecordEvent();
#endif
    platform::EnableProfiler(config_.use_gpu() ? platform::ProfilerState::kAll
                                               : platform::ProfilerState::kCPU);
  }

  if (!status_is_cloned_) {
    root_predictor_id_ = predictor_id_;
  }

  // no matter with or without OneDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  std::string model_path = config_.prog_file();
  if (!model_path.empty()) {
    load_pir_model_ =
        model_path.substr(model_path.find_last_of(".") + 1) == "json";
  } else if (!config_.model_dir().empty()) {
    std::string model_dir = config_.model_dir();
    load_pir_model_ = false;
    for (const auto &entry : std::filesystem::directory_iterator(model_dir)) {
      if (entry.is_regular_file() &&
          entry.path().filename() == "__model__.json") {
        load_pir_model_ = true;
        config_.SetProgFile(config_.model_dir() + "/__model__.json");
        break;
      }
    }
  }
  if (load_pir_model_) {
    config_.use_pir_ = true;
    config_.use_new_executor_ = true;
  }

  // Use Optimized model to inference
  if (config_.use_optimized_model_) {
    std::string optimized_model_path = GetOptimizedModelPath();
    std::string optimized_model;
    if (config_.new_ir_enabled()) {
      optimized_model = optimized_model_path + "/" + "_optimized.json";
    } else {
      optimized_model = optimized_model_path + "/" + "_optimized.pdmodel";
    }
    std::string optimized_params =
        optimized_model_path + "/" + "_optimized.pdiparams";
    if (FileExists(optimized_model) && FileExists(optimized_params)) {
      config_.SetModel(optimized_model, optimized_params);
      if (config_.new_ir_enabled()) {
        load_pir_model_ = true;
      }
      LOG(INFO) << "Load Optimized model from " << optimized_model
                << " and Load Optimized optimized_params from "
                << optimized_params;
    } else {
      LOG(WARNING)
          << "The optimized model is not found, fallback to original model. "
             "EnableSaveOptimModel will be turned on and the optimized model "
             "can be available next time.";
      config_.EnableSaveOptimModel(true);
      config_.UseOptimizedModel(false);
    }
  }

  if (!PrepareScope(parent_scope)) {
    return false;
  }
  InitPlace();

  if (!CreateExecutor()) {
    return false;
  }

  if (load_pir_model_) {
    if (!PreparePirProgram()) {
      return false;
    }
  } else {
    if (!PrepareProgram(program)) {
      return false;
    }
  }

  // Get the feed_target_names and fetch_target_names

  PrepareFeedFetch();

  // Prepare executor, create local variables.
  if (!PrepareExecutor()) {
    return true;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // TODO(inference): Now only gpu with external stream support private
  // device_context.
  if (config_.use_gpu_ && config_.use_external_stream_) {
    private_context_ = true;
  }
  if (private_context_) {
    if (!status_is_cloned_) {
      predictor_stream_ = config_.GetExecStream();
    }
    // NOTE: If the external_stream equals to global_device_contexts's stream,
    // then fallback.
    auto global_stream = static_cast<phi::GPUContext *>(
                             phi::DeviceContextPool::Instance().Get(place_))
                             ->stream();
    if (predictor_stream_ != global_stream) {
      InitResourceManager(predictor_stream_);
      InitDeviceContexts();
    }
  }
#endif
#if defined(PADDLE_WITH_XPU)
  if (config_.use_xpu_) {
    private_context_ = true;
    if (!status_is_cloned_ && config_.external_stream_enabled()) {
      predictor_stream_ = config_.GetExecStream();
    }
    if (predictor_stream_ == nullptr) {
      auto *global_context = static_cast<phi::XPUContext *>(
          phi::DeviceContextPool::Instance().Get(place_));
      predictor_stream_ = global_context->stream();
    }
    InitDeviceContexts();
  }
#endif

  TryShrinkMemory();

  inference::DisplayMemoryInfo(place_, "Init predictor");
  return true;
}

void AnalysisPredictor::InitPlace() {
  if (config_.use_gpu()) {
    PADDLE_ENFORCE_EQ(config_.use_xpu(),
                      false,
                      common::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    place_ = phi::GPUPlace(config_.gpu_device_id());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (config_.thread_local_stream_enabled()) {
      LOG_FIRST_N(WARNING, 1) << "We will remove this interface in the future. "
                                 "Please use config.SetExecStream instead.";
    }
#endif
  } else if (config_.use_xpu()) {
#ifdef PADDLE_WITH_XPU
    phi::backends::xpu::SetXPUDeviceId(config_.xpu_device_id());
    place_ = phi::XPUPlace(config_.xpu_device_id());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "You tried to use XPU forward propagation (inference without lite "
        "engine), but Paddle was not compiled "
        "with WITH_XPU."));
#endif  // PADDLE_WITH_XPU
  } else if (config_.use_ipu()) {
#ifdef PADDLE_WITH_IPU
    place_ = phi::IPUPlace();
#else
    PADDLE_THROW(common::errors::Unavailable(
        "You tried to use IPU forward propagation, but Paddle was not compiled "
        "with WITH_IPU."));
#endif
  } else if (config_.use_custom_device()) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    place_ = phi::CustomPlace(config_.custom_device_type(),
                              config_.custom_device_id());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "You tried to use CustomDevice forward propagation, but Paddle was not "
        "compiled "
        "with WITH_CUSTOM_DEVICE."));
#endif
  } else {
    place_ = phi::CPUPlace();
  }
}

std::string AnalysisPredictor::GetOptimizedModelPath() {
  std::string model_opt_cache_dir = config_.opt_cache_dir_;
  if (!model_opt_cache_dir.empty()) {
    if (!PathExists(model_opt_cache_dir)) {
      PADDLE_ENFORCE_NE(
          MKDIR(model_opt_cache_dir.c_str()),
          -1,
          common::errors::PreconditionNotMet(
              "Can not create optimize cache directory: %s, Make sure you "
              "have permission to write",
              model_opt_cache_dir));
    }
  } else {
    model_opt_cache_dir =
        !config_.model_dir().empty()
            ? config_.model_dir()
            : inference::analysis::GetDirRoot(config_.prog_file());
  }
  return model_opt_cache_dir;
}

void AnalysisPredictor::ClearExtraParams() {
  auto var_names = scope_->LocalVarNames();
  std::vector<std::string> trt_repetitive_params;
  for (auto &op_desc : inference_program_->Block(0).AllOps()) {
    if (op_desc->Type() == "tensorrt_engine") {
      auto trt_params = PADDLE_GET_CONST(std::vector<std::string>,
                                         op_desc->GetAttr("parameters"));
      trt_repetitive_params.insert(
          trt_repetitive_params.end(), trt_params.begin(), trt_params.end());
      // NOTE(ming1753): This is a trick solution to the problem of possible
      // absolute paths in the model_opt_cache_dir and shape_range_info_path
      // attributes in tensorrt_engine op.
      auto model_opt_cache_dir_from_model = PADDLE_GET_CONST(
          std::string, op_desc->GetAttr("model_opt_cache_dir"));
      auto model_opt_cache_dir = GetOptimizedModelPath();
      if (op_desc->HasAttr("model_opt_cache_dir")) {
        op_desc->SetAttr("model_opt_cache_dir", model_opt_cache_dir);
      }
      if (op_desc->HasAttr("shape_range_info_path")) {
        if (config_.shape_range_info_path_.empty()) {
          op_desc->SetAttr(
              "shape_range_info_path",
              model_opt_cache_dir + "/" + "shape_range_info.pbtxt");
        } else {
          op_desc->SetAttr("shape_range_info_path",
                           config_.shape_range_info_path_);
        }
      }
      if (op_desc->HasAttr("predictor_id")) {
        op_desc->SetAttr("predictor_id", predictor_id_);
      }
    }
#ifdef PADDLE_WITH_OPENVINO
    if (op_desc->Type() == "openvino_engine") {
      if (op_desc->HasAttr("inference_num_threads")) {
        op_desc->SetAttr("inference_num_threads",
                         config_.cpu_math_library_num_threads_);
      }
    }
#endif
  }

  std::vector<std::string> extra_params;
  for (auto &var_desc : inference_program_->Block(0).AllVars()) {
    if (var_desc->Persistable()) {
      // Clear repetitive parameters in tensorrt
      if (scope_->FindVar(var_desc->Name()) &&
          std::count(trt_repetitive_params.begin(),
                     trt_repetitive_params.end(),
                     var_desc->Name())) {
        extra_params.emplace_back(var_desc->Name());
      }
    }
  }

  scope_->EraseVars(extra_params);
  VLOG(1) << "Clear " << extra_params.size() << " extra params.";
}

void AnalysisPredictor::InitResourceManager(void *stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  predictor_stream_ =
      ResourceManager::Instance().InitGPUResource(place_, stream);
#endif
}

void AnalysisPredictor::InitDeviceContexts() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Init GPUContext.
  if (place_.GetType() == phi::AllocationType::GPU) {
    device_contexts_.emplace(
        place_, std::async(std::launch::deferred, [=] {
          auto *gpu_resource =
              ResourceManager::Instance().GetGPUResource(predictor_stream_);
          auto *gpu_context = new InferGPUContext(place_);
          UpdatePrivateDeviceContext(gpu_context, gpu_resource, place_);
          return std::unique_ptr<phi::DeviceContext>(gpu_context);
        }));
  }
#endif
#ifdef PADDLE_WITH_XPU
  if (place_.GetType() == phi::AllocationType::XPU) {
    device_contexts_.emplace(
        place_, std::async(std::launch::deferred, [=] {
          auto &instance = memory::allocation::AllocatorFacade::Instance();
          auto *xpu_context =
              new InferXPUContext(place_, config_.xpu_config().context_gm_size);
          xpu_context->SetConvAutotuneInfo(
              config_.xpu_config_.conv_autotune_file,
              config_.xpu_config_.conv_autotune_level,
              config_.xpu_config_.conv_autotune_file_writeback,
              place_);
          xpu_context->SetFcAutotuneInfo(
              config_.xpu_config_.fc_autotune_file,
              config_.xpu_config_.fc_autotune_level,
              config_.xpu_config_.fc_autotune_file_writeback,
              place_);
          if (config_.xpu_config_.transformer_softmax_optimize_level > 0) {
            xpu_context->SetContextOption(
                "XPU_SOFTMAX_OPT",
                std::to_string(
                    config_.xpu_config_.transformer_softmax_optimize_level)
                    .c_str());
          }
          xpu_context->SetAllocator(instance.GetAllocator(place_).get());
          xpu_context->SetGenerator(
              phi::DefaultXPUGenerator(place_.GetDeviceId()).get());
          xpu_context->SetHostAllocator(
              instance.GetAllocator(phi::CPUPlace()).get());
          xpu_context->SetHostGenerator(phi::DefaultCPUGenerator().get());
          xpu_context->SetZeroAllocator(
              instance.GetZeroAllocator(place_).get());
          xpu_context->SetHostZeroAllocator(
              instance.GetZeroAllocator(phi::CPUPlace()).get());
          xpu_context->SetStream(predictor_stream_);
          return std::unique_ptr<phi::DeviceContext>(xpu_context);
        }));
  }
#endif
}

void *AnalysisPredictor::GetExecStream() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place_.GetType() == phi::AllocationType::GPU) {
    if (private_context_) {
      return predictor_stream_;
    } else {
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      return reinterpret_cast<const phi::GPUContext *>(pool.Get(place_))
          ->stream();
    }
  }
#endif
#if defined(PADDLE_WITH_XPU)
  if (place_.GetType() == phi::AllocationType::XPU) {
    if (private_context_) {
      return predictor_stream_;
    } else {
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      return reinterpret_cast<const phi::XPUContext *>(pool.Get(place_))
          ->stream();
    }
  }
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  if (place_.GetType() == phi::AllocationType::CUSTOM) {
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    return reinterpret_cast<const phi::CustomContext *>(pool.Get(place_))
        ->stream();
  }
#endif
  // TODO(inference): Support other backends.
  return nullptr;
}

const void *AnalysisPredictor::GetDeviceContexts() const {
  if (private_context_) {
    return &device_contexts_;
  } else {
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    const auto &dev_ctxs = pool.device_contexts();
    return &dev_ctxs;
  }
}

bool AnalysisPredictor::PrepareScope(
    const std::shared_ptr<framework::Scope> &parent_scope) {
#ifdef PADDLE_WITH_XPU
  // Set "XPU_PADDLE_L3_SIZE" to "0" to avoid malloc l3 cache when xpu_context
  // init.
  setenv("XPU_PADDLE_L3_SIZE", "0", 0);
#endif
  if (parent_scope) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        common::errors::PreconditionNotMet(
            "Both program and parent_scope should be set in Clone mode."));
    scope_ = parent_scope;
    status_is_cloned_ = true;
  } else {
    paddle::framework::InitMemoryMethod();
    paddle::framework::InitDevices();
    paddle::framework::InitDefaultKernelSignatureMap();
    // TODO(wilber): we need to release memory occupied by weights.
    scope_ = std::make_unique<paddle::framework::Scope>();
    status_is_cloned_ = false;
  }
  sub_scope_ = &scope_->NewScope();
  return true;
}

void AnalysisPredictor::OptimizeInferencePirProgram() {
  auto ir_printing_conditions = [this](::pir::Pass *pass,
                                       ::pir::Operation *op) {
    if (this->config_.ir_debug_passes_.empty()) {
      return true;
    }
    return std::find(this->config_.ir_debug_passes_.begin(),
                     this->config_.ir_debug_passes_.end(),
                     pass->name()) != this->config_.ir_debug_passes_.end();
  };

  auto AddAutoLayoutPasses = [&](pir::PassManager &pass_manager) {
    auto &pass_registry = pir::PassRegistry::Instance();
    std::vector<std::string> passes = {"auto_layout_pass"};

    for (const auto &pass_name : passes) {
      if (std::find(config_.deleted_passes_.begin(),
                    config_.deleted_passes_.end(),
                    pass_name) == config_.deleted_passes_.end()) {
        pass_manager.AddPass(pass_registry.Get(pass_name));
      }
    }
  };

  auto AddAutoMixedPrecisionPass = [&](pir::PassManager &pass_manager) {
    auto auto_mixed_precision_pass = ::pir::CreateAutoMixedPrecisionPass();
    if (std::find(config_.deleted_passes_.begin(),
                  config_.deleted_passes_.end(),
                  auto_mixed_precision_pass->name()) ==
        config_.deleted_passes_.end()) {
      auto_mixed_precision_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place_);
      auto_mixed_precision_pass->Set("mixed_precision_mode",
                                     new phi::DataType(paddle::ConvertPrecision(
                                         config_.mixed_precision_mode_)));
      auto_mixed_precision_pass->Set(
          "enable_low_precision_io",
          new bool(config_.enable_low_precision_io_));
      auto_mixed_precision_pass->Set(
          "mixed_black_list",
          new std::unordered_set<std::string>(config_.mixed_black_list_));
      auto_mixed_precision_pass->Set(
          "mixed_white_list",
          new std::unordered_set<std::string>(config_.mixed_white_list_));

      pass_manager.AddPass(std::move(auto_mixed_precision_pass));
    }
  };

  if (!config_.use_optimized_model_) {
#ifdef PADDLE_WITH_CINN
    auto CreatePassMgr = [&] {
      pir::IrContext *ctx = pir::IrContext::Instance();
      ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
      ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
      auto pass_manager = std::make_shared<::pir::PassManager>(
          ::pir::IrContext::Instance(), config_.pm_opt_level_);
      if (!config_.glog_info_disabled()) {
        pass_manager->EnablePrintStatistics();
      }
      if (config_.ir_debug_) {
        pass_manager->EnableIRPrinting(
            std::make_unique<pir::PassManager::IRPrinterOption>(
                ir_printing_conditions, ir_printing_conditions));
      }
      auto &shape_analysis =
          pir::ShapeAnalysisManager::Instance().Get(pir_program_.get());
      pass_manager->SetValueReplacedHook([&](pir::Value from, pir::Value to) {
        shape_analysis.ShareShapeOrData(from, to);
      });
      return pass_manager;
    };

    if (config_.cinn_enabled() && !config_.custom_pass_only_) {
      ::pir::PassManager delete_assert_op_pm(::pir::IrContext::Instance(),
                                             config_.pm_opt_level_);
      delete_assert_op_pm.AddPass(pir::CreateDeleteAssertOpPass());
      delete_assert_op_pm.Run(pir_program_.get());
    }

    if (config_.use_gpu() && config_.cinn_enabled()) {
      if (!config_.custom_pass_only_) {
        ::pir::PassManager fused_op_pm(::pir::IrContext::Instance(),
                                       config_.pm_opt_level_);
        auto &shape_analysis =
            pir::ShapeAnalysisManager::Instance().Get(pir_program_.get());
        fused_op_pm.SetValueReplacedHook([&](pir::Value from, pir::Value to) {
          shape_analysis.ShareShapeOrData(from, to);
        });
        // Infer symbol shape for all ops before fused pass
        fused_op_pm.AddPass(pir::CreateShapeOptimizationPass());
        const std::vector<std::string> FusedOpPasses{
            // Operator fusion pass
            "map_op_to_another_pass",
            "conv2d_bn_fuse_pass",
            "conv2d_add_act_fuse_pass",
            "conv2d_add_fuse_pass",
            "matmul_add_act_fuse_pass"};

        for (const auto &fused_op : FusedOpPasses) {
          fused_op_pm.AddPass(pir::PassRegistry::Instance().Get(fused_op));
        }

        if (config_.enable_gpu_mixed_) {
          AddAutoMixedPrecisionPass(fused_op_pm);
          if (FLAGS_enable_auto_layout_pass) {
            AddAutoLayoutPasses(fused_op_pm);
          } else {
            fused_op_pm.AddPass(
                pir::PassRegistry::Instance().Get("transfer_layout_pass"));
          }
        }
        fused_op_pm.Run(pir_program_.get());
      }
    }

    if (paddle::prim::PrimCommonUtils::IsFwdPrimEnabled()) {
      VLOG(4) << "[Prim] Decomp program in predictor begin.";
      DecompProgram decomp_object(pir_program_.get());
      decomp_object.decomp_program();

      cinn::dialect::ir::CheckInferSymbolicIfNeed(pir_program_.get(),
                                                  CreatePassMgr);
    }

    if (config_.cinn_enabled()) {
      VLOG(4) << "[CINN] Begin ApplyCinnPass";
      cinn::dialect::ir::ApplyCinnPass(
          pir_program_.get(), CreatePassMgr, false);
    }
#endif

    // Apply some optimization passes required by the inference
    ::pir::PassManager pass_pm(::pir::IrContext::Instance(),
                               config_.pm_opt_level_);
    if (!config_.custom_passes_.empty()) {
      for (const auto &custom_pass : config_.custom_passes_) {
        pass_pm.AddPass(pir::PassRegistry::Instance().Get(custom_pass));
      }
    }
    if (config_.use_gpu()) {
      // gpu
      if (!config_.custom_pass_only_) {
        for (const auto &gpu_pass : kPirGpuPasses) {
          if (std::find(config_.deleted_passes_.begin(),
                        config_.deleted_passes_.end(),
                        gpu_pass) == config_.deleted_passes_.end()) {
            pass_pm.AddPass(pir::PassRegistry::Instance().Get(gpu_pass));
          }
        }
      }
#ifdef PADDLE_WITH_XPU
    } else if (config_.use_xpu()) {
      // xpu
      if (!config_.custom_pass_only_) {
        for (const auto &xpu_pass : kPirXpuPasses) {
          if (std::find(config_.deleted_passes_.begin(),
                        config_.deleted_passes_.end(),
                        xpu_pass) == config_.deleted_passes_.end()) {
            pass_pm.AddPass(
                std::move(pir::PassRegistry::Instance().Get(xpu_pass)));
          }
        }
      }
#endif

#ifdef PADDLE_WITH_DNNL
    } else if (config_.mkldnn_enabled()) {
      // mkldnn
      if (!config_.custom_pass_only_) {
        for (const auto &mkldnn_pass : kPirMkldnnPasses) {
          if (std::find(config_.deleted_passes_.begin(),
                        config_.deleted_passes_.end(),
                        mkldnn_pass) == config_.deleted_passes_.end()) {
            pass_pm.AddPass(pir::PassRegistry::Instance().Get(mkldnn_pass));
          }
        }
        if (config_.mkldnn_bfloat16_enabled()) {
          for (const auto &mkldnn_pass : kPirMkldnnBf16Passes) {
            if (std::find(config_.deleted_passes_.begin(),
                          config_.deleted_passes_.end(),
                          mkldnn_pass) == config_.deleted_passes_.end()) {
              pass_pm.AddPass(pir::PassRegistry::Instance().Get(mkldnn_pass));
            }
          }
        }
      }
#endif
    } else {
      // cpu
      if (!config_.custom_pass_only_) {
        for (const auto &cpu_pass : kPirCpuPasses) {
          if (std::find(config_.deleted_passes_.begin(),
                        config_.deleted_passes_.end(),
                        cpu_pass) == config_.deleted_passes_.end()) {
            pass_pm.AddPass(pir::PassRegistry::Instance().Get(cpu_pass));
          }
        }
      }
    }

    // set attr
    for (const auto &pass : pass_pm.passes()) {
      pass->SetNotOwned(pir::Pass::kParamScopeAttr, sub_scope_);
      pass->SetNotOwned(pir::Pass::kPlaceAttr, &place_);
      pass->Set("enable_gpu_mixed", new bool(config_.enable_gpu_mixed_));
      if (pass->name() == "matmul_add_act_fuse_pass" ||
          pass->name() == "conv2d_add_act_fuse_pass" ||
          pass->name() == "conv2d_add_fuse_pass") {
        pass->Set("use_cutlass", new bool(config_.use_cutlass_));
      }
    }

    if (!config_.glog_info_disabled()) {
      pass_pm.EnablePrintStatistics();
    }
    if (config_.ir_debug_) {
      pass_pm.EnableIRPrinting(
          std::make_unique<pir::PassManager::IRPrinterOption>(
              ir_printing_conditions, ir_printing_conditions));
    }

    pass_pm.Run(pir_program_.get());

    if (config_.save_optimized_model_) {
      std::string optimized_model =
          GetOptimizedModelPath() + "/" + "_optimized.json";
      pir::WriteModule(*pir_program_, optimized_model, 1, true, false, true);
      LOG(INFO) << "Optimized model saved to " << optimized_model;
      SaveOrLoadPirParameters(true);
    }
  }

  // Apply some basic passes required by the framework
  ::pir::PassManager basic_pass_pm(::pir::IrContext::Instance(),
                                   config_.pm_opt_level_);
  if (config_.enable_gpu_mixed_) {
    if (!config_.cinn_enabled()) {
      AddAutoMixedPrecisionPass(basic_pass_pm);
    }
  }
  if (FLAGS_enable_auto_layout_pass) {
    AddAutoLayoutPasses(basic_pass_pm);
  } else {
    auto transfer_layout_pass = ::pir::CreateTransferLayoutPass();
    if (std::find(config_.deleted_passes_.begin(),
                  config_.deleted_passes_.end(),
                  transfer_layout_pass->name()) ==
        config_.deleted_passes_.end()) {
      basic_pass_pm.AddPass(std::move(transfer_layout_pass));
    }
  }
  auto common_subexpression_elimination_pass =
      ::pir::CreateCommonSubexpressionEliminationPass();
  if (std::find(config_.deleted_passes_.begin(),
                config_.deleted_passes_.end(),
                common_subexpression_elimination_pass->name()) ==
      config_.deleted_passes_.end()) {
    basic_pass_pm.AddPass(std::move(common_subexpression_elimination_pass));
  }
  auto params_sync_among_devices_pass =
      ::pir::CreateParamsSyncAmongDevicesPass();
  if (std::find(config_.deleted_passes_.begin(),
                config_.deleted_passes_.end(),
                params_sync_among_devices_pass->name()) ==
      config_.deleted_passes_.end()) {
    params_sync_among_devices_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place_);
    params_sync_among_devices_pass->SetNotOwned(pir::Pass::kParamScopeAttr,
                                                sub_scope_);
    basic_pass_pm.AddPass(std::move(params_sync_among_devices_pass));
  }
  auto constant_folding_pass = ::pir::CreateConstantFoldingPass();
  if (std::find(config_.deleted_passes_.begin(),
                config_.deleted_passes_.end(),
                constant_folding_pass->name()) ==
      config_.deleted_passes_.end()) {
    constant_folding_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place_);
    constant_folding_pass->SetNotOwned(pir::Pass::kParamScopeAttr, sub_scope_);
    basic_pass_pm.AddPass(std::move(constant_folding_pass));
  }
  auto dead_code_elimination_pass = ::pir::CreateDeadCodeEliminationPass();
  if (std::find(config_.deleted_passes_.begin(),
                config_.deleted_passes_.end(),
                dead_code_elimination_pass->name()) ==
      config_.deleted_passes_.end()) {
    dead_code_elimination_pass->SetNotOwned(pir::Pass::kParamScopeAttr,
                                            sub_scope_);
    basic_pass_pm.AddPass(std::move(dead_code_elimination_pass));
  }
  auto replace_fetch_with_shadow_output_pass =
      ::pir::CreateReplaceFetchWithShadowOutputPass();
  if (std::find(config_.deleted_passes_.begin(),
                config_.deleted_passes_.end(),
                replace_fetch_with_shadow_output_pass->name()) ==
      config_.deleted_passes_.end()) {
    basic_pass_pm.AddPass(std::move(replace_fetch_with_shadow_output_pass));
  }
  if (!config_.glog_info_disabled()) {
    basic_pass_pm.EnablePrintStatistics();
  }
  if (config_.ir_debug_) {
    basic_pass_pm.EnableIRPrinting(
        std::make_unique<pir::PassManager::IRPrinterOption>(
            ir_printing_conditions, ir_printing_conditions));
  }
  basic_pass_pm.Run(pir_program_.get());
  //----------------------------------------------------------------------------------------------//

  pir_program_ =
      paddle::dialect::PdOpLowerToKernelPass(pir_program_.get(), place_);

  ::pir::PassManager lowered_pm(::pir::IrContext::Instance(), 3);
  auto remove_shadow_feed_pass = ::pir::CreateRemoveShadowFeedPass();
  if (std::find(config_.deleted_passes_.begin(),
                config_.deleted_passes_.end(),
                remove_shadow_feed_pass->name()) ==
      config_.deleted_passes_.end()) {
    remove_shadow_feed_pass->Set("used_for_inference", new bool(true));
    lowered_pm.AddPass(std::move(remove_shadow_feed_pass));
  }
  if (FLAGS_pir_apply_inplace_pass) {
    auto inplace_pass = ::pir::CreateInplacePass();
    if (std::find(config_.deleted_passes_.begin(),
                  config_.deleted_passes_.end(),
                  inplace_pass->name()) == config_.deleted_passes_.end()) {
      lowered_pm.AddPass(std::move(inplace_pass));
    }
  }
  if (!config_.glog_info_disabled()) {
    lowered_pm.EnablePrintStatistics();
  }
  if (config_.ir_debug_) {
    lowered_pm.EnableIRPrinting(
        std::make_unique<pir::PassManager::IRPrinterOption>(
            ir_printing_conditions, ir_printing_conditions));
  }
  lowered_pm.Run(pir_program_.get());

  LOG(INFO) << "======= pir optimization completed =======";
}

bool AnalysisPredictor::SaveOrLoadPirParameters(bool for_save) {
  std::vector<std::pair<std::string, pir::Value>> param_name_var_pairs;
  int feed_idx = 0;
  pir_feeds_.clear();
  for (auto op : pir_program_->block()->ops()) {
    // put pd-op.data and pd-op.fetch into idx2feeds and idx2feeds
    if (op->isa<paddle::dialect::FetchOp>()) {
      int idx = op->attribute("col").dyn_cast<pir::Int32Attribute>().data();
      if (pir_fetches_.size() <= static_cast<size_t>(idx)) {
        pir_fetches_.resize(idx + 1);
        pir_fetches_[idx] = op;
        std::string fetch_name =
            op->attribute("name").dyn_cast<pir::StrAttribute>().AsString();
        idx2fetches_[idx] = fetch_name;
        fetch_name2shapes_[fetch_name] =
            pir::GetShapeFromValue(op->operand_source(0));
      }
    } else if (op->isa<paddle::dialect::DataOp>() ||
               op->isa<paddle::dialect::FeedOp>()) {
      std::string data_name =
          op->attribute("name").dyn_cast<pir::StrAttribute>().AsString();
      if (!load_pir_model_ && for_save) {
        sub_scope_->Var(data_name);
      }
      idx2feeds_[feed_idx] = data_name;
      feed_names_[data_name] = feed_idx;
      feed_idx++;
      pir_feeds_.emplace_back(op);
      feed_name2shapes_[data_name] = pir::GetShapeFromValue(op->result(0));
    }

    if (op->isa<::pir::ParameterOp>()) {
      std::string var_name =
          op->attribute<pir::StrAttribute>("parameter_name").AsString();
      auto var = op->result(0);
      param_name_var_pairs.emplace_back(var_name, var);
    }
  }

  std::sort(param_name_var_pairs.begin(),
            param_name_var_pairs.end(),
            [](const std::pair<std::string, pir::Value> &a,
               const std::pair<std::string, pir::Value> &b) {
              return a.first < b.first;
            });

  std::vector<std::string> param_names, filter_param_names;
  std::vector<pir::Value> vars;
  for (const auto &pair : param_name_var_pairs) {
    param_names.emplace_back(pair.first);
    vars.emplace_back(pair.second);
  }

  size_t len = vars.size();
  std::vector<phi::DenseTensor *> tensor_out;
  for (size_t i = 0; i < len; ++i) {
    auto *var = sub_scope_->FindVar(param_names[i]);
    pir::Value value = vars[i];

    if (var == nullptr) {
      if (value && value.type().isa<pir::DenseTensorType>()) {
        var = sub_scope_->Var(param_names[i]);
        auto *tensor_temp = var->GetMutable<phi::DenseTensor>();
        tensor_temp->Resize(common::make_ddim(pir::GetShapeFromValue(value)));
        phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
        const phi::DeviceContext *dev_ctx = nullptr;
        dev_ctx = pool.Get(phi::CPUPlace());
        pir::Type type_ = pir::GetDataTypeFromValue(value);
        phi::DataType type_data = paddle::dialect::TransToPhiDataType(type_);
        dev_ctx->Alloc(tensor_temp, type_data);
      } else {
        PADDLE_THROW(common::errors::Unavailable(
            "Only support parameter data of type DenseTensor."));
      }
    }
    // we only load params which are persistable(means TRUE parameters))
    auto *tensor_temp = var->GetMutable<phi::DenseTensor>();
    if (value.attribute("persistable")
            .dyn_cast<::pir::BoolAttribute>()
            .data()) {
      tensor_out.push_back(tensor_temp);
      filter_param_names.emplace_back(param_names[i]);
    } else {
      VLOG(3) << param_names[i]
              << " persistable is false, will ignore it when load variables.";
    }
  }
  bool load_separate_params_ = true;
  if (!for_save && config_.model_dir().empty()) {
    // Combine model
    load_separate_params_ = false;
  }

  if (for_save) {
    std::string optimized_params =
        GetOptimizedModelPath() + "/" + "_optimized.pdiparams";
    std::vector<const phi::DenseTensor *> const_tensor_out(tensor_out.begin(),
                                                           tensor_out.end());
    pir::SaveCombineFunction(
        const_tensor_out, param_names, optimized_params, true, false, true);
    LOG(INFO) << "Optimized params saved to " << optimized_params;
  } else {
    if (load_separate_params_) {
      std::string params_dir = config_.model_dir();

      auto process_params = [this, &params_dir, &filter_param_names](
                                size_t start_idx, size_t end_idx) {
        std::vector<phi::DenseTensor *> local_tensor_out;

        for (size_t j = start_idx; j < end_idx; ++j) {
          const auto &param_name = filter_param_names[j];
          std::string param_file = params_dir + "/" + param_name;

          auto *var = sub_scope_->FindVar(param_name);
          VLOG(4) << "persistable variable's name: " << param_name;
          if (var == nullptr) {
            VLOG(4) << "Variable " << param_name << " not found in scope";
            continue;
          }
          auto *tensor_temp = var->GetMutable<phi::DenseTensor>();

          pir::LoadFunction(param_file, -1, {}, false, tensor_temp, place_);

          local_tensor_out.push_back(tensor_temp);
        }

        return local_tensor_out;
      };

      size_t num_threads = 8;
      size_t chunk_size = std::max(static_cast<size_t>(1),
                                   filter_param_names.size() / num_threads);
      num_threads =
          std::min(num_threads, filter_param_names.size() / chunk_size);
      size_t remain_size = filter_param_names.size() % num_threads;
      VLOG(4) << "Start Load with multi-thread: " << num_threads
              << " chund size: " << chunk_size;

      std::vector<std::future<std::vector<phi::DenseTensor *>>> futures;

      for (size_t i = 0; i < num_threads; ++i) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = start_idx + chunk_size;

        futures.push_back(
            std::async(std::launch::async, process_params, start_idx, end_idx));
      }
      if (remain_size > 0) {
        futures.push_back(std::async(std::launch::async,
                                     process_params,
                                     filter_param_names.size() - remain_size,
                                     filter_param_names.size()));
      }

      std::vector<phi::DenseTensor *> tensor_out;
      for (auto &future : futures) {
        auto local_tensor_out = future.get();
        tensor_out.insert(
            tensor_out.end(), local_tensor_out.begin(), local_tensor_out.end());
      }

    } else {
      pir::LoadCombineFunction(config_.params_file(),
                               filter_param_names,
                               &tensor_out,
                               false,
                               place_);
    }
  }
  return true;
}

bool AnalysisPredictor::PreparePirProgram() {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  PADDLE_ENFORCE_EQ(
      pir_program_,
      nullptr,
      common::errors::Fatal("Here, pir_program must be a nullptr!"));

  pir_program_ = std::make_shared<pir::Program>(pir::IrContext::Instance());
  pir::ReadModule(config_.prog_file(), pir_program_.get(), 1 /*pir_version*/);
  if (!SaveOrLoadPirParameters(false)) {
    return false;
  }
  OptimizeInferencePirProgram();
  return true;
}

bool AnalysisPredictor::PrepareProgram(
    const std::shared_ptr<framework::ProgramDesc> &program) {
  if (!program) {
    if (!LoadProgramDesc()) return false;
    // If not cloned, the parameters should be loaded.
    // If config_.ir_optim() is True, parameters is loaded in
    // OptimizeInferenceProgram(), but other persistable variables
    // (like RAW type var) are not created in scope.
    // If config_.ir_optim() is False, parameters is loaded in LoadParameters(),
    // still need to create other persistable variables.
    // So in both case, create persistable variables at first.
    executor_->CreateVariables(*inference_program_, 0, true, sub_scope_);

    // if enable_ir_optim_ is false,
    // the analysis pass(op fuse, graph analysis, trt subgraph, onednn etc) will
    // not be executed.
    model_precision_ =
        paddle::inference::GetModelPrecision(*inference_program_);
#ifdef PADDLE_WITH_TENSORRT
    if (config_.tensorrt_engine_enabled()) {
      inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
          predictor_id_;
      VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
              << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
    }
#endif
    if (config_.use_optimized_model_) {
      LoadParameters();
      ClearExtraParams();
#ifdef PADDLE_WITH_CUDA
      if (config_.use_gpu()) {
        paddle::platform::EmptyCache();
      }
#endif
    } else {
      OptimizeInferenceProgram();
    }
  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
    if (config_.apply_optim_) {
      VLOG(3)
          << "apply_optim is enabled, will call OptimizeInferenceProgram().";
      OptimizeInferenceProgram();
    }
  }

  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);

  if (config_.new_ir_enabled()) {
    PADDLE_ENFORCE_EQ(
        pir_program_,
        nullptr,
        common::errors::Fatal("Here, pir_program must be a nullptr!"));
    pir_program_ = paddle::TranslateLegacyProgramToProgram(*inference_program_);
    OptimizeInferencePirProgram();
  }
  return true;
}

bool AnalysisPredictor::CreateExecutor() {
  executor_ = std::make_unique<paddle::framework::NaiveExecutor>(place_);
  return true;
}

static bool IsPrepareDataOptTargetOp(framework::OpDesc *op) {
  // here is prepare data optimization related bad cases:
  // let's assume an op behind conditional_block and if conditional_block
  // chooses branch 1, the op need to call prepare data. else the op don't need
  // to call prepare data. In running, if predictor chooses branch 2, then
  // optimization takes effect, later issue is followed if predictor chooses
  // branch 1, because the op lost chance to prepare data.
  std::vector<std::string> op_type = {"conditional_block_infer",
                                      "select_input"};
  for (const auto &type : op_type) {
    if (op->Type() == type) {
      return true;
    }
  }
  return false;
}

static void DisablePrepareDataOpt(
    std::shared_ptr<framework::ProgramDesc> inference_program,
    int block,
    bool pre_disable_opt) {
  bool disable_opt = false;
  auto &infer_block = inference_program->Block(block);
  for (auto *op : infer_block.AllOps()) {
    if (disable_opt || pre_disable_opt) {
      op->SetAttr("inference_force_prepare_data", true);
    }
    if (op->HasAttr("sub_block")) {
      int blockID = op->GetBlockAttrId("sub_block");
      DisablePrepareDataOpt(
          inference_program, blockID, disable_opt || pre_disable_opt);
    }
    // disable prepare data if unfriendly op is found
    if (!disable_opt) {
      disable_opt = IsPrepareDataOptTargetOp(op);
    }
  }
}

bool AnalysisPredictor::PrepareExecutor() {
  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          common::errors::PreconditionNotMet(
                              "The sub_scope should not be nullptr."));

  if (config_.new_ir_enabled()) {
    executor_->Prepare(sub_scope_);
  } else {
    DisablePrepareDataOpt(inference_program_, 0, false);
    executor_->Prepare(sub_scope_, *inference_program_, 0);
  }

  if (config_.new_executor_enabled()) {
    framework::interpreter::ExecutionConfig execution_config;
    execution_config.create_local_scope = false;
    execution_config.used_for_inference = true;

    auto input_names = GetInputNames();

    execution_config.skip_gc_vars.insert(input_names.begin(),
                                         input_names.end());
    auto output_names = GetOutputNames();

    execution_config.skip_gc_vars.insert(output_names.begin(),
                                         output_names.end());

    if (config_.new_ir_enabled()) {
      executor_->PrepareInterpreterCore(
          sub_scope_, *pir_program_, execution_config);
    } else {
      executor_->PrepareInterpreterCore(
          sub_scope_, *inference_program_, execution_config);
    }
  }

  if (config_.enable_memory_optim_ && !config_.use_optimized_model_) {
    auto *pass_res_info =
        inference::analysis::PassResultInfoForRuntime::Instance();
    auto reuse_table =
        pass_res_info->Get<std::unordered_map<std::string, std::string>>(
            root_predictor_id_, "memory_optimize_pass");
    executor_->MakeReusePlan(reuse_table);
  }
  return true;
}

void AnalysisPredictor::MkldnnPreSet(const std::vector<PaddleTensor> &inputs) {
#ifdef PADDLE_WITH_DNNL
  std::vector<std::vector<int>> inputs_shape;
  for (const auto &input : inputs) {
    inputs_shape.emplace_back(input.shape);
  }
  MkldnnPreSet(inputs_shape);
#endif
}

void AnalysisPredictor::MkldnnPreSet(
    const std::vector<paddle::Tensor> &inputs) {
#ifdef PADDLE_WITH_DNNL
  std::vector<std::vector<int>> inputs_shape;
  for (const auto &input : inputs) {
    inputs_shape.emplace_back(common::vectorize<int>(input.dims()));
  }
  MkldnnPreSet(inputs_shape);
#endif
}

void AnalysisPredictor::MkldnnPreSet(
    const std::vector<std::vector<int>> &inputs_shape) {
#ifdef PADDLE_WITH_DNNL
  VLOG(2) << "AnalysisPredictor::ZeroCopyRun get_cur_mkldnn_session_id="
          << phi::OneDNNContext::tls().get_cur_mkldnn_session_id();
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0) {
    VLOG(2) << "In mkldnn cache clear mode.";
    phi::OneDNNContext::tls().set_cur_mkldnn_session_id(
        phi::OneDNNContextThreadLocals::kMKLDNNSessionID_CacheClearing);
    // Set current_input_shape for caching dynamic shape.
    std::stringstream ss;
    for (const auto &input_shape : inputs_shape) {
      for (int item : input_shape) {
        ss << item << "-";
      }
    }
    VLOG(2) << "Set input shape=" << ss.str();
    phi::OneDNNContext::tls().set_cur_input_shape_str(ss.str());
  }
  phi::OneDNNContext::tls().set_cur_input_shape_cache_capacity(
      config_.mkldnn_cache_capacity_);

#endif
}

void AnalysisPredictor::MkldnnPostReset() {
#ifdef PADDLE_WITH_DNNL
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0 &&
      static_cast<phi::OneDNNContext *>(
          (&phi::DeviceContextPool::Instance())->Get(phi::CPUPlace()))
              ->GetCachedObjectsNumber() > 0) {
    if (VLOG_IS_ON(2)) {
      auto shape_blob_size =
          static_cast<phi::OneDNNContext *>(
              (&phi::DeviceContextPool::Instance())->Get(phi::CPUPlace()))
              ->GetShapeBlobSize();
      PADDLE_ENFORCE_LE(shape_blob_size,
                        static_cast<size_t>(config_.mkldnn_cache_capacity_),
                        common::errors::InvalidArgument(
                            "Required shape_blob_size should be less than or "
                            "equal to config_.mkldnn_cache_capacity_. "));
    }
    // We cannot reset to the default cache settings
    // as there maybe CopyToCPU method used and oneDNN
    // primitives are used there so cache would grow
  }
#endif
}

bool AnalysisPredictor::Run(const std::vector<PaddleTensor> &inputs,
                            std::vector<PaddleTensor> *output_data,
                            int batch_size) {
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPreSet(inputs);
#endif
  VLOG(3) << "Predictor::predict";
  // set feed variable
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::PreconditionNotMet("The scope should not be nullptr."));
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }
#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif

  if (config_.new_ir_enabled()) {
    ::paddle::framework::RunFeedHooks(*pir_program_, *scope);
  }
  if (config_.shape_range_info_collected()) {
    HookCollectShapeRangeInfo();
  }

  if (config_.new_executor_enabled()) {  // NOLINT
    executor_->RunInterpreterCore();
  } else {
    // Run the inference program
    // if share variables, we need not create variables
    executor_->Run();
  }

  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }

  // All the containers in the scope will be hold in inference, but the
  // operators assume that the container will be reset after each batch.
  // Here is a bugfix, collect all the container variables, and reset then to a
  // bool; the next time, the operator will call MutableData and construct a new
  // container again, so that the container will be empty for each batch.
  if (sub_scope_) {
    tensor_array_batch_cleaner_.CollectNoTensorVars(sub_scope_);
  }
  tensor_array_batch_cleaner_.ResetNoTensorVars();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  phi::dynload::MKL_Free_Buffers();
#endif
  return true;
}

bool AnalysisPredictor::Run(const std::vector<paddle::Tensor> &inputs,
                            std::vector<paddle::Tensor> *outputs) {
  inference::DisplayMemoryInfo(place_, "before run");
  if (private_context_) {
    phi::DeviceContextPool::SetDeviceContexts(&device_contexts_);
    auto &pool = paddle::experimental::DeviceContextPool::Instance();
    pool.SyncDeviceContext(place_);
  }
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPreSet(inputs);
#endif
  VLOG(3) << "predict start";
  // set feed variable
  framework::Scope *scope{nullptr};

  scope = executor_->GetScope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::PreconditionNotMet("The scope should not be nullptr."));
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }
#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif

  if (config_.new_ir_enabled()) {
    ::paddle::framework::RunFeedHooks(*pir_program_, *scope);
  }
  if (config_.shape_range_info_collected()) {
    HookCollectShapeRangeInfo();
  }
#ifdef PADDLE_WITH_XPU
  InferXPUContext *infer_xpu_ctx = nullptr;
  if (config_.use_xpu_) {
    PADDLE_ENFORCE(
        private_context_,
        common::errors::Fatal(
            "Must use private context if run predictor on xpu place."));
    auto *dev_ctxs = reinterpret_cast<const std::map<
        phi::Place,
        std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        this->GetDeviceContexts());
    infer_xpu_ctx =
        static_cast<InferXPUContext *>(dev_ctxs->at(place_).get().get());
    auto *x_context = static_cast<xpu::Context *>(config_.xpu_config_.context);
    if (x_context != nullptr) {
      infer_xpu_ctx->SetXContext(x_context);
    }
    infer_xpu_ctx->SetStream(predictor_stream_);
    infer_xpu_ctx->SetL3Info(config_.xpu_config_.l3_size,
                             config_.xpu_config_.l3_ptr,
                             config_.xpu_config_.l3_autotune_size,
                             place_);
  }
#endif

  if (config_.new_executor_enabled()) {  // NOLINT
    executor_->RunInterpreterCore();
  } else {
    // Run the inference program
    // if share variables, we need not create variables
    executor_->Run();
  }

  inference::DisplayMemoryInfo(place_, "after run");
#ifdef PADDLE_WITH_XPU
  if (config_.use_xpu_ && infer_xpu_ctx != nullptr) {
    infer_xpu_ctx->L3CacheAutotune();
  }
#endif
  // get fetch variable
  if (!GetFetch(outputs, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }

  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(sub_scope_);
  tensor_array_batch_cleaner_.ResetTensorArray();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
  if (private_context_) {
    phi::DeviceContextPool::SetDeviceContexts(nullptr);
  }
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  phi::dynload::MKL_Free_Buffers();
#endif
  return true;
}

bool AnalysisPredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
                                framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (inputs.size() != feeds_.size()) {
    LOG(ERROR) << "wrong feed input size, need " << feeds_.size() << " but get "
               << inputs.size();
    return false;
  }

  // Cache the inputs memory for better concurrency performance.
  feed_tensors_.resize(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    phi::DenseTensor *input = &feed_tensors_[i];
    if (!PaddleTensorToDenseTensor(inputs[i], input, place_)) {
      return false;
    }
    int idx = -1;
    if (config_.specify_input_name_) {
      auto name = inputs[i].name;
      if (feed_names_.find(name) == feed_names_.end()) {
        LOG(ERROR) << "feed names from program do not have name: [" << name
                   << "] from specified input";
      }
      idx = static_cast<int>(feed_names_[name]);
    } else {
      idx = PADDLE_GET_CONST(int, feeds_[i]->GetAttr("col"));
    }
    auto &t = framework::GetVariableTensor(*scope, idx2feeds_[idx]);
    t.ShareDataWith(*input);
    t.set_lod(input->lod());
  }
  return true;
}

bool AnalysisPredictor::SetFeed(const std::vector<paddle::Tensor> &inputs,
                                framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (load_pir_model_) {
    PADDLE_ENFORCE_EQ(inputs.size(),
                      pir_feeds_.size(),
                      common::errors::InvalidArgument(
                          "wrong feed input size, need %d but get %d.",
                          pir_feeds_.size(),
                          inputs.size()));
  } else {
    PADDLE_ENFORCE_EQ(inputs.size(),
                      feeds_.size(),
                      common::errors::InvalidArgument(
                          "wrong feed input size, need %d but get %d.",
                          feeds_.size(),
                          inputs.size()));
  }

  for (const auto &input : inputs) {
    PADDLE_ENFORCE_EQ(input.defined(),
                      true,
                      common::errors::InvalidArgument(
                          "The input Tensor expected to be defined."));
    PADDLE_ENFORCE_EQ(
        input.is_dense_tensor(),
        true,
        common::errors::InvalidArgument(
            "The input Tensor expected to be type of dense tensor."));
  }

  if (std::all_of(inputs.cbegin(), inputs.cend(), [&](const paddle::Tensor &t) {
        return !t.name().empty() && feed_names_.count(t.name());
      })) {
    for (const auto &input : inputs) {
      auto &t = framework::GetVariableTensor(*scope, input.name());
      t.ShareDataWith(
          *std::dynamic_pointer_cast<phi::DenseTensor>(input.impl()));
      t.set_lod(
          std::dynamic_pointer_cast<phi::DenseTensor>(input.impl())->lod());
    }
  } else {
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto &t = framework::GetVariableTensor(*scope, idx2feeds_[i]);
      t.ShareDataWith(
          *std::dynamic_pointer_cast<phi::DenseTensor>(inputs[i].impl()));
      t.set_lod(
          std::dynamic_pointer_cast<phi::DenseTensor>(inputs[i].impl())->lod());
    }
  }
  return true;
}

template <typename T>
void AnalysisPredictor::GetFetchOne(const phi::DenseTensor &fetch,
                                    PaddleTensor *output) {
  // set shape.
  auto shape = common::vectorize(fetch.dims());
  output->shape.assign(shape.begin(), shape.end());
  // set data.
  int num_elems = inference::VecReduceToInt(shape);
  output->data.Resize(num_elems * sizeof(T));
  paddle::memory::Copy(phi::CPUPlace(),
                       output->data.data(),
                       fetch.place(),
                       fetch.data<T>(),
                       num_elems * sizeof(T));
  // set lod
  output->lod.clear();
  for (auto &level : fetch.lod()) {
    output->lod.emplace_back(level.begin(), level.end());
  }
}

bool AnalysisPredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                 framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetches_.size());
  for (size_t i = 0; i < fetches_.size(); ++i) {
    int idx = PADDLE_GET_CONST(int, fetches_[i]->GetAttr("col"));
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(idx),
        i,
        common::errors::InvalidArgument(
            "Fetch op's col attr(%d) should be equal to the index(%d)",
            idx,
            i));
    auto &t = framework::GetVariableTensor(*scope, idx2fetches_[idx]);
    auto type = framework::TransToProtoVarType(t.dtype());
    auto output = &(outputs->at(i));
    output->name = fetches_[idx]->Input("X")[0];
    if (type == framework::proto::VarType::FP32) {
      GetFetchOne<float>(t, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == framework::proto::VarType::INT64) {
      GetFetchOne<int64_t>(t, output);
      output->dtype = PaddleDType::INT64;
    } else if (type == framework::proto::VarType::INT32) {
      GetFetchOne<int32_t>(t, output);
      output->dtype = PaddleDType::INT32;
    } else if (type == framework::proto::VarType::FP16) {
      GetFetchOne<float16>(t, output);
      output->dtype = PaddleDType::FLOAT16;
    } else if (type == framework::proto::VarType::BF16) {
      GetFetchOne<bfloat16>(t, output);
      output->dtype = PaddleDType::BFLOAT16;
    } else {
      LOG(ERROR)
          << "unknown type, only support float32, float16, bfloat16, int64 and "
             "int32 now.";
    }
  }
  return true;
}

bool AnalysisPredictor::GetFetch(std::vector<paddle::Tensor> *outputs,
                                 framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  if (load_pir_model_) {
    outputs->resize(pir_fetches_.size());
    for (size_t i = 0; i < pir_fetches_.size(); ++i) {
      auto const &name = idx2fetches_[i];
      auto &t = framework::GetVariableTensor(*scope, name);
      (*outputs)[i] =
          paddle::Tensor(std::make_shared<phi::DenseTensor>(t), name);
    }
    return true;
  }
  outputs->resize(fetches_.size());
  for (size_t i = 0; i < fetches_.size(); ++i) {
    auto const &name = idx2fetches_[i];
    auto &t = framework::GetVariableTensor(*scope, name);
    (*outputs)[i] = paddle::Tensor(std::make_shared<phi::DenseTensor>(t), name);
  }
  return true;
}

void AnalysisPredictor::PrepareArgument() {
  VLOG(3) << "AnalysisPredictor::PrepareArgument";
  // Init std::unique_ptr argument_.
  argument_ = std::make_unique<Argument>();
  argument_->SetUseGPU(config_.use_gpu());
  argument_->SetUseCutlass(config_.use_cutlass_);
  argument_->SetUseFcPadding(config_.use_fc_padding());
  argument_->SetGPUDeviceId(config_.gpu_device_id());
  argument_->SetEnableIrOptim(config_.enable_ir_optim_);
  argument_->SetEnableMemoryOptim(config_.enable_memory_optim());
  argument_->SetModelFromMemory(config_.model_from_memory_);
  argument_->SetUsePIR(config_.new_ir_enabled());
  // Analyze inference_program
  argument_->SetPredictorID(predictor_id_);
  argument_->SetRootPredictorID(root_predictor_id_);
  argument_->SetSaveOptimizedModel(config_.save_optimized_model_);
  argument_->SetOptimCacheDir(config_.opt_cache_dir_);
  if (!config_.model_dir().empty()) {
    argument_->SetModelDir(config_.model_dir());
  } else {
    PADDLE_ENFORCE_EQ(config_.prog_file().empty(),
                      false,
                      common::errors::PreconditionNotMet(
                          "Either model_dir or prog_file should be set."));

    argument_->SetModelProgramPath(config_.prog_file());
    argument_->SetModelParamsPath(config_.params_file());
  }
  argument_->SetOptimizedModelSavePath(GetOptimizedModelPath());
  // For JITLayer
  argument_->SetSkipLoadParams(config_.skip_load_params_);

  argument_->SetTensorRtPrecisionMode(static_cast<int>(
      paddle::ConvertPrecision(config_.tensorrt_precision_mode_)));
  argument_->SetTensorRtUseOSS(config_.trt_use_varseqlen_);
  argument_->SetTensorRtWithInterleaved(config_.trt_with_interleaved_);
  argument_->SetTensorRtTransformerPosid(config_.tensorrt_transformer_posid_);
  argument_->SetTensorRtTransformerMaskid(config_.tensorrt_transformer_maskid_);
  argument_->SetMinInputShape(config_.min_input_shape_);
  argument_->SetMaxInputShape(config_.max_input_shape_);
  argument_->SetOptimInputShape(config_.optim_input_shape_);
  argument_->SetTensorRtTunedDynamicShape(
      config_.tuned_tensorrt_dynamic_shape());
  argument_->SetUseTensorRT(false);
  if (config_.use_gpu() && config_.tensorrt_engine_enabled()) {
    LOG(INFO) << "TensorRT subgraph engine is enabled";
    argument_->SetUseTensorRT(true);
    argument_->SetTensorRtWorkspaceSize(config_.tensorrt_workspace_size_);
    argument_->SetTensorRtMaxBatchSize(config_.tensorrt_max_batchsize_);
    argument_->SetTensorRtMinSubgraphSize(config_.tensorrt_min_subgraph_size_);
    argument_->SetTRTMarkOutput(config_.trt_mark_output_);
    argument_->SetTRTOutputTensorNames(config_.trt_output_tensor_names_);
    argument_->SetTRTParameterRunFp16(config_.trt_parameters_run_fp16_);
    argument_->SetTRTParameterRunInt8(config_.trt_parameters_run_int8_);
    argument_->SetTRTParameterRunBfp16(config_.trt_parameters_run_bfp16_);
    argument_->SetTensorRtDisabledOPs(config_.trt_disabled_ops_);
    argument_->SetTRTExcludeVarNames(config_.trt_exclude_var_names_);
    argument_->SetTRTForbidDynamicOp(config_.trt_forbid_dynamic_op_);

    argument_->SetTensorRtUseDLA(config_.trt_use_dla_);
    argument_->SetTensorRtDLACore(config_.trt_dla_core_);
    argument_->SetTensorRtUseStaticEngine(config_.trt_use_static_engine_);

    argument_->SetTensorRtUseCalibMode(config_.trt_use_calib_mode_);
    argument_->SetTensorRtUseCudaGraph(config_.trt_use_cuda_graph_);
    argument_->SetCloseTrtPluginFp16(config_.disable_trt_plugin_fp16_);
    argument_->SetTensorRtShapeRangeInfoPath(config_.shape_range_info_path());
    argument_->SetTensorRtAllowBuildAtRuntime(
        config_.trt_allow_build_at_runtime());
    argument_->SetTensorRtUseInspector(config_.trt_use_inspector_);
    argument_->SetTensorRtInspectorSerialize(config_.trt_inspector_serialize_);
    argument_->SetTensorRtUseExplicitQuantization(
        config_.trt_use_explicit_quantization_);
    argument_->SetTrtEngineMemorySharing(config_.trt_engine_memory_sharing());
    argument_->SetTensorRtOptimizationLevel(config_.trt_optimization_level_);
    argument_->SetTensorRtOpsRunFloat(config_.trt_ops_run_float_);
  }

  argument_->SetUseXpu(config_.use_xpu_);
#ifdef PADDLE_WITH_OPENVINO
  argument_->SetUseOpenVINO(config_.use_openvino_);
  argument_->SetCpuMathLibraryNumThreads(config_.cpu_math_library_num_threads_);
  argument_->SetOpenvinoInferencePrecision(static_cast<int>(
      paddle::ConvertPrecision(config_.openvino_inference_precision_)));
#endif
#ifdef PADDLE_WITH_IPU
  argument_->SetUseIpu(config_.use_ipu());
  argument_->SetIpuDeviceNum(config_.ipu_device_num());
  argument_->SetIpuMicroBatchSize(config_.ipu_micro_batch_size_);
  argument_->SetIpuEnablePipelining(config_.ipu_enable_pipelining_);
  argument_->SetIpuBatchesPerStep(config_.ipu_batches_per_step_);
  argument_->SetIpuEnableFp16(config_.ipu_enable_fp16_);
  argument_->SetIpuReplicaNum(config_.ipu_replica_num_);
  argument_->SetIpuAvailableMemoryProportion(
      config_.ipu_available_memory_proportion_);
  argument_->SetIpuEnableHalfPartial(config_.ipu_enable_half_partial_);
  argument_->SetIpuEnableModelRuntimeExecutor(
      config_.ipu_enable_model_runtime_executor_);
  argument_->SetIpuCustomOpsInfo(config_.ipu_custom_ops_info_);
  argument_->SetIpuCustomPatterns(config_.ipu_custom_patterns_);
#endif

  if (config_.mkldnn_enabled() && !config_.use_gpu()) {
    LOG(INFO) << "MKLDNN is enabled";
    argument_->SetMKLDNNEnabledOpTypes(config_.mkldnn_enabled_op_types_);
  }

  if (config_.cinn_enabled()) {
    argument_->SetUseCinnCompiler(true);
  }

#ifdef PADDLE_WITH_DNNL
  if (config_.mkldnn_bfloat16_enabled()) {
    LOG(INFO) << "Bfloat16 is enabled";
    argument_->SetBfloat16EnabledOpTypes(config_.bfloat16_enabled_op_types_);
  }

  if (config_.mkldnn_int8_enabled()) {
    LOG(INFO) << "Int8 is enabled";
    argument_->SetQuantizeEnabledOpTypes(config_.quantize_enabled_op_types_);
    argument_->SetQuantizeExcludedOpIds(config_.quantize_excluded_op_ids_);
    argument_->SetQuantVarScales({});
  }
#endif

  argument_->SetUseCustomDevice(config_.use_custom_device());
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (config_.use_custom_device()) {
    LOG(INFO) << "CustomDevice is enabled";
    argument_->SetCustomDeviceType(config_.custom_device_type());
    argument_->SetCustomDeviceId(config_.custom_device_id());
  }
#endif

  argument_->SetUseXpu(config_.use_xpu_);
  argument_->SetXpuDeviceId(config_.xpu_config_.device_id);
  argument_->SetXpuL3Size(config_.xpu_config_.l3_size);
  argument_->SetXpuL3Ptr(config_.xpu_config_.l3_ptr);
  argument_->SetXpuL3AutotuneSize(config_.xpu_config_.l3_autotune_size);
  argument_->SetXpuContextGmSize(config_.xpu_config_.context_gm_size);
  argument_->SetXpuContext(config_.xpu_config_.context);
  argument_->SetXpuStream(config_.xpu_config_.stream);
  argument_->SetXpuConvAutotuneLevel(config_.xpu_config_.conv_autotune_level);
  argument_->SetXpuConvAutotuneFile(config_.xpu_config_.conv_autotune_file);
  argument_->SetXpuConvAutotuneFileWriteback(
      config_.xpu_config_.conv_autotune_file_writeback);
  argument_->SetXpuFcAutotuneLevel(config_.xpu_config_.fc_autotune_level);
  argument_->SetXpuFcAutotuneFile(config_.xpu_config_.fc_autotune_file);
  argument_->SetXpuFcAutotuneFileWriteback(
      config_.xpu_config_.fc_autotune_file_writeback);
  argument_->SetXpuGemmComputePrecision(
      config_.xpu_config_.gemm_compute_precision);
  argument_->SetXpuQuantPostDynamicWeightMethods(
      config_.xpu_config_.quant_post_dynamic_weight_methods);
  argument_->SetXpuTransformerSoftmaxOptimizeLevel(
      config_.xpu_config_.transformer_softmax_optimize_level);
  argument_->SetXpuTransformerEncoderAdaptiveSeqlen(
      config_.xpu_config_.transformer_encoder_adaptive_seqlen);
  argument_->SetXpuQuantPostStaticGeluOutThreshold(
      config_.xpu_config_.quant_post_static_gelu_out_threshold);
  argument_->SetXpuQuantPostDynamicActivationMethod(
      config_.xpu_config_.quant_post_dynamic_activation_method);
  argument_->SetXpuQuantPostDynamicWeightPrecision(
      config_.xpu_config_.quant_post_dynamic_weight_precision);
  argument_->SetXpuQuantPostDynamicOpTypes(
      config_.xpu_config_.quant_post_dynamic_op_types);

  auto *pass_builder = config_.pass_builder();
  // TODO(inference): Need to reconstruct the pass_builder, pass should be
  // processed in a single
  if (model_precision_ != phi::DataType::FLOAT32) {
    LOG(INFO) << "Model is mixed precision type with " << model_precision_
              << ", we will use a new PassStrategy. Note that only GPU/XPU "
                 "backend is supported for now.";
    if (!config_.cinn_enabled()) {
      const auto &deleted_passes = pass_builder->GetAllDeletedPasses();
      if (config_.tensorrt_engine_enabled()) {
        pass_builder->ClearPasses();
        for (const auto &pass : kTrtLowerPrecisionPasses) {
          if (deleted_passes.count(pass)) continue;
          pass_builder->AppendPass(pass);
        }
      } else if (config_.use_gpu()) {
        pass_builder->ClearPasses();
        for (const auto &pass : kGpuLowerPrecisionPasses) {
          if (deleted_passes.count(pass)) continue;
          pass_builder->AppendPass(pass);
        }
      } else if (config_.use_xpu()) {  // NOLINT
        // All passes support fp16. Not reset pass_builder.
      } else if (config_.use_custom_device()) {
        // All passes support fp16. Not reset pass_builder.
      } else {
        pass_builder->ClearPasses();
      }
    }
  }

  if (!config_.ir_optim()) {
    argument_->SetEnableIrOptim(false);
    if (config_.enable_gpu_mixed_ &&
        model_precision_ == phi::DataType::FLOAT32) {
      argument_->SetEnableIrOptim(true);
      pass_builder->ClearPasses();
      if (!config_.new_ir_enabled()) {
        pass_builder->AppendPass("map_op_to_another_pass");
        pass_builder->AppendPass("simplify_with_basic_ops_pass");
        pass_builder->AppendPass("is_test_pass");
        pass_builder->AppendPass("constant_folding_pass");
        pass_builder->AppendPass("auto_mixed_precision_pass");
        pass_builder->AppendPass("inplace_op_var_pass");
      }
      LOG(INFO) << "This model run in GPU mixed precision mode with no ir "
                   "optimization.";
      if (config_.ir_debug_) {
        pass_builder->TurnOnDebug();
      }
    } else {
      LOG(INFO)
          << "Ir optimization is turned off, no ir pass will be executed.";
    }
  } else {
    if (config_.ir_debug_) {
      pass_builder->TurnOnDebug();
    }
    if (config_.enable_gpu_mixed_) {
      LOG(INFO) << "This model run in GPU mixed precision mode.";
    }
  }

  argument_->SetEnableCustomDeviceMixed(config_.enable_custom_device_mixed());
  if (config_.enable_custom_device_mixed_) {
    argument_->SetEnableIrOptim(true);
    pass_builder->AppendPass("auto_mixed_precision_pass");
    LOG(INFO) << "This model run in Custom Device mixed precision mode.";
  }

  argument_->SetDisableLogs(config_.glog_info_disabled());
  argument_->SetIrAnalysisPasses(pass_builder->AllPasses());
  argument_->SetAnalysisPasses(pass_builder->AnalysisPasses());
  argument_->SetScopeNotOwned(scope_.get());

  // mixed precision.
  argument_->SetModelPrecision(static_cast<int>(model_precision_));
  argument_->SetMixedBlackList(config_.mixed_black_list_);
  argument_->SetMixedWhiteList(config_.mixed_white_list_);
  argument_->SetEnableGPUMixed(config_.enable_gpu_mixed_);
  argument_->SetMixedPrecisionMode(static_cast<int>(
      paddle::ConvertPrecision(config_.mixed_precision_mode_)));
  argument_->SetEnableLowPrecisionIO(config_.enable_low_precision_io_);
}

// NOTE All the members in AnalysisConfig should be copied to Argument.
void AnalysisPredictor::OptimizeInferenceProgram() {
  PrepareArgument();
  Analyzer().Run(argument_.get());
  PADDLE_ENFORCE_EQ(
      argument_->scope_valid(),
      true,
      common::errors::InvalidArgument("The argument scope should be valid."));
  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((argument_.get()), ir_analyzed_program);
  inference_program_.reset(
      new framework::ProgramDesc(argument_->ir_analyzed_program()),
      [](framework::ProgramDesc *prog) {
// Note, please do NOT use any member variables, because member variables may
// have been destructed in multiple threads.
#ifdef PADDLE_WITH_TENSORRT
        auto &block = prog->Block(0);
        for (auto &op_desc : block.AllOps()) {
          if (op_desc->Type() == "tensorrt_engine") {
            std::string engine_key =
                PADDLE_GET_CONST(std::string, op_desc->GetAttr("engine_key"));
            int engine_predictor_id =
                PADDLE_GET_CONST(int, op_desc->GetAttr("predictor_id"));
            std::string engine_name =
                engine_key + std::to_string(engine_predictor_id);
            if (paddle::inference::Singleton<
                    inference::tensorrt::TRTEngineManager>::Global()
                    .Has(engine_name)) {
              paddle::inference::Singleton<
                  inference::tensorrt::TRTEngineManager>::Global()
                  .DeleteKey(engine_name);
            }
          }
        }
#endif
        delete prog;
      });

#if defined(PADDLE_WITH_TESTING)
  fusion_statis_ = *argument_->fusion_statis_ptr();
#endif
  // The argument take a lot of storage,
  // when the predictor settings are complete, we release these stores.
#if defined(_WIN32)
  argument_->PartiallyRelease();
#else
  if (config_.mkldnn_enabled() ||
      (config_.tensorrt_engine_enabled() &&
       config_.tensorrt_precision_mode_ ==
           AnalysisConfig::Precision::kInt8)) {  // NOLINT
    argument_->PartiallyRelease();
  } else {
    argument_.reset(nullptr);
  }
#endif
  LOG(INFO) << "======= ir optimization completed =======";
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
    const AnalysisConfig &config) {
  PADDLE_ENFORCE_EQ(
      config.is_valid(),
      true,
      common::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));

  // Register custom operators compiled by the user.
  // This function can only be executed once per process.
  static std::once_flag custom_operators_registered;
  std::call_once(custom_operators_registered, [config]() {
    inference::RegisterAllCustomOperator(config.new_ir_enabled());
  });

  auto SetGflags = [](const AnalysisConfig &config) {
    auto SetGflag = [](const char *name, const char *value) {
      bool success = paddle::flags::SetFlagValue(name, value);
      PADDLE_ENFORCE_EQ(
          success,
          true,
          common::errors::InvalidArgument(
              "Fail to set gflag: %s, please make sure the gflag exists.",
              name));
      VLOG(3) << "set gflag: --" << name << "=" << value;
    };
    // TODO(NHZlX): Should add the link to the doc of
    // paddle_infer::CreatePredictor<paddle_infer::Config>
    if (config.glog_info_disabled()) {
      FLAGS_logtostderr = true;
      FLAGS_minloglevel = 2;  // GLOG_ERROR
    }

    if (config.use_gpu()) {
      static std::once_flag gflags_initialized;
      static bool process_level_allocator_enabled;

      std::call_once(gflags_initialized, [&]() {
        PADDLE_ENFORCE_GE(
            config.memory_pool_init_size_mb(),
            0.f,
            common::errors::InvalidArgument(
                "The size of memory pool should be greater than 0."));
        PADDLE_ENFORCE_GE(config.gpu_device_id(),
                          0,
                          common::errors::InvalidArgument(
                              "Invalid device id (%d). The device id should be "
                              "greater than 0.",
                              config.gpu_device_id()));

        float fraction_of_gpu_memory = config.fraction_of_gpu_memory_for_pool();
        if (fraction_of_gpu_memory > 0.95f) {
          LOG(ERROR)
              << "Allocate too much memory for the GPU memory pool, assigned "
              << config.memory_pool_init_size_mb() << " MB";
          LOG(ERROR) << "Try to shrink the value by setting "
                        "AnalysisConfig::EnableUseGpu(...)";
        }
        if (fraction_of_gpu_memory >= 0.0f || fraction_of_gpu_memory <= 0.95f) {
          std::string value = std::to_string(fraction_of_gpu_memory);
          SetGflag("fraction_of_gpu_memory_to_use", value.data());
        }

        // TODO(Shixiaowei02): Add a mandatory scheme to use the thread local
        // allocator when multi-stream is enabled.
        if (config.thread_local_stream_enabled()) {
          SetGflag("allocator_strategy", "thread_local");
          process_level_allocator_enabled = false;
        } else {
          process_level_allocator_enabled = true;
        }

        // for inference, the following default values are better.
        if (std::getenv("FLAGS_conv_workspace_size_limit") == nullptr) {
          SetGflag("conv_workspace_size_limit", "32");
        }
        if (std::getenv("FLAGS_initial_cpu_memory_in_mb") == nullptr) {
          SetGflag("initial_cpu_memory_in_mb", "0");
        }
        if (std::getenv("FLAGS_cache_inference_while_scope") == nullptr) {
          SetGflag("cache_inference_while_scope", "1");
        }
      });

      if (config.thread_local_stream_enabled() &&
          process_level_allocator_enabled) {
        PADDLE_THROW(common::errors::Fatal(
            "When binding threads and streams, the use of "
            "process-level allocators will result in undefined result "
            "errors due to memory asynchronous operations."
            "The thread and stream binding configuration of all "
            "predictors should be the same in a single process."));
      }
    }
  };
  SetGflags(config);

  VLOG(3) << "create AnalysisPredictor";

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  // Each config can only be used for one predictor.
  config.SetInValid();
  auto predictor_p = dynamic_cast<AnalysisPredictor *>(predictor.get());

#ifdef PADDLE_WITH_TENSORRT
  paddle::framework::ir::patterns::KeyCounter::Instance().CleanCounter();
#endif

  if (!predictor_p->Init(nullptr)) {
    return nullptr;
  }

  return predictor;
}

void AnalysisPredictor::PrepareFeedFetch() {
  if (load_pir_model_) {
    return;
  }
  PADDLE_ENFORCE_NOT_NULL(
      sub_scope_,
      common::errors::InvalidArgument("The sub_scope should not be nullptr."));
  CreateFeedFetchVar(sub_scope_);
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == framework::kFeedOpType) {
      int idx = PADDLE_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
      idx2feeds_[idx] = op->Output("Out")[0];
    } else if (op->Type() == framework::kFetchOpType) {
      int idx = PADDLE_GET_CONST(int, op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
      idx2fetches_[idx] = op->Input("X")[0];
    }
  }
}

void AnalysisPredictor::CreateFeedFetchVar(framework::Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::InvalidArgument("The scope should not be nullptr."));
  auto *var = scope->Var(framework::kFeedOpType);
  var->GetMutable<framework::FeedList>();
  var = scope->Var(framework::kFetchOpType);
  var->GetMutable<framework::FetchList>();
}

std::vector<std::string> AnalysisPredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto &item : idx2feeds_) {
    input_names.push_back(item.second);
  }
  return input_names;
}

std::map<std::string, std::vector<int64_t>>
AnalysisPredictor::GetInputTensorShape() {
  if (load_pir_model_) {
    return feed_name2shapes_;
  }
  std::map<std::string, std::vector<int64_t>> input_shapes;
  std::vector<std::string> names = GetInputNames();
  for (std::string const &name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::PreconditionNotMet("Input %s does not exist.", name));
    input_shapes[name] = var->GetShape();
  }
  return input_shapes;
}

std::map<std::string, paddle_infer::DataType>
AnalysisPredictor::GetInputTypes() {
  std::map<std::string, paddle_infer::DataType> input_type;
  std::vector<std::string> names = GetInputNames();
  for (const auto &name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::PreconditionNotMet(
            "Input %s does not exist inference_program_.", name));
    auto dtype = var->GetDataType();
    if (dtype == paddle::framework::proto::VarType::FP32) {
      input_type[name] = paddle_infer::DataType::FLOAT32;
    } else if (dtype == paddle::framework::proto::VarType::FP16) {
      input_type[name] = paddle_infer::DataType::FLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::BF16) {
      input_type[name] = paddle_infer::DataType::BFLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::INT64) {
      input_type[name] = paddle_infer::DataType::INT64;
    } else if (dtype == paddle::framework::proto::VarType::INT32) {
      input_type[name] = paddle_infer::DataType::INT32;
    } else if (dtype == paddle::framework::proto::VarType::UINT8) {
      input_type[name] = paddle_infer::DataType::UINT8;
    } else if (dtype == paddle::framework::proto::VarType::INT8) {
      input_type[name] = paddle_infer::DataType::INT8;
    } else if (dtype == paddle::framework::proto::VarType::FP64) {
      input_type[name] = paddle_infer::DataType::FLOAT64;
    } else if (dtype == paddle::framework::proto::VarType::BOOL) {
      input_type[name] = paddle_infer::DataType::BOOL;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported data type `%s` when get input dtype ", dtype));
    }
  }
  return input_type;
}

std::vector<std::string> AnalysisPredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto &item : idx2fetches_) {
    output_names.push_back(item.second);
  }
  return output_names;
}

std::map<std::string, std::vector<int64_t>>
AnalysisPredictor::GetOutputTensorShape() {
  if (load_pir_model_) {
    return fetch_name2shapes_;
  }
  std::map<std::string, std::vector<int64_t>> output_shapes;
  std::vector<std::string> names = GetOutputNames();
  for (std::string const &name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::PreconditionNotMet("Output %s does not exist.", name));
    output_shapes[name] = var->GetShape();
  }
  return output_shapes;
}

std::map<std::string, paddle_infer::DataType>
AnalysisPredictor::GetOutputTypes() {
  std::map<std::string, paddle_infer::DataType> output_type;
  std::vector<std::string> names = GetOutputNames();
  for (const auto &name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::PreconditionNotMet(
            "Output %s does not exist inference_program_.", name));
    auto dtype = var->GetDataType();
    if (dtype == paddle::framework::proto::VarType::FP32) {
      output_type[name] = paddle_infer::DataType::FLOAT32;
    } else if (dtype == paddle::framework::proto::VarType::FP16) {
      output_type[name] = paddle_infer::DataType::FLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::BF16) {
      output_type[name] = paddle_infer::DataType::BFLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::INT64) {
      output_type[name] = paddle_infer::DataType::INT64;
    } else if (dtype == paddle::framework::proto::VarType::INT32) {
      output_type[name] = paddle_infer::DataType::INT32;
    } else if (dtype == paddle::framework::proto::VarType::UINT8) {
      output_type[name] = paddle_infer::DataType::UINT8;
    } else if (dtype == paddle::framework::proto::VarType::INT8) {
      output_type[name] = paddle_infer::DataType::INT8;
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported data type `%s` when get output dtype ", dtype));
    }
  }
  return output_type;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetInputTensor(
    const std::string &name) {
  framework::Scope *scope = nullptr;
  scope = executor_->GetScope();
  PADDLE_ENFORCE_NOT_NULL(
      scope->FindVar(name),
      common::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the executor.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(
      static_cast<void *>(scope), this->GetDeviceContexts()));
  res->input_or_output_ = true;
  res->SetName(name);
  if (phi::is_cpu_place(place_)) {  // NOLINT
    res->SetPlace(PaddlePlace::kCPU);
  } else if (phi::is_ipu_place(place_)) {
    // Currently, IPUPlace's tensor copy between cpu and ipu has been set in
    // IpuBackend.
    res->SetPlace(PaddlePlace::kCPU);
  } else if (phi::is_xpu_place(place_)) {
    auto xpu_place = place_;
    res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
  } else if (phi::is_custom_place(place_)) {
    auto custom_place = place_;
    res->SetPlace(PaddlePlace::kCUSTOM,
                  custom_place.GetDeviceId(),
                  custom_place.GetDeviceType());
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  framework::Scope *scope;  // NOLINT
  scope = executor_->GetScope();
  PADDLE_ENFORCE_NOT_NULL(
      scope->FindVar(name),
      common::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the executor.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(
      static_cast<void *>(scope), this->GetDeviceContexts()));
  res->input_or_output_ = false;
  res->SetName(name);
  if (phi::is_cpu_place(place_)) {  // NOLINT
    res->SetPlace(PaddlePlace::kCPU);
  } else if (phi::is_ipu_place(place_)) {
    // Currently, IPUPlace's tensor copy between cpu and ipu has been set in
    // IpuBackend.
    res->SetPlace(PaddlePlace::kCPU);
  } else if (phi::is_xpu_place(place_)) {
    auto xpu_place = place_;
    res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
  } else if (phi::is_custom_place(place_)) {
    auto custom_place = place_;
    res->SetPlace(PaddlePlace::kCUSTOM,
                  custom_place.GetDeviceId(),
                  custom_place.GetDeviceType());
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

bool AnalysisPredictor::ZeroCopyRun(bool switch_stream) {
  inference::DisplayMemoryInfo(place_, "before run");
  if (private_context_) {
    phi::DeviceContextPool::SetDeviceContexts(&device_contexts_);
    auto &pool = paddle::experimental::DeviceContextPool::Instance();
    pool.SyncDeviceContext(place_);
  }
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) {
    std::vector<std::vector<int>> shape_vector;
    auto names = GetInputNames();
    for (auto &name : names) {
      auto in_tensor = GetInputTensor(name);
      shape_vector.emplace_back(in_tensor->shape());
    }
    MkldnnPreSet(shape_vector);
  }
#endif

#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif

  if (config_.new_ir_enabled()) {
    auto *scope = sub_scope_ ? sub_scope_ : scope_.get();
    if (scope != nullptr) {
      ::paddle::framework::RunFeedHooks(*pir_program_, *scope);
    }
  }
  if (config_.shape_range_info_collected()) {
    HookCollectShapeRangeInfo();
  }
#ifdef PADDLE_WITH_XPU
  InferXPUContext *infer_xpu_ctx = nullptr;
  if (config_.use_xpu_) {
    PADDLE_ENFORCE(
        private_context_,
        common::errors::Fatal(
            "Must use private context if run predictor on xpu place."));
    auto *dev_ctxs = reinterpret_cast<const std::map<
        phi::Place,
        std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        this->GetDeviceContexts());
    infer_xpu_ctx =
        static_cast<InferXPUContext *>(dev_ctxs->at(place_).get().get());
    auto *x_context = static_cast<xpu::Context *>(config_.xpu_config_.context);
    if (x_context != nullptr) {
      infer_xpu_ctx->SetXContext(x_context);
    }
    infer_xpu_ctx->SetStream(predictor_stream_);
    infer_xpu_ctx->SetL3Info(config_.xpu_config_.l3_size,
                             config_.xpu_config_.l3_ptr,
                             config_.xpu_config_.l3_autotune_size,
                             place_);
  }
#endif

  if (config_.new_executor_enabled()) {  // NOLINT
    executor_->RunInterpreterCore({}, false, switch_stream);
  } else {
    executor_->Run();
  }
  inference::DisplayMemoryInfo(place_, "after run");

#ifdef PADDLE_WITH_XPU
  if (config_.use_xpu_ && infer_xpu_ctx != nullptr &&
      config_.xpu_config_.l3_autotune_size > 0) {
    static std::once_flag set_output_holder_map;
    std::call_once(set_output_holder_map, [&]() {
      auto scope = executor_->GetScope();
      VLOG(4) << "Set output tensor's holder.";
      for (auto name : GetOutputNames()) {
        auto out_tensor = scope->FindVar(name)->GetMutable<phi::DenseTensor>();

        phi::Allocation *holder =
            reinterpret_cast<phi::DenseTensor *>(out_tensor)->Holder().get();
        infer_xpu_ctx->SetOutHolder(holder);
      }
    });
    infer_xpu_ctx->L3CacheAutotune();
  }
#endif

  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(sub_scope_);
  tensor_array_batch_cleaner_.ResetTensorArray();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
  if (private_context_) {
    phi::DeviceContextPool::SetDeviceContexts(nullptr);
  }
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  phi::dynload::MKL_Free_Buffers();
#endif
  return true;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
bool AnalysisPredictor::ExpRunWithExternalStream(const gpuStream_t stream) {
  if (!private_context_) {
    PADDLE_THROW(common::errors::Fatal(
        "Please use config.SetExecStream to init gpu resources, and then we "
        "will bind gpu resources to execution stream."));
  }
  bool switch_stream = false;
  if (stream != predictor_stream_) {
#ifdef PADDLE_WITH_HIP
    hipStreamSynchronize(static_cast<gpuStream_t>(predictor_stream_));
#else
    cudaStreamSynchronize(static_cast<gpuStream_t>(predictor_stream_));
#endif
    ResourceManager::Instance().GpuResourceSwitchStream(predictor_stream_,
                                                        stream);
    predictor_stream_ = stream;

    auto *dev_ctxs = const_cast<
        std::map<phi::Place,
                 std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        reinterpret_cast<const std::map<
            phi::Place,
            std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
            this->GetDeviceContexts()));

    dev_ctxs->erase(place_);
    dev_ctxs->emplace(
        place_, std::async(std::launch::deferred, [=] {
          auto *gpu_resource =
              ResourceManager::Instance().GetGPUResource(predictor_stream_);
          auto *gpu_context = new InferGPUContext(place_);
          UpdatePrivateDeviceContext(gpu_context, gpu_resource, place_);
          return std::unique_ptr<phi::DeviceContext>(gpu_context);
        }));
    switch_stream = true;
  }
  return ZeroCopyRun(switch_stream);
}
#endif

void AnalysisPredictor::HookCollectShapeRangeInfo() {
  if (config_.new_executor_enabled()) {
    LOG_FIRST_N(WARNING, 1)
        << "When collecting shapes, it is recommended to run multiple loops to "
           "obtain more accurate shape information.";
  }

  auto hook = [&](const std::string &op_type,
                  const std::string &input_name,
                  const paddle::Tensor &input_tensor) -> void {
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    if (config_.use_gpu()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto *dev_ctx = pool.Get(place_);
      auto stream = static_cast<phi::GPUContext *>(dev_ctx)->stream();
#ifdef PADDLE_WITH_HIP
      hipStreamSynchronize(stream);
#else
      cudaStreamSynchronize(stream);
#endif
#endif
    }

    if (!input_tensor.is_dense_tensor()) return;
    auto tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(input_tensor.impl()).get();
    phi::DDim dim = tensor->dims();
    std::vector<int32_t> shape(dim.size());
    for (int i = 0; i < static_cast<int>(shape.size()); ++i)
      shape[i] = static_cast<int32_t>(dim[i]);
    if (!shape.empty()) {
      shape_info_[input_name].emplace_back(shape);
    } else if (tensor->numel() > 0) {
      // This must be a zero dimension tensor.
      PADDLE_ENFORCE_EQ(tensor->numel(),
                        1UL,
                        common::errors::PreconditionNotMet(
                            "This tensor must have one element, but got %ld.",
                            tensor->numel()));
      std::vector<int32_t> zero_shape(1, 1);
      shape_info_[input_name].emplace_back(zero_shape);
    }

    // We need collect value range for shape tensor for Paddle-TRT's use.
    // To be noticed, this method to identify all shape tensors is based on
    // assumption that all shape tensors in the model have numbers <= 8.
    // This is a simple method to identify all shape tensors with some
    // mistakes, but it doesn't matter.
    auto is_shape_tensor = tensor->numel() <= 8 && tensor->numel() >= 1;
    if ((tensor->dtype() == phi::DataType::INT32 ||
         tensor->dtype() == phi::DataType::INT64) &&
        is_shape_tensor) {
      std::vector<int> int32_host(tensor->numel());

      if (phi::is_cpu_place(tensor->place())) {
        auto &int32_tensor = *tensor;
        if (tensor->dtype() == phi::DataType::INT64) {
          auto *cpu_ctx = pool.Get(phi::CPUPlace());
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::CPUContext &>(*cpu_ctx),
              *tensor,
              DataType::INT32);
        }
        paddle::memory::Copy(phi::CPUPlace(),
                             int32_host.data(),
                             phi::CPUPlace(),
                             int32_tensor.data<int>(),
                             int32_tensor.numel() * sizeof(int));
      } else if (phi::is_gpu_place(tensor->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        auto *dev_ctx = pool.Get(tensor->place());
        auto &int32_tensor = *tensor;
        if (tensor->dtype() == phi::DataType::INT64) {
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::GPUContext &>(*dev_ctx),
              *tensor,
              DataType::INT32);
        }
        paddle::memory::Copy(phi::CPUPlace(),
                             int32_host.data(),
                             int32_tensor.place(),
                             int32_tensor.data<int>(),
                             int32_tensor.numel() * sizeof(int),
                             nullptr);
#endif
      }
      shape_tensor_value_[input_name].emplace_back(int32_host);
    }
  };
  RegisterInputHook(hook);
}

bool AnalysisPredictor::ExpRunWithRuntimeConfig(void *config) {
#ifdef PADDLE_WITH_XPU
  auto xpu_runtime_config =
      reinterpret_cast<paddle_infer::experimental::XpuRuntimeConfig *>(config);

  config_.xpu_config_.context = xpu_runtime_config->context;
  auto *stream = xpu_runtime_config->stream;
  if (stream != nullptr && stream != predictor_stream_) {
    paddle::platform::XPUStreamSync(
        static_cast<paddle::xpuStream>(predictor_stream_));
    predictor_stream_ = stream;
  }

  auto l3_size = xpu_runtime_config->l3_size;
  auto l3_autotune_size = xpu_runtime_config->l3_autotune_size;
  PADDLE_ENFORCE_LE(
      l3_autotune_size,
      l3_size,
      common::errors::InvalidArgument(
          "l3_autotune_size(%zu) should be less than or equal to l3_size(%zu).",
          l3_autotune_size,
          l3_size));
  config_.xpu_config_.l3_size = l3_size;
  config_.xpu_config_.l3_ptr = xpu_runtime_config->l3_ptr;
  config_.xpu_config_.l3_autotune_size = l3_autotune_size;

  return ZeroCopyRun();
#endif
  return false;
}

void AnalysisPredictor::StatisticShapeRangeInfo() {
  std::map<std::string, std::vector<int32_t>> min_shapes;
  std::map<std::string, std::vector<int32_t>> max_shapes;
  std::map<std::string, std::vector<int32_t>> opt_shapes;
  std::map<std::string, std::vector<int32_t>> min_values;
  std::map<std::string, std::vector<int32_t>> max_values;
  std::map<std::string, std::vector<int32_t>> opt_values;

  auto extract_min_max_opt =
      [](std::map<std::string, std::vector<int32_t>> &min_data,
         decltype(min_data) max_data,
         decltype(min_data) opt_data,
         decltype(shape_info_) shape_data) {
        for (auto const &it : shape_data) {
          auto name = it.first;
          auto shapes = it.second;

          std::vector<int32_t> min_shape(shapes[0].begin(), shapes[0].end());
          std::vector<int32_t> max_shape(shapes[0].begin(), shapes[0].end());
          std::vector<int32_t> opt_shape(shapes[0].begin(), shapes[0].end());

          auto ShapeMaxFreq =
              [](const std::map<int32_t, int32_t> &m) -> int32_t {
            std::vector<std::pair<int32_t, int32_t>> counter;
            for (auto &it : m) counter.emplace_back(it);
            std::sort(counter.begin(),
                      counter.end(),
                      [](std::pair<int32_t, int32_t> &a,
                         std::pair<int32_t, int32_t> &b) {
                        return a.second > b.second;
                      });
            return counter[0].first;
          };

          for (size_t d = 0; d < shapes[0].size(); ++d) {
            std::map<int32_t, int32_t> counter;
            for (auto &shape : shapes) {
              counter[shape[d]] += 1;
              if (shape[d] < min_shape[d]) min_shape[d] = shape[d];
              if (shape[d] > max_shape[d]) max_shape[d] = shape[d];
            }
            opt_shape[d] = ShapeMaxFreq(counter);
          }

          min_data[name] = min_shape;
          max_data[name] = max_shape;
          opt_data[name] = opt_shape;
        }
      };
  extract_min_max_opt(min_shapes, max_shapes, opt_shapes, shape_info_);
  extract_min_max_opt(min_values, max_values, opt_values, shape_tensor_value_);

  inference::SerializeShapeRangeInfo(config_.shape_range_info_path(),
                                     min_shapes,
                                     max_shapes,
                                     opt_shapes,
                                     min_values,
                                     max_values,
                                     opt_values);
}

bool AnalysisPredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir().empty()) {  // NOLINT
    filename = config_.model_dir() + "/__model__";
  } else if (!config_.prog_file().empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    filename = config_.prog_file();
  } else {
    if (config_.model_dir().empty() && config_.prog_file().empty()) {
      LOG(ERROR)
          << "Either model_dir or (prog_file, param_file) should be set.";
      return false;
    }
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s'.",
        config_.model_dir(),
        config_.params_file());
    return false;
  }

  // Create ProgramDesc
  framework::proto::ProgramDesc proto;
  if (!config_.model_from_memory()) {
    std::string pb_content;
    // Read binary
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin.is_open()),
        true,
        common::errors::NotFound(
            "Cannot open file %s, please confirm whether the file is normal.",
            filename));
    fin.seekg(0, std::ios::end);
    pb_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(pb_content.at(0)), pb_content.size());  // NOLINT
    fin.close();

    proto.ParseFromString(pb_content);
  } else {
    proto.ParseFromString(config_.prog_file());
  }
  inference_program_ = std::make_unique<framework::ProgramDesc>(proto);
  return true;
}

bool AnalysisPredictor::LoadParameters() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          common::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));

  const auto &global_block = inference_program_->MutableBlock(0);

  // create a temporary program to load parameters.

  std::unique_ptr<framework::ProgramDesc> load_program(
      new framework::ProgramDesc());
  framework::BlockDesc *load_block = load_program->MutableBlock(0);
  std::vector<std::string> params;

  for (auto *var : global_block->AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();

      framework::VarDesc *new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      if (!config_.params_file().empty()) {
        params.push_back(new_var->Name());
      } else {
        // append_op
        framework::OpDesc *op = load_block->AppendOp();
        op->SetType("load");
        op->SetOutput("Out", {new_var->Name()});
        op->SetAttr("file_path", {config_.model_dir() + "/" + new_var->Name()});
        op->CheckAttrs();
      }
    }
  }

  if (!config_.params_file().empty()) {
    // sort paramlist to have consistent ordering
    std::sort(params.begin(), params.end());
    // append just the load_combine op
    framework::OpDesc *op = load_block->AppendOp();
    op->SetType("load_combine");
    op->SetOutput("Out", params);
    op->SetAttr("file_path", {config_.params_file()});
    op->CheckAttrs();
  }

  // Use NaiveExecutor to Load parameters.
  framework::NaiveExecutor e(place_);
  e.Prepare(scope_.get(), *load_program, 0);
  e.Run();
  VLOG(3) << "get " << scope_->LocalVarNames().size() << " vars after load";

  return true;
}

uint64_t AnalysisPredictor::TryShrinkMemory() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (config_.use_gpu()) {
    paddle::platform::EmptyCache();
  }
#endif
  return paddle::memory::Release(place_);
}

void AnalysisPredictor::ClearIntermediateTensor() {
  if (config_.new_ir_enabled()) {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Don't need to use this API [ClearIntermediateTensor] when PIR is "
        "enabled."));
  }
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          common::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));
  const auto &global_block = inference_program_->MutableBlock(0);
  for (auto *var : global_block->AllVars()) {
    if (!IsPersistable(var)) {
      const std::string name = var->Name();
      auto *variable = executor_->GetScope()->FindVar(name);
      if (variable != nullptr && variable->IsType<phi::DenseTensor>() &&
          name != framework::kFeedOpType && name != framework::kFetchOpType) {
        VLOG(3) << "Clear Intermediate Tensor: " << name;
        auto *t = variable->GetMutable<phi::DenseTensor>();
        t->clear();
      }
    }
  }
}

#ifdef PADDLE_WITH_TENSORRT
using inference::Singleton;
bool AnalysisPredictor::SaveTrtCalibToDisk() {
  PADDLE_ENFORCE_EQ(config_.tensorrt_engine_enabled(),
                    true,
                    common::errors::PreconditionNotMet(
                        "This func can be invoked only in trt mode"));
  auto &block = inference_program_->Block(0);
  for (auto &op_desc : block.AllOps()) {
    if (op_desc->Type() == "tensorrt_engine") {
      std::string engine_name = PADDLE_GET_CONST(
          std::string, op_desc->GetAttr("calibration_engine_key"));
      if (!Singleton<TRTCalibratorEngineManager>::Global().Has(engine_name)) {
        LOG(ERROR) << "You should run the predictor(with trt) on the real data "
                      "to generate calibration info";
        return false;
      }
      TRTCalibratorEngine *calib_engine =
          Singleton<TRTCalibratorEngineManager>::Global().Get(engine_name);
      LOG(INFO) << "Wait for calib threads done.";
      calib_engine->calib_->waitAndSetDone();
      LOG(INFO) << "Generating TRT Calibration table data, this may cost a lot "
                   "of time...";
      calib_engine->thr_->join();
      std::string calibration_table_data =
          calib_engine->calib_->getCalibrationTableAsString();

      if (calibration_table_data.empty()) {
        LOG(ERROR) << "the calibration table is empty.";
        return false;
      }

      std::string model_opt_cache_dir =
          argument_->Has("model_dir") ? argument_->model_dir()
                                      : inference::analysis::GetDirRoot(
                                            argument_->model_program_path());

      std::string calibration_table_data_path =
          inference::analysis::GetTrtCalibPath(
              inference::analysis::GetOrCreateModelOptCacheDir(
                  model_opt_cache_dir),
              engine_name);

      std::ofstream ofile(calibration_table_data_path, std::ios::out);
      LOG(INFO) << "Write Paddle-TRT INT8 calibration table data to file "
                << calibration_table_data_path;
      ofile << calibration_table_data;
      ofile.close();
    }
  }
  // Free all calibrator resources.
  Singleton<TRTCalibratorEngineManager>::Global().DeleteALL();
  return true;
}
#endif

AnalysisPredictor::~AnalysisPredictor() {  // NOLINT
#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled() &&
      config_.tensorrt_precision_mode_ == AnalysisConfig::Precision::kInt8 &&
      Singleton<TRTCalibratorEngineManager>::Global().Has()) {
    SaveTrtCalibToDisk();
  }
#endif

  if (config_.with_profile_) {
#ifdef PADDLE_WITH_NVTX
    platform::NvprofDisableRecordEvent();
    platform::CudaProfilerStop();
#endif
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }

  if (sub_scope_) {
    if (framework::global_transfer_scope_key().find(sub_scope_) !=
        framework::global_transfer_scope_key().end()) {
      auto scope_key_set = framework::global_transfer_scope_key()[sub_scope_];
      for (auto item : scope_key_set) {
        framework::global_transfer_data_cache().erase(item);
      }
      framework::global_transfer_scope_key().erase(sub_scope_);
    }
    for (auto &var_name : scope_->LocalVarNames()) {
      auto *var = scope_->FindVar(var_name);
      if (var->IsType<phi::DenseTensor>()) {
        auto *tensor = var->GetMutable<phi::DenseTensor>();
        tensor->clear();
      }
    }

    scope_->DeleteScope(sub_scope_);
  }

  if (config_.shape_range_info_collected()) {
    StatisticShapeRangeInfo();
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (predictor_stream_ != nullptr) {
    ResourceManager::Instance().DestroyGPUResource(predictor_stream_);
  }
#endif

  if (place_.GetType() != phi::AllocationType::UNDEFINED) {
    memory::Release(place_);
  }
  device_contexts_.clear();

#ifdef PADDLE_WITH_TENSORRT
  if (config_.trt_engine_memory_sharing()) {
    inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
        .ReleaseContextMemory(predictor_id_);
  }
#endif
}

std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone(void *stream) {
  VLOG(3) << "AnalysisPredictor::Clone";
  std::lock_guard<std::mutex> lk(clone_mutex_);
  auto *x = new AnalysisPredictor(config_);
  x->status_is_cloned_ = true;
  x->root_predictor_id_ = this->root_predictor_id_;
  x->config_.apply_optim_ = false;
  if (config_.use_external_stream_ && stream == nullptr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "config has been configured to use external stream, but the Clone "
        "function has not received a valid stream parameter."));
  } else if (!config_.use_external_stream_ && stream != nullptr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "config has not been configured to use external stream, but the Clone "
        "function has received a stream parameter."));
  }
  x->predictor_stream_ = stream;
  x->Init(scope_, inference_program_);
#ifdef PADDLE_WITH_TENSORRT
  x->executor_->ResetTrtOps(++AnalysisPredictor::clone_num_);
#endif
  return std::unique_ptr<PaddlePredictor>(x);
}

std::string AnalysisPredictor::GetSerializedProgram() const {
  return inference_program_->Proto()->SerializeAsString();
}

void AnalysisPredictor::RegisterOutputHook(
    const OutputTensorHookFunc &hookfunc) {
  if (config_.new_ir_enabled()) {
    std::call_once(register_output_hook_flag_, [this] {
      executor_->RegisterOutputHook(
          [this](framework::InstructionBase *instr,
                 framework::ValueExecutionInfo *value_exe_info,
                 framework::Scope *scope) {
            for (auto &output : instr->Outputs()) {
              auto var_name = value_exe_info->GetVarName(output.first);
              auto *var = scope->FindVar(var_name);
              if (!var || !var->IsType<phi::DenseTensor>()) continue;
              auto dense_tensor = var->Get<phi::DenseTensor>();
              if (!dense_tensor.initialized()) continue;
              auto tensor = paddle::Tensor(
                  std::make_shared<phi::DenseTensor>(dense_tensor), var_name);
              for (auto &hookfunc : this->output_hookfuncs_) {
                hookfunc(instr->Name() + ":" + std::to_string(instr->Id()),
                         var_name,
                         tensor);
              }
            }
          });
    });
    output_hookfuncs_.push_back(hookfunc);
  } else {
    std::call_once(register_output_hook_flag_, [this] {
      executor_->RegisterOutputHook(
          [this](framework::OperatorBase *op, framework::Scope *scope) {
            for (auto &output : op->Outputs()) {
              for (auto &var_name : output.second) {
                auto *var = scope->FindVar(var_name);
                if (!var || !var->IsType<phi::DenseTensor>()) continue;
                auto dense_tensor = var->Get<phi::DenseTensor>();
                if (!dense_tensor.initialized()) continue;
                auto tensor = paddle::Tensor(
                    std::make_shared<phi::DenseTensor>(dense_tensor), var_name);
                for (auto &hookfunc : this->output_hookfuncs_) {
                  hookfunc(op->Type(), var_name, tensor);
                }
              }
            }
          });
    });
    output_hookfuncs_.push_back(hookfunc);
  }
}

void AnalysisPredictor::RegisterInputHook(const InputTensorHookFunc &hookfunc) {
  if (config_.new_ir_enabled()) {
    std::call_once(register_input_hook_flag_, [this] {
      executor_->RegisterInputHook(
          [this](framework::InstructionBase *instr,
                 framework::ValueExecutionInfo *value_exe_info,
                 framework::Scope *scope) {
            for (auto &input : instr->Inputs()) {
              auto var_name = value_exe_info->GetVarName(input.first);
              auto *var = scope->FindVar(var_name);
              if (!var || !var->IsType<phi::DenseTensor>()) continue;
              auto dense_tensor = var->Get<phi::DenseTensor>();
              if (!dense_tensor.initialized()) continue;
              auto tensor = paddle::Tensor(
                  std::make_shared<phi::DenseTensor>(dense_tensor), var_name);
              for (auto &hookfunc : this->input_hookfuncs_) {
                hookfunc(instr->Name() + ":" + std::to_string(instr->Id()),
                         var_name,
                         tensor);
              }
            }
          });
    });
    input_hookfuncs_.push_back(hookfunc);
  } else {
    std::call_once(register_input_hook_flag_, [this] {
      executor_->RegisterInputHook(
          [this](framework::OperatorBase *op, framework::Scope *scope) {
            for (auto &input : op->Inputs()) {
              for (auto &var_name : input.second) {
                auto *var = scope->FindVar(var_name);
                if (!var || !var->IsType<phi::DenseTensor>()) continue;
                auto dense_tensor = var->Get<phi::DenseTensor>();
                if (!dense_tensor.initialized()) continue;
                auto tensor = paddle::Tensor(
                    std::make_shared<phi::DenseTensor>(dense_tensor), var_name);
                for (auto &hookfunc : this->input_hookfuncs_) {
                  hookfunc(op->Type(), var_name, tensor);
                }
              }
            }
          });
    });
    input_hookfuncs_.push_back(hookfunc);
  }
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<AnalysisConfig>(
    const AnalysisConfig &config) {
  LOG(WARNING) << "Deprecated. Please use CreatePredictor instead.";
  return CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
      config);
}

}  // namespace paddle

#ifdef PADDLE_WITH_TENSORRT
USE_TRT_CONVERTER(elementwise_add_weight);
USE_TRT_CONVERTER(elementwise_sub_weight);
USE_TRT_CONVERTER(elementwise_mul_weight);
USE_TRT_CONVERTER(elementwise_div_weight);
USE_TRT_CONVERTER(elementwise_min_weight);
USE_TRT_CONVERTER(elementwise_max_weight);
USE_TRT_CONVERTER(elementwise_pow_weight);
USE_TRT_CONVERTER(elementwise_mod_weight);
USE_TRT_CONVERTER(elementwise_floordiv_weight);
USE_TRT_CONVERTER(elementwise_add_tensor);
USE_TRT_CONVERTER(elementwise_sub_tensor);
USE_TRT_CONVERTER(elementwise_div_tensor);
USE_TRT_CONVERTER(elementwise_mul_tensor);
USE_TRT_CONVERTER(elementwise_max_tensor);
USE_TRT_CONVERTER(elementwise_min_tensor);
USE_TRT_CONVERTER(elementwise_pow_tensor);
USE_TRT_CONVERTER(elementwise_floordiv_tensor);
USE_TRT_CONVERTER(elementwise_mod_tensor);
USE_TRT_CONVERTER(less_than);
USE_TRT_CONVERTER(greater_than);
USE_TRT_CONVERTER(logical_or);
USE_TRT_CONVERTER(logical_xor);
USE_TRT_CONVERTER(logical_and);
USE_TRT_CONVERTER(less_equal);
USE_TRT_CONVERTER(greater_equal);
USE_TRT_CONVERTER(transpose);
USE_TRT_CONVERTER(transpose2);
USE_TRT_CONVERTER(flatten);
USE_TRT_CONVERTER(flatten_contiguous_range);
USE_TRT_CONVERTER(matrix_multiply);
USE_TRT_CONVERTER(bmm);
USE_TRT_CONVERTER(conv2d);
USE_TRT_CONVERTER(relu);
USE_TRT_CONVERTER(sigmoid);
USE_TRT_CONVERTER(pool2d);
USE_TRT_CONVERTER(softmax);
USE_TRT_CONVERTER(batch_norm);
USE_TRT_CONVERTER(concat);
USE_TRT_CONVERTER(dropout);
USE_TRT_CONVERTER(pad);
USE_TRT_CONVERTER(bitwise_and);
USE_TRT_CONVERTER(bitwise_or);
USE_TRT_CONVERTER(size);
#if IS_TRT_VERSION_GE(8200)
USE_TRT_CONVERTER(pad3d);
USE_TRT_CONVERTER(einsum)
#endif
USE_TRT_CONVERTER(hard_sigmoid);
USE_TRT_CONVERTER(hard_swish);
USE_TRT_CONVERTER(split);
USE_TRT_CONVERTER(fill_any_like);
USE_TRT_CONVERTER(prelu);
USE_TRT_CONVERTER(conv2d_transpose);
USE_TRT_CONVERTER(leaky_relu);
USE_TRT_CONVERTER(shuffle_channel);
USE_TRT_CONVERTER(where);
USE_TRT_CONVERTER(bitwise_not);
USE_TRT_CONVERTER(one_hot);
USE_TRT_CONVERTER(one_hot_v2);
USE_TRT_CONVERTER(swish);
USE_TRT_CONVERTER(silu);
USE_TRT_CONVERTER(group_norm);
USE_TRT_CONVERTER(instance_norm);
USE_TRT_CONVERTER(layer_norm);
USE_TRT_CONVERTER(gelu);
USE_TRT_CONVERTER(multihead_matmul);
USE_TRT_CONVERTER(multihead_matmul_roformer);
USE_TRT_CONVERTER(skip_layernorm);
USE_TRT_CONVERTER(slice);
USE_TRT_CONVERTER(scale);
USE_TRT_CONVERTER(stack);
USE_TRT_CONVERTER(clip);
USE_TRT_CONVERTER(gather);
USE_TRT_CONVERTER(anchor_generator);
USE_TRT_CONVERTER(yolo_box);
USE_TRT_CONVERTER(yolo_box_head);
USE_TRT_CONVERTER(arg_max);
USE_TRT_CONVERTER(arg_min);
USE_TRT_CONVERTER(roi_align);
USE_TRT_CONVERTER(affine_channel);
USE_TRT_CONVERTER(multiclass_nms);
USE_TRT_CONVERTER(multiclass_nms3);
USE_TRT_CONVERTER(nearest_interp);
USE_TRT_CONVERTER(nearest_interp_v2);
USE_TRT_CONVERTER(bilinear_interp_v2);
USE_TRT_CONVERTER(linear_interp_v2);
USE_TRT_CONVERTER(reshape);
USE_TRT_CONVERTER(reshape2);
USE_TRT_CONVERTER(gather_nd);
USE_TRT_CONVERTER(reduce_mean);
USE_TRT_CONVERTER(reduce_max);
USE_TRT_CONVERTER(reduce_min);
USE_TRT_CONVERTER(reduce_sum);
USE_TRT_CONVERTER(reduce_prod);
USE_TRT_CONVERTER(reduce_any);
USE_TRT_CONVERTER(reduce_all);
USE_TRT_CONVERTER(tile);
USE_TRT_CONVERTER(conv3d);
USE_TRT_CONVERTER(conv3d_transpose);
USE_TRT_CONVERTER(mish);
USE_TRT_CONVERTER(deformable_conv);
USE_TRT_CONVERTER(pool3d)
USE_TRT_CONVERTER(square);
// unary op
USE_TRT_CONVERTER(exp);
USE_TRT_CONVERTER(log);
USE_TRT_CONVERTER(sqrt);
USE_TRT_CONVERTER(reciprocal);
USE_TRT_CONVERTER(abs);
USE_TRT_CONVERTER(sin);
USE_TRT_CONVERTER(cos);
USE_TRT_CONVERTER(tan);
USE_TRT_CONVERTER(sinh);
USE_TRT_CONVERTER(cosh);
USE_TRT_CONVERTER(tanh);
USE_TRT_CONVERTER(asin);
USE_TRT_CONVERTER(acos);
USE_TRT_CONVERTER(atan);
USE_TRT_CONVERTER(asinh);
USE_TRT_CONVERTER(acosh);
USE_TRT_CONVERTER(atanh);
USE_TRT_CONVERTER(ceil);
USE_TRT_CONVERTER(floor);
#if IS_TRT_VERSION_GE(8200)
USE_TRT_CONVERTER(round);
USE_TRT_CONVERTER(sign);
#endif
USE_TRT_CONVERTER(rsqrt);
USE_TRT_CONVERTER(fused_preln_embedding_eltwise_layernorm)
USE_TRT_CONVERTER(prompt_tuning_emb_eltwise_layernorm);
USE_TRT_CONVERTER(fused_embedding_eltwise_layernorm);
USE_TRT_CONVERTER(preln_skip_layernorm)
USE_TRT_CONVERTER(fused_bias_dropout_residual_layer_norm)
USE_TRT_CONVERTER(c_allreduce_sum)
USE_TRT_CONVERTER(roll)
USE_TRT_CONVERTER(strided_slice)
USE_TRT_CONVERTER(rnn)
USE_TRT_CONVERTER(fill_constant_batch_size_like)
USE_TRT_CONVERTER(transformer_input_convert)
USE_TRT_CONVERTER(cast)
USE_TRT_CONVERTER(recover_padding)
USE_TRT_CONVERTER(remove_padding)
USE_TRT_CONVERTER(equal);
USE_TRT_CONVERTER(not_equal);
USE_TRT_CONVERTER(top_k)
USE_TRT_CONVERTER(top_k_v2)
USE_TRT_CONVERTER(range)
USE_TRT_CONVERTER(squeeze2)
USE_TRT_CONVERTER(unsqueeze2)
USE_TRT_CONVERTER(sum)
USE_TRT_CONVERTER(shape)
USE_TRT_CONVERTER(fill_constant)
USE_TRT_CONVERTER(fused_token_prune)
USE_TRT_CONVERTER(celu)
USE_TRT_CONVERTER(layernorm_shift_partition)
USE_TRT_CONVERTER(reverse_roll)
USE_TRT_CONVERTER(preln_layernorm_shift_partition)
USE_TRT_CONVERTER(merge_layernorm)
USE_TRT_CONVERTER(trans_layernorm)
USE_TRT_CONVERTER(skip_merge_layernorm)
USE_TRT_CONVERTER(generic_plugin_creator)
USE_TRT_CONVERTER(custom_plugin_creater)  // typos: disable-line
USE_TRT_CONVERTER(custom_generic_plugin_creator)
USE_TRT_CONVERTER(fuse_eleadd_transpose)
USE_TRT_CONVERTER(tanh_shrink)
USE_TRT_CONVERTER(logsigmoid)
USE_TRT_CONVERTER(lookup_table)
USE_TRT_CONVERTER(lookup_table_v2)
USE_TRT_CONVERTER(expand_v2)
USE_TRT_CONVERTER(expand_as_v2)
USE_TRT_CONVERTER(argsort)
USE_TRT_CONVERTER(take_along_axis)
USE_TRT_CONVERTER(skip_groupnorm_act)
USE_TRT_CONVERTER(preln_groupnorm_act)
USE_TRT_CONVERTER(cumsum)
USE_TRT_CONVERTER(assign)
USE_TRT_CONVERTER(p_norm)
USE_TRT_CONVERTER(unbind)
USE_TRT_CONVERTER(index_put)
USE_TRT_CONVERTER(flip)
USE_TRT_CONVERTER(isnan_v2)
USE_TRT_CONVERTER(share_data)
#if IS_TRT_VERSION_GE(8522)
USE_TRT_CONVERTER(flash_multihead_matmul)
USE_TRT_CONVERTER(cross_multihead_matmul)
USE_TRT_CONVERTER(qk_multihead_matmul)
#endif
#if IS_TRT_VERSION_GE(8510)
USE_TRT_CONVERTER(grid_sampler)
#endif
#if IS_TRT_VERSION_GE(8200)
USE_TRT_CONVERTER(set_value)
USE_TRT_CONVERTER(index_select);
USE_TRT_CONVERTER(temporal_shift)
#endif
#if PADDLE_WITH_CUSPARSELT && IS_TRT_VERSION_GE(8000)
USE_TRT_CONVERTER(sparse_fc)
USE_TRT_CONVERTER(sparse_multihead_matmul)
#endif
#if IS_TRT_VERSION_GE(8000)
USE_TRT_CONVERTER(quantize_linear)
USE_TRT_CONVERTER(dequantize_linear)
#endif
#endif

namespace paddle_infer {

Predictor::Predictor(const Config &config) : predictor_(nullptr) {
  // The second parameter indicates that the discard log is not printed
  if (config.use_onnxruntime()) {
#ifdef PADDLE_WITH_ONNXRUNTIME
    if (config.use_gpu()) {
      LOG(WARNING) << "The current ONNXRuntime backend doesn't support GPU,"
                      "and it falls back to use Paddle Inference.";
    } else if (!paddle::CheckConvertToONNX(config)) {
      LOG(WARNING)
          << "Paddle2ONNX do't support convert the Model, fall back to using "
             "Paddle Inference.";
    } else {
      predictor_ =
          paddle::CreatePaddlePredictor<Config,
                                        paddle::PaddleEngineKind::kONNXRuntime>(
              config);
      return;
    }
#else
    LOG(WARNING)
        << "The onnxruntime backend isn't enabled,"
           " and please re-compile Paddle with WITH_ONNXRUNTIME option,"
           "fall back to using Paddle Inference.";
#endif
  }
  predictor_ =
      paddle::CreatePaddlePredictor<Config,
                                    paddle::PaddleEngineKind::kAnalysis>(
          config);
}

std::vector<std::string> Predictor::GetInputNames() {
  return predictor_->GetInputNames();
}

std::map<std::string, std::vector<int64_t>> Predictor::GetInputTensorShape() {
  return predictor_->GetInputTensorShape();
}

std::map<std::string, DataType> Predictor::GetInputTypes() {
  return predictor_->GetInputTypes();
}

std::unique_ptr<Tensor> Predictor::GetInputHandle(const std::string &name) {
  return predictor_->GetInputTensor(name);
}

std::vector<std::string> Predictor::GetOutputNames() {
  return predictor_->GetOutputNames();
}

std::unique_ptr<Tensor> Predictor::GetOutputHandle(const std::string &name) {
  return predictor_->GetOutputTensor(name);
}

std::map<std::string, std::vector<int64_t>> Predictor::GetOutputTensorShape() {
  return predictor_->GetOutputTensorShape();
}

std::map<std::string, DataType> Predictor::GetOutputTypes() {
  return predictor_->GetOutputTypes();
}

bool Predictor::Run() { return predictor_->ZeroCopyRun(); }

bool Predictor::Run(const std::vector<paddle::Tensor> &inputs,
                    std::vector<paddle::Tensor> *outputs) {
  return predictor_->Run(inputs, outputs);
}

std::unique_ptr<Predictor> Predictor::Clone(void *stream) {
  auto analysis_pred = predictor_->Clone(stream);
  std::unique_ptr<Predictor> pred(new Predictor(std::move(analysis_pred)));
  return pred;
}

void Predictor::ClearIntermediateTensor() {
  predictor_->ClearIntermediateTensor();
}

uint64_t Predictor::TryShrinkMemory() { return predictor_->TryShrinkMemory(); }

void Predictor::RegisterOutputHook(const OutputTensorHookFunc &hookfunc) {
  predictor_->RegisterOutputHook(hookfunc);
}
void Predictor::RegisterInputHook(const InputTensorHookFunc &hookfunc) {
  predictor_->RegisterInputHook(hookfunc);
}

void *Predictor::GetExecStream() const { return predictor_->GetExecStream(); }

int GetNumBytesOfDataType(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return -1;
  }
}

std::string GetVersion() { return paddle::get_version(); }

std::tuple<int, int, int> GetTrtCompileVersion() {
#ifdef PADDLE_WITH_TENSORRT
  return paddle::inference::tensorrt::GetTrtCompileVersion();
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

std::tuple<int, int, int> GetTrtRuntimeVersion() {
#ifdef PADDLE_WITH_TENSORRT
  return paddle::inference::tensorrt::GetTrtRuntimeVersion();
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

void UpdateDllFlag(const char *name, const char *value) {
  paddle::UpdateDllFlag(name, value);
}

void ConvertToMixedPrecision(const std::string &model_file,
                             const std::string &params_file,
                             const std::string &mixed_model_file,
                             const std::string &mixed_params_file,
                             PrecisionType mixed_precision,
                             paddle_infer::PlaceType backend,
                             bool keep_io_types,
                             std::unordered_set<std::string> black_list,
                             std::unordered_set<std::string> white_list) {
  auto phi_backend = paddle::ConvertBackend(backend);
  auto phi_precision = paddle::ConvertPrecision(mixed_precision);
  paddle::inference::analysis::ConvertToMixedPrecision(model_file,
                                                       params_file,
                                                       mixed_model_file,
                                                       mixed_params_file,
                                                       phi_precision,
                                                       phi_backend,
                                                       keep_io_types,
                                                       black_list,
                                                       white_list);
}

}  // namespace paddle_infer

namespace paddle_infer {
std::shared_ptr<Predictor> CreatePredictor(const Config &config) {  // NOLINT
  std::shared_ptr<Predictor> predictor(new Predictor(config));
  return predictor;
}

namespace services {
PredictorPool::PredictorPool(const Config &config, size_t size) : preds_() {
  PADDLE_ENFORCE_GE(
      size,
      1UL,
      common::errors::InvalidArgument(
          "The predictor pool size should be greater than 1, but it's (%d)",
          size));
  Config copy_config(config);
  main_pred_ = std::make_unique<Predictor>(config);
  for (size_t i = 0; i < size - 1; i++) {
    if (config.tensorrt_engine_enabled()) {
      Config config_tmp(copy_config);
      preds_.emplace_back(new Predictor(config_tmp));
    } else {
      preds_.emplace_back(main_pred_->Clone());
    }
  }
}

Predictor *PredictorPool::Retrieve(size_t idx) {
  PADDLE_ENFORCE_LT(
      idx,
      preds_.size() + 1,
      common::errors::InvalidArgument(
          "There are (%d) predictors in the pool, but the idx is (%d)",
          idx,
          preds_.size() + 1));
  if (idx == 0) {
    return main_pred_.get();
  }
  return preds_[idx - 1].get();
}
}  // namespace services

namespace experimental {

// Note: Can only be used under thread_local semantics.
bool InternalUtils::RunWithExternalStream(paddle_infer::Predictor *p,
                                          cudaStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  auto pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  return pred->ExpRunWithExternalStream(stream);
#endif
  return false;
}
bool InternalUtils::RunWithExternalStream(paddle_infer::Predictor *p,
                                          hipStream_t stream) {
#ifdef PADDLE_WITH_HIP
  auto pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  return pred->ExpRunWithExternalStream(stream);
#endif
  return false;
}

bool InternalUtils::RunWithRuntimeConfig(paddle_infer::Predictor *p,
                                         void *config) {
  auto pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  return pred->ExpRunWithRuntimeConfig(config);
}

void InternalUtils::UpdateConfigInterleaved(paddle_infer::Config *c,
                                            bool with_interleaved) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  c->trt_with_interleaved_ = with_interleaved;
#endif
}

void InternalUtils::SetTransformerPosid(
    paddle_infer::Config *c, const std::string &tensorrt_transformer_posid) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  c->tensorrt_transformer_posid_ = tensorrt_transformer_posid;
#endif
}

void InternalUtils::SetTransformerMaskid(
    paddle_infer::Config *c, const std::string &tensorrt_transformer_maskid) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  c->tensorrt_transformer_maskid_ = tensorrt_transformer_maskid;
#endif
}

void InternalUtils::DisableTensorRtHalfOps(
    paddle_infer::Config *c, const std::unordered_set<std::string> &ops) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  c->trt_ops_run_float_ = ops;
#endif
}

void InternalUtils::SyncStream(paddle_infer::Predictor *p) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto *pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  auto *dev_ctx = reinterpret_cast<phi::GPUContext *>(pool.Get(pred->place_));
  paddle::gpuStreamSynchronize(dev_ctx->stream());
#endif
}
void InternalUtils::SyncStream(cudaStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  cudaStreamSynchronize(stream);
#endif
}

void InternalUtils::SyncStream(hipStream_t stream) {
#ifdef PADDLE_WITH_HIP
  hipStreamSynchronize(stream);
#endif
}

}  // namespace experimental
}  // namespace paddle_infer
