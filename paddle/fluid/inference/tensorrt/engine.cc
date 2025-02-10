/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/engine.h"
#include <NvInfer.h>
#include <glog/logging.h>
#include <string>
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"  // NOLINT

#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle::inference::tensorrt {

thread_local int TensorRTEngine::predictor_id_per_thread = 0;

void TensorRTEngine::Weight::SetDataType(phi::DataType type) {
  nvinfer1::DataType nv_type = nvinfer1::DataType::kFLOAT;
  switch (type) {
    case phi::DataType::FLOAT32:
      nv_type = nvinfer1::DataType::kFLOAT;
      break;
    case phi::DataType::FLOAT16:
      nv_type = nvinfer1::DataType::kHALF;
      break;
    case phi::DataType::INT32:
      nv_type = nvinfer1::DataType::kINT32;
      break;
    case phi::DataType::INT8:
      nv_type = nvinfer1::DataType::kINT8;
      break;
#if IS_TRT_VERSION_GE(7000)
    case phi::DataType::BOOL:
      nv_type = nvinfer1::DataType::kBOOL;
      break;
#endif
    default:
      common::errors::InvalidArgument(
          "Paddle-TRT loads weights failed, found not supported data type %s.",
          type);
      break;
  }
  w_.type = nv_type;
}

void TensorRTEngine::InitNetwork() {
  FreshDeviceId();
  infer_builder_.reset(createInferBuilder(&logger_));
#if IS_TRT_VERSION_GE(8500)
  infer_network_.reset(infer_builder_->createNetworkV2(
      1U << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
#else
  if (with_dynamic_shape()) {
    infer_network_.reset(infer_builder_->createNetworkV2(
        1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  } else {
    infer_network_.reset(infer_builder_->createNetworkV2(0U));
  }
#endif
  infer_builder_config_.reset(infer_builder_->createBuilderConfig());
  optim_profiles_.resize(max_profile_num_);
  for (int i = 0; i < max_profile_num_; i++)
    optim_profiles_[i] = infer_builder_->createOptimizationProfile();
}

nvinfer1::IExecutionContext *TensorRTEngine::context() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (infer_context_.find(predictor_id_per_thread) == infer_context_.end()) {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        common::errors::InvalidArgument(
            "You should build engine first and then set the context."));
    // We may see trt warning: Profile 0 has been chosen by another
    // IExecutionContext...
    // It's ok. We will set it later.
    nvinfer1::IExecutionContext *infer_context{nullptr};
    if (params_.context_memory_sharing) {
      infer_context =
          infer_engine_->createExecutionContextWithoutDeviceMemory();
    } else {
      infer_context = infer_engine_->createExecutionContext();
    }
    PADDLE_ENFORCE_NOT_NULL(
        infer_context,
        common::errors::InvalidArgument(
            "TensorRT engine can not build execution context."));
    if (with_dynamic_shape()) {
      // need new profile if it's not the first
      if (cur_profile_num_ > 0) {
#if IS_TRT_VERSION_GE(8600)
        infer_context->setOptimizationProfileAsync(cur_profile_num_, nullptr);
#else
        infer_context->setOptimizationProfile(cur_profile_num_);
#endif
      }
      profile_index_[predictor_id_per_thread] = cur_profile_num_;
      ++cur_profile_num_;
    }
    infer_context_[predictor_id_per_thread].reset(infer_context);
  }
  return infer_context_[predictor_id_per_thread].get();
}

void TensorRTEngine::Execute(int batch_size,
                             std::vector<void *> *buffers,
                             cudaStream_t stream) {
  FreshDeviceId();
  auto infer_context = context();
  if (params_.context_memory_sharing) {
    void *context_memory{nullptr};
    context_memory =
        inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
            .GetContextMemory(
                predictor_id_per_thread,
                phi::GPUPlace(device_id()),
                phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    infer_context->setDeviceMemory(context_memory);
  }

  // TODO(wilber): Is cudaGraph has conflict with memory sharing?
  if (startup_with_cudagraph_ && !cudagraph_inited_) {
    // Avoid capturing initialization calls by executing the enqueue function at
    // least once before starting CUDA graph capture.
    const auto ret = Enqueue(infer_context, buffers, batch_size, stream);
    PADDLE_ENFORCE_EQ(
        ret,
        true,
        common::errors::PreconditionNotMet("Trt CudaGraph test run failed."));
    cudaStreamSynchronize(stream);

    cuda_graph_.BeginCapture(stream);
    // The built TRT engine may contain operations that are not permitted under
    // CUDA graph capture mode. When the stream is capturing, the call may
    // return false if the current CUDA graph capture fails.
    if (Enqueue(infer_context, buffers, batch_size, stream)) {
      cuda_graph_.EndCapture(stream);
      cudagraph_inited_ = true;
    } else {
      cuda_graph_.EndCaptureOnError(stream);
      // Ensure any CUDA error has been cleaned up.
      PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
      LOG(WARNING) << "The built TensorRT engine contains operations that are "
                      "not permitted under "
                      "CUDA graph capture mode. The specified UseCudaGraph "
                      "flag has been ignored. The inference will be "
                      "launched without using CUDA graph launch.";
      cudagraph_inited_ = false;
    }
    startup_with_cudagraph_ = false;
  }

  Enqueue(infer_context, buffers, batch_size, stream);
}

bool TensorRTEngine::Enqueue(nvinfer1::IExecutionContext *context,
                             std::vector<void *> *buffers,
                             int batch_size,
                             cudaStream_t stream) {
  if (cudagraph_inited_) {
    VLOG(1) << "cuda_graph init success, so we will use cuda graph launch the "
               "entire graph.";
    return cuda_graph_.Launch(stream);
  }

#if IS_TRT_VERSION_GE(8500)
  for (size_t j = 0; j < buffers->size(); ++j) {
    auto name = context->getEngine().getIOTensorName(j);
    if (context->getEngine().isShapeInferenceIO(name) &&
        context->getEngine().getTensorIOMode(name) ==
            nvinfer1::TensorIOMode::kINPUT) {
      continue;
    } else {
      context->setTensorAddress(name, (*buffers)[j]);
    }
  }
#endif

  bool ret;
#if IS_TRT_VERSION_GE(8500)
  ret = context->enqueueV3(stream);
#else
  if (!with_dynamic_shape()) {
    ret = context->enqueue(batch_size, buffers->data(), stream, nullptr);
  } else {
#if IS_TRT_VERSION_GE(8500)
    ret = context->enqueueV3(stream);
#else
    ret = context->enqueueV2(buffers->data(), stream, nullptr);
#endif
  }
#endif
  return ret;
}

void TensorRTEngine::FreezeNetwork() {
  FreshDeviceId();
  VLOG(3) << "TRT to freeze network";
  PADDLE_ENFORCE_NOT_NULL(infer_builder_,
                          common::errors::InvalidArgument(
                              "Inference builder of TRT is null. Please make "
                              "sure you call InitNetwork first."));
  PADDLE_ENFORCE_NOT_NULL(network(),
                          common::errors::InvalidArgument(
                              "Call InitNetwork first to initialize network."));
  // build engine.
#if IS_TRT_VERSION_LT(10000)
  if (!with_dynamic_shape()) {
    infer_builder_->setMaxBatchSize(params_.max_batch_size);
  }
#endif
#if IS_TRT_VERSION_GE(8300)
  infer_builder_config_->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE, params_.max_workspace_size);
#else
  infer_builder_config_->setMaxWorkspaceSize(params_.max_workspace_size);
#endif

  bool enable_fp16 = (precision() == phi::DataType::FLOAT16);
  if (enable_fp16) {
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (!support_fp16) {
      LOG(INFO) << "You specify FP16 mode, but the hardware do not support "
                   "FP16 speed up, use FP32 instead.";
    } else {
      LOG(INFO) << "Run Paddle-TRT FP16 mode";
    }
  }

  if (precision() == phi::DataType::BFLOAT16) {
#if IS_TRT_VERSION_GE(9000)
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kBF16);
    LOG(INFO) << "Run Paddle-TRT BF16 mode";
#else
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    if (!support_fp16) {
      LOG(INFO) << "Because the version of TensorRT is less than 9.0, and the "
                   "hardware do not support FP16, run Paddle-TRT FP32 mode";
    } else {
      LOG(INFO) << "Because the version of TensorRT is less than 9.0, run "
                   "Paddle-TRT FP16 mode";
    }
#endif
  }

  bool enable_int8 = (precision() == phi::DataType::INT8);
  if (enable_int8) {
    if (!use_dla()) {
      infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);

    if (params_.calibrator) {
      infer_builder_config_->setInt8Calibrator(params_.calibrator);
    } else if (!params_.use_explicit_quantization) {
      infer_builder_config_->setInt8Calibrator(nullptr);

      for (auto &quant_range : quant_dynamic_range_) {
        auto tensor = quant_range.first;
        float range = quant_range.second;
        tensor->setDynamicRange(-range, range);
      }

      std::unordered_set<nvinfer1::ITensor *> all_t;
      for (int i = 0; i < network()->getNbLayers(); i++) {
        auto layer = network()->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
          all_t.insert(layer->getOutput(j));
        }
      }

      for (int i = 0; i < network()->getNbInputs(); i++) {
        all_t.insert(network()->getInput(i));
      }

      for (auto &t : all_t) {
        if (!quant_dynamic_range_.count(t)) {
          VLOG(3) << "We are in trt int8 mode(not calibration), scale not set"
                  << " for tensor " << t->getName()
                  << ", this might be ok when trt does not need this range";
        }
      }
    }
  }

  if (use_dla()) {
    if (!enable_int8 && !enable_fp16) {
      LOG(WARNING) << "TensorRT DLA must be used with int8 or fp16, but you "
                      "set float32, so DLA is not used.";
    } else if (infer_builder_->getNbDLACores() == 0) {
      LOG(WARNING)
          << "TensorRT DLA is set by config, but your device does not have "
             "DLA, so DLA is not used.";
    } else {
      if (params_.dla_core < 0 ||
          params_.dla_core >= infer_builder_->getNbDLACores()) {
        params_.dla_core = 0;
        LOG(WARNING) << "Invalid DLACore, must be 0 < DLACore < "
                     << infer_builder_->getNbDLACores() << ", but got "
                     << params_.dla_core << ", so use use 0 as default.";
      }
      infer_builder_config_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      infer_builder_config_->setDLACore(params_.dla_core);
      infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      LOG(INFO) << "TensorRT DLA enabled in FreezeNetwork(), DLACore "
                << params_.dla_core;
    }
  }
  if (!refit_params_path().empty()) {
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kREFIT);
    VLOG(3) << "TensorRT Refittable enabled in FreezeNetwork()";
  }

  if (with_dynamic_shape()) {
    LOG(INFO) << "Run Paddle-TRT Dynamic Shape mode.";
    for (int i = 0; i < max_profile_num_; i++) {
      for (auto &input : min_input_shape()) {
#if IS_TRT_VERSION_LT(7100)
        // trt6/trt7011 will check all_of input > 0
        if (!(std::all_of(input.second.begin(),
                          input.second.end(),
                          [](int x) { return x > 0; }) &&
              std::all_of(max_input_shape()[input.first].begin(),
                          max_input_shape()[input.first].end(),
                          [](int x) { return x > 0; }) &&
              std::all_of(optim_input_shape()[input.first].begin(),
                          optim_input_shape()[input.first].end(),
                          [](int x) { return x > 0; }))) {
          continue;
        }
#endif
        VLOG(4) << "TRT dynamic_shape set " << input.first
                << " min: " << Vec2Str(input.second)
                << ", max: " << Vec2Str(max_input_shape()[input.first])
                << ", opt: " << Vec2Str(optim_input_shape()[input.first]);

        optim_profiles_[i]->setDimensions(
            input.first.c_str(),
            nvinfer1::OptProfileSelector::kMIN,
            Vec2TRT_Dims(input.second, input.first, true));
        optim_profiles_[i]->setDimensions(
            input.first.c_str(),
            nvinfer1::OptProfileSelector::kMAX,
            Vec2TRT_Dims(max_input_shape()[input.first], input.first, true));
        optim_profiles_[i]->setDimensions(
            input.first.c_str(),
            nvinfer1::OptProfileSelector::kOPT,
            Vec2TRT_Dims(optim_input_shape()[input.first], input.first, true));
      }

      for (int input_id = 0; input_id < network()->getNbInputs(); input_id++) {
        auto input_name = network()->getInput(input_id)->getName();
        if (!itensor_map_.count(input_name)) continue;
        if (!GetITensor(input_name)->isShapeTensor()) continue;
        PADDLE_ENFORCE_EQ(min_shape_tensor().count(input_name) > 0 &&
                              max_shape_tensor().count(input_name) > 0 &&
                              optim_shape_tensor().count(input_name) > 0,
                          true,
                          common::errors::InvalidArgument(
                              "Fail to find min/max/optim shape value for TRT "
                              "network's shape tensor input named %s.",
                              input_name));
        auto min_vec = min_shape_tensor().at(input_name);
        optim_profiles_[i]->setShapeValues(input_name,
                                           nvinfer1::OptProfileSelector::kMIN,
                                           min_vec.data(),
                                           min_vec.size());
        optim_profiles_[i]->setShapeValues(
            input_name,
            nvinfer1::OptProfileSelector::kMAX,
            max_shape_tensor()[input_name].data(),
            min_vec.size());
        optim_profiles_[i]->setShapeValues(
            input_name,
            nvinfer1::OptProfileSelector::kOPT,
            optim_shape_tensor()[input_name].data(),
            min_vec.size());
      }

      infer_builder_config_->addOptimizationProfile(optim_profiles_[i]);
    }
    if (WithFp16() && disable_trt_plugin_fp16()) {
      LOG(INFO) << "NOTE: In order to achieve higher accuracy, you have "
                   "disabled the fp16 mode of TRT Plugin,\n"
                << "you can reopen it with "
                   "'config.SetDynamicShapeInfo(min_shape, max_shape, "
                   "opt_shape, false /*disable_trt_plugin_fp16*/)'";
    }
  }
#if IS_TRT_VERSION_GE(8200)
  if (params_.use_inspector) {
    infer_builder_config_->setProfilingVerbosity(
        nvinfer1::ProfilingVerbosity::kDETAILED);
  }
#endif

#if IS_TRT_VERSION_GE(8600)
  VLOG(4) << "Set the TensorRT optimization level to be "
          << params_.optimization_level;
  infer_builder_config_->setBuilderOptimizationLevel(
      params_.optimization_level);
#endif

#if IS_TRT_VERSION_GE(8210)
  if (!trt_ops_run_float_.empty()) {
    infer_builder_config_->setFlag(
        nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
  }
#endif

#if IS_TRT_VERSION_LT(8000)
  infer_engine_.reset(infer_builder_->buildEngineWithConfig(
      *network(), *infer_builder_config_));
#else
  ihost_memory_.reset(infer_builder_->buildSerializedNetwork(
      *network(), *infer_builder_config_));
  PADDLE_ENFORCE_NOT_NULL(
      ihost_memory_,
      common::errors::Fatal(
          "Build TensorRT serialized network failed! Please recheck "
          "you configurations related to paddle-TensorRT."));

  infer_runtime_.reset(createInferRuntime(&logger_));
  PADDLE_ENFORCE_NOT_NULL(
      infer_runtime_,
      common::errors::Fatal("Build TensorRT runtime failed! Please recheck "
                            "you configurations related to paddle-TensorRT."));

  infer_engine_.reset(infer_runtime_->deserializeCudaEngine(
      ihost_memory_->data(), ihost_memory_->size()));
#endif

  PADDLE_ENFORCE_NOT_NULL(
      infer_engine_,
      common::errors::Fatal("Build TensorRT cuda engine failed! Please recheck "
                            "you configurations related to paddle-TensorRT."));

#if IS_TRT_VERSION_GE(8600)
  binding_num_ = infer_engine_->getNbIOTensors();
#else
  binding_num_ = infer_engine_->getNbBindings();
#endif
  // reset status for dynamic shape clone
  if (max_profile_num_ > 1) {
    infer_context_.clear();
    cur_profile_num_ = 0;
  }
  // for engine context memory sharing
  if (params_.context_memory_sharing) {
    inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
        .UpdateContextMemorySize(infer_engine_->getDeviceMemorySize(),
                                 predictor_id_per_thread);
  }
  if (params_.use_inspector) {
    GetEngineInfo(params_.engine_info_path);
  }
  if (!refit_params_path().empty()) {
    infer_refitter_.reset(createInferRefitter(infer_engine_.get(), &logger_));
    PADDLE_ENFORCE_NOT_NULL(
        infer_refitter_,
        common::errors::InvalidArgument(
            "Failed to create refitter for the TRT engine."));
  }
}

nvinfer1::ITensor *TensorRTEngine::DeclareInput(const std::string &name,
                                                nvinfer1::DataType dtype,
                                                const nvinfer1::Dims &dims) {
  PADDLE_ENFORCE_EQ(network() != nullptr,
                    true,
                    common::errors::InvalidArgument(
                        "The TRT network should be initialized first."));
  auto *input = network()->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE_NOT_NULL(
      input,
      common::errors::InvalidArgument("Adding input %s failed in "
                                      "TensorRT inference network. "
                                      "Please recheck your input.",
                                      name));
  PADDLE_ENFORCE_EQ(input->isNetworkInput(),
                    true,
                    common::errors::InvalidArgument(
                        "Input %s is not the input of TRT inference network. "
                        "Please recheck your input.",
                        name));
  TensorRTEngine::SetITensor(name, input);
  return input;
}

void TensorRTEngine::DeclareOutput(const nvinfer1::ILayer *layer,
                                   int offset,
                                   const std::string &name) {
  auto *output = layer->getOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      common::errors::InvalidArgument(
          "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(),
                    false,
                    common::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
  PADDLE_ENFORCE_EQ(output->isNetworkOutput(),
                    true,
                    common::errors::InvalidArgument(
                        "The output %s of TRT engine should be the output "
                        "of the network.",
                        name));
}

void TensorRTEngine::DeclareOutput(const std::string &name) {
  auto *output = TensorRTEngine::GetITensor(name);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      common::errors::InvalidArgument(
          "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(),
                    false,
                    common::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
}

void TensorRTEngine::DeclareOutput(const std::string &name,
                                   nvinfer1::DataType dtype) {
  auto *output = TensorRTEngine::GetITensor(name);
  DeclareOutput(name);
  output->setType(dtype);
}

void TensorRTEngine::DeleteITensor(const std::string &name,
                                   nvinfer1::ITensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      common::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      true,
      itensor_map_.count(name),
      common::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null", name));
  itensor_map_.erase(name);
}

void TensorRTEngine::SetITensor(const std::string &name,
                                nvinfer1::ITensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      common::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      0,
      itensor_map_.count(name),
      common::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be duplicated", name));
  itensor_map_[name] = tensor;
}

nvinfer1::ITensor *TensorRTEngine::GetITensor(const std::string &name,
                                              bool scalar) {
  if (scalar) {
    return ConvertWeight2ITensor(name, true);
  }
  if (itensor_map_.count(name)) {
    return itensor_map_[name];
  } else {
    ConvertWeight2ITensor(name);
    return itensor_map_[name];
  }
}

// For cases when input is not middle-tensor , but persistable tensor
// you should call this.
nvinfer1::ITensor *TensorRTEngine::ConvertWeight2ITensor(
    const std::string &name, bool scalar) {
  auto *var_v = scope_->FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var_v,
      common::errors::NotFound("You are converting a persistable weight to a "
                               "tensor, but there is no "
                               "persistable variable called %s in scope.",
                               name));
  auto *var_t = var_v->GetMutable<phi::DenseTensor>();
  auto weight = this->GetTrtWeight(name, *var_t);

  // Now we have create weights, then we need create a itensor
  auto var_dims = var_t->dims();
  nvinfer1::Dims trt_in_shape;
  trt_in_shape.nbDims = var_t->dims().size();
  for (int64_t i = 0; i < trt_in_shape.nbDims; i++) {
    trt_in_shape.d[i] = var_dims[i];
  }
  // Make 0-D tensor to 1-D tensor.
  if (trt_in_shape.nbDims == 0) {
    trt_in_shape.nbDims = 1;
    trt_in_shape.d[0] = 1;
  }
  if (scalar) {
    trt_in_shape.nbDims = 0;
    trt_in_shape.d[0] = var_dims[0];
  }
  nvinfer1::ILayer *layer =
      TRT_ENGINE_ADD_LAYER(this, Constant, trt_in_shape, weight.get());
  if (!scalar) {
    this->SetITensor(name, layer->getOutput(0));
  }
  return layer->getOutput(0);
}

std::unordered_map<std::string, nvinfer1::ITensor *>
    *TensorRTEngine::GetITensorMap() {
  return &itensor_map_;
}

void TensorRTEngine::Deserialize(const std::string &engine_serialized_data) {
  FreshDeviceId();
  infer_runtime_.reset(createInferRuntime(&logger_));

  if (use_dla()) {
    if (precision() != phi::DataType::INT8 &&
        precision() != phi::DataType::FLOAT16) {
      LOG(WARNING) << "TensorRT DLA must be used with int8 or fp16, but you "
                      "set float32, so DLA is not used.";
    } else if (infer_runtime_->getNbDLACores() == 0) {
      LOG(WARNING)
          << "TensorRT DLA is set by config, but your device does not have "
             "DLA, so DLA is not used.";
    } else {
      if (params_.dla_core < 0 ||
          params_.dla_core >= infer_runtime_->getNbDLACores()) {
        params_.dla_core = 0;
        LOG(WARNING) << "Invalid DLACore, must be 0 < DLACore < "
                     << infer_runtime_->getNbDLACores() << ", but got "
                     << params_.dla_core << ", so use use 0 as default.";
      }
      infer_runtime_->setDLACore(params_.dla_core);
      LOG(INFO) << "TensorRT DLA enabled in Deserialize(), DLACore "
                << params_.dla_core;
    }
  }

  infer_engine_.reset(infer_runtime_->deserializeCudaEngine(
      engine_serialized_data.c_str(), engine_serialized_data.size()));

  PADDLE_ENFORCE_NOT_NULL(
      infer_engine_,
      common::errors::Fatal(
          "Building TRT cuda engine failed when deserializing engine info. "
          "Please check:\n1. Your TRT serialization is generated and "
          "loaded "
          "on the same GPU architecture;\n2. The Paddle Inference version "
          "of "
          "generating serialization file and doing inference are "
          "consistent."));

#if IS_TRT_VERSION_GE(10000)
  binding_num_ = infer_engine_->getNbIOTensors();
#else
  binding_num_ = infer_engine_->getNbBindings();
#endif
  // for engine context memory sharing
  if (params_.context_memory_sharing) {
    inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
        .UpdateContextMemorySize(infer_engine_->getDeviceMemorySize(),
                                 predictor_id_per_thread);
  }
  if (params_.use_inspector) {
    GetEngineInfo(params_.engine_info_path);
  }
}

// Note: Only for support plugin.
TensorRTEngine::Weight TensorRTEngine::GetFp16TrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  phi::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    common::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));
  weight_map[name_with_suffix].reset(new phi::DenseTensor());
  weight_map[name_with_suffix]->Resize(weight_tensor.dims());

  TensorRTEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());

  // if trt not support dtype, we need to cast to fp16.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::FLOAT16);
    auto *fp16_data =
        weight_map[name_with_suffix]->mutable_data<float16>(phi::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp16_data[i] = static_cast<float16>(bf16_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT16);
    weight.SetValues(fp16_data);
  } else if (weight_tensor.dtype() == phi::DataType::FLOAT32) {
    phi::DenseTensor fp32_tensor;
    fp32_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &fp32_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::FLOAT16);
    auto *fp16_data =
        weight_map[name_with_suffix]->mutable_data<float16>(phi::CPUPlace());
    auto *fp32_data = fp32_tensor.mutable_data<float>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp16_data[i] = static_cast<float16>(fp32_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT16);
    weight.SetValues(fp16_data);
  } else if (weight_tensor.dtype() == phi::DataType::INT64) {
    phi::DenseTensor int64_tensor;
    int64_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &int64_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::INT32);
    auto *int32_data =
        weight_map[name_with_suffix]->mutable_data<int32_t>(phi::CPUPlace());
    auto *int64_data = int64_tensor.mutable_data<int64_t>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      int32_data[i] = int64_data[i];
    }
    weight.SetDataType(phi::DataType::INT32);
    weight.SetValues(int32_data);
  } else {
    paddle::framework::TensorCopySync(
        weight_tensor, cpu_place, weight_map[name_with_suffix].get());
    weight.SetDataType(weight_tensor.dtype());
    weight.SetValues(weight_map[name_with_suffix]->data());
  }
  name_suffix_counter += 1;
  return weight;
}

// Note: Only for support plugin.
TensorRTEngine::Weight TensorRTEngine::GetFp32TrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  phi::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    common::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));
  weight_map[name_with_suffix].reset(new phi::DenseTensor());
  weight_map[name_with_suffix]->Resize(weight_tensor.dims());

  TensorRTEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());

  // if trt not support dtype, we need to cast to fp32.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::FLOAT32);
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(phi::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(bf16_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT32);
    weight.SetValues(fp32_data);
  } else if (weight_tensor.dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor fp16_tensor;
    fp16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &fp16_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::FLOAT32);
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(phi::CPUPlace());
    auto *fp16_data = fp16_tensor.mutable_data<float16>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(fp16_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT32);
    weight.SetValues(fp32_data);
  } else if (weight_tensor.dtype() == phi::DataType::INT64) {
    phi::DenseTensor int64_tensor;
    int64_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &int64_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::INT32);
    auto *int32_data =
        weight_map[name_with_suffix]->mutable_data<int32_t>(phi::CPUPlace());
    auto *int64_data = int64_tensor.mutable_data<int64_t>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      int32_data[i] = int64_data[i];
    }
    weight.SetDataType(phi::DataType::INT32);
    weight.SetValues(int32_data);
  } else {
    paddle::framework::TensorCopySync(
        weight_tensor, cpu_place, weight_map[name_with_suffix].get());
    weight.SetDataType(weight_tensor.dtype());
    weight.SetValues(weight_map[name_with_suffix]->data());
  }
  name_suffix_counter += 1;
  return weight;
}

TensorRTEngine::Weight TensorRTEngine::GetTrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  phi::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    common::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));

  if (weight_tensor.place() == PlaceType::kGPU ||
      weight_tensor.dtype() != phi::DataType::FLOAT32) {
    weight_map[name_with_suffix].reset(new phi::DenseTensor());
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
  }

  TensorRTEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());

  // if trt not support dtype, we need to cast to fp32.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::FLOAT32);
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(phi::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(bf16_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT32);
    weight.SetValues(fp32_data);
  } else if (weight_tensor.dtype() == phi::DataType::INT64) {
    phi::DenseTensor int64_tensor;
    int64_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, phi::CPUPlace(), &int64_tensor);
    weight_map[name_with_suffix]->set_type(phi::DataType::INT32);
    auto *int32_data =
        weight_map[name_with_suffix]->mutable_data<int32_t>(phi::CPUPlace());
    auto *int64_data = int64_tensor.mutable_data<int64_t>(phi::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      int32_data[i] = int64_data[i];
    }
    weight.SetDataType(phi::DataType::INT32);
    weight.SetValues(int32_data);
  } else {
    if (weight_tensor.place() == PlaceType::kGPU) {
      paddle::framework::TensorCopySync(
          weight_tensor, cpu_place, weight_map[name_with_suffix].get());
      weight.SetDataType(weight_tensor.dtype());
      weight.SetValues(weight_map[name_with_suffix]->data());
    } else {
      weight.SetDataType(weight_tensor.dtype());
      weight.SetValues(weight_tensor.data());
    }
  }

  name_suffix_counter += 1;
  return weight;
}

bool TensorRTEngine::setRefitWeights(
    const std::string &weight_name, const phi::DenseTensor &new_weight_tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      infer_refitter_,
      common::errors::InvalidArgument(
          "Refitter is not initialized. Make sure you enabled refit at build "
          "time by calling use_refittable()."));

  auto new_weights = this->GetTrtWeight(weight_name, new_weight_tensor);
  const nvinfer1::Weights &final_weights = new_weights.get();

  bool set_result =
      infer_refitter_->setNamedWeights(weight_name.c_str(), final_weights);
  if (!set_result) {
    PADDLE_ENFORCE_EQ(
        set_result,
        true,
        common::errors::InvalidArgument(
            "Failed to set named weights for weight '%s' during refitting.",
            weight_name.c_str()));
    return false;
  }
  return true;
}

bool TensorRTEngine::FinalizeRefit() {
  PADDLE_ENFORCE_NOT_NULL(
      infer_refitter_,
      phi::errors::InvalidArgument(
          "Refit is not initialize.Make sure you enabled refit."));
  int missing_count = infer_refitter_->getMissingWeights(0, nullptr);
  if (missing_count > 0) {
    std::vector<const char *> missing_names(missing_count);
    infer_refitter_->getMissingWeights(missing_count, missing_names.data());
    for (int i = 0; i < missing_count; ++i) {
      LOG(ERROR) << "Missing weight: " << missing_names[i];
    }
    return false;
  }
  bool success = infer_refitter_->refitCudaEngine();
  if (!success) {
    LOG(ERROR) << "Failed to refit engine.";
    return false;
  }
  return success;
}

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPlugin(
    nvinfer1::ITensor *const *inputs,
    int num_inputs,
    plugin::PluginTensorRT *plugin) {
  owned_plugin_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPluginV2Ext(
    nvinfer1::ITensor *const *inputs,
    int num_inputs,
    plugin::PluginTensorRTV2Ext *plugin) {
  owned_plugin_v2ext_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPluginV2IOExt(
    nvinfer1::ITensor *const *inputs,
    int num_inputs,
    nvinfer1::IPluginV2IOExt *plugin) {
  owned_plugin_v2ioext_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

void TensorRTEngine::FreshDeviceId() {
  int count;
  cudaGetDeviceCount(&count);
  PADDLE_ENFORCE_LT(device_id(),
                    count,
                    common::errors::OutOfRange(
                        "Device id %d exceeds the current device count: %d.",
                        device_id(),
                        count));
  platform::SetDeviceId(device_id());
}

void TensorRTEngine::GetEngineInfo(const std::string &engine_info_path) {
#if IS_TRT_VERSION_GE(8200)
  std::unique_ptr<nvinfer1::IEngineInspector> infer_inspector(
      infer_engine_->createEngineInspector());
  auto *infer_context = context();
  infer_inspector->setExecutionContext(infer_context);
  if (engine_info_path.empty()) {
    LOG(INFO) << "====== engine info ======";
    for (int i = 0; i < infer_engine_->getNbLayers(); ++i) {
      LOG(INFO) << infer_inspector->getLayerInformation(
          i, nvinfer1::LayerInformationFormat::kJSON);
    }
    LOG(INFO) << "====== engine info end ======";
  } else {
    std::fstream out_file;
    out_file.open(engine_info_path, std::ios_base::out);
    out_file << "[";
    for (int i = 0; i < infer_engine_->getNbLayers(); ++i) {
      out_file << infer_inspector->getLayerInformation(
                      i, nvinfer1::LayerInformationFormat::kJSON)
               << "\n";
      if (i != infer_engine_->getNbLayers() - 1) {
        out_file << ",";
      }
    }
    out_file << "]";
    out_file.close();
  }
#else
  LOG(INFO) << "Inspector needs TensorRT version 8.2 and after.";
#endif
}

}  // namespace paddle::inference::tensorrt
