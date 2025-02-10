/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include "NvInferRuntimeCommon.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/tensorrt/engine_params.h"
#include "paddle/fluid/platform/tensorrt/helper.h"
#include "paddle/fluid/platform/tensorrt/trt_plugin.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/stream.h"

COMMON_DECLARE_bool(trt_ibuilder_cache);

namespace paddle {
namespace platform {

// The code is mainly from TensorRT, thanks to the project.
class TrtCudaGraph {
 public:
  TrtCudaGraph() = default;
  ~TrtCudaGraph() {
    if (cuda_graph_exec_) {
      cudaGraphExecDestroy(cuda_graph_exec_);
    }
  }

  void BeginCapture(cudaStream_t stream) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  }

  bool Launch(cudaStream_t stream) {
    return cudaGraphLaunch(cuda_graph_exec_, stream);
  }

  void EndCapture(cudaStream_t stream) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamEndCapture(stream, &cuda_graph_));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphInstantiate(
        &cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphDestroy(cuda_graph_));
  }

  void EndCaptureOnError(cudaStream_t stream) {
    // There are two possibilities why stream capture would fail:
    // (1) stream is in cudaErrorStreamCaptureInvalidated state.
    // (2) TRT reports a failure.
    // In case (1), the returning cuda_graph_ should be nullptr.
    // In case (2), the returning cuda_graph_ is not nullptr, but it should not
    // be used.
    const auto ret = cudaStreamEndCapture(stream, &cuda_graph_);
    if (ret == cudaErrorStreamCaptureInvalidated) {
      PADDLE_ENFORCE_EQ(cuda_graph_ == nullptr,
                        true,
                        common::errors::PreconditionNotMet(
                            "CudaGraph capture stream failed."));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(ret);
      PADDLE_ENFORCE_NOT_NULL(cuda_graph_,
                              common::errors::PreconditionNotMet(
                                  "CudaGraph capture stream failed."));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphDestroy(cuda_graph_));
      cuda_graph_ = nullptr;
    }
    // Clean up any cuda error.
    cudaGetLastError();
    LOG(WARNING) << "The TRT CUDA graph capture on the stream has failed.";
  }

 private:
  DISABLE_COPY_AND_ASSIGN(TrtCudaGraph);
  cudaGraph_t cuda_graph_{};
  cudaGraphExec_t cuda_graph_exec_{};
};

/*
 * TensorRT Engine.
 *
 * There are two alternative ways to use it, one is to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */
class TensorRTEngine {
  using ShapeMapType = std::map<std::string, std::vector<int>>;
  using PredictorID = int;

 public:
  // Weight is model parameter.
  class Weight {
   public:
    Weight() { w_ = nvinfer1::Weights{}; }
    Weight(nvinfer1::DataType dtype, void* value, size_t num_elem) {
      w_.type = dtype;
      w_.values = value;
      w_.count = num_elem;
    }
    const nvinfer1::Weights& get() { return w_; }

    void SetDataType(nvinfer1::DataType type) { w_.type = type; }

    void SetDataType(phi::DataType type);

    void SetValues(const void* values) { w_.values = values; }

    void SetCount(int64_t num) { w_.count = num; }

    std::vector<int64_t> dims;

   private:
    nvinfer1::Weights w_;
  };

  TensorRTEngine(const EngineParams& params,
                 nvinfer1::ILogger& logger = NaiveLogger::Global())
      : params_(params), logger_(logger) {
    dy::initLibNvInferPlugins(&logger_, "");
    static std::once_flag trt_plugin_registered;
    std::call_once(trt_plugin_registered,
                   []() { TrtPluginRegistry::Global()->RegisterToTrt(); });
  }

  // Add an input and set its name, data type and dimension.
  nvinfer1::ITensor* DeclareInput(const std::string& name,
                                  nvinfer1::DataType dtype,
                                  const nvinfer1::Dims& dim);
  // Set the offset-th output from a layer as the network's output, and set its
  // name.
  void DeclareOutput(const nvinfer1::ILayer* layer,
                     int offset,
                     const std::string& name);
  // Set the itensor_map_[name] as the network's output, and set its name.
  void DeclareOutput(const std::string& name);
  // Set the itensor_map_[name] as the network's output, and set its name and
  // data type.
  void DeclareOutput(const std::string& name, nvinfer1::DataType dtype);
  void ClearTensorMap() { itensor_map_.clear(); }

  void DeleteITensor(const std::string& name, nvinfer1::ITensor* tensor);
  void SetITensor(const std::string& name, nvinfer1::ITensor* tensor);
  // Get an ITensor called name.
  nvinfer1::ITensor* GetITensor(const std::string& name, bool scalar = false);
  nvinfer1::ITensor* ConvertWeight2ITensor(const std::string& name,
                                           bool scalar = false);
  std::unordered_map<std::string, nvinfer1::ITensor*>* GetITensorMap();

  nvinfer1::ICudaEngine* engine() { return infer_engine_.get(); }
  nvinfer1::IExecutionContext* context();

  int GetBindingsOffset() {
    return (binding_num_ / max_profile_num_) * GetProfileIndex();
  }

  int GetNbBindings() { return binding_num_; }

  void ResetContext() {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        common::errors::InvalidArgument(
            "You should build engine first and then set the context."));
    std::unique_lock<std::mutex> lock(mutex_);
    infer_context_[predictor_id_per_thread].reset(nullptr);
    infer_context_.erase(predictor_id_per_thread);
    cur_profile_num_ = 0;
  }

  nvinfer1::IHostMemory* Serialize() {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        common::errors::InvalidArgument(
            "The TensorRT engine must be built first before serialization"));
#if IS_TRT_VERSION_LT(8000)
    ihost_memory_.reset(infer_engine_->serialize());
#else
    PADDLE_ENFORCE_NOT_NULL(
        ihost_memory_,
        common::errors::InvalidArgument(
            "TensorRT >= 8.0 requires that buildSerializedNetwork is called"));
#endif
    return ihost_memory_.get();
  }

  void Deserialize(const std::string& engine_serialized_data);

  bool WithFp16() {
    bool enable_fp16 = (precision() == phi::DataType::FLOAT16);
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    // below is consistent with setFlag in engine.cc
    bool fall_back_fp16 = WithInt8() && !use_dla();
    return (enable_fp16 || fall_back_fp16) && support_fp16;
  }

  bool WithInt8() {
    bool enable_int8 = (precision() == phi::DataType::INT8);
    bool support_int8 = infer_builder_->platformHasFastInt8();
    return enable_int8 && support_int8;
  }

  nvinfer1::IPluginV2Layer* AddPluginV2IOExt(nvinfer1::ITensor* const* inputs,
                                             int num_inputs,
                                             nvinfer1::IPluginV2IOExt* plugin);

  void SetTensorDynamicRange(nvinfer1::ITensor* tensor, float range) {
    quant_dynamic_range_[tensor] = range;
  }

  // Get fp16 trt weight. If src weight is not fp16, we will cast.
  Weight GetFp16TrtWeight(const std::string& name,
                          const phi::DenseTensor& weight_tensor);

  // Get fp32 trt weight. If src weight is not fp32, we will cast.
  Weight GetFp32TrtWeight(const std::string& name,
                          const phi::DenseTensor& weight_tensor);

  // if the src weight type is fp16, then return fp16 trt weight, etc.
  Weight GetTrtWeight(const std::string& name,
                      const phi::DenseTensor& weight_tensor);

  float GetTensorDynamicRange(nvinfer1::ITensor* tensor) {
    return quant_dynamic_range_[tensor];
  }

  bool DynamicRangeIsSet(nvinfer1::ITensor* tensor) {
    return quant_dynamic_range_.count(tensor);
  }

  void SetRunFloat(const std::unordered_set<std::string>& ops) {
    trt_ops_run_float_ = ops;
  }

  bool OpIsRunFloat(const std::string& op) const {
    return trt_ops_run_float_.count(op) > 0;
  }

  // A pointer to CPU memory is needed of the TRT weight.
  // Before TRT runs, fluid loads weight into GPU storage.
  // so we need to copy the weights from GPU to CPU in our op converter.
  // We use a map to store these weights for the weight memory is not released
  // in advance, which affecting the construction of TRT Op.
  std::unordered_map<std::string /*name*/, std::unique_ptr<phi::DenseTensor>>
      weight_map;

  // When setting weight_map, a self-increasing suffix is needed for the names
  // so as to avoid repeatedly setting weights with the same name.
  void SetWeights(std::string w_name,
                  std::unique_ptr<phi::DenseTensor> w_tensor) {
    static int suffix_counter = 0;
    std::string suffix = std::to_string(suffix_counter);
    std::string splitter = "__";
    std::string name_with_suffix = w_name + splitter + suffix;
    PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                      0,
                      common::errors::AlreadyExists(
                          "The weight named %s is set into the weight map "
                          "twice in TRT OP converter.",
                          name_with_suffix));
    weight_map[name_with_suffix] = std::move(w_tensor);
    suffix_counter += 1;
  }

  void ClearWeights() {
    for (auto& weight_pair : weight_map) {
      weight_pair.second.reset(nullptr);
    }
  }

  // NOTE: The func below was modified to adapt the dynamic shape.
  // Initialize the inference network, so that TensorRT layers can add to this
  // network.
  void InitNetwork();
  // After finishing adding ops, freeze this network and creates the execution
  // environment.
  void FreezeNetwork();
  void Execute(int batch_size,
               std::vector<void*>* buffers,
               cudaStream_t stream = nullptr);

  bool Enqueue(nvinfer1::IExecutionContext* context,
               std::vector<void*>* buffers,
               int batch,
               cudaStream_t stream);

  nvinfer1::INetworkDefinition* network() { return infer_network_.get(); }

  ShapeMapType& min_input_shape() { return params_.min_input_shape; }
  ShapeMapType& max_input_shape() { return params_.max_input_shape; }
  ShapeMapType& optim_input_shape() { return params_.optim_input_shape; }
  ShapeMapType& min_shape_tensor() { return params_.min_shape_tensor; }
  ShapeMapType& max_shape_tensor() { return params_.max_shape_tensor; }
  ShapeMapType& optim_shape_tensor() { return params_.optim_shape_tensor; }

  bool AdjustDynamicShapeRange(const ShapeMapType& runtime_input_shape,
                               const ShapeMapType& runtime_shape_tensor,
                               std::vector<std::string>* changed,
                               std::vector<std::string>* tensor_changed) {
    bool ret = false;
    changed->clear();
    tensor_changed->clear();
    for (const auto& it : runtime_input_shape) {
      auto name = it.first;
      auto input_shape = it.second;
      // Make 0-D tensor to 1-D tensor.
      if (input_shape.empty()) {
        input_shape.push_back(1);
      }
      bool min_change = false;
      bool max_change = false;
      std::vector<int> bak_min_shape;
      std::vector<int> bak_max_shape;
      if (!params_.min_input_shape.count(name)) {
        params_.min_input_shape[name] = input_shape;
        params_.max_input_shape[name] = input_shape;
        params_.optim_input_shape[name] = input_shape;
        min_change = true;
        max_change = true;
        ret = true;
      } else {
        PADDLE_ENFORCE_EQ(params_.min_input_shape[name].size(),
                          input_shape.size(),
                          common::errors::InvalidArgument(
                              "TRT dynamic_shape min_input_shape %s size not "
                              "equal, the min_input_shape[%s].size()=%d"
                              ", but the runtime_input_shape[%s].size()=%d.",
                              name,
                              name,
                              params_.min_input_shape[name].size(),
                              name,
                              input_shape.size()));

        bak_min_shape = params_.min_input_shape[name];
        bak_max_shape = params_.max_input_shape[name];
        for (size_t d = 0; d < input_shape.size(); ++d) {
          if (input_shape[d] < params_.min_input_shape[name][d]) {
            ret = true;
            min_change = true;
            params_.min_input_shape[name][d] = input_shape[d];
          }
          if (input_shape[d] > params_.max_input_shape[name][d]) {
            ret = true;
            max_change = true;
            params_.max_input_shape[name][d] = input_shape[d];
          }
        }
      }
      if (min_change)
        LOG(INFO) << "refactor tensor shape range: " << name
                  << ", min_shape from " << Vec2Str(bak_min_shape) << " to "
                  << Vec2Str(params_.min_input_shape[name]);
      if (max_change)
        LOG(INFO) << "refactor tensor shape range: " << name
                  << ", max_shape from " << Vec2Str(bak_max_shape) << " to "
                  << Vec2Str(params_.max_input_shape[name]);
      if (min_change || max_change) changed->push_back(name);
    }
    for (const auto& it : runtime_shape_tensor) {
      auto name = it.first;
      auto shape_tensor = it.second;
      bool min_change = false;
      bool max_change = false;
      std::vector<int> bak_min_shape;
      std::vector<int> bak_max_shape;
      if (!params_.min_shape_tensor.count(name)) {
        params_.min_shape_tensor[name] = shape_tensor;
        params_.max_shape_tensor[name] = shape_tensor;
        params_.optim_shape_tensor[name] = shape_tensor;
        min_change = true;
        max_change = true;
        ret = true;
      } else {
        PADDLE_ENFORCE_EQ(params_.min_shape_tensor[name].size(),
                          shape_tensor.size(),
                          common::errors::InvalidArgument(
                              "TRT dynamic_shape min_shape_tensor %s size not "
                              "equal, the min_shape_tensor[%s].size()=%d"
                              ", but the runtime_shape_tensor[%s].size()=%d.",
                              name,
                              name,
                              params_.min_shape_tensor[name].size(),
                              name,
                              shape_tensor.size()));

        bak_min_shape = params_.min_shape_tensor[name];
        bak_max_shape = params_.max_shape_tensor[name];
        for (size_t d = 0; d < shape_tensor.size(); ++d) {
          if (shape_tensor[d] < params_.min_shape_tensor[name][d]) {
            ret = true;
            min_change = true;
            params_.min_shape_tensor[name][d] = shape_tensor[d];
          }
          if (shape_tensor[d] > params_.max_shape_tensor[name][d]) {
            ret = true;
            max_change = true;
            params_.max_shape_tensor[name][d] = shape_tensor[d];
          }
        }
      }
      if (min_change)
        LOG(INFO) << "refactor shape tensor range: " << name
                  << ", min_shape from " << Vec2Str(bak_min_shape) << " to "
                  << Vec2Str(params_.min_shape_tensor[name]);
      if (max_change)
        LOG(INFO) << "refactor shape tensor range: " << name
                  << ", max_shape from " << Vec2Str(bak_max_shape) << " to "
                  << Vec2Str(params_.max_shape_tensor[name]);
      if (min_change || max_change) tensor_changed->push_back(name);
    }
    return ret;
  }

  bool use_varseqlen() { return params_.use_varseqlen; }
  bool use_dla() { return params_.use_dla; }
  bool with_interleaved() { return params_.with_interleaved; }
  const std::string& tensorrt_transformer_posid() {
    return params_.tensorrt_transformer_posid;
  }
  const std::string& tensorrt_transformer_maskid() {
    return params_.tensorrt_transformer_maskid;
  }
  bool disable_trt_plugin_fp16() { return params_.disable_trt_plugin_fp16; }
  phi::DataType precision() { return params_.precision; }

  void SetProfileNum(int num) { max_profile_num_ = num; }

  void SetScope(const framework::Scope* scope) { scope_ = scope; }

  void SetAllNodesLowerToTrt(bool all_nodes_offload_to_trt) {
    // all nodes are in trt, so we can use cudaGraph to optimize runtime.
    startup_with_cudagraph_ = all_nodes_offload_to_trt;
  }

  bool LowPrecisionIOEnabled() const { return params_.enable_low_precision_io; }

  bool use_explicit_quantization() const {
    return params_.use_explicit_quantization;
  }

 private:
  // Each ICudaEngine object is bound to a specific GPU when it is instantiated,
  // ensure that the thread is associated with the correct device by calling
  // FreshDeviceId().
  void FreshDeviceId();

  void GetEngineInfo(const std::string& engine_info_path);

  int device_id() { return params_.device_id; }

  int GetProfileIndex() {
    if (max_profile_num_ > 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      return profile_index_[predictor_id_per_thread];
    } else {
      return 0;
    }
  }

 private:
  //
  // Construction parameters.
  //
  EngineParams params_;

  //
  // The following are runtime parameters.
  //

  int max_profile_num_{1};
  int cur_profile_num_{0};
  std::unordered_map<PredictorID, int> profile_index_;

  nvinfer1::ILogger& logger_;

  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, nvinfer1::ITensor* /*ITensor*/>
      itensor_map_;

  std::vector<std::unique_ptr<nvinfer1::IPluginV2IOExt>> owned_plugin_v2ioext_;

  // TensorRT related internal members
  infer_ptr<nvinfer1::IBuilder> infer_builder_;
  infer_ptr<nvinfer1::INetworkDefinition> infer_network_;
  infer_ptr<nvinfer1::IRuntime> infer_runtime_;
  infer_ptr<nvinfer1::ICudaEngine> infer_engine_;
  std::unordered_map<PredictorID, infer_ptr<nvinfer1::IExecutionContext>>
      infer_context_;
  infer_ptr<nvinfer1::IHostMemory> ihost_memory_;
  std::unordered_map<nvinfer1::ITensor*, float> quant_dynamic_range_;

  // cudagraph related
  TrtCudaGraph cuda_graph_;
  bool cudagraph_inited_{false};
  bool startup_with_cudagraph_{false};

  // Used for convert weight into Itensor
  const framework::Scope* scope_{nullptr};

  // specify run on float to avoid overflow
  std::unordered_set<std::string> trt_ops_run_float_;

#if IS_TRT_VERSION_GE(6000)
  int binding_num_;
  infer_ptr<nvinfer1::IBuilderConfig> infer_builder_config_;
  std::vector<nvinfer1::IOptimizationProfile*> optim_profiles_;
#endif
  std::mutex mutex_;

 public:
  thread_local static int predictor_id_per_thread;
};  // class TensorRTEngine

// Add a layer__ into engine__ with args ARGS.
// For example:
//
// Reference
// https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#charRNN_define_network
//
// will add a fully connected layer into the engine.
// TensorRT has too many layers, so that is not wise to add member functions for
// them, and an macro like this is more extensible when underlying TensorRT
// library add new layer supports.
#define TRT_ENGINE_ADD_LAYER(engine__, layer__, ...) \
  engine__->network()->add##layer__(__VA_ARGS__)

class TRTEngineManager {
  using PredictorID = int;
  using AllocationPtr = phi::Allocator::AllocationPtr;

 public:
  TRTEngineManager() {
    // createInferBuilder loads trt kernels and take a few second
    // But as long as one IBuilder lives, trt kernel will not be unloaded
    // Hence, a persistent IBuilder to avoid TensorRT unload/reload kernels
    if (FLAGS_trt_ibuilder_cache) {
      holder_.reset(createInferBuilder(&NaiveLogger::Global()));
    }
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.empty();
  }

  bool Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }

  TensorRTEngine* Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.at(name).get();
  }

  TensorRTEngine* Create(const std::string& name,
                         const EngineParams& params,
                         nvinfer1::ILogger& logger = NaiveLogger::Global()) {
    auto engine = std::make_unique<TensorRTEngine>(params, logger);
    std::lock_guard<std::mutex> lock(mutex_);
    engines_[name].reset(engine.release());
    return engines_[name].get();
  }

  void DeleteAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : engines_) {
      item.second.reset(nullptr);
    }
    engines_.clear();
  }

  void DeleteKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = engines_.find(key);
    if (iter != engines_.end()) {
      iter->second.reset(nullptr);
      engines_.erase(iter);
    }
  }

  void UpdateContextMemorySize(size_t mem_size, PredictorID predictor_id) {
    VLOG(3) << "TensorRT engine context memory size is "
            << mem_size / 1024.0 / 1024.0 << "MiB in predictor id "
            << predictor_id;
    bool size_updated{false};

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (max_ctx_mem_size_ < mem_size) {
        max_ctx_mem_size_ = mem_size;
        size_updated = true;
      }
    }

    if (size_updated) {
      ReleaseContextMemory(predictor_id);
    }
  }

  void* GetContextMemory(PredictorID predictor_id,
                         const phi::GPUPlace& place,
                         const phi::Stream& stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    static auto alignment = GetAlignmentSize(place);
    if (context_memorys_.count(predictor_id) == 0) {
      auto context_memory =
          memory::Alloc(place, max_ctx_mem_size_ + alignment, stream);
      context_memorys_[predictor_id] = std::move(context_memory);
    }
    return GetAlignedMemory(context_memorys_[predictor_id]->ptr(), alignment);
  }

  void ReleaseContextMemory(PredictorID predictor_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (context_memorys_.count(predictor_id)) {
      context_memorys_[predictor_id].reset(nullptr);
      context_memorys_.erase(predictor_id);
    }
  }

 private:
  size_t GetAlignmentSize(const phi::GPUPlace& place) {
    const auto& prop = platform::GetDeviceProperties(place.GetDeviceId());
    return prop.textureAlignment;
  }

  void* GetAlignedMemory(void* addr, size_t alignment) {
    return reinterpret_cast<void*>(uintptr_t(addr) & (~(alignment - 1)));
  }

  mutable std::mutex mutex_;
  size_t max_ctx_mem_size_{0};
  std::unordered_map<PredictorID, AllocationPtr> context_memorys_;
  std::unordered_map<std::string, std::unique_ptr<TensorRTEngine>> engines_;
  infer_ptr<nvinfer1::IBuilder> holder_;
};

}  // namespace platform
}  // namespace paddle
