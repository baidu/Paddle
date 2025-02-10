// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <NvInfer.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/tensorrt/helper.h"
#include "paddle/fluid/platform/tensorrt/trt_plugin_utils.h"
#include "paddle/phi/core/enforce.h"

namespace nvinfer1 {
class ITensor;
}  // namespace nvinfer1

PD_DECLARE_bool(profile);

namespace paddle::platform {

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

class PluginTensorRT;

typedef std::function<PluginTensorRT*(const void*, size_t)>
    PluginDeserializeFunc;

typedef std::function<PluginTensorRT*(void)> PluginConstructFunc;

// Deprecated. Do not inherit this class, please refer to PluginTensorRTV2Ext
class PluginTensorRT : public nvinfer1::IPluginV2 {
 public:
  PluginTensorRT() : with_fp16_(false) {}

  // It was used for TensorRT deserialization.
  // It should not be called by users.
  PluginTensorRT(const void* serialized_data, size_t length) {}

  virtual ~PluginTensorRT() {}

  nvinfer1::Dims const& getInputDims(int index) const {
    return input_dims_.at(index);
  }

  nvinfer1::DataType getDataType() const { return data_type_; }

  nvinfer1::PluginFormat getDataFormat() const { return data_format_; }

  // IPluginV2
  virtual const char* getPluginType() const TRT_NOEXCEPT = 0;

  virtual const char* getPluginVersion() const TRT_NOEXCEPT { return "1"; }

  int getNbOutputs() const TRT_NOEXCEPT { return 1; }

  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims* input_dims,
                                             int num_inputs) TRT_NOEXCEPT = 0;

  // Check format support. The default is FLOAT32 and kLINEAR.
  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override;

  // Configure the layer
  void configureWithFormat(const nvinfer1::Dims* input_dims,
                           int num_inputs,
                           const nvinfer1::Dims* output_dims,
                           int num_outputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int max_batch_size) TRT_NOEXCEPT override;

  // Initialize the layer for execution.
  int initialize() TRT_NOEXCEPT override { return 0; }

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override {}

  // Find the workspace size required by the layer
  size_t getWorkspaceSize(int) const TRT_NOEXCEPT override { return 0; }

  // Execute the layer
  virtual int enqueue(int batch_size,
                      const void* const* inputs,
                      void* const* outputs,
                      void* workspace,
                      cudaStream_t stream) TRT_NOEXCEPT = 0;

  // Find the size of the serialization buffer required
  virtual size_t getSerializationSize() const TRT_NOEXCEPT = 0;

  // Serialize the layer config to buffer.
  // TensorRT will call this func to serialize the configuration of TensorRT
  // engine. It should not be called by users.
  virtual void serialize(void* buffer) const TRT_NOEXCEPT = 0;

  void destroy() TRT_NOEXCEPT override { delete this; }

  virtual nvinfer1::IPluginV2* clone() const TRT_NOEXCEPT = 0;

  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
    namespace_ = plugin_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return namespace_.c_str();
  }

 protected:
  // Deserialize input_dims, max_batch_size, data_type, data_format
  void deserializeBase(void const*& serial_data,  // NOLINT
                       size_t& serial_length);    // NOLINT
  size_t getBaseSerializationSize() const;
  // Serialize input_dims, max_batch_size, data_type, data_format
  void serializeBase(void*& buffer) const;  // NOLINT

  std::vector<nvinfer1::Dims> input_dims_;
  nvinfer1::DataType data_type_;
  nvinfer1::PluginFormat data_format_;

  bool with_fp16_;

 private:
  std::string namespace_;
};

// TensorRT introduced IPluginV2Ext after 5.1, Paddle no longer supports
// versions before 5.1
class PluginTensorRTV2Ext : public nvinfer1::IPluginV2Ext {
 public:
  PluginTensorRTV2Ext() : with_fp16_(false) {}
  PluginTensorRTV2Ext(const void* serialized_data, size_t length) {}

  nvinfer1::Dims const& getInputDims(int index) const {
    return input_dims_.at(index);
  }
  nvinfer1::DataType getDataType() const { return data_type_; }
  nvinfer1::PluginFormat getDataFormat() const { return data_format_; }

  // The Func in IPluginV2Ext
  virtual nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT = 0;

  virtual bool isOutputBroadcastAcrossBatch(int32_t output_index,
                                            const bool* input_is_broadcasted,
                                            int32_t nb_inputs) const
      TRT_NOEXCEPT {
    return false;
  }

  virtual bool canBroadcastInputAcrossBatch(int32_t input_index) const
      TRT_NOEXCEPT {
    return false;
  }

  void configurePlugin(const nvinfer1::Dims* input_dims,
                       int32_t nb_inputs,
                       const nvinfer1::Dims* output_dims,
                       int32_t nb_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format,
                       int32_t max_batch_size) TRT_NOEXCEPT override;

  virtual IPluginV2Ext* clone() const TRT_NOEXCEPT = 0;

  void attachToContext(cudnnContext*,
                       cublasContext*,
                       nvinfer1::IGpuAllocator*) TRT_NOEXCEPT override {}

  void detachFromContext() TRT_NOEXCEPT override {}

  // The Func in IPluginV2
  virtual const char* getPluginType() const TRT_NOEXCEPT = 0;
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  virtual int32_t getNbOutputs() const TRT_NOEXCEPT { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int32_t index,
                                             const nvinfer1::Dims* inputs,
                                             int32_t nb_input) TRT_NOEXCEPT = 0;
  // Check format support. The default is FLOAT32 and NCHW.
  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  }
  // Initialize the layer for execution.
  // This is called when the engine is created.
  int initialize() TRT_NOEXCEPT override { return 0; }

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override {}

  // Find the workspace size required by the layer
  size_t getWorkspaceSize(int) const TRT_NOEXCEPT override { return 0; }

  // Execute the layer

  virtual int enqueue(int batch_size,
                      const void* const* inputs,
                      void* const* outputs,
                      void* workspace,
                      cudaStream_t stream) TRT_NOEXCEPT = 0;

  // Find the size of the serialization buffer required
  virtual size_t getSerializationSize() const TRT_NOEXCEPT = 0;

  // Serialize the layer config to buffer.
  // TensorRT will call this func to serialize the configuration of TensorRT
  // engine. It should not be called by users.
  virtual void serialize(void* buffer) const TRT_NOEXCEPT = 0;

  virtual void destroy() TRT_NOEXCEPT = 0;

  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
    name_space_ = plugin_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return name_space_.c_str();
  }

 protected:
  void deserializeBase(void const*& serial_data,  // NOLINT
                       size_t& serial_length);    // NOLINT
  size_t getBaseSerializationSize() const;
  void serializeBase(void*& buffer) const;  // NOLINT

 protected:
  std::vector<nvinfer1::Dims> input_dims_;
  nvinfer1::DataType data_type_;
  nvinfer1::PluginFormat data_format_;
  bool with_fp16_;

 private:
  std::string name_space_;
};

#if IS_TRT_VERSION_GE(6000)
class DynamicPluginTensorRT : public nvinfer1::IPluginV2DynamicExt {
 public:
  DynamicPluginTensorRT() : with_fp16_(false) {}
  DynamicPluginTensorRT(const void* serialized_data, size_t length) {}

  // The Func in IPluginExt or IpluginExtV2
  virtual const char* getPluginVersion() const TRT_NOEXCEPT { return "1"; }
  virtual const char* getPluginType() const TRT_NOEXCEPT = 0;
  int getNbOutputs() const TRT_NOEXCEPT { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  void terminate() TRT_NOEXCEPT override{};

  virtual size_t getSerializationSize() const TRT_NOEXCEPT = 0;
  virtual void serialize(void* buffer) const TRT_NOEXCEPT = 0;

  // The Func in IPluginV2
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT = 0;
  virtual nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT = 0;  // NOLINT

  virtual bool supportsFormatCombination(
      int pos,
      const nvinfer1::PluginTensorDesc* in_out,
      int nb_inputs,
      int nb_outputs) TRT_NOEXCEPT = 0;

  virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                               int nb_inputs,
                               const nvinfer1::DynamicPluginTensorDesc* out,
                               int nb_outputs) TRT_NOEXCEPT = 0;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override {
    return 0;
  }

  virtual int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                      const nvinfer1::PluginTensorDesc* output_desc,
                      const void* const* inputs,
                      void* const* outputs,
                      void* workspace,
                      cudaStream_t stream) TRT_NOEXCEPT = 0;

  virtual nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT = 0;
  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
    name_space_ = plugin_namespace;
  }
  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return name_space_.c_str();
  }
  virtual void destroy() TRT_NOEXCEPT = 0;

 protected:
  void deserializeBase(void const*& serial_data,  // NOLINT
                       size_t& serial_length);    // NOLINT
  size_t getBaseSerializationSize() const;
  void serializeBase(void*& buffer) const;  // NOLINT
  bool with_fp16_;

 private:
  std::string name_space_;
  std::string plugin_base_;
};
#endif

class TensorRTPluginCreator : public nvinfer1::IPluginCreator {
 public:
  TensorRTPluginCreator() = default;

  virtual const char* getPluginName() const TRT_NOEXCEPT = 0;

  virtual const char* getPluginVersion() const TRT_NOEXCEPT = 0;

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override;

  virtual nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                                 const void* serial_data,
                                                 size_t serial_length)
      TRT_NOEXCEPT = 0;

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override;

  const char* getPluginNamespace() const TRT_NOEXCEPT override;

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};

class TrtPluginRegistry {
 public:
  static TrtPluginRegistry* Global() {
    static TrtPluginRegistry registry;
    return &registry;
  }
  bool Register(const std::string& name, const std::function<void()>& func) {
    map.emplace(name, func);
    return true;
  }
  void RegisterToTrt() {
    for (auto& it : map) {
      it.second();
    }
  }

 private:
  std::unordered_map<std::string, std::function<void()>> map;
};

template <typename T>
class TrtPluginRegistrarV2 {
 public:
  TrtPluginRegistrarV2() {
    static auto func_ptr = GetPluginRegistry();
    if (func_ptr != nullptr) {
      func_ptr->registerCreator(creator, "");
    }
  }

 private:
  T creator;
};

#define REGISTER_TRT_PLUGIN_V2(name) REGISTER_TRT_PLUGIN_V2_HELPER(name)

#define REGISTER_TRT_PLUGIN_V2_HELPER(name)                     \
  UNUSED static bool REGISTER_TRT_PLUGIN_V2_HELPER##name =      \
      paddle::platform::TrtPluginRegistry::Global()->Register(  \
          #name, []() -> void {                                 \
            static paddle::platform::TrtPluginRegistrarV2<name> \
                plugin_registrar_##name{};                      \
          });

}  // namespace paddle::platform
