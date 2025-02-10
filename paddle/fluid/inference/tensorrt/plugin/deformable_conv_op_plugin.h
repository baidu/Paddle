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

#include <cstdio>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/dynload/cublas.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class DeformableConvPlugin : public nvinfer1::IPluginV2Ext {
 public:
  explicit DeformableConvPlugin(const nvinfer1::DataType data_type,
                                const nvinfer1::Weights& weights,
                                const std::vector<int>& kernel_dims,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& dilations,
                                const int groups,
                                const int deformable_groups,
                                const int im2col_step,
                                const bool with_fp16);
  explicit DeformableConvPlugin(const nvinfer1::DataType data_type,
                                const nvinfer1::Weights& weights,
                                const std::vector<int>& kernel_dims,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& dilations,
                                const int groups,
                                const int deformable_groups,
                                const int im2col_step,
                                const std::vector<int>& input_dim,
                                const std::vector<int>& offset_dim,
                                const std::vector<int>& mask_dim,
                                const std::vector<int>& output_dim,
                                const bool with_fp16);
  DeformableConvPlugin(const void* data, size_t length);
  ~DeformableConvPlugin() override;

  const char* getPluginType() const TRT_NOEXCEPT override;
  const char* getPluginVersion() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nb_input_dims) TRT_NOEXCEPT override;
  bool supportsFormat(nvinfer1::DataType type, nvinfer1::TensorFormat format)
      const TRT_NOEXCEPT override;
  size_t getWorkspaceSize(int max_batch_size) const TRT_NOEXCEPT override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
#else
  int enqueue(int batch_size,
              const void* const* inputs,
              void* const* outputs,
#endif
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;
  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override;
  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override;
  const char* getPluginNamespace() const TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_type,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;
  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool* input_is_broadcast,
                                    int nb_inputs) const TRT_NOEXCEPT override;
  bool canBroadcastInputAcrossBatch(int input_index) const
      TRT_NOEXCEPT override;

  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator)
      TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::Dims* input_dims,
                       int nb_inputs,
                       const nvinfer1::Dims* output_dims,
                       int nb_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format,
                       int max_batch_size) TRT_NOEXCEPT override;
  nvinfer1::IPluginV2Ext* clone() const TRT_NOEXCEPT override;

 private:
  template <typename T>
  int enqueue_impl(int batch_size,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);
  nvinfer1::Weights copyToDevice(const void* hostData, size_t count);
  void serializeFromDevice(void** hostBuffer,
                           const nvinfer1::Weights& deviceWeights) const;
  nvinfer1::Weights deserializeToDevice(const void** hostBuffer, size_t count);

  bool with_fp16_;
  nvinfer1::DataType data_type_;
  nvinfer1::Weights weights_;
  std::vector<int> kernel_dims_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups_;
  int deformable_groups_;
  int im2col_step_;
  std::string namespace_;

  std::vector<int> input_dim_;
  std::vector<int> offset_dim_;
  std::vector<int> mask_dim_;
  std::vector<int> output_dim_;

  cublasHandle_t cublasHandle_;
};

class DeformableConvPluginCreator : public nvinfer1::IPluginCreator {
 public:
  DeformableConvPluginCreator() = default;
  ~DeformableConvPluginCreator() override = default;

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override;
  const char* getPluginNamespace() const TRT_NOEXCEPT override;
  const char* getPluginName() const TRT_NOEXCEPT override;
  const char* getPluginVersion() const TRT_NOEXCEPT override;
  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override;
  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serial_data,
                                            size_t serial_length)
      TRT_NOEXCEPT override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};

REGISTER_TRT_PLUGIN_V2(DeformableConvPluginCreator);

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

class DeformableConvPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit DeformableConvPluginDynamic(const nvinfer1::DataType data_type,
                                       const nvinfer1::Weights& weights,
                                       const std::vector<int>& kernel_dims,
                                       const std::vector<int>& strides,
                                       const std::vector<int>& paddings,
                                       const std::vector<int>& dilations,
                                       const int groups,
                                       const int deformable_groups,
                                       const int im2col_step,
                                       const bool with_fp16);
  DeformableConvPluginDynamic(const void* data, size_t length);
  ~DeformableConvPluginDynamic() override;
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new DeformableConvPluginDynamic(data_type_,
                                           weights_,
                                           kernel_dims_,
                                           strides_,
                                           paddings_,
                                           dilations_,
                                           groups_,
                                           deformable_groups_,
                                           im2col_step_,
                                           with_fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "deformable_conv_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override { delete this; }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

 private:
  nvinfer1::Weights copyToDevice(const void* hostData, size_t count);
  void serializeFromDevice(void** hostBuffer,
                           const nvinfer1::Weights& deviceWeights) const;
  nvinfer1::Weights deserializeToDevice(const void** hostBuffer, size_t count);
  template <typename T>
  int enqueue_impl(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);
  bool with_fp16_;
  nvinfer1::DataType data_type_;
  nvinfer1::Weights weights_;
  std::vector<int> kernel_dims_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups_;
  int deformable_groups_;
  int im2col_step_;
  std::string namespace_;

  cublasHandle_t cublasHandle_;
};

class DeformableConvPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override {
    namespace_ = lib_namespace;
  }
  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return namespace_.c_str();
  }
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "deformable_conv_plugin_dynamic";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serial_data,
                                            size_t serial_length)
      TRT_NOEXCEPT override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class PIRDeformableConvPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit PIRDeformableConvPluginDynamic(const nvinfer1::DataType data_type,
                                          const nvinfer1::Weights& weights,
                                          const std::vector<int>& kernel_dims,
                                          const std::vector<int>& strides,
                                          const std::vector<int>& paddings,
                                          const std::vector<int>& dilations,
                                          const int groups,
                                          const int deformable_groups,
                                          const int im2col_step,
                                          const bool with_fp16);
  PIRDeformableConvPluginDynamic(const void* data, size_t length);
  ~PIRDeformableConvPluginDynamic() override;
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new PIRDeformableConvPluginDynamic(data_type_,
                                              weights_,
                                              kernel_dims_,
                                              strides_,
                                              paddings_,
                                              dilations_,
                                              groups_,
                                              deformable_groups_,
                                              im2col_step_,
                                              with_fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "deformable_conv_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override { delete this; }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

 private:
  nvinfer1::Weights copyToDevice(const void* hostData, size_t count);
  void serializeFromDevice(void** hostBuffer,
                           const nvinfer1::Weights& deviceWeights) const;
  nvinfer1::Weights deserializeToDevice(const void** hostBuffer, size_t count);
  template <typename T>
  int enqueue_impl(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);
  bool with_fp16_;
  nvinfer1::DataType data_type_;
  nvinfer1::Weights weights_;
  std::vector<int> kernel_dims_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups_;
  int deformable_groups_;
  int im2col_step_;
  std::string namespace_;

  cublasHandle_t cublasHandle_;
};

class PIRDeformableConvPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  PIRDeformableConvPluginDynamicCreator() = default;
  ~PIRDeformableConvPluginDynamicCreator() override = default;

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override;
  const char* getPluginNamespace() const TRT_NOEXCEPT override;
  const char* getPluginName() const TRT_NOEXCEPT override;
  const char* getPluginVersion() const TRT_NOEXCEPT override;
  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override;
  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serial_data,
                                            size_t serial_length)
      TRT_NOEXCEPT override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};

REGISTER_TRT_PLUGIN_V2(PIRDeformableConvPluginDynamicCreator);

REGISTER_TRT_PLUGIN_V2(DeformableConvPluginDynamicCreator);
#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
