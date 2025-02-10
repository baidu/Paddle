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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
class RoiAlignPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit RoiAlignPluginDynamic(const nvinfer1::DataType data_type,
                                 const int pooled_height,
                                 const int pooled_width,
                                 float spatial_scale,
                                 int sampling_ratio,
                                 bool aligned);
  RoiAlignPluginDynamic(void const* data, size_t length);
  ~RoiAlignPluginDynamic() = default;
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder)  // NOLINT
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

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  const char* getPluginType() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;
  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override;
  const char* getPluginVersion() const TRT_NOEXCEPT override;

 private:
  template <typename T, typename OutT>
  int enqueue_impl(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);

  nvinfer1::DataType data_type_;
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
  int sampling_ratio_;
  int smem_per_block_;
  bool aligned_;
  std::string namespace_;
};

class RoiAlignPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  RoiAlignPluginDynamicCreator();
  ~RoiAlignPluginDynamicCreator() override = default;

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

class PIRRoiAlignPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit PIRRoiAlignPluginDynamic(const nvinfer1::DataType data_type,
                                    const int pooled_height,
                                    const int pooled_width,
                                    float spatial_scale,
                                    int sampling_ratio,
                                    bool aligned);
  PIRRoiAlignPluginDynamic(void const* data, size_t length);
  ~PIRRoiAlignPluginDynamic() = default;
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder)  // NOLINT
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

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  const char* getPluginType() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;
  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override;
  const char* getPluginVersion() const TRT_NOEXCEPT override;

 private:
  template <typename T, typename OutT>
  int enqueue_impl(const nvinfer1::PluginTensorDesc* inputDesc,
                   const nvinfer1::PluginTensorDesc* outputDesc,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);

  nvinfer1::DataType data_type_;
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
  int sampling_ratio_;
  int smem_per_block_;
  bool aligned_;
  std::string namespace_;
};

class PIRRoiAlignPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  PIRRoiAlignPluginDynamicCreator();
  ~PIRRoiAlignPluginDynamicCreator() override = default;

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

REGISTER_TRT_PLUGIN_V2(RoiAlignPluginDynamicCreator);
REGISTER_TRT_PLUGIN_V2(PIRRoiAlignPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
