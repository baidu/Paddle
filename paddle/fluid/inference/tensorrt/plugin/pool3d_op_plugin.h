// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdio.h>

#include <cassert>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

static std::vector<int> CalcOutputSize(const std::vector<int>& input_shape,
                                       const bool& ceil_mode,
                                       const bool& adaptive,
                                       const std::vector<int>& ksize,
                                       const std::vector<int>& strides,
                                       const std::vector<int>& paddings) {
  std::vector<int> output_shape = input_shape;
  if (adaptive) {
    output_shape[0] = ksize[0];
    output_shape[1] = ksize[1];
    output_shape[2] = ksize[2];
  } else {
    int output_d =
        (input_shape[0] - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
    int output_h =
        (input_shape[1] - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
    int output_w =
        (input_shape[2] - ksize[2] + 2 * paddings[2]) / strides[2] + 1;
    if (ceil_mode) {
      output_d =
          (input_shape[0] - ksize[0] + 2 * paddings[0] + strides[0] - 1) /
              strides[0] +
          1;
      output_h =
          (input_shape[1] - ksize[1] + 2 * paddings[1] + strides[1] - 1) /
              strides[1] +
          1;
      output_w =
          (input_shape[2] - ksize[2] + 2 * paddings[2] + strides[2] - 1) /
              strides[2] +
          1;
    }
    output_shape[0] = output_d;
    output_shape[1] = output_h;
    output_shape[2] = output_w;
  }
  return output_shape;
}

class Pool3DPlugin : public PluginTensorRTV2Ext {
 public:
  size_t getSerializationSize() const TRT_NOEXCEPT override;
  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  void serialize(void* buffer) const TRT_NOEXCEPT override;

  enum class Pool3DType {
    max = 0,
    avg,
  };
  Pool3DPlugin() {}
  Pool3DPlugin(bool ceil_mode,
               Pool3DType pool3d_type,
               bool adaptive,
               std::vector<int> ksize,
               std::vector<int> strides,
               std::vector<int> paddings,
               std::vector<int> input_shape)
      : ceil_mode_(ceil_mode),
        pool3d_type_(pool3d_type),
        adaptive_(adaptive),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        input_shape_(input_shape) {
    output_shape_ = input_shape_;
    std::vector<int> output_shape =
        CalcOutputSize({input_shape_[1], input_shape_[2], input_shape_[3]},
                       ceil_mode_,
                       adaptive_,
                       ksize_,
                       strides_,
                       paddings_);
    output_shape_[1] = output_shape[0];
    output_shape_[2] = output_shape[1];
    output_shape_[3] = output_shape[2];
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  Pool3DPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &ceil_mode_);
    DeserializeValue(&serialData, &serialLength, &pool3d_type_);
    DeserializeValue(&serialData, &serialLength, &adaptive_);
    DeserializeValue(&serialData, &serialLength, &ksize_);
    DeserializeValue(&serialData, &serialLength, &strides_);
    DeserializeValue(&serialData, &serialLength, &paddings_);
    DeserializeValue(&serialData, &serialLength, &input_shape_);
    DeserializeValue(&serialData, &serialLength, &output_shape_);
  }

  Pool3DPlugin* clone() const TRT_NOEXCEPT override;

  const char* getPluginType() const TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;

  int getNbOutputs() const TRT_NOEXCEPT override;

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nbInputDims) TRT_NOEXCEPT override;

  int initialize() TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override;

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize,
              const void* const* inputs,
              void** outputs,
#else
  int enqueue(int batchSize,
              const void* const* inputs,
              void* const* outputs,
#endif
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

 private:
  bool ceil_mode_;
  Pool3DType pool3d_type_;
  bool adaptive_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
};

class Pool3DPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "pool3d_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new Pool3DPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(Pool3DPluginCreator);

class Pool3DPluginDynamic : public DynamicPluginTensorRT {
 public:
  Pool3DPluginDynamic() {}
  Pool3DPluginDynamic(const bool& ceil_mode,
                      const std::string& pool3d_type,
                      const bool& adaptive,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const bool& is_global)
      : ceil_mode_(ceil_mode),
        pool3d_type_(pool3d_type),
        adaptive_(adaptive),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        is_global_(is_global) {}

  Pool3DPluginDynamic(void const* serialData, size_t serialLength);
  ~Pool3DPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;
  const char* getPluginType() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
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
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
  bool ceil_mode_;
  std::string pool3d_type_;
  bool adaptive_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool is_global_;
};

class Pool3DPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "pool3d_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new Pool3DPluginDynamic(serial_data, serial_length);
  }
};

class PIRPool3DPluginDynamic : public DynamicPluginTensorRT {
 public:
  PIRPool3DPluginDynamic() {}
  PIRPool3DPluginDynamic(const bool& ceil_mode,
                         const std::string& pool3d_type,
                         const bool& adaptive,
                         const std::vector<int>& ksize,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const bool& is_global)
      : ceil_mode_(ceil_mode),
        pool3d_type_(pool3d_type),
        adaptive_(adaptive),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        is_global_(is_global) {}

  PIRPool3DPluginDynamic(void const* serialData, size_t serialLength);
  ~PIRPool3DPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;
  const char* getPluginType() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
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
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
  bool ceil_mode_;
  std::string pool3d_type_;
  bool adaptive_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool is_global_;
};

class PIRPool3DPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "pir_pool3d_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new PIRPool3DPluginDynamic(serial_data, serial_length);
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    bool ceil_mode = false;
    std::string pool3d_type;
    bool adaptive = false;
    std::vector<int> ksize;
    std::vector<int> strides;
    std::vector<int> paddings;
    bool is_global = false;

    for (int i = 0; i < fc->nbFields; ++i) {
      const nvinfer1::PluginField& field = fc->fields[i];
      const std::string field_name(field.name);

      if (field_name.compare("ceil_mode") == 0) {
        ceil_mode = *static_cast<const bool*>(field.data);
      } else if (field_name.compare("pool3d_type") == 0) {
        pool3d_type = std::string(static_cast<const char*>(fc->fields[i].data),
                                  fc->fields[i].length);
      } else if (field_name.compare("adaptive") == 0) {
        adaptive = *static_cast<const bool*>(field.data);
      } else if (field_name.compare("ksize") == 0) {
        const int length = fc->fields[i].length;
        const int* data = static_cast<const int*>(fc->fields[i].data);
        ksize.insert(ksize.end(), data, data + length);
      } else if (field_name.compare("strides") == 0) {
        const int length = fc->fields[i].length;
        const int* data = static_cast<const int*>(fc->fields[i].data);
        strides.insert(strides.end(), data, data + length);
      } else if (field_name.compare("paddings") == 0) {
        const int length = fc->fields[i].length;
        const int* data = static_cast<const int*>(fc->fields[i].data);
        paddings.insert(paddings.end(), data, data + length);
      } else if (field_name.compare("is_global") == 0) {
        is_global = *static_cast<const bool*>(field.data);
      } else {
        assert(false && "Unknown plugin field name.");
      }
    }
    return new Pool3DPluginDynamic(
        ceil_mode, pool3d_type, adaptive, ksize, strides, paddings, is_global);
  }
};
REGISTER_TRT_PLUGIN_V2(Pool3DPluginDynamicCreator);
REGISTER_TRT_PLUGIN_V2(PIRPool3DPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
