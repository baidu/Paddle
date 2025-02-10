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

class YoloBoxPlugin : public nvinfer1::IPluginV2Ext {
 public:
  explicit YoloBoxPlugin(const nvinfer1::DataType data_type,
                         const std::vector<int>& anchors,
                         const int class_num,
                         const float conf_thresh,
                         const int downsample_ratio,
                         const bool clip_bbox,
                         const float scale_x_y,
                         const bool iou_aware,
                         const float iou_aware_factor,
                         const int input_h,
                         const int input_w);
  YoloBoxPlugin(const void* data, size_t length);
  ~YoloBoxPlugin() override;

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
  template <typename T>
  int enqueue_impl(int batch_size,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);
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
  nvinfer1::DataType data_type_;
  std::vector<int> anchors_;
  int* anchors_device_;
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
  int input_h_;
  int input_w_;
  bool iou_aware_;
  float iou_aware_factor_;
  std::string namespace_;
};

class YoloBoxPluginCreator : public nvinfer1::IPluginCreator {
 public:
  YoloBoxPluginCreator();
  ~YoloBoxPluginCreator() override = default;

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

class PIRYoloBoxPlugin : public nvinfer1::IPluginV2Ext {
 public:
  explicit PIRYoloBoxPlugin(const nvinfer1::DataType data_type,
                            const std::vector<int>& anchors,
                            const int class_num,
                            const float conf_thresh,
                            const int downsample_ratio,
                            const bool clip_bbox,
                            const float scale_x_y,
                            const bool iou_aware,
                            const float iou_aware_factor,
                            const int input_h,
                            const int input_w);
  PIRYoloBoxPlugin(const void* data, size_t length);
  ~PIRYoloBoxPlugin() override;

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
  template <typename T>
  int enqueue_impl(int batch_size,
                   const void* const* inputs,
                   void* const* outputs,
                   void* workspace,
                   cudaStream_t stream);
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
  nvinfer1::DataType data_type_;
  std::vector<int> anchors_;
  int* anchors_device_;
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
  int input_h_;
  int input_w_;
  bool iou_aware_;
  float iou_aware_factor_;
  std::string namespace_;
};

class PIRYoloBoxPluginCreator : public nvinfer1::IPluginCreator {
 public:
  PIRYoloBoxPluginCreator();
  ~PIRYoloBoxPluginCreator() override = default;

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

REGISTER_TRT_PLUGIN_V2(YoloBoxPluginCreator);
REGISTER_TRT_PLUGIN_V2(PIRYoloBoxPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
