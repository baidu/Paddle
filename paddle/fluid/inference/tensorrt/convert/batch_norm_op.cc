/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace nvinfer1 {
class IScaleLayer;  // NOLINT
}  // namespace nvinfer1

namespace paddle::inference::tensorrt {

class BatchNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a batch norm op to tensorrt batch_norm";

    framework::OpDesc op_desc(op, nullptr);
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    // Declare weights
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Mean_v = scope.FindVar(op_desc.Input("Mean").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    auto* Variance_v = scope.FindVar(op_desc.Input("Variance").front());
    const float eps = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));
    auto output_name = op_desc.Output("Y").front();
    PADDLE_ENFORCE_NOT_NULL(
        Bias_v,
        common::errors::NotFound(
            "Variable of Bias of batch_norm TRT converter is not found."));
    PADDLE_ENFORCE_NOT_NULL(
        Mean_v,
        common::errors::NotFound(
            "Variable of Mean of batch_norm TRT converter is not found."));
    PADDLE_ENFORCE_NOT_NULL(
        Scale_v,
        common::errors::NotFound(
            "Variable of Scale of batch_norm TRT converter is not found."));
    PADDLE_ENFORCE_NOT_NULL(
        Variance_v,
        common::errors::NotFound(
            "Variable of Variance of batch_norm TRT converter is not found."));

    // get tensor
    auto* Bias_t = Bias_v->GetMutable<phi::DenseTensor>();
    auto* Mean_t = Mean_v->GetMutable<phi::DenseTensor>();
    auto* Scale_t = Scale_v->GetMutable<phi::DenseTensor>();
    auto* Variance_t = Variance_v->GetMutable<phi::DenseTensor>();

    // create temp tensor for weights
    phi::DenseTensor bias_tensor;
    phi::DenseTensor mean_tensor;
    phi::DenseTensor scale_tensor;
    phi::DenseTensor variance_tensor;

    bias_tensor.Resize(Bias_t->dims());
    mean_tensor.Resize(Mean_t->dims());
    scale_tensor.Resize(Scale_t->dims());
    variance_tensor.Resize(Variance_t->dims());

    phi::CPUPlace cpu_place;
    // copy data from gpu to cpu
    paddle::framework::TensorCopySync((*Bias_t), cpu_place, &bias_tensor);
    paddle::framework::TensorCopySync((*Mean_t), cpu_place, &mean_tensor);
    paddle::framework::TensorCopySync((*Scale_t), cpu_place, &scale_tensor);
    paddle::framework::TensorCopySync(
        (*Variance_t), cpu_place, &variance_tensor);

    auto* bias_data = bias_tensor.mutable_data<float>(phi::CPUPlace());
    auto* mean_data = mean_tensor.mutable_data<float>(phi::CPUPlace());
    auto* scale_data = scale_tensor.mutable_data<float>(phi::CPUPlace());
    auto* variance_data = variance_tensor.mutable_data<float>(phi::CPUPlace());

    std::unique_ptr<phi::DenseTensor> combine_scale_tensor(
        new phi::DenseTensor());
    std::unique_ptr<phi::DenseTensor> combine_bias_tensor(
        new phi::DenseTensor());

    combine_scale_tensor->Resize(scale_tensor.dims());
    combine_bias_tensor->Resize(bias_tensor.dims());

    auto* combine_scale_data =
        combine_scale_tensor->mutable_data<float>(phi::CPUPlace());
    auto* combine_bias_data =
        combine_bias_tensor->mutable_data<float>(phi::CPUPlace());

    size_t ele_num = combine_scale_tensor->memory_size() / sizeof(float);

    for (size_t i = 0; i < ele_num; i++) {
      float scale = scale_data[i];
      float bias = bias_data[i];
      float mean = mean_data[i];
      float variance = variance_data[i];
      combine_scale_data[i] = scale / sqrtf(variance + eps);
      combine_bias_data[i] = bias - mean * combine_scale_data[i];
    }

    TensorRTEngine::Weight scale_weights{
        nvinfer1::DataType::kFLOAT,
        static_cast<void*>(combine_scale_data),
        combine_scale_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight shift_weights{
        nvinfer1::DataType::kFLOAT,
        static_cast<void*>(combine_bias_data),
        combine_bias_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight power_weights{
        nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ILayer* layer = nullptr;
    nvinfer1::IShuffleLayer* expand_layer = nullptr;
    nvinfer1::IShuffleLayer* squeeze_layer = nullptr;

    auto x_dim = X->getDimensions();
    if (x_dim.nbDims < 4) {
      nvinfer1::Dims expand_shape;
      expand_shape.nbDims = 4;
      for (int i = 0; i < 4; i++) {
        if (i < x_dim.nbDims) {
          expand_shape.d[i] = x_dim.d[i] < 0 ? 0 : x_dim.d[i];
        } else {
          expand_shape.d[i] = 1;
        }
      }
      expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      expand_layer->setReshapeDimensions(expand_shape);
      X = expand_layer->getOutput(0);
      expand_layer->getOutput(0)->setName(
          ("reshape_before_batchnorm_out: " + output_name).c_str());
      expand_layer->setName(
          ("BN_Shuffle: (Output: " + output_name + ")").c_str());
    }

    layer = TRT_ENGINE_ADD_LAYER(engine_,
                                 ScaleNd,
                                 *X,
                                 nvinfer1::ScaleMode::kCHANNEL,
                                 shift_weights.get(),
                                 scale_weights.get(),
                                 power_weights.get(),
                                 1);

    engine_->SetWeights(op_desc.Input("Bias").front(),
                        std::move(combine_bias_tensor));
    engine_->SetWeights(op_desc.Input("Scale").front(),
                        std::move(combine_scale_tensor));
    if (x_dim.nbDims < 4) {
      layer->getOutput(0)->setName(("BN: ScaleNd: " + output_name).c_str());
      layer->setName(("BN: ScaleNd: (Output: " + output_name + ")").c_str());
      nvinfer1::Dims squeeze_shape;
      squeeze_shape.nbDims = x_dim.nbDims;
      for (int i = 0; i < squeeze_shape.nbDims; i++) {
        squeeze_shape.d[i] = x_dim.d[i] < 0 ? 0 : x_dim.d[i];
      }
      squeeze_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
      squeeze_layer->setReshapeDimensions(squeeze_shape);
      ReplenishLayerAndOutput(
          squeeze_layer, "batchnorm_add_scale", {output_name}, test_mode);
    } else {
      ReplenishLayerAndOutput(
          layer, "batchnorm_add_scale", {output_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(batch_norm, BatchNormOpConverter);
