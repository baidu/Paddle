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

namespace paddle::inference::tensorrt {

/*
 * Scale Op
 */
class ScaleOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a scale op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> itensors;
    std::string input_name = op_desc.Input("X").front();
    std::string out_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    bool bias_after_scale =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("bias_after_scale"));
    float bias = PADDLE_GET_CONST(float, op_desc.GetAttr("bias"));
    float scale = PADDLE_GET_CONST(float, op_desc.GetAttr("scale"));
    bool is_int = input->getType() == nvinfer1::DataType::kINT32;
    nvinfer1::ILayer* layer = nullptr;
    nvinfer1::ITensor* bias_tensor =
        is_int ? Add1DConstantLayer(
                     static_cast<int>(bias > 0 ? bias + 0.5 : bias - 0.5))
               : Add1DConstantLayer(bias);
    bool is_bias_0 = bias == 0;

    std::vector<int32_t> bias_shapes(input->getDimensions().nbDims, 1);
    auto* bias_shapes_tensor = Add1DConstantLayer(bias_shapes);
    auto* reshape_layer_bias =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *bias_tensor);
    reshape_layer_bias->setInput(1, *bias_shapes_tensor);

    bool has_scale_tensor;
    nvinfer1::ITensor* scale_tensor;
    bool is_scale_1;

    auto scale_inputs = op_desc.Inputs();
    if (scale_inputs.find("ScaleTensor") != scale_inputs.end() &&
        !op_desc.Input("ScaleTensor").empty()) {  // has EndsTensor input
      has_scale_tensor = true;
      scale_tensor = engine_->GetITensor(op_desc.Input("ScaleTensor")[0]);
      is_scale_1 = false;
    } else {
      has_scale_tensor = false;
      scale_tensor = is_int ? Add1DConstantLayer(static_cast<int>(
                                  scale > 0 ? scale + 0.5 : scale - 0.5))
                            : Add1DConstantLayer(scale);
      is_scale_1 = scale == 1;
    }

    std::vector<int32_t> scale_shapes(input->getDimensions().nbDims, 1);
    auto* scale_shapes_tensor = Add1DConstantLayer(scale_shapes);
    auto* reshape_layer_scale =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *scale_tensor);
    reshape_layer_scale->setInput(1, *scale_shapes_tensor);

    if (!has_scale_tensor && is_scale_1 && is_bias_0) {
      layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input);
    } else {
      if (bias_after_scale) {
        if (!is_scale_1) {
          layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *input,
                                       *reshape_layer_scale->getOutput(0),
                                       nvinfer1::ElementWiseOperation::kPROD);
          input = layer->getOutput(0);
        }
        if (!is_bias_0) {
          layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *input,
                                       *reshape_layer_bias->getOutput(0),
                                       nvinfer1::ElementWiseOperation::kSUM);
        }
      } else {
        if (!is_bias_0) {
          layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *input,
                                       *reshape_layer_bias->getOutput(0),
                                       nvinfer1::ElementWiseOperation::kSUM);
          input = layer->getOutput(0);
        }
        if (!is_scale_1) {
          layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *input,
                                       *reshape_layer_scale->getOutput(0),
                                       nvinfer1::ElementWiseOperation::kPROD);
        }
      }
    }
    ReplenishLayerAndOutput(layer, "scale", {out_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(scale, ScaleOpConverter);
