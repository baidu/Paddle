/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
namespace paddle::inference::tensorrt {

class LinearInterpolateV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a linear_interp_v2_op op to tensorrt OP";

    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);

    auto data_layout = common::StringToDataLayout(
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_layout")));
    auto interp_method =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("interp_method"));
    bool align_corners =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("align_corners"));

    auto resize_inputs = op_desc.Inputs();
    auto input_names = op_desc.Input("X");

    auto out_w = PADDLE_GET_CONST(int, op_desc.GetAttr("out_w"));

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
#if IS_TRT_VERSION_GE(8600)
    layer->setResizeMode(nvinfer1::InterpolationMode::kLINEAR);
#else
    layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
#endif
    // scale > outputsize

#if IS_TRT_VERSION_GE(8000)
    if (align_corners == true) {
      layer->setCoordinateTransformation(
          nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS);
    } else {
      layer->setCoordinateTransformation(
          nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    }
#endif
#if !IS_TRT_VERSION_GE(8000)
    layer->setAlignCorners(align_corners);
#endif
    auto in_dim = input->getDimensions();
    float scale_w = -1.f;

    // Scales Priority: Scale(tensor) > scale(attr) > out_d/out_h/out_w(attr)
    bool has_scale_input_attr =
        (resize_inputs.find("Scale") != resize_inputs.end());
    bool has_scale_input =
        has_scale_input_attr && (!op_desc.Input("Scale").empty());
    if (has_scale_input) {
      auto* scale_var = scope.FindVar(op_desc.Input("Scale")[0]);
      auto* scale_tensor = scale_var->GetMutable<phi::DenseTensor>();
      auto* scale_d = scale_tensor->data<float>();
      scale_w = scale_d[0];
    } else {
      const std::vector<float> scale_attr =
          PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("scale"));
      if (scale_attr.size() > 0) {
        scale_w = scale_attr[0];
      }
    }

    // axis are different in static/dynamic mode
    bool with_dynamic = true;
    int w_axis = (data_layout == phi::DataLayout::kNCHW) + with_dynamic;

    if (scale_w > 0.) {
      out_w = static_cast<int>(in_dim.d[w_axis] * scale_w);
    }

    // Priority: Input(OutSize) > Input(SizeTensor)> attr(out_h/out_w) >
    // attr(scale)
    nvinfer1::ITensor* outsize_tensor = nullptr;
    if (resize_inputs.find("OutSize") != resize_inputs.end()) {
      if (!op_desc.Input("OutSize").empty()) {
        outsize_tensor = engine_->GetITensor(op_desc.Input("OutSize")[0]);
      }
    }

#if IS_TRT_VERSION_GE(8200)
    if (outsize_tensor == nullptr) {
      if (resize_inputs.find("SizeTensor") != resize_inputs.end()) {
        if (op_desc.Input("SizeTensor").size() >= 1) {
          outsize_tensor = engine_->GetITensor(op_desc.Input("SizeTensor")[0]);
        }
      }
    }
#endif

    if (out_w > 0 && !(scale_w > 0.)) {
      scale_w =
          static_cast<float>(out_w) / static_cast<float>(in_dim.d[w_axis]);
    }

    std::vector<float> scales;
    scales.push_back(1.f);
    if (data_layout == phi::DataLayout::kNCHW) {
      scales.push_back(1.f);
      scales.push_back(scale_w);
    } else if (data_layout == phi::DataLayout::kNHWC) {
      scales.push_back(scale_w);
      scales.push_back(1.f);
    }

    if (outsize_tensor != nullptr) {
      std::vector<nvinfer1::ITensor*> outsize_itensors;
      auto* input_shape = Shape(input);
      outsize_itensors.push_back(GetEleTensorOfShape(input_shape, 0));

      if (data_layout == phi::DataLayout::kNCHW) {
        outsize_itensors.push_back(GetEleTensorOfShape(input_shape, 1));
        outsize_itensors.push_back(outsize_tensor);
      } else if (data_layout == phi::DataLayout::kNHWC) {
        outsize_itensors.push_back(outsize_tensor);
        outsize_itensors.push_back(GetEleTensorOfShape(input_shape, 2));
      }
      layer->setInput(1, *Concat(outsize_itensors));
    } else {
      layer->setScales(scales.data(), scales.size());
    }

    ReplenishLayerAndOutput(
        layer, "linear_interp_v2_op", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(linear_interp_v2, LinearInterpolateV2OpConverter);
