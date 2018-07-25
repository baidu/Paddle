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

namespace paddle {
namespace inference {
namespace tensorrt {

class Conv2dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    LOG(INFO)
        << "convert a fluid conv2d op to tensorrt conv layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(), 1);  // Y is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("Input").front());
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input("Filter").front());
    PADDLE_ENFORCE_NOT_NULL(Y_v);
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    auto* weight_data = Y_t->mutable_data<float>(platform::CPUPlace());

    PADDLE_ENFORCE(Y_t->dims().size(), 4UL);
    const int n_output = Y_t->dims()[0];
    const int filter_h = Y_t->dims()[2];
    const int filter_w = Y_t->dims()[3];

    const int groups = boost::get<int>(op_desc.GetAttr("groups"));
    const std::vector<int> dilations =
        boost::get<std::vector<int>>(op_desc.GetAttr("dilations"));
    const std::vector<int> strides =
        boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
    const std::vector<int> paddings =
        boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));

    nvinfer1::DimsHW nv_ksize(filter_h, filter_w);
    nvinfer1::DimsHW nv_dilations(dilations[0], dilations[1]);
    nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
    nvinfer1::DimsHW nv_paddings(paddings[0], paddings[1]);

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  Y_t->memory_size() / sizeof(float)};

    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Convolution, *const_cast<nvinfer1::ITensor*>(X), n_output,
        nv_ksize, weight.get(), bias.get());
    PADDLE_ENFORCE(layer != nullptr);
    layer->setStride(nv_strides);
    layer->setPadding(nv_paddings);
    layer->setDilation(nv_dilations);
    layer->setNbGroups(groups);

    auto output_name = op_desc.Output("Output").front();
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(conv2d, Conv2dOpConverter);
