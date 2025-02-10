/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class SquareOpConverter : public OpConverter {
 public:
  SquareOpConverter() = default;
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert a square op to tensorrt layer ";
    nvinfer1::ITensor* input_tensor =
        engine_->GetITensor(op_desc.Input("X")[0]);

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *input_tensor,
                                       *input_tensor,
                                       nvinfer1::ElementWiseOperation::kPROD);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "square", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(square, SquareOpConverter);
