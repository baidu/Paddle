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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class YoloBoxHeadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "yolo_box_head");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "yolo_box_head");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class YoloBoxHeadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "The input tensor");
    AddAttr<std::vector<int>>("anchors",
                              "The anchor width and height, "
                              "it will be parsed pair by pair.");
    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<float>("conf_thresh",
                   "The confidence scores threshold of detection boxes. "
                   "Boxes with confidence scores under threshold should "
                   "be ignored.");
    AddAttr<int>("downsample_ratio",
                 "The downsample ratio from network input to YoloBox operator "
                 "input, so 32, 16, 8 should be set for the first, second, "
                 "and thrid YoloBox operators.");
    AddAttr<bool>("clip_bbox",
                  "Whether clip output bonding box in Input(ImgSize) "
                  "boundary. Default true.");
    AddAttr<float>("scale_x_y",
                   "Scale the center point of decoded bounding "
                   "box. Default 1.0");
    AddOutput("Out", "The output tensor");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolo_box_head, ops::YoloBoxHeadOp, ops::YoloBoxHeadOpMaker);
