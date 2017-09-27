/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/sigmoid_cross_entropy_with_logits_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SigmoidCrossEntropyWithLogitsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Labels");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2,
                      "Input(Labels)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], labels_dims[0],
                      "The 1st dimension of Input(X) and Input(Labels) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[1], labels_dims[1],
                      "The 2nd dimension of Input(X) and Input(Labels) should "
                      "be equal.");

    ctx->SetOutputDim("Y", x_dims);
    ctx->ShareLoD("X", /*->*/ "Y");
  }
};

class SigmoidCrossEntropyWithLogitsGradOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Labels");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2,
                      "Input(Labels)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(dy_dims.size(), 2, "Input(Y@Grad)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], labels_dims[0],
                      "The 1st dimension of Input(X) and Input(Labels) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[1], labels_dims[1],
                      "The 2nd dimension of Input(X) and Input(Labels) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[0], dy_dims[0],
                      "The 1st dimension of Input(X) and Input(Y@Grad) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[1], dy_dims[1],
                      "The 2nd dimension of Input(X) and Input(Y@Grad) should "
                      "be equal.");

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

// TODO(aroraabhinav) : Complete proto documentation
class SigmoidCrossEntropyWithLogitsOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  SigmoidCrossEntropyWithLogitsOpMaker(framework::OpProto* proto,
                                       framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("Labels", "");
    AddOutput("Y", "");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sigmoid_cross_entropy_with_logits,
            ops::SigmoidCrossEntropyWithLogitsOp,
            ops::SigmoidCrossEntropyWithLogitsOpMaker,
            sigmoid_cross_entropy_with_logits_grad,
            ops::SigmoidCrossEntropyWithLogitsGradOp);
REGISTER_OP_CPU_KERNEL(sigmoid_cross_entropy_with_logits,
                       ops::SigmoidCrossEntropyWithLogitsKernel<
                           paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                       ops::SigmoidCrossEntropyWithLogitsGradKernel<
                           paddle::platform::CPUPlace, float>);
