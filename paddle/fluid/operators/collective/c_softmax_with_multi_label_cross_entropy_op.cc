/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_softmax_with_multi_label_cross_entropy_op.h"

namespace paddle {
namespace operators {

class CSoftmaxWithMultiLabelCrossEntropyOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Logits"),
                   "Input",
                   "Logits",
                   "CSoftmaxWithMultiLabelCrossEntropyOp");
    OP_INOUT_CHECK(ctx->HasInput("Label"),
                   "Input",
                   "Label",
                   "CSoftmaxWithMultiLabelCrossEntropyOp");

    OP_INOUT_CHECK(ctx->HasInput("SmoothWeight"),
                   "Input",
                   "SmoothWeight",
                   "CSoftmaxWithMultiLabelCrossEntropyOp");

    OP_INOUT_CHECK(ctx->HasOutput("Softmax"),
                   "Output",
                   "Softmax",
                   "CSoftmaxWithMultiLabelCrossEntropyOp");
    OP_INOUT_CHECK(ctx->HasOutput("Loss"),
                   "Output",
                   "Loss",
                   "CSoftmaxWithMultiLabelCrossEntropyOp");

    auto logits_dims = ctx->GetInputDim("Logits");
    auto labels_dims = ctx->GetInputDim("Label");
    auto smooth_weight_dims = ctx->GetInputDim("SmoothWeight");

    auto logits_rank = logits_dims.size();
    auto labels_rank = labels_dims.size();
    auto axis = logits_rank - 1;
    for (int i = 0; i < logits_rank; i++) {
      if (i != axis) {
        if (ctx->IsRuntime() || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
          PADDLE_ENFORCE_EQ(logits_dims[i],
                            labels_dims[i],
                            common::errors::InvalidArgument(
                                "Input(Logits) and Input(Label) should in "
                                "same shape in dimensions except axis."));
        }
      }
    }

    PADDLE_ENFORCE_GE(
        labels_dims[logits_rank - 1],
        1UL,
        common::errors::InvalidArgument(
            "the last dimension of Input(Label) should be greater than or "
            "equal to 1."
            "But received: the last dimension of Input(Label) is [%d],"
            "the last dimension is [%d]",
            labels_dims[logits_rank - 1],
            logits_rank - 1));

    for (int i = 0; i < labels_rank; ++i) {
      if (ctx->IsRuntime() ||
          (labels_dims[i] > 0 && smooth_weight_dims[i] > 0)) {
        PADDLE_ENFORCE_EQ(labels_dims[i],
                          smooth_weight_dims[i],
                          common::errors::InvalidArgument(
                              "Input(Label) and Input(SmoothWeight) should in "
                              "same shape in dimensions."));
      }
    }

    ctx->SetOutputDim("Softmax", logits_dims);

    bool sum_multi_label_loss = ctx->Attrs().Get<bool>("sum_multi_label_loss");
    if (sum_multi_label_loss) {
      labels_dims[axis] = 1;
    }
    ctx->SetOutputDim("Loss", labels_dims);

    ctx->ShareLoD("Logits", /*->*/ "Softmax");
    ctx->ShareLoD("Logits", /*->*/ "Loss");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(
        OperatorWithKernel::IndicateVarDataType(ctx, "Logits"), ctx.GetPlace());
  }
};

class CSoftmaxWithMultiLabelCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The input tensor of unscaled "
             "log probabilities, whose dimension :attr:`axis` should be scaled "
             "by softmax.");
    AddInput("Label",
             "(Tensor) The input tensor of groud truth label. Label is a "
             "Tensor<int64> in same shape with "
             "Input(Logits) except the shape in dimension :attr:`axis` as C.");
    AddInput("SmoothWeight",
             "(Tensor) The input tensor is a Tensor<float> in same shape with "
             "Input(Label)");
    AddOutput(
        "Softmax",
        "(Tensor, default: Tensor<float>), A tensor in same shape with "
        "Input(Logits). "
        "The outputs value of softmax activation by given the input batch, "
        "which will be used in backward calculation.");
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A tensor in same shape with "
              "Input(Logits) "
              "except the shape in dimension :attr:`axis` as 1. The cross "
              "entropy loss.");
    AddAttr<int64_t>("ignore_index",
                     "(int default -100) Specifies a target value "
                     "that is ignored and does not contribute to the loss.")
        .SetDefault(-100);
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>(
        "rank",
        "(int default 0) rank id for CSoftmaxWithMultiLabelCrossEntropy.")
        .SetDefault(0);
    AddAttr<int>(
        "nranks",
        "(int default 1) nranks id for CSoftmaxWithMultiLabelCrossEntropy.")
        .SetDefault(0);
    AddAttr<bool>("sum_multi_label_loss",
                  "(bool default true) sum_multi_label_loss for "
                  "CSoftmaxWithMultiLabelCrossEntropy.")
        .SetDefault(true);
    AddComment(R"DOC(
CSoftmaxWithMultiLabelCrossEntropy Operator

)DOC");
  }
};

class CSoftmaxWithMultiLabelCrossEntropyOpGrad
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Loss")),
                      true,
                      common::errors::InvalidArgument(
                          "Input(Loss@Grad) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Softmax"),
        true,
        common::errors::InvalidArgument("Input(Softmax) should be not null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Label"),
        true,
        common::errors::InvalidArgument("Input(Label) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("SmoothWeight"),
                      true,
                      common::errors::InvalidArgument(
                          "Input(SmoothWeight) should be not null."));

    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Logits")),
                      true,
                      common::errors::InvalidArgument(
                          "Output(Logits@Grad) should be not null."));

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Softmax"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Loss")),
                          ctx.GetPlace());
  }
};

template <typename T>
class CSoftmaxWithMultiLabelCrossEntropyOpGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("c_softmax_with_multi_label_cross_entropy_grad");

    op->SetInput("Softmax", this->Output("Softmax"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("SmoothWeight", this->Input("SmoothWeight"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));
  }
};

DECLARE_INPLACE_OP_INFERER(CSoftmaxWithMultiLabelCrossEntropyInplaceInferer,
                           {"Logits", "Softmax"});

DECLARE_INPLACE_OP_INFERER(CSoftmaxWithMultiLabelCrossEntropyGradInplaceInferer,
                           {"Softmax", framework::GradVarName("Logits")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_softmax_with_multi_label_cross_entropy,
                  ops::CSoftmaxWithMultiLabelCrossEntropyOp,
                  ops::CSoftmaxWithMultiLabelCrossEntropyOpMaker,
                  ops::CSoftmaxWithMultiLabelCrossEntropyOpGradMaker<
                      paddle::framework::OpDesc>,
                  ops::CSoftmaxWithMultiLabelCrossEntropyOpGradMaker<
                      paddle::imperative::OpBase>,
                  ops::CSoftmaxWithMultiLabelCrossEntropyInplaceInferer);

REGISTER_OPERATOR(c_softmax_with_multi_label_cross_entropy_grad,
                  ops::CSoftmaxWithMultiLabelCrossEntropyOpGrad,
                  ops::CSoftmaxWithMultiLabelCrossEntropyGradInplaceInferer);

PD_REGISTER_STRUCT_KERNEL(c_softmax_with_multi_label_cross_entropy,
                          CPU,
                          ALL_LAYOUT,
                          ops::CSoftmaxWithMultiLabelCrossEntropyOpCPUKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
