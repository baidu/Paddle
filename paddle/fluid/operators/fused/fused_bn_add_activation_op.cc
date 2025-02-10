/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_bn_add_activation_op.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle::operators {

void FusedBatchNormAddActOp::InferShape(
    framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(ctx->HasInput("Z"), "Input", "Z", "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(
      ctx->HasInput("Scale"), "Input", "Scale", "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(
      ctx->HasInput("Bias"), "Input", "Bias", "FusedBatchNormAddActOp");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(
      ctx->HasOutput("MeanOut"), "Output", "MeanOut", "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(ctx->HasOutput("VarianceOut"),
                 "Output",
                 "VarianceOut",
                 "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(ctx->HasOutput("SavedMean"),
                 "Output",
                 "SavedMean",
                 "FusedBatchNormAddActOp");
  OP_INOUT_CHECK(ctx->HasOutput("SavedVariance"),
                 "Output",
                 "SavedVariance",
                 "FusedBatchNormAddActOp");

  const auto x_dims = ctx->GetInputDim("X");
  const auto z_dims = ctx->GetInputDim("Z");
  PADDLE_ENFORCE_EQ(
      x_dims,
      z_dims,
      common::errors::InvalidArgument("ShapeError: the shapes of input "
                                      "must be equal. But received: the shape "
                                      "of input X = [%s], and the shape of "
                                      "input Y = [%s]",
                                      x_dims,
                                      z_dims));
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::InvalidArgument("ShapeError: the dimensions of input "
                                      "must greater than or equal to 2."
                                      "But received: the shape of input "
                                      "= [%s], the dimension of input = "
                                      "[%d]",
                                      x_dims,
                                      x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      common::errors::InvalidArgument("ShapeError: the dimensions of input "
                                      "must smaller than or equal to 5."
                                      "But received: the shape of input "
                                      "= [%s], the dimension of input = "
                                      "[%d]",
                                      x_dims,
                                      x_dims.size()));

  const int64_t C = x_dims[x_dims.size() - 1];

  auto scale_dim = ctx->GetInputDim("Scale");
  auto bias_dim = ctx->GetInputDim("Bias");

  PADDLE_ENFORCE_EQ(
      scale_dim.size(),
      1UL,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of scale must equal to 1."
          "But received: the shape of scale is [%s], the dimension "
          "of scale is [%d]",
          scale_dim,
          scale_dim.size()));
  PADDLE_ENFORCE_EQ(bias_dim.size(),
                    1UL,
                    common::errors::InvalidArgument(
                        "ShapeError: the dimension of bias must equal to 1."
                        "But received: the shape of bias is [%s],the dimension "
                        "of bias is [%d]",
                        bias_dim,
                        bias_dim.size()));

  bool check = true;
  if ((!ctx->IsRuntime()) &&
      (common::product(scale_dim) <= 0 || common::product(bias_dim) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0],
                      C,
                      common::errors::InvalidArgument(
                          "ShapeError: the shape of scale must equal to [%d]"
                          "But received: the shape of scale is [%d]",
                          C,
                          scale_dim[0]));
    PADDLE_ENFORCE_EQ(bias_dim[0],
                      C,
                      common::errors::InvalidArgument(
                          "ShapeError: the shape of bias must equal to [%d]"
                          "But received: the shape of bias is [%d]",
                          C,
                          bias_dim[0]));
  }
  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("MeanOut", {C});
  ctx->SetOutputDim("VarianceOut", {C});
  ctx->SetOutputDim("SavedMean", {C});
  ctx->SetOutputDim("SavedVariance", {C});
  ctx->ShareLoD("X", "Y");
}

phi::KernelKey FusedBatchNormAddActOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should be float when input tensor's dtype is float16.
  auto bn_param_type = framework::proto::VarType::FP32;

  PADDLE_ENFORCE_EQ(
      bn_param_type,
      framework::TransToProtoVarType(
          ctx.Input<phi::DenseTensor>("Scale")->dtype()),
      common::errors::InvalidArgument("Scale input should be of float type"));
  PADDLE_ENFORCE_EQ(
      bn_param_type,
      framework::TransToProtoVarType(
          ctx.Input<phi::DenseTensor>("Bias")->dtype()),
      common::errors::InvalidArgument("Bias input should be of float type"));

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

void FusedBatchNormAddActOpMaker::Make() {
  AddInput("X", "The input tensor");
  AddInput("Z", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Mean",
           "Mean is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Variance",
           "Variance is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddOutput("Y", "result after normalization");
  AddOutput("MeanOut",
            "Share memory with Mean. "
            "Store the global mean when training");
  AddOutput("VarianceOut",
            "Share memory with Variance. "
            "Store the global Variance when training");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("ReserveSpace",
            "Reserve GPU space for triggering the new semi-persistent "
            "NHWC kernel");
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                          true,
                          common::errors::InvalidArgument(
                              "'epsilon' should be between 0.0 and 0.001."));
      });
  AddAttr<std::string>("act_type", "The activation type to be fused.")
      .SetDefault("relu");
  AddComment(R"DOC(
Fused Batch Normalization with activation.

Batch Norm has been implemented as discussed in the paper:
https://arxiv.org/pdf/1502.03167.pdf
Batch Norm can be used as a normalizer function for conv2d and fully_connected operations.
Now, the required data format for FusedBatchNormAddActOp is NHWC `[batch, in_height, in_width, in_channels]`.

)DOC");
}

void FusedBatchNormAddActGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  // check input
  OP_INOUT_CHECK(
      ctx->HasInput("X"), "Input", "X", "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(
      ctx->HasInput("Scale"), "Input", "Scale", "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"),
                 "Input",
                 "SavedMean",
                 "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"),
                 "Input",
                 "SavedVariance",
                 "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                 "Input",
                 framework::GradVarName("Y"),
                 "FusedBatchNormAddActGradOp");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                 "Output",
                 framework::GradVarName("X"),
                 "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Z")),
                 "Output",
                 framework::GradVarName("Z"),
                 "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Scale")),
                 "Output",
                 framework::GradVarName("Scale"),
                 "FusedBatchNormAddActGradOp");
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Bias")),
                 "Output",
                 framework::GradVarName("Bias"),
                 "FusedBatchNormAddActGradOp");

  const auto in_dims = ctx->GetInputDim("X");
  const int C = in_dims[in_dims.size() - 1];

  ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  ctx->SetOutputDim(framework::GradVarName("Z"), in_dims);
  ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
  ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
}

phi::KernelKey FusedBatchNormAddActGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(common::errors::NotFound(
        "Can not find Y@GRAD in the execution context."));
  }
  const phi::DenseTensor *t = nullptr;
  if (var->IsType<phi::DenseTensor>()) {
    t = &var->Get<phi::DenseTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        common::errors::NotFound("Can not get the tensor value of Y@GRAD."));
  }

  return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                        ctx.GetPlace());
}

}  // namespace paddle::operators

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_bn_add_activation,
    ops::FusedBatchNormAddActOp,
    ops::FusedBatchNormAddActOpMaker,
    ops::FusedBatchNormAddActOpInferVarType,
    ops::FusedBatchNormAddActGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedBatchNormAddActGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_bn_add_activation_grad,
                  ops::FusedBatchNormAddActGradOp);
