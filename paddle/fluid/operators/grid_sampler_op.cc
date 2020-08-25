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

#include "paddle/fluid/operators/grid_sampler_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class GridSampleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GridSampler");
    OP_INOUT_CHECK(ctx->HasInput("Grid"), "Input", "Grid", "GridSampler");
    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "GridSampler");

    auto x_dims = ctx->GetInputDim("X");
    auto grid_dims = ctx->GetInputDim("Grid");
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "Input(X) of GridSampleOp should be 4-D Tensor, but "
                          "received X dimension size(%d)",
                          x_dims.size()));
    PADDLE_ENFORCE_EQ(grid_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "Input(Grid) of GridSampleOp should be 4-D Tensor, "
                          "but received X dimension size(%d)",
                          grid_dims.size()));
    if (ctx->IsRuntime() || grid_dims[3] > 0) {
      PADDLE_ENFORCE_EQ(
          grid_dims[3], 2,
          platform::errors::InvalidArgument(
              "Input(Grid) dimension[3] should be 2, but received %d",
              grid_dims[3]));
    }
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          grid_dims[0], x_dims[0],
          platform::errors::InvalidArgument(
              "Input(X) and Input(Grid) dimension[0] should be equal, but "
              "received X dimension[0](%d) != Grid dimension[0](%d)",
              x_dims[0], grid_dims[0]));
    }

    ctx->SetOutputDim("Output",
                      {x_dims[0], x_dims[1], grid_dims[1], grid_dims[2]});
    ctx->ShareLoD("X", "Output");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_CUDA
    if (platform::CanCUDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kCUDNN;
    }
#endif
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        framework::DataLayout::kAnyLayout, library_);
  }
};

class GridSampleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input data of GridSampleOp, "
             "This is a 4-D tensor with shape of [N, C, H, W]");
    AddInput(
        "Grid",
        "(Tensor) The input grid of GridSampleOp generated by AffineGridOp, "
        "This is a 4-D tensor with shape of [N, H, W, 2] is the concatenation "
        "of x and y coordinates with shape [N, H, W] in last dimension");
    AddOutput("Output", "(Tensor) Output tensor with shape [N, C, H, W]");
    AddAttr<bool>(
        "use_cudnn",
        "(bool, default true) Only used in cudnn kernel, need install cudnn")
        .SetDefault(true);

    AddAttr<bool>(
        "align_corners",
        "(bool, default true) If align_corners is true, it will project"
        "-1 and 1 to the centers of the corner pixels. Otherwise, it will "
        "project"
        "-1 and 1 to the image edges.")
        .SetDefault(true);

    AddAttr<std::string>(
        "mode",
        "(bool, default true) The interpolation method which can be 'bilinear'"
        " or 'nearest'.")
        .SetDefault("bilinear");

    AddAttr<std::string>(
        "padding_mode",
        "(bool, default true) The padding method used when source"
        "index is out of input images. It can be 'zeros', 'reflect' and "
        "'border'.")
        .SetDefault("zeros");

    AddComment(R"DOC(
      This operation samples input X by using bilinear or nearest interpolation based on 
      flow field grid, which is usually generated by affine_grid. The grid of
      shape [N, H, W, 2] is the concatenation of (grid_x, grid_y) coordinates 
      with shape [N, H, W] each, where grid_x is indexing the 4th dimension 
      (in width dimension) of input data x and grid_y is indexing the 3rd 
      dimension (in height dimension), finally results is the bilinear 
      interpolation value or nearest value of 4 nearest corner points.

      For bilinear interpolation mode:
      Step 1:
        Get (x, y) grid coordinates and scale to [0, H-1/W-1].

        grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
        grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

      Step 2:
        Indices input data X with grid (x, y) in each [H, W] area, and bilinear 
        interpolate point value by 4 nearest points.

          wn ------- y_n ------- en
          |           |           |
          |          d_n          |
          |           |           |
         x_w --d_w-- grid--d_e-- x_e
          |           |           |
          |          d_s          |
          |           |           |
          ws ------- y_s ------- wn

        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord

        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side

        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
               + ws * d_e * d_n + es * d_w * d_n
        )DOC");
  }
};

class GridSampleOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    auto input_dims = ctx->GetInputDim("X");
    auto grid_dims = ctx->GetInputDim("Grid");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), input_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Grid"))) {
      ctx->SetOutputDim(framework::GradVarName("Grid"), grid_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_CUDA
    if (platform::CanCUDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kCUDNN;
    }
#endif
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        framework::DataLayout::kAnyLayout, library_);
  }
};

template <typename T>
class GridSampleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("grid_sampler_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Grid", this->Input("Grid"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Grid"), this->InputGrad("Grid"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(grid_sampler, ops::GridSampleOp, ops::GridSampleOpMaker,
                  ops::GridSampleGradMaker<paddle::framework::OpDesc>,
                  ops::GridSampleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(grid_sampler_grad, ops::GridSampleOpGrad);

REGISTER_OP_CPU_KERNEL(
    grid_sampler,
    ops::GridSampleOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GridSampleOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    grid_sampler_grad,
    ops::GridSampleGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GridSampleGradOpKernel<paddle::platform::CPUDeviceContext, double>);
