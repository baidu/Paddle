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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace paddle {
namespace operators {

class SliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "slice");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "slice");

    // Case 1: Special treatment when input is a tensor array.
    auto x_var_type = ctx->GetInputsVarType("Input")[0];
    auto axes = ctx->Attrs().Get<std::vector<int>>("axes");
    if (x_var_type == framework::proto::VarType::DENSE_TENSOR_ARRAY) {
      PADDLE_ENFORCE_EQ(axes.size(),
                        1,
                        common::errors::InvalidArgument(
                            "The size of axes must be 1 when the Input of "
                            "SliceOp is phi::TensorArray, "
                            "but received %d.",
                            axes.size()));
      if (ctx->IsRuntime()) {
        // If the var type of input is DENSE_TENSOR_ARRAY,
        // the output shape is determined by SliceKernel:Compute in runtime.
        return;
      } else {
        // NOTE(liym27): A better way is needed to get accurate dims of tensor
        // array.
        // The resulted dim of GetInputDim("Input") is the dim of the
        // last item written into TensorArray "Input". Maybe it's a bug to fix.
        ctx->SetOutputDim("Out", ctx->GetInputDim("Input"));
        return;
      }
    }

    // Case 2: input is a tensor.
    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_LT(in_dims.size(),
                      7,
                      common::errors::InvalidArgument(
                          "The rank of input should be less than 7."));
    phi::DDim out_dims(in_dims);

    auto starts = ctx->Attrs().Get<std::vector<int>>("starts");
    auto ends = ctx->Attrs().Get<std::vector<int>>("ends");
    auto decrease_axis = ctx->Attrs().Get<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx->Attrs().Get<std::vector<int>>("infer_flags");
    if (infer_flags.empty()) {
      // Initialize infer_flags with 1.
      // To be compatible with other op tests in which infer_flags is not set.
      infer_flags = std::vector<int>(axes.size(), 1);
    }

    // 2.1 Check attrs.
    auto starts_size = starts.size();
    auto ends_size = ends.size();

    if (ctx->HasInputs("StartsTensorList")) {
      starts_size = ctx->Inputs("StartsTensorList").size();
      PADDLE_ENFORCE_GT(starts_size,
                        0,
                        common::errors::InvalidArgument(
                            "StartsTensorList size can't be zero"));
    }
    if (ctx->HasInputs("EndsTensorList")) {
      ends_size = ctx->Inputs("EndsTensorList").size();
      PADDLE_ENFORCE_GT(
          ends_size,
          0,
          common::errors::InvalidArgument("EndsTensorList size can't be zero"));
    }

    if (!ctx->HasInput("StartsTensor")) {
      PADDLE_ENFORCE_EQ(
          starts_size,
          axes.size(),
          common::errors::InvalidArgument(
              "The size of starts must be equal to the size of axes."));
    }
    if (!ctx->HasInput("EndsTensor")) {
      PADDLE_ENFORCE_EQ(
          ends_size,
          axes.size(),
          common::errors::InvalidArgument(
              "The size of ends must be equal to the size of axes."));
    }
    for (auto &axis : axes) {
      if (axis < 0) {
        axis = std::max(0, axis + in_dims.size());
      }
    }
    phi::funcs::CheckAndUpdateSliceAttrs<int>(
        in_dims, axes, &starts, &ends, nullptr, &infer_flags);

    auto slice_dims = phi::funcs::GetSliceDims<int>(
        in_dims, axes, starts, ends, nullptr, &infer_flags);
    if (ctx->IsRuntime()) {
      out_dims = phi::funcs::GetDecreasedDims<int>(
          slice_dims, decrease_axis, &infer_flags);
    } else {
      out_dims =
          phi::funcs::GetDecreasedDims<int>(slice_dims, decrease_axis, nullptr);
    }

    ctx->SetOutputDim("Out", out_dims);
    if (!axes.empty() && axes[0] != 0) {
      ctx->ShareLoD("Input", /*->*/ "Out");
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *in_var = ctx.InputVar("Input");
    if (in_var->IsType<phi::DenseTensor>()) {
      auto &in_tensor = in_var->Get<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(
          in_tensor.IsInitialized(),
          true,
          common::errors::InvalidArgument(
              "The tensor Input (Input) of Slice op is not initialized."));
      // NOTE: cuda pinned tensor need to copy its data to target place
      if (in_tensor.place().GetType() == phi::AllocationType::GPUPINNED) {
        return phi::KernelKey(framework::TransToProtoVarType(in_tensor.dtype()),
                              ctx.GetPlace());
      }

#ifdef PADDLE_WITH_DNNL
      auto input_data_type =
          framework::OperatorWithKernel::IndicateVarDataType(ctx, "Input");
      auto vec_dims = common::vectorize(in_tensor.dims());
      bool all_zero_dims = std::all_of(
          vec_dims.cbegin(), vec_dims.cend(), [](int64_t i) { return i == 0; });
      if (!all_zero_dims && this->CanMKLDNNBeUsed(ctx, input_data_type)) {
        // OneDNN uses blocking format, which cannot be always supported with
        // reorders, because if blocked dimension is not divisible by 8 or
        // 16(depending on which blocking format is used) submemory cannot be
        // created, so in that scenario a fallback is needed
        if (ctx.Input<phi::DenseTensor>("Input")
                ->mem_desc()
                .get_inner_nblks() == 0) {
          return phi::KernelKey(phi::Backend::ONEDNN,
                                phi::DataLayout::ONEDNN,
                                phi::TransToPhiDataType(input_data_type));
        }
      }
#endif

      return phi::KernelKey(framework::TransToProtoVarType(in_tensor.dtype()),
                            in_tensor.place());
    }
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
                          ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class SliceOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = "Input";
    auto out_name = "Out";
    auto decrease_axis = ctx->GetAttr("decrease_axis");
    auto not_decrease = paddle::get<std::vector<int>>(decrease_axis).empty();
    if (not_decrease) {
      // The default type of out is phi::DenseTensor.
      // However, if no axis is decreased and the type of input is not
      // phi::DenseTensor, the type of out should be the same as input.
      // For example, input is a phi::TensorArray and no axis is decreased, the
      // output should be a phi::TensorArray.
      ctx->SetOutputType(out_name, ctx->GetInputType(x_name));
      ctx->SetOutputDataType(out_name, ctx->GetInputDataType(x_name));
    }
  }
};

class SliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) Tensor of data to extract slices from.");
    AddInput("StartsTensor",
             "(Tensor<int32>, optional) If provided, slice will use this."
             "It has the highest priority of StartsTensor, StartsTensorList "
             "and attr(starts).")
        .AsDispensable();
    AddInput("EndsTensor",
             "(Tensor<int32>, optional) If provided, slice will use this."
             "It has the highest priority of EndsTensor, EndsTensorList and "
             "attr(ends).")
        .AsDispensable();
    AddInput(
        "StartsTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(starts).")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "EndsTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(ends).")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "Sliced data tensor.");
    AddAttr<std::vector<int>>(
        "axes",
        "(list<int>) Axes that `starts` and `ends` apply to. It's optional."
        "If not present, will be treated as [0, 1, ..., len(`starts`) - 1].");
    AddAttr<std::vector<int>>(
        "starts",
        "(list<int>) Starting indices of corresponding axis in `axes`")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "ends", "(list<int>) Ending indices of corresponding axis in `axes`.")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "infer_flags", "(list<int>) Flags of inferring dims in attributes.")
        .SetDefault({});
    AddAttr<std::vector<int>>("decrease_axis", "(list<int>) decrease_axis")
        .SetDefault({});
    AddComment(R"DOC(
Slice Operator.

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slice uses `axes`, `starts` and `ends` attributes to specify the start and
end dimension for each axis in the list of axes, it uses this information
to slice the input data tensor. If a negative value is passed for any of
the start or end indices, it represents number of elements before the end
of that dimension. If the value passed to start or end is larger than
the n (the number of elements in this dimension), it represents n.
For slicing to the end of a dimension with unknown size, it is recommended
to pass in INT_MAX. The size of axes must be equal to starts\' and ends\'.
Following examples will explain how slice works:

.. code-block:: text

    Case1:
        Given:
            data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
            axes = [0, 1]
            starts = [1, 0]
            ends = [2, 3]
        Then:
            result = [ [5, 6, 7], ]

    Case2:
        Given:
            data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
            starts = [0, 1]
            ends = [-1, 1000]
        Then:
            result = [ [2, 3, 4], ]
)DOC");
  }
};

class SliceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"),
        true,
        common::errors::InvalidArgument("Input should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")),
        true,
        common::errors::InvalidArgument("Input(Out@GRAD) should not be null"));
    auto x_var_type = ctx->GetInputsVarType("Input")[0];
    if (x_var_type == framework::proto::VarType::DENSE_TENSOR_ARRAY) {
      // If the var type of input is DENSE_TENSOR_ARRAY,
      // the output shape is determined by SliceGradKernel:Compute in runtime.
      if (ctx->IsRuntime()) {
        return;
      }
    }
    auto x_dims = ctx->GetInputDim("Input");
    auto x_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_DNNL
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      // OneDNN uses blocking format, which cannot be always supported with
      // reorders, because if blocked dimension is not divisible by 8 or
      // 16(depending on which blocking format is used) submemory cannot be
      // created, so in that scenario a fallback is needed
      if (ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"))
              ->mem_desc()
              .get_inner_nblks() == 0) {
        return phi::KernelKey(phi::Backend::ONEDNN,
                              phi::DataLayout::ONEDNN,
                              phi::TransToPhiDataType(input_data_type));
      }
    }
#endif
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class SliceOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x = "Input";
    auto d_out = framework::GradVarName("Out");
    auto out = framework::GradVarName("Input");
    // The types of grad_input and input should always be the same.
    // The default type of out is phi::DenseTensor, but the type of input can be
    // phi::DenseTensor or phi::DenseTensorArray,
    // so set the type of both to be the same.
    ctx->SetOutputType(out, ctx->GetInputType(x));
    ctx->SetOutputDataType(out, ctx->GetInputDataType(d_out));
  }
};

template <typename T>
class SliceOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput("Input", this->Input("Input"));
    if (this->HasInput("StartsTensor")) {
      bind->SetInput("StartsTensor", this->Input("StartsTensor"));
    }
    if (this->HasInput("EndsTensor")) {
      bind->SetInput("EndsTensor", this->Input("EndsTensor"));
    }
    if (this->HasInput("StartsTensorList")) {
      bind->SetInput("StartsTensorList", this->Input("StartsTensorList"));
    }
    if (this->HasInput("EndsTensorList")) {
      bind->SetInput("EndsTensorList", this->Input("EndsTensorList"));
    }
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("slice_grad");
  }
};

class SliceCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor input = this->GetSingleForwardInput("Input");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor input_grad = this->GetSingleInputGrad("Input");

    auto dx_ptr = this->GetOutputPtr(&input_grad);
    std::string dx_name = this->GetOutputName(input_grad);
    auto axes = this->Attr<std::vector<int>>("axes");
    auto starts = this->Attr<std::vector<int>>("starts");
    auto ends = this->Attr<std::vector<int>>("ends");
    auto infer_flags = this->Attr<std::vector<int>>("infer_flags");
    auto decrease_axis = this->Attr<std::vector<int>>("decrease_axis");
    VLOG(6) << "Running slice_grad composite func";
    std::vector<int64_t> new_axes =
        std::vector<int64_t>(axes.begin(), axes.end());
    std::vector<int64_t> new_infer_flags =
        std::vector<int64_t>(infer_flags.begin(), infer_flags.end());
    std::vector<int64_t> new_decrease_axis =
        std::vector<int64_t>(decrease_axis.begin(), decrease_axis.end());
    prim::slice_grad<prim::DescTensor>(input,
                                       out_grad,
                                       new_axes,
                                       paddle::experimental::IntArray(starts),
                                       paddle::experimental::IntArray(ends),
                                       new_infer_flags,
                                       new_decrease_axis,
                                       dx_ptr);
    this->RecoverOutputName(input_grad, dx_name);
  }
};
template <typename T>
class SliceDoubleOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    if (this->HasInput("StartsTensor")) {
      bind->SetInput("StartsTensor", this->Input("StartsTensor"));
    }
    if (this->HasInput("EndsTensor")) {
      bind->SetInput("EndsTensor", this->Input("EndsTensor"));
    }
    if (this->HasInput("StartsTensorList")) {
      bind->SetInput("StartsTensorList", this->Input("StartsTensorList"));
    }
    if (this->HasInput("EndsTensorList")) {
      bind->SetInput("EndsTensorList", this->Input("EndsTensorList"));
    }
    bind->SetInput("Input", this->OutputGrad(framework::GradVarName("Input")));
    bind->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("slice");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SliceOpGradNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(slice,
                  ops::SliceOp,
                  ops::SliceOpMaker,
                  ops::SliceOpGradMaker<paddle::framework::OpDesc>,
                  ops::SliceOpGradMaker<paddle::imperative::OpBase>,
                  ops::SliceCompositeGradOpMaker,
                  ops::SliceOpVarTypeInference);
REGISTER_OPERATOR(slice_grad,
                  ops::SliceOpGrad,
                  ops::SliceDoubleOpGradMaker<paddle::framework::OpDesc>,
                  ops::SliceDoubleOpGradMaker<paddle::imperative::OpBase>,
                  ops::SliceOpGradNoNeedBufferVarsInferer,
                  ops::SliceOpGradVarTypeInference);
