// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace ops = paddle::operators;
namespace paddle::operators {
class ReduceBaseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceBaseOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ReduceBaseOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    PADDLE_ENFORCE_GT(dims.size(),
                      0,
                      common::errors::InvalidArgument(
                          "The input dim dimensions of ReduceBaseOp "
                          "should be greater than 0. But received the dim "
                          "dimensions of Reduce = %d.",
                          dims.size()));

    for (size_t i = 0; i < dims.size(); ++i) {
      PADDLE_ENFORCE_LT(
          dims[i],
          x_rank,
          common::errors::InvalidArgument(
              "The reduce dim index %d should be in the "
              "range [-dimension(X), dimension(X)] "
              "which dimension = %d. But received dim index = %d.",
              i,
              x_rank,
              dims[i]));
      PADDLE_ENFORCE_GE(
          dims[i],
          -x_rank,
          common::errors::InvalidArgument(
              "The reduce dim index %d should be in the "
              "range [-dimension(X), dimension(X)] "
              "which dimension = %d. But received dim index = %d.",
              i,
              x_rank,
              dims[i]));
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
    }
    sort(dims.begin(), dims.end());
    bool reduce_all = ctx->Attrs().Get<bool>("reduce_all");
    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    if (reduce_all) {
      if (keep_dim)
        ctx->SetOutputDim("Out",
                          common::make_ddim(std::vector<int64_t>(x_rank, 1)));
      else
        ctx->SetOutputDim("Out", {1});
    } else {
      auto dims_vector = common::vectorize(x_dims);
      if (keep_dim) {
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = 1;
        }
      } else {
        const int kDelFlag = -2;
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = kDelFlag;
        }
        dims_vector.erase(
            remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
            dims_vector.end());
      }
      if (!keep_dim && dims_vector.size() == 0) {
        dims_vector.push_back(1);
      }
      auto out_dims = common::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (dims.size() > 0 && dims[0] != 0) {
        // Only pass LoD when not reducing on the first dim.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

  // oneDNN's reduction kernel is optimized only for reducing throughout the
  // most outer dims, so in case of another type of reduction, it would be
  // better to fallback to native implementation
  static bool HasOptimizedOneDNNKernel(const framework::ExecutionContext& ctx) {
    // native reduce kernels don't support bf16
    // so oneDNN kernel is enforced in that case
    if (ctx.Input<phi::DenseTensor>("X")->dtype() == phi::DataType::BFLOAT16)
      return true;

    if (!ctx.HasAttr("dim") || !ctx.HasAttr("reduce_all")) {
      return false;
    }

    auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
    const bool reduce_all = ctx.Attr<bool>("reduce_all");
    int ndims = ctx.Input<phi::DenseTensor>("X")->dims().size();

    if (reduce_all) {
      return true;
    }

    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      if (reduce_dims[i] < 0) reduce_dims[i] = ndims + reduce_dims[i];
    }
    sort(reduce_dims.begin(), reduce_dims.end());
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      if (reduce_dims[reduce_dims.size() - i - 1] !=
          static_cast<int>(ndims - i - 1)) {
        return false;
      }
    }

    return true;
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // choose cudnn kernel if the runtime supported.
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

    // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
    if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5 ||
        !HasOptimizedOneDNNKernel(ctx)) {
      this->SetDnnFallback(true);
    }
    // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

    if (input_data_type == framework::proto::VarType::FP16) {
      PADDLE_ENFORCE_EQ(
          ctx.GetPlace().GetType() == phi::AllocationType::GPU ||
              ctx.GetPlace().GetType() == phi::AllocationType::XPU ||
              ctx.GetPlace().GetType() == phi::AllocationType::CUSTOM,
          true,
          common::errors::InvalidArgument(
              "float16 can only be used on GPU or XPU place"));
    }
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class ReduceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceBaseOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "ReduceBaseOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    // TODO(dev): We should delete Infershape and migrate it into
    // UnchangeInferMeta.In case of 'dim' is Variable, it will
    // not exist in Attrs but in Inputs.
    if (ctx->HasAttr("dim")) {
      auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
      for (size_t i = 0; i < dims.size(); ++i) {
        PADDLE_ENFORCE_LT(
            dims[i],
            x_rank,
            common::errors::InvalidArgument(
                "The reduce dim index %d should be in the "
                "range [-dimension(X), dimension(X)], "
                "which dimension = %d. But received dim index = %d.",
                i,
                x_rank,
                dims[i]));
        if (dims[i] < 0) dims[i] = x_rank + dims[i];
      }
    }

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    int out_dtype = ctx.Attr<int>("out_dtype");
    auto input_data_type =
        (out_dtype >= 0)
            ? static_cast<framework::proto::VarType::Type>(out_dtype)
            : OperatorWithKernel::IndicateVarDataType(
                  ctx, framework::GradVarName("Out"));

    // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
    // max 5D tensor is supported
    if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5) {
      dnn_fallback_ = true;
    }
    // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

// NOTE(dengkaipeng): Input(Out) is unnecessary in reduce_mean_grad
// calculation, but will incur a reduce_mean_grad op after
// reduce_mean_grad_grad, delete Input(Out) here.
// This change has no effect on reduce_mean_grad calculations.
template <typename T>
class ReduceMeanOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("reduce_mean_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

class ReduceMeanDoubleGradDescMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    auto x_gg = OutputGrad(framework::GradVarName("X"));  // input ddx
    auto out_grads = InputGrad(framework::GradVarName("Out"));
    if (!out_grads.empty()) {
      auto* out_grad_op = new framework::OpDesc();
      out_grad_op->SetType("reduce_mean");
      out_grad_op->SetInput("X", x_gg);
      out_grad_op->SetAttrMap(Attrs());
      out_grad_op->SetOutput("Out", out_grads);
      ops.emplace_back(out_grad_op);
    }

    return ops;
  }
};

class ReduceMeanDoubleGradOpBaseMaker : public imperative::GradOpBaseMakerBase {
 public:
  using imperative::GradOpBaseMakerBase::GradOpBaseMakerBase;

  std::shared_ptr<imperative::GradOpNode> operator()() const override {
    auto out_grads = InputGrad(framework::GradVarName("Out"));
    if (!out_grads.empty()) {
      auto x_gg = OutputGrad(framework::GradVarName("X"));  // input ddx
      auto node = this->NewGradNode();
      {
        imperative::TracedGradOp op(node);
        op.SetType("reduce_mean");
        op.SetInput("X", x_gg);
        op.SetAttrMap(Attrs());
        op.SetDefaultAttrsMap(DefaultAttrsMap());
        op.SetOutput("Out", out_grads);
      }
      return node;
    } else {
      return nullptr;
    }
  }
};
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ReduceMeanGradNoNeedBufferVarInferer, "X");

class ReduceBaseOpMaker : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X",
             "(Tensor) The input tensor. Tensors with rank at most 6 are "
             "supported.");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<std::vector<int>>(
        "dim",
        "(list<int>, default {0}) The dimensions to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `dim[i] < 0`, the dims[i] to reduce is `rank + dims[i]`. "
        "Note that reducing on the first dim will make the LoD info lost.")
        .SetDefault({0})
        .SupportTensor();
    AddAttr<bool>("keep_dim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    AddAttr<int>("in_dtype",
                 "(int, default -1)"
                 "The dtype of input, default value is -1, the user could not "
                 "set this value.")
        .SetDefault(-1);
    AddAttr<int>(
        "out_dtype",
        "(int, default -1)"
        "The dtype of output, default value is -1, the dtype is same as input")
        .SetDefault(-1);
    AddComment(string::Sprintf(R"DOC(
%s Operator.

This operator computes the %s of input tensor along the given dimension.
The result tensor has 1 fewer dimension than the input unless keep_dim is true.
If reduce_all is true, just reduce along all dimensions and output a scalar.

)DOC",
                               GetOpType(),
                               GetName()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetOpType() const = 0;
};
}  // namespace paddle::operators

class __reduce_meanMaker__ : public ops::ReduceBaseOpMaker {
 protected:
  std::string GetName() const override { return "reduce_mean"; }
  std::string GetOpType() const override { return "Reduce reduce_mean"; }
};

DECLARE_INFER_SHAPE_FUNCTOR(
    reduce_mean,
    ReduceMeanInferShapeFunctor,
    PD_INFER_META(phi::ReduceIntArrayAxisInferMetaBase));

REGISTER_OPERATOR(reduce_mean,
                  ops::ReduceBaseOp,
                  __reduce_meanMaker__,
                  ops::ReduceMeanOpGradMaker<paddle::framework::OpDesc>,
                  ops::ReduceMeanOpGradMaker<paddle::imperative::OpBase>,
                  ReduceMeanInferShapeFunctor);
REGISTER_OPERATOR(reduce_mean_grad,
                  ops::ReduceGradOp,
                  ops::ReduceMeanDoubleGradDescMaker,
                  ops::ReduceMeanDoubleGradOpBaseMaker,
                  ops::ReduceMeanGradNoNeedBufferVarInferer);
