/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/infermeta/unary.h"
namespace paddle::operators {

class CIdentityOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "c_identity");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "c_identity");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for c_identity must be non-negative.", ring_id));
    phi::DDim dim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class CIdentityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) identity tensor.");
    AddOutput("Out", "(Tensor) identity tensor.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default true) eject CUDA operations to calculation stream.")
        .SetDefault(true);
    AddAttr<bool>("use_model_parallel",
                  "(bool default true) use this op with model parallel.")
        .SetDefault(true);
    AddComment(R"DOC(
Identity Operator which returns a copy of itself.
)DOC");
  }
};

template <typename T>
class CIdentityOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("all_reduce");
    retv->SetInput("x", this->OutputGrad("Out"));
    retv->SetOutput("out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetAttr("reduce_type", static_cast<int>(phi::ReduceType::kRedSum));
  }
};
}  // namespace paddle::operators

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(c_identity,
                            CIdentityShapeFunctor,
                            PD_INFER_META(phi::CIdentityInferMeta));

REGISTER_OPERATOR(c_identity,
                  ops::CIdentityOp,
                  ops::CIdentityOpGradMaker<paddle::framework::OpDesc>,
                  ops::CIdentityOpGradMaker<paddle::imperative::OpBase>,
                  ops::CIdentityOpMaker,
                  CIdentityShapeFunctor);
