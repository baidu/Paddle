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

#include "paddle/fluid/operators/randperm_op.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class RandpermOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "The output(Out) of randperm op must not be null."));
    int n = ctx->Attrs().Get<int>("n");
    PADDLE_ENFORCE_GT(
        n, 0, platform::errors::InvalidArgument(
                  "The input(n) of randperm op must be greater than 0."));

    ctx->SetOutputDim("Out", framework::make_ddim({n}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class RandpermOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The output tensor of randperm op.");

    AddAttr<int>(
        "n", "The upper bound (exclusive), and it should be greater than 0.");
    AddAttr<int>("dtype",
                 "The data type of output tensor. "
                 "Default: 3[int64].")
        .SetDefault(framework::proto::VarType::INT64);
    AddAttr<int>("seed",
                 "Random seed used for permute samples. "
                 "0 means use a seed generated by the system."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random permutation every time. "
                 "Default: 0.")
        .SetDefault(0);

    AddComment(R"DOC( 
This operator returns a random permutation of integers from 0 to n-1.
)DOC");
  }
};

class RandpermOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_data_type = static_cast<framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, ctx->GetAttr("dtype")));
    ctx->SetOutputDataType("Out", var_data_type);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    randperm, paddle::operators::RandpermOp, paddle::operators::RandpermOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::RandpermOpVarTypeInference);

template <typename T>
using kernel =
    paddle::operators::RandpermKernel<paddle::platform::CPUDeviceContext, T>;

REGISTER_OP_CPU_KERNEL(randperm, kernel<int64_t>, kernel<int>);
