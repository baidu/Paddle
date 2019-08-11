// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/filter_by_instag_op.h"

#include <memory>
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {
class FilterByInstagOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Ins"), "Input(Ins) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Ins_tag"), 
            "Input(Ins_tag) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Filter_tag"), 
            "Input(Filter_tag) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("LossWeight"),
                   "Output(LossWeight) shoudl not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Map"), "Output(Map) should be not null.");

    auto x1_dims = ctx->GetInputDim("Ins");  // batch_size * vec

    ctx->SetOutputDim("Out", framework::make_ddim({-1, x1_dims[1]}));
    ctx->SetOutputDim("LossWeight", framework::make_ddim({-1, 1}));
    ctx->SetOutputDim("Map", framework::make_ddim({-1, 2}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Ins"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FilterByInstagOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ins", "(LoDTensor) embeded tensor");
    AddInput("Ins_tag", "(LoDTensor) ins tag list");
    AddInput("Filter_tag", "(1D Tensor) filter tag list");
    AddAttr<bool>("is_lod", "is Ins with LoD info or not, default True");
    AddOutput("Out", "(LoDTensor) embeded tensor filtered by instag");
    AddOutput("LossWeight", "(Tensor) loss weight.");
    AddOutput("Map", "(LoDTensor) mapping from Out rows to X1 rows");
    AddComment(R"DOC(
Filter By Instag Op 

This operator is used to filter embeded ins.

There are 3 inputs. First is embeded ins, Second is tags for ins, 
Third is tags to filter.

There are 3 outputs. First is filtered embeded ins, Second is Loss Weight,
Third is the Map from Out line number to X1 line number. 
)DOC");
  }
};

class FilterByInstagOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Map"), "Input(Map) should be not null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Grad Input(Out) should be not null");
    PADDLE_ENFORCE(ctx->HasInput("Ins"), "Input(Ins) should be not null");
    PADDLE_ENFORCE(ctx->HasInput("LossWeight"),
                   "Input(LossWeight) should be not null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Ins")),
                   "Grad Output(Ins) should be not null");

    auto grad_out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x1_dims = ctx->GetInputDim("Ins");
    ctx->SetOutputDim(framework::GradVarName("Ins"),
                      framework::make_ddim({x1_dims[0], grad_out_dims[1]}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("Out")));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FilterByInstagGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("filter_by_instag_grad");
    op->SetInput("Map", Output("Map"));
    op->SetInput("Ins", Input("Ins"));
    op->SetAttrMap(Attrs());
    op->SetInput("LossWeight", Output("LossWeight"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Ins"), InputGrad("Ins"));
    return op;
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(filter_by_instag, ops::FilterByInstagOp,
                  ops::FilterByInstagOpMaker,
                  ops::FilterByInstagGradOpDescMaker);

REGISTER_OPERATOR(filter_by_instag_grad, ops::FilterByInstagOpGrad);

REGISTER_OP_CPU_KERNEL(filter_by_instag, ops::FilterByInstagKernel<float>,
                       ops::FilterByInstagKernel<double>,
                       ops::FilterByInstagKernel<int32_t>,
                       ops::FilterByInstagKernel<int64_t>);

REGISTER_OP_CPU_KERNEL(filter_by_instag_grad,
                       ops::FilterByInstagGradKernel<float>,
                       ops::FilterByInstagGradKernel<double>,
                       ops::FilterByInstagGradKernel<int32_t>,
                       ops::FilterByInstagGradKernel<int64_t>);
