//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sparse_attention_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class SparseAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "tensor of shape ");
    AddInput("K", "tensor of shape ");
    AddInput("V", "tensor of shape ");
    AddInput("offset", "tensor of offset in CSR format ");
    AddInput("columns", "tensor of columns in CSR format ");

    AddOutput("Out", "The output tensor of sparse_mat operator");
    AddOutput("ResultSdd", "The computation result of sdd operation.")
        .AsIntermediate();
    AddOutput("ResultSoftmax", "The computation result of softmax operation.")
        .AsIntermediate();
    AddComment(R"DOC(
      )DOC");
  }
};

class SparseAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Q"), "Input", "Q", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("K"), "Input", "K", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("V"), "Input", "V", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("offset"), "Input", "offset",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasInput("columns"), "Input", "columns",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("ResultSdd"), "Output", "ResultSdd",
                   "sparse_attention");
    OP_INOUT_CHECK(ctx->HasOutput("ResultSoftmax"), "Output", "ResultSoftmax",
                   "sparse_attention");

    std::vector<int64_t> dims_q =
        paddle::framework::vectorize(ctx->GetInputDim("Q"));
    std::vector<int64_t> dims_k =
        paddle::framework::vectorize(ctx->GetInputDim("K"));
    std::vector<int64_t> dims_v =
        paddle::framework::vectorize(ctx->GetInputDim("V"));
    std::vector<int64_t> dims_columns =
        paddle::framework::vectorize(ctx->GetInputDim("columns"));

    auto ndims_q = dims_q.size();
    auto ndims_k = dims_k.size();
    auto ndims_v = dims_v.size();

    int64_t columns_dim = dims_columns[0];

    // q,k,v must be 4 dims
    PADDLE_ENFORCE_EQ(ndims_q, 4,
                      "ShapeError: Dimension in query' shapes must be 4. ");
    PADDLE_ENFORCE_EQ(ndims_k, 4,
                      "ShapeError: Dimension in key' shapes must be 4. ");
    PADDLE_ENFORCE_EQ(ndims_v, 4,
                      "ShapeError: Dimension in value' shapes must be 4. ");

    std::vector<int64_t> new_dims;
    new_dims.assign(dims_q.begin(), dims_q.end());

    auto out_dims = framework::make_ddim(new_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("ResultSdd", {columns_dim});
    ctx->SetOutputDim("ResultSoftmax", {columns_dim});
    ctx->ShareLoD("Q", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "Q", "K");

    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class SparseAttentionOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasInput("Q"), "Input", "Q", "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("K"), "Input", "K", "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("V"), "Input", "V", "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("offset"), "Input", "offset",
                   "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("columns"), "Input", "columns",
                   "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("ResultSdd"), "Input", "ResultSdd",
                   "sparse_attention");
    OP_INOUT_CHECK(context->HasInput("ResultSoftmax"), "Input", "ResultSoftmax",
                   "sparse_attention");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "sparse_attention");

    auto x_dims = context->GetInputDim("Q");
    auto y_dims = context->GetInputDim("K");
    auto z_dims = context->GetInputDim("V");

    auto x_grad_name = framework::GradVarName("Q");
    auto y_grad_name = framework::GradVarName("K");
    auto z_grad_name = framework::GradVarName("V");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(y_grad_name)) {
      context->SetOutputDim(y_grad_name, y_dims);
    }
    if (context->HasOutput(z_grad_name)) {
      context->SetOutputDim(z_grad_name, z_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class SparseAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sparse_attention_grad");
    op->SetInput("Q", this->Input("Q"));
    op->SetInput("K", this->Input("K"));
    op->SetInput("V", this->Input("V"));
    op->SetInput("offset", this->Input("offset"));
    op->SetInput("columns", this->Input("columns"));
    op->SetInput("ResultSdd", this->Output("ResultSdd"));
    op->SetInput("ResultSoftmax", this->Output("ResultSoftmax"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Q"), this->InputGrad("Q"));
    op->SetOutput(framework::GradVarName("K"), this->InputGrad("K"));
    op->SetOutput(framework::GradVarName("V"), this->InputGrad("V"));
  }
};

template <typename T>
class SparseAttentionDoubleGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("sparse_attention");
    grad_op->SetInput("Q", this->Input(("Q")));
    grad_op->SetInput("K", this->Input(("K")));
    grad_op->SetInput("V", this->Input(("V")));
    grad_op->SetInput("offset", this->Input(("offset")));
    grad_op->SetInput("columns", this->Input(("columns")));
    grad_op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    grad_op->SetOutput("DDQ", this->OutputGrad(framework::GradVarName("Q")));
    grad_op->SetOutput("DDK", this->OutputGrad(framework::GradVarName("K")));
    grad_op->SetOutput("DDV", this->OutputGrad(framework::GradVarName("V")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sparse_attention, ops::SparseAttentionOp,
                  ops::SparseAttentionOpMaker,
                  ops::SparseAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::SparseAttentionGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    sparse_attention_grad, ops::SparseAttentionOpGrad,
    ops::SparseAttentionDoubleGradOpMaker<paddle::framework::OpDesc>,
    ops::SparseAttentionDoubleGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    sparse_attention,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SparseAttentionKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    sparse_attention_grad,
    ops::SparseAttentionGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SparseAttentionGradKernel<paddle::platform::CPUDeviceContext, double>);
