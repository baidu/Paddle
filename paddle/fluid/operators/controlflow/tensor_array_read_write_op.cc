/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/array_operator.h"

namespace paddle {
namespace framework {
class OpDesc;
class Scope;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class WriteToArrayOp : public ArrayOp {
 public:
  WriteToArrayOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const phi::Place &place) const override {
    auto *x = scope.FindVar(Input("X"));
    if (x == nullptr) return;
    auto &x_tensor = x->Get<phi::DenseTensor>();
    size_t offset = GetOffset(scope, place);
    auto *out = scope.FindVar(Output("Out"))->GetMutable<phi::TensorArray>();
    if (offset >= out->size()) {
      VLOG(10) << "Resize " << Output("Out") << " from " << out->size()
               << " to " << offset + 1;
      out->resize(offset + 1);
    }
    auto *out_tensor = &out->at(offset);
    out_tensor->set_lod(x_tensor.lod());
    if (x_tensor.memory_size() > 0) {
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);

      paddle::framework::TensorCopy(x_tensor, place, dev_ctx, out_tensor);
    } else {
      VLOG(10) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                  "nothing has been written to output array["
               << offset << "].";
    }
  }
};

class WriteToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(phi::DenseTensor) the tensor will be written to tensor array");
    AddInput(
        "I",
        "(Tensor) the subscript index in tensor array. The number of element "
        "should be 1");
    AddOutput("Out", "(TensorArray) the tensor array will be written");
    AddComment(R"DOC(
WriteToArray Operator.

This operator writes a phi::DenseTensor to a phi::DenseTensor array.

Assume $T$ is phi::DenseTensor, $i$ is the subscript of the array, and $A$ is the array. The
equation is

$$A[i] = T$$

)DOC");
  }
};

class WriteToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInput("I"),
        true,
        common::errors::NotFound("Input(I) of WriteToArrayOp is not found."));

    // TODO(wangchaochaohu) control flow Op do not support runtime infer shape
    // Later we add [context->GetInputDim("I")) == 1] check when it's supported

    if (!context->HasInput("X")) {
      return;
    }

    PADDLE_ENFORCE_EQ(context->HasOutput("Out"),
                      true,
                      common::errors::NotFound(
                          "Output(Out) of WriteToArrayOp is not found."));
    context->SetOutputDim("Out", context->GetInputDim("X"));

    // When compile time, we need to:
    // - for ReadFromArray, share tensor_array X's lod_level to Out
    // - for WriteToArray, share X's lod_level to tensor_array Out
    // When runtime, we need to:
    // - for ReadFromArray, share X[I]'s lod to Out
    // - for WriteToArray, share X's lod to Out[I]
    // but we cannot get I's value here, so leave this work to detail
    // kernel implementation.
    if (!context->IsRuntime()) {
      context->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class WriteToArrayInferVarType : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto out_name = Output(ctx, "Out")[0];
    VLOG(10) << "Set Variable " << out_name << " as DENSE_TENSOR_ARRAY";
    SetType(ctx, out_name, framework::proto::VarType::DENSE_TENSOR_ARRAY);
    if (HasVar(ctx, x_name)) {
      SetDataType(ctx, out_name, GetDataType(ctx, x_name));
    }
  }
};

class ReadFromArrayOp : public ArrayOp {
 public:
  ReadFromArrayOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : ArrayOp(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const phi::Place &place) const override {
    auto *x = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(
        x,
        common::errors::NotFound("Input(X) of ReadFromArrayOp is not found."));
    auto &x_array = x->Get<phi::TensorArray>();
    auto *out = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE_NOT_NULL(
        out,
        common::errors::NotFound(
            "Output(Out) of ReadFromArrayOp is not found."));
    size_t offset = GetOffset(scope, place);
    if (offset < x_array.size()) {
      auto *out_tensor = out->GetMutable<phi::DenseTensor>();
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);
      framework::TensorCopy(x_array[offset], place, dev_ctx, out_tensor);
      out_tensor->set_lod(x_array[offset].lod());
    } else {
      VLOG(10) << "offset " << offset << " >= " << x_array.size();
      // set grad of the written tensor to 0 when used as write_to_array_grad
      auto *fw_var = scope.FindVar(Input("X_W"));
      if (fw_var == nullptr) return;
      auto &fw_var_tensor = fw_var->Get<phi::DenseTensor>();

      framework::AttributeMap attrs;
      attrs["dtype"] = framework::TransToProtoVarType(fw_var_tensor.dtype());
      attrs["shape"] = common::vectorize<int>(fw_var_tensor.dims());
      attrs["value"] = 0.0f;

      auto zero_op = framework::OpRegistry::CreateOp(
          "fill_constant", {}, {{"Out", {Output("Out")}}}, attrs);
      zero_op->Run(scope, place);
      auto *out_tensor = out->GetMutable<phi::DenseTensor>();
      out_tensor->set_lod(fw_var_tensor.lod());
    }
  }
};

class ReadFromArrayProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(TensorArray) the array will be read from.");
    AddInput("I",
             "(Tensor) the subscript index in tensor array. The number of "
             "element should be 1");
    AddInput("X_W",
             "(Tensor) the written tensor when used as the grad op of "
             "write_to_array. We use this to fill zero gradient.")
        .AsDispensable();
    AddOutput("Out", "(phi::DenseTensor) the tensor will be read from.");
    AddComment(R"DOC(
ReadFromArray Operator.

Read a phi::DenseTensor from a phi::DenseTensor Array.

Assume $T$ is phi::DenseTensor, $i$ is the subscript of the array, and $A$ is the array. The
equation is

$$T = A[i]$$

)DOC");
  }
};

class ReadFromArrayInferShape : public WriteToArrayInferShape {};

template <typename T>
class WriteToArrayGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("read_from_array");
    grad_op->SetInput("I", this->Input("I"));
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetInput("X_W", this->Input("X"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ReadFromArrayGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("write_to_array");
    grad_op->SetInput("I", this->Input("I"));
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(write_to_array,
                  ops::WriteToArrayOp,
                  ops::WriteToArrayInferShape,
                  ops::WriteToArrayOpProtoMaker,
                  ops::WriteToArrayGradMaker<paddle::framework::OpDesc>,
                  ops::WriteToArrayGradMaker<paddle::imperative::OpBase>,
                  ops::WriteToArrayInferVarType);
REGISTER_OPERATOR(read_from_array,
                  ops::ReadFromArrayOp,
                  ops::ReadFromArrayInferShape,
                  ops::ReadFromArrayProtoMaker,
                  ops::ReadFromArrayGradMaker<paddle::framework::OpDesc>,
                  ops::ReadFromArrayGradMaker<paddle::imperative::OpBase>);
