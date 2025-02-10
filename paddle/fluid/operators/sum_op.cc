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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class SumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto x_vars = ctx.MultiInputVar("X");
    auto x_vars_name = ctx.InputNames("X");

    PADDLE_ENFORCE_GT(
        x_vars.size(),
        0,
        common::errors::InvalidArgument("Input[X] should not be empty"));

    PADDLE_ENFORCE_NOT_NULL(
        x_vars[0],
        common::errors::NotFound("Input var[%s] should not be nullptr",
                                 x_vars_name[0]));

    if (x_vars[0]->IsType<phi::DenseTensor>()) {
      int dtype = -1;
      for (size_t idx = 0; idx < x_vars.size(); ++idx) {
        PADDLE_ENFORCE_NOT_NULL(
            x_vars[idx],
            common::errors::NotFound("Input var[%s] should not be nullptr",
                                     x_vars_name[idx]));
        auto tensor =
            framework::GetDenseTensorOrSelectedRowsValueFromVar(*x_vars[idx]);
        if (!tensor->IsInitialized()) {
          continue;
        }
        if (dtype == -1) {
          dtype = framework::TransToProtoVarType(tensor->dtype());
        } else {
          PADDLE_ENFORCE_EQ(dtype,
                            framework::TransToProtoVarType(tensor->dtype()),
                            common::errors::InvalidArgument(
                                "The inputs type of sum op must be same"));
        }
      }
      PADDLE_ENFORCE_NE(dtype,
                        -1,
                        common::errors::InvalidArgument(
                            "Sum operator should have at least one tensor"));

      auto data_type = static_cast<framework::proto::VarType::Type>(dtype);

      // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
      if (!((data_type == framework::proto::VarType::FP32 ||
             data_type == framework::proto::VarType::BF16) &&
            ctx.OutputVar("Out")->IsType<phi::DenseTensor>())) {  // NOLINT
        this->SetDnnFallback(true);
      } else if (!std::all_of(x_vars.begin(),
                              x_vars.end(),
                              [](const framework::Variable* v) {
                                return v->IsType<phi::DenseTensor>();
                              })) {
        this->SetDnnFallback(true);
      }
      // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

      return phi::KernelKey(data_type, ctx.GetPlace());
    } else if (x_vars[0]->IsType<phi::SelectedRows>()) {
      for (auto& var : x_vars) {
        auto& value = var->Get<phi::SelectedRows>().value();
        if (value.IsInitialized()) {
          return phi::KernelKey(framework::TransToProtoVarType(value.dtype()),
                                ctx.GetPlace());
        }
      }
      // if input sparse vars are not initialized, use an default kernel type.
      return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
    } else if (x_vars[0]->IsType<phi::TensorArray>()) {
      for (auto& x_var : x_vars) {
        auto& array = x_var->Get<phi::TensorArray>();
        for (auto& each : array) {
          if (each.numel() != 0 && each.IsInitialized()) {
            return phi::KernelKey(framework::TransToProtoVarType(each.dtype()),
                                  ctx.GetPlace());
          }
        }
      }
      PADDLE_THROW(common::errors::InvalidArgument(
          "Expected each tensor in Input(x) in sum op has be initialized, but "
          "some tensor in Input(x) is not be initialized, please check your "
          "code.",
          framework::ToTypeName(x_vars[0]->Type())));
    }
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected type of Input(X) must be Tensor,  SelectedRows or "
        "DenseTensorArray. But got "
        "unsupported type: %s.",
        framework::ToTypeName(x_vars[0]->Type())));
  }
};

class SumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "A Variable list. The shape and data type of the list elements"
        "should be consistent. Variable can be multi-dimensional Tensor"
        "or phi::DenseTensor, and data types can be: float32, float64, int32, "
        "int64.")
        .AsDuplicable();
    AddOutput("Out",
              "the sum of input :code:`x`. its shape and data types are "
              "consistent with :code:`x`.");
    AddComment(
        R"DOC(This OP is used to sum one or more Tensor or phi::DenseTensor
                    of the input. If the input is phi::DenseTensor, the output only
                    shares LoD information with the first input.)DOC");
  }
};

class SumOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    if (!ctx->IsDygraph()) {
      auto var_type = framework::proto::VarType::SELECTED_ROWS;
      if (VLOG_IS_ON(10)) {
        for (size_t ind = 0; ind < ctx->InputSize("X"); ++ind) {
          VLOG(10) << ctx->InputVarName("X", static_cast<int>(ind)) << " "
                   << ctx->GetInputType("X", static_cast<int>(ind));
        }
      }

      if (ctx->InputTypeAnyOf("X",
                              framework::proto::VarType::DENSE_TENSOR_ARRAY)) {
        if (!ctx->InputTypeAllOf(
                "X", framework::proto::VarType::DENSE_TENSOR_ARRAY)) {
          std::ostringstream os;
          for (size_t ind = 0; ind < ctx->InputSize("X"); ++ind) {
            os << "    " << ctx->InputVarName("X", static_cast<int>(ind))
               << " type is " << ctx->GetInputType("X", static_cast<int>(ind))
               << "\n";
          }
          PADDLE_THROW(common::errors::InvalidArgument(
              "Not all inputs are tensor array:\n%s", os.str()));
        }
        var_type = framework::proto::VarType::DENSE_TENSOR_ARRAY;
      } else if (ctx->InputTypeAnyOf("X",
                                     framework::proto::VarType::DENSE_TENSOR)) {
        var_type = framework::proto::VarType::DENSE_TENSOR;
      }

      ctx->SetOutputType("Out", var_type);
      ctx->SetOutputDataType("Out", ctx->GetInputDataType("X"));
    }
  }
};

class SumGradDescMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    auto x_grads = InputGrad("X", false);
    std::vector<std::unique_ptr<framework::OpDesc>> grad_ops;
    grad_ops.reserve(x_grads.size());
    auto og = OutputGrad("Out");
    std::transform(x_grads.begin(),
                   x_grads.end(),
                   std::back_inserter(grad_ops),
                   [&og](const std::string& x_grad) {
                     auto* grad_op = new framework::OpDesc();
                     grad_op->SetType("scale");
                     grad_op->SetInput("X", og);
                     grad_op->SetOutput("Out", {x_grad});
                     grad_op->SetAttr("scale", 1.0f);
                     return std::unique_ptr<framework::OpDesc>(grad_op);
                   });

    return grad_ops;
  }
};

class SumGradOpBaseMaker : public imperative::GradOpBaseMakerBase {
 public:
  using imperative::GradOpBaseMakerBase::GradOpBaseMakerBase;

  std::shared_ptr<imperative::GradOpNode> operator()() const override {
    auto x_grads = InputGrad("X", false);
    using InputGradsType = decltype(x_grads);

    if (!x_grads.empty()) {
      auto node = this->NewGradNode();
      node->reserve(x_grads.size());
      auto og = OutputGrad("Out");
      for (auto& x_grad : x_grads) {
        imperative::TracedGradOp op(node);
        op.SetType("scale");
        op.SetInput("X", og);
        op.SetOutput("Out", InputGradsType{x_grad});
        op.SetAttr("scale", 1.0f);
        op.SetDefaultAttrsMap(DefaultAttrsMap());
      }
      return node;
    } else {
      return nullptr;
    }
  }
};

DECLARE_INPLACE_OP_INFERER(SumInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(sum,
                            AddNInferShapeFunctor,
                            PD_INFER_META(phi::AddNTensorArrayInferMeta));

REGISTER_OPERATOR(sum,
                  ops::SumOp,
                  ops::SumOpMaker,
                  ops::SumGradDescMaker,
                  ops::SumGradOpBaseMaker,
                  ops::SumOpVarTypeInference,
                  ops::SumInplaceInferer,
                  AddNInferShapeFunctor);
