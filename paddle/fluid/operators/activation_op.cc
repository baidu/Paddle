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

#include "paddle/fluid/operators/activation_op.h"

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/operators/mkldnn/mkldnn_activation_op.h"
#include "paddle/phi/backends/dynload/port.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

template <typename GradFunctor>
static constexpr bool CanInplaceAct() {
  return GradFunctor::FwdDeps() == ActBwdOpFwdDeps::kDepOut ||
         GradFunctor::FwdDeps() == ActBwdOpFwdDeps::kNoDeps;
}

#define REGISTER_ACTIVATION_OP_MAKER(OP_NAME, OP_COMMENT)           \
  class OP_NAME##OpMaker                                            \
      : public ::paddle::framework::OpProtoAndCheckerMaker {        \
   public:                                                          \
    void Make() override {                                          \
      AddInput("X",                                                 \
               "Input of " #OP_NAME                                 \
               " operator, an N-D Tensor, with data type float32, " \
               "float64 or float16.");                              \
      AddOutput("Out",                                              \
                "Output of " #OP_NAME                               \
                " operator, a Tensor with shape same as input.");   \
      AddComment(OP_COMMENT);                                       \
    }                                                               \
  }

template <ActBwdOpFwdDeps kDepValue, typename T>
class ActivationGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());

    if ((static_cast<int>(kDepValue) &
         static_cast<int>(ActBwdOpFwdDeps::kDepX)) ||
        FLAGS_use_mkldnn ||
        (op->HasAttr("use_mkldnn") &&
         PADDLE_GET_CONST(bool, op->GetAttr("use_mkldnn")))) {
      op->SetInput("X", this->Input("X"));  // x
    }

    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      op->SetInput("Out", this->Output("Out"));  // out
    }
  }
};

framework::OpKernelType GetKernelType(const framework::ExecutionContext& ctx,
                                      const framework::OperatorWithKernel& oper,
                                      const std::string& name) {
  auto data_type = oper.IndicateVarDataType(ctx, name);
  // FIXME(liuwei1031) temporarily disable the code to unblock users
  // TODO(liuwei1031) figure out the reason behind
  // https://github.com/PaddlePaddle/Paddle/issues/16096
  // and re-enable this in the future
  // #ifdef PADDLE_WITH_CUDA
  //   auto it1 = oper.Attrs().find("use_cudnn");
  //   if (it1 != oper.Attrs().end() && platform::CanCUDNNBeUsed(ctx)) {
  //     library = framework::LibraryType::kCUDNN;
  //   }
  // #endif
  return framework::OpKernelType(data_type, ctx.GetPlace());
}

class ActivationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }
};

class ActivationOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class ActivationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    ctx->ShareDim(out_grad_name, framework::GradVarName("X"));
    ctx->ShareLoD(out_grad_name, framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, framework::GradVarName("Out"));
  }
};

UNUSED constexpr char SigmoidDoc[] = R"DOC(
Sigmoid Activation

$$out = \frac{1}{1 + e^{-x}}$$

)DOC";

UNUSED constexpr char ReluDoc[] = R"DOC(
Relu Activation Operator.

$$out = \max(x, 0)$$

)DOC";

UNUSED constexpr char TanhDoc[] = R"DOC(
Tanh Activation Operator.

$$out = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

UNUSED constexpr char TanhShrinkDoc[] = R"DOC(
TanhShrink Activation Operator.

$$out = x - \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

)DOC";

UNUSED constexpr char SqrtDoc[] = R"DOC(
Sqrt Activation Operator.

$$out=\\sqrt{x}=x^{1/2}$$

**Note**:
  input value must be greater than or equal to zero.

)DOC";

UNUSED constexpr char RsqrtDoc[] = R"DOC(
Rsqrt Activation Operator.

Please make sure input is legal in case of numeric errors.

$$out = \\frac{1}{\\sqrt{x}}$$

)DOC";

UNUSED constexpr char LogDoc[] = R"DOC(
Log Activation Operator.

$$out = \ln(x)$$

Natural logarithm of x.

)DOC";

UNUSED constexpr char SquareDoc[] = R"DOC(
The OP square each elements of the inputs.

$$out = x^2$$

)DOC";

UNUSED constexpr char SoftsignDoc[] = R"DOC(
Softsign Activation Operator.

$$out = \\frac{x}{1 + \|x\|}$$

)DOC";

class LeakyReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "A LoDTensor or Tensor representing preactivation values. Must be "
             "one of the following types: float32, float64.");
    AddOutput(
        "Out",
        "A LoDTensor or Tensor with the same type and size as that of x.");
    AddAttr<float>("alpha", "Slope of the activation function at x < 0.")
        .SetDefault(0.02f);
    AddComment(R"DOC(
LeakyRelu Activation Operator.

$$out = \max(x, \alpha * x)$$

)DOC");
  }
};

class SoftplusOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input of Softplus operator, an N-D Tensor, with data type "
             "float32, float64 or float16.");
    AddOutput(
        "Out",
        "Output of Softplus operator, a Tensor with shape same as input.");
    AddAttr<float>("beta", "The value of beta for Softplus.").SetDefault(1.0f);
    AddAttr<float>("threshold", "The value of threshold for Softplus.")
        .SetDefault(20.0f);
    AddComment(R"DOC(
:strong:`Softplus Activation Operator`

..  math::
    out = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) \\
    \text{For numerical stability, the implementation reverts to the linear function when :}\,x \times \beta > threshold.

)DOC");
  }
};

class SoftShrinkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Softshrink operator");
    AddOutput("Out", "Output of Softshrink operator");
    AddAttr<float>("lambda", "non-negative offset").SetDefault(0.5f);
    AddComment(R"DOC(
:strong:`Softshrink Activation Operator`

..  math::
    out = \begin{cases}
         x - \lambda, \text{if } x > \lambda \\
         x + \lambda, \text{if } x < -\lambda \\
         0,  \text{otherwise}
         \end{cases}

)DOC");
  }
};

class BReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input is a multi-dimensional Tensor. The data type is "
             "float32, float64.");
    AddOutput("Out",
              "The output is a multi-dimensional Tensor which has same "
              "dimension and data type as the ``X``.");
    AddAttr<float>("t_min", "The min marginal value of BRelu")
        .SetDefault(static_cast<float>(0));
    AddAttr<float>("t_max", "The max marginal value of BRelu")
        .SetDefault(static_cast<float>(24));
    AddComment(R"DOC(
BRelu Activation Operator.

$$out = \min(\max(x, t_{min}), t_{max})$$

)DOC");
  }
};

class SoftReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of SoftRelu operator");
    AddOutput("Out", "Output of SoftRelu operator");
    AddAttr<float>("threshold", "The threshold value of SoftRelu")
        .SetDefault(40.0f);
    AddComment(R"DOC(
SoftRelu Activation Operator.

$$out = \ln(1 + \exp(\max(\min(x, threshold), -threshold)))$$

)DOC");
  }
};

class ELUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input is a multi-dimensional Tensor. The data type is "
             "float32 or float64.");
    AddOutput("Out",
              "The output is a multi-dimensional Tensor which has same "
              "dimension and data type as the ``x``.");
    AddAttr<float>("alpha", "The alpha value of ELU").SetDefault(1.0f);
    AddComment(R"DOC(
ELU Activation Operator.

Applies the following element-wise computation on the input according to
https://arxiv.org/abs/1511.07289.

$$out = \max(0, x) + \min(0, \alpha * (e^x - 1))$$

)DOC");
  }
};

template <typename T>
class ELUGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elu_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class CELUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input is a multi-dimensional Tensor. The data type is "
             "float32 or float64.");
    AddOutput("Out",
              "The output is a multi-dimensional Tensor which has same "
              "dimension and data type as the ``x``.");
    AddAttr<float>("alpha", "The alpha value of CELU").SetDefault(1.0f);
    AddComment(R"DOC(
CELU Activation Operator.

Applies the following element-wise computation on the input according to
https://arxiv.org/abs/1704.07483.

$$out = \max(0, x) + \min(0, \alpha * (e^(x/\alpha) - 1))$$

)DOC");
  }
};

class Relu6OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input of relu6 operator, an N-D Tensor, "
             "with data type float32, float64.");
    AddOutput(
        "Out",
        "Output of relu6 operator, a Tensor with the same shape as input.");
    AddAttr<float>("threshold",
                   "The threshold value of Relu6. Default is 6.0. ")
        .SetDefault(6.0f);
    AddComment(R"DOC(
Relu6 Activation Operator.

$$out = \min(\max(0, x), threshold)$$

)DOC");
  }
};

class PowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Pow operator");
    AddInput("FactorTensor",
             "(Tensor<float>, optional). If provided, pow will use this"
             "The shape of FactorTensor MUST BE [1]."
             "it has higher priority than attr(factor).")
        .AsDispensable();
    AddOutput("Out", "Output of Pow operator");
    AddAttr<float>("factor", "The exponential factor of Pow").SetDefault(1.0f);
    AddComment(R"DOC(
Pow Activation Operator.

$$out = x^{factor}$$

)DOC");
  }
};

class STanhOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input of STanh operator."
             " A Tensor with type float32, float64.");
    AddOutput("Out", "Output of STanh operator. A Tensor with type float32.");
    AddAttr<float>("scale_a", "The scale parameter of a for the input. ")
        .SetDefault(0.67f);
    AddAttr<float>("scale_b", "The scale parameter of b for the input")
        .SetDefault(1.7159f);
    AddComment(R"DOC(
STanh Activation Operator.

$$out = b * \\frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}$$

)DOC");
  }
};

class ThresholdedReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of ThresholdedRelu operator");
    AddOutput("Out", "Output of ThresholdedRelu operator");
    AddAttr<float>("threshold",
                   "The threshold location of activation. [default 1.0].")
        .SetDefault(1.0f);
    AddComment(R"DOC(
:strong:`ThresholdedRelu activation operator`

..  math::

    out = \begin{cases}
             x,  \text{if } x > threshold \\
             0,  \text{otherwise}
          \end{cases}
)DOC");
  }
};

class SwishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Swish operator");
    AddOutput("Out", "Output of Swish operator");
    AddAttr<float>("beta", "Constant beta of swish operator").SetDefault(1.0f);
    AddComment(R"DOC(
Swish Activation Operator.

$$out = \\frac{x}{1 + e^{- \beta \ x}}$$

)DOC");
  }
};

class MishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Mish operator");
    AddOutput("Out", "Output of Mish operator");
    AddAttr<float>(
        "threshold",
        "Constant threshold of softplus in Mish operator. Approximate value "
        "of softplus will be used if absolute value of input is greater than "
        ":attr:`threshold`")
        .SetDefault(20.f);
    AddComment(R"DOC(
Mish Activation Operator.

..  math::
    softplus(x) = \begin{cases}
            x, \text{if } x > \text{threshold} \\
            \ln(1 + e^{x}),  \text{otherwise}
          \end{cases}

    out = x * \tanh(softplus(x))

)DOC");
  }
};

class HardSwishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of HardSwish operator");
    AddOutput("Out", "Output of HardSwish operator");
    AddAttr<float>("threshold", "The threshold parameter of HardSwish operator")
        .SetDefault(6.0f);
    AddAttr<float>("scale", "The scale parameter of HardSwish operator")
        .SetDefault(6.0f);
    AddAttr<float>("offset", "The offset parameter of HardSwish operator")
        .SetDefault(3.0f);
    AddComment(R"DOC(
HardSwish Activation Operator.

The hard version of swish(https://arxiv.org/pdf/1905.02244.pdf).

$$out = \frac{x * (min(max(0, x+offset), threshold))}{scale}$$

The threshold and scale should be positive. The offset can be either positive or negative.
The default parameters are set according to the above reference.
It is recommended to use the defaults for this activation.

)DOC");
  }
};

REGISTER_ACTIVATION_OP_MAKER(Sigmoid, SigmoidDoc);
REGISTER_ACTIVATION_OP_MAKER(Relu, ReluDoc);
REGISTER_ACTIVATION_OP_MAKER(Tanh, TanhDoc);
REGISTER_ACTIVATION_OP_MAKER(TanhShrink, TanhShrinkDoc);
REGISTER_ACTIVATION_OP_MAKER(Sqrt, SqrtDoc);
REGISTER_ACTIVATION_OP_MAKER(Rsqrt, RsqrtDoc);
REGISTER_ACTIVATION_OP_MAKER(Log, LogDoc);
REGISTER_ACTIVATION_OP_MAKER(Square, SquareDoc);
REGISTER_ACTIVATION_OP_MAKER(Softsign, SoftsignDoc);

template <ActBwdOpFwdDeps kDepValue>
class ActivationOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      if (ctx->HasOutput("DX")) {
        ctx->ShareDim("X", "DX");
        ctx->ShareLoD("X", "DX");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      if (ctx->HasOutput("DOut")) {
        ctx->ShareDim("Out", "DOut");
        ctx->ShareLoD("Out", "DOut");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("Out", "DDOut");
        ctx->ShareLoD("Out", "DDOut");
      }
      if (ctx->HasOutput("DOutNew")) {
        ctx->ShareDim("Out", "DOutNew");
        ctx->ShareLoD("Out", "DOutNew");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "DDX");
  }
};

template <ActBwdOpFwdDeps kDepValue>
class ActivationOpDoubleGrad2 : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("Out", "DDOut");
        ctx->ShareLoD("Out", "DDOut");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "DDX");
  }
};

template <ActBwdOpFwdDeps kDepValue>
class ActivationOpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
      if (ctx->HasOutput("DX")) {
        ctx->ShareDim("X", "DX");
        ctx->ShareLoD("X", "DX");
      }
      if (ctx->HasOutput("DDOut")) {
        ctx->ShareDim("X", "DDOut");
        ctx->ShareLoD("X", "DDOut");
      }
    }
    if (static_cast<int>(kDepValue) &
        static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
      if (ctx->HasOutput("D_DOut")) {
        ctx->ShareDim("Out", "D_DOut");
        ctx->ShareLoD("Out", "D_DOut");
      }
      if (ctx->HasOutput("D_OutNew")) {
        ctx->ShareDim("Out", "D_OutNew");
        ctx->ShareLoD("Out", "D_OutNew");
      }
      if (ctx->HasOutput("D_DDx")) {
        ctx->ShareDim("DDX", "D_DDx");
        ctx->ShareLoD("DDX", "D_DDx");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "DDX");
  }
};

template <typename T>
class SigmoidDoubleGradMaker
    : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sigmoid_grad_grad");
    // input1: Out
    op->SetInput("Out", this->Input("Out"));
    // input2: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetAttrMap(this->Attrs());
    // output: ddy
    op->SetOutput("DOutNew", this->InputGrad("Out"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

template <typename T>
class SigmoidTripleGradMaker
    : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sigmoid_triple_grad");
    // Out, DDX, DOut, D_DDOut, D_DOut_New   // input
    // D_OutNew, D_DOut, D_DDx               // output
    // input1: Out
    op->SetInput("Out", this->Input("Out"));
    // input2: ddx
    op->SetInput("DDX", this->Input("DDX"));
    // input3: dout
    op->SetInput("DOut", this->Input("DOut"));
    // input4: d_ddout
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));
    // input5: d_dout_new
    op->SetInput("D_DOut_New", this->OutputGrad("DOutNew"));
    op->SetAttrMap(this->Attrs());

    // output: d_dOut, d_OutNew, d_ddx
    op->SetOutput("D_OutNew", this->InputGrad("Out"));
    op->SetOutput("D_DOut", this->InputGrad("DOut"));
    op->SetOutput("D_DDx", this->InputGrad("DDX"));
  }
};

template <typename T>
class TanhDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tanh_grad_grad");
    // input1: Out
    op->SetInput("Out", this->Input("Out"));
    // input2: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetAttrMap(this->Attrs());
    // output: ddy
    op->SetOutput("DOutNew", this->InputGrad("Out"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

template <typename T>
class TanhTripleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tanh_triple_grad");
    // Out, DDX, DOut, D_DDOut, D_DOut_New   // input
    // D_OutNew, D_DOut, D_DDx               // output
    // input1: Out
    op->SetInput("Out", this->Input("Out"));
    // input2: ddx
    op->SetInput("DDX", this->Input("DDX"));
    // input3: dout
    op->SetInput("DOut", this->Input("DOut"));
    // input4: d_ddout
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));
    // input5: d_dout_new
    op->SetInput("D_DOut_New", this->OutputGrad("DOutNew"));
    op->SetAttrMap(this->Attrs());

    // output: d_dOut, d_OutNew, d_ddx
    op->SetOutput("D_OutNew", this->InputGrad("Out"));
    op->SetOutput("D_DOut", this->InputGrad("DOut"));
    op->SetOutput("D_DDx", this->InputGrad("DDX"));
  }
};
// ReluGrad: dx = dy if y >= 0 else 0
// ReluGradGrad: ddy = ddx if y >= 0 else 0
template <typename T>
class ReluDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("relu_grad_grad");
    // input1: Out
    op->SetInput("Out", this->Input("Out"));
    // input2: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());
    // output: ddy
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// leaky_relu Grad: dx=dy if x>=0 else alpha * dy
// leaky_relu GradGrad: ddy=ddx if x>=0 else alpha * ddx
template <typename T>
class LeakyReluDoubleGradMaker
    : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("leaky_relu_grad_grad");
    // input1: X
    op->SetInput("X", this->Input("X"));
    // X@GRAD@GRAD: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());
    // Out@GRAD@GRAD: ddy
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// elu grad: dx=dy if y>0 else alpha*dy*x.exp()
// elu gradgrad: ddx=ddy if y>0 else alpha*ddy*x.exp()
template <typename T>
class ELUDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elu_grad_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    // X@GRAD@GRAD: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());

    // Out@GRAD@GRAD: ddy
    op->SetOutput("DX", this->InputGrad("X"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// celu grad: dx=dy if y>0 else dy*(x/alpha).exp()
// celu gradgrad: ddx=ddy if y>0 else ddy*(x/alpha).exp()/alpha
template <typename T>
class CELUDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("celu_grad_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    // X@GRAD@GRAD: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());

    // Out@GRAD@GRAD: ddy
    op->SetOutput("DX", this->InputGrad("X"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// sqrt Grad: dx = 0.5 * dy / y
// sqrt GradGrad: ddy = 0.5 * ddx / y, dy = -1 * dx * ddx
template <typename T>
class SqrtDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sqrt_grad_grad");
    op->SetInput("Out", this->Input("Out"));
    op->SetInput("DX", this->Output(framework::GradVarName("X")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());
    op->SetOutput("DOut", this->InputGrad("Out"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// rsqrt Grad: dx = -0.5 * dy * y * y * y
// rsqrt GradGrad: ddy = -0.5 * ddx * y * y * y, dy = (3/y) * ddx
template <typename T>
class RsqrtDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("rsqrt_grad_grad");
    op->SetInput("Out", this->Input("Out"));
    op->SetInput("DX", this->Output(framework::GradVarName("X")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetAttrMap(this->Attrs());
    op->SetOutput("DOut", this->InputGrad("Out"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// square Grad: dx=2x*dy
// square GradGrad: ddy=2x*ddx, dx=2dy*ddx
template <typename T>
class SquareDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("square_grad_grad");
    op->SetInput("X", this->Input("X"));
    // Out@GRAD: dy
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    // X@GRAD@GRAD: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));

    op->SetAttrMap(this->Attrs());

    // X@GRAD: dx
    op->SetOutput("DX", this->InputGrad("X"));
    // Out@GRAD@GRAD: ddy
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

// log Grad: dx = dout / x
// log Grad Grad: ddout = ddx / x; dx = -(dout / x) * (ddx / x)
template <typename T>
class LogDoubleGradMaker : public ::paddle::framework::SingleGradOpMaker<T> {
 public:
  using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("log_grad_grad");
    op->SetInput("X", this->Input("X"));
    // X@GRAD@GRAD: ddx
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetAttrMap(this->Attrs());
    // X@GRAD: dx
    op->SetOutput("DX", this->InputGrad("X"));
    // Out@GRAD@GRAD: ddy
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

DECLARE_INPLACE_OP_INFERER(ActivationGradOpInplaceInferer,
                           {framework::GradVarName("Out"),  // dout
                            framework::GradVarName("X")});  // dx
DECLARE_INPLACE_OP_INFERER(ActivationDoubleGradOpInplaceInferer,
                           {"DDX", "DDOut"});
DECLARE_INPLACE_OP_INFERER(ActivationTripleGradOpInplaceInferer,
                           {"DDX", "D_DOut"});

template <typename T>
class PowGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pow_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetInput("FactorTensor", this->Input("FactorTensor"));
    op->SetAttrMap(this->Attrs());
  }
};
class PowOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "FactorTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class PowOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    ctx->ShareDim(out_grad_name, framework::GradVarName("X"));
    ctx->ShareLoD(out_grad_name, framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, framework::GradVarName("Out"));
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "FactorTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};
DECLARE_INPLACE_OP_INFERER(ActFwdInplaceInferer, {"X", "Out"});
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_OP(KERNEL_TYPE, OP_NAME, functor, grad_functor) \
  REGISTER_OPERATOR(                                                        \
      KERNEL_TYPE,                                                          \
      ops::ActivationOp,                                                    \
      ops::OP_NAME##OpMaker,                                                \
      ops::ActivationOpInferVarType,                                        \
      ops::ActivationGradOpMaker<ops::grad_functor<float>::FwdDeps(),       \
                                 paddle::framework::OpDesc>,                \
      ops::ActivationGradOpMaker<ops::grad_functor<float>::FwdDeps(),       \
                                 paddle::imperative::OpBase>,               \
      std::conditional<ops::CanInplaceAct<ops::grad_functor<float>>(),      \
                       ops::ActFwdInplaceInferer,                           \
                       void>::type);                                        \
  REGISTER_OPERATOR(KERNEL_TYPE##_grad,                                     \
                    ops::ActivationOpGrad,                                  \
                    ops::ActivationGradOpInplaceInferer);

#define REGISTER_ACTIVATION_CPU_KERNEL(                                     \
    act_type, op_name, functor, grad_functor)                               \
  REGISTER_OP_CPU_KERNEL(                                                   \
      act_type,                                                             \
      ops::ActivationKernel<phi::CPUContext, ops::functor<float>>,          \
      ops::ActivationKernel<phi::CPUContext, ops::functor<double>>);        \
  REGISTER_OP_CPU_KERNEL(                                                   \
      act_type##_grad,                                                      \
      ops::ActivationGradKernel<phi::CPUContext, ops::grad_functor<float>>, \
      ops::ActivationGradKernel<phi::CPUContext, ops::grad_functor<double>>);

FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_OP);
FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_CPU_KERNEL);

REGISTER_ACTIVATION_OP(brelu, BRelu, BReluFunctor, BReluGradFunctor);
REGISTER_ACTIVATION_OP(thresholded_relu,
                       ThresholdedRelu,
                       ThresholdedReluFunctor,
                       ThresholdedReluGradFunctor);
REGISTER_ACTIVATION_OP(relu6, Relu6, Relu6Functor, Relu6GradFunctor);
REGISTER_ACTIVATION_OP(softshrink,
                       SoftShrink,
                       SoftShrinkFunctor,
                       SoftShrinkGradFunctor);
REGISTER_ACTIVATION_OP(tanh_shrink,
                       TanhShrink,
                       TanhShrinkFunctor,
                       TanhShrinkGradFunctor);
REGISTER_ACTIVATION_OP(softsign,
                       Softsign,
                       SoftsignFunctor,
                       SoftsignGradFunctor);
REGISTER_ACTIVATION_OP(softplus,
                       Softplus,
                       SoftplusFunctor,
                       SoftplusGradFunctor);
REGISTER_ACTIVATION_OP(mish, Mish, MishFunctor, MishGradFunctor);
REGISTER_ACTIVATION_OP(stanh, STanh, STanhFunctor, STanhGradFunctor);
REGISTER_ACTIVATION_OP(hard_swish,
                       HardSwish,
                       HardSwishFunctor,
                       HardSwishGradFunctor);
REGISTER_ACTIVATION_OP(swish, Swish, SwishFunctor, SwishGradFunctor);

/* ==========================    sigmoid register  =============================
 */
// 1. Register Sigmoid Operator
REGISTER_OPERATOR(
    sigmoid,
    ops::ActivationOp,
    ops::SigmoidOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::SigmoidGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::SigmoidGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    std::conditional<ops::CanInplaceAct<ops::SigmoidGradFunctor<float>>(),
                     ops::ActFwdInplaceInferer,
                     void>::type);

// 2. Register Sigmoid Grad Operator
REGISTER_OPERATOR(sigmoid_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::SigmoidDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::SigmoidDoubleGradMaker<paddle::imperative::OpBase>);

// 3. Register Sigmoid DoubleGrad Operator
REGISTER_OPERATOR(
    sigmoid_grad_grad,
    ops::ActivationOpDoubleGrad<ops::SigmoidGradGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer,
    ops::SigmoidTripleGradMaker<paddle::framework::OpDesc>,
    ops::SigmoidTripleGradMaker<paddle::imperative::OpBase>);

// 4. Register Sigmoid TripleGrad Operator
REGISTER_OPERATOR(sigmoid_triple_grad,
                  ops::ActivationOpTripleGrad<
                      ops::SigmoidTripleGradFunctor<float>::FwdDeps()>,
                  ops::ActivationTripleGradOpInplaceInferer);

/* ========================================================================== */

/* ==========================    tanh register  ============================= */
REGISTER_OPERATOR(
    tanh,
    ops::ActivationOp,
    ops::TanhOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::TanhGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::TanhGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    std::conditional<ops::CanInplaceAct<ops::TanhGradFunctor<float>>(),
                     ops::ActFwdInplaceInferer,
                     void>::type);
REGISTER_OPERATOR(tanh_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::TanhDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::TanhDoubleGradMaker<paddle::imperative::OpBase>)
REGISTER_OPERATOR(
    tanh_grad_grad,
    ops::ActivationOpDoubleGrad<ops::TanhGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer,
    ops::TanhTripleGradMaker<paddle::framework::OpDesc>,
    ops::TanhTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    tanh_triple_grad,
    ops::ActivationOpTripleGrad<ops::TanhTripleGradFunctor<float>::FwdDeps()>,
    ops::ActivationTripleGradOpInplaceInferer);

/* ========================================================================== */

/* ==========================    relu register  ============================= */
REGISTER_OPERATOR(
    relu,
    ops::ActivationOp,
    ops::ReluOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::ReluGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::ReluGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(relu_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::ReluDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::ReluDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    relu_grad_grad,
    ops::ActivationOpDoubleGrad2<ops::ReluGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ======================== leaky relu register  ============================ */
REGISTER_OPERATOR(
    leaky_relu,
    ops::ActivationOp,
    ops::LeakyReluOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::LeakyReluGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::LeakyReluGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(leaky_relu_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::LeakyReluDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::LeakyReluDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    leaky_relu_grad_grad,
    ops::ActivationOpDoubleGrad2<ops::LeakyReluGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ========================    elu  register     ============================ */
REGISTER_OPERATOR(elu,
                  ops::ActivationOp,
                  ops::ELUOpMaker,
                  ops::ActivationOpInferVarType,
                  ops::ELUGradOpMaker<paddle::framework::OpDesc>,
                  ops::ELUGradOpMaker<paddle::imperative::OpBase>,
                  ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(elu_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::ELUDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::ELUDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    elu_grad_grad,
    ops::ActivationOpDoubleGrad<ops::ELUGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ========================    celu  register     ============================
 */
REGISTER_OPERATOR(
    celu,
    ops::ActivationOp,
    ops::CELUOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::CELUGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::CELUGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(celu_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::CELUDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::CELUDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    celu_grad_grad,
    ops::ActivationOpDoubleGrad<ops::CELUGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ===========================   sqrt register  ============================= */
REGISTER_OPERATOR(
    sqrt,
    ops::ActivationOp,
    ops::SqrtOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::SqrtGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::SqrtGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(sqrt_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::SqrtDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::SqrtDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    sqrt_grad_grad,
    ops::ActivationOpDoubleGrad<ops::SqrtGradGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ===========================   rsqrt register  =============================
 */
REGISTER_OPERATOR(
    rsqrt,
    ops::ActivationOp,
    ops::RsqrtOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::RsqrtGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::RsqrtGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(rsqrt_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::RsqrtDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::RsqrtDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    rsqrt_grad_grad,
    ops::ActivationOpDoubleGrad<ops::RsqrtGradGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ==========================   square register  ============================ */
REGISTER_OPERATOR(
    square,
    ops::ActivationOp,
    ops::SquareOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::SquareGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::SquareGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(square_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::SquareDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::SquareDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(
    square_grad_grad,
    ops::ActivationOpDoubleGrad<ops::SquareGradGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ========================================================================== */

/* ==========================   pow register  ============================ */

REGISTER_OPERATOR(
    pow,
    ops::PowOp,
    ops::PowOpMaker,
    ops::ActivationOpInferVarType,
    ops::PowGradOpMaker<paddle::framework::OpDesc>,
    ops::PowGradOpMaker<paddle::imperative::OpBase>,
    std::conditional<ops::CanInplaceAct<ops::PowGradFunctor<float>>(),
                     ops::ActFwdInplaceInferer,
                     void>::type);
REGISTER_OPERATOR(pow_grad,
                  ops::PowOpGrad,
                  ops::ActivationGradOpInplaceInferer);
/* ========================================================================== */

/* ==========================  Log register ==================================*/
REGISTER_OPERATOR(
    log,
    ops::ActivationOp,
    ops::LogOpMaker,
    ops::ActivationOpInferVarType,
    ops::ActivationGradOpMaker<ops::LogGradFunctor<float>::FwdDeps(),
                               paddle::framework::OpDesc>,
    ops::ActivationGradOpMaker<ops::LogGradFunctor<float>::FwdDeps(),
                               paddle::imperative::OpBase>,
    ops::ActFwdInplaceInferer);
REGISTER_OPERATOR(log_grad,
                  ops::ActivationOpGrad,
                  ops::ActivationGradOpInplaceInferer,
                  ops::LogDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::LogDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    log_grad_grad,
    ops::ActivationOpDoubleGrad<ops::LogGradGradFunctor<float>::FwdDeps()>,
    ops::ActivationDoubleGradOpInplaceInferer);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(leaky_relu)
    .AddCheckpoint(
        R"ROC(fix leaky_relu, bahavior changed when alpha < 0 or alpha > 1)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .BugfixWithBehaviorChanged(
                "leaky_relu calculate formula before checkponit: out = max(x, "
                "alpha * x); after checkpoint: out = x if x > 0 else alpha * "
                "x"));

REGISTER_OP_VERSION(hard_shrink)
    .AddCheckpoint(
        R"ROC(fix hard_shrink, bahavior changed when threshold<0)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .BugfixWithBehaviorChanged(
                "hard_shrink calculate formula before checkponit: out = x * "
                "((x < -threshold) + (x > threshold)); after checkpoint: out = "
                "x * (((x < -threshold) + (x > threshold)) > 0)"));

REGISTER_OP_VERSION(softplus).AddCheckpoint(
    R"ROC(add new attributes [beta] and [threshold], and the formula is changed to "
         " softplus(x) = \\frac{1}{beta} * \\log(1 + e^{beta * x}) \\\\ \\text{For numerical"
         " stability, the implementation reverts to the linear function when: beta * x > threshold.})ROC",
    paddle::framework::compatible::OpVersionDesc()
        .NewAttr("beta", "The beta value of the new formula", 1.0f)
        .NewAttr("threshold", "The threshold value of the new formula", 20.0f));

REGISTER_OP_VERSION(mish).AddCheckpoint(
    R"ROC(add new attributes [use_mkldnn], and when computing softplus the formula is changed as the new veriosn of softplus)ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "use_mkldnn",
        "(bool, default false) Only used in mkldnn kernel",
        false));

/* ========================================================================== */
