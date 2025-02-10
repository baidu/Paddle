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

#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle::operators {

using BlockDesc = framework::BlockDesc;

class RunProgramOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"),
                      true,
                      common::errors::NotFound(
                          "Input(X) of RunProgramOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutputs("Out"),
                      true,
                      common::errors::NotFound(
                          "Output(Out) of RunProgramOp should not be null."));
  }

 protected:
  /* [Why use single type kernel]:
   *
   * This op is similar to a control flow op, it doses not need
   * a op kernel, but in order to make it execute under dynamic
   * graph mode, implement it with op kernel.
   *
   * So whether the kernel data type is int, float or other type,
   * which has no effect on its execution logic, so directly
   * specified a data type here.
   *
   * Of course, the data type here is also not important.
   */
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          expected_kernel_type.layout(),
                          expected_kernel_type.dtype());
  }
};

class RunProgramOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<phi::DenseTensor>)"
             "The input tensors of RunProgram operator, also the feed targets "
             "of loaded program.")
        .AsDuplicable();
    AddInput("Params",
             "(vector<phi::DenseTensor or SelectedRows>)"
             "The input parameter of RunProgram operator, also the parameters "
             "of the loaded program.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "(vector<phi::DenseTensor>)"
              "The output tensors of RunProgram operator, also the fetch "
              "targets of the loaded program.")
        .AsDuplicable();
    AddOutput("OutScope",
              "(StepScopeVar)"
              "A vector of execution scope in RunProgram operator, which "
              "contains at most one scope."
              "NOTE: Do not use Scope directly because Scope output is not "
              "currently supported.");
    AddOutput("DOut",
              "(vector<phi::DenseTensor>)"
              "The output tensors for GRAD Tensors in RunProgram forward "
              "operator, the forward operator contains GRAD Tensors when it "
              "computes double grad.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("CUDAGraph", "The output CUDA Graph when use_cuda_graph=True.")
        .AsDispensable();
    AddAttr<BlockDesc*>("global_block",
                        "(BlockDesc *)"
                        "The global block of executed program desc.");
    AddAttr<int64_t>("start_op_index",
                     "(int64_t)"
                     "The index of the op to start execution");
    AddAttr<int64_t>("end_op_index",
                     "(int64_t)"
                     "The index of the op to stop execution");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);
    AddAttr<bool>(
        "in_pir_pt_mode",
        "(bool, default false) Set to true when need to run in pir mode")
        .SetDefault(false);
    AddAttr<int64_t>(
        "program_id",
        "(int64_t)"
        "The unique hash id used as cache key for ExecutorInfoCache.");
    AddAttr<std::string>("cuda_graph_capture_mode",
                         "(str, default '') The CUDA Graph capture mode. "
                         "Default '' means no CUDA Graph capturing.")
        .SetDefault("");
    AddAttr<int64_t>("cuda_graph_pool_id",
                     "(int64_t, default 0) The CUDA Graph memory pool ID.")
        .SetDefault(0);
    AddAttr<bool>("use_interpretorcore",
                  "(bool, default false) Set to true for use interpretercore.")
        .SetDefault(false);
    AddAttr<BlockDesc*>("forward_global_block",
                        "(BlockDesc *)"
                        "The global block of executed forward program desc.")
        .SetDefault(nullptr);
    AddAttr<BlockDesc*>("backward_global_block",
                        "(BlockDesc *)"
                        "The global block of executed backward program desc.")
        .SetDefault(nullptr);
    AddAttr<std::vector<std::string>>("param_grad_names",
                                      "std::vector<std::string>"
                                      "The names of parameter gradients.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("out_grad_names",
                                      "std::vector<std::string>"
                                      "The names of output gradients.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("x_names",
                                      "std::vector<std::string>"
                                      "The names of input tensors.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("x_grad_names",
                                      "std::vector<std::string>"
                                      "The names of input gradients.")
        .SetDefault({});
    AddComment(R"DOC(
RunProgram operator.

The RunProgram operator receives a program's feed targets, fetch targets,
and parameters, and receives the forward and backward program desc
as attributes, and then executes the program by executor.

NOTE: This operator is added so that the inference model stored by
`fluid.io.save_inference_model` under the static graph mode can be loaded
under the dynamic graph mode for fine-tuning or inferencing.

)DOC");
  }
};

class RunProgramGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"),
                      true,
                      common::errors::NotFound(
                          "Input(X) of RunProgramGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs(framework::GradVarName("Out")),
        true,
        common::errors::NotFound(
            "Input(Out@GRAD) of RunProgramGradOp should not be null."));
    // NOTE: The X@GRAD and Params@GRAD may not exist,
    // because they can be set stop_gradient = True
  }

 protected:
  /* see [Why use single type kernel] */
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          expected_kernel_type.layout(),
                          expected_kernel_type.dtype());
  }
};

template <typename T>
struct FilterHelper {};

template <>
struct FilterHelper<imperative::OpBase> {
  static void filter(const BlockDesc* desc,
                     imperative::TracedVarList<imperative::VarBase,
                                               imperative::kBackward>* vec) {
    auto f = [desc](std::shared_ptr<imperative::VarBase> ptr) {
      return !desc->HasVar(ptr->Name());
    };
    auto new_end = std::remove_if(vec->begin(), vec->end(), f);
    vec->resize(new_end - vec->begin());
  }
};

template <>
struct FilterHelper<framework::OpDesc> {
  static void filter(const BlockDesc* desc, std::vector<std::string>* vec) {
    auto f = [desc](const std::string& name) { return !desc->HasVar(name); };
    auto new_end = std::remove_if(vec->begin(), vec->end(), f);
    vec->resize(new_end - vec->begin());
  }
};

template <typename T>
class RunProgramGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("run_program_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Params", this->Input("Params"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetInput("OutScope", this->Output("OutScope"));
    grad_op->SetInput("DOut", this->Output("DOut"));
    if (this->HasOutput("CUDAGraph")) {
      grad_op->SetInput("CUDAGraph", this->Output("CUDAGraph"));
    }
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    auto block_desc =
        PADDLE_GET_CONST(BlockDesc*, this->GetAttr("global_block"));
    auto params_grad = this->InputGrad("Params");
    FilterHelper<T>::filter(block_desc, &params_grad);  // filter the vector.
    grad_op->SetOutput(framework::GradVarName("Params"), params_grad);
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;
REGISTER_OPERATOR(run_program,
                  ops::RunProgramOp,
                  ops::RunProgramOpMaker,
                  ops::RunProgramGradOpMaker<paddle::framework::OpDesc>,
                  ops::RunProgramGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(run_program_grad, ops::RunProgramGradOp);
