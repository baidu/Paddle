/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <vector>
#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using LoDTensor = framework::LoDTensor;

constexpr char kStepBlock[] = "step_block";
constexpr char kCondition[] = "Condition";
constexpr char kStepScopes[] = "StepScopes";
constexpr char kParameters[] = "X";
constexpr char kParamGrads[] = "X@Grad";
constexpr char kOutputs[] = "Out";

class WhileOp : public framework::OperatorBase {
 public:
  WhileOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(Input(kCondition)));
    auto &cond = scope.FindVar(Input(kCondition))->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(cond.dims(), paddle::framework::make_ddim({1}));

    framework::Executor executor(dev_ctx);
    auto *block = Attr<framework::BlockDescBind *>(kStepBlock);
    auto *program = block->Program();

    auto step_scopes =
        scope.FindVar(Output(kStepScopes))->GetMutable<StepScopeVar>();

    while (cond.data<bool>()[0]) {
      auto &current_scope = scope.NewScope();
      step_scopes->push_back(&current_scope);

      executor.Run(*program, &current_scope, block->ID(),
                   false /*create_local_scope*/);
    }
  }
};

class WhileOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  WhileOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(kParameters,
             "A set of variables, which are required by operators inside the "
             "block of While Op.")
        .AsDuplicable();
    AddInput(
        kCondition,
        "(Bool) An scalar. When it's False, the While Op will be terminated.")
        .AsDuplicable();
    AddOutput(kOutputs,
              "A set of variables, which will be assigned with values "
              "generated by the operators inside the block of While Op.")
        .AsDuplicable();
    AddOutput(kStepScopes,
              "(StepScopeVar) A vector of local scope, which size equals the "
              "step number of While Op. The i'th scope storages temporary "
              "variables generated in the i'th step.");
    AddAttr<framework::BlockDescBind *>(kStepBlock,
                                        "The step block inside WhileOp");
    AddComment(R"DOC(
)DOC");
  }
};

class WhileGradOp : public framework::OperatorBase {
 public:
  WhileGradOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    //    PADDLE_ENFORCE(...)

    framework::Executor executor(dev_ctx);
    auto *block = Attr<framework::BlockDescBind *>(kStepBlock);
    auto *program = block->Program();

    auto *step_scopes =
        scope.FindVar(Input(kStepScopes))->GetMutable<StepScopeVar>();

    for (auto cur_scope_iter = step_scopes->rbegin();
         cur_scope_iter != step_scopes->rend(); ++cur_scope_iter) {
      executor.Run(*program, *cur_scope_iter, block->ID(), false);

      auto &pg_names = Outputs(kParamGrads);
      auto &p_names = Inputs(kParameters);
      PADDLE_ENFORCE_EQ(pg_names.size(), p_names.size());
      for (size_t prog_id = 0; prog_id < pg_names.size(); ++prog_id) {
        auto inside_grad_name = framework::GradVarName(p_names[prog_id]);

        //  // TODO(tonyyang-savil: Not sure we need the following
        //  // If does not compute gradient of that variable inside rnn,
        //  just
        //  // continue
        //  if (local_var_names.find(inside_grad_name) ==
        //  local_var_names.end()) {
        //    continue;
        //  }

        // zero gradient variable in step 0
        if (cur_scope_iter == step_scopes->rbegin()) {
          auto *var = (*cur_scope_iter)->FindVar(inside_grad_name);
          PADDLE_ENFORCE_NOT_NULL(var);
          if (var->IsType<LoDTensor>()) {
            auto &inside_tensor = var->Get<framework::LoDTensor>();
            framework::AttributeMap attrs;
            attrs["data_type"] = framework::ToDataType(inside_tensor.type());
            attrs["shape"] = framework::vectorize2int(inside_tensor.dims());
            attrs["value"] = 0.0f;

            auto zero_op = framework::OpRegistry::CreateOp(
                "fill_constant", {}, {{"Out", {pg_names[prog_id]}}}, attrs);
            zero_op->Run(scope, dev_ctx);
          }
        }

        // sum gradient
        auto *outside_var = scope.FindVar(pg_names[prog_id]);
        PADDLE_ENFORCE_NOT_NULL(outside_var);
        auto &outside_tensor = *outside_var->GetMutable<framework::LoDTensor>();

        std::string result_var_name;
        auto *local_result_var = (*cur_scope_iter)->Var(&result_var_name);
        auto &local_result_tensor =
            *local_result_var->GetMutable<framework::LoDTensor>();

        local_result_tensor.ShareDataWith(outside_tensor);

        auto sum_op = framework::OpRegistry::CreateOp(
            "sum", {{"X", {result_var_name, inside_grad_name}}},
            {{"Out", {result_var_name}}}, {});
        sum_op->Run(**cur_scope_iter, dev_ctx);
      }
    }
  }
};

class WhileGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDescBind> Apply() const {
    auto *grad = new framework::OpDescBind();
    grad->SetType("while_grad");
    for (auto &input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(framework::GradVarName(input_param),
                      this->InputGrad(input_param));
    }

    for (auto &output_param : this->OutputNames()) {
      grad->SetInput(output_param, this->Output(output_param));
      if (output_param != kStepScopes) {
        grad->SetInput(framework::GradVarName(output_param),
                       this->OutputGrad(output_param));
      }
    }
    grad->SetAttrMap(this->Attrs());
    grad->SetBlockAttr(kStepBlock, *grad_block_[0]);

    return std::unique_ptr<framework::OpDescBind>(grad);
  }
};

class WhileGradOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    std::vector<std::string> input{kParameters};
    for (auto &s : input) {
      for (auto &in_name : op_desc.Input(s)) {
        auto *in_var = block->FindVarRecursive(in_name);
        PADDLE_ENFORCE_NOT_NULL(in_var, "Can't find %s", in_name);
        auto in_var_type = in_var->GetType();
        block->FindRecursiveOrCreateVar(framework::GradVarName(in_name))
            ->SetType(in_var_type);
      }
    }
  }
};

class WhileGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    std::vector<std::string> input{kParameters};
    std::vector<std::string> output{kOutputs};
    for (auto &s : input) {
      PADDLE_ENFORCE(ctx->HasInputs(s));
      PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(s)));
    }
    for (auto &s : output) {
      PADDLE_ENFORCE(ctx->HasInputs(s));
      PADDLE_ENFORCE(ctx->HasInputs(framework::GradVarName(s)));
    }
    for (auto &s : input) {
      auto names = ctx->Inputs(s);
      auto dims = ctx->GetInputsDim(s);
      auto var_types = ctx->GetInputsVarType(s);
      std::vector<std::string> names_to_set;
      std::vector<framework::DDim> dims_to_set;
      for (size_t i = 0; i < names.size(); ++i) {
        if (var_types[i] == framework::VarDesc::LOD_TENSOR) {
          names_to_set.push_back(framework::GradVarName(names[i]));
          dims_to_set.push_back(dims[i]);
        } else if (var_types[i] == framework::VarDesc::LOD_TENSOR_ARRAY) {
          // not sure how to set the dim of LOD_TENSOR_ARRAY
          names_to_set.push_back(framework::GradVarName(names[i]));
          dims_to_set.push_back(dims[i]);
        }
      }
      ctx->SetDims(names_to_set, dims_to_set);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(while, paddle::operators::WhileOp,
                  paddle::operators::WhileOpMaker,
                  paddle::operators::WhileGradOpDescMaker);
REGISTER_OPERATOR(while_grad, paddle::operators::WhileGradOp,
                  paddle::operators::WhileGradOpShapeInference,
                  paddle::operators::WhileGradOpVarTypeInference);
