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
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using LoDTensor = framework::LoDTensor;

constexpr char kStepBlock[] = "step_block";
constexpr char kCondition[] = "Condition";
constexpr char kStepScopes[] = "StepScopes";
constexpr char kParameters[] = "X";
constexpr char kParamGrads[] = "X@GRAD";
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

    auto outside_og_names = Inputs(framework::GradVarName(kOutputs));
    auto inside_og_names =
        Attr<std::vector<std::string>>("original_output_grad");

    PADDLE_ENFORCE_EQ(outside_og_names.size(), inside_og_names.size());

    for (auto cur_scope_iter = step_scopes->rbegin();
         cur_scope_iter != step_scopes->rend(); ++cur_scope_iter) {
      VLOG(3) << "Start backward at time_step "
              << cur_scope_iter - step_scopes->rbegin();
      framework::Scope &cur_scope = **cur_scope_iter;
      // Link OG from outside to inside
      for (size_t i = 0; i < outside_og_names.size(); ++i) {
        auto outside_og_name = outside_og_names[i];
        auto inside_og_name = inside_og_names[i];
        VLOG(10) << "Linking outside " << outside_og_name << " --> inside "
                 << inside_og_name;
        auto &og_outside = detail::Ref(scope.FindVar(outside_og_name));
        auto &og_inside = detail::Ref(cur_scope.Var(inside_og_name));
        if (og_outside.Type().hash_code() ==
            typeid(framework::LoDTensor).hash_code()) {
          auto &outside_tensor = og_outside.Get<framework::LoDTensor>();
          auto &inside_tensor =
              detail::Ref(og_inside.GetMutable<framework::LoDTensor>());
          inside_tensor.set_lod(outside_tensor.lod());
          inside_tensor.ShareDataWith(outside_tensor);
        } else if (og_outside.Type().hash_code() ==
                   typeid(framework::LoDTensorArray).hash_code()) {
          auto &outside_array = og_outside.Get<framework::LoDTensorArray>();
          auto &inside_array =
              detail::Ref(og_inside.GetMutable<framework::LoDTensorArray>());
          VLOG(10) << outside_og_name << " size = " << outside_array.size();
          inside_array.resize(outside_array.size());

          for (size_t j = 0; j < inside_array.size(); ++j) {
            VLOG(10) << j << " " << outside_array[j].numel();
            if (outside_array[j].numel() != 0) {
              inside_array[j].set_lod(outside_array[j].lod());
              inside_array[j].ShareDataWith(outside_array[j]);
            } else {
              PADDLE_ENFORCE_EQ(inside_array[j].numel(), 0);
            }
          }
        }
      }

      executor.Run(*program, *cur_scope_iter, block->ID(), false);

      auto &pg_names = Outputs(kParamGrads);
      auto &p_names = Inputs(kParameters);
      PADDLE_ENFORCE_EQ(pg_names.size(), p_names.size());
      for (size_t param_id = 0; param_id < pg_names.size(); ++param_id) {
        if (pg_names[param_id] == framework::kEmptyVarName) {
          continue;  // iterator doesn't have gradient
        }
        auto inside_grad_name = framework::GradVarName(p_names[param_id]);

        //  // TODO(tonyyang-svail): Not sure we need the following
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
          PADDLE_ENFORCE_NOT_NULL(var, "Can not find var %s", inside_grad_name);
          if (var->IsType<LoDTensor>()) {
            auto &inside_tensor = var->Get<framework::LoDTensor>();
            framework::AttributeMap attrs;
            attrs["dtype"] = framework::ToDataType(inside_tensor.type());
            attrs["shape"] = framework::vectorize2int(inside_tensor.dims());
            attrs["value"] = 0.0f;

            auto zero_op = framework::OpRegistry::CreateOp(
                "fill_constant", {}, {{"Out", {pg_names[param_id]}}}, attrs);
            zero_op->Run(scope, dev_ctx);
          }
        }

        // sum gradient
        auto new_inside_name = cur_scope.Rename(inside_grad_name);
        auto sum_op = framework::OpRegistry::CreateOp(
            "sum", {{"X", {pg_names[param_id], new_inside_name}}},
            {{"Out", {pg_names[param_id]}}}, {});
        sum_op->Run(cur_scope, dev_ctx);
        cur_scope.Rename(new_inside_name, inside_grad_name);
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
    grad->SetInput(kParameters, Input(kParameters));
    grad->SetOutput(
        framework::GradVarName(kParameters),
        InputGrad(kParameters, /*do not drop empty gradient*/ false));
    grad->SetInput(kOutputs, Output(kOutputs));

    // OG should be re-calculated by step blocks, since many outputs of while op
    // do not need to calculate gradients.
    std::unordered_set<std::string> block_ins;
    {
      for (auto &p : Input(kParameters)) {
        block_ins.insert(p);
      }
      for (auto &o : Output(kOutputs)) {
        block_ins.insert(o);
      }
    }
    std::unordered_set<std::string> extra_inputs;
    for (size_t i = 0; i < grad_block_[0]->OpSize(); ++i) {
      for (auto &input_name : grad_block_[0]->Op(i)->InputArgumentNames()) {
        if (block_ins.find(input_name) != block_ins.end()) {
          continue;
        }
        extra_inputs.insert(input_name);
      }

      for (auto &output_name : grad_block_[0]->Op(i)->OutputArgumentNames()) {
        block_ins.insert(output_name);
      }
    }

    std::vector<std::string> extra_inputs_list;
    extra_inputs_list.resize(extra_inputs.size());
    std::copy(extra_inputs.begin(), extra_inputs.end(),
              extra_inputs_list.begin());
    grad->SetInput(framework::GradVarName(kOutputs), extra_inputs_list);
    grad->SetInput(kStepScopes, Output(kStepScopes));
    grad->SetAttrMap(this->Attrs());
    grad->SetBlockAttr(kStepBlock, *grad_block_[0]);
    // record the original output gradient names, since the gradient name of
    // while operator could be renamed.
    grad->SetAttr("original_output_grad", extra_inputs_list);

    return std::unique_ptr<framework::OpDescBind>(grad);
  }
};

class WhileGradOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    auto p_names = op_desc.Input(kParameters);
    auto pg_names = op_desc.Output(framework::GradVarName(kParameters));

    for (size_t i = 0; i < p_names.size(); ++i) {
      auto &p_var = detail::Ref(block->FindVarRecursive(p_names[i]));
      auto *g_var = block->FindVarRecursive(pg_names[i]);
      if (g_var != nullptr) {  // Gradient could be @EMPTY@
        VLOG(5) << "Setting " << pg_names[i] << " following " << p_names[i]
                << " type: " << p_var.GetType();
        g_var->SetType(p_var.GetType());
        g_var->SetDataType(p_var.GetDataType());
      }
    }
  }
};

class WhileGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    ctx->HasInputs(kParameters);
    ctx->HasOutputs(framework::GradVarName(kParameters));
    ctx->HasInputs(kOutputs);
    ctx->HasInputs(framework::GradVarName(kOutputs));

    auto p_names = ctx->Inputs(kParameters);
    auto pg_names = ctx->Outputs(kParamGrads);
    auto dims = ctx->GetInputsDim(kParameters);
    auto var_types = ctx->GetInputsVarType(kParameters);
    std::vector<std::string> names_to_set;
    std::vector<framework::DDim> dims_to_set;
    for (size_t i = 0; i < p_names.size(); ++i) {
      if (pg_names[i] == framework::kEmptyVarName) {
        continue;
      }
      if (var_types[i] == framework::VarDesc::LOD_TENSOR) {
        names_to_set.push_back(pg_names[i]);
        dims_to_set.push_back(dims[i]);
      } else if (var_types[i] == framework::VarDesc::LOD_TENSOR_ARRAY) {
        // not sure how to set the dim of LOD_TENSOR_ARRAY
        names_to_set.push_back(pg_names[i]);
        dims_to_set.push_back(dims[i]);
      }
    }
    ctx->SetDims(names_to_set, dims_to_set);
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
