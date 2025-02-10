// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/cinn/hlir/framework/pir/compilation_task.h"

#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace hlir {
namespace framework {

void GroupCompilationContext::SetLoweredFuncs(
    BucketLoweredFuncsWrapper&& funcs) {
  for (std::tuple<ir::SymbolicPredicate, ir::LoweredFunc, int>& predicate2func :
       funcs.predicate2funcs) {
    predicates_.push_back(std::move(std::get<0>(predicate2func)));
    lowered_funcs_.push_back(std::move(std::get<1>(predicate2func)));
    priorities_.push_back(std::move(std::get<2>(predicate2func)));
  }
  for (std::pair<ir::SymbolicPredicate, ir::LoweredFunc>& predicate2func :
       funcs.predicate2funcsCX86) {
    CX86_predicates_.push_back(std::move(predicate2func.first));
    CX86_lowered_funcs_.push_back(std::move(predicate2func.second));
  }
  // TODO(Dmovic): remove expr body after update all the backend.
  for (ir::LoweredFunc& func : lowered_funcs_) {
    func->body_block = ir::ConvertExprBlockToStmtBlock(func->body);
  }
  for (ir::LoweredFunc& func : CX86_lowered_funcs_) {
    func->body_block = ir::ConvertExprBlockToStmtBlock(func->body);
  }
  infer_shape_lowered_func_ = std::move(funcs.infer_shape_func);

  need_x86_kernel_ = (CX86_predicates_.size() > 0);
}

std::string GroupCompilationContext::PrintPredicate2Funcs() const {
  std::stringstream ss;
  for (int i = 0; i < predicates_.size(); ++i) {
    ss << "[CONDITION " << i << "]: " << predicates_[i] << "\n";
    ss << "[LOWEREDFUNC " << i << "]: " << lowered_funcs_[i] << "\n\n";
  }
  return ss.str();
}

void GroupCompilationContext::PrepareModuleBuilder() {
  PADDLE_ENFORCE_EQ(predicates_.size(),
                    lowered_funcs_.size(),
                    ::common::errors::InvalidArgument(
                        "The size of predicates and lowered_funcs should be "
                        "the same."));
  PADDLE_ENFORCE_EQ(predicates_.size(),
                    priorities_.size(),
                    ::common::errors::InvalidArgument(
                        "The size of predicates and priorities should be "
                        "the same."));
  for (const ir::Expr& predicate : predicates_) {
    module_builder_.AddPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : lowered_funcs_) {
    module_builder_.AddFunction(func);
  }
  for (int& priority : priorities_) {
    module_builder_.AddPriority(priority);
  }
  module_builder_.SetInferShapeFunc(infer_shape_lowered_func_);

  PADDLE_ENFORCE_EQ(CX86_predicates_.size(),
                    CX86_lowered_funcs_.size(),
                    ::common::errors::InvalidArgument(
                        "The size of predicates and lowered_funcs should be "
                        "the same."));

  for (const ir::Expr& predicate : CX86_predicates_) {
    CX86_module_builder_.AddPredicate(predicate);
  }
  for (const ir::LoweredFunc& func : CX86_lowered_funcs_) {
    CX86_module_builder_.AddFunction(func);
  }
}

/**
 * For functions belonging to different broadcast groups, int args and the name
 * of the tensor args may be variate, but the number of the tensor args should
 * be fixed. So we need to unify the tensor args and symbol args. For example,
 * func1(_var, _var_1, S4, S5); func2(_var, _var_2, S1) would be unified to
 * func1(_var, _var_1, S4, S5, S1); func2(_var, _var_2, S4, S5, S1).
 */
void UnifyBroadcastGroupFuncArgs(
    std::vector<GroupCompilationContext>* contexts,
    pir::OpLoweringGroupPtr origin_group,
    std::unordered_map<int, ir::Var>* symbolic_shape_var_index) {
  std::vector<ir::Argument> new_args_vec;

  const auto& AddTensorArgs = [&]() {
    const auto& func_args = (*contexts)[0].lowered_funcs_[0]->args;
    for (size_t arg_idx = 0; arg_idx < func_args.size(); ++arg_idx) {
      if (func_args[arg_idx].is_buffer()) {
        new_args_vec.emplace_back(func_args[arg_idx]);
      }
    }
  };

  std::unordered_set<std::string> symbol_args_set;
  const auto& AddSymbolArgs = [&](::pir::Value input, const int& input_idx) {
    enum ArgType { Dim, Value };
    const auto& AddSymbolArgFromDimExprVec =
        [&](ArgType arg_type, const std::vector<symbol::DimExpr>& expr_vec) {
          int vec_size = expr_vec.size();
          for (int idx = 0; idx < vec_size; idx++) {
            if (expr_vec[idx].isa<std::string>()) {
              const std::string& symbol_name =
                  expr_vec[idx].dyn_cast<std::string>();
              if (symbol_args_set.count(symbol_name) != 0) {
                continue;
              }
              symbol_args_set.insert(symbol_name);
              const auto& arg = ir::Var(symbol_name, cinn::common::Int(64));
              new_args_vec.emplace_back(ir::Argument{arg});
              int arg_idx = new_args_vec.size() - 1;
              symbolic_shape_var_index->insert({arg_idx, arg});
              if (arg_type == Dim) {
                origin_group->mut_symbol_args_map()[arg_idx] =
                    pir::CINNKernelInfo::ArgDimIdx{input_idx, idx};
              } else {
                origin_group->mut_symbol_args_map()[arg_idx] =
                    pir::CINNKernelInfo::ArgValueIdx{input_idx, idx};
              }
            }
          }
        };
    const auto& shape_or_data = origin_group->GetShapeOrDataExprs(input);
    // Add dim symbol args
    AddSymbolArgFromDimExprVec(ArgType::Dim, shape_or_data.shape());
    // Add value symbol args
    if (shape_or_data.data())
      AddSymbolArgFromDimExprVec(ArgType::Value, shape_or_data.data().value());
  };

  const auto& UpdateAllFuncArgs = [&](GroupCompilationContext& context) {
    for (ir::LoweredFunc& func : context.lowered_funcs_) {
      func->args = new_args_vec;
    }
  };

  AddTensorArgs();
  origin_group->mut_symbol_args_map().clear();
  const auto& group_inputs = pir::GetBlockOutsideInput(origin_group->ops());
  for (size_t input_idx = 0; input_idx < group_inputs.size(); ++input_idx)
    AddSymbolArgs(group_inputs[input_idx], input_idx);
  for (int i = 0; i < contexts->size(); ++i) {
    UpdateAllFuncArgs((*contexts)[i]);
  }
}

std::shared_ptr<pir::CompilationResult> CompilationTask::operator()() {
  Lowering();
  return CodegenAndJit();
}

void CompilationTask::Lowering() {
  VLOG(5) << "Begin to lowering group: " << *context_->group_;
  auto op_lowerer = CreateOpLowerer<pir::OpLoweringGroupPtr>(context_->target_);
  context_->SetLoweredFuncs(op_lowerer.BucketLower(context_->group_));

  if (context_->group_->IsBroadcastLeaf()) {
    const auto& broadcast_condition_dimexprs =
        context_->group_->GetBroadcastConditions();

    using BranchType = pir::OpLoweringGroup::BranchType;
    const auto& GetSingleBranchExprFromBroadcastCond =
        [](const symbol::Broadcastable<symbol::DimExpr>& broadcast_expr,
           const BranchType& condition) -> ir::Expr {
      const auto& expr_converter = common::DimExprConverter();
      ir::Expr lhs = expr_converter.ConvertToIrExpr(broadcast_expr->lhs);
      ir::Expr rhs = expr_converter.ConvertToIrExpr(broadcast_expr->rhs);
      ir::Expr one = ir::Expr(1);
      ir::Expr condition_expr;
      if (condition == BranchType::LHS_EQ_RHS) {
        condition_expr = ir::EQ::Make(lhs, rhs);
      } else {
        condition_expr = ir::NE::Make(lhs, rhs);
        ir::Expr eq_one_expr;
        if (condition == BranchType::LHS_EQ_ONE) {
          eq_one_expr = ir::EQ::Make(lhs, one);
        } else {  // BranchType::RHS_EQ_ONE
          eq_one_expr = ir::EQ::Make(rhs, one);
        }
        condition_expr = ir::And::Make(condition_expr, eq_one_expr);
      }
      return condition_expr;
    };

    const auto& ChangeBroadcastConditionToExpr = [&]() -> ir::Expr {
      ir::Expr result = ir::Expr(true);
      for (const auto& [broadcast_expr, condition] :
           broadcast_condition_dimexprs) {
        result = ir::And::Make(
            result,
            GetSingleBranchExprFromBroadcastCond(broadcast_expr, condition));
      }
      return result;
    };

    context_->broadcast_condition_ = ChangeBroadcastConditionToExpr();
  }
  VLOG(5) << "End to lowering: " << context_->PrintPredicate2Funcs();
}

std::shared_ptr<pir::CompilationResult> CompilationTask::CodegenAndJit() {
  context_->PrepareModuleBuilder();

  ir::Module ir_module = context_->module_builder_.Build();
  ir::Module ir_moduleCX86 = context_->CX86_module_builder_.Build();
  return BuildPirCINNKernelInfo(
      ir_module, ir_moduleCX86, context_->NeedCompileCX86Kernel());
}

std::shared_ptr<pir::CompilationResult> CompilationTask::BuildPirCINNKernelInfo(
    const ir::Module& module,
    const ir::Module& CX86module,
    bool need_x86_kernel) {
  auto compilation_result = std::make_shared<pir::CompilationResult>(
      context_->target_, need_x86_kernel);
  auto backend_resource = std::make_shared<pir::BackendResource>(
      context_->target_,
      context_->group_->FuncName(),
      context_->group_->FuncName() + "_infer_shape",
      context_->group_->symbol_args_map(),
      context_->group_->temp_space_sizes());
  VLOG(5) << "Start to compile module into cuda kernel...";
  backend_resource->GetBackendCompiler()->Build(module, "");
  backend_resource->GetBackendCompiler()->AppendCX86(CX86module);
  backend_resource->GetBackendCompiler()->EndCompile();
  compilation_result->SetBackendResource(backend_resource);

  VLOG(5) << "End to compile module into cuda kernel.";
  return compilation_result;
}

std::shared_ptr<pir::CompilationResult>
CompilationTask::CompileBroadcastModules(
    std::vector<GroupCompilationContext>* leaf_group_contexts,
    const std::unordered_map<int, ir::Var>& symbolic_shape_var_index) {
  auto compilation_result =
      std::make_shared<pir::CompilationResult>(context_->target_);
  auto backend_resource = std::make_shared<pir::BackendResource>(
      context_->target_,
      context_->group_->FuncName(),
      context_->group_->FuncName() + "_infer_shape",
      context_->group_->symbol_args_map(),
      context_->group_->temp_space_sizes());

  std::vector<std::string> case_func_names;
  std::vector<ir::Expr> broadcast_conditions;
  for (auto& context : *leaf_group_contexts) {
    context.PrepareModuleBuilder();
    case_func_names.emplace_back(context.group_->FuncName());
    broadcast_conditions.emplace_back(context.broadcast_condition_);
    ir::Module ir_module = context.module_builder_.Build();
    ir::Module ir_moduleCX86 = context.CX86_module_builder_.Build();
    backend_resource->GetBackendCompiler()->Build(ir_module, "");
    backend_resource->GetBackendCompiler()->AppendCX86(ir_moduleCX86);
  }

  ir::Module wrapper_module(
      cinn::backends::CreateSwitchWithBroadcastConditionModule(
          broadcast_conditions,
          case_func_names,
          context_->group_->FuncName(),
          symbolic_shape_var_index));
  backend_resource->GetBackendCompiler()->AppendBroadcastSwitchModule(
      wrapper_module);
  backend_resource->GetBackendCompiler()->EndCompile();
  compilation_result->SetBackendResource(backend_resource);
  VLOG(5) << "End to compile module into cuda kernel.";
  return compilation_result;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
