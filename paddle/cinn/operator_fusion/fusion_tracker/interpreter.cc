// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/fusion_tracker/interpreter.h"
#include "glog/logging.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"

namespace cinn::fusion {

void RunCopyInstr(const std::shared_ptr<CopyInstr>& instr,
                  FusionInterpreter* interpreter) {
  interpreter->scope[instr->new_name_] =
      interpreter->scope.at(instr->origin_name_);
}

void RunCombineInstr(const std::shared_ptr<CombineInstr>& instr,
                     FusionInterpreter* interpreter) {
  // TODO(@wuzhanfei)
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  for (const auto& name : instr->names_) {
    const auto& to_insert = interpreter->scope.at(name);
    new_pattern->Extend(to_insert->fusion_ops);
  }
  VLOG(4) << "After CombineInstr Pattern: \n"
          << GetFusibleOpsExpr(new_pattern->fusion_ops);
  interpreter->scope[instr->result_] = new_pattern;
}

void RunInitPatternInstr(const std::shared_ptr<InitPatternInstr>& instr,
                         FusionInterpreter* interpreter) {
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  new_pattern->fusion_ops.emplace_back(
      interpreter->initialized_lowered_op[(instr->get_idx())]);
  interpreter->scope[instr->result_] = new_pattern;
}

void RunTrivialInlineInstr(const std::shared_ptr<TrivialInlineInstr>& instr,
                           FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->upstream_]->fusion_ops.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "Upstream op must have only one fusion_op."));
  auto upstream_op = std::get<TrivialOp>(
      interpreter->scope[instr->upstream_]->fusion_ops.front());
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();

  auto DoTrivialFusion = [upstream_op](const auto& downstream_op) -> FusibleOp {
    return cinn::hlir::framework::pir::trivial_fusion_detail::
        TrivalxOther_Fusion(upstream_op, downstream_op);
  };

  for (auto downstream_op :
       interpreter->scope[instr->downstream_]->fusion_ops) {
    new_pattern->fusion_ops.emplace_back(
        std::visit(DoTrivialFusion, downstream_op));
  }
  interpreter->scope[instr->result_] = new_pattern;
}

void RunTmpTransformInstr(const std::shared_ptr<TmpTransformInstr>& instr,
                          FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_GT(
      interpreter->scope.count(instr->upstream_),
      0,
      ::common::errors::NotFound("Can not find TmpTransformInstr upstream."));
  PADDLE_ENFORCE_GT(
      interpreter->scope.count(instr->downstream_),
      0,
      ::common::errors::NotFound("Can not find TmpTransformInstr downstream."));

  PADDLE_ENFORCE_EQ(
      interpreter->scope[instr->downstream_]->fusion_ops.size(),
      1,
      ::common::errors::InvalidArgument(
          "Downstream %s must have only one fusion_op.", instr->downstream_));
  auto downstream_op =
      interpreter->scope[instr->downstream_]->fusion_ops.front();
  std::vector<FusibleOp> changed_upstreams;
  for (auto fusion_op : interpreter->scope[instr->upstream_]->fusion_ops) {
    auto upstream_op = std::get<ReduceOp>(fusion_op);
    changed_upstreams = ConcatVector(
        changed_upstreams,
        cinn::hlir::framework::pir::trivial_fusion_detail::
            TransformReduceLoopRange(
                upstream_op, &downstream_op, instr->fake_reduce_iter_idx_));
  }

  // inplace set the upstream
  interpreter->scope[instr->out_upstream_] = std::make_shared<ScopeElement>();
  interpreter->scope[instr->out_upstream_]->fusion_ops = changed_upstreams;
  // inplace set the downstream
  interpreter->scope[instr->out_downstream_] = std::make_shared<ScopeElement>();
  interpreter->scope[instr->out_downstream_]->fusion_ops.push_back(
      downstream_op);
}

void RunTrivialLoopAlignInstr(
    const std::shared_ptr<TrivialLoopAlignInstr>& instr,
    FusionInterpreter* interpreter) {
  PADDLE_ENFORCE_EQ(interpreter->scope[instr->downstream_]->fusion_ops.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "Downstream op must have only one fusion_op."));
  auto upstream_op = std::get<ReduceOp>(
      interpreter->scope[instr->upstream_]->fusion_ops.front());
  auto downstream_op = std::get<TrivialOp>(
      interpreter->scope[instr->downstream_]->fusion_ops.front());
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  new_pattern->fusion_ops.emplace_back(
      cinn::hlir::framework::pir::trivial_fusion_detail::SinkTrivialLoopAlign(
          downstream_op, upstream_op, instr->fake_reduce_iter_idx_));
  interpreter->scope[instr->result_] = new_pattern;
}

void RunItersTransformInstr(const std::shared_ptr<ItersTransformInstr>& instr,
                            FusionInterpreter* interpreter) {
  auto iters_transform = [transform_route = instr->iters_transform_route_](
                             ir::Expr op_expr,
                             ir::Expr aligned_expr) -> ir::Expr {
    for (auto trans : transform_route) {
      op_expr = std::visit(ApplyItersTransform(op_expr, aligned_expr), trans);
    }
    return op_expr;
  };

  auto new_pattern = std::make_shared<ScopeElement>();
  auto fusion_ops = interpreter->scope[instr->source_]->fusion_ops;
  PADDLE_ENFORCE(interpreter->scope.count(instr->aligned_) &&
                     !interpreter->scope[instr->aligned_]->fusion_ops.empty(),
                 ::common::errors::PreconditionNotMet(
                     "ItersTransform to aligend op must be initialized."));
  ir::Expr aligned_expr =
      std::visit(FusibleOp2Expr(),
                 interpreter->scope[instr->aligned_]->fusion_ops.back())[0];
  for (const auto& fusion_op : fusion_ops) {
    ir::Expr op_expr = std::visit(FusibleOp2Expr(), fusion_op).back();
    VLOG(4) << "[ItersTransform] expr before transform: \n" << op_expr;
    ir::Expr transformed_expr = iters_transform(op_expr, aligned_expr);
    if (cinn::hlir::framework::pir::trivial_fusion_detail::IsReduceBody(
            transformed_expr)) {
      new_pattern->fusion_ops.emplace_back(ReduceOp(transformed_expr));
    } else {
      new_pattern->fusion_ops.emplace_back(TrivialOp(transformed_expr));
    }
  }
  interpreter->scope[instr->target_] = new_pattern;
}

void RunReshapeAlignInstr(const std::shared_ptr<ReshapeAlignInstr>& instr,
                          FusionInterpreter* interpreter) {
  const auto expr = std::visit(
      FusibleOp2Expr(), interpreter->scope[instr->input_]->fusion_ops[0])[0];
  VLOG(4) << "Before RunReshapeAlignInstr: \n" << expr;
  auto result = cinn::hlir::framework::pir::trivial_fusion_detail::ReshapeLoop(
      expr, instr->in_shape_, instr->out_shape_);

  auto new_pattern = std::make_shared<ScopeElement>();
  new_pattern->fusion_ops.emplace_back(TrivialOp(result));
  interpreter->scope[instr->result_] = new_pattern;
  VLOG(4) << "After ReshapeAlignInstr: \n" << result;
}

void RunPaddingInstr(const std::shared_ptr<PaddingInstr>& instr,
                     FusionInterpreter* interpreter) {
  ScopeElementPtr new_pattern = std::make_shared<ScopeElement>();
  for (auto fusion_op : interpreter->scope[instr->target_]->fusion_ops) {
    new_pattern->Extend(DoPadding(fusion_op, instr->padding_pos_));
  }
  interpreter->scope[instr->result_] = new_pattern;
}

void RunReturnInstr(const std::shared_ptr<ReturnInstr>& instr,
                    FusionInterpreter* interpreter) {
  using namespace cinn::hlir::framework::pir::trivial_fusion_detail;  // NOLINT
  for (auto fusion_op : interpreter->scope[instr->target_]->fusion_ops) {
    auto exprs = std::visit(FusibleOp2Expr(), fusion_op);
    // Insert if for append loops
    for (const auto& expr : exprs) {
      // interpreter->ret_expr.push_back(expr);
      std::vector<std::string> load_tensor_names;
      for (const auto& tensor : GetOutputTensors(expr)) {
        load_tensor_names.push_back(tensor->name);
      }
      if (AnyFirstInSecond(load_tensor_names, interpreter->global_var_names)) {
        interpreter->ret_expr.push_back(
            ExprTransformerUtils::InsertIfForAppendVarsTransformer()(expr));
      } else {
        interpreter->ret_expr.push_back(expr);
      }
    }
  }
}

std::vector<ir::Expr> FusionInterpreter::Run() {
  /*
   * instruction can't inplace change.
   */
  for (auto instr : tracker->instructions_) {
    VLOG(4) << "FusionInterpreter Start Run " << instr->DebugStr();
    switch (instr->type()) {
      case T_Copy:
        RunCopyInstr(dynamic_cast_instr_with_err<CopyInstr>(instr), this);
        break;
      case T_Combine:
        RunCombineInstr(dynamic_cast_instr_with_err<CombineInstr>(instr), this);
        break;
      case T_InitPattern:
        RunInitPatternInstr(
            dynamic_cast_instr_with_err<InitPatternInstr>(instr), this);
        break;
      case T_TrivialInline:
        RunTrivialInlineInstr(
            dynamic_cast_instr_with_err<TrivialInlineInstr>(instr), this);
        break;
      case T_TmpTransform:
        RunTmpTransformInstr(
            dynamic_cast_instr_with_err<TmpTransformInstr>(instr), this);
        break;
      case T_Padding:
        RunPaddingInstr(dynamic_cast_instr_with_err<PaddingInstr>(instr), this);
        break;
      case T_Return:
        RunReturnInstr(dynamic_cast_instr_with_err<ReturnInstr>(instr), this);
        break;
      case T_TrivialLoopAlign:
        RunTrivialLoopAlignInstr(
            dynamic_cast_instr_with_err<TrivialLoopAlignInstr>(instr), this);
        break;
      case T_ItersTransform:
        RunItersTransformInstr(
            dynamic_cast_instr_with_err<ItersTransformInstr>(instr), this);
        break;
      case T_ReshapeAlign:
        RunReshapeAlignInstr(
            dynamic_cast_instr_with_err<ReshapeAlignInstr>(instr), this);
        break;
      default:
        PADDLE_THROW(
            ::common::errors::Unavailable("Unsupported Fusion Instruction"));
    }
  }

  return TopoSort(ret_expr);
}

}  // namespace cinn::fusion
