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
#include "paddle/cinn/operator_fusion/fusion_tracker/expr_utils.h"
#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"

namespace cinn::fusion {

using namespace cinn::hlir::framework::pir::trivial_fusion_detail;  // NOLINT
using namespace ExprSetFinderUtils;                                 // NOLINT
using namespace ExprTransformerUtils;                               // NOLINT

ir::Expr ApplyItersTransform::operator()(const TransposeItersTransform& trans) {
  auto result = TransposeForsTransformer(trans.perm_)(expr_);
  VLOG(4) << "[ItersTransform] After TransposeItersTransform: " << result;
  return result;
}

ir::Expr ApplyItersTransform::operator()(const RemoveOnesTransform& trans) {
  VLOG(4) << "[ItersTransform] Before RemoveOnesTransform("
          << utils::Join(trans.ones_, ",") << "): " << expr_;
  auto result = RemoveOnesTransformer(trans.ones_)(expr_);
  VLOG(4) << "[ItersTransform] After  RemoveOnesTransform: " << result;
  return result;
}

ir::Expr ApplyItersTransform::operator()(const AppendItersTransform& trans) {
  VLOG(4) << "[ItersTransform] Start AppendItersTransform: "
          << trans.DebugStr();
  auto aligned_vars = GetAllLoopVars(aligned_expr_);
  PADDLE_ENFORCE_LT(trans.axis_.back(),
                    aligned_vars.size(),
                    ::common::errors::InvalidArgument(
                        "The last axis to append iters should be less than the "
                        "size of aligned_vars."));

  std::vector<ir::Var> append_vars;
  for (size_t i = 0; i < trans.axis_.size(); ++i) {
    const auto upper_bound = aligned_vars[trans.axis_[i]]->upper_bound;
    append_vars.push_back(ir::Var(upper_bound, trans.var_names_[i]));
  }
  auto result = InsertForsTransformer(trans.axis_, append_vars)(expr_);
  VLOG(4) << "[ItersTransform] After AppendItersTransform: " << result;
  return result;
}

std::vector<ir::Expr> GetFusibleOpsExpr(std::vector<FusibleOp> fusion_ops) {
  std::vector<ir::Expr> exprs;
  for (auto& fusion_op : fusion_ops) {
    auto expr = std::visit(FusibleOp2Expr(), fusion_op).front();
    exprs.push_back(expr);
  }
  return exprs;
}

// tmp transform for reduce_tree and reduce_tree_trivial.
std::vector<ir::Tensor> GetOutputTensors(const ir::Expr& op_expr) {
  const auto& tensors =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       ChildTensorStores)(op_expr);
  std::function<ir::Tensor(ir::Expr)> func = [](const ir::Expr& expr) {
    return expr.As<ir::Store>()->tensor.as_tensor_ref();
  };
  return MapVector(tensors, func);
}

std::vector<ir::Tensor> GetInputTensors(const ir::Expr& op_expr) {
  const auto& exprs =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       ChildTensorLoads)(op_expr);
  std::function<ir::Tensor(ir::Expr)> func = [](const ir::Expr& expr) {
    return expr.As<ir::Load>()->tensor.as_tensor_ref();
  };
  const auto& inputs = MapVector(exprs, func);
  const auto& outputs = GetOutputTensors(op_expr);
  return FilterVector(inputs, [&outputs](const ir::Tensor& tensor) {
    return std::find(outputs.begin(), outputs.end(), tensor) == outputs.end();
  });
}

std::vector<ir::Expr> TopoSort(const std::vector<ir::Expr>& op_exprs) {
  // Topo Sort is important for CINN GroupSchedule.
  std::map<ir::Tensor, std::vector<const ir::Expr*>> tensor2defining_op;
  std::map<ir::Tensor, std::vector<const ir::Expr*>> tensor2used_op;
  for (const auto& op : op_exprs) {
    auto inputs = GetInputTensors(op);
    auto outputs = GetOutputTensors(op);

    if (VLOG_IS_ON(5)) {
      VLOG(4) << "Ir::Expr is: \n" << op;
      VLOG(4) << "Inputs: ";
      for (const auto& input : inputs) {
        VLOG(4) << input->name;
      }
      VLOG(4) << "Outputs: ";
      for (const auto& output : outputs) {
        VLOG(4) << output->name;
      }
    }
    for (const auto& input : inputs) {
      tensor2used_op[input].push_back(&op);
    }
    for (const auto& output : outputs) {
      tensor2defining_op[output].push_back(&op);
    }
  }

  // Collect Downstreams
  std::map<const ir::Expr*, std::vector<const ir::Expr*>> op2downstreams;
  std::map<const ir::Expr*, int> degrees;
  for (const auto& op : op_exprs) {
    degrees[&op] = 0;
  }
  for (const auto& op : op_exprs) {
    auto outputs = GetOutputTensors(op);
    std::vector<const ir::Expr*> downstreams;
    for (const auto& output : outputs) {
      downstreams = ConcatVector(downstreams, tensor2used_op[output]);
    }
    for (const auto& downstream : downstreams) {
      degrees[downstream]++;
    }
    op2downstreams[&op] = downstreams;
  }

  // Topo Sort
  std::vector<const ir::Expr*> result;
  std::queue<const ir::Expr*> q;
  for (const auto& op : op_exprs) {
    if (degrees[&op] == 0) {
      q.push(&op);
    }
  }
  while (!q.empty()) {
    auto* cur = q.front();
    VLOG(4) << "Topo Sort Visit Order is:" << GetOutputTensors(*cur)[0]->name;
    q.pop();
    result.push_back(cur);
    for (const auto& downstream : op2downstreams[cur]) {
      degrees[downstream]--;
      if (degrees[downstream] == 0) {
        q.push(downstream);
      }
    }
  }
  PADDLE_ENFORCE_EQ(result.size(),
                    op_exprs.size(),
                    ::common::errors::PreconditionNotMet(
                        "[Error info] the size of result should be equal to "
                        "the size of op_exprs."));
  std::vector<ir::Expr> sorted_result;
  for (const auto& op : result) {
    sorted_result.push_back(*op);
  }
  return sorted_result;
}

static std::vector<ir::Var> GetAllForIters(const ir::Expr& expr) {
  const auto& all_father_fors =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       FindFather(expr) * IsFor)(expr);
  std::vector<ir::Var> vars;
  for (const auto& for_expr : all_father_fors) {
    vars.push_back(for_expr.As<ir::For>()->loop_var);
  }
  VLOG(4) << "GetAllForIters : " << expr
          << "\n var is : " << utils::Join(vars, ",");
  return vars;
}

static thread_local std::atomic<int> counter = 0;
ir::Expr UnSqueezeExpr(const ir::Expr& expr,
                       const std::vector<int>& padding_vec) {
  VLOG(4) << "UnSqueezeExpr: " << expr
          << "\npadding vector: " << utils::Join(padding_vec, ", ");
  const auto& vars_in_expr = AppendBound(GetAllForIters(expr), expr);
  // get the all vars.
  auto GenNextName = []() {
    counter += 1;
    return "expand_var_" + std::to_string(counter);
  };
  std::vector<ir::Var> vars;
  int pointer = 0;
  for (int i = 0; i < vars_in_expr.size() + padding_vec.size(); i++) {
    if (std::find(padding_vec.begin(), padding_vec.end(), i) !=
        padding_vec.end()) {
      vars.emplace_back(Expr(0), Expr(1), GenNextName());
    } else {
      vars.push_back(vars_in_expr[pointer++]);
    }
  }
  // update the is_reduce of expand_var.
  for (int i : padding_vec) {
    if (i == 0) {
      vars[i]->is_reduce_axis = false;
    } else {
      vars[i]->is_reduce_axis = vars[i - 1]->is_reduce_axis;
    }
  }

  // sequencely unsqueeze the ir::Expr.
  ir::Expr result = expr;
  for (int i : padding_vec) {
    if (i > 0) {
      result = UnsqueezeForTransformer((ChildFors * IsForIterVar(vars[i - 1])),
                                       vars[i])(result);
    } else {
      result = UnsqueezeForTransformer(ChildRootScheduleBlockRealizes,
                                       vars[i])(result);
    }
  }
  return result;
}

std::vector<FusibleOp> DoPadding(const FusibleOp& fusion_op,
                                 const std::vector<int>& padding_pos) {
  std::vector<FusibleOp> results;
  auto expr_vec = std::visit(FusibleOp2Expr(), fusion_op);
  for (auto expr : expr_vec) {
    auto squeezed = UnSqueezeExpr(expr, padding_pos);
    if (IsReduceBody(expr)) {
      results.emplace_back(ReduceOp(squeezed));
    } else {
      results.emplace_back(TrivialOp(squeezed));
    }
  }
  return results;
}

}  // namespace cinn::fusion
