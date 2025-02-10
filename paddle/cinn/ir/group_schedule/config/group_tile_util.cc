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

#include "paddle/cinn/ir/group_schedule/config/group_tile_util.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

namespace cinn {

using hlir::framework::pir::trivial_fusion_detail::GetAllForIters;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ChildScheduleBlockRealizes;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ChildTensorLoads;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ScheduleBlockRealizeIsNotInit;

namespace ir {
namespace {

struct VarReplacer : public ir::IRMutator<ir::Expr*> {
  std::unordered_set<ir::Var> iter_vars;
  ir::Var inspecting_var;

  explicit VarReplacer(const std::vector<ir::Var>& _iter_vars)
      : iter_vars(_iter_vars.begin(), _iter_vars.end()) {}

  virtual void Visit(const ir::_Var_* op, ir::Expr* expr) {
    ir::Var var = op->Copy().as_var_ref();
    if (inspecting_var.defined() && var == inspecting_var) {
      *expr = ir::Expr(1);
    } else if (iter_vars.find(var) != iter_vars.end()) {
      *expr = ir::Expr(0);
    } else {
      // We can replace shape variables (e.g. S0) with any constant, and here
      // we just choose to replace them with 32.
      *expr = ir::Expr(32);
    }
  }
};

std::vector<int64_t> GetVarStrides(ir::Expr load_offset,
                                   const std::vector<ir::Var>& iter_vars) {
  VarReplacer replacer(iter_vars);

  const auto Evaluate = [&](const ir::Var var) {
    ir::Expr expr = ir::ir_utils::IRCopy(load_offset);
    replacer.inspecting_var = var;
    replacer.IRMutator::Visit(&expr, &expr);
    ir::Expr res = optim::ArithSimplify(expr);
    if (res.is_constant()) {
      return res.as_int64();
    }
    return int64_t(0);
  };

  const int64_t base = Evaluate(ir::Var());

  std::vector<int64_t> strides;
  for (const auto& var : iter_vars) {
    int64_t stride = Evaluate(var) - base;
    strides.push_back(stride);
  }
  return strides;
}

ir::Expr GetLargestLoad(const std::vector<ir::Expr>& exprs) {
  common::cas_intervals_t var_intervals =
      common::CollectVarIntervalsOfExprs(exprs);
  common::SymbolicExprAnalyzer symbolic_expr_analyzer(var_intervals);

  const auto GetLoadSize = [](const ir::Expr& expr) {
    auto* load = expr.As<ir::Load>();
    auto* tensor = load->tensor.As<ir::_Tensor_>();
    if (tensor->shape.size() == 0) {
      return ir::Expr(1);
    }
    ir::Expr size = tensor->shape[0];
    for (size_t i = 1; i < tensor->shape.size(); i++) {
      size = size * tensor->shape[i];
    }
    return optim::ArithSimplify(size);
  };

  ir::Expr res = exprs[0];
  ir::Expr res_size = GetLoadSize(res);
  for (size_t i = 1; i < exprs.size(); i++) {
    ir::Expr cur_size = GetLoadSize(exprs[i]);
    std::optional<bool> gt = symbolic_expr_analyzer.ProveGT(cur_size, res_size);
    if (gt.has_value() && gt.value()) {
      res = exprs[i];
      res_size = cur_size;
    }
  }
  return res;
}

std::vector<ir::Expr> GetOpComputeBodyScheduleBlockRealizeExprSet(
    const ir::Expr& body) {
  using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ExprSetFinder;
  ExprSetFinder finder =
      ChildScheduleBlockRealizes * ExprSetFinder::GetIdentity();
  std::vector<ir::Expr> exprs = finder(body);
  return std::move(exprs);
}

bool ScheduleBlockRealizesShouldVectorizeCheck(
    const std::vector<ir::Expr>& exprs) {
  if (exprs.size() != 1) return false;

  ir::Expr expr_schedule_block_realize = exprs[0];
  bool is_reduce = ir::analyzer::IsReductionSBlock(expr_schedule_block_realize);
  if (is_reduce) return false;
  return true;
}

void CollectScheduleBlockRealizeLoadTensorsAndIndex(
    ir::Expr block,
    std::unordered_map<std::string, std::vector<std::vector<Expr>>>&
        load_tensor_and_indexes) {
  ir::ir_utils::CollectIRNodesWithoutTensor(
      block,
      [&](const ir::Expr* expr) {
        if (expr->As<ir::Load>()) {
          auto* node = expr->As<ir::Load>();
          PADDLE_ENFORCE_NOT_NULL(
              node,
              ::common::errors::InvalidArgument(
                  "Expected Load node, but received nullptr."));
          auto* tensor = node->tensor.As<ir::_Tensor_>();
          PADDLE_ENFORCE_NOT_NULL(
              tensor,
              ::common::errors::InvalidArgument(
                  "Expected _Tensor_ node in load, but received nullptr."));
          load_tensor_and_indexes[tensor->name].push_back(node->indices);
          return true;
        }
        return false;
      },
      /* uniq_target = */ false);
  return;
}

void CollectScheduleBlockRealizeStoreTensorsAndIndex(
    ir::Expr block,
    std::unordered_map<std::string, std::vector<std::vector<Expr>>>&
        store_tensor_and_indexes) {
  ir::ir_utils::CollectIRNodesWithoutTensor(
      block,
      [&](const ir::Expr* expr) {
        if (expr->As<ir::Store>()) {
          auto* node = expr->As<ir::Store>();
          PADDLE_ENFORCE_NOT_NULL(
              node,
              ::common::errors::InvalidArgument(
                  "Expected Load node, but received nullptr."));
          auto* tensor = node->tensor.As<ir::_Tensor_>();
          PADDLE_ENFORCE_NOT_NULL(
              tensor,
              ::common::errors::InvalidArgument(
                  "Expected _Tensor_ node in load, but received nullptr."));
          store_tensor_and_indexes[tensor->name].push_back(node->indices);
          return true;
        }
        return false;
      },
      /* uniq_target = */ false);
  return;
}

bool ScheduleBlockRealizeHasSpecialOp(
    Expr block, std::function<bool(const ir::Expr* e)>&& special_op_check) {
  bool has_special_op = false;
  ir::ir_utils::CollectIRNodesWithoutTensor(
      block,
      [&](const ir::Expr* expr) {
        if (special_op_check(expr)) {
          has_special_op = true;
          return true;
        }
        return false;
      },
      /* uniq_target = */ false);
  return has_special_op;
}

bool CheckTensorIsBroadcastAndContinuous(
    const std::vector<Expr>& indices,
    const std::vector<ir::Var>& for_iters,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value) {
  int loop_idx = 0;
  bool is_broadcast = false;
  for (int i = 0; i < indices.size(); ++i) {
    ir::Expr index = indices[i];
    cinn::optim::Simplify(&index);
    if (index.is_constant() && index.get_constant() == 0) {
      is_broadcast = true;
      continue;
    }

    if (!index.is_var()) return false;
    ir::Var iter_var = index.as_var_ref();
    if (!iter_var2value.count(iter_var)) {
      return false;
    }
    ir::Expr iter_value = iter_var2value.at(iter_var);

    if (!iter_value.as_var() && !iter_value.is_constant()) return false;

    for (; loop_idx < for_iters.size(); ++loop_idx) {
      if (for_iters[loop_idx] == iter_value.as_var_ref()) {
        break;
      }
    }

    if (loop_idx == for_iters.size()) {
      return false;
    }
  }

  if (is_broadcast || indices.size() < for_iters.size()) {
    return true;
  }
  return false;
}

bool CheckTensorIsContinuous(
    const std::vector<Expr>& indices,
    const std::vector<ir::Var>& for_iters,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value) {
  for (int i = 0; i < indices.size(); ++i) {
    ir::Expr index = indices[i];
    cinn::optim::Simplify(&index);
    if (index.is_constant()) return false;
    if (!index.is_var()) return false;
    ir::Var iter_var = index.as_var_ref();
    if (!iter_var2value.count(iter_var)) {
      return false;
    }
    ir::Expr iter_value = iter_var2value.at(iter_var);
    if (!iter_value.as_var() && !iter_value.is_constant()) return false;
    if (for_iters[i] != iter_value.as_var_ref()) {
      return false;
    }
  }
  return true;
}

bool TensorCanBeVectorized(
    const std::string& tensor_name,
    const std::vector<std::vector<Expr>>& indices,
    const std::vector<ir::Var>& for_iters,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value,
    std::unordered_set<std::string>* continue_tensors) {
  bool can_vectorize = true;
  for (auto indice : indices) {
    if (CheckTensorIsBroadcastAndContinuous(
            indice, for_iters, iter_var2value)) {
      continue;
    }

    if (CheckTensorIsContinuous(indice, for_iters, iter_var2value)) {
      continue_tensors->insert(tensor_name);
      continue;
    }
    can_vectorize = false;
    break;
  }

  return can_vectorize;
}

bool ScheduleBlockRealizeCanVectorize(
    const ir::Expr& expr_schedule_block_realize,
    const std::vector<ir::Var>& for_iters,
    std::unordered_set<std::string>* continue_tensors) {
  if (!expr_schedule_block_realize.As<ir::ScheduleBlockRealize>()) return false;
  std::vector<ir::Expr> iter_values =
      expr_schedule_block_realize.As<ir::ScheduleBlockRealize>()->iter_values;
  std::unordered_map<ir::Var, ir::Expr> iter_var2value =
      ir::analyzer::GetIterVarToValueOfSBlock(expr_schedule_block_realize);
  std::unordered_map<std::string, std::vector<std::vector<Expr>>>
      load_tensor_and_indexes;
  CollectScheduleBlockRealizeLoadTensorsAndIndex(expr_schedule_block_realize,
                                                 load_tensor_and_indexes);
  for (const auto& tensor : load_tensor_and_indexes) {
    if (TensorCanBeVectorized(tensor.first,
                              tensor.second,
                              for_iters,
                              iter_var2value,
                              continue_tensors)) {
      continue;
    }
    return false;
  }

  std::unordered_map<std::string, std::vector<std::vector<Expr>>>
      store_tensor_and_indexes;
  CollectScheduleBlockRealizeStoreTensorsAndIndex(expr_schedule_block_realize,
                                                  store_tensor_and_indexes);

  for (const auto& tensor : store_tensor_and_indexes) {
    if (TensorCanBeVectorized(tensor.first,
                              tensor.second,
                              for_iters,
                              iter_var2value,
                              continue_tensors)) {
      continue;
    }
    return false;
  }

  return true;
}

int CalculateContinueTensorSizeInGroupArgs(
    const std::unordered_set<std::string>& group_args,
    const std::unordered_set<std::string>& continuous_tensors) {
  int is_continuous_tensor_size = 0;
  for (auto tensor_name : continuous_tensors) {
    if (group_args.count(tensor_name)) {
      is_continuous_tensor_size++;
    }
  }

  return is_continuous_tensor_size;
}

}  // namespace

std::vector<int64_t> GetLoopStrides(const ir::Expr& body) {
  ir::Expr expr_block =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit)
          .GetSingle(body);
  auto* block = expr_block.As<ir::ScheduleBlockRealize>();
  auto& iter_values = block->iter_values;
  auto& iter_vars = block->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
  const std::vector<ir::Var> for_iters = GetAllForIters(body);

  const auto GetLoopIndex = [&](size_t var_index) {
    auto it = std::find(for_iters.begin(),
                        for_iters.end(),
                        iter_values[var_index].as_var_ref());
    PADDLE_ENFORCE_NE(it,
                      for_iters.end(),
                      ::common::errors::PreconditionNotMet(
                          "iter var %s was not found in loop vars: %s",
                          iter_values[var_index],
                          body));
    return std::distance(for_iters.begin(), it);
  };

  const auto& all_loads = ChildTensorLoads(expr_block);
  std::vector<int64_t> loop_strides(for_iters.size());
  if (all_loads.empty()) {
    return loop_strides;
  }
  const ir::Expr largest_load = GetLargestLoad(all_loads);
  ir::Expr load_offset = largest_load.As<ir::Load>()->index();
  std::vector<int64_t> var_strides = GetVarStrides(load_offset, iter_vars);
  for (size_t i = 0; i < iter_vars.size(); i++) {
    loop_strides[GetLoopIndex(i)] = var_strides[i];
  }
  return loop_strides;
}

bool GetCanApplyGridReduce(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<int64_t>& reduce_axis) {
  // Names of tensors that are downstream of reduce.
  // A tensor is downstream of reduce either if it is produced by a reduce, or
  // if it has data dependency on another tensor that is downstream of reduce.
  std::unordered_set<std::string> reduce_downstream_tensor_names;

  const auto IsReduceDownstream = [&](const ir::Expr& expr_block) {
    for (auto& expr_load : ChildTensorLoads(expr_block)) {
      std::string load_tensor_name = expr_load.As<ir::Load>()->name();
      if (reduce_downstream_tensor_names.count(load_tensor_name) > 0) {
        return true;
      }
    }
    return false;
  };

  const auto AddReduceDownstream = [&](const ir::Expr& expr_block) {
    auto expr_store = analyzer::GetStoreOfSBlock(expr_block);
    std::string store_tensor_name = expr_store.As<ir::Store>()->name();
    reduce_downstream_tensor_names.insert(store_tensor_name);
  };

  const auto CheckOutputHasReduceAxis = [&](const ir::Expr& body,
                                            const ir::Expr& expr_block) {
    std::vector<ir::Var> all_loop_vars = GetAllForIters(body);
    std::unordered_set<std::string> reduce_loop_vars;
    for (int64_t axis : reduce_axis) {
      reduce_loop_vars.insert(all_loop_vars[axis]->name);
    }

    std::unordered_set<std::string> reduce_iter_vars;
    auto* block = expr_block.As<ir::ScheduleBlockRealize>();
    auto& iter_vars = block->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
    for (int i = 0; i < iter_vars.size(); i++) {
      if (block->iter_values[i].is_var() &&
          reduce_loop_vars.count(block->iter_values[i].as_var()->name) > 0) {
        reduce_iter_vars.insert(iter_vars[i]->name);
      }
    }

    // The result is true if the indices of the output tensor contain any
    // reduce iter vars.
    auto expr_store = analyzer::GetStoreOfSBlock(expr_block);
    for (auto& index_expr : expr_store.As<ir::Store>()->indices) {
      if (index_expr.is_var() &&
          reduce_iter_vars.count(index_expr.as_var_ref()->name) > 0) {
        return true;
      }
    }
    return false;
  };

  for (const auto& body : op_compute_bodies) {
    ir::Expr expr_block =
        (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit)
            .GetSingle(body);
    bool is_reduce = analyzer::IsReductionSBlock(expr_block);
    bool is_reduce_downstream = IsReduceDownstream(expr_block);
    bool output_has_reduce_axis = CheckOutputHasReduceAxis(body, expr_block);

    if (is_reduce_downstream || is_reduce) {
      AddReduceDownstream(expr_block);
    }

    // When a block is downstream of reduce, its output shouldn't contain
    // reduce axis. Otherwise, it broadcasts the result of reduce. If this
    // is the case, we cannot apply grid reduce.
    if (is_reduce_downstream && output_has_reduce_axis) {
      VLOG(4) << "grid reduce is prohibited by block: " << expr_block;
      return false;
    }
  }
  return true;
}

GroupVectorizeInfo GetCanApplyVectorize(
    const std::vector<ir::Expr>& op_compute_bodies,
    const std::unordered_set<std::string>& group_args) {
  bool can_vectorize = true;
  bool has_if_else_op = false;
  bool has_select_op = false;
  std::unordered_set<std::string> continuous_tensors;

  for (const auto& body : op_compute_bodies) {
    std::vector<ir::Expr> blocks =
        GetOpComputeBodyScheduleBlockRealizeExprSet(body);

    if (!ScheduleBlockRealizesShouldVectorizeCheck(blocks)) continue;
    ir::Expr expr_schedule_block_realize = blocks[0];
    const std::vector<ir::Var> for_iters =
        hlir::framework::pir::trivial_fusion_detail::GetAllForIters(body);
    if (ScheduleBlockRealizeHasSpecialOp(
            expr_schedule_block_realize,
            [](const ir::Expr* e) { return e->As<ir::IfThenElse>(); })) {
      has_if_else_op = true;
    }

    if (ScheduleBlockRealizeHasSpecialOp(
            expr_schedule_block_realize,
            [](const ir::Expr* e) { return e->As<ir::Select>(); })) {
      has_select_op = true;
    }

    if (ScheduleBlockRealizeCanVectorize(
            expr_schedule_block_realize, for_iters, &continuous_tensors))
      continue;
    can_vectorize = false;
    break;
  }

  int continue_tensor_nums =
      CalculateContinueTensorSizeInGroupArgs(group_args, continuous_tensors);

  return {can_vectorize,
          has_if_else_op,
          has_select_op,
          continue_tensor_nums,
          group_args.size()};
}

}  // namespace ir
}  // namespace cinn
