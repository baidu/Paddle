// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/ir_simplify_pass.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

Tensor GetTensor(const Expr& block) {
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::NotFound(
          "Param block should be a ir::ScheduleBlockRealize node."));
  auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
      block, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  PADDLE_ENFORCE_EQ(find_tensor.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "One block should only have one Store node!(except for "
                        "root block)"));
  PADDLE_ENFORCE_NOT_NULL(
      (*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor(),
      ::common::errors::NotFound("Store node's tensor should be tensor."));
  Tensor tensor =
      (*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor_ref();
  return tensor;
}

Tensor GetReadTensor(const Expr& block, int index) {
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::NotFound(
          "Param block should be a ir::ScheduleBlockRealize node."));
  auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
      block, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  PADDLE_ENFORCE_EQ(find_tensor.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "One block should only have one Store node!(except for "
                        "root block)"));
  std::vector<Tensor> res;
  auto find_read_tensor =
      ir::ir_utils::CollectIRNodesWithoutTensor(block, [&](const Expr* x) {
        if (x->As<ir::Load>())
          res.push_back(x->As<ir::Load>()->tensor.as_tensor_ref());
        return x->As<ir::Load>();
      });
  PADDLE_ENFORCE_EQ(
      find_read_tensor.size(),
      res.size(),
      ::common::errors::InvalidArgument(
          "The number of Load nodes should be equal to the number "
          "of read tensors!"));
  PADDLE_ENFORCE_EQ(
      !find_read_tensor.empty(),
      true,
      ::common::errors::NotFound("Didn't find Load tensor in block!"));
  PADDLE_ENFORCE_LT(
      index,
      (int)find_read_tensor.size(),
      ::common::errors::InvalidArgument("Index is not < read tensor's size!"));
  return res[index];
}

int GetLoopExtent(const Expr& loop) {
  PADDLE_ENFORCE_NOT_NULL(loop.As<ir::For>(),
                          ::common::errors::NotFound(
                              "The input of GetLoopExtent should be ir::For!"));
  PADDLE_ENFORCE_EQ(
      cinn::common::is_zero(loop.As<ir::For>()->min),
      true,
      ::common::errors::InvalidArgument("For node's min should be zero."));
  PADDLE_ENFORCE_EQ(loop.As<ir::For>()->extent.is_constant(),
                    true,
                    ::common::errors::InvalidArgument(
                        "For node's extent should be constant."));
  return static_cast<int>(loop.As<ir::For>()->extent.get_constant());
}

int GetLoopExtent(const ir::stmt::For loop) {
  PADDLE_ENFORCE_EQ(
      cinn::common::is_zero(loop->min()),
      true,
      ::common::errors::InvalidArgument("For node's min should be zero."));
  PADDLE_ENFORCE_EQ(loop->extent().is_constant(),
                    true,
                    ::common::errors::InvalidArgument(
                        "For node's extent should be constant."));
  return static_cast<int>(loop->extent().get_constant());
}

void SetCudaAxisInfo(ir::LoweredFunc lowered_func) {
  auto CannotProveLT = [](const ir::Expr& lhs, const ir::Expr& rhs) -> bool {
    std::vector<ir::Expr> exprs{rhs, lhs};
    common::cas_intervals_t var_intervals =
        common::CollectVarIntervalsOfExprs(exprs);
    common::SymbolicExprAnalyzer analyzer{var_intervals};
    std::optional<bool> proved_lt = analyzer.ProveLT(lhs, rhs);
    return !proved_lt.has_value() || !proved_lt.value();
  };
  auto func_body = lowered_func->body;
  CudaAxisInfo info;
  ir::ir_utils::CollectIRNodes(func_body, [&](const Expr* x) {
    if (x->As<ir::For>() && x->As<ir::For>()->bind_info().valid()) {
      PADDLE_ENFORCE_EQ(
          cinn::common::is_zero(x->As<ir::For>()->min),
          true,
          ::common::errors::InvalidArgument("For node's min should be zero."));
      auto bind_info = x->As<ir::For>()->bind_info();
      info.set_valid(true);
      ir::Expr range = x->As<ir::For>()->extent;
      if (bind_info.for_type == ForType::GPUThread) {
        if (CannotProveLT(range, info.block_dim(bind_info.offset))) {
          VLOG(3) << "Set block dim[" << bind_info.offset << "] with range "
                  << range;
          info.set_block_dim(bind_info.offset, range);
        }
      } else if (bind_info.for_type == ForType::GPUBlock) {
        if (CannotProveLT(range, info.grid_dim(bind_info.offset))) {
          VLOG(3) << "Set grid dim[" << bind_info.offset << "] with range "
                  << range;
          info.set_grid_dim(bind_info.offset, range);
        }
      } else {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "The for loop's bind info should be gpu block or thread!"));
      }
    }
    return (x->As<ir::For>() && x->As<ir::For>()->bind_info().valid());
  });
  lowered_func->cuda_axis_info = info;
}

bool Contains(const Expr& container, const Expr& expr) {
  auto find_expr = ir::ir_utils::CollectIRNodesWithoutTensor(
      container,
      [&](const Expr* x) {
        return (x->node_type() == expr.node_type() && *x == expr);
      },
      true);
  return (!find_expr.empty());
}

Expr GetNextForLoop(const Expr& for_loop) {
  Expr result;
  PADDLE_ENFORCE_NOT_NULL(
      for_loop.As<ir::For>(),
      ::common::errors::NotFound("Param for_loop should be a ir::For node."));
  Expr for_body = for_loop.As<ir::For>()->body;
  ir::Block* for_body_block = for_body.As<ir::Block>();
  PADDLE_ENFORCE_NOT_NULL(
      for_body_block,
      ::common::errors::NotFound("The for_loop's body should be Block!"));

  // Only support for body block contains a sub for loop
  int next_idx = -1;
  for (int i = 0; i < for_body_block->stmts.size(); ++i) {
    Expr stmt = for_body_block->stmts[i];
    if (stmt.As<IfThenElse>() || stmt.As<ir::For>()) {
      if (next_idx == -1) {
        next_idx = i;
      } else {
        // More then one sub for loop, Return undefined.
        return result;
      }
    }
  }
  if (next_idx == -1) {
    // More then one sub for loop, Return undefined.
    return result;
  }

  Expr block_body = for_body_block->stmts[next_idx];
  if (block_body.As<IfThenElse>()) {
    // TODO(zhhsplendid): is it right to only handle true case?
    // It may be wrong, but the code is written by previous developer, for us,
    // we will check it later in the future.
    PADDLE_ENFORCE_NOT_NULL(
        block_body.As<IfThenElse>()->true_case.As<ir::Block>(),
        ::common::errors::NotFound(
            "IfThenElse node's true_case should be Block!"));
    Expr true_case = block_body.As<IfThenElse>()->true_case;
    if (true_case.As<ir::Block>()->stmts.size() != 1U ||
        !true_case.As<ir::Block>()->stmts[0].As<ir::For>())
      return result;
    result = true_case.As<ir::Block>()->stmts[0];
    return result;
  } else if (block_body.As<ir::For>()) {
    return block_body;
  } else {
    return result;
  }
}

std::vector<Expr> GetIfThenElseInRange(const Expr& top, const Expr& bottom) {
  std::vector<Expr> if_nodes;
  PADDLE_ENFORCE_NOT_NULL(
      top.As<ir::For>(),
      ::common::errors::NotFound(
          "Param top of GetIfThenElseInRange should be ir::For node."));
  PADDLE_ENFORCE_NOT_NULL(
      bottom.As<ir::For>(),
      ::common::errors::NotFound(
          "Param bottom of GetIfThenElseInRange should be ir::For node."));
  for (auto loop_iter = top; loop_iter != bottom;) {
    PADDLE_ENFORCE_NOT_NULL(
        loop_iter.As<ir::For>(),
        ::common::errors::NotFound("Param loop_iter should be ir::For node."));
    PADDLE_ENFORCE_NOT_NULL(
        loop_iter.As<ir::For>()->body.As<ir::Block>(),
        ::common::errors::NotFound("For node's body should be Block!"));
    auto block = loop_iter.As<ir::For>()->body.As<ir::Block>();
    for (Expr tmp : block->stmts) {
      if (tmp.As<IfThenElse>()) {
        if_nodes.push_back(tmp);
        PADDLE_ENFORCE_NOT_NULL(
            tmp.As<IfThenElse>()->true_case.As<ir::Block>(),
            ::common::errors::NotFound(
                "IfThenElse node's true_case should be Block!"));
        Expr true_case = tmp.As<IfThenElse>()->true_case;
        PADDLE_ENFORCE_EQ(true_case.As<ir::Block>()->stmts.size() == 1U &&
                              true_case.As<ir::Block>()->stmts[0].As<ir::For>(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Block node's stmts should be For! And the size "
                              "of stmts should be 1, but got %d.",
                              true_case.As<ir::Block>()->stmts.size()));
        tmp = true_case.As<ir::Block>()->stmts[0];
      }
      if (tmp.As<ir::For>()) {
        loop_iter = tmp;
      }
    }
  }
  return if_nodes;
}

void ReplaceExpr(Expr* source,
                 const std::vector<Var>& replaced,
                 const std::vector<Expr>& candidates) {
  PADDLE_ENFORCE_EQ(
      replaced.size(),
      candidates.size(),
      ::common::errors::InvalidArgument(
          "In ReplaceExpr, the size of Vars to be replaced must "
          "be equal to the size of candidate Exprs! Please check."));
  if (replaced.empty()) return;
  std::map<Var, Expr, CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
      continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  MappingVarToExprMutator mapper(replacing_map);
  mapper(source);
  return;
}

void ReplaceExpr(Expr* source,
                 const std::map<Var, Expr, CompVar>& replacing_map) {
  if (replacing_map.empty()) return;
  MappingVarToExprMutator mapper(replacing_map);
  mapper(source);
  return;
}

std::vector<int> ValidateFactors(const std::vector<int>& factors,
                                 int total_extent,
                                 const ModuleExpr& module_expr) {
  const std::string primitive = "split";
  PADDLE_ENFORCE_EQ(
      !factors.empty(),
      true,
      ::common::errors::InvalidArgument(
          "The factors param of Split should not be empty! Please check."));
  bool has_minus_one = false;
  int product = 1;
  int idx = -1;
  for (auto& i : factors) {
    idx++;
    PADDLE_ENFORCE_EQ(
        i != 0 && i >= -1, true, ::common::errors::InvalidArgument([&]() {
          std::ostringstream os;
          os << "[IRScheduleError] An Error occurred in the schedule primitive "
                "<"
             << primitive << ">.\n"
             << "[Error info] The params in factors of Split should be "
                "positive. However, the factor at position "
             << idx << " is " << i << ".\n"
             << "[Expr info] The Expr of current schedule is "
             << module_expr.GetExprs() << ".";
          return os.str();
        }()));
    PADDLE_ENFORCE_EQ(i == -1 && has_minus_one,
                      false,
                      ::common::errors::InvalidArgument([&]() {
                        std::ostringstream os;
                        os << "[IRScheduleError] An Error occurred in the "
                              "schedule primitive <"
                           << primitive << ">.\n"
                           << "[Error info] The params in factors of Split "
                              "should not be less than -1 or "
                           << "have more than one -1!\n"
                           << "[Expr info] The Expr of current schedule is "
                           << module_expr.GetExprs() << ".";
                        return os.str();
                      }()));
    if (i == -1) {
      has_minus_one = true;
    } else {
      product *= i;
    }
  }
  std::vector<int> validated_factors = factors;
  if (!has_minus_one) {
    PADDLE_ENFORCE_GE(
        product, total_extent, ::common::errors::PreconditionNotMet([&]() {
          std::ostringstream os;
          os << "[IRScheduleError] An Error occurred in the schedule primitive "
                "<"
             << primitive << ">.\n"
             << "[Error info] In Split, the factors' product[" << product
             << "] should be not larger than or equal "
                "to original loop's extent["
             << total_extent << "]!\n"
             << "[Expr info] The Expr of current schedule is "
             << module_expr.GetExprs() << ".";
          return os.str();
        }()));
    return validated_factors;
  } else {
    int minus_one_candidate = static_cast<int>(
        ceil(static_cast<double>(total_extent) / static_cast<double>(product)));
    for (int i = 0; i < validated_factors.size(); ++i) {
      if (validated_factors[i] == -1) {
        validated_factors[i] = minus_one_candidate;
      }
    }
    return validated_factors;
  }
}

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis) {
  auto* rf_for = rf_loop.As<ir::For>();
  PADDLE_ENFORCE_NOT_NULL(
      rf_for,
      ::common::errors::NotFound(
          "Expr param of Rfactor must be For node! Please check."));
  // check the rf_loop only has one schedule block
  auto block_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
      rf_loop,
      [&](const Expr* x) { return x->As<ScheduleBlockRealize>(); },
      true);
  PADDLE_ENFORCE_EQ(block_nodes.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "Rfactor Loop should only have one schedule block!"));
  auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
      rf_loop, [&](const Expr* x) { return x->As<Store>(); }, true);
  PADDLE_ENFORCE_EQ(find_store.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "Rfactor Loop should only have one Store node!"));
  auto indice = find_store.begin()->As<Store>()->indices;
  // check rf_axis
  PADDLE_ENFORCE_LE(
      rf_axis,
      indice.size(),
      ::common::errors::InvalidArgument(
          "rf_axis should not be greater than store's domain size"));
  // check rfactor loop is reduce
  auto* sch_block_realize = block_nodes.begin()->As<ScheduleBlockRealize>();
  auto* sch_block = sch_block_realize->schedule_block.As<ScheduleBlock>();
  PADDLE_ENFORCE_NOT_NULL(
      sch_block,
      ::common::errors::NotFound("ScheduleBlockRealize node's schedule_block "
                                 "should be ScheduleBlock."));
  auto& iter_values = sch_block_realize->iter_values;
  auto& iter_vars = sch_block->iter_vars;
  PADDLE_ENFORCE_EQ(iter_values.size(),
                    iter_vars.size(),
                    ::common::errors::InvalidArgument(
                        "iter_values size should be equal to iter_vars size"));
  auto rf_loop_var = rf_for->loop_var;
  Var rf_block_var;
  for (int i = 0; i < iter_values.size(); ++i) {
    if (ContainVar({iter_values[i]}, rf_loop_var->name)) {
      PADDLE_ENFORCE_EQ(
          !rf_block_var.defined(),
          true,
          ::common::errors::InvalidArgument(
              "The rfactor loop var can only be binded to one block var."));
      auto iter_value = iter_values[i].As<_Var_>();
      PADDLE_ENFORCE_NOT_NULL(
          iter_value,
          ::common::errors::NotFound(
              "The iter value don't support complex reduce bindings."));
      rf_block_var = iter_vars[i];
      auto it = std::find_if(indice.begin(), indice.end(), [&](const Expr& x) {
        return x.As<_Var_>() && x.As<_Var_>()->name == rf_block_var->name;
      });
      PADDLE_ENFORCE_EQ(
          it == indice.end(),
          true,
          ::common::errors::InvalidArgument(
              "Param rfactor loop var is not reduce, please check!"));
    }
  }
}

std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root) {
  auto loop_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) { return x->As<ir::For>() && Contains(*x, expr); });
  std::vector<Expr> result(loop_nodes.begin(), loop_nodes.end());
  if (result.empty()) {
    std::stringstream ss;
    ss << "Didn't find expr's : \n" << expr << "\n loops in root : \n" << root;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });
  return result;
}

IterRange GetAccessedRange(const Expr& index,
                           const std::vector<Var>& iter_vars,
                           const std::vector<IterRange>& iter_ranges) {
  PADDLE_ENFORCE_EQ(iter_vars.size(),
                    iter_ranges.size(),
                    ::common::errors::InvalidArgument(
                        "The size of iter_vars should be equal to the size of "
                        "iter_ranges! Please check."));
  std::vector<Expr> var_mins, var_maxs;
  for (const auto& range : iter_ranges) {
    var_mins.emplace_back(range.min);
    var_maxs.emplace_back(range.min + range.extent - 1);
  }

  Expr indice_min = ir::ir_utils::IRCopy(index);
  Expr indice_max = ir::ir_utils::IRCopy(index);
  // replace the var by the corresponding iter_value
  ReplaceExpr(&indice_min, iter_vars, var_mins);
  ReplaceExpr(&indice_max, iter_vars, var_maxs);
  // simplify expression
  indice_min = optim::ArithSimplify(indice_min);
  indice_max = optim::ArithSimplify(indice_max);

  Expr indice_extent;
  Expr mod_extent(0);
  if (indice_min.As<Mod>() && indice_min.As<Mod>()->b().is_constant()) {
    Expr mod_right_min = indice_min.As<Mod>()->a();
    Expr mod_right_max = indice_max.As<Mod>()->a();
    Expr mod_right_extent =
        optim::ArithSimplify(mod_right_max - mod_right_min + 1);
    mod_extent = indice_min.As<Mod>()->b();
    if (mod_right_extent.get_constant() < mod_extent.get_constant()) {
      mod_extent = mod_right_extent;
    }
  }

  if (indice_min == indice_max) {
    if (cinn::common::is_zero(mod_extent)) {
      // If a index keeps constant, its extent should be 1.
      indice_extent = Expr(1);
    } else {
      indice_extent = mod_extent;
    }
  } else {
    indice_extent = optim::ArithSimplify(optim::ArithSimplify(indice_max) -
                                         optim::ArithSimplify(indice_min) + 1);
  }

  if (indice_extent.is_constant() && indice_extent.get_constant() < 0) {
    VLOG(3) << "deduced indices are not constant";
    indice_min = indice_max;
    indice_extent = Expr(-indice_extent.get_constant());
  }
  VLOG(3) << "indice_min=" << indice_min << ", indice_max=" << indice_max
          << ", indice_extent=" << indice_extent;
  return IterRange(indice_min, indice_extent);
}

std::vector<IterRange> CalculateTensorRegions(
    const Expr& block,
    const std::vector<Expr>& tensor_indices,
    const Tensor& tensor,
    const Expr& root) {
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ScheduleBlockRealize>(),
      ::common::errors::NotFound("Param block of CalculateTensorRegions "
                                 "should be ScheduleBlockRealize node!"));
  auto iter_vars = block.As<ir::ScheduleBlockRealize>()
                       ->schedule_block.As<ir::ScheduleBlock>()
                       ->iter_vars;
  auto iter_values = block.As<ir::ScheduleBlockRealize>()->iter_values;

  std::vector<Var> loop_vars;
  std::vector<IterRange> loop_ranges;

  auto outer_loops = GetLoopsOfExpr(block, root);
  for (auto& loop : outer_loops) {
    PADDLE_ENFORCE_NOT_NULL(
        loop.As<For>(),
        ::common::errors::NotFound("Param loop should be For node."));
    loop_vars.emplace_back(loop.As<For>()->loop_var);
    loop_ranges.emplace_back(
        IterRange(loop.As<For>()->min, loop.As<For>()->extent));
  }

  std::vector<IterRange> result;
  for (int i = 0; i < tensor_indices.size(); ++i) {
    Expr binded_index = ir::ir_utils::IRCopy(tensor_indices[i]);
    ReplaceExpr(&binded_index, iter_vars, iter_values);
    auto range = GetAccessedRange(binded_index, loop_vars, loop_ranges);

    // in generally, the range should be constant, but in some cases our
    // AutoSimplify (algebraic simplification function) can't simplify
    // completely where we use the whole shape in this indice as the accessed
    // range conservatively
    if (!range.min.is_constant() || !range.extent.is_constant()) {
      VLOG(3) << "deduced range is not constant, range.min=" << range.min
              << ", range.extent=" << range.extent;
      if (tensor->buffer.defined()) {
        PADDLE_ENFORCE_GT((int)tensor->buffer->shape.size(),
                          i,
                          ::common::errors::InvalidArgument(
                              "The size of tensor's shape should be greater "
                              "than or equal to the size of tensor_indices!"));
        result.emplace_back(IterRange(Expr(0), tensor->buffer->shape[i]));
      } else {
        PADDLE_ENFORCE_GT((int)tensor->shape.size(),
                          i,
                          ::common::errors::InvalidArgument(
                              "The size of tensor's shape should be greater "
                              "than or equal to the size of tensor_indices!"));
        result.emplace_back(IterRange(Expr(0), tensor->shape[i]));
      }
    } else {
      result.emplace_back(std::move(range));
    }
  }

  return result;
}

Expr GetNthAccessExpr(const Expr& block, int index, bool is_write) {
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ScheduleBlockRealize>(),
      ::common::errors::NotFound("Param block of GetNthAccessExpr should be "
                                 "ScheduleBlockRealize node."));
  auto compute_body = block.As<ScheduleBlockRealize>()
                          ->schedule_block.As<ScheduleBlock>()
                          ->body;
  if (is_write) {
    std::vector<Expr> find_store_vec;
    auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
        compute_body, [&](const Expr* x) {
          if (x->As<ir::Store>()) find_store_vec.push_back(*x);
          return x->As<ir::Store>();
        });
    PADDLE_ENFORCE_EQ(find_store.size(),
                      find_store_vec.size(),
                      ::common::errors::InvalidArgument(
                          "The number of Store nodes should be equal to the "
                          "number of find_store_vec!"));
    PADDLE_ENFORCE_LT(
        index,
        (int)find_store.size(),
        ::common::errors::InvalidArgument("Index is not < store's size!"));
    Expr store_index = find_store_vec[index];
    return store_index;
  } else {
    std::vector<Expr> find_load_vec;
    auto find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
        compute_body, [&](const Expr* x) {
          if (x->As<ir::Load>()) find_load_vec.push_back(*x);
          return x->As<ir::Load>();
        });
    PADDLE_ENFORCE_EQ(find_load.size(),
                      find_load_vec.size(),
                      ::common::errors::InvalidArgument(
                          "The number of Load nodes should be equal to the "
                          "number of find_load_vec!"));
    PADDLE_ENFORCE_LT(
        index,
        (int)find_load.size(),
        ::common::errors::InvalidArgument("Index is not < load's size!"));
    Expr load_index = find_load_vec[index];
    return load_index;
  }
}

Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type) {
  auto cache_tensor = lang::Compute(
      tensor->shape,
      [=](const std::vector<Expr>& dims) { return tensor(dims); },
      tensor->name + "_" + memory_type + "_temp_buffer");
  cache_tensor->WithBuffer(memory_type);
  return cache_tensor;
}

Expr MakeCacheBlock(const std::vector<IterRange>& buffer_ranges,
                    CacheBlockInfo* info,
                    const std::string& memory_type,
                    DeviceAPI device_api) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<Expr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const auto& range : buffer_ranges) {
    Var loop_var(
        cinn::common::UniqName("cache_ax" + std::to_string(loop_vars.size())));
    // Var loop_var("ax" + std::to_string(loop_vars.size()));
    loop_vars.push_back(loop_var);
    iter_values.push_back(optim::ArithSimplify(range.min + loop_var));
  }
  // block variables
  std::vector<Var> block_vars;
  Tensor new_tensor = info->alloc;
  // Create block vars, block's accessed region and accessing indices
  PADDLE_ENFORCE_EQ(new_tensor->buffer.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The new tensor's buffer should be defined!"));
  for (auto& dim : new_tensor->buffer->shape) {
    Var var(Expr(0), dim, "v" + std::to_string(block_vars.size()), false);
    block_vars.push_back(var);
  }
  auto body = new_tensor->tensor_store_expanded_body();
  std::vector<Var> axis_vars =
      cinn::common::GenDefaultAxis(new_tensor->domain.size());
  axis_vars.insert(axis_vars.end(),
                   new_tensor->reduce_axis.begin(),
                   new_tensor->reduce_axis.end());
  for (int i = 0; i < axis_vars.size(); ++i) {
    optim::ReplaceVarWithExpr(&body, axis_vars[i], block_vars[i]);
  }
  Expr block = ir::ScheduleBlockRealize::Make(
      iter_values,
      ir::ScheduleBlock::Make(
          block_vars, {}, {}, new_tensor->name, Block::Make({body})));
  Expr new_body = block;
  for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; i--) {
    new_body = For::Make(loop_vars[i],
                         Expr(0),
                         optim::ArithSimplify(buffer_ranges[i].extent),
                         ir::ForType::Serial,
                         device_api,
                         ir::Block::Make({new_body}));
  }
  info->cache_block = std::move(new_body);
  return block;
}

void FindInsertionPoint(const Expr& root, CacheBlockInfo* info, bool is_write) {
  Expr find_tensor =
      is_write ? Expr(info->write_tensor) : Expr(info->read_tensor);
  auto find_produce_read =
      ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
        return x->As<ir::Store>() && x->As<ir::Store>()->tensor == find_tensor;
      });

  if (find_produce_read.empty()) {
    PADDLE_ENFORCE_NOT_NULL(
        root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>(),
        ::common::errors::NotFound(
            "ScheduleBlockRealize node's schedule_block should be "
            "ScheduleBlock!"));
    PADDLE_ENFORCE_NOT_NULL(root.As<ScheduleBlockRealize>()
                                ->schedule_block.As<ScheduleBlock>()
                                ->body.As<Block>(),
                            ::common::errors::NotFound(
                                "ScheduleBlock node's body should be Block!"));
    info->loc_block = root.As<ScheduleBlockRealize>()
                          ->schedule_block.As<ScheduleBlock>()
                          ->body;
    info->loc_pos = 0;
    return;
  }

  PADDLE_ENFORCE_EQ(find_produce_read.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Store nodes should be equal to 1!"));
  Expr producer = *(find_produce_read.begin());

  PADDLE_ENFORCE_NOT_NULL(
      root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>(),
      ::common::errors::NotFound("ScheduleBlockRealize node's schedule_block "
                                 "should be ScheduleBlock!"));
  PADDLE_ENFORCE_NOT_NULL(
      root.As<ScheduleBlockRealize>()
          ->schedule_block.As<ScheduleBlock>()
          ->body.As<Block>(),
      ::common::errors::NotFound("ScheduleBlock node's body should be Block!"));
  info->loc_block =
      root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body;
  for (int i = 0;
       i < static_cast<int>(info->loc_block.As<Block>()->stmts.size());
       ++i) {
    if (Contains(info->loc_block.As<Block>()->stmts[i], producer)) {
      info->loc_pos = i + 1;
      break;
    }
  }
}

const std::set<Expr, CompExpr> CollectLoopsToSet(
    const std::vector<Expr>& loops) {
  std::set<Expr, CompExpr> for_loops;
  for (auto& i : loops) {
    PADDLE_ENFORCE_NOT_NULL(
        i.As<ir::For>(),
        ::common::errors::NotFound(
            "Param loops should be For node! Please check."));
    auto inserted = for_loops.insert(i);
    if (!inserted.second) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "There should be no duplicate elements in loops! Please check."));
    }
  }
  return for_loops;
}

// This function is used in Reorder schedule primitive. Since input loop
// Expr(s) of Reorder doesn't give original for loop order, we have to
// find the top (most outer) loop and bottom (most inner) among loop Expr(s)
std::pair<Expr, Expr> GetBoundaryOfReorderRange(
    const std::set<Expr, CompExpr>& loop_set) {
  Expr top = *loop_set.begin();
  Expr bottom;
  std::set<Expr, CompExpr> visited;
  bool first_traversal = true;
  for (Expr loop_i : loop_set) {
    if (visited.count(loop_i)) {
      continue;
    }
    Expr v_for = loop_i;
    PADDLE_ENFORCE_NOT_NULL(
        v_for.As<ir::For>(),
        ::common::errors::NotFound(
            "Param v_for should be a ir::For node! Please check."));
    while (v_for.defined()) {
      // If loop_i's sub loop is visited it must be pre-visited top.
      // Then loop_i should be the new top
      if (visited.count(v_for)) {
        if (v_for != top) {
          PADDLE_THROW(::common::errors::InvalidArgument(
              "Loops in GetBoundaryOfReorderRange is not a chain! "
              "Please check."));
        }
        top = loop_i;
        break;
      }

      // This while loop always GetNextForLoop(sub loop), so the last
      // visited v_for in the first traversal will be the bottom.
      if (first_traversal && loop_set.count(v_for)) {
        bottom = v_for;
      }
      visited.insert(v_for);
      v_for = GetNextForLoop(v_for);
    }
    first_traversal = false;
  }
  PADDLE_ENFORCE_NOT_NULL(
      top.As<ir::For>(),
      ::common::errors::NotFound("Param top should be a ir::For node."));
  PADDLE_ENFORCE_EQ(bottom.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Param bottom should be defined! Please check."));
  PADDLE_ENFORCE_NOT_NULL(
      bottom.As<ir::For>(),
      ::common::errors::NotFound("Param bottom should be a ir::For node."));
  return std::make_pair(top, bottom);
}

std::vector<Expr> GetLoopsInRange(const Expr& top, const Expr& bottom) {
  std::vector<Expr> chain;
  PADDLE_ENFORCE_NOT_NULL(
      top.As<ir::For>(),
      ::common::errors::NotFound(
          "Param top of GetLoopsInRange should be a ir::For node."));
  PADDLE_ENFORCE_NOT_NULL(
      bottom.As<ir::For>(),
      ::common::errors::NotFound(
          "Param bottom of GetLoopsInRange should be a ir::For node."));
  for (auto loop_iter = top; loop_iter != bottom;) {
    Expr tmp = GetNextForLoop(loop_iter);
    if (!tmp.defined())
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Loops in GetLoopsInReorderRange is not a chain! Please check."));
    chain.push_back(loop_iter);
    loop_iter = tmp;
  }
  chain.push_back(bottom);
  return chain;
}

// Construct a loop chain such that:
//
//   loops[i_1] {
//     loops[i_2] {
//       ...
//        loops[i_n] {
//          stmts;
//        }
//     }
//   }
//
// where reordered_indices = {i_1, i_2, ... i_n }
//
// This is a helper function which constructs non-main chain for other body
// statements in Reorder. See comment and call place in ConstructNewLoopChain
Expr ConstructOtherStmtChain(const std::vector<Expr>& stmts,
                             const std::vector<Expr>& loops,
                             const std::vector<int> reordered_indices) {
  Expr new_loop;
  for (int i = reordered_indices.size() - 1; i >= 0; --i) {
    Expr temp = ir::ir_utils::IRCopy(loops[reordered_indices[i]]);
    PADDLE_ENFORCE_EQ(temp.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Param temp should be defined! Please check."));
    PADDLE_ENFORCE_NOT_NULL(
        temp.As<ir::For>(),
        ::common::errors::NotFound("Param temp should be a ir::For node."));
    if (new_loop.defined()) {
      temp.As<ir::For>()->body = Block::Make({new_loop});
    } else {
      temp.As<ir::For>()->body = Block::Make({stmts});
    }
    new_loop = temp;
  }
  return new_loop;
}

Expr ConstructNewLoopChain(const std::vector<Expr>& chain,
                           const std::vector<Expr>& ordered_loops,
                           const std::set<Expr, CompExpr>& loop_set,
                           std::vector<Expr>& if_nodes) {  // NOLINT
  std::vector<std::set<std::string>> condition_vars;
  // In each IfThenElse node, find the vars its condition depends on.
  for (auto& if_expr : if_nodes) {
    PADDLE_ENFORCE_NOT_NULL(
        if_expr.As<IfThenElse>(),
        ::common::errors::NotFound(
            "Param if_nodes should be IfThenElse node! Please check."));
    auto var_set = ir::ir_utils::CollectIRNodes(
        if_expr.As<IfThenElse>()->condition,
        [&](const Expr* x) { return x->as_var(); });
    std::set<std::string> var_name_set;
    for (auto& i : var_set) var_name_set.insert(i.as_var()->name);
    condition_vars.push_back(var_name_set);
  }
  Expr new_loop;
  int index = static_cast<int>(ordered_loops.size()) - 1;

  std::vector<Expr> reordered_loop_chain;
  // Construct the main loop chain from bottom to top.
  for (int i = static_cast<int>(chain.size()) - 1; i >= 0; i--) {
    auto& loop_in_chain = chain[i];
    PADDLE_ENFORCE_NOT_NULL(
        loop_in_chain.As<ir::For>(),
        ::common::errors::NotFound(
            "Param loop_in_chain should be ir::For node! Please check."));
    Expr temp;
    if (loop_set.count(loop_in_chain)) {
      PADDLE_ENFORCE_GE(index,
                        0,
                        ::common::errors::InvalidArgument(
                            "The index should be greater than or equal to 0!"));
      temp = ir::ir_utils::IRCopy(ordered_loops[index]);
      --index;
    } else {
      temp = ir::ir_utils::IRCopy(loop_in_chain);
    }
    PADDLE_ENFORCE_EQ(temp.defined(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Param temp should be defined! Please check."));
    PADDLE_ENFORCE_NOT_NULL(
        temp.As<ir::For>(),
        ::common::errors::NotFound(
            "Param temp should be ir::For node! Please check."));
    // Main chain, each loop's body only contains sub_loop or bottom loop's body
    if (new_loop.defined()) {
      temp.As<ir::For>()->body = Block::Make({new_loop});
    } else {
      temp.As<ir::For>()->body = loop_in_chain.As<ir::For>()->body;
    }
    Expr original_temp = temp;
    // Here we handle the IfThenElse nodes.
    for (int i = 0; i < static_cast<int>(if_nodes.size()); ++i) {
      if (condition_vars[i].count(
              original_temp.As<ir::For>()->loop_var->name)) {
        Expr temp_body = temp.As<ir::For>()->body;
        if (temp_body.As<Block>() && temp_body.As<Block>()->stmts.size() == 1U)
          temp_body = temp_body.As<Block>()->stmts[0];
        temp.As<ir::For>()->body =
            IfThenElse::Make(if_nodes[i].As<IfThenElse>()->condition,
                             temp_body,
                             if_nodes[i].As<IfThenElse>()->false_case);
        temp.As<ir::For>()->body = Block::Make({temp.As<ir::For>()->body});
        if_nodes.erase(if_nodes.begin() + i);
        condition_vars.erase(condition_vars.begin() + i);
        i--;
      }
    }
    new_loop = temp;
    reordered_loop_chain.push_back(new_loop);
  }
  PADDLE_ENFORCE_EQ(new_loop.defined(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Param new loop should be defined! Please check."));

  // new_loop_chain, which represents the main loop chain, now is from top to
  // bottom.
  std::reverse(reordered_loop_chain.begin(), reordered_loop_chain.end());

  // In the main loop chain, each loop's body only contains sub_loop or bottom
  // loop's body, but the origin loop chain may contain some other body stmts.
  // The main loop chain lost those other body stmts.
  // For example:
  //
  // for (i, 0, 32) {         Reorder j, i         for (j, 0, 64) {
  //   other_body_stmts       above main chine
  //   for (j, 0, 64) {      ------------------>     for (i, 0, 32) {
  //     bottom_loop_body                              bottom_loop_body
  //   }                                             }
  // }                                             }
  //
  // We go through origin loop and check other body stmts, adding it as another
  // chain, such as:
  //
  // for (i, 0, 32) {
  //   other_body_stmts
  // }
  // for (j, 0, 64) {
  //   for (i, 0, 32) {
  //     bottom_loop_body
  //   }
  // }
  //

  // Construct the complete loop chain from origin loop top to bottom.
  PADDLE_ENFORCE_EQ(
      chain.size(),
      reordered_loop_chain.size(),
      ::common::errors::InvalidArgument(
          "origin loop chain size not equals reordered requirement "
          "when ConstructNewLoopChain in Reorder"));
  std::unordered_set<std::string> origin_loop_var_names;
  Expr ret = new_loop;

  // Maintain an index to add stmt (other body stmt chain)
  //
  //  stmt  stmt  MainChainLoop  stmt   stmt
  //               index        index+1
  //
  // The index of this MainChainLoop points the place before next MainChainLoop
  // We can insert statements before MainChainLoop at the index, and insert
  // statements after MainChainLoop at the index + 1
  int add_other_chain_index = 0;

  for (int i = 0; i < chain.size() - 1; ++i) {
    // we just check i < chain.size() - 1
    // because bottom loop's body stmts have been all added

    const ir::For* loop_in_chain = chain[i].As<ir::For>();
    ir::For* reordered_in_chain = reordered_loop_chain[i].As<ir::For>();

    origin_loop_var_names.insert(loop_in_chain->loop_var->name);
    PADDLE_ENFORCE_EQ(
        origin_loop_var_names.size(),
        i + 1,
        ::common::errors::InvalidArgument(
            "Duplicate loop var name in origin Chain during Reorder"));

    const ir::Block* body_block = loop_in_chain->body.As<ir::Block>();

    if (body_block != nullptr && body_block->stmts.size() > 1) {
      // contains other body stmts

      // Get the other body statements before loop and after loop
      bool other_stmt_body_before_loop = true;
      std::vector<Expr> stmts_before_loop;
      std::vector<Expr> stmts_after_loop;
      for (int j = 0; j < body_block->stmts.size(); ++j) {
        if (body_block->stmts[j].As<ir::For>() &&
            body_block->stmts[j].As<ir::For>()->loop_var->name ==
                chain[i + 1].As<ir::For>()->loop_var->name) {
          other_stmt_body_before_loop = false;
          continue;
        }
        if (other_stmt_body_before_loop) {
          stmts_before_loop.push_back(body_block->stmts[j]);
        } else {
          stmts_after_loop.push_back(body_block->stmts[j]);
        }
      }

      // Find the chain that other body stmts shares with main loop chain
      std::vector<int> reordered_indices;
      for (int j = 0; j < reordered_loop_chain.size(); ++j) {
        if (origin_loop_var_names.count(
                reordered_loop_chain[j].As<ir::For>()->loop_var->name)) {
          reordered_indices.push_back(j);
        }
      }
      PADDLE_ENFORCE_EQ(reordered_indices.size(),
                        origin_loop_var_names.size(),
                        ::common::errors::InvalidArgument(
                            "Reordered chain loop var names doesn't match "
                            "other stmt chain loop var names"));
      // Add other stmts chain to root Block if other stmts exist
      if (!stmts_before_loop.empty()) {
        Expr before_chain = ConstructOtherStmtChain(
            stmts_before_loop, reordered_loop_chain, reordered_indices);
        if (ret.As<ir::Block>() == nullptr) {
          ret = ir::Block::Make({ret});
        }
        std::vector<Expr>& inplace_stmts = ret.As<ir::Block>()->stmts;
        auto pos = inplace_stmts.begin() + add_other_chain_index;
        inplace_stmts.insert(pos, before_chain);
        ++add_other_chain_index;
      }

      if (!stmts_after_loop.empty()) {
        Expr after_chain = ConstructOtherStmtChain(
            stmts_after_loop, reordered_loop_chain, reordered_indices);
        if (ret.As<ir::Block>() == nullptr) {
          ret = ir::Block::Make({ret});
        }
        std::vector<Expr>& inplace_stmts = ret.As<ir::Block>()->stmts;
        auto pos = inplace_stmts.begin() + add_other_chain_index + 1;
        inplace_stmts.insert(pos, after_chain);
      }
    }
  }

  return ret;
}

std::vector<Expr> GetProducers(const Expr& block, const Expr& root) {
  PADDLE_ENFORCE_NOT_NULL(block.As<ir::ScheduleBlockRealize>(),
                          ::common::errors::NotFound(
                              "Param block of GetProducers should be "
                              "ir::ScheduleBlockRealize node! Please check."));
  PADDLE_ENFORCE_NOT_NULL(root.As<ir::ScheduleBlockRealize>(),
                          ::common::errors::NotFound(
                              "Param root of GetProducers should be "
                              "ir::ScheduleBlockRealize node! Please check."));
  std::vector<Expr> producers;

  // collect all producers' tensor names
  std::set<std::string> producer_tensor_names;
  auto compute_body = block.As<ir::ScheduleBlockRealize>()
                          ->schedule_block.As<ir::ScheduleBlock>()
                          ->body;
  std::string block_name = block.As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;
  ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&producer_tensor_names, &block_name](const Expr* x) {
        const ir::Load* load = x->As<ir::Load>();
        if (load) {
          producer_tensor_names.insert(load->tensor.as_tensor()->name);
          if (load->tensor.as_tensor()->name == block_name) {
            producer_tensor_names.insert(
                GenReduceInitTensorNameOf(load->tensor.as_tensor()->name));
          }
          return true;
        }
        const ir::Store* store = x->As<ir::Store>();
        if (store) {
          std::set<ir::Expr> call_nodes =
              ir::ir_utils::CollectIRNodesWithoutTensor(
                  store->value,
                  [](const ir::Expr* x) { return x->As<ir::Call>(); });
          for (ir::Expr call : call_nodes) {
            const std::vector<ir::Expr>& read_args =
                call.As<ir::Call>()->read_args;
            for (const ir::Expr& arg : read_args) {
              if (arg.as_tensor()) {
                producer_tensor_names.insert(arg.as_tensor_ref()->name);
              }
            }
          }
        }
        return false;
      });

  // traverse each of other blocks and filter those ones which contain at least
  // one producer tensor;
  auto find_blocks = ir::ir_utils::CollectIRNodesWithoutTensor(
      root, [&block, &root](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() && *x != block && *x != root;
      });
  for (auto&& cur : find_blocks) {
    auto* cur_block = cur.As<ir::ScheduleBlockRealize>()
                          ->schedule_block.As<ir::ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        cur_block,
        ::common::errors::NotFound(
            "Param block result should be a ScheduleBlockRealize node."));
    auto find_stores = ir::ir_utils::CollectIRNodesWithoutTensor(
        cur_block->body, [&producer_tensor_names](const Expr* x) {
          return x->As<ir::Store>() &&
                 producer_tensor_names.count(
                     x->As<ir::Store>()->tensor.as_tensor()->name) > 0;
        });
    if (!find_stores.empty()) producers.emplace_back(cur);
  }
  return producers;
}

std::vector<Expr> GetConsumers(const Expr& block, const Expr& root) {
  PADDLE_ENFORCE_NOT_NULL(block.As<ir::ScheduleBlockRealize>(),
                          ::common::errors::NotFound(
                              "Param block of GetConsumers should be "
                              "ir::ScheduleBlockRealize node! Please check."));
  PADDLE_ENFORCE_NOT_NULL(root.As<ir::ScheduleBlockRealize>(),
                          ::common::errors::NotFound(
                              "Param root of GetConsumers should be "
                              "ir::ScheduleBlockRealize node! Please check."));
  std::vector<Expr> consumers;
  std::string block_tensor = GetTensor(block)->name;
  if (IsReduceInitTensorName(block_tensor)) {
    std::string consumer_name = GetOriginalReduceTensorName(block_tensor);
    auto consumer =
        ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
          return x->As<ir::ScheduleBlockRealize>() &&
                 x->As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ir::ScheduleBlock>()
                         ->name == consumer_name;
        });
    PADDLE_ENFORCE_EQ(consumer.size(),
                      1,
                      ::common::errors::InvalidArgument(
                          "The number of consumer should be equal to 1!"));
    return {*consumer.begin()};
  }

  auto find_block =
      ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() && *x != block && *x != root;
      });
  for (auto& i : find_block) {
    PADDLE_ENFORCE_NOT_NULL(
        i.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>(),
        ::common::errors::NotFound(
            "ScheduleBlockRealize node's schedule_block should "
            "be ScheduleBlock!"));
    auto block_body = i.As<ir::ScheduleBlockRealize>()
                          ->schedule_block.As<ir::ScheduleBlock>()
                          ->body;
    auto find_load_or_call = ir::ir_utils::CollectIRNodesWithoutTensor(
        block_body, [&](const Expr* x) {
          if (x->As<ir::Call>()) {
            const std::vector<ir::Expr>& read_args =
                x->As<ir::Call>()->read_args;
            for (const ir::Expr& arg : read_args) {
              if (arg.as_tensor() &&
                  arg.as_tensor_ref()->name == block_tensor) {
                return true;
              }
            }
          }
          return x->As<ir::Load>() &&
                 x->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                     block_tensor;
        });
    if (!find_load_or_call.empty()) consumers.emplace_back(i);
  }
  return consumers;
}

void CheckComputeAtValidation(const Expr& block,
                              const Expr& loop,
                              const Expr& root) {
  auto find_block = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() && *x == block;
      },
      true);
  PADDLE_ENFORCE_EQ(!find_block.empty(),
                    true,
                    ::common::errors::NotFound("Didn't find block in root!"));

  auto find_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) { return x->As<ir::For>() && *x == loop; },
      true);
  PADDLE_ENFORCE_EQ(!find_loop.empty(),
                    true,
                    ::common::errors::NotFound("Didn't find loop in root!"));

  auto find_block_in_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
      loop,
      [&](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() && *x == block;
      },
      true);
  PADDLE_ENFORCE_EQ(find_block_in_loop.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The loop should not be block's ancestor!"));
}

void InsertBlock(Expr& for_loop, const Expr& insertion, int index) {  // NOLINT
  PADDLE_ENFORCE_NOT_NULL(
      for_loop.As<ir::For>(),
      ::common::errors::NotFound("Param for_loop of GetConsumers should be a "
                                 "ir::For node! Please check."));
  PADDLE_ENFORCE_NOT_NULL(
      for_loop.As<ir::For>()->body.As<Block>(),
      ::common::errors::NotFound("For node's body should be Block!"));
  ir::Block* dst_block = for_loop.As<ir::For>()->body.As<Block>();
  PADDLE_ENFORCE_EQ(
      index == -1 || index >= 0 && index < dst_block->stmts.size(),
      true,
      ::common::errors::InvalidArgument(
          "The index should be -1 or between [0, block stmts size), but got %d",
          index));

  if (index == -1) {
    dst_block->stmts.emplace_back(insertion);
  } else {
    auto dst_it = dst_block->stmts.begin() + index;
    if (dst_it->As<IfThenElse>()) {
      auto* inserted_block = dst_it->As<IfThenElse>()->true_case.As<Block>();
      PADDLE_ENFORCE_NOT_NULL(
          inserted_block,
          ::common::errors::NotFound("The IfThenElse node to be inserted "
                                     "should contain a true_case block."));
      inserted_block->stmts.insert(inserted_block->stmts.begin(), insertion);
    } else {
      dst_block->stmts.insert(dst_it, insertion);
    }
  }
}

IterRange RangeUnion(const IterRange& range1, const IterRange& range2) {
  Expr new_min = optim::ArithSimplify(Min::Make(range1.min, range2.min));
  Expr new_extent = optim::ArithSimplify(
      optim::ArithSimplify(
          Max::Make(range1.min + range1.extent, range2.min + range2.extent)) -
      new_min);
  return IterRange(new_min, new_extent);
}

std::vector<IterRange> CalculateRequiredRegions(
    const Expr& block,
    const Expr& loop,
    const Expr& root,
    const std::vector<Expr>& required_blocks,
    bool is_store_provided) {
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::NotFound(
          "Param block should be a ir::ScheduleBlockRealize node."));
  PADDLE_ENFORCE_NOT_NULL(
      loop.As<ir::For>(),
      ::common::errors::NotFound("Param loop should be a ir::For node."));

  std::set<Expr> provided_nodes;
  if (is_store_provided) {
    provided_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
        block, [&](const Expr* x) { return x->As<ir::Store>(); });
  } else {
    provided_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
        block, [&](const Expr* x) { return x->As<ir::Load>(); });
  }

  std::vector<IterRange> required_buffer_range;
  // deduce accessed regions of the provided tensor in block by iterating each
  // required block
  for (const Expr& pro_node : provided_nodes) {
    std::string provided_tensor_name =
        is_store_provided ? pro_node.As<ir::Store>()->tensor.as_tensor()->name
                          : pro_node.As<ir::Load>()->tensor.as_tensor()->name;
    if (IsReduceInitTensorName(provided_tensor_name)) {
      provided_tensor_name = GetOriginalReduceTensorName(provided_tensor_name);
    }
    for (const Expr& req_block : required_blocks) {
      PADDLE_ENFORCE_NOT_NULL(
          req_block.As<ir::ScheduleBlockRealize>(),
          ::common::errors::NotFound(
              "Param req_block should be a ir::ScheduleBlockRealize node."));
      Expr block_body =
          ir::ir_utils::IRCopy(req_block.As<ir::ScheduleBlockRealize>()
                                   ->schedule_block.As<ir::ScheduleBlock>()
                                   ->body);
      auto iter_vars = req_block.As<ir::ScheduleBlockRealize>()
                           ->schedule_block.As<ir::ScheduleBlock>()
                           ->iter_vars;
      auto iter_values = req_block.As<ir::ScheduleBlockRealize>()->iter_values;
      ReplaceExpr(&block_body, iter_vars, iter_values);

      // Notice that we look for For nodes in loop's body instead of loop
      // itself.
      auto find_loops = ir::ir_utils::CollectIRNodesWithoutTensor(
          loop.As<ir::For>()->body, [&](const Expr* x) {
            return x->As<ir::For>() && Contains(*x, req_block);
          });

      // collect vars and their ranges of each loop under the input loop
      std::vector<Var> loop_vars;
      std::vector<IterRange> loop_ranges;
      for (const auto& for_loop : find_loops) {
        loop_vars.emplace_back(for_loop.As<ir::For>()->loop_var);
        loop_ranges.emplace_back(for_loop.As<ir::For>()->min,
                                 for_loop.As<ir::For>()->extent);
      }

      std::set<Expr> required_nodes;
      if (is_store_provided) {
        required_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
            block_body, [&](const Expr* x) {
              return x->As<ir::Load>() &&
                     x->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                         provided_tensor_name;
            });
      } else {
        required_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
            block_body, [&](const Expr* x) {
              return x->As<ir::Store>() &&
                     x->As<ir::Store>()->tensor.as_tensor_ref()->name ==
                         provided_tensor_name;
            });
      }

      // deducing range by indices of each required node
      for (const Expr& req_node : required_nodes) {
        const auto& indices = is_store_provided
                                  ? req_node.As<ir::Load>()->indices
                                  : req_node.As<ir::Store>()->indices;

        if (find_loops.empty()) {
          for (int i = 0; i < indices.size(); ++i) {
            if (i >= required_buffer_range.size())
              required_buffer_range.emplace_back(indices[i], Expr(1));
            else
              required_buffer_range[i] = RangeUnion(
                  required_buffer_range[i], IterRange(indices[i], Expr(1)));
          }
        } else {
          for (int i = 0; i < indices.size(); ++i) {
            auto range = GetAccessedRange(indices[i], loop_vars, loop_ranges);
            if (i >= required_buffer_range.size()) {
              required_buffer_range.emplace_back(std::move(range));
            } else {
              required_buffer_range[i] =
                  RangeUnion(required_buffer_range[i], range);
            }
          }
        }
      }  // end for load_nodes
    }
  }

  int iter_size = block.As<ir::ScheduleBlockRealize>()->iter_values.size();
  // maybe some dimensions are not accessed by consumers so we should append
  // them
  if (iter_size > required_buffer_range.size()) {
    for (int i = required_buffer_range.size(); i < iter_size; ++i) {
      PADDLE_ENFORCE_EQ(
          block.As<ir::ScheduleBlockRealize>()->iter_values[i].as_var() ||
              block.As<ir::ScheduleBlockRealize>()
                  ->iter_values[i]
                  .is_constant(),
          true,
          ::common::errors::InvalidArgument(
              "ScheduleBlockRealize node's iter "
              "values should be var or constant."));
      if (block.As<ir::ScheduleBlockRealize>()->iter_values[i].as_var()) {
        auto find_for_loops =
            ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
              return x->As<ir::For>() &&
                     x->As<ir::For>()->loop_var->name ==
                         block.As<ir::ScheduleBlockRealize>()
                             ->iter_values[i]
                             .as_var_ref()
                             ->name;
            });
        PADDLE_ENFORCE_EQ(find_for_loops.size(),
                          1U,
                          ::common::errors::InvalidArgument(
                              "The number of For nodes should be equal to 1!"));
        required_buffer_range.emplace_back(
            (*find_for_loops.begin()).As<ir::For>()->min,
            (*find_for_loops.begin()).As<ir::For>()->extent);
      } else {
        int cons = static_cast<int>(
            block.As<ir::ScheduleBlockRealize>()->iter_values[i].is_constant());
        required_buffer_range.emplace_back(Expr(cons), Expr(1));
      }
    }
  }
  return required_buffer_range;
}

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block,
                                             const Expr& root) {
  PADDLE_ENFORCE_NOT_NULL(
      schedule_block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::NotFound(
          "Param schedule_block should be a ir::ScheduleBlockRealize node."));
  auto compute_body = schedule_block.As<ir::ScheduleBlockRealize>()
                          ->schedule_block.As<ir::ScheduleBlock>()
                          ->body;
  // 1. Check the schedule block to be inlined is not a reduce tensor.
  auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  PADDLE_ENFORCE_EQ(find_store.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Store nodes should be equal to 1!"));
  Expr tensor = (*find_store.begin()).As<ir::Store>()->tensor;
  PADDLE_ENFORCE_EQ(!tensor.as_tensor_ref()->is_reduce_tensor(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Param tensor should not be a reduce tensor!"));
  // 2. Check this schedule block is the only writer of the tensor.
  find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) {
        return x->As<ir::Store>() &&
               (x->As<ir::Store>()->tensor).as_tensor_ref()->name ==
                   tensor.as_tensor_ref()->name;
      },
      true);
  PADDLE_ENFORCE_EQ(find_store.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Store nodes should be equal to 1!"));
  // 3. Check there is no overlap between the buffers the schedule block reads
  // and writes.
  auto find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) {
        return x->As<ir::Load>() && x->As<ir::Load>()->tensor == tensor;
      });
  PADDLE_ENFORCE_EQ(
      find_load.empty(),
      true,
      ::common::errors::InvalidArgument("The find_load should be empty!"));
  return (*find_store.begin());
}

std::tuple<Expr, Expr, Expr> CheckReverseComputeInlineValidationAndGetExprs(
    const Expr& schedule_block, const Expr& root) {
  PADDLE_ENFORCE_NOT_NULL(
      schedule_block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::NotFound(
          "Param schedule_block should be a ir::ScheduleBlockRealize node."));
  auto compute_body = schedule_block.As<ir::ScheduleBlockRealize>()
                          ->schedule_block.As<ir::ScheduleBlock>()
                          ->body;
  // 1. Check the schedule block to be reverse inlined is not a reduce tensor.
  auto find_inlined_load = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Load>(); }, true);
  PADDLE_ENFORCE_EQ(find_inlined_load.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Load nodes should be equal to 1!"));
  Expr tensor = (*find_inlined_load.begin()).As<ir::Load>()->tensor;
  PADDLE_ENFORCE_EQ(!tensor.as_tensor_ref()->is_reduce_tensor(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Param tensor should not be a reduce tensor!"));
  auto inlined_load = *find_inlined_load.begin();
  // 2. Check this schedule block is the only reader of the tensor.
  auto find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) {
        return x->As<ir::Load>() &&
               (x->As<ir::Load>()->tensor).as_tensor_ref()->name ==
                   tensor.as_tensor_ref()->name;
      },
      true);
  PADDLE_ENFORCE_EQ(find_load.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Load nodes should be equal to 1!"));
  // 3. Check there is no overlap between the buffers the schedule block reads
  // and writes.
  auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) {
        return x->As<ir::Store>() && x->As<ir::Store>()->tensor == tensor;
      });
  PADDLE_ENFORCE_EQ(
      find_store.empty(),
      true,
      ::common::errors::InvalidArgument("The find_store should be empty!"));
  // 4. Get store that will be inlined.
  auto find_inlined_store =
      ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
        return x->As<ir::Store>() && x->As<ir::Store>()->tensor == tensor;
      });
  PADDLE_ENFORCE_EQ(find_inlined_store.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Store nodes should be equal to 1!"));
  auto inlined_store = *find_inlined_store.begin();
  // 5. Get target store.
  auto find_target_store = ir::ir_utils::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  PADDLE_ENFORCE_EQ(find_target_store.size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The number of Store nodes should be equal to 1!"));
  auto target_store = *find_target_store.begin();
  return {inlined_load, inlined_store, target_store};
}

bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name) {
  for (auto& expr : exprs) {
    auto find_expr = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr,
        [&](const Expr* x) {
          return x->As<_Var_>() && x->As<_Var_>()->name == var_name;
        },
        true);
    if (!find_expr.empty()) return true;
  }
  return false;
}

std::unordered_map<int, int> PrimeFactorize(int n) {
  std::unordered_map<int, int> factors;
  while (n % 2 == 0) {
    ++factors[2];
    n /= 2;
  }
  for (int i = 3; i <= sqrt(n); i += 2) {
    while (n % i == 0) {
      ++factors[i];
      n /= i;
    }
  }
  if (n > 2) {
    factors[n] = 1;
  }
  return factors;
}

std::vector<int> SampleTile(utils::LinearRandomEngine::StateType* rand_seed,
                            int n,
                            int extent) {
  std::vector<int> tile;
  while (n > 1) {
    std::unordered_map<int, int> factors = PrimeFactorize(extent);
    int product = 1;
    for (auto& factor : factors) {
      if (factor.second >= 1) {
        int num = utils::SampleUniformInt(1, factor.second + 1, rand_seed);
        product *= std::pow(factor.first, num);
      }
    }
    tile.push_back(product);
    extent /= product;
    --n;
  }
  tile.push_back(extent);
  return tile;
}

bool ContainDynamicShape(const Expr& expr) {
  auto loop_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
      expr, [&](const Expr* x) { return x->As<ir::For>(); });
  for (const auto& n : loop_nodes) {
    auto for_node = n.As<ir::For>();
    // we only deal static index shape now.
    if (!for_node->extent.is_index()) return true;
    if (for_node->extent.as_index().IsDynamic()) return true;
  }
  return false;
}
}  // namespace ir
}  // namespace cinn
