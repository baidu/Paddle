// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/arrange_storage_tactic.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

class ArrangeStorageTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "ArrangeStorageTactic"; }

 private:
  std::unordered_set<std::string> output_names_;
};

// [block_name, [var, for_node]]
using VarToForMap =
    std::unordered_map<std::string, std::unordered_map<ir::Var, ir::Expr>>;
using IntSet = common::SingleIntervalIntSet;

enum class CudaAxisType : int {
  kCudaBlock = 0,
  kCudaThread = 1,
};

struct CudaAxisSpace {
  IntSet x{Expr(0), Expr(0)};
  IntSet y{Expr(0), Expr(0)};
  IntSet z{Expr(0), Expr(0)};
  CudaAxisType type;
};

struct FixedCudaIterVarName {
  static constexpr char* kCudaBlockX = "fixed_cuda_block_x";
  static constexpr char* kCudaBlockY = "fixed_cuda_block_y";
  static constexpr char* kCudaBlockZ = "fixed_cuda_block_z";
  static constexpr char* kCudaThreadX = "fixed_cuda_thread_x";
  static constexpr char* kCudaThreadY = "fixed_cuda_thread_y";
  static constexpr char* kCudaThreadZ = "fixed_cuda_thread_z";
};

std::optional<bool> IsSubCudaAxisSpace(const CudaAxisSpace& lhs,
                                       const CudaAxisSpace& rhs) {
  PADDLE_ENFORCE_EQ(
      lhs.type,
      rhs.type,
      ::common::errors::InvalidArgument(
          "The type of 'lhs' must be equal to the type of 'rhs'. "));
  std::optional<bool> prove_sub_x = lhs.x.ProveSubSet(rhs.x);
  std::optional<bool> prove_sub_y = lhs.y.ProveSubSet(rhs.y);
  std::optional<bool> prove_sub_z = lhs.z.ProveSubSet(rhs.z);
  if (!prove_sub_x.has_value() || !prove_sub_y.has_value() ||
      !prove_sub_z.has_value()) {
    return std::nullopt;
  }
  return prove_sub_x.value() && prove_sub_y.value() && prove_sub_z.value();
}

std::tuple<CudaAxisSpace, CudaAxisSpace> GetCudaAxisSpace(
    const VarToForMap& var2for_map, const std::string block_name) {
  CudaAxisSpace cuda_block_space{IntSet{Expr(0), Expr(0)},
                                 IntSet{Expr(0), Expr(0)},
                                 IntSet{Expr(0), Expr(0)},
                                 CudaAxisType::kCudaBlock};
  CudaAxisSpace cuda_thread_space{IntSet{Expr(0), Expr(0)},
                                  IntSet{Expr(0), Expr(0)},
                                  IntSet{Expr(0), Expr(0)},
                                  CudaAxisType::kCudaThread};
  PADDLE_ENFORCE_GT(var2for_map.count(block_name),
                    0,
                    ::common::errors::InvalidArgument("block_name not found"));
  for (const auto& var2for : var2for_map.at(block_name)) {
    const Expr& for_expr = var2for.second;
    const ir::For* for_node = for_expr.As<ir::For>();
    PADDLE_ENFORCE_NOT_NULL(
        for_node, ::common::errors::InvalidArgument("for_node is nullptr"));
    IntSet interval{
        for_node->min,
        common::AutoSimplify(for_node->min + for_node->extent - Expr(1))};
    if (for_node->is_gpu_thread_binded()) {
      if (for_node->bind_info().offset == 0) {
        cuda_thread_space.x = interval;
      } else if (for_node->bind_info().offset == 1) {
        cuda_thread_space.y = interval;
      } else if (for_node->bind_info().offset == 2) {
        cuda_thread_space.z = interval;
      }
    } else if (for_node->is_gpu_block_binded()) {
      if (for_node->bind_info().offset == 0) {
        cuda_block_space.x = interval;
      } else if (for_node->bind_info().offset == 1) {
        cuda_block_space.y = interval;
      } else if (for_node->bind_info().offset == 2) {
        cuda_block_space.z = interval;
      }
    }
  }
  VLOG(6) << "GetCudaAxisSpace of block: " << block_name
          << "\n cuda_block_space: ["
          << "x = [" << cuda_block_space.x.Min() << " : "
          << cuda_block_space.x.Max() << "] "
          << "y = [" << cuda_block_space.y.Min() << " : "
          << cuda_block_space.y.Max() << "] "
          << "z = [" << cuda_block_space.z.Min() << " : "
          << cuda_block_space.z.Max() << "]]"
          << "\n cuda_thread_space: ["
          << "x = [" << cuda_thread_space.x.Min() << " : "
          << cuda_thread_space.x.Max() << "] "
          << "y = [" << cuda_thread_space.y.Min() << " : "
          << cuda_thread_space.y.Max() << "] "
          << "z = [" << cuda_thread_space.z.Min() << " : "
          << cuda_thread_space.z.Max() << "]]";
  return {cuda_block_space, cuda_thread_space};
}

IntSet Evaluate(Expr expr,
                const std::unordered_map<ir::Var, ir::Var>& fixed,
                const std::unordered_map<ir::Var, IntSet>& var_domain) {
  Expr copy_for_upper_bound = ir::ir_utils::IRCopy(expr);
  Expr copy_for_lower_bound = ir::ir_utils::IRCopy(expr);
  common::cas_intervals_t var_intervals;
  std::vector<ir::Expr> var_vec = ir::ir_utils::CollectIRNodesWithoutTensor(
      expr, [](const ir::Expr* x) { return x->as_var(); });
  for (Expr var_expr : var_vec) {
    ir::Var var = var_expr.as_var_ref();
    if (fixed.count(var) != 0) {
      const ir::Var& fixed_var = fixed.at(var);
      var_intervals.emplace(
          fixed_var->name,
          common::CasInterval(fixed_var->lower_bound, fixed_var->upper_bound));
      optim::ReplaceVarWithExpr(&copy_for_lower_bound, var, Expr(fixed_var));
      optim::ReplaceVarWithExpr(&copy_for_upper_bound, var, Expr(fixed_var));
    } else if (var_domain.count(var) != 0) {
      Expr var_min = var_domain.at(var).Min();
      Expr var_max = var_domain.at(var).Max();
      optim::ReplaceVarWithExpr(&copy_for_lower_bound, var, var_min);
      optim::ReplaceVarWithExpr(&copy_for_upper_bound, var, var_max);
    } else if (var->is_symbolic_constant) {
      continue;
    } else {
      PADDLE_ENFORCE_EQ(
          var->lower_bound.defined(),
          true,
          ::common::errors::InvalidArgument(
              "The 'lower_bound' of the variable must be defined."));
      PADDLE_ENFORCE_EQ(
          var->upper_bound.defined(),
          true,
          ::common::errors::InvalidArgument(
              "The 'upper_bound' of the variable must be defined."));
      optim::ReplaceVarWithExpr(&copy_for_lower_bound, var, var->lower_bound);
      optim::ReplaceVarWithExpr(&copy_for_upper_bound, var, var->upper_bound);
    }
  }
  ir::Expr lower_bound =
      common::AutoSimplify(copy_for_lower_bound, var_intervals);
  ir::Expr upper_bound =
      common::AutoSimplify(copy_for_upper_bound, var_intervals);
  lower_bound = common::EnhancedSimplifyModExpr(lower_bound, var_intervals);
  upper_bound = common::EnhancedSimplifyModExpr(upper_bound, var_intervals);
  return IntSet(lower_bound, upper_bound, var_intervals);
}

std::unordered_map<ir::Var, ir::Var> GetFixedVar(
    const VarToForMap& var2for_map,
    const std::string& block_name,
    const CudaAxisSpace& cuda_space) {
  if (var2for_map.count(block_name) == 0) return {};
  std::unordered_map<ir::Var, ir::Var> fix_var_map;
  const CudaAxisType& type = cuda_space.type;
  for (const std::pair<ir::Var, ir::Expr>& var2for :
       var2for_map.at(block_name)) {
    const ir::For* for_node = var2for.second.As<ir::For>();
    if (type == CudaAxisType::kCudaBlock && for_node->is_gpu_block_binded()) {
      if (for_node->bind_info().offset == 0) {
        fix_var_map.insert(
            {var2for.first,
             ir::_Var_::Make(cuda_space.x.Min(),
                             cuda_space.x.Max(),
                             FixedCudaIterVarName::kCudaBlockX,
                             var2for.first->is_reduce_axis,
                             /* is_symbolic_constant = */ true)});
      } else if (for_node->bind_info().offset == 1) {
        fix_var_map.insert(
            {var2for.first,
             ir::_Var_::Make(cuda_space.y.Min(),
                             cuda_space.y.Max(),
                             FixedCudaIterVarName::kCudaBlockY,
                             var2for.first->is_reduce_axis,
                             /* is_symbolic_constant = */ true)});
      } else if (for_node->bind_info().offset == 2) {
        fix_var_map.insert(
            {var2for.first,
             ir::_Var_::Make(cuda_space.z.Min(),
                             cuda_space.z.Max(),
                             FixedCudaIterVarName::kCudaBlockZ,
                             var2for.first->is_reduce_axis,
                             /* is_symbolic_constant = */ true)});
      }
    } else if (type == CudaAxisType::kCudaThread &&
               for_node->is_gpu_thread_binded()) {
      if (for_node->bind_info().offset == 0) {
        fix_var_map.insert(
            {var2for.first,
             ir::_Var_::Make(cuda_space.x.Min(),
                             cuda_space.x.Max(),
                             FixedCudaIterVarName::kCudaThreadX,
                             var2for.first->is_reduce_axis,
                             /* is_symbolic_constant = */ true)});
      } else if (for_node->bind_info().offset == 1) {
        fix_var_map.insert(
            {var2for.first,
             ir::_Var_::Make(cuda_space.y.Min(),
                             cuda_space.y.Max(),
                             FixedCudaIterVarName::kCudaThreadY,
                             var2for.first->is_reduce_axis,
                             /* is_symbolic_constant = */ true)});
      } else if (for_node->bind_info().offset == 2) {
        fix_var_map.insert(
            {var2for.first,
             ir::_Var_::Make(cuda_space.z.Min(),
                             cuda_space.z.Max(),
                             FixedCudaIterVarName::kCudaThreadZ,
                             var2for.first->is_reduce_axis,
                             /* is_symbolic_constant = */ true)});
      }
    }
  }
  return fix_var_map;
}

std::unordered_map<ir::Var, IntSet> GetVarDomainOfSBlock(
    const VarToForMap& var2for_map, const std::string& block_name) {
  if (var2for_map.count(block_name) == 0) return {};
  std::unordered_map<ir::Var, IntSet> var_domains;
  for (const std::pair<ir::Var, ir::Expr>& var2for :
       var2for_map.at(block_name)) {
    const ir::For* for_node = var2for.second.As<ir::For>();
    var_domains.emplace(
        var2for.first,
        IntSet(for_node->min,
               common::AutoSimplify(for_node->min + for_node->extent -
                                    ir::Expr(1))));
  }
  return var_domains;
}

std::optional<CudaAxisType> AnalyzeCrossType(const VarToForMap& var2for_map,
                                             Expr store,
                                             Expr load,
                                             Expr store_block,
                                             Expr load_block) {
  PADDLE_ENFORCE_NOT_NULL(
      store_block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "The 'store_block' must be of type 'ir::ScheduleBlockRealize'."));
  PADDLE_ENFORCE_NOT_NULL(
      load_block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "The 'load_block' must be of type 'ir::ScheduleBlockRealize'."));
  std::string store_block_name = store_block.As<ir::ScheduleBlockRealize>()
                                     ->schedule_block.As<ir::ScheduleBlock>()
                                     ->name;
  std::string load_block_name = load_block.As<ir::ScheduleBlockRealize>()
                                    ->schedule_block.As<ir::ScheduleBlock>()
                                    ->name;
  VLOG(6) << "Analyzing cross type of Store: [" << store << "] and Load: ["
          << load << "]";

  // 1. Determine domain range
  CudaAxisSpace cuda_block_space_of_store;
  CudaAxisSpace cuda_thread_space_of_store;
  std::tie(cuda_block_space_of_store, cuda_thread_space_of_store) =
      GetCudaAxisSpace(var2for_map, store_block_name);
  CudaAxisSpace cuda_block_space_of_load;
  CudaAxisSpace cuda_thread_space_of_load;
  std::tie(cuda_block_space_of_load, cuda_thread_space_of_load) =
      GetCudaAxisSpace(var2for_map, load_block_name);
  std::optional<bool> is_block_sub_space =
      IsSubCudaAxisSpace(cuda_block_space_of_load, cuda_block_space_of_store);
  if (!is_block_sub_space.has_value() || !is_block_sub_space.value()) {
    VLOG(6) << "load cuda block space is not sub space of store";
    return CudaAxisType::kCudaBlock;
  }
  VLOG(6) << "load cuda block space is sub space of store";
  std::optional<bool> is_thread_sub_space =
      IsSubCudaAxisSpace(cuda_thread_space_of_load, cuda_thread_space_of_store);
  if (!is_thread_sub_space.has_value() || !is_thread_sub_space.value()) {
    VLOG(6) << "load cuda thread space is not sub space of store";
    return CudaAxisType::kCudaThread;
  }
  VLOG(6) << "load cuda thread space is sub space of store";

  // 2. Determine value range
  std::unordered_map<ir::Var, ir::Var> cuda_block_fixed_var_of_store =
      GetFixedVar(var2for_map, store_block_name, cuda_block_space_of_load);
  std::unordered_map<ir::Var, ir::Var> cuda_block_fixed_var_of_load =
      GetFixedVar(var2for_map, load_block_name, cuda_block_space_of_load);
  std::unordered_map<ir::Var, ir::Var> cuda_thread_fixed_var_of_store =
      GetFixedVar(var2for_map, store_block_name, cuda_thread_space_of_load);
  std::unordered_map<ir::Var, ir::Var> cuda_thread_fixed_var_of_load =
      GetFixedVar(var2for_map, load_block_name, cuda_thread_space_of_load);
  std::unordered_map<ir::Var, ir::Var> cuda_block_thread_fixed_var_of_store =
      cuda_block_fixed_var_of_store;
  cuda_block_thread_fixed_var_of_store.insert(
      cuda_thread_fixed_var_of_store.begin(),
      cuda_thread_fixed_var_of_store.end());
  std::unordered_map<ir::Var, ir::Var> cuda_block_thread_fixed_var_of_load =
      cuda_block_fixed_var_of_load;
  cuda_block_thread_fixed_var_of_store.insert(
      cuda_thread_fixed_var_of_load.begin(),
      cuda_thread_fixed_var_of_load.end());
  std::unordered_map<ir::Var, IntSet> store_var_domain =
      GetVarDomainOfSBlock(var2for_map, store_block_name);
  std::unordered_map<ir::Var, IntSet> load_var_domain =
      GetVarDomainOfSBlock(var2for_map, load_block_name);
  std::vector<ir::Expr> iter_values_of_store =
      analyzer::GetIterValuesOfAccess(store, store_block);
  std::vector<ir::Expr> iter_values_of_load =
      analyzer::GetIterValuesOfAccess(load, load_block);
  PADDLE_ENFORCE_EQ(iter_values_of_load.size(),
                    iter_values_of_store.size(),
                    ::common::errors::InvalidArgument(
                        "The number of iter values of store and load should be "
                        "the same"));

  for (int i = 0; i < iter_values_of_load.size(); ++i) {
    IntSet block_store_range = Evaluate(iter_values_of_store[i],
                                        cuda_block_fixed_var_of_store,
                                        store_var_domain);
    IntSet block_load_range = Evaluate(
        iter_values_of_load[i], cuda_block_fixed_var_of_load, load_var_domain);
    VLOG(6) << "block_store_range of [" << iter_values_of_store[i] << "] = ["
            << block_store_range.Min() << " : " << block_store_range.Max()
            << "]";
    VLOG(6) << "block_load_range of [" << iter_values_of_load[i] << "] = ["
            << block_load_range.Min() << " : " << block_load_range.Max() << "]";
    std::optional<bool> is_block_sub_set =
        block_load_range.ProveSubSet(block_store_range);
    if (!is_block_sub_set.has_value() || !is_block_sub_set.value()) {
      VLOG(6) << "load range of a cuda block is not sub set of store";
      return CudaAxisType::kCudaBlock;
    }

    IntSet thread_store_range = Evaluate(iter_values_of_store[i],
                                         cuda_block_thread_fixed_var_of_store,
                                         store_var_domain);
    IntSet thread_load_range = Evaluate(iter_values_of_load[i],
                                        cuda_block_thread_fixed_var_of_load,
                                        load_var_domain);
    VLOG(6) << "thread_store_range of [" << iter_values_of_store[i] << "] = ["
            << thread_store_range.Min() << " : " << thread_store_range.Max()
            << "]";
    VLOG(6) << "thread_load_range of [" << iter_values_of_load[i] << "] = ["
            << thread_load_range.Min() << " : " << thread_load_range.Max()
            << "]";
    std::optional<bool> is_thread_sub_set =
        thread_load_range.ProveSubSet(thread_store_range);
    if (!is_thread_sub_set.has_value() || !is_thread_sub_set.value()) {
      VLOG(6) << "load range of a cuda thread is not sub set of store";
      return CudaAxisType::kCudaThread;
    }
  }

  return std::nullopt;
}

void ArrangeStorageTactic::Init(ScheduleContext* context) {
  output_names_ = context->output_names;
}

void ArrangeStorageTactic::Apply(ir::IRSchedule* sch,
                                 const std::string& block_id) {
  ir::Expr store_block = sch->GetBlock(block_id);
  ir::Tensor store_tensor = analyzer::GetStoreTensorOfSBlock(store_block);
  // Skip if the store tensor has already been allocated to GPU shared or local
  // memory.
  if (store_tensor->buffer.defined() && store_tensor->buffer->is_on_gpu())
    return;

  ir::Expr root_block = sch->GetRootBlock(store_block);
  ir::Expr store = analyzer::GetStoreOfSBlock(store_block);
  VarToForMap var2for_map =
      analyzer::CollectVarToForMap({root_block}, sch->GetAllBlocks());

  // Traverse load nodes to check if there are loads that cross cuda blocks or
  // threads
  std::vector<std::pair<Expr, Expr>> loads_and_blocks =
      analyzer::GetConsumerLoadsAndSBlocks(store_block, root_block);

  ir::MemoryType memory_type = ir::MemoryType::GPULocal;
  for (const auto& load_and_block : loads_and_blocks) {
    ir::Expr load = load_and_block.first;
    ir::Expr load_block = load_and_block.second;
    std::optional<CudaAxisType> cross_type =
        AnalyzeCrossType(var2for_map, store, load, store_block, load_block);
    if (!cross_type.has_value()) {
      memory_type = ir::MemoryType::GPULocal;
    } else if (cross_type.value() == CudaAxisType::kCudaThread) {
      memory_type = ir::MemoryType::GPUShared;
    } else if (cross_type.value() == CudaAxisType::kCudaBlock) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Fusion requires synchronization across blocks, but "
          "currently we do not support it."));
      break;
    } else {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    }
  }

  // Set output tensor to global
  if (output_names_.count(block_id) > 0) {
    memory_type = ir::MemoryType::Auto;
  }
  // Set the reduce_init tensor and the real tensor to the same memory
  if (ir::IsReduceInitTensorName(block_id)) {
    ir::Expr block = sch->GetBlock(ir::GetOriginalReduceTensorName(block_id));
    memory_type = analyzer::GetStoreTensorOfSBlock(block)->buffer->memory_type;
  }
  // Do schedule
  std::unordered_set<std::string> sync_mark;
  if (memory_type == ir::MemoryType::Auto) {
    VLOG(6) << "Set store tensor of block " << block_id << " to global";
  } else if (memory_type == ir::MemoryType::GPUShared) {
    VLOG(6) << "Set store tensor of block " << block_id << " to shared";
    sch->SetBuffer(store_block, "shared");
    std::vector<ir::Expr> loops = sch->GetLoops(store_block);
    if (sync_mark.count(ir::GetOriginalReduceTensorName(block_id)) == 0) {
      sch->SyncThreads(loops.back(), true);
      sync_mark.insert(ir::GetOriginalReduceTensorName(block_id));
    }
  } else if (memory_type == ir::MemoryType::GPULocal) {
    VLOG(6) << "Set store tensor of block " << block_id << " to register";
    sch->SetBuffer(store_block, "local");
  }
}

std::unique_ptr<ScheduleTactic> CreateArrangeStorageTactic() {
  return std::make_unique<ArrangeStorageTactic>();
}

}  // namespace ir
}  // namespace cinn
