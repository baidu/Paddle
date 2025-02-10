// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/align_iter_space_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

namespace cinn {
namespace ir {
namespace {

/**
 * Reorder the loops according to the memory-consistent order of input or output
 * to make memory access as coalesced as possible.
 *
 * This tactic uses different alignment policies for Reduce and Trivial:
 * 1) Reduce: align with the input, because after reduction, the output data is
 *    significantly smaller than the input data, so it's more critical to make
 *    input coalesced.
 * 2) Trivial: align with the output, because discrete writes incur higher costs
 *    than discrete reads for the same volume of data due to the hardware design
 *    of cache. Therefore, we should ensure coalesced writes in priority.
 *
 * Note: we reorder spatial and reduce loops separately, because we need to
 * maintain the relative order between spatial and reduce loops, so as for later
 * tactics to work properly. Thus, we use two lists sp_loop_perm & rd_loop_perm
 * to record the permutation of spatial and reduce loops respectively.
 *
 *
 * Examples:
 * 1. Reduce
 * Input:
 *   for (i, 0, 8):          # S
 *     for (j, 0, 32):       # S
 *       for (k, 0, 128):    # R
 *         for (a, 0, 256):  # R
 *           var_1[i, j] += var_0[j, a, k, i]
 * Analysis:
 *   We align Reduce to the input `var_0[j, a, k, i]`. In the indices of var_0,
 *   the mapping from each index to the loop index is:
 *      indices[0] = j   =>  loops[1]  # S
 *      indices[1] = a   =>  loops[3]  # R
 *      indices[2] = k   =>  loops[2]  # R
 *      indices[3] = i   =>  loops[0]  # S
 *   To make the indices of var_0 consistent with its original memory layout, we
 *   need to permute the loops in the order {1, 3, 2, 0}. However, as we reorder
 *   spatial and reduce loop separately, we split the permutation into sp & rd,
 *   getting sp_loop_perm = {1, 0} and rd_loop_perm = {3, 2}.
 * Output:
 *   for (j, 0, 32):         # S
 *     for (i, 0, 8):        # S
 *       for (a, 0, 256):    # R
 *         for (k, 0, 128):  # R
 *           var_1[i, j] += var_0[j, a, k, i]
 *
 * 2. Trivial
 * Input:
 *   for (i, 0, 32):
 *     for (j, 0, 128):
 *       for (k, 0, 256):
 *         var_1[k, i, j] = exp(var_0[j, i, k])
 * Analysis:
 *   We align Trivial to the output `var_1[k, i, j]`. In the indices of var_1,
 *   the mapping from each index to the loop index is:
 *      indices[0] = k  => loops[2]
 *      indices[1] = i  => loops[0]
 *      indices[2] = j  => loops[1]
 *   Like example 1, we should permute the loops in the order {2, 0, 1}. As this
 *   graph doesn't contain reduce loops, all we get is sp_loop_perm = {2, 0, 1},
 *   and rd_loop_perm = {}.
 * Output:
 *   for (k, 0, 256):
 *     for (i, 0, 32):
 *       for (j, 0, 128):
 *         var_1[k, i, j] = exp(var_0[j, i, k])
 */
class AlignIterSpaceTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "AlignIterSpaceTactic"; }

 private:
  /**
   * Get the common memory-consistent order of loops according to the outputs.
   * Returns null if not all outputs share the same order.
   */
  std::vector<int> GetCommonOutputLoopPerm(ir::IRSchedule* sch);

 private:
  ScheduleContext* context_;

  // The permutation of spatial and reduce loops, in other to achieve the
  // memory-consistent alignment.
  std::vector<int> sp_loop_perm_;
  std::vector<int> rd_loop_perm_;
};

void AlignIterSpaceTactic::Init(ScheduleContext* context, ir::IRSchedule* sch) {
  context_ = context;
  sp_loop_perm_.clear();
  rd_loop_perm_.clear();

  auto& loop_strides = context_->config.base_info->loop_strides;
  auto& reduce_axis = context_->config.base_info->reduce_axis;
  std::set<int> reduce_axis_set(reduce_axis.begin(), reduce_axis.end());

  if (!loop_strides.empty()) {
    // If this is a Reduce, calculate the loop_perm by sorting the loops in the
    // descending order of their strides according to the input, then split the
    // loop_perm into sp_loop_perm & rd_loop_perm.
    std::vector<int> loop_perm(loop_strides.size());
    std::iota(loop_perm.begin(), loop_perm.end(), 0);
    std::stable_sort(loop_perm.begin(), loop_perm.end(), [&](int a, int b) {
      return loop_strides[a] > loop_strides[b];
    });

    for (int axis : loop_perm) {
      if (reduce_axis_set.count(axis) > 0) {
        rd_loop_perm_.push_back(axis);
      } else if (loop_strides[axis] != 0) {
        sp_loop_perm_.push_back(axis);
      }
    }
  } else {
    // If this is a Trvial, calculate the sp_loop_perm according to the output.
    sp_loop_perm_ = GetCommonOutputLoopPerm(sch);
  }

  VLOG(4) << "AlignIterSpaceTactic:\n"
          << "sp_loop_perm: " << utils::Join(sp_loop_perm_, ", ") << "\n"
          << "rd_loop_perm: " << utils::Join(rd_loop_perm_, ", ");
}

std::unordered_map<ir::Var, int> GetLoopVarToIndex(
    const std::vector<ir::Expr>& loops) {
  std::unordered_map<ir::Var, int> loop_var2index;
  for (int i = 0; i < loops.size(); ++i) {
    auto* node = loops[i].As<ir::For>();
    loop_var2index[node->loop_var] = i;
  }
  return loop_var2index;
}

/**
 * Check whether this is an effective permutation.
 * A permutation is ineffective if it's entirely in ascending order.
 */
bool IsPermutationEffective(const std::vector<int>& perm) {
  for (int i = 1; i < perm.size(); ++i) {
    if (perm[i - 1] > perm[i]) return true;
  }
  return false;
}

std::vector<int> AlignIterSpaceTactic::GetCommonOutputLoopPerm(
    ir::IRSchedule* sch) {
  std::vector<int> common_loop_perm;

  for (auto& block : sch->GetAllBlocks()) {
    std::string block_id = ir::analyzer::GetBlockName(block);
    if (context_->output_names.count(block_id) == 0) continue;

    auto store = ir::analyzer::GetStoreOfSBlock(block);
    auto& indices = store.As<ir::Store>()->indices;
    std::unordered_map<ir::Var, ir::Expr> iter_var2iter_value =
        ir::analyzer::GetIterVarToValueOfSBlock(block);
    std::unordered_map<ir::Var, int> loop_var2index =
        GetLoopVarToIndex(sch->GetLoops(block));

    std::vector<int> loop_perm;
    for (auto& index : indices) {
      if (index.is_constant()) continue;
      if (!index.is_var()) return {};
      ir::Expr iter_value = iter_var2iter_value[index.as_var_ref()];
      if (!iter_value.is_var()) return {};
      ir::Expr loop_var = iter_value.as_var_ref();
      loop_perm.push_back(loop_var2index[loop_var]);
    }

    if (common_loop_perm.empty()) {
      common_loop_perm = std::move(loop_perm);
    } else if (common_loop_perm != loop_perm) {
      return {};
    }
  }

  return common_loop_perm;
}

void AlignIterSpaceTactic::Apply(ir::IRSchedule* sch,
                                 const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;
  if (IsPermutationEffective(sp_loop_perm_)) {
    sch->Reorder(block_id, sp_loop_perm_);
  }
  if (IsPermutationEffective(rd_loop_perm_)) {
    sch->Reorder(block_id, rd_loop_perm_);
  }
}

}  // namespace

std::unique_ptr<ScheduleTactic> CreateAlignIterSpaceTactic() {
  return std::make_unique<AlignIterSpaceTactic>();
}

}  // namespace ir
}  // namespace cinn
