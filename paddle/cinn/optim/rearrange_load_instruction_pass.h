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
#include "paddle/cinn/pass/pass.h"

PD_DECLARE_bool(cinn_enable_rearrange_load);

namespace cinn {
namespace optim {
class RearrangeLoadInstructionPass : public FuncPass {
 public:
  RearrangeLoadInstructionPass() : FuncPass("rearrange_load_instruction") {}
  LogicalResult Run(ir::LoweredFunc func) override;
};

/*
 * Rearrange global memory loads in front of expressions to optimize the
 * instruction pipeline at the assembly level for GPUs.
 *
 * This pass operates on leaf blocks (blocks in the inner-most loops). It first
 * extracts loads from each schedule block in a leaf block, then places these
 * loads at the beginning of the block. By doing so, it overlaps the memory
 * latency of multiple loads, minimizes pipeline stalls, and therefore improves
 * the throughput.
 *
 *
 * Background:
 * GPU architectures are characterized by deep, in-order execution pipelines.
 * Unlike modern CPUs, which can execute instructions out of order at the
 * hardware level, GPUs follow a strict in-order execution model. Therefore,
 * when a subsequent instruction depends on a previous one that requires a
 * significant amount of time to complete, the pipeline will stall, severely
 * impacting performance.
 *
 * For example, consider the following assembly code:
 *   (I1)  LOAD x1, [s1]     // x1 = *s1
 *   (I2)  ADD  x2, x0, x1   // x2 = x0 + x1
 *   (I3)  LOAD x3, [s2]     // x3 = *s2
 *   (I4)  MUL  x4, x2, x3   // x4 = x2 * x3
 * In this sequence, instruction (I2) depends on the result of (I1). If (I1) is
 * a long-latency load operation, taking a significant amount of time (let's say
 * T0), (I2) cannot be issued until (I1) completes. This dependency effectively
 * blocks all succeeding instructions from being dispatched. Moreover, (I3)
 * cannot be issued until both (I1) and (I2) are completed. If (I3) is also a
 * long-latency load taking the same time, T0, we would spend approximately 2*T0
 * on this segment of code.
 *
 * However, by observing that (I2) and (I3) are independent of each other, we
 * can rearrange the instructions as follows:
 *   (I1)  LOAD x1, [s1]     // x1 = *s1
 *   (I3)  LOAD x3, [s2]     // x3 = *s2
 *   (I2)  ADD  x2, x0, x1   // x2 = x0 + x1
 *   (I4)  MUL  x4, x2, x3   // x4 = x2 * x3
 * In this reordered sequence, (I1) and (I3) can be issued in parallel because
 * they do not have dependencies on each other. If there is sufficient memory
 * bandwidth, (I1) and (I3) will complete concurrently in T0, reducing the total
 * execution time to nearly T0!
 *
 *
 * Performance Impact:
 * This pass can enhance performance by up to 20% for both Reduce and Trvial.
 * The improvement is often more pronounced when expressions involve complex ops
 * (e.g. div, exp and rsqrt) and when multiple schedule blocks exist within one
 * leaf block. The performance gain comes from that the NVCC tends to conserve
 * registers and employs a `lazy` approach to software pipelining. By applying
 * this pass, we force NVCC to use more registers and engage in more aggressive
 * software pipelining.
 *
 * However, there are also random cases where this pass may decrease
 * performance. The reason is unclear yet (perhaps because of suboptimal
 * unrolling and register overflow). We have used some strategies to avoid these
 * cases, such as limiting the maximum number of loads to rearrange and
 * forbidding certain patterns. While we cannot currently guarantee a consistent
 * improvement, our experiments indicate that the performance degradation is
 * within 5% in the worst case.
 *
 *
 * Limitations:
 * 1) The Select op is handled carefully, as for loads in Select's branches, we
 *    only rearrange those that appear on both branches.
 * 2) Indicrect loads (i.e. loads in indices) are not handled at all.
 * 3) If there are too many candidate loads to rearrange in a leaf block, we
 *    heuristically choose only 8 loads to rearrange.
 *
 *
 * Examples:
 * 1. Single Reduce schedule block:
 * =>
 *   for (k, 0, 32) {
 *     ScheduleBlock(var_2) {
 *       var_2[k] = var_2[k] + var_0[k] * var_1[k]
 *     }
 *   }
 * <=
 *   for (k, 0, 32) {
 *     var_0_local = var_0[k]
 *     var_1_local = var_1[k]
 *     ScheduleBlock(var_2) {
 *       var_2[k] = var_2[k] + var_0_local * var_1_local
 *     }
 *   }
 * Note:
 *   The reduce var itself (var_2[k]) is not rearranged.
 *
 * 2. Multiple Trivial schedule blocks:
 * =>
 *   for (k, 0, 32) {
 *     ScheduleBlock(var_3) {
 *       var_3[k] = var_0[k] + var_1[k]
 *     }
 *     ScheduleBlock(var_4) {
 *       var_4[k] = var_0[k] * var_2[k]
 *     }
 *   }
 * <=
 *   for (k, 0, 32) {
 *     var_0_local = var_0[k]
 *     var_1_local = var_1[k]
 *     var_2_local = var_2[k]
 *     ScheduleBlock(var_3) {
 *       var_3[k] = var_0_local + var_1_local
 *     }
 *     ScheduleBlock(var_4) {
 *       var_4[k] = var_0_local * var_2_local
 *     }
 *   }
 * Note:
 *   `var_0` is used twice but only loaded once.
 *
 * 3. Counter example:
 * =>
 *   for (k, 0, 32) {
 *     ScheduleBlock(var_3) {
 *       var_3[k] = (var_1[var_0[k]] > 0.0f) ? var_2[k] : 0.0f;
 *     }
 *     ScheduleBlock(var_4) {
 *       var_4[k] = var_3[k] * 2.0f
 *     }
 *   }
 * <=
 *   NO CHANGE!
 * Note:
 *   `var_1[var_0[k]]` has indirect indices, `var_2[k]` only appears in one
 *   branch of Select, `var_3[k]` in ScheduleBlock(var_4) has data dependency
 *   with ScheduleBlock(var_3); none of them can be rearranged.
 */
std::unique_ptr<FuncPass> CreateRearrangeLoadInstructionPass();

}  // namespace optim
}  // namespace cinn
