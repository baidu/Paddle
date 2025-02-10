// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {

class ReplaceCrossThreadReductionPass : public FuncPass {
 public:
  ReplaceCrossThreadReductionPass()
      : FuncPass("replace_cross_thread_reduction") {}

  LogicalResult Run(ir::LoweredFunc func) override;
};

/**
 * Replaces cross-thread reduction operations with more efficient external calls.
 *
 * This pass optimizes GPU reduction operations by replacing them with optimized
 * external function calls. It is applicable in scenarios where multiple GPU threads
 * need to perform reduction operations (like sum, max, min) across thread boundaries.
 *
 * The pass performs the following steps:
 * 1. Identifies reducible operations for optimization in cross-thread scenarios.
 * 2. Replaces the original reduction operation with an optimized external function call.
 * 3. Creates shared memory buffers for intermediate results.
 * 4. Transforms the reduction pattern based on the selected method (None/Warp/Block/Discrete).
 *
 * Supported reduction operations:
 * - Addition (ir::Add)
 * - Multiplication (ir::Mul)
 * - Maximum (ir::Max)
 * - Minimum (ir::Min)
 * - Logical AND (ir::And)
 * - Logical OR (ir::Or)
 *
 * Performance impact:
 * - Reduces thread synchronization overhead.
 * - Optimizes memory access patterns through shared memory usage.
 * - Enables efficient parallel reduction at different granularities (warp/block).
 * - Minimizes global memory access during reduction operations.
 *
 * Risks and limitations:
 * - Limited to reduction operations with thread-bound loops.
 * - Maximum thread block size constraint of 1024.
 * - Requires sufficient shared memory resources.
 * - May increase register pressure due to shared memory usage.
 *
 * Example:
 * Input IR:
 *   void reduce_sum_kernel(const float* data, float* total_sum, int n) {
 *     int tid = threadIdx.x;
 *     int i = blockIdx.x * blockDim.x + threadIdx.x;
 *     float my_sum = 0.0;
 *     while (i < n) {
 *       my_sum += data[i];
 *       i += blockDim.x * gridDim.x;
 *     }
 *     shared_sum[tid] = my_sum;
 *     __syncthreads();
 *     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
 *       if (tid % (2*s) == 0) {
 *         shared_sum[tid] += shared_sum[tid + s];
 *       }
 *       __syncthreads();
 *     }
 *     if (tid == 0) {
 *       total_sum[blockIdx.x] = shared_sum[0];
 *     }
 *   }
 *
 * Output IR:
 *   buffer shm32_float_reduce[32];
 *   sum = __cinn_cuda_reduce_sum(data, shm32_float_reduce, false);
 *
 * Not applicable scenarios:
 * 1. Non-reduction loop:
 *    for (i = 0; i < n; i++) {
 *      output[i] = input[i] + 1;
 *    }
 *    Reason: No reduction operation present, pass should not be applied.
 * 2. Large reduction with thread count > 1024:
 *    for (i = 0; i < 2048; i++) {
 *      sum += data[i];
 *    }
 *    Reason: Exceeds maximum supported thread block size.
 * 3. float sum = 0.0;
 *    float data[1024];
 *    __global__ void incorrectReduction() { sum += data[threadIdx.x];}
 *    Reason: Reduction axes are not bound to threads.
 */
void ReplaceCrossThreadReduction(ir::LoweredFunc fn);

}  // namespace optim
}  // namespace cinn
