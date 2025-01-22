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
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {
/**
 * Optimizes GPU expressions by transforming variables, buffer indices, and
 * memory access patterns for efficient GPU execution.
 *
 * This pass is applicable in scenarios where GPU-specific expressions need to
 * be optimized for execution on GPU backends. This pass is essential in
 * compiler pipelines that generate or transform GPU code, ensuring that
 * variables and memory accesses are correctly mapped and optimized for GPU
 * architecture.
 *
 * When applied, this pass performs a series of transformations on the IR to
 * optimize expressions for GPU execution:
 *   1) Variable and Loop Transformation
 *   2) Buffer and Memory Access Optimization
 *   3) Expression Simplification and Type Casting
 */
void OptimizeExprGPU(ir::stmt::BlockRef func_body);
// void OptimizeExprGPU(ir::Expr* func_body);

/**
 * Remove the GPU block/thread-bound For loops, add IfThenElse guards if needed.
 *
 * It's usually safe to remove bound loops, because when launching the kernel,
 * we are expected to choose dim sizes that match the extents of these loops.
 * However, there are cases where we cannot simply remove a loop, but need to
 * add an IfThenElse as guard:
 *   1) if the loop doesn't start from 0.
 *   2) if we cannot prove that the loop's extent is always equal to or greater
 *      than the corresponding dim size.
 *
 * Example 1:
 *   # assume blockDim.x == 256
 *   thread_bind[threadIdx.x] for (k, 0, 256):
 *     ScheduleBlock(A)
 * =>
 *   ScheduleBlock(A)
 *
 * Example 2:
 *   # assume gridDim.x == 8
 *   thread_bind[blockIdx.x] for (k, 2, min(S0, 8)):
 *     ScheduleBlock(A)
 * =>
 *   if (blockIdx.x >= 2 && blockIdx.x < min(S0, 8)):
 *     ScheduleBlock(A)
 *
 * @param fn The LoweredFunc to process.
 */
class RemoveGpuForLoopsPass : public FuncPass {
 public:
  RemoveGpuForLoopsPass() : FuncPass("remove_gpu_for_loops") {}

  LogicalResult Run(ir::LoweredFunc fn) override;
};
std::unique_ptr<FuncPass> CreateRemoveGpuForLoopsPass();

/**
 * Removes conditional wrappers around CUDA thread synchronization calls.
 *
 * This pass is applicable in scenarios where CUDA synchronization functions,
 * such as `cuda_sync_threads`, are enclosed within conditional statements
 * (`IfThenElse`) that check if a certain variable equals zero. Such scenarios
 * are common in auto-generated code or optimized code paths where
 * synchronization is conditionally performed based on loop iterations or
 * specific flags.
 *
 * When applied, this pass traverses the Intermediate Representation (IR) of a
 * lowered function to identify `IfThenElse` nodes that contain
 * `cuda_sync_threads` calls with conditions checking for equality to zero. For
 * each identified conditional synchronization:
 *   1) It verifies that the `IfThenElse` condition is an equality (`EQ`)
 *      comparison where the second operand is zero.
 *   2) It replaces the entire `IfThenElse` node with the `cuda_sync_threads`
 *      call, effectively removing the conditional check.
 *
 * Example 1:
 *   if (xxxx == 0) { __syncthreads(); }
 * =>
 *   __syncthreads();
 *
 * Example 2:
 *   if (xxxx > 0) { __syncthreads(); }
 * =>
 *   if (xxxx > 0) { __syncthreads(); }
 */
class CudaSyncThreadsDropIfThenElsePass : public FuncPass {
 public:
  CudaSyncThreadsDropIfThenElsePass()
      : FuncPass("cuda_sync_threads_drop_ifthenelse") {}

  LogicalResult Run(ir::LoweredFunc fn) override;
};
std::unique_ptr<FuncPass> CreateCudaSyncThreadsDropIfThenElsePass();

}  // namespace optim
}  // namespace cinn
