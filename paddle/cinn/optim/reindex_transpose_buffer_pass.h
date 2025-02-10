// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

namespace cinn {
namespace optim {

/**
 * Re-index the load and stores of transpose buffers using the swizzling
 * pattern, and replace all transpose buffers with one union buffer.
 *
 * This pass applies two changes to the transpose buffers in the function to
 * improve the efficiency of transpose:
 *
 * 1. Swizzle the load & store indices of transpose buffers to prevent bank
 *    conflict in shared memory. Swizzling is a symmetric index pattern for
 *    a pair of store and load, in the form:
 *       Store :  shm[y, x ^ y]
 *       Load  :  shm[x, y ^ x]
 *    So that we transpose shm[y][x] to shm[x][y], while neither the store nor
 *    the load cause bank conflict (because both `x ^ y` and `y ^ x` produce 32
 *    unique values within a warp of 32 threads).
 *
 * 2. Replace multiple individual transpose buffers for each transpose with a
 *    single union buffer to save shared memory. This is valid because transpose
 *    buffers are not used in the same time.
 *
 *
 * Example:
 * Suppose we are transposing `var_0` to `var_1` using buffer `var_0_shm`.
 * Input IR:
 *    for (k, 0, 4):
 *      for (thread.y, 0, 8):
 *        for (thread.x, 0, 32):
 *          var_0_shm[k * 8 + thread.y, thread.x] = var_0[...]
 *    for (k, 0, 4):
 *      for (thread.y, 0, 8):
 *        for (thread.x, 0, 32):
 *          var_1[...] = var_0_shm[thread.x, k * 8 + thread.y]
 * Output IR:
 *    for (k, 0, 4):
 *      for (thread.y, 0, 8):
 *        for (thread.x, 0, 32):
 *          transpose_union_shm[
 *              k * 8 + thread.y, thread.x ^ (k * 8 + thread.y)] = var_0[...]
 *    for (k, 0, 4):
 *      for (thread.y, 0, 8):
 *        for (thread.x, 0, 32):
 *          var_1[...] = transpose_union_shm[
 *              thread.x, (k * 8 + thread.y) ^ thread.x]
 */
class ReindexTransposeBufferPass : public FuncPass {
 public:
  ReindexTransposeBufferPass() : FuncPass("reindex_transpose_buffer") {}
  LogicalResult Run(ir::LoweredFunc func) override;
};

std::unique_ptr<FuncPass> CreateReindexTransposeBufferPass();

}  // namespace optim
}  // namespace cinn
