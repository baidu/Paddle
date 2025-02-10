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

#pragma once
#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {

/**
 * UpdateBufferAxisPass optimizes buffer access by formalizing indices and
 * replacing redundant accesses with zero.
 *
 * This pass is used in `OptimizeExprGpu` and is applicable in scenarios
 * where buffer accesses in shared or local GPU memory have consistent index
 * expressions across the same axis. In such cases, the pass analyzes the Expr
 * AST to determine if these consistent indices imply that less memory is
 * needed. By setting these redundant indices to zero, the pass can help
 * minimize memory usage.
 *
 * When applied, this pass analyzes buffer access patterns and identifies
 * indices that are consistently accessed with the same expression across the
 * same axis in shared or local GPU memory. It then replaces these indices with
 * zero, which can lead to reduced memory allocation requirements and
 * streamlined memory usage.
 *
 * Performance impact: This pass addresses memory optimization in GPU
 * environments by potentially reducing memory allocation and improving access
 * efficiency, which can enhance overall performance.
 *
 * Examples:
 * 1. Consistent Index Access in GPU Shared Memory:
 *    Input IR:
 *      `A[i * 3][j] = ...`
 *      `... = A[k][j]`
 *    Output IR:
 *      `A[i * 3][0] = ...`
 *      `... = A[k][0]`
 *
 * 2. Single Dimension Access Simplified:
 *    Input IR:
 *      B[i * n + j] = ...
 *      ... = B[k * n + j]
 *    Output IR:
 *      B[i * n + 0] = ...
 *      ... = B[k * n + 0]
 */

class UpdateBufferAxisPass : public BlockPass {
 public:
  UpdateBufferAxisPass() : BlockPass("update_buffer_axis_pass") {}

  LogicalResult Run(ir::stmt::BlockRef) override;
};

std::unique_ptr<BlockPass> CreateUpdateBufferAxisPass();

}  // namespace optim
}  // namespace cinn
