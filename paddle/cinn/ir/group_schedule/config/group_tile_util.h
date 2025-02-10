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

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace hlir::framework::pir {
struct GroupVectorizeInfo;
}  // namespace hlir::framework::pir

using hlir::framework::pir::GroupVectorizeInfo;

namespace ir {

/**
 * Get the strides of loops according to the largest input of Reduce.
 *
 * For example, in the following compute body of Reduce:
 *   for (i, 0, 8):
 *     for (j, 0, 16):
 *       for (k, 0, 24):
 *         for (m, 0, 32):
 *           var[i] = var[i] + var_1[k, i, j, m] * var_2[i, k, j]
 *
 * There are two inputs, whose sizes are:
 *    var_1[k, i, j, m]: 24*8*16*32 = 98304
 *    var_2[i, k, j]: 8*24*16 = 3072
 * We can see that the largest input is var_1[k, i, j, m].
 *
 * Therefore, the strides of loops (i, j, k and m) according to the largest
 * input (var_1[k, i, j, m]) are:
 *    i:   16*32 = 512
 *    j:      32 = 32
 *    k: 8*16*32 = 4096
 *    m:       1 = 1
 *
 * Limitations:
 * 1) If there are multiple inputs of the same size, we simply choose the first
 *    visited input.
 * 2) If the input's size contains symbols (e.g. S0, S1), we just replace each
 *    symbol with 32 so as to get a constant stride.
 *
 * @param reduce_compute_body The root For node of Reduce.
 */
std::vector<int64_t> GetLoopStrides(const ir::Expr& reduce_compute_body);

// Check whether we can apply grid reduce in this group.
// We can apply grid reduce if there is no reduce-then-broadcast dependency
// in this group.
bool GetCanApplyGridReduce(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<int64_t>& reduce_axis);

// Check whether we can apply vectorize in this group.
GroupVectorizeInfo GetCanApplyVectorize(
    const std::vector<ir::Expr>& op_compute_bodies,
    const std::unordered_set<std::string>& group_args);

}  // namespace ir
}  // namespace cinn
