/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {
/**
 * A **hack** rule with a strong assumption that the first dimension of
 * all the input and output tensors is the batch dimension (broadcast
 * dimension), therefore, if any tensor's first dimension is sharded, the
 * sharding would be propagating to all the other tensors (for tensor first
 * dimension). All the other axes of tensors would be set as unshard (-1).
 *
 *
 * This rule is used to support emerging op for hybrid parallelism quickly, and
 * once there is a specific rule for that op,  we should remove that op from
 * this rule.
 *
 * Vector of input tensors and output tensors used as arguments (for both
 * inferfw & inferbw) to support any kind of op.
 *
 */
SpmdInfo DefaultDataParallelInferSpmd(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs);

SpmdInfo DefaultDataParallelInferSpmdReverse(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs);

// For phi api
template <typename... Args>
SpmdInfo VariadicDefaultDataParallelInferSpmd(const Args&... args) {
  return detail::VariadicSpmdRuleArgumentParser<DefaultDataParallelInferSpmd>()
      .apply(args...)
      .InferForward();
}

template <typename... Args>
SpmdInfo VariadicDefaultDataParallelInferSpmdReverse(const Args&... args) {
  return detail::VariadicSpmdRuleArgumentParser<
             DefaultDataParallelInferSpmdReverse>()
      .apply(args...)
      .InferBackward();
}

}  // namespace distributed
}  // namespace phi
