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
#include "paddle/phi/infermeta/spmd_rules/dim_trans.h"

namespace phi {
namespace distributed {

std::vector<int64_t> InferTargetShape(const std::vector<int64_t>& shape,
                                      int64_t len);

std::vector<DimTrans*> MakeReshapeDimTrans(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& tgt_shape);

SpmdInfo ReshapeInferSpmd(const DistMetaTensor& x,
                          const std::vector<int64_t>& shape);

SpmdInfo ReshapeInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& out,
                                 const std::vector<int64_t>& shape);
}  // namespace distributed
}  // namespace phi
