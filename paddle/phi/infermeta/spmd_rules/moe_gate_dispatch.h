/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace distributed {

SpmdInfo MoEGateDispatchFwdInferSpmd(const DistMetaTensor& x,
                                     const DistMetaTensor& gate_logits,
                                     int64_t k,
                                     int64_t capacity,
                                     bool use_pad);
// out: "y", "combine_weights", "scatter_index", "expert_offset", "expert_id"

SpmdInfo MoEGateDispatchBwdInferSpmd(const DistMetaTensor& combine_weights,
                                     const DistMetaTensor& scatter_index,
                                     const DistMetaTensor& expert_id,
                                     const DistMetaTensor& grad_y,
                                     const DistMetaTensor& grad_combine_weights,
                                     int64_t k,
                                     int64_t capacity,
                                     bool use_pad);
// out: "x_grad", "gate_logits_grad"

}  // namespace distributed
}  // namespace phi
