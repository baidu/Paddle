// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/fwd/forward.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/fwd/forward_grad.h"

namespace egr {

// forward_ad namespace is used to implement forward automatic differentiation.
// namespace forward_ad {

uint64_t enter_dual_level() {
  return forward_ad::ForwardADLevel::get_next_idx();
}

void exit_dual_level(uint64_t level) {
  forward_ad::ForwardADLevel::release_idx(level);
}

paddle::Tensor make_dual(const paddle::Tensor& primal,
                         const paddle::Tensor& tangent,
                         int64_t level) {
  PADDLE_ENFORCE_EQ(!primal._fw_grad(level).defined(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Making a dual Tensor based on a Tensor that "
                        "already has a forward gradient at the same level ",
                        level,
                        " is not supported."));

  auto dual_tensor = view_shape_ad_func(primal, primal.shape());
  dual_tensor._set_fw_grad(
      tangent, /*dual_tensor,*/ level, /* is_inplace_op */ false);
  return dual_tensor;
}

std::tuple<paddle::Tensor, paddle::Tensor> unpack_dual(
    const paddle::Tensor& tensor, int64_t level) {
  return std::tuple<paddle::Tensor, paddle::Tensor>(tensor,
                                                    tensor._fw_grad(level));
}

// } // namespace forward_ad

}  // namespace egr
