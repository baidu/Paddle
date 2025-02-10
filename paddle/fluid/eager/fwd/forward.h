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

#pragma once

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/phi/api/all.h"
#include "paddle/utils/test_macros.h"

namespace egr {

// forward_ad namespace is used to implement forward automatic differentiation.
// namespace forward_ad {

TEST_API uint64_t enter_dual_level();

TEST_API void exit_dual_level(uint64_t level);

TEST_API paddle::Tensor make_dual(const paddle::Tensor& primal,
                                  const paddle::Tensor& tangent,
                                  int64_t level);

TEST_API std::tuple<paddle::Tensor, paddle::Tensor> unpack_dual(
    const paddle::Tensor& tensor, int64_t level);

// } // namespace forward_ad

}  // namespace egr
