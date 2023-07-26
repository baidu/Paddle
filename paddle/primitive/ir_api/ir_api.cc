// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/primitive/ir_api/ir_api.h"
#include "paddle/fluid/ir/dialect/pd_op.h"

namespace ir {
namespace api {
std::vector<std::vector<ir::OpResult>> tanh_grad(ir::OpResult out,
                                                 ir::OpResult grad_out) {
  std::vector<std::vector<ir::OpResult>> res;

  return res;
}
}  // namespace api
}  // namespace ir
