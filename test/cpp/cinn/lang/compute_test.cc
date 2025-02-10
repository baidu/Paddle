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

#include "paddle/cinn/lang/compute.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace lang {

TEST(Call, basic) {
  Expr M(100);

  Placeholder<float> x("x", {M, Expr(10)});
  Placeholder<float> y("y", {M, Expr(10)});

  std::vector<ReturnType> return_types(
      {{Float(32), std::vector<Expr>{{M, Expr(20)}}, "C"}});
  auto tensors = CallLowered("lowered_fun0", {Expr(x), Expr(y)}, return_types);
}

}  // namespace lang
}  // namespace cinn
