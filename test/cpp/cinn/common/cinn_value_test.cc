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

#include "paddle/cinn/common/cinn_value.h"

#include <gtest/gtest.h>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn {
namespace common {

TEST(CINNValue, test) {
  {
    CINNValue value(32);
    ASSERT_EQ(int(value), 32);  // NOLINT
  }
  {
    CINNValue value(32.f);
    ASSERT_NEAR(float(value), 32.f, 1e-6);  // NOLINT
  }
}

TEST(CINNValue, buffer) {
  cinn_buffer_t* v = nullptr;
  CINNValue value(v);
  ASSERT_EQ((cinn_buffer_t*)value, nullptr);
}

TEST(CINNValue, Expr) {
  Expr a(1);

  {
    CINNValue value(a);
    ASSERT_TRUE(a == value);
  }

  {
    CINNValue copied = CINNValue(a);
    ASSERT_TRUE(copied == cinn::common::make_const(1));
  }
}

}  // namespace common
}  // namespace cinn
