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

#include <gtest/gtest.h>

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/optim/ir_simplify.h"
namespace cinn::optim {

TEST(CastSimplify, same_type) {
  Var n("n");
  Expr a = ir::Cast::Make(Int(32), n);
  LOG(INFO) << n->type();
  LOG(INFO) << a;
  SimplifyCast(&a);
  ASSERT_EQ(utils::GetStreamCnt(a), "n");
}

TEST(CastSimplify, Imm_int) {
  Expr a = ir::Cast::Make(Int(64), Expr(1));
  Expr c = ir::Cast::Make(Int(32), a);
  LOG(INFO) << c;
  SimplifyCast(&c);
  LOG(INFO) << c;
  ASSERT_EQ(utils::GetStreamCnt(c), "1");
  ASSERT_EQ(c.type(), Int(32));
}

TEST(CastSimplify, Imm_double) {
  Expr a = ir::Cast::Make(Float(64), Expr(2.33));
  Expr c = ir::Cast::Make(Int(32), a);
  LOG(INFO) << c;
  SimplifyCast(&c);
  LOG(INFO) << c;
  ASSERT_EQ(utils::GetStreamCnt(c), "2");
  ASSERT_EQ(c.type(), Int(32));
}

TEST(CastSimplify, Imm_uint) {
  Expr a = ir::Cast::Make(UInt(64), Expr(1));
  Expr c = ir::Cast::Make(UInt(32), a);
  LOG(INFO) << c;
  SimplifyCast(&c);
  LOG(INFO) << c;
  ASSERT_EQ(utils::GetStreamCnt(c), "1");
  ASSERT_EQ(c.type(), UInt(32));
}

}  // namespace cinn::optim
