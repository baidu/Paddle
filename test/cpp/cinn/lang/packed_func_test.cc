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

#include "paddle/cinn/lang/packed_func.h"

#include <gtest/gtest.h>

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace lang {

TEST(Function, test) {
  PackedFunc::body_t func_body = [](Args args, RetValue* ret) {
    int a = args[0];
    int b = args[1];
    *ret = (a + b);
  };
  PackedFunc func(func_body);

  int c = func(1, 2);
  LOG(INFO) << "c " << c;
}

TEST(Function, test1) {
  PackedFunc::body_t body = [](Args args, RetValue* ret) {
    auto* msg = static_cast<const char*>(args[0]);
    (*ret) = msg;
  };

  PackedFunc func(body);
  const char* msg = "hello world";
  char* c = func(msg);
  LOG(INFO) << static_cast<char*>(c);
}

TEST(Function, Expr) {
  PackedFunc::body_t body = [](Args args, RetValue* ret) {
    Expr a = args[0];
    Expr b = args[1];

    ASSERT_EQ(a->__ref_count__.val(), 4);
    ASSERT_EQ(b->__ref_count__.val(), 4);

    Expr c = a + b;
    (*ret) = CINNValue(c);
  };

  PackedFunc func(body);

  Expr a(1);
  Expr b(2);
  ASSERT_EQ(a->__ref_count__.val(), 1);
  ASSERT_EQ(b->__ref_count__.val(), 1);

  Expr ret = func(a, b);

  ASSERT_EQ(utils::GetStreamCnt(ret), "(1 + 2)");
}

TEST(Function, ReturnMultiValue) {
  PackedFunc::body_t body = [](Args args, RetValue* ret) {
    int a = args[0];
    int b = args[1];
    int c = a + b;
    int d = a - b;

    *ret = cinn::common::CINNValuePack{
        {cinn::common::CINNValue(c), cinn::common::CINNValue(d)}};
  };

  PackedFunc func(body);

  cinn::common::CINNValuePack ret = func(1, 2);
  int c = ret[0];
  int d = ret[1];

  EXPECT_EQ(c, 3);
  EXPECT_EQ(d, -1);
}

}  // namespace lang
}  // namespace cinn
