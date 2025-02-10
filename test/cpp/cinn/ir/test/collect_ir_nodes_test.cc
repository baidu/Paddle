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

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"

namespace cinn {
namespace ir {
namespace ir_utils {

TEST(CollectIRNodes, basic0) {
  Expr C = Expr(1) + 2;

  auto exprs =
      CollectIRNodes(C, [](const Expr* x) { return x->As<ir::Add>(); });
  ASSERT_EQ(exprs.size(), 1UL);

  auto ints =
      CollectIRNodes(C, [](const Expr* x) { return x->As<ir::IntImm>(); });
  ASSERT_EQ(ints.size(), 2UL);
}

TEST(CollectIRNodes, basic) {
  Expr M(100);
  Expr N(200);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  ast_gen_ius::TensorGroup tensor_group({C});

  auto fn = LowerToAst("fn", {A, B, C}, &tensor_group);

  LOG(INFO) << "fn:\n" << fn;

  Expr expr_func_body = ConvertStmtBlockToExprBlock(fn->body_block);

  auto tensors = CollectIRNodes(expr_func_body,
                                [](const Expr* x) { return x->as_tensor(); });
  ASSERT_EQ(tensors.size(), 3UL);

  LOG(INFO) << "fn.body:\n" << expr_func_body;
  auto tensors2 = CollectIRNodes(expr_func_body,
                                 [](const Expr* x) { return x->as_tensor(); });
  auto exprs = CollectIRNodes(expr_func_body, [](const Expr* x) { return x; });
}
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
