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

#include <sstream>

#include "gtest/gtest.h"

#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn::common::test {

using namespace symbol;  // NOLINT

TEST(Convert, AddExpr) {
  List<DimExpr> num_lists{DimExpr(4), DimExpr(5), DimExpr("sym_0")};
  DimExpr dim_expr{Add<DimExpr>{num_lists}};
  ir::Expr src_expr = DimExprConverter().ConvertToIrExpr(dim_expr);

  ir::Expr expr1 =
      ir::Add::Make(ir::Expr(std::int64_t(4)), ir::Expr(std::int64_t(5)));
  ir::Expr dst_expr =
      ir::Add::Make(expr1,
                    ir::_Var_::Make(ir::Expr(static_cast<int64_t>(1)),
                                    ir::Expr(INT32_MAX),
                                    "sym_0",
                                    /* is_reduce  = */ false,
                                    /* is_symbolic_constant = */ true));
  ASSERT_TRUE(MathEqual(src_expr, dst_expr));
}

TEST(Convert, SubExpr) {
  DimExpr dim_expr = DimExpr(4) - DimExpr("sym_0");
  ir::Expr src_expr = DimExprConverter().ConvertToIrExpr(dim_expr);

  ir::Expr expr1 =
      ir::Sub::Make(ir::Expr(std::int64_t(0)),
                    ir::_Var_::Make(ir::Expr(static_cast<int64_t>(1)),
                                    ir::Expr(INT32_MAX),
                                    "sym_0",
                                    /* is_reduce  = */ false,
                                    /* is_symbolic_constant = */ true));
  ir::Expr dst_expr = ir::Add::Make(ir::Expr(std::int64_t(4)), expr1);
  ASSERT_TRUE(MathEqual(src_expr, dst_expr));
}

TEST(Convert, MulExpr) {
  List<DimExpr> num_lists{DimExpr(4), DimExpr(5), DimExpr("sym_0")};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};
  ir::Expr src_expr = DimExprConverter().ConvertToIrExpr(dim_expr);

  ir::Expr expr1 =
      ir::Mul::Make(ir::Expr(std::int64_t(4)), ir::Expr(std::int64_t(5)));
  ir::Expr dst_expr =
      ir::Mul::Make(expr1,
                    ir::_Var_::Make(ir::Expr(static_cast<int64_t>(1)),
                                    ir::Expr(INT32_MAX),
                                    "sym_0",
                                    /* is_reduce  = */ false,
                                    /* is_symbolic_constant = */ true));
  ASSERT_TRUE(MathEqual(src_expr, dst_expr));
}

TEST(Convert, MaxExpr) {
  List<DimExpr> num_lists{DimExpr(4), DimExpr(5), DimExpr("sym_0")};
  DimExpr dim_expr{Max<DimExpr>{num_lists}};
  ir::Expr src_expr = DimExprConverter().ConvertToIrExpr(dim_expr);

  std::ostringstream stream;
  stream << src_expr;
  ASSERT_EQ(stream.str(), "cinn_max(cinn_max(4ll, 5ll), sym_0)");
}

TEST(Convert, MinExpr) {
  List<DimExpr> num_lists{DimExpr(4), DimExpr(5), DimExpr("sym_0")};
  DimExpr dim_expr{Min<DimExpr>{num_lists}};
  ir::Expr src_expr = DimExprConverter().ConvertToIrExpr(dim_expr);

  std::ostringstream stream;
  stream << src_expr;
  ASSERT_EQ(stream.str(), "cinn_min(cinn_min(4ll, 5ll), sym_0)");
}

}  // namespace cinn::common::test
