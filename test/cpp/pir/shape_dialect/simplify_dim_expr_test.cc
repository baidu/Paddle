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

#include <atomic>
#include "gtest/gtest.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace symbol::test {

namespace {

DimExpr BD(const DimExpr& lhs, const DimExpr& rhs) {
  return Broadcast<DimExpr>{{lhs, rhs}};
}

DimExpr MakeSymbolic() {
  static std::atomic<int64_t> cnt(0);
  return DimExpr{std::to_string(cnt++)};
}

DimExpr MakeConstant(std::int64_t value) { return DimExpr{value}; }

}  // namespace

TEST(DimExpr, flatten_bd) {
  DimExpr sym0 = MakeSymbolic();
  DimExpr sym1 = MakeSymbolic();
  DimExpr sym2 = MakeSymbolic();
  DimExpr origin = BD(BD(sym0, sym1), sym2);
  DimExpr expected = Broadcast<DimExpr>{{sym0, sym1, sym2}};
  ASSERT_EQ(SimplifyDimExpr(origin), expected);
}

TEST(Simplify, NumberAdd) {
  List<DimExpr> num_lists{DimExpr(5), Negative<DimExpr>(5)};
  DimExpr dim_expr{Add<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, DoubleNegative) {
  DimExpr inner_expr{Negative<DimExpr>(DimExpr{1})};
  DimExpr expr{Negative<DimExpr>(inner_expr)};

  DimExpr simplified_dim_expr = SimplifyDimExpr(expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, UnitNegative) {
  DimExpr unit{Negative<DimExpr>{DimExpr{0}}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(unit);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, NumberNaiveMul) {
  List<DimExpr> num_lists{DimExpr(5), DimExpr(5)};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 25);
}

TEST(Simplify, NumberNaiveDiv) {
  DimExpr dim_expr{Div<DimExpr>{DimExpr(5), DimExpr(5)}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestNumberAddDiv) {
  DimExpr div_expr{Div<DimExpr>{DimExpr(5), DimExpr(5)}};
  List<DimExpr> sum_lists{DimExpr(0), div_expr};
  DimExpr dim_expr{Add<DimExpr>{sum_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestNumberMulAdd) {
  List<DimExpr> num_lists{DimExpr(5), Negative<DimExpr>(5)};
  List<DimExpr> product_lists{DimExpr(5), Add<DimExpr>{num_lists}};
  DimExpr dim_expr{Mul<DimExpr>{product_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, SymbolicMul) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{DimExpr(1), sym};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::string>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, SymbolicDiv) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{sym, DimExpr(1)};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::string>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, SymbolicMulUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{sym, DimExpr(1)};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, SymbolicDivUnit) {
  DimExpr sym = MakeSymbolic();
  DimExpr dim_expr{
      Div<DimExpr>{Mul<DimExpr>{List<DimExpr>{DimExpr(2), sym}}, sym}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 2);
}

TEST(Simplify, NestSymbolicMulAddUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> sum_lists{DimExpr(6), Negative<DimExpr>{DimExpr(5)}};
  List<DimExpr> product_lists = List<DimExpr>{Add<DimExpr>{sum_lists}, sym};
  DimExpr dim_expr{Mul<DimExpr>{product_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::string>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, NestSymbolicDivAddUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> sum_lists{DimExpr(6), Negative<DimExpr>{DimExpr(5)}};
  DimExpr dim_expr{Div<DimExpr>{sym, Add<DimExpr>{sum_lists}}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::string>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, ConstantMaxMin) {
  List<DimExpr> max_lists{DimExpr(4), DimExpr(6)};
  DimExpr dim_expr1{Max<DimExpr>{max_lists}};

  DimExpr simplified_dim_expr1 = SimplifyDimExpr(dim_expr1);
  ASSERT_TRUE((simplified_dim_expr1.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr1.Get<std::int64_t>()), 6);

  List<DimExpr> min_lists{DimExpr(2), DimExpr(3)};
  DimExpr dim_expr2{Min<DimExpr>{min_lists}};

  DimExpr simplified_dim_expr2 = SimplifyDimExpr(dim_expr2);
  ASSERT_TRUE((simplified_dim_expr2.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr2.Get<std::int64_t>()), 2);
}

TEST(Simplify, FoldBroadcast) {
  DimExpr sym0{"S0"};
  DimExpr sym1{"S1"};
  DimExpr mul{Mul<DimExpr>{{sym0, sym1}}};
  DimExpr broadcast0{Broadcast<DimExpr>{{mul, sym0}}};
  DimExpr broadcast1{Broadcast<DimExpr>{{sym1, mul}}};
  DimExpr simplify_broadcast0 = SimplifyDimExpr(broadcast0);
  DimExpr simplify_broadcast1 = SimplifyDimExpr(broadcast1);

  DimExpr add{Add<DimExpr>{{sym0, sym1}}};
  DimExpr broadcast2{Broadcast<DimExpr>{{add, sym0}}};
  DimExpr broadcast3{Broadcast<DimExpr>{{sym1, add}}};
  DimExpr simplify_broadcast2 = SimplifyDimExpr(broadcast2);
  DimExpr simplify_broadcast3 = SimplifyDimExpr(broadcast3);

  ASSERT_TRUE(simplify_broadcast0 == mul);
  ASSERT_TRUE(simplify_broadcast1 == mul);
  ASSERT_TRUE(simplify_broadcast2 == add);
  ASSERT_TRUE(simplify_broadcast3 == add);
}

TEST(Simplify, FoldRedundantBroadcast) {
  DimExpr S0{"S0"};
  DimExpr S1{"S1"};
  DimExpr bc{Broadcast<DimExpr>{{S0, S0, S1, S1}}};
  DimExpr simplify_bc = SimplifyDimExpr(bc);
  ASSERT_TRUE((simplify_bc == Broadcast<DimExpr>{{S0, S1}}));
}

TEST(Simplify, Case1) {
  // Div(Mul(Div(Mul(Broadcast(S11, S8), Broadcast(S10, S13, S4, S7),
  // Broadcast(S12, S3, S6, S9)), S0)), 16), 49)
  DimExpr S11{"S11"};
  DimExpr S8{"S8"};
  DimExpr mul_op1 = Broadcast<DimExpr>{{S11, S8}};

  DimExpr S10{"S10"};
  DimExpr S13{"S13"};
  DimExpr S4{"S4"};
  DimExpr S7{"S7"};
  DimExpr mul_op2 = Broadcast<DimExpr>{{S10, S13, S4, S7}};

  DimExpr S12{"S12"};
  DimExpr S3{"S3"};
  DimExpr S6{"S6"};
  DimExpr S9{"S9"};
  DimExpr mul_op3 = Broadcast<DimExpr>{{S12, S3, S6, S9}};

  DimExpr S0{"S0"};
  DimExpr mul_op4 =
      Div<DimExpr>{Mul<DimExpr>{List<DimExpr>{mul_op1, mul_op2, mul_op3}}, S0};

  DimExpr dim_expr = Div<DimExpr>{
      Mul<DimExpr>{List<DimExpr>{mul_op4, DimExpr{16}}}, DimExpr(49)};

  ASSERT_TRUE((SimplifyDimExpr(dim_expr)) == dim_expr);
}

TEST(Simplify, Case2) {
  // Div(Mul(S2, S3, 8, 7, 7), Mul( Div(S0, 7), Div(S1, 7), 8, 7, 7，1, 2))
  DimExpr S2{"S2"};
  DimExpr S3{"S3"};
  DimExpr mul_op1 =
      Mul<DimExpr>{List<DimExpr>{S2, S3, DimExpr(8), DimExpr(7), DimExpr(7)}};

  DimExpr S0{"S0"};
  DimExpr S1{"S1"};
  DimExpr mul_op2 = Mul<DimExpr>{List<DimExpr>{Div<DimExpr>{S0, DimExpr(7)},
                                               Div<DimExpr>{S1, DimExpr(7)},
                                               DimExpr(8),
                                               DimExpr(7),
                                               DimExpr(7),
                                               DimExpr(1),
                                               DimExpr(2)}};
  DimExpr dim_expr{Div<DimExpr>{mul_op1, mul_op2}};

  DimExpr expected = Div<DimExpr>{
      Mul<DimExpr>{List<DimExpr>{S2, S3}},
      Mul<DimExpr>{List<DimExpr>{
          Div<DimExpr>{S0, DimExpr(7)}, Div<DimExpr>{S1, DimExpr(7)}, 2}}};
  ASSERT_TRUE((SimplifyDimExpr(dim_expr)) == expected);
}

}  // namespace symbol::test
