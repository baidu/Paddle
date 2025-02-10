// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/integer_set.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace common {

class TestSymbolicExprAnalyzer : public ::testing::Test {
 public:
  void SetUp() override {
    i = ir::Var(ir::Expr(0), ir::Expr(7), "i");
    j = ir::Var(ir::Expr(0), ir::Expr(15), "j");
    var_intervals = {
        {"i", CasInterval(i->lower_bound, i->upper_bound)},
        {"j", CasInterval(j->lower_bound, j->upper_bound)},
    };
  }

  ir::Var i;
  ir::Var j;
  cas_intervals_t var_intervals;
  SymbolicExprAnalyzer analyzer{var_intervals};
};

TEST_F(TestSymbolicExprAnalyzer, bound) {
  ir::Expr e1 = i + j;
  EXPECT_EQ(analyzer.LowerBound(e1), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e1), ir::Expr(22));

  ir::Expr e2 = 16 * i + j;
  EXPECT_EQ(analyzer.LowerBound(e2), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e2), ir::Expr(127));

  ir::Expr e3 = 16 * i + j + 1;
  EXPECT_EQ(analyzer.LowerBound(e3), ir::Expr(1));
  EXPECT_EQ(analyzer.UpperBound(e3), ir::Expr(128));

  ir::Expr e4 = (16 * i + j) / 16;
  EXPECT_EQ(analyzer.LowerBound(e4), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e4), ir::Expr(7));

  ir::Expr e5 = (16 * i + j) % 16;
  EXPECT_EQ(analyzer.LowerBound(e5), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e5), ir::Expr(15));

  ir::Expr e6 = i - j;
  EXPECT_EQ(analyzer.LowerBound(e6), ir::Expr(-15));
  EXPECT_EQ(analyzer.UpperBound(e6), ir::Expr(7));

  ir::Expr e7 = 0 - i - j;
  EXPECT_EQ(analyzer.LowerBound(e7), ir::Expr(-22));
  EXPECT_EQ(analyzer.UpperBound(e7), ir::Expr(0));

  ir::Expr e8 = -1 * i - j;
  EXPECT_EQ(analyzer.LowerBound(e8), ir::Expr(-22));
  EXPECT_EQ(analyzer.UpperBound(e8), ir::Expr(0));
}

TEST_F(TestSymbolicExprAnalyzer, compare) {
  // case 1
  ir::Expr e1 = 4 * i + 2 * j;
  ir::Expr e2 = 2 * i + j;

  EXPECT_TRUE(analyzer.ProveEQ(e1, e1).value() &&
              analyzer.Prove(ir::EQ::Make(e1, e1)).value());
  EXPECT_FALSE(analyzer.ProveEQ(e1, e2).has_value() ||
               analyzer.Prove(ir::EQ::Make(e1, e2)).has_value());
  EXPECT_FALSE(analyzer.ProveNE(e1, e1).value() &&
               analyzer.Prove(ir::NE::Make(e1, e1)).value());
  EXPECT_FALSE(analyzer.ProveNE(e1, e2).has_value() ||
               analyzer.Prove(ir::NE::Make(e1, e2)).has_value());

  EXPECT_TRUE(analyzer.ProveGE(e1, e2).value() &&
              analyzer.Prove(e1 >= e2).value());
  EXPECT_FALSE(analyzer.ProveGE(e2, e1).has_value() ||
               analyzer.Prove(e2 >= e1).has_value());
  EXPECT_TRUE(analyzer.ProveLE(e2, e1).value() &&
              analyzer.Prove(e2 <= e1).value());
  EXPECT_FALSE(analyzer.ProveLE(e1, e2).has_value() ||
               analyzer.Prove(e1 <= e2).has_value());

  EXPECT_FALSE(analyzer.ProveGT(e1, e2).has_value() ||
               analyzer.Prove(e1 > e2).has_value());
  EXPECT_FALSE(analyzer.ProveGT(e2, e1).value() &&
               analyzer.Prove(e2 > e1).value());
  EXPECT_FALSE(analyzer.ProveLT(e2, e1).has_value() ||
               analyzer.Prove(e2 < e1).has_value());
  EXPECT_FALSE(analyzer.ProveLT(e1, e2).value() &&
               analyzer.Prove(e1 < e2).value());

  // case 2
  ir::Expr e3 = i + j + 1;
  ir::Expr e4 = i + j;

  EXPECT_TRUE(analyzer.ProveEQ(e3, e3).value() &&
              analyzer.Prove(ir::EQ::Make(e3, e3)).value());
  EXPECT_FALSE(analyzer.ProveEQ(e3, e4).value() &&
               analyzer.Prove(ir::EQ::Make(e3, e4)).value());
  EXPECT_TRUE(analyzer.ProveNE(e3, e4).value() &&
              analyzer.Prove(ir::NE::Make(e3, e4)).value());
  EXPECT_FALSE(analyzer.ProveNE(e4, e4).value() &&
               analyzer.Prove(ir::NE::Make(e4, e4)).value());

  EXPECT_TRUE(analyzer.ProveGE(e3, e4).value() &&
              analyzer.Prove(e3 >= e4).value());
  EXPECT_FALSE(analyzer.ProveGE(e4, e3).value() &&
               analyzer.Prove(e4 >= e3).value());
  EXPECT_TRUE(analyzer.ProveLE(e4, e3).value() &&
              analyzer.Prove(e4 <= e3).value());
  EXPECT_FALSE(analyzer.ProveLE(e3, e4).value() &&
               analyzer.Prove(e3 <= e4).value());

  EXPECT_TRUE(analyzer.ProveGT(e3, e4).value() &&
              analyzer.Prove(e3 > e4).value());
  EXPECT_FALSE(analyzer.ProveGT(e4, e3).value() &&
               analyzer.Prove(e4 > e3).value());
  EXPECT_TRUE(analyzer.ProveLT(e4, e3).value() &&
              analyzer.Prove(e4 < e3).value());
  EXPECT_FALSE(analyzer.ProveLT(e3, e4).value() &&
               analyzer.Prove(e3 < e4).value());
}

TEST_F(TestSymbolicExprAnalyzer, Divisible) {
  auto x = ir::Var(ir::Expr(1), ir::Expr(7), "x");
  auto y = ir::Var(ir::Expr(1), ir::Expr(15), "y");
  auto S = ir::Var(ir::Expr(16), ir::Expr(256), "S");

  cas_intervals_t divisible_var_intervals = {
      {"x", CasInterval(x->lower_bound, x->upper_bound)},
      {"y", CasInterval(y->lower_bound, y->upper_bound)},
      {"S", CasInterval(S->lower_bound, S->upper_bound)},
  };
  SymbolicExprAnalyzer divisible_analyzer{divisible_var_intervals};

  // case 1
  ir::Expr e1 = 4 * x + 2 * y * x;
  ir::Expr e2 = x;
  ir::Expr e3 = y;

  EXPECT_TRUE(divisible_analyzer.ProveDivisible(e1, e2).value_or(false));
  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e1, e3).value_or(false));

  // case 2
  ir::Expr e4 = y + y * x + 4 * y - x * y;

  EXPECT_TRUE(divisible_analyzer.ProveDivisible(e4, e3).value_or(false));
  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e4, e2).value_or(false));

  // case 3
  ir::Expr e5 = x / y + x + y;

  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e5, e3).value_or(false));
  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e5, e2).value_or(false));

  // case 4
  ir::Expr e6 = S * x / 4 + x * y;

  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e6, e2).value_or(false));
  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e6, e3).value_or(false));

  ir::Expr e7 = 16 * x / 4 + x * y;

  EXPECT_TRUE(divisible_analyzer.ProveDivisible(e7, e2).value_or(false));
  EXPECT_FALSE(divisible_analyzer.ProveDivisible(e7, e3).value_or(false));
}

TEST(SingleIntervalIntSet, constant) {
  SingleIntervalIntSet empty_set(ir::Expr(0), ir::Expr(-1));
  SingleIntervalIntSet all_set(SymbolicExprLimit::negative_inf,
                               SymbolicExprLimit::positive_inf);
  SingleIntervalIntSet single_point(ir::Expr(0), ir::Expr(0));
  SingleIntervalIntSet interval_0_2_set(ir::Expr(0), ir::Expr(2));
  SingleIntervalIntSet interval_0_4_set(ir::Expr(0), ir::Expr(4));
  SingleIntervalIntSet interval_2_6_set(ir::Expr(2), ir::Expr(6));
  SingleIntervalIntSet interval_8_9_set(ir::Expr(8), ir::Expr(9));

  EXPECT_TRUE(empty_set.ProveEmpty().value());
  EXPECT_FALSE(empty_set.ProveAll().value());
  EXPECT_FALSE(all_set.ProveEmpty().value());
  EXPECT_TRUE(all_set.ProveAll().value());
  EXPECT_TRUE(single_point.ProvePoint().value());
  EXPECT_FALSE(interval_0_2_set.ProvePoint().value());
  EXPECT_TRUE(interval_0_2_set.ProveSubSet(interval_0_4_set).value());
  EXPECT_FALSE(interval_0_4_set.ProveSubSet(interval_0_2_set).value());
  EXPECT_FALSE(interval_0_2_set.ProveSuperSet(interval_0_4_set).value());
  EXPECT_TRUE(interval_0_4_set.ProveSuperSet(interval_0_2_set).value());

  EXPECT_TRUE(ProveEQ(interval_0_2_set, interval_0_2_set).value());
  EXPECT_FALSE(ProveEQ(interval_0_2_set, interval_0_4_set).value());

  SingleIntervalIntSet union_0_6_set =
      ProvedUnion(interval_0_2_set, interval_2_6_set).value();
  EXPECT_EQ(union_0_6_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_6_set.Max(), ir::Expr(6));
  union_0_6_set = ProvedUnion(interval_2_6_set, interval_0_2_set).value();
  EXPECT_EQ(union_0_6_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_6_set.Max(), ir::Expr(6));
  SingleIntervalIntSet union_0_4_set =
      ProvedUnion(interval_0_2_set, interval_0_4_set).value();
  EXPECT_EQ(union_0_4_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_4_set.Max(), ir::Expr(4));
  union_0_4_set = ProvedUnion(interval_0_4_set, interval_0_2_set).value();
  EXPECT_EQ(union_0_4_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_4_set.Max(), ir::Expr(4));
  SingleIntervalIntSet union_0_9_set =
      ProvedUnion(interval_0_4_set, interval_8_9_set).value();
  EXPECT_EQ(union_0_9_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_9_set.Max(), ir::Expr(9));
  union_0_9_set = ProvedUnion(interval_8_9_set, interval_0_4_set).value();
  EXPECT_EQ(union_0_9_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_9_set.Max(), ir::Expr(9));

  SingleIntervalIntSet intersect_0_2_set =
      ProvedIntersect(interval_0_2_set, interval_0_4_set).value();
  EXPECT_EQ(intersect_0_2_set.Min(), ir::Expr(0));
  EXPECT_EQ(intersect_0_2_set.Max(), ir::Expr(2));
  intersect_0_2_set =
      ProvedIntersect(interval_0_4_set, interval_0_2_set).value();
  EXPECT_EQ(intersect_0_2_set.Min(), ir::Expr(0));
  EXPECT_EQ(intersect_0_2_set.Max(), ir::Expr(2));
  SingleIntervalIntSet intersect_2_2_set =
      ProvedIntersect(interval_0_2_set, interval_2_6_set).value();
  EXPECT_EQ(intersect_2_2_set.Min(), ir::Expr(2));
  EXPECT_EQ(intersect_2_2_set.Max(), ir::Expr(2));
  intersect_2_2_set =
      ProvedIntersect(interval_2_6_set, interval_0_2_set).value();
  EXPECT_EQ(intersect_2_2_set.Min(), ir::Expr(2));
  EXPECT_EQ(intersect_2_2_set.Max(), ir::Expr(2));
  SingleIntervalIntSet intersect_empty_set =
      ProvedIntersect(interval_0_4_set, interval_8_9_set).value();
  EXPECT_TRUE(intersect_empty_set.ProveEmpty().value());
  intersect_empty_set =
      ProvedIntersect(interval_8_9_set, interval_0_4_set).value();
  EXPECT_TRUE(intersect_empty_set.ProveEmpty().value());
}

TEST(SingleIntervalIntSet, case_0) {
  ir::Var S0 = ir::Var(ir::Expr(0), ir::Expr(7), "S0");
  ir::Expr e1 = S0 * 16;
  ir::Expr e2 = S0 * 16 + 7;
  ir::Expr e3 = S0 * 16 + 15;
  SingleIntervalIntSet empty_set(e2, e1);
  SingleIntervalIntSet single_point(e3, e3);
  SingleIntervalIntSet set_0(e1, e2);
  SingleIntervalIntSet set_1(e1, e3);

  EXPECT_TRUE(empty_set.ProveEmpty().value());
  EXPECT_FALSE(empty_set.ProveAll().value());
  EXPECT_TRUE(single_point.ProvePoint().value());
  EXPECT_FALSE(set_0.ProvePoint().value());
  EXPECT_TRUE(ProveEQ(set_0, set_0).value());
  EXPECT_FALSE(ProveEQ(set_0, set_1).value());

  EXPECT_TRUE(set_0.ProveSubSet(set_1).value());
  EXPECT_FALSE(set_1.ProveSubSet(set_0).value());
  EXPECT_FALSE(set_0.ProveSuperSet(set_1).value());
  EXPECT_TRUE(set_1.ProveSuperSet(set_0).value());

  EXPECT_TRUE(ProveEQ(ProvedUnion(set_0, set_1).value(), set_1).value());
  EXPECT_TRUE(ProveEQ(ProvedIntersect(set_0, set_1).value(), set_0).value());
  EXPECT_TRUE(ProveEQ(ProvedUnion(set_1, single_point).value(), set_1).value());
  EXPECT_TRUE(
      ProveEQ(ProvedIntersect(set_1, single_point).value(), single_point)
          .value());
  EXPECT_TRUE(ProveEQ(ProvedUnion(set_0, empty_set).value(), set_0).value());
  EXPECT_TRUE(
      ProveEQ(ProvedIntersect(set_0, empty_set).value(), empty_set).value());
  EXPECT_TRUE(ProveEQ(ProvedUnion(set_0, single_point).value(), set_1).value());
  EXPECT_TRUE(ProvedIntersect(set_0, single_point).value().ProveEmpty());
}

TEST(SingleIntervalIntSet, case_1) {
  ir::Var S0 = ir::Var(ir::Expr(0), ir::Expr(7), "S0");
  ir::Var S1 = ir::Var(ir::Expr(0), ir::Expr(15), "S1");
  ir::Expr e1 = S0 * 16;
  ir::Expr e2 = S0 * 16 + S1;
  ir::Expr e3 = S0 * 16 + S1 * 2 + 1;
  SingleIntervalIntSet empty_set(e3, e1);
  SingleIntervalIntSet single_point(e3, e3);
  SingleIntervalIntSet set_0(e1, e2);
  SingleIntervalIntSet set_1(e1, e3);

  EXPECT_TRUE(empty_set.ProveEmpty().value());
  EXPECT_FALSE(empty_set.ProveAll().value());
  EXPECT_TRUE(single_point.ProvePoint().value());
  EXPECT_FALSE(set_0.ProvePoint().has_value());
  EXPECT_TRUE(ProveEQ(set_0, set_0).value());
  EXPECT_FALSE(ProveEQ(set_0, set_1).value());

  EXPECT_TRUE(set_0.ProveSubSet(set_1).value());
  EXPECT_FALSE(set_1.ProveSubSet(set_0).value());
  EXPECT_FALSE(set_0.ProveSuperSet(set_1).value());
  EXPECT_TRUE(set_1.ProveSuperSet(set_0).value());

  EXPECT_TRUE(ProveEQ(ProvedUnion(set_0, set_1).value(), set_1).value());
  EXPECT_TRUE(ProveEQ(ProvedIntersect(set_0, set_1).value(), set_0).value());
  EXPECT_TRUE(ProveEQ(ProvedUnion(set_1, single_point).value(), set_1).value());
  EXPECT_TRUE(
      ProveEQ(ProvedIntersect(set_1, single_point).value(), single_point)
          .value());
  EXPECT_TRUE(ProveEQ(ProvedUnion(set_0, empty_set).value(), set_0).value());
  EXPECT_TRUE(
      ProveEQ(ProvedIntersect(set_0, empty_set).value(), empty_set).value());
  EXPECT_TRUE(ProveEQ(ProvedUnion(set_0, single_point).value(), set_1).value());
  EXPECT_TRUE(
      ProvedIntersect(set_0, single_point).value().ProveEmpty().value());
}

TEST(SingleIntervalIntSet, case_2) {
  ir::Var S = ir::Var(ir::Expr(0), ir::Expr(0), "S");

  SingleIntervalIntSet set_0{S, S + Expr(1)};
  SingleIntervalIntSet set_1{Expr(0), Expr(1)};
  SingleIntervalIntSet set_2{Expr(0), Expr(2)};

  EXPECT_TRUE(ProveEQ(set_0, set_1).value());
  EXPECT_FALSE(ProveEQ(set_0, set_2).value());
  EXPECT_TRUE(set_0.ProveSubSet(set_2).value());
  EXPECT_TRUE(set_2.ProveSuperSet(set_0).value());
}

}  // namespace common
}  // namespace cinn
