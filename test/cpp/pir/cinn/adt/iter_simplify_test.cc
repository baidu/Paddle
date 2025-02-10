// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/iter_simplify.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/schedule_base.h"

namespace cinn {
namespace common {

#define ITER_MARK_VAR(var) \
  ir::IterMark::Make(ir::IndexExpr(var.ptr()), var->upper_bound)
#define ITER_MARK_SUM(sum, ext) ir::IterMark::Make(sum, ext)
#define ITER_SPLIT(mark, ...) ir::IterSplit::Make(mark, ##__VA_ARGS__)
#define ITER_SUM(...) ir::IterSum::Make({__VA_ARGS__}, ir::IndexExpr(0))
#define ITER_SUM_WITH_BASE(base, ...) ir::IterSum::Make({__VA_ARGS__}, base)

#define TEST_EXPR(expr, expected, expr_norm) \
  rewriter.Rewrite(&expr);                   \
  EXPECT_EQ(expr, Expr(expected));           \
  normalizer.Convert(&expr);                 \
  EXPECT_EQ(expr, expr_norm);

class TestIterSimplify : public ::testing::Test {
 public:
  void SetUp() override {
    i = ir::Var(ir::Expr(0), ir::Expr(2), "i").set_index(1);
    j = ir::Var(ir::Expr(0), ir::Expr(4), "j").set_index(1);
    k = ir::Var(ir::Expr(0), ir::Expr(8), "k").set_index(1);
    i_j_k_fused =
        ir::Var(ir::Expr(0), ir::Expr(64), "i_j_k_fused").set_index(1);
    var_intervals = {
        {"i", CasInterval(i->lower_bound, i->upper_bound)},
        {"j", CasInterval(j->lower_bound, j->upper_bound)},
        {"k", CasInterval(k->lower_bound, k->upper_bound)},
        {"i_j_k_fused",
         CasInterval(i_j_k_fused->lower_bound, i_j_k_fused->upper_bound)}};
  };

  ir::Var i;
  ir::Var j;
  ir::Var k;
  ir::Var i_j_k_fused;
  cas_intervals_t var_intervals;
  SymbolicExprAnalyzer analyzer{var_intervals};
};

TEST_F(TestIterSimplify, IterExprMake) {
  // IterMark Make func.
  auto mark_expr = ITER_MARK_VAR(i);
  auto mark_expr_ = ITER_MARK_VAR(j);
  // IterSplit Make func.
  auto split_0_expr = ITER_SPLIT(mark_expr);
  auto split_1_expr = ITER_SPLIT(mark_expr, ir::IndexExpr(1));
  auto split_2_expr = ITER_SPLIT(
      mark_expr, ir::IndexExpr(1), ir::IndexExpr(2), ir::IndexExpr(1));
  auto split_3_expr = ITER_SPLIT(
      mark_expr, ir::IndexExpr(2), ir::IndexExpr(2), ir::IndexExpr(1));
  auto split_4_expr = ITER_SPLIT(
      mark_expr_, ir::IndexExpr(1), ir::IndexExpr(2), ir::IndexExpr(1));
  // IterSum Make func.
  auto sum_expr = ITER_SUM(split_0_expr, split_1_expr, split_2_expr);

  auto mark = mark_expr.As<ir::IterMark>();
  auto split_0 = split_0_expr.As<ir::IterSplit>();
  auto split_1 = split_1_expr.As<ir::IterSplit>();
  auto split_2 = split_2_expr.As<ir::IterSplit>();
  auto sum = sum_expr.As<ir::IterSum>();

  EXPECT_EQ(mark->source, ir::IndexExpr(i.ptr()));
  EXPECT_EQ(mark->extent, ir::IndexExpr(2));

  EXPECT_EQ(split_0->source, mark_expr);
  EXPECT_EQ(split_0->lower_factor, ir::IndexExpr(1));
  EXPECT_EQ(split_0->extent, ir::IndexExpr(2));
  EXPECT_EQ(split_0->scale, ir::IndexExpr(1));

  EXPECT_EQ(split_1->source, mark_expr);
  EXPECT_EQ(split_1->lower_factor, ir::IndexExpr(1));
  EXPECT_EQ(split_1->extent, ir::IndexExpr(2));
  EXPECT_EQ(split_1->scale, ir::IndexExpr(1));

  EXPECT_EQ(split_2->source, mark_expr);
  EXPECT_EQ(split_2->lower_factor, ir::IndexExpr(1));
  EXPECT_EQ(split_2->extent, ir::IndexExpr(2));
  EXPECT_EQ(split_2->scale, ir::IndexExpr(1));

  EXPECT_EQ(sum->args.size(), 3);
  EXPECT_EQ(sum->base, Expr(0));

  EXPECT_NE(mark_expr, mark_expr_);

  EXPECT_EQ(split_0_expr, split_1_expr);
  EXPECT_EQ(split_1_expr, split_2_expr);
  EXPECT_NE(split_2_expr, split_3_expr);
}

TEST_F(TestIterSimplify, conversion) {
  IterMapRewriter rewriter{{i}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};
  ir::Expr e1 = i;
  auto gt = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i)));
  TEST_EXPR(e1, gt, e1);
}

TEST_F(TestIterSimplify, add) {
  IterMapRewriter rewriter{{i, j, k}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 =
      ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i)), ITER_SPLIT(ITER_MARK_VAR(j)));
  auto gt2 = ITER_SUM_WITH_BASE(ir::IndexExpr(5),
                                ITER_SPLIT(ITER_MARK_VAR(i)),
                                ITER_SPLIT(ITER_MARK_VAR(j)),
                                ITER_SPLIT(ITER_MARK_VAR(k)));
  auto gt3 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i), ir::IndexExpr(2)));
  auto gt4 = ITER_SUM_WITH_BASE(ir::IndexExpr(12));

  ir::Expr e1 = i + j;
  ir::Expr e2 = i + j + k + 5;
  ir::Expr e3 = i + i;
  ir::Expr e4 = Expr(7) + Expr(5);

  TEST_EXPR(e1, gt1, i + j);
  TEST_EXPR(e2, gt2, i + j + k + 5);
  TEST_EXPR(e3, gt3, i * 2);
  TEST_EXPR(e4, gt4, Expr(12));
}

TEST_F(TestIterSimplify, sub) {
  IterMapRewriter rewriter{{i, j, k}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i)),
                      ITER_SPLIT(ITER_MARK_VAR(j), ir::IndexExpr(-1)));
  auto gt2 =
      ITER_SUM_WITH_BASE(ir::IndexExpr(5),
                         ITER_SPLIT(ITER_MARK_VAR(i)),
                         ITER_SPLIT(ITER_MARK_VAR(j)),
                         ITER_SPLIT(ITER_MARK_VAR(k), ir::IndexExpr(-1)));
  auto gt3 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i), ir::IndexExpr(0)));
  auto gt4 = ITER_SUM_WITH_BASE(ir::IndexExpr(2));

  ir::Expr e1 = i - j;
  ir::Expr e2 = i + j - k + 5;
  ir::Expr e3 = i - i;
  ir::Expr e4 = Expr(7) - Expr(5);
  TEST_EXPR(e1, gt1, (j * -1) + i);
  TEST_EXPR(e2, gt2, i + j + (k * -1) + 5);
  TEST_EXPR(e3, gt3, Expr(0));
  TEST_EXPR(e4, gt4, Expr(2));
}

TEST_F(TestIterSimplify, mul) {
  IterMapRewriter rewriter{{i, j, k}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i), ir::IndexExpr(2)),
                      ITER_SPLIT(ITER_MARK_VAR(j)));
  auto gt2 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i), ir::IndexExpr(2)),
                      ITER_SPLIT(ITER_MARK_VAR(j), ir::IndexExpr(2)),
                      ITER_SPLIT(ITER_MARK_VAR(k)));

  auto gt3 = ITER_SUM_WITH_BASE(ir::IndexExpr(10),
                                ITER_SPLIT(ITER_MARK_VAR(i), ir::IndexExpr(2)),
                                ITER_SPLIT(ITER_MARK_VAR(j), ir::IndexExpr(2)),
                                ITER_SPLIT(ITER_MARK_VAR(k)));
  auto gt4 = ITER_SUM_WITH_BASE(ir::IndexExpr(35));

  ir::Expr e1 = i * 2 + j;
  ir::Expr e2 = (i + j) * 2 + k;
  ir::Expr e3 = (i + j + 5) * 2 + k;
  ir::Expr e4 = Expr(7) * Expr(5);

  TEST_EXPR(e1, gt1, i * 2 + j);
  TEST_EXPR(e2, gt2, (i + j) * 2 + k);
  TEST_EXPR(e3, gt3, (i + j) * 2 + k + 10);
  TEST_EXPR(e4, gt4, Expr(35));
}

TEST_F(TestIterSimplify, div) {
  IterMapRewriter rewriter{{i, j, k, i_j_k_fused}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(1)));
  auto gt2 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(32),
                                 ir::IndexExpr(2),
                                 ir::IndexExpr(1)));
  auto gt3 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused)));
  auto gt4 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused), ir::IndexExpr(2)));
  auto gt5 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(2),
                                 ir::IndexExpr(32),
                                 ir::IndexExpr(1)));
  auto gt6 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM_WITH_BASE(ir::IndexExpr(8),
                                       ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused))),
                    ir::IndexExpr(72)),
      ir::IndexExpr(16),
      ir::IndexExpr(5),
      ir::IndexExpr(1)));
  auto gt7 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM_WITH_BASE(ir::IndexExpr(1),
                                       ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused))),
                    ir::IndexExpr(65)),
      ir::IndexExpr(2),
      ir::IndexExpr(33),
      ir::IndexExpr(1)));
  auto gt8 = ITER_SUM_WITH_BASE(ir::IndexExpr(2),
                                ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                           ir::IndexExpr(8),
                                           ir::IndexExpr(8),
                                           ir::IndexExpr(1)));
  auto gt9 = ITER_SUM_WITH_BASE(
      ir::IndexExpr(2),
      ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused), ir::IndexExpr(2)));
  auto gt10 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM_WITH_BASE(ir::IndexExpr(1),
                                       ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused))),
                    ir::IndexExpr(65)),
      ir::IndexExpr(8),
      ir::IndexExpr(9),
      ir::IndexExpr(1)));
  auto gt11 = ITER_SUM_WITH_BASE(ir::IndexExpr(3));
  auto gt12 = ITER_SUM_WITH_BASE(ir::IndexExpr(3));
  auto gt13 = ITER_SUM_WITH_BASE(ir::IndexExpr(15));
  auto gt14 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));

  ir::Expr e1 = i_j_k_fused / 8;
  ir::Expr e2 = i_j_k_fused / 8 / 4;
  ir::Expr e3 = i_j_k_fused / 1;
  ir::Expr e4 = i_j_k_fused * 16 / 8;
  ir::Expr e5 = i_j_k_fused * 8 / 16;
  ir::Expr e6 = (i_j_k_fused + 8) / 16;
  ir::Expr e7 = (i_j_k_fused * 8 + 8) / 16;
  ir::Expr e8 = (i_j_k_fused + 16) / 8;
  ir::Expr e9 = (i_j_k_fused * 16 + 16) / 8;
  ir::Expr e10 = (i_j_k_fused + 1) / 8;
  ir::Expr e11 = Expr(15) / Expr(5);
  ir::Expr e12 = Expr(15) / Expr(4);
  ir::Expr e13 = Expr(15) / Expr(1);
  ir::Expr e14 = Expr(0) / Expr(4);

  TEST_EXPR(e1, gt1, i_j_k_fused / 8);
  TEST_EXPR(e2, gt2, i_j_k_fused / 32);
  TEST_EXPR(e3, gt3, i_j_k_fused);
  TEST_EXPR(e4, gt4, i_j_k_fused * 2);
  TEST_EXPR(e5, gt5, i_j_k_fused / 2);
  TEST_EXPR(e6, gt6, (i_j_k_fused + 8) / 16);
  TEST_EXPR(e7, gt7, (i_j_k_fused + 1) / 2);
  TEST_EXPR(e8, gt8, i_j_k_fused / 8 + 2);
  TEST_EXPR(e9, gt9, i_j_k_fused * 2 + 2);
  TEST_EXPR(e10, gt10, (i_j_k_fused + 1) / 8);
  TEST_EXPR(e11, gt11, Expr(3));
  TEST_EXPR(e12, gt12, Expr(3));
  TEST_EXPR(e13, gt13, Expr(15));
  TEST_EXPR(e14, gt14, Expr(0));
}

TEST_F(TestIterSimplify, mod) {
  IterMapRewriter rewriter{{i, j, k, i_j_k_fused}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(1),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(1)));
  auto gt2 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(4),
                                 ir::IndexExpr(1)));
  auto gt3 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));
  auto gt4 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));
  auto gt5 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(1),
                                 ir::IndexExpr(2),
                                 ir::IndexExpr(8)));
  auto gt6 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM_WITH_BASE(ir::IndexExpr(8),
                                       ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused))),
                    ir::IndexExpr(72)),
      ir::IndexExpr(1),
      ir::IndexExpr(16),
      ir::IndexExpr(1)));
  auto gt7 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM_WITH_BASE(ir::IndexExpr(1),
                                       ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                                  ir::IndexExpr(1),
                                                  ir::IndexExpr(64),
                                                  ir::IndexExpr(1))),
                    ir::IndexExpr(65)),
      ir::IndexExpr(1),
      ir::IndexExpr(2),
      ir::IndexExpr(8)));
  auto gt8 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(1),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(1)));
  auto gt9 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));
  auto gt10 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM_WITH_BASE(ir::IndexExpr(1),
                                       ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused))),
                    ir::IndexExpr(65)),
      ir::IndexExpr(1),
      ir::IndexExpr(8),
      ir::IndexExpr(1)));
  auto gt11 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));
  auto gt12 = ITER_SUM_WITH_BASE(ir::IndexExpr(3));
  auto gt13 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));
  auto gt14 = ITER_SUM_WITH_BASE(ir::IndexExpr(0));

  ir::Expr e1 = i_j_k_fused % 8;
  ir::Expr e2 = i_j_k_fused / 8 % 4;
  ir::Expr e3 = i_j_k_fused % 1;
  ir::Expr e4 = i_j_k_fused * 16 % 8;
  ir::Expr e5 = i_j_k_fused * 8 % 16;
  ir::Expr e6 = (i_j_k_fused + 8) % 16;
  ir::Expr e7 = (i_j_k_fused * 8 + 8) % 16;
  ir::Expr e8 = (i_j_k_fused + 16) % 8;
  ir::Expr e9 = (i_j_k_fused * 16 + 16) % 8;
  ir::Expr e10 = (i_j_k_fused + 1) % 8;
  ir::Expr e11 = Expr(15) % Expr(5);
  ir::Expr e12 = Expr(15) % Expr(4);
  ir::Expr e13 = Expr(15) % Expr(1);
  ir::Expr e14 = Expr(0) % Expr(4);

  TEST_EXPR(e1, gt1, i_j_k_fused % 8);
  TEST_EXPR(e2, gt2, i_j_k_fused % 32 / 8);
  TEST_EXPR(e3, gt3, Expr(0));
  TEST_EXPR(e4, gt4, Expr(0));
  TEST_EXPR(e5, gt5, i_j_k_fused % 2 * 8);
  TEST_EXPR(e6, gt6, (i_j_k_fused + 8) % 16);
  TEST_EXPR(e7, gt7, (i_j_k_fused + 1) % 2 * 8);
  TEST_EXPR(e8, gt8, i_j_k_fused % 8);
  TEST_EXPR(e9, gt9, Expr(0));
  TEST_EXPR(e10, gt10, (i_j_k_fused + 1) % 8);
  TEST_EXPR(e11, gt11, Expr(0));
  TEST_EXPR(e12, gt12, Expr(3));
  TEST_EXPR(e13, gt13, Expr(0));
  TEST_EXPR(e14, gt14, Expr(0));
}

TEST_F(TestIterSimplify, fuse_not_same_source) {
  IterMapRewriter rewriter{{i, j, k, i_j_k_fused}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};

  auto gt1 = ITER_SUM(ITER_SPLIT(
      ITER_MARK_SUM(ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i), ir::IndexExpr(32)),
                             ITER_SPLIT(ITER_MARK_VAR(j), ir::IndexExpr(8)),
                             ITER_SPLIT(ITER_MARK_VAR(k), ir::IndexExpr(1))),
                    ir::IndexExpr(64)),
      ir::IndexExpr(8),
      ir::IndexExpr(8),
      ir::IndexExpr(1)));

  ir::Expr e1 = (i * 32 + j * 8 + k) / 8;
  ir::Expr e2 = (i * 32 + j * 7) / 8;

  TEST_EXPR(e1, gt1, (i * 32 + j * 8 + k) / 8);
  EXPECT_ANY_THROW(rewriter.Rewrite(&e2));
}

TEST_F(TestIterSimplify, fuse_same_source) {
  IterMapRewriter rewriter{{i, j, k, i_j_k_fused}, analyzer};
  IterMapToExprNormalizer normalizer{analyzer};

  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(32),
                                 ir::IndexExpr(2),
                                 ir::IndexExpr(1)));
  auto gt2 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(4),
                                 ir::IndexExpr(1)));
  auto gt3 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(1),
                                 ir::IndexExpr(8),
                                 ir::IndexExpr(1)));
  auto gt4 = ITER_SUM(ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(32),
                                 ir::IndexExpr(2),
                                 ir::IndexExpr(1)),
                      ITER_SPLIT(ITER_MARK_VAR(i_j_k_fused),
                                 ir::IndexExpr(32),
                                 ir::IndexExpr(1),
                                 ir::IndexExpr(1)));

  ir::Expr e1 = (i_j_k_fused / 16 / 2 * 32 + i_j_k_fused / 16 % 2 * 16 +
                 i_j_k_fused % 16) /
                8 / 4;
  ir::Expr e2 =
      (i_j_k_fused / 32 * 32 + i_j_k_fused / 16 % 2 * 16 + i_j_k_fused % 16) /
      8 % 4;
  ir::Expr e3 =
      (i_j_k_fused / 32 * 32 + i_j_k_fused / 16 % 2 * 16 + i_j_k_fused % 16) %
      8;

  ir::Expr e4 =
      ((i_j_k_fused / 16) / 2) +
      ((((i_j_k_fused % 16) / 8) + (2 * ((i_j_k_fused / 16) % 2))) / 4);
  ir::Expr e5 = (((i_j_k_fused % 16) / 8) + ((4 * ((i_j_k_fused / 16) / 2)) +
                                             (2 * ((i_j_k_fused / 16) % 2)))) %
                4;
  ir::Expr e6 = ((i_j_k_fused % 16) + ((32 * ((i_j_k_fused / 16) / 2)) +
                                       (16 * ((i_j_k_fused / 16) % 2)))) %
                8;

  TEST_EXPR(e1, gt1, i_j_k_fused / 32);
  TEST_EXPR(e2, gt2, i_j_k_fused % 32 / 8);
  TEST_EXPR(e3, gt3, i_j_k_fused % 8);
  TEST_EXPR(e4, gt4, i_j_k_fused / 32);
  TEST_EXPR(e5, gt2, i_j_k_fused % 32 / 8);
  TEST_EXPR(e6, gt3, i_j_k_fused % 8);
}

TEST_F(TestIterSimplify, SimplifyBindings) {
  std::vector<ir::Var> block_vars;
  std::vector<ir::Expr> iter_values;
  std::vector<ir::Expr> shape = {Expr(2), Expr(4), Expr(8)};
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  // Create block vars and axis vars
  for (int i = 0; i < shape.size(); ++i) {
    block_vars.push_back(ir::Var(Expr(0),
                                 shape[i],
                                 cinn::UniqName("b" + std::to_string(i)),
                                 false,
                                 false)
                             .set_index(1));
    axis_vars[i]->is_reduce_axis = false;
    iter_values.push_back(axis_vars[i]);
  }

  // Create ScheduleBlock body
  ir::Expr body_ = ir::ScheduleBlockRealize::Make(
      iter_values,
      ir::ScheduleBlock::Make(block_vars, {}, {}, "Test", Expr(0)));

  // Create For loops
  auto body = body_;
  for (int i = shape.size() - 1; i >= 0; --i) {
    ir::Var loop_var = axis_vars[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         ir::Block::Make({body}));
  }

  // Create outer ScheduleBlockRealize
  ir::Expr body_outer = ir::ScheduleBlockRealize::Make(
      {}, ir::ScheduleBlock::Make({}, {}, {}, "test1", body));

  // Create ir schedule
  ir::ModuleExpr mod_expr({ir::Block::Make({body_outer})});
  ir::IRSchedule ir_sch(mod_expr);
  std::vector<ir::Expr> loops = ir_sch.GetLoops(body_);

  // Apply Fuse and Split
  ir::Expr loop_fuse = ir_sch.Fuse(loops);
  std::vector<ir::Expr> loops_split = ir_sch.Split(loop_fuse, {2, 2, 16});
  ir::Expr loop_fuse_2 = ir_sch.Fuse(loops_split);

  // Apply SimplifyBindings
  SimplifyBlockBinding::SimplifyBindings(loop_fuse_2, {}, analyzer);

  // Check result
  auto for_op = loop_fuse_2.As<ir::For>();
  auto simplified_values = for_op->body.As<ir::Block>()
                               ->stmts[0]
                               .As<ir::ScheduleBlockRealize>()
                               ->iter_values;
  auto f = for_op->loop_var;

  EXPECT_EQ(simplified_values[0], f / 32);
  EXPECT_EQ(simplified_values[1], f % 32 / 8);
  EXPECT_EQ(simplified_values[2], f % 8);
}

TEST_F(TestIterSimplify, MergeMulMod) {
  auto S0 = ir::Var(ir::Expr(0), ir::Expr(4), "S0").set_index(1);
  auto S1 = ir::Var(ir::Expr(0), ir::Expr(256), "S1").set_index(1);
  auto S2 = ir::Var(ir::Expr(0), ir::Expr(13), "S2").set_index(1);

  auto e1 = ((((((((S0 * 256) + S1) + (S2 * 1024)) / 2500) * 50) +
               (((((S0 * 256) + S1) + (S2 * 1024)) % 2500) / 50)) *
              50) +
             ((((S0 * 256) + S1) + (S2 * 1024)) % 50));
  auto e2 = ((((((S0 * 256) + S1) + (S2 * 1024)) / 2500) + -4) * 2500) +
            ((((S0 * 256) + S1) + (S2 * 1024)) % 2500);

  auto e3 = (S1 / 784 * 28 + S1 % 784 / 28) * 28 + S1 % 28;

  EXPECT_EQ(MergeMulMod(e1), (((S0 * 256) + S1) + (S2 * 1024)));
  EXPECT_EQ(MergeMulMod(e2), ((((S0 * 256) + S1) + (S2 * 1024)) + -10000));
  EXPECT_EQ(MergeMulMod(e3), S1);
}
}  // namespace common
}  // namespace cinn
