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

#include "gtest/gtest.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/ir_context.h"

namespace symbol::test {

// Construct DimExpr by overloaded operator(+, - , *, /)
TEST(DimExpr, DimExprNaive) {
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  DimExpr output = (sym0 + sym1) * constant1;
}

// Construct DimExpr by DimExprBuilder
TEST(DimExpr, DimExprBuilder) {
  DimExprBuilder builder;
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  DimExpr add = builder.Add(sym0, sym1);
  DimExpr out = builder.Broadcast(add, constant1);
}

/*
  Simulate the ShapeOrDataDimExprs result of below codes:
  def (x, y):
    extend_x = x.shape
    out = pd.reshape(y, extend_x)
*/
TEST(DimExpr, DataShapeExpr) {
  // Show ideal ShapeOrDataDimExprs of each pir::Value
  std::vector<DimExpr> x_shapes{DimExpr("S0"), DimExpr(2)};
  std::vector<DimExpr> y_shapes{DimExpr(1), DimExpr("S1"), DimExpr(2)};
  // x => {shape: [S0, 2], data: nullopt}
  ShapeOrDataDimExprs x_data_shape{symbol::TensorShapeOrDataDimExprs(x_shapes)};
  // y => {shape: [1, S1, 2], data: nullopt}
  ShapeOrDataDimExprs y_data_shape{symbol::TensorShapeOrDataDimExprs(y_shapes)};
  // out => {shape: [S0, 2], data: nullopt}
  ShapeOrDataDimExprs out_value_shape{
      symbol::TensorShapeOrDataDimExprs(x_shapes)};
}

/*
  Simulate the ShapeOrDataDimExprs result of below codes:
  def (x, y):
    out = pd.combine(x, y)
*/
TEST(DimExpr, TensorListShapeOrDataDimExprs) {
  std::vector<DimExpr> x_shapes{DimExpr("S0"), DimExpr("S1"), DimExpr(2)};
  std::vector<DimExpr> y_shapes{DimExpr(1), DimExpr("S3"), DimExpr(2)};
  // x => {shape: [S0, S1, 2], data: nullopt}
  ShapeOrDataDimExprs x_data_shape{symbol::TensorShapeOrDataDimExprs(x_shapes)};
  // y => {shape: [1, S3, 2], data: nullopt}
  ShapeOrDataDimExprs y_data_shape{symbol::TensorShapeOrDataDimExprs(y_shapes)};

  // out => {shape: [S0, S1, 2], data: nullopt, shape: [1, S3, 2], data:
  // nullopt}
  ShapeOrDataDimExprs out_data_shape_list(
      {symbol::TensorShapeOrDataDimExprs(x_shapes),
       symbol::TensorShapeOrDataDimExprs(y_shapes)});
}

TEST(Simplify, NumberArithmetic) {
  DimExpr number = DimExpr(5);
  DimExpr add_minus = number + number - number;
  ASSERT_TRUE((add_minus.Has<std::int64_t>()));
  ASSERT_EQ((add_minus.Get<std::int64_t>()), 5);
  DimExpr mul_div = number * DimExpr(1) / number;
  ASSERT_TRUE((mul_div.Has<std::int64_t>()));
  ASSERT_EQ((mul_div.Get<std::int64_t>()), 1);
}

TEST(DimExpr, Equal) {
  DimExprBuilder builder;
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  ASSERT_EQ(sym0 + sym1, sym0 + sym1);
  ASSERT_EQ(sym0 + sym1, sym1 + sym0);
  ASSERT_EQ(sym0 + constant1, DimExpr("S0") + constant1);
  ASSERT_EQ(sym0 - sym1, sym0 - sym1);
  ASSERT_NE(sym0 - sym1, sym1 - sym0);
  ASSERT_EQ(sym0 - constant1, DimExpr("S0") - constant1);
  ASSERT_EQ(sym0 * sym1, sym0 * sym1);
  ASSERT_EQ(sym0 * sym1, sym1 * sym0);
  ASSERT_EQ(sym0 * constant1, DimExpr("S0") * constant1);
  ASSERT_EQ(sym0 / sym1, sym0 / sym1);
  ASSERT_NE(sym0 / sym1, sym1 / sym0);
  ASSERT_EQ(sym0 / constant1, DimExpr("S0") / constant1);
  ASSERT_EQ(builder.Max(sym0, sym1), builder.Max(sym0, sym1));
  ASSERT_NE(builder.Max(sym0, sym1), builder.Max(sym1, sym0));
  ASSERT_EQ(builder.Max(sym0, constant1),
            builder.Max(DimExpr("S0"), constant1));
  ASSERT_EQ(builder.Min(sym0, sym1), builder.Min(sym0, sym1));
  ASSERT_NE(builder.Min(sym0, sym1), builder.Min(sym1, sym0));
  ASSERT_EQ(builder.Min(sym0, constant1),
            builder.Min(DimExpr("S0"), constant1));
  ASSERT_EQ(builder.Broadcast(sym0, sym1), builder.Broadcast(sym0, sym1));
  ASSERT_EQ(builder.Broadcast(sym0, sym1), builder.Broadcast(sym1, sym0));
  ASSERT_EQ(builder.Broadcast(sym0, constant1),
            builder.Broadcast(DimExpr("S0"), constant1));
}

TEST(DimExpr, Print) {
  DimExprBuilder builder;
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  ASSERT_EQ((ToString(sym0 + sym1)), "Add(S0, S1)");
  ASSERT_EQ((ToString(sym0 - sym1)), "Add(S0, -S1)");
  ASSERT_EQ((ToString(sym0 * sym1)), "Mul(S0, S1)");
  ASSERT_EQ((ToString(sym0 / sym1)), "Div(S0, S1)");
  ASSERT_EQ((ToString(builder.Max(sym0, sym1))), "Max(S0, S1)");
  ASSERT_EQ((ToString(builder.Min(sym0, sym1))), "Min(S0, S1)");
  ASSERT_EQ((ToString(builder.Broadcast(sym0, sym1))), "Broadcast(S0, S1)");
}

TEST(DimExpr, Hash) {
  DimExprBuilder builder;
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  ASSERT_EQ((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(sym0 + sym1)));
  ASSERT_EQ((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(sym1 + sym0)));
  ASSERT_NE((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(sym0 - sym1)));
  ASSERT_NE((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(sym0 * sym1)));
  ASSERT_NE((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(sym0 / sym1)));
  ASSERT_NE((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(builder.Max(sym0, sym1))));
  ASSERT_NE((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(builder.Min(sym0, sym1))));
  ASSERT_NE((std::hash<DimExpr>()(sym0 + sym1)),
            (std::hash<DimExpr>()(builder.Broadcast(sym0, sym1))));
}

}  // namespace symbol::test
