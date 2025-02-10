// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/common/broadcast_tree.h"

#include "gtest/gtest.h"

namespace cinn::common {
using namespace symbol;  // NOLINT

namespace {

DimExpr MakeBroadcastDimExpr(const DimExpr& expr1, const DimExpr& expr2) {
  List<DimExpr> operands{expr1, expr2};
  return Broadcast<DimExpr>{operands};
}

bool DimExprNonBroadcast(const DimExpr& dim_expr) {
  if (dim_expr.Has<Broadcast<DimExpr>>()) {
    return false;
  } else {
    return true;
  }
}

void CheckLeafNonBroadcast(const BroadcastLeaf& leaf) {
  for (const auto& operands : *leaf) {
    for (const auto& operand : operands) {
      ASSERT_TRUE(DimExprNonBroadcast(operand));
    }
  }
}

void CheckInnerBranchNonBroadcast(
    const BroadcastBranch<BroadcastTree>& branch) {
  const auto& [_, lhs_eq_rhs_tree, lhs_eq_one_tree, rhs_eq_one_tree] =
      branch.tuple();
  ASSERT_TRUE(lhs_eq_rhs_tree.Has<BroadcastLeaf>());
  ASSERT_TRUE(lhs_eq_one_tree.Has<BroadcastLeaf>());
  ASSERT_TRUE(rhs_eq_one_tree.Has<BroadcastLeaf>());
  CheckLeafNonBroadcast(lhs_eq_rhs_tree.Get<BroadcastLeaf>());
  CheckLeafNonBroadcast(lhs_eq_one_tree.Get<BroadcastLeaf>());
  CheckLeafNonBroadcast(rhs_eq_one_tree.Get<BroadcastLeaf>());
}

}  // namespace

TEST(BroadcastTree, Naive) {
  DimExpr expr1("S1");
  DimExpr expr2("S2");
  DimExpr expr3("S3");
  DimExpr expr4("S4");
  std::vector<DimExpr> tensor_shape{expr1,
                                    expr2,
                                    MakeBroadcastDimExpr(expr1, expr2),
                                    MakeBroadcastDimExpr(expr3, expr4)};
  BroadcastLeaf leaf = adt::List<std::vector<DimExpr>>{tensor_shape};
  int num_of_leaves = 0;
  BroadcastTree tree = ConstructBroadcastTree(leaf, &num_of_leaves);
  ASSERT_TRUE(tree.Has<BroadcastBranch<BroadcastTree>>());
  const auto& branch = tree.Get<BroadcastBranch<BroadcastTree>>();
  const auto& [cstr_broadcastable,
               lhs_eq_rhs_tree,
               lhs_eq_one_tree,
               rhs_eq_one_tree] = branch.tuple();
  ASSERT_EQ(cstr_broadcastable->lhs, DimExpr("S1"));
  ASSERT_EQ(cstr_broadcastable->rhs, DimExpr("S2"));
  ASSERT_TRUE(lhs_eq_rhs_tree.Has<BroadcastBranch<BroadcastTree>>());
  ASSERT_TRUE(lhs_eq_one_tree.Has<BroadcastBranch<BroadcastTree>>());
  ASSERT_TRUE(rhs_eq_one_tree.Has<BroadcastBranch<BroadcastTree>>());
  CheckInnerBranchNonBroadcast(
      lhs_eq_rhs_tree.Get<BroadcastBranch<BroadcastTree>>());
  CheckInnerBranchNonBroadcast(
      lhs_eq_one_tree.Get<BroadcastBranch<BroadcastTree>>());
  CheckInnerBranchNonBroadcast(
      rhs_eq_one_tree.Get<BroadcastBranch<BroadcastTree>>());
}

TEST(BroadcastTree, SimplifyConstantBroadcast) {
  DimExpr expr1("S1");
  DimExpr expr2("S2");
  DimExpr expr3("S3");
  DimExpr expr4(4);
  std::vector<DimExpr> tensor_shape{expr1,
                                    expr2,
                                    MakeBroadcastDimExpr(expr1, expr2),
                                    MakeBroadcastDimExpr(expr3, expr4)};
  BroadcastLeaf leaf = adt::List<std::vector<DimExpr>>{tensor_shape};
  int num_of_leaves = 0;
  BroadcastTree tree = ConstructBroadcastTree(leaf, &num_of_leaves);
  ASSERT_TRUE(tree.Has<BroadcastBranch<BroadcastTree>>());
  const auto& branch = tree.Get<BroadcastBranch<BroadcastTree>>();
  const auto& [cstr_broadcastable,
               lhs_eq_rhs_tree,
               lhs_eq_one_tree,
               rhs_eq_one_tree] = branch.tuple();
  ASSERT_EQ(cstr_broadcastable->lhs, DimExpr("S1"));
  ASSERT_EQ(cstr_broadcastable->rhs, DimExpr("S2"));
  ASSERT_TRUE(lhs_eq_rhs_tree.Has<BroadcastLeaf>());
  ASSERT_TRUE(lhs_eq_one_tree.Has<BroadcastLeaf>());
  ASSERT_TRUE(rhs_eq_one_tree.Has<BroadcastLeaf>());
  CheckLeafNonBroadcast(lhs_eq_rhs_tree.Get<BroadcastLeaf>());
  CheckLeafNonBroadcast(lhs_eq_one_tree.Get<BroadcastLeaf>());
  CheckLeafNonBroadcast(rhs_eq_one_tree.Get<BroadcastLeaf>());
}

}  // namespace cinn::common
