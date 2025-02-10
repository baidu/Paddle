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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/optim/merge_block_utils.h"

namespace cinn {
namespace optim {

namespace {

bool IsBlockForAllEqual(const ForTreeNode& first, const ForTreeNode& second) {
  auto ForVarExtentEqual = [&](const ForTreeNode& first,
                               const ForTreeNode& second) -> bool {
    const ir::Expr lhs = first.val->extent();
    const ir::Expr rhs = second.val->extent();
    if (lhs != rhs) {
      return false;
    }
    return true;
  };

  if (!ForVarExtentEqual(first, second)) return false;
  if (first.children.size() != second.children.size()) return false;
  for (size_t i = 0; i < first.children.size(); ++i) {
    if (!IsBlockForAllEqual(first.children[i], second.children[i])) {
      return false;
    }
  }

  return true;
}

ir::stmt::For MakeForLoops(const std::vector<int> extents, int index) {
  ir::stmt::StmtRef body_stmt;
  if (index == extents.size() - 1) {
    body_stmt = ir::stmt::Schedule(
        std::vector<Var>(),
        std::vector<Expr>(),
        std::vector<Expr>(),
        std::vector<Expr>(),
        "block",
        ir::stmt::BlockRef(std::vector<ir::stmt::StmtRef>()));
  } else {
    body_stmt = MakeForLoops(extents, index + 1);
  }

  std::vector<ir::stmt::StmtRef> body = {body_stmt};
  return ir::stmt::For(ir::Var("i"),
                       ir::Expr(0),
                       ir::Expr(extents[index]),
                       ir::ForType::Serial,
                       ir::DeviceAPI::CUDA,
                       ir::stmt::BlockRef(body),
                       ir::VectorizeInfo(),
                       ir::BindInfo());
}

void TestHelper(const std::vector<int>& extents1,
                const std::vector<int>& extents2,
                bool is_same) {
  auto for_loop1 = MakeForLoops(extents1, 0);
  auto for_loop2 = MakeForLoops(extents2, 0);

  if (is_same) {
    EXPECT_TRUE(CanMergeBlocks(for_loop1, for_loop2, IsBlockForAllEqual));
  } else {
    EXPECT_FALSE(CanMergeBlocks(for_loop1, for_loop2, IsBlockForAllEqual));
  }
}

void TestHelper2(const std::vector<std::vector<int>>& extents1,
                 const std::vector<std::vector<int>>& extents2,
                 bool is_same) {
  auto MakeNestLoops =
      [&](const std::vector<std::vector<int>>& extents) -> ir::stmt::For {
    std::vector<ir::stmt::StmtRef> for_loops;
    for (size_t i = 0; i < extents.size(); ++i) {
      for_loops.push_back(MakeForLoops(extents[i], 0));
    }
    ir::stmt::BlockRef block(for_loops);
    ir::stmt::For for_stmt = ir::stmt::For(ir::Var("i"),
                                           ir::Expr(0),
                                           ir::Expr(1),
                                           ir::ForType::Serial,
                                           ir::DeviceAPI::CUDA,
                                           block,
                                           ir::VectorizeInfo(),
                                           ir::BindInfo());
    return for_stmt;
  };

  auto for_stmt1 = MakeNestLoops(extents1);
  auto for_stmt2 = MakeNestLoops(extents2);

  if (is_same) {
    EXPECT_TRUE(CanMergeBlocks(for_stmt1, for_stmt2, IsBlockForAllEqual));
  } else {
    EXPECT_FALSE(CanMergeBlocks(for_stmt1, for_stmt2, IsBlockForAllEqual));
  }
}

TEST(ForInfo, ForInfoEqual) {
  TestHelper({10}, {10}, true);
  TestHelper({10, 5}, {10, 5}, true);
  TestHelper({10, 5, 3}, {10, 5, 3}, true);

  TestHelper2({{10}, {10}}, {{10}, {10}}, true);
  TestHelper2({{10, 5}, {4, 7}}, {{10, 5}, {4, 7}}, true);
  TestHelper2(
      {{10, 5, 3}, {4, 7, 9}, {2, 8}}, {{10, 5, 3}, {4, 7, 9}, {2, 8}}, true);
}

TEST(ForInfo, ForInfoNotEqual) {
  TestHelper({10}, {9}, false);
  TestHelper({10, 5}, {10, 4}, false);
  TestHelper({10, 5, 3}, {10, 5, 2}, false);

  TestHelper2({{10}, {10}}, {{10}, {9}}, false);
  TestHelper2({{10, 5}, {4, 7}}, {{10, 5}, {4, 3}}, false);
  TestHelper2(
      {{10, 5, 3}, {4, 7, 9}, {2, 8}}, {{10, 5, 3}, {4, 7, 9}, {2, 7}}, false);
}

}  // namespace

}  // namespace optim
}  // namespace cinn
