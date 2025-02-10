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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/common/dfs_topo_walker.h"

namespace cinn {
namespace common {

TEST(DfsTopoWalker, simple) {
  std::vector<std::pair<int, int>> edges{
      {0, 1}, {2, 3}, {1, 3}, {0, 3}, {3, 4}};
  DfsTopoWalker<int> walker(
      [&](int node, const std::function<void(int)>& NodeHandler) {
        for (const auto& pair : edges) {
          if (pair.second == node) {
            NodeHandler(pair.first);
          }
        }
      },
      [&](int node, const std::function<void(int)>& NodeHandler) {
        for (const auto& pair : edges) {
          if (pair.first == node) {
            NodeHandler(pair.second);
          }
        }
      });
  std::vector<int> sources{0, 2};
  std::vector<int> outputs;
  walker(sources.begin(), sources.end(), [&](int node) {
    outputs.push_back(node);
  });
  for (auto output : outputs) {
    LOG(INFO) << output;
  }
  std::vector<int> expected{0, 1, 2, 3, 4};
  EXPECT_TRUE((outputs == expected));
}

}  // namespace common
}  // namespace cinn
