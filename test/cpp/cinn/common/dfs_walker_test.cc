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

#include "paddle/cinn/common/dfs_walker.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace cinn {
namespace common {

TEST(DfsWalker, simple_on_push) {
  DfsWalker<int> visitor(
      [](int node, const std::function<void(int)>& NodeHandler) {
        if (node == 0) {
          NodeHandler(3);
        } else if (node == 1) {
          NodeHandler(2);
          NodeHandler(3);
        } else if (node == 2 || node == 3) {
          NodeHandler(4);
        }
      });
  std::vector<int> sources{0, 1};
  std::vector<int> outputs;
  visitor(sources.begin(), sources.end(), [&](int node) {
    LOG(ERROR) << node;
    outputs.push_back(node);
  });
  std::vector<int> expected{0, 3, 4, 1, 2};
  EXPECT_TRUE((outputs == expected));
}

TEST(DfsWalker, simple_on_pop) {
  DfsWalker<int> visitor(
      [](int node, const std::function<void(int)>& NodeHandler) {
        if (node == 0) {
          NodeHandler(3);
        } else if (node == 1) {
          NodeHandler(2);
          NodeHandler(3);
        } else if (node == 2 || node == 3) {
          NodeHandler(4);
        }
      });
  std::vector<int> sources{0, 1};
  std::vector<int> outputs;
  visitor(
      sources.begin(),
      sources.end(),
      [](int) {},
      [&](int node) {
        LOG(ERROR) << node;
        outputs.push_back(node);
      });
  std::vector<int> expected{4, 3, 0, 2, 1};
  EXPECT_TRUE((outputs == expected));
}

}  // namespace common
}  // namespace cinn
