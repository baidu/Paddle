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

#include "paddle/cinn/common/is_reachable_predicator.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace cinn {
namespace common {

TEST(IsReachablePredicator, simple) {
  IsReachablePredicator<int> IsReachable(
      // Get min depth
      [](int x) { return std::abs(x); },
      // Get max depth
      [](int x) { return std::abs(x); },
      // visit next node
      [](int x, const std::function<void(int)>& Handler) {
        Handler(x + (x / std::abs(x)));
      });
  EXPECT_TRUE(IsReachable(33, 99, [](int) {}));
  EXPECT_FALSE(IsReachable(33, -99, [](int) {}));
}

}  // namespace common
}  // namespace cinn
