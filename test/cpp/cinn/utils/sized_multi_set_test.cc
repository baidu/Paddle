// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/sized_multi_set.h"

#include <gtest/gtest.h>

#include <vector>

namespace cinn {
namespace utils {

TEST(SizedMultiSet, PopMax) {
  SizedMultiSet<int> sized_multi_set(5);

  for (int i = 0; i < 10; ++i) {
    sized_multi_set.Push(i);
    if (i < 5) {
      EXPECT_EQ(sized_multi_set.Size(), static_cast<size_t>(i + 1));
      EXPECT_EQ(sized_multi_set.MaxValue(), i);
      EXPECT_EQ(sized_multi_set.MinValue(), 0);
    } else {
      EXPECT_EQ(sized_multi_set.Size(), 5);
      EXPECT_EQ(sized_multi_set.MaxValue(), 4);
      EXPECT_EQ(sized_multi_set.MinValue(), 0);
    }
  }

  std::vector<int> vec = sized_multi_set.ReturnAsContainer<std::vector<int>>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(vec[i], i);
  }

  for (int i = 0; i < 4; ++i) {
    sized_multi_set.Pop();
    EXPECT_EQ(sized_multi_set.Size(), static_cast<size_t>(4 - i));
    EXPECT_EQ(sized_multi_set.MaxValue(), static_cast<size_t>(3 - i));
    EXPECT_EQ(sized_multi_set.MinValue(), static_cast<size_t>(0));
  }
}

TEST(SizedMultiSet, PopMin) {
  SizedMultiSet<int> sized_multi_set(5, /* pop_max_when_full = */ false);
  for (int i = 0; i < 10; ++i) {
    sized_multi_set.Push(i);
    if (i < 5) {
      EXPECT_EQ(sized_multi_set.Size(), static_cast<size_t>(i + 1));
      EXPECT_EQ(sized_multi_set.MaxValue(), i);
      EXPECT_EQ(sized_multi_set.MinValue(), 0);
    } else {
      EXPECT_EQ(sized_multi_set.Size(), 5);
      EXPECT_EQ(sized_multi_set.MaxValue(), i);
      EXPECT_EQ(sized_multi_set.MinValue(), i - 4);
    }
  }

  std::vector<int> vec = sized_multi_set.ReturnAsContainer<std::vector<int>>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(vec[i], i + 5);
  }

  for (int i = 0; i < 4; ++i) {
    sized_multi_set.Pop();
    EXPECT_EQ(sized_multi_set.Size(), static_cast<size_t>(4 - i));
    EXPECT_EQ(sized_multi_set.MaxValue(), static_cast<size_t>(9));
    EXPECT_EQ(sized_multi_set.MinValue(), static_cast<size_t>(6 + i));
  }
}

}  // namespace utils
}  // namespace cinn
