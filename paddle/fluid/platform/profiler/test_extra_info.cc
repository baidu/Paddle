// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/profiler/extra_info.h"

using paddle::platform::ExtraInfo;

TEST(ExtraInfoTest, case0) {
  ExtraInfo instance;
  instance.AddExtraInfo(std::string("info1"), std::string("%d"), 20);
  instance.AddExtraInfo(std::string("info2"), std::string("%s"), "helloworld");
  std::unordered_map<std::string, std::string> map = instance.GetExtraInfo();
  EXPECT_EQ(map["info1"], "20");
  EXPECT_EQ(map["info2"], "helloworld");
  EXPECT_EQ(map.size(), 2u);
  instance.Clear();
  EXPECT_EQ(map.size(), 0u);
}
