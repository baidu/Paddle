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

#include <gtest/gtest.h>

#include "paddle/common/enforce.h"
#include "paddle/phi/core/enforce.h"

class FatalClass {
 public:
  FatalClass() {}
  ~FatalClass() { PADDLE_FATAL("fatal occurred in deconstructor!"); }
};

void throw_exception_in_func() {
  FatalClass test_case;
  PADDLE_THROW(::common::errors::External("throw exception in func"));
}

void terminate_in_func() { FatalClass test_case; }

TEST(paddle_fatal_test, base) {
  EXPECT_FALSE(::common::enforce::IsPaddleFatalSkip());
  EXPECT_DEATH(terminate_in_func(), "fatal occurred in deconstructor!.*");
  EXPECT_THROW(throw_exception_in_func(), common::enforce::EnforceNotMet);
  EXPECT_FALSE(::common::enforce::IsPaddleFatalSkip());
  ::common::enforce::SkipPaddleFatal(true);
  // skip fatal.
  terminate_in_func();
  // unskip paddle fatal.
  ::common::enforce::SkipPaddleFatal(false);
}
