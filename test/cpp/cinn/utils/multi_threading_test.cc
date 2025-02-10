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

#include "paddle/cinn/utils/multi_threading.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "paddle/common/enforce.h"

namespace cinn {
namespace utils {

TEST(JobDispatcher, SequenceDispatcher) {
  std::unique_ptr<JobDispatcher> dispatcher =
      std::make_unique<SequenceDispatcher>(1, 3);
  ASSERT_EQ(1, dispatcher->Next());
  ASSERT_EQ(2, dispatcher->Next());
  // check reach the end
  ASSERT_EQ(-1, dispatcher->Next());
}

TEST(parallel_run, Basic) {
  std::vector<int> results(100, -1);
  auto worker_fn = [&results](int index) {
    PADDLE_ENFORCE_LT(index,
                      results.size(),
                      ::common::errors::InvalidArgument("invalid index!"));
    results[index] = index;
  };
  // check process every index in the extent of [0, 100) with step 1
  parallel_run(worker_fn, SequenceDispatcher(0, 100), 2);
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(results[i], i);
  }

  // check only indexes in the extent of [0, 100) with step 3 are processed
  results.assign(100, -1);
  parallel_run(worker_fn, SequenceDispatcher(0, 100, 3), 3);
  for (int i = 0; i < 100; ++i) {
    if (i % 3 == 0) {
      ASSERT_EQ(results[i], i);
    } else {
      ASSERT_EQ(results[i], -1);
    }
  }
}

}  // namespace utils
}  // namespace cinn
