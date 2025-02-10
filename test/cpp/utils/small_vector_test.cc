// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/utils/small_vector.h"

#include <cstdlib>
#include <ctime>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"

template <typename T, unsigned N>
static std::vector<T> ToStdVector(const paddle::small_vector<T, N> &vec) {
  std::vector<T> std_vec;
  std_vec.reserve(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    std_vec.emplace_back(vec[i]);
  }
  return std_vec;
}

template <size_t N>
void SmallVectorCheck(size_t n) {
  std::srand(std::time(nullptr));

  std::vector<int> std_vec;
  paddle::small_vector<int, N> vec;

  for (size_t i = 0; i < n; ++i) {
    int value = rand();  // NOLINT

    std_vec.emplace_back(value);
    vec.emplace_back(value);

    PADDLE_ENFORCE_EQ(std_vec.size(),
                      vec.size(),
                      common::errors::InvalidArgument(
                          "The sizes of std_vec and vec should be equal, but "
                          "received std_vec.size() = %zu and vec.size() = %zu.",
                          std_vec.size(),
                          vec.size()));
    PADDLE_ENFORCE_EQ(
        std_vec.back(),
        vec.back(),
        common::errors::InvalidArgument(
            "The last elements of std_vec and vec should be equal, but "
            "received std_vec.back() = %d and vec.back() = %d.",
            std_vec.back(),
            vec.back()));

    PADDLE_ENFORCE_EQ(vec.back(),
                      value,
                      common::errors::InvalidArgument(
                          "The last element of vec should be equal to value, "
                          "but received vec.back() = %d and value = %d.",
                          vec.back(),
                          value));
  }

  bool is_equal = (std_vec == ToStdVector(vec));

  PADDLE_ENFORCE_EQ(
      is_equal,
      true,
      common::errors::InvalidArgument(
          "The std_vec and vec should be equal, but they are not."));

  for (size_t i = 0; i < n; ++i) {
    PADDLE_ENFORCE_EQ(std_vec.size(),
                      vec.size(),
                      common::errors::InvalidArgument(
                          "The sizes of std_vec and vec should be equal, but "
                          "received std_vec.size() = %zu and vec.size() = %zu.",
                          std_vec.size(),
                          vec.size()));
    PADDLE_ENFORCE_EQ(
        std_vec.back(),
        vec.back(),
        common::errors::InvalidArgument(
            "The last elements of std_vec and vec should be equal, but "
            "received std_vec.back() = %d and vec.back() = %d.",
            std_vec.back(),
            vec.back()));
    std_vec.pop_back();
    vec.pop_back();
    PADDLE_ENFORCE_EQ(std_vec.size(),
                      vec.size(),
                      common::errors::InvalidArgument(
                          "The sizes of std_vec and vec should be equal, but "
                          "received std_vec.size() = %zu and vec.size() = %zu.",
                          std_vec.size(),
                          vec.size()));
  }

  PADDLE_ENFORCE_EQ(
      std_vec.size(),
      static_cast<size_t>(0),
      common::errors::InvalidArgument(
          "The size of std_vec should be 0, but received vec.size() = %zu.",
          vec.size()));
  PADDLE_ENFORCE_EQ(
      vec.size(),
      static_cast<size_t>(0),
      common::errors::InvalidArgument(
          "The size of vec should be 0, but received vec.size() = %zu.",
          vec.size()));
}

TEST(small_vector, small_vector) {
  for (size_t i = 0; i < 20; ++i) {
    SmallVectorCheck<1>(i);
    SmallVectorCheck<10>(i);
    SmallVectorCheck<15>(i);
    SmallVectorCheck<20>(i);
    SmallVectorCheck<21>(i);
    SmallVectorCheck<25>(i);
  }
}
