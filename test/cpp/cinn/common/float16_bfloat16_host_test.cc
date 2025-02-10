// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <random>
#include <vector>

#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/float16.h"

namespace cinn {
namespace common {

std::vector<float16> test_fp16_host_kernel(const float16* x,
                                           const float16* y,
                                           const int num) {
  std::vector<float16> out(num);
  for (int idx = 0; idx < num; ++idx) {
    float16 x_i = x[idx], y_i = y[idx];
    x_i += float16(1);

    out[idx] = (x_i + y_i) * (x_i - y_i);
  }
  return out;
}

std::vector<bfloat16> test_bf16_host_kernel(const bfloat16* x,
                                            const bfloat16* y,
                                            const int num) {
  std::vector<bfloat16> out(num);
  for (int idx = 0; idx < num; ++idx) {
    bfloat16 x_i = x[idx], y_i = y[idx];
    x_i += bfloat16(1);

    out[idx] = (x_i + y_i) * (x_i - y_i);
  }
  return out;
}

std::vector<float> test_fp32_host_kernel(const float* x,
                                         const float* y,
                                         const int num) {
  std::vector<float> out(num);
  for (int idx = 0; idx < num; ++idx) {
    float x_i = x[idx], y_i = y[idx];
    x_i += 1.0f;

    out[idx] = (x_i + y_i) * (x_i - y_i);
  }
  return out;
}

TEST(FP16_BF16, basic_host) {
  int num = 2048;
  // int num = 2;
  std::vector<float16> x_fp16(num), y_fp16(num);
  std::vector<bfloat16> x_bf16(num), y_bf16(num);
  std::vector<float> x_fp32(num), y_fp32(num);

  std::random_device r;
  std::default_random_engine eng(r());
  std::uniform_real_distribution<float> dis(1e-5f, 1.0f);

  for (int i = 0; i < num; ++i) {
    x_fp16[i] = x_fp32[i] = dis(eng);
    y_fp16[i] = y_fp32[i] = dis(eng);

    x_fp16[i] = x_fp32[i];
    y_fp16[i] = y_fp32[i];

    x_bf16[i] = x_fp32[i];
    y_bf16[i] = y_fp32[i];
  }

  auto out_fp16 = test_fp16_host_kernel(x_fp16.data(), y_fp16.data(), num);
  ASSERT_EQ(out_fp16.size(), num);

  auto out_bf16 = test_bf16_host_kernel(x_bf16.data(), y_bf16.data(), num);
  ASSERT_EQ(out_bf16.size(), num);

  auto out_fp32 = test_fp32_host_kernel(x_fp32.data(), y_fp32.data(), num);
  ASSERT_EQ(out_fp32.size(), num);

  for (int i = 0; i < num; ++i) {
    ASSERT_NEAR(static_cast<float>(out_fp16[i]), out_fp32[i], 1e-2f);
    ASSERT_NEAR(static_cast<float>(out_bf16[i]), out_fp32[i], 1e-1f);
  }
}

}  // namespace common
}  // namespace cinn
