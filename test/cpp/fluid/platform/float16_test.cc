/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/float16.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

namespace paddle::platform {
using float16 = phi::dtype::float16;
using namespace phi::dtype;  // NOLINT

TEST(float16, conversion_cpu) {
  // Conversion from float
  EXPECT_EQ(float16(1.0f).x, 0x3c00);
  EXPECT_EQ(float16(0.5f).x, 0x3800);
  EXPECT_EQ(float16(0.33333f).x, 0x3555);
  EXPECT_EQ(float16(0.0f).x, 0x0000);
  EXPECT_EQ(float16(-0.0f).x, 0x8000);
  EXPECT_EQ(float16(65504.0f).x, 0x7bff);
  EXPECT_EQ(float16(65536.0f).x, 0x7c00);

  // Conversion from double
  EXPECT_EQ(float16(1.0).x, 0x3c00);
  EXPECT_EQ(float16(0.5).x, 0x3800);
  EXPECT_EQ(float16(0.33333).x, 0x3555);
  EXPECT_EQ(float16(0.0).x, 0x0000);
  EXPECT_EQ(float16(-0.0).x, 0x8000);
  EXPECT_EQ(float16(65504.0).x, 0x7bff);
  EXPECT_EQ(float16(65536.0).x, 0x7c00);

  // Conversion from int
  EXPECT_EQ(float16(-1).x, 0xbc00);
  EXPECT_EQ(float16(0).x, 0x0000);
  EXPECT_EQ(float16(1).x, 0x3c00);
  EXPECT_EQ(float16(2).x, 0x4000);
  EXPECT_EQ(float16(3).x, 0x4200);

  // Conversion from bool
  EXPECT_EQ(float16(true).x, 0x3c00);
  EXPECT_EQ(float16(false).x, 0x0000);

  // Assignment operator
  float16 v_assign;
  v_assign = float16(0);
  EXPECT_EQ(v_assign.x, 0x0000);
  v_assign = 0.5f;
  EXPECT_EQ(v_assign.x, 0x3800);
  v_assign = 0.33333;
  EXPECT_EQ(v_assign.x, 0x3555);
  v_assign = -1;
  EXPECT_EQ(v_assign.x, 0xbc00);
  v_assign = true;
  EXPECT_EQ(v_assign.x, 0x3c00);

  // Conversion operator
  EXPECT_EQ(static_cast<float>(float16(0.5f)), 0.5f);
  EXPECT_NEAR(static_cast<double>(float16(0.33333)), 0.33333, 0.0001);
  EXPECT_EQ(static_cast<int>(float16(-1)), -1);
  EXPECT_EQ(static_cast<bool>(float16(true)), true);
}

TEST(float16, arithmetic_cpu) {
  EXPECT_EQ(static_cast<float>(float16(1) + float16(1)), 2);
  EXPECT_EQ(static_cast<float>(float16(5) + float16(-5)), 0);
  EXPECT_NEAR(
      static_cast<float>(float16(0.33333f) + float16(0.66667f)), 1.0f, 0.001);
  EXPECT_EQ(static_cast<float>(float16(3) - float16(5)), -2);
  EXPECT_NEAR(static_cast<float>(float16(0.66667f) - float16(0.33333f)),
              0.33334f,
              0.001);
  EXPECT_NEAR(static_cast<float>(float16(3.3f) * float16(2.0f)), 6.6f, 0.01);
  EXPECT_NEAR(static_cast<float>(float16(-2.1f) * float16(-3.0f)), 6.3f, 0.01);
  EXPECT_NEAR(
      static_cast<float>(float16(2.0f) / float16(3.0f)), 0.66667f, 0.001);
  EXPECT_EQ(static_cast<float>(float16(1.0f) / float16(2.0f)), 0.5f);
  EXPECT_EQ(static_cast<float>(-float16(512.0f)), -512.0f);
  EXPECT_EQ(static_cast<float>(-float16(-512.0f)), 512.0f);
}

TEST(float16, comparison_cpu) {
  EXPECT_TRUE(float16(1.0f) == float16(1.0f));
  EXPECT_FALSE(float16(-1.0f) == float16(-0.5f));
  EXPECT_TRUE(float16(1.0f) != float16(0.5f));
  EXPECT_FALSE(float16(-1.0f) != float16(-1.0f));
  EXPECT_TRUE(float16(1.0f) < float16(2.0f));
  EXPECT_FALSE(float16(-1.0f) < float16(-1.0f));
  EXPECT_TRUE(float16(1.0f) <= float16(1.0f));
  EXPECT_TRUE(float16(2.0f) > float16(1.0f));
  EXPECT_FALSE(float16(-2.0f) > float16(-2.0f));
  EXPECT_TRUE(float16(2.0f) >= float16(2.0f));

  EXPECT_TRUE(float16(0.0f) == float16(-0.0f));
  EXPECT_TRUE(float16(0.0f) <= float16(-0.0f));
  EXPECT_TRUE(float16(0.0f) >= float16(-0.0f));
  EXPECT_FALSE(float16(0.0f) < float16(-0.0f));
  EXPECT_FALSE(float16(-0.0f) < float16(0.0f));
  EXPECT_FALSE(float16(0.0f) > float16(-0.0f));
  EXPECT_FALSE(float16(-0.0f) > float16(0.0f));
}

TEST(float16, lod_tensor_cpu) {
  phi::DenseTensor lod_tensor;

  std::vector<float16> input_data = {
      float16(1.0f), float16(0.5f), float16(0.33333f), float16(0.0f)};
  EXPECT_EQ(input_data[0].x, 0x3c00);
  EXPECT_EQ(input_data[1].x, 0x3800);
  EXPECT_EQ(input_data[2].x, 0x3555);
  EXPECT_EQ(input_data[3].x, 0x0000);

  lod_tensor.Resize({4, 1});
  lod_tensor.set_lod(phi::LegacyLoD({{0, 2, 4}}));
  float16* data_ptr = lod_tensor.mutable_data<float16>(CPUPlace());

  EXPECT_NE(data_ptr, nullptr);
  EXPECT_EQ(input_data.size(), static_cast<size_t>(lod_tensor.numel()));
  for (size_t i = 0; i < input_data.size(); ++i) {
    data_ptr[i] = input_data[i];
    EXPECT_EQ(data_ptr[i].x, input_data[i].x);
  }
}

TEST(float16, floating) {
  // compile time assert.
  PADDLE_ENFORCE_EQ(
      std::is_floating_point<float16>::value,
      true,
      common::errors::Unavailable("The float16 support in CPU failed."));
}

TEST(float16, print) {
  float16 a = float16(1.0f);
  std::cout << a << std::endl;
}

// CPU test
TEST(float16, isinf) {
  float16 a;
  a.x = 0x7c00;
  float16 b = float16(INFINITY);
  float16 c = static_cast<float16>(INFINITY);
  EXPECT_EQ(std::isinf(a), true);
  EXPECT_EQ(std::isinf(b), true);
  EXPECT_EQ(std::isinf(c), true);
}

TEST(float16, isnan) {
  float16 a;
  a.x = 0x7fff;
  float16 b = float16(NAN);
  float16 c = static_cast<float16>(NAN);
  EXPECT_EQ(std::isnan(a), true);
  EXPECT_EQ(std::isnan(b), true);
  EXPECT_EQ(std::isnan(c), true);
}

}  // namespace paddle::platform
