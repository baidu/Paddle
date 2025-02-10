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

#include "paddle/cinn/runtime/cinn_runtime.h"

#include <gtest/gtest.h>

TEST(buffer, basic) {
  auto* buffer =
      cinn_buffer_t::new_(cinn_x86_device, cinn_float32_t(), {3, 10});
  ASSERT_TRUE(buffer);
  ASSERT_TRUE(buffer->device_interface);
  ASSERT_EQ(buffer->device_interface, cinn_x86_device_interface());
  buffer->device_interface->impl->malloc(NULL, buffer);
  auto* data = reinterpret_cast<float*>(buffer->memory);
  data[0] = 0.f;
  data[1] = 1.f;
  EXPECT_EQ(data[0], 0.f);
  EXPECT_EQ(data[1], 1.f);
}

TEST(cinn_print_debug_string, basic) {
  cinn_print_debug_string("hello world");
  cinn_print_debug_string("should be 1, %d", 1);
  int a = 1;
  cinn_print_debug_string("should be pointer, %p", &a);
  cinn_print_debug_string("should be 1, %d", a);
  cinn_print_debug_string("v3[%d %d %d], ", 1, 2, 3);
}

TEST(cinn_args_construct, basic) {
  cinn_pod_value_t arr[4];
  cinn_pod_value_t a0(0);
  cinn_pod_value_t a1(1);
  cinn_pod_value_t a2(2);
  cinn_pod_value_t a3(3);
  cinn_args_construct(arr, 4, &a0, &a1, &a2, &a3);
  for (int i = 0; i < 4; i++) ASSERT_EQ((int)arr[i], i);
}
