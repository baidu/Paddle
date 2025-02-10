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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/stats.h"

#include "gtest/gtest.h"

namespace paddle {
namespace memory {
TEST(XPUOverloadAllocTest, EnvTest) {
  setenv("XPUAPI_DEFAULT_SIZE", "4096", 1);
  // use alloc overload
  unsetenv("XPU_PADDLE_DISABLE_ALLOC_OVERLOAD");
  phi::XPUContext dev_ctx_overload(
      phi::XPUPlace(phi::backends::xpu::GetXPUCurrentDeviceId()));
  EXPECT_STREQ(dev_ctx_overload.x_context()->get_option("XPUAPI_DEFAULT_SIZE"),
               "1");
  EXPECT_NE(dev_ctx_overload.x_context()->overload_alloc_gm, nullptr);
  // do not use alloc overload
  setenv("XPU_PADDLE_DISABLE_ALLOC_OVERLOAD", "1", 1);
  phi::XPUContext dev_ctx_origin(
      phi::XPUPlace(phi::backends::xpu::GetXPUCurrentDeviceId()));
  EXPECT_STREQ(dev_ctx_origin.x_context()->get_option("XPUAPI_DEFAULT_SIZE"),
               "4096");
  EXPECT_EQ(dev_ctx_origin.x_context()->overload_alloc_gm, nullptr);
  unsetenv("XPU_PADDLE_DISABLE_ALLOC_OVERLOAD");
  unsetenv("XPUAPI_DEFAULT_SIZE");
}

TEST(XPUOverloadAllocTest, BasicTest) {
  phi::XPUContext dev_ctx(
      phi::XPUPlace(phi::backends::xpu::GetXPUCurrentDeviceId()));
  int numel = 64;
  int alignment = phi::backends::xpu::XPUMinChunkSize();
  int expected_alloc_size =
      allocation::AlignedSize(numel * sizeof(int), alignment);
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int pre_alloc_value = DEVICE_MEMORY_STAT_CURRENT_VALUE(
      Allocated, dev_ctx.GetPlace().GetDeviceId());
  int* buffer = RAII_GUARD.alloc<int>(numel);
  int after_alloc_value = DEVICE_MEMORY_STAT_CURRENT_VALUE(
      Allocated, dev_ctx.GetPlace().GetDeviceId());
  EXPECT_NE(buffer, nullptr);
  EXPECT_EQ(after_alloc_value - pre_alloc_value, expected_alloc_size);
}

TEST(XPUOverloadAllocTest, NestedScopeTest) {
  phi::XPUContext dev_ctx(
      phi::XPUPlace(phi::backends::xpu::GetXPUCurrentDeviceId()));
  xpu::ctx_guard RAII_GUARD1(dev_ctx.x_context());
  int pre_alloc_value = DEVICE_MEMORY_STAT_CURRENT_VALUE(
      Allocated, dev_ctx.GetPlace().GetDeviceId());
  int* buffer_outer = RAII_GUARD1.alloc<int>(64);
  EXPECT_NE(buffer_outer, nullptr);
  {
    // The destruction of inner guard should not free the memory allocated from
    // outer guard.
    xpu::ctx_guard RAII_GUARD2(dev_ctx.x_context());
    int* buffer_inner = RAII_GUARD2.alloc<int>(64);
    EXPECT_NE(buffer_inner, nullptr);
  }
  int post_alloc_value = DEVICE_MEMORY_STAT_CURRENT_VALUE(
      Allocated, dev_ctx.GetPlace().GetDeviceId());
  EXPECT_NE(post_alloc_value, pre_alloc_value);
}

TEST(XPUOverloadAllocTest, MultiStreamTest) {
  // Test whether stream 1 use the memory poll of stream 0.
  int size = 64;
  setenv("XPU_CDNN_CLUSTER_PARALLEL", "1", 1);
  phi::XPUContext dev_ctx(
      phi::XPUPlace(phi::backends::xpu::GetXPUCurrentDeviceId()));
  xpu::ctx_guard RAII_GUARD0(dev_ctx.x_context(0));
  xpu::ctx_guard RAII_GUARD1(dev_ctx.x_context(1));
  int pre_alloc_value = DEVICE_MEMORY_STAT_CURRENT_VALUE(
      Allocated, dev_ctx.GetPlace().GetDeviceId());
  int* buffer0 = RAII_GUARD1.alloc<int>(size);
  EXPECT_NE(buffer0, nullptr);
  {
    int* buffer1 = RAII_GUARD0.alloc<int>(size);
    EXPECT_NE(buffer1, nullptr);
  }
  int post_alloc_value = DEVICE_MEMORY_STAT_CURRENT_VALUE(
      Allocated, dev_ctx.GetPlace().GetDeviceId());

  EXPECT_NE(pre_alloc_value, post_alloc_value);
  unsetenv("XPU_CDNN_CLUSTER_PARALLEL");
}
}  // namespace memory
}  // namespace paddle
