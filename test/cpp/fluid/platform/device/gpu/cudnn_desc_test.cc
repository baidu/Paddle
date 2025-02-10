// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/device/gpu/gpu_dnn.h"

namespace paddle::platform {

TEST(TensorDescriptor, Empty) {
  phi::backends::gpu::ActivationDescriptor a;
  phi::backends::gpu::TensorDescriptor t;
  phi::backends::gpu::TensorDescriptor t1;
  phi::backends::gpu::TensorDescriptor *t11 =
      new phi::backends::gpu::TensorDescriptor();
  delete t11;
  std::unique_ptr<phi::backends::gpu::TensorDescriptor> tt(
      new phi::backends::gpu::TensorDescriptor());
}

TEST(TensorDescriptor, Normal) {
  phi::DenseTensor tt;
  tt.Resize({2, 3, 4});
  tt.mutable_data<float>(phi::CPUPlace());

  phi::backends::gpu::TensorDescriptor desc;
  desc.set(tt);
  EXPECT_TRUE(desc.desc() != nullptr);
}

}  // namespace paddle::platform
