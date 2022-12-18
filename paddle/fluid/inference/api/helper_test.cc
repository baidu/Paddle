/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/api/helper.h"

#include "gtest/gtest.h"

namespace paddle {

TEST(inference_api_helper, DataType) {
  ASSERT_TRUE(
      paddle::inference::IsFloatVar(paddle::framework::proto::VarType::FP64));
  ASSERT_TRUE(
      paddle::inference::IsFloatVar(paddle::framework::proto::VarType::FP32));
  ASSERT_TRUE(
      paddle::inference::IsFloatVar(paddle::framework::proto::VarType::FP16));
  ASSERT_TRUE(
      paddle::inference::IsFloatVar(paddle::framework::proto::VarType::BF16));

  ASSERT_FALSE(
      paddle::inference::IsFloatVar(paddle::framework::proto::VarType::INT32));
}

}  // namespace paddle
