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

#include <gtest/gtest.h>

#include "paddle/cinn/hlir/pe/schedule.h"

namespace cinn {
namespace hlir {
namespace pe {
using ir::Tensor;

TEST(load_x86_params, load_x86_params) {
  auto &res = ScheduleParam::get_x86_instance().GetParam();
  std::string key =
      "X86ScheduleConv input 1 3 224 224 weight 64 3 7 7 stride 2 2 padding 3 "
      "3 dilation 1 1";
  ASSERT_EQ(res.count(key), 1);

  absl::flat_hash_map<std::string, int> conv2d_factors;
  auto target = cinn::common::DefaultHostTarget();
  std::vector<int> shape_input = {1, 64, 56, 56};
  std::vector<int> shape_weights = {64, 64, 3, 3};
  std::vector<int> strides = {1, 1};
  std::vector<int> pads = {1, 1};
  std::vector<int> dilations = {1, 1};
  key =
      GenerateX86ConvKey(shape_input, shape_weights, strides, pads, dilations);
  GetConv2dFactors(&conv2d_factors, -1, -1, -1, -1, -1, Float(32), target, key);
  int ic_bn_size = conv2d_factors["ic_bn"];
  int oc_bn_size = conv2d_factors["oc_bn"];
  int fc_bn_size = conv2d_factors["fc_bn"];
  int ow_bn_size = conv2d_factors["ow_bn"];
  int unroll_kw = conv2d_factors["unroll_kw"];
  ASSERT_EQ(ic_bn_size, 64);
  ASSERT_EQ(fc_bn_size, 64);
  ASSERT_EQ(oc_bn_size, 32);
  ASSERT_EQ(ow_bn_size, 7);
  ASSERT_EQ(unroll_kw, 1);
}

TEST(load_cuda_params, load_cuda_params) {
  auto &res = ScheduleParam::get_cuda_instance().GetParam();
  if (res.empty()) {
    CreateCudaSerialData();
    LoadSerialData(&res);
  }
  std::string key = "CudaDirectConvSchedule 1 3 230 230 64 3 7 7 1 64 112 112";
  ASSERT_EQ(res.count(key), 1);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
