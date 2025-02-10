// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/nodes.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/type_promotion_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/dygraph_api.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/type_promotion.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

void PrintMeanMaxMin(const std::string& op,
                     const std::string& var,
                     const paddle::Tensor& x) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);

  std::cout << "LXJ shape Dbg: " << op << "--" << var << " shape = " << x.dims()
            << std::endl;
  auto mean_tensor = paddle::experimental::copy_to(
      paddle::experimental::mean(x, paddle::experimental::IntArray({}), false),
      phi::CPUPlace(),
      true);
  auto max_tensor = paddle::experimental::copy_to(
      paddle::experimental::max(x, paddle::experimental::IntArray({}), false),
      phi::CPUPlace(),
      true);
  auto min_tensor = paddle::experimental::copy_to(
      paddle::experimental::min(x, paddle::experimental::IntArray({}), false),
      phi::CPUPlace(),
      true);

  auto u =
      paddle::experimental::mean(x, paddle::experimental::IntArray({}), false);
  auto sub = paddle::experimental::subtract(x, u);
  auto pow = paddle::experimental::pow(sub, 2);
  auto out_ten = paddle::experimental::sum(
      pow, paddle::experimental::IntArray({}), pow.dtype(), false);

  auto out_tensor =
      paddle::experimental::copy_to(out_ten, phi::CPUPlace(), true);

  double mean, max, min, out;
  switch (mean_tensor.dtype()) {
    case phi::DataType::BOOL:
      mean = *mean_tensor.data<bool>();
      break;
    case phi::DataType::UINT8:
      mean = *mean_tensor.data<uint8_t>();
      break;
    case phi::DataType::INT8:
      mean = *mean_tensor.data<int8_t>();
      break;
    case phi::DataType::BFLOAT16:
      mean = *mean_tensor.data<phi::dtype::bfloat16>();
      break;
    case phi::DataType::FLOAT16:
      mean = *mean_tensor.data<phi::dtype::float16>();
      break;
    case phi::DataType::INT16:
      mean = *mean_tensor.data<int16_t>();
      break;
    case phi::DataType::UINT16:
      mean = *mean_tensor.data<uint16_t>();
      break;
    case phi::DataType::FLOAT32:
      mean = *mean_tensor.data<float>();
      break;
    case phi::DataType::INT32:
      mean = *mean_tensor.data<int>();
      break;
    case phi::DataType::FLOAT64:
      mean = *mean_tensor.data<double>();
      break;
    case phi::DataType::INT64:
      mean = *mean_tensor.data<int64_t>();
      break;
    case phi::DataType::UINT64:
      mean = *mean_tensor.data<uint64_t>();
      break;
    default:
      mean = -1;
      break;
  }
  switch (max_tensor.dtype()) {
    case phi::DataType::BOOL:
      max = *max_tensor.data<bool>();
      break;
    case phi::DataType::UINT8:
      max = *max_tensor.data<uint8_t>();
      break;
    case phi::DataType::INT8:
      max = *max_tensor.data<int8_t>();
      break;
    case phi::DataType::BFLOAT16:
      max = *max_tensor.data<phi::dtype::bfloat16>();
      break;
    case phi::DataType::FLOAT16:
      max = *max_tensor.data<phi::dtype::float16>();
      break;
    case phi::DataType::INT16:
      max = *max_tensor.data<int16_t>();
      break;
    case phi::DataType::UINT16:
      max = *max_tensor.data<uint16_t>();
      break;
    case phi::DataType::FLOAT32:
      max = *max_tensor.data<float>();
      break;
    case phi::DataType::INT32:
      max = *max_tensor.data<int>();
      break;
    case phi::DataType::FLOAT64:
      max = *max_tensor.data<double>();
      break;
    case phi::DataType::INT64:
      max = *max_tensor.data<int64_t>();
      break;
    case phi::DataType::UINT64:
      max = *max_tensor.data<uint64_t>();
      break;
    default:
      max = -1;
      break;
  }
  switch (min_tensor.dtype()) {
    case phi::DataType::BOOL:
      min = *min_tensor.data<bool>();
      break;
    case phi::DataType::UINT8:
      min = *min_tensor.data<uint8_t>();
      break;
    case phi::DataType::INT8:
      min = *min_tensor.data<int8_t>();
      break;
    case phi::DataType::BFLOAT16:
      min = *min_tensor.data<phi::dtype::bfloat16>();
      break;
    case phi::DataType::FLOAT16:
      min = *min_tensor.data<phi::dtype::float16>();
      break;
    case phi::DataType::INT16:
      min = *min_tensor.data<int16_t>();
      break;
    case phi::DataType::UINT16:
      min = *min_tensor.data<uint16_t>();
      break;
    case phi::DataType::FLOAT32:
      min = *min_tensor.data<float>();
      break;
    case phi::DataType::INT32:
      min = *min_tensor.data<int>();
      break;
    case phi::DataType::FLOAT64:
      min = *min_tensor.data<double>();
      break;
    case phi::DataType::INT64:
      min = *min_tensor.data<int64_t>();
      break;
    case phi::DataType::UINT64:
      min = *min_tensor.data<uint64_t>();
      break;
    default:
      min = -1;
      break;
  }
  switch (out_tensor.dtype()) {
    case phi::DataType::BOOL:
      out = *out_tensor.data<bool>();
      break;
    case phi::DataType::UINT8:
      out = *out_tensor.data<uint8_t>();
      break;
    case phi::DataType::INT8:
      out = *out_tensor.data<int8_t>();
      break;
    case phi::DataType::BFLOAT16:
      out = *out_tensor.data<phi::dtype::bfloat16>();
      break;
    case phi::DataType::FLOAT16:
      out = *out_tensor.data<phi::dtype::float16>();
      break;
    case phi::DataType::INT16:
      out = *out_tensor.data<int16_t>();
      break;
    case phi::DataType::UINT16:
      out = *out_tensor.data<uint16_t>();
      break;
    case phi::DataType::FLOAT32:
      out = *out_tensor.data<float>();
      break;
    case phi::DataType::INT32:
      out = *out_tensor.data<int>();
      break;
    case phi::DataType::FLOAT64:
      out = *out_tensor.data<double>();
      break;
    case phi::DataType::INT64:
      out = *out_tensor.data<int64_t>();
      break;
    case phi::DataType::UINT64:
      out = *out_tensor.data<uint64_t>();
      break;
    default:
      out = -1;
      break;
  }
  auto x_numel = x.numel();

  egr::Controller::Instance().SetHasGrad(grad_tmp);

  std::cout << "LXJ Dbg: " << op << "--" << var << ": mean = " << mean
            << ", max = " << max << ", min = " << min
            << ", std = " << out / x_numel << std::endl;
}

void PrintMeanMaxMin(const std::string& op,
                     const std::string& var,
                     const paddle::optional<paddle::Tensor>& x) {
  if (x) {
    PrintMeanMaxMin(op, var, *x);
  }
}

void PrintMeanMaxMin(const std::string& op,
                     const std::string& var,
                     const std::vector<paddle::Tensor>& x) {
  for (auto& t : x) {
    PrintMeanMaxMin(op, var, t);
  }
}

void PrintMeanMaxMin(const std::string& op,
                     const std::string& var,
                     const paddle::optional<std::vector<paddle::Tensor>>& x) {
  if (x) {
    for (auto& t : *x) {
      PrintMeanMaxMin(op, var, t);
    }
  }
}
