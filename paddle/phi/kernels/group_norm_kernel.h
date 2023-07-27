// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/inference/tensorrt/plugin/common/groupNormPluginCommon.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/dense_tensor.h"

using paddle::inference::tensorrt::plugin::GroupNormNHWCParams;
namespace phi {

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* variance);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T, typename AccT = T>
class GroupNormDirectCUDAFunctor {
 public:
  void operator()(gpuStream_t stream,
                  const T* input,
                  std::vector<int> input_shape,
                  const T* bias,
                  const T* scale,
                  AccT* temp_variance,
                  int groups,
                  float eps,
                  T* output,
                  AccT* mean,
                  AccT* variance,
                  const DataLayout data_layout);
};
#endif

template <typename T>
class groupNormNHWCSum {
 public:
  void operator()(GroupNormNHWCParams<T>* params, const gpuStream_t stream);
};

template <typename T>
class groupNormNHWCScale {
 public:
  void operator()(const GroupNormNHWCParams<T>& params,
                  const gpuStream_t stream);
};

}  // namespace phi
