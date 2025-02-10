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

#pragma once
#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/phi/backends/context_pool.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/random.h>

#include "paddle/phi/core/generator.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#endif

#include "glog/logging.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace funcs {

template <typename T>
inline void UniformRealDistribution(T* data,
                                    const int64_t& size,
                                    const float& min,
                                    const float& max,
                                    const unsigned int seed) {
  VLOG(4) << "[CPU] UniformRandomKernel<T>";
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  auto engine = phi::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <>
inline void UniformRealDistribution(phi::dtype::bfloat16* data,
                                    const int64_t& size,
                                    const float& min,
                                    const float& max,
                                    const unsigned int seed) {
  VLOG(4) << "[CPU] UniformRandomKernel<bfloat16>";
  std::uniform_real_distribution<float> dist(min, max);
  auto engine = phi::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<phi::dtype::bfloat16>(dist(*engine));
  }
}

inline std::vector<int64_t> GetNewDataFromShapeTensor(
    const phi::DenseTensor* new_data_tensor) {
  phi::DenseTensor cpu_starts_tensor;
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().Get(cpu_starts_tensor.place());
  if (new_data_tensor->dtype() == phi::DataType::INT64) {
    auto* new_data = new_data_tensor->data<int64_t>();
    if (new_data_tensor->place().GetType() == phi::AllocationType::GPU) {
      phi::Copy(*dev_ctx,
                *new_data_tensor,
                phi::CPUPlace(),
                true,
                &cpu_starts_tensor);
      new_data = cpu_starts_tensor.data<int64_t>();
    }
    std::vector<int64_t> vec_new_data(new_data,
                                      new_data + new_data_tensor->numel());
    return vec_new_data;
  } else if (new_data_tensor->dtype() == phi::DataType::INT32) {
    auto* new_data = new_data_tensor->data<int32_t>();
    std::vector<int64_t> vec_new_data;
    if (new_data_tensor->place().GetType() == phi::AllocationType::GPU) {
      phi::Copy(*dev_ctx,
                *new_data_tensor,
                phi::CPUPlace(),
                true,
                &cpu_starts_tensor);
      new_data = cpu_starts_tensor.data<int32_t>();
    }
    for (int i = 0; i < new_data_tensor->numel(); ++i) {
      vec_new_data.push_back(static_cast<int64_t>(*(new_data + i)));
    }
    return vec_new_data;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Expected dtype of ShapeTensor must be int32, int64. But got "
        "unsupported dtype: %s.",
        new_data_tensor->dtype()));
  }
}

inline std::vector<int64_t> GetNewDataFromShapeTensorList(
    const std::vector<const phi::DenseTensor*>& list_new_shape_tensor) {
  phi::DenseTensor temp;
  auto* dev_ctx = phi::DeviceContextPool::Instance().Get(temp.place());
  std::vector<int64_t> vec_new_shape;
  vec_new_shape.reserve(list_new_shape_tensor.size());
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        common::make_ddim({1}),
        common::errors::InvalidArgument(
            "Shape of dim tensor in uniform_random_op should be [1]"
            "But received tensor's dim=%s.",
            tensor->dims()));

    if (tensor->dtype() == phi::DataType::INT32) {
      if (tensor->place().GetType() == phi::AllocationType::GPU) {
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(static_cast<int64_t>(*temp.data<int32_t>()));
      } else {
        vec_new_shape.push_back(static_cast<int64_t>(*tensor->data<int32_t>()));
      }
    } else if (tensor->dtype() == phi::DataType::INT64) {
      if (tensor->place().GetType() == phi::AllocationType::GPU) {
        phi::DenseTensor temp;
        phi::Copy(*dev_ctx, *tensor, phi::CPUPlace(), true, &temp);
        vec_new_shape.push_back(*temp.data<int64_t>());
      } else {
        vec_new_shape.push_back(*tensor->data<int64_t>());
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Expected dtype of ShapeTensorList of %d-th must be int32, int64. "
          "But got "
          "unsupported dtype: %s.",
          i,
          phi::DataTypeToString(tensor->dtype())));
    }
  }

  return vec_new_shape;
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  __host__ __device__ UniformGenerator(
      T min, T max, int seed, int diag_num, int diag_step, T diag_val)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
  }
};

template <typename T>
void UniformRandom(const phi::GPUContext& dev_ctx,
                   phi::DenseTensor* tensor,
                   int attr_seed,
                   float attr_min,
                   float attr_max,
                   int attr_diag_num,
                   int attr_diag_step,
                   float attr_diag_val) {
  int64_t size = tensor->numel();
  T* data = dev_ctx.Alloc<T>(tensor);
  if (size <= 0) return;
  unsigned int seed = static_cast<unsigned int>(attr_seed);

  T min = static_cast<T>(attr_min);
  T max = static_cast<T>(attr_max);
  unsigned int diag_num = static_cast<unsigned int>(attr_diag_num);
  unsigned int diag_step = static_cast<unsigned int>(attr_diag_step);
  T diag_val = static_cast<T>(attr_diag_val);

  if (seed == 0) {
    // Use global Generator seed
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    phi::funcs::uniform_distribution<MT> dist;
    phi::funcs::uniform_real_transform<MT> trans(min, max);
    phi::funcs::distribution_and_transform<T>(dev_ctx, tensor, dist, trans);
  } else {
    // Use OP seed
    auto func =
        UniformGenerator<T>(min, max, seed, diag_num, diag_step, diag_val);
    phi::IndexKernel<T, UniformGenerator<T>>(dev_ctx, tensor, func);
  }
}
#endif
}  // namespace funcs
}  // namespace phi
