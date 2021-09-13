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
#pragma once

#include <cstring>
#include <random>
#include <string>

#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename MaskType, int VecSize>
__global__ void DropoutGradCUDAKernel(const T* dout, const MaskType* mask,
                                      const T factor, const int64_t size,
                                      T* dx) {
  using LoadT = platform::AlignedVector<T, VecSize>;
  using MaskLoadT = platform::AlignedVector<MaskType, VecSize>;

  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    LoadT dout_val;
    platform::Load<T, VecSize>(&dout[i], &dout_val);

    MaskLoadT mask_val;
    platform::Load<MaskType, VecSize>(&mask[i], &mask_val);

    LoadT dx_val;

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      dx_val[j] = dout_val[j] * static_cast<T>(mask_val[j]) * factor;
    }

    platform::Store<T, VecSize>(dx_val, &dx[i]);
  }
}
#endif

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class CPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      size_t size = framework::product(mask->dims());

      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        std::memset(y_data, 0, size * sizeof(*y_data));        // NOLINT
        std::memset(mask_data, 0, size * sizeof(*mask_data));  // NOLINT
        return;
      }
      // std::minstd_rand engine;
      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      int seed_data = 0;
      if (seed) {
        seed_data = *(seed->data<int>());
      } else {
        seed_data =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : 0;
      }
      auto engine = framework::GetCPURandomEngine(seed_data);

      std::uniform_real_distribution<float> dist(0, 1);

      for (size_t i = 0; i < size; ++i) {
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
          y_data[i] = 0;
        } else {
          mask_data[i] = 1;
          if (upscale_in_train) {
            y_data[i] = x_data[i] / static_cast<T>(1.0f - dropout_prob);
          } else {
            y_data[i] = x_data[i];
          }
        }
      }
    } else {
      if (upscale_in_train) {
        const auto* X_data = x->data<T>();
        auto* Y_data = y->mutable_data<T>(context.GetPlace());
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < x->numel(); i++) {
          Y_data[i] = X_data[i];
        }
      } else {
        auto X = EigenMatrix<T>::Reshape(*x, 1);
        auto Y = EigenMatrix<T>::Reshape(*y, 1);
        auto& place =
            *context.template device_context<DeviceContext>().eigen_device();
        Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    auto size = grad_x->numel();

    auto dX = EigenVector<T>::Flatten(*grad_x);
    auto dY = EigenVector<T>::Flatten(*grad_y);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    if (context.Attr<bool>("is_test") == true) {
      if (dropout_implementation == "upscale_in_train") {
        dX.device(place) = static_cast<T>(1) * dY;
      } else {
        float dropout_prob = context.Attr<float>("dropout_prob");
        dX.device(place) = dY * static_cast<T>(1.0f - dropout_prob);
      }
    } else {
      auto M = EigenVector<uint8_t>::Flatten(*mask);
      if (dropout_implementation == "upscale_in_train") {
        float dropout_prob = context.Attr<float>("dropout_prob");
        if (dropout_prob == 1.0f) {
          dX.device(place) = static_cast<T>(0) * dY;
        } else {
          int vec_size = platform::GetVectorizedSize<T>(grad_y->data<T>());
          if (platform::is_gpu_place(context.GetPlace()) && vec_size == 4 &&
              size % 4 == 0) {
#if defined(__NVCC__) || defined(__HIPCC__)
            auto factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
            auto stream = context.cuda_device_context().stream();
            platform::GpuLaunchConfig config = platform::GetGpuLaunchConfig1D(
                context.cuda_device_context(), size);
            DropoutGradCUDAKernel<T, uint8_t, 4><<<
                config.block_per_grid, config.thread_per_block, 0, stream>>>(
                grad_y->data<T>(), mask->data<uint8_t>(), factor, size,
                grad_x->data<T>());
#endif
          } else {
            dX.device(place) =
                dY * M.cast<T>() / static_cast<T>(1.0f - dropout_prob);
          }
        }
      } else {
        dX.device(place) = dY * M.cast<T>();
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
