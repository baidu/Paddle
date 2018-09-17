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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/momentum_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void MomentumKernel(const T* p, const T* g, const T* v,
                               const T* learning_rate, const T mu,
                               const int64_t num, bool use_nesterov, T* p_out,
                               T* v_out) {
  T lr = learning_rate[0];
  if (use_nesterov) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += blockDim.x * gridDim.x) {
      T g_val = g[i];
      T v_new = v[i] * mu + g_val;
      v_out[i] = v_new;
      p_out[i] = p[i] - (g_val + v_new * mu) * lr;
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += blockDim.x * gridDim.x) {
      T v_new = v[i] * mu + g[i];
      v_out[i] = v_new;
      p_out[i] = p[i] - lr * v_new;
    }
  }
}

template <typename T>
__global__ void MomentumLarsKernel(const T* p, const T* g, const T* v,
                                   const T* learning_rate, const T mu,
                                   const int64_t num, const T lars_coeff,
                                   const T lars_weight_decay, const T* p_norm,
                                   const T* g_norm, T* p_out, T* v_out) {
  T lr = learning_rate[0];
  T local_lr = learning_rate[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (p_norm[0] > 0 && g_norm[0] > 0) {
      local_lr = lr * lars_coeff * p_norm[0] /
                 (g_norm[0] + lars_weight_decay * p_norm[0]);
    }
    T v_new = v[i] * mu + (g[i] + lars_weight_decay * p[i]);
    v_out[i] = v_new;
    p_out[i] = p[i] - local_lr * v_new;
  }
}

template <typename DeviceContext, typename T>
class MomentumOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto grad = ctx.Input<framework::Tensor>("Grad");
    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    T* p_out = param_out->mutable_data<T>(ctx.GetPlace());
    T* v_out = velocity_out->mutable_data<T>(ctx.GetPlace());

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    bool use_lars = ctx.Attr<bool>("use_lars");
    T lars_coeff = ctx.Attr<float>("lars_coeff");
    T lars_weight_decay = ctx.Attr<float>("lars_weight_decay");

    auto* p = param->data<T>();
    auto* v = velocity->data<T>();
    auto* g = grad->data<T>();
    auto* lr = learning_rate->data<T>();

    int block = 512;
    int grid = (param->numel() + block - 1) / block;
    if (!use_lars) {
      MomentumKernel<T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
          p, g, v, lr, mu, param->numel(), use_nesterov, p_out, v_out);
    } else {
      auto eigen_p = framework::EigenVector<T>::Flatten(*param);
      auto eigen_g = framework::EigenVector<T>::Flatten(*grad);
      // calculate norms using eigein and launch the kernel.
      framework::Tensor p_norm_t, g_norm_t;
      p_norm_t.Resize({1});
      g_norm_t.Resize({1});
      auto* p_norm_data = p_norm_t.mutable_data<T>(ctx.GetPlace());
      auto* g_norm_data = g_norm_t.mutable_data<T>(ctx.GetPlace());
      auto ep_norm = framework::EigenScalar<T>::From(p_norm_t);
      auto eg_norm = framework::EigenScalar<T>::From(g_norm_t);

      auto* place = ctx.template device_context<DeviceContext>().eigen_device();
      ep_norm.device(*place) = eigen_p.square().sum().sqrt();
      eg_norm.device(*place) = eigen_g.square().sum().sqrt();
      MomentumLarsKernel<<<grid, block, 0,
                           ctx.cuda_device_context().stream()>>>(
          p, g, v, lr, mu, param->numel(), lars_coeff, lars_weight_decay,
          p_norm_data, g_norm_data, p_out, v_out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    momentum,
    ops::MomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, double>);
