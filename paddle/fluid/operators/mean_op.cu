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
#include "cub/cub.cuh"
#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};

template <typename T>
__global__ void MeanRunKernel(const T in_data, T* out_data, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_data[idx] = in_data / (static_cast<T>(N));
  }
}

template <typename DeviceContext, typename T>
class MeanCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    output->mutable_data<T>(context.GetPlace());
    auto size_prob = input->numel();
    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(context.GetPlace());
    auto stream = context.cuda_device_context().stream();

    DivideFunctor<T> transformer(size_prob);
    cub::TransformInputIterator<T, DivideFunctor<T>, const T*> trans_x(
        in_data, transformer);
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, trans_x, out_data,
                           size_prob, stream);
    framework::Tensor tmp;
    auto* temp_storage = tmp.mutable_data<uint8_t>(
        framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
        context.GetPlace());
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, trans_x, out_data,
                           size_prob, stream);
  }
};

template <typename DeviceContext, typename T>
class MeanCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE(OG->numel() == 1, "Mean Gradient should be scalar");
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());

    T in_data = OG[0];
    auto size_prob = IG->numel();
    auto out_data = IG->data<T>();
    int threads = 512;
    int grid = (size_prob + threads - 1) / threads;
    auto stream = context.cuda_device_context().stream();
    MeanRunKernel<T><<<grid, threads, 0, stream>>>(in_data, out_data,
                                                   size_prob);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    mean, ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanCUDAKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    mean_grad, ops::MeanGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MeanGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MeanGradKernel<paddle::platform::CUDADeviceContext, plat::float16>);
