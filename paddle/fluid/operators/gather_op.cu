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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"

namespace paddle {
namespace operators {

template <typename T>
class GatherGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *index = ctx.Input<Tensor>("Index");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));

    int axis = ctx.Attr<int>("axis");
    if (ctx.HasInput("Axis")) {
      const Tensor *axis_tensor = ctx.Input<Tensor>("Axis");
      Tensor cpu_axis;
      framework::TensorCopy(*axis_tensor, platform::CPUPlace(), &cpu_axis);
      const auto &axis_type =
          framework::TransToProtoVarType(axis_tensor->dtype());
      if (axis_type == framework::proto::VarType::INT32) {
        axis = static_cast<int>(cpu_axis.data<int32_t>()[0]);
      } else if (axis_type == framework::proto::VarType::INT64) {
        axis = static_cast<int>(cpu_axis.data<int64_t>()[0]);
      }
    }

    const auto &dev_ctx = ctx.cuda_device_context();
    const auto &index_type = framework::TransToProtoVarType(index->dtype());
    if (axis != 0) {
      if (index_type == framework::proto::VarType::INT32) {
        phi::funcs::GatherV2GradCUDAFunction<T, int32_t>(dO, index, axis, dX,
                                                         dev_ctx);
      } else if (index_type == framework::proto::VarType::INT64) {
        phi::funcs::GatherV2GradCUDAFunction<T, int64_t>(dO, index, axis, dX,
                                                         dev_ctx);
      }
      return;
    }

    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;
    if (index_type == framework::proto::VarType::INT32) {
      phi::funcs::GPUScatterAssign<T, int>(dev_ctx, *dO, *index, dX,
                                           ctx.Attr<bool>("overwrite"));
    } else if (index_type == framework::proto::VarType::INT64) {
      phi::funcs::GPUScatterAssign<T, int64_t>(dev_ctx, *dO, *index, dX,
                                               ctx.Attr<bool>("overwrite"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(gather_grad, ops::GatherGradOpCUDAKernel<float>,
                        ops::GatherGradOpCUDAKernel<double>,
                        ops::GatherGradOpCUDAKernel<int64_t>,
                        ops::GatherGradOpCUDAKernel<int>,
                        ops::GatherGradOpCUDAKernel<plat::float16>,
                        ops::GatherGradOpCUDAKernel<plat::bfloat16>);
