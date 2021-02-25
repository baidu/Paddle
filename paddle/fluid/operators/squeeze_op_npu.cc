/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/squeeze_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SqueezeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto axes = ctx.Attr<std::vector<int>>("axes");
    framework::AttributeMap attr_input = {{"axis", axes}};

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto runner = NpuOpRunner("Squeeze", {*in}, {*out}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    squeeze,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    squeeze2,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::SqueezeNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
#endif
