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

#include "paddle/fluid/operators/range_op.h"

namespace paddle {
namespace operators {

template <typename T>
class RangeMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* start_t = context.Input<framework::Tensor>("Start");
    auto* end_t = context.Input<framework::Tensor>("End");
    auto* step_t = context.Input<framework::Tensor>("Step");
    auto* out = context.Output<framework::Tensor>("Out");

    framework::Tensor n;
    framework::TensorCopy(
        *start_t,
        platform::CPUPlace(),
        context.template device_context<platform::MLUDeviceContext>(),
        &n);
    context.template device_context<paddle::platform::MLUDeviceContext>()
        .Wait();
    T start = n.data<T>()[0];
    framework::TensorCopy(
        *end_t,
        platform::CPUPlace(),
        context.template device_context<platform::MLUDeviceContext>(),
        &n);
    context.template device_context<paddle::platform::MLUDeviceContext>()
        .Wait();
    T end = n.data<T>()[0];
    framework::TensorCopy(
        *step_t,
        platform::CPUPlace(),
        context.template device_context<platform::MLUDeviceContext>(),
        &n);
    context.template device_context<paddle::platform::MLUDeviceContext>()
        .Wait();
    T step = n.data<T>()[0];

    int64_t size = 0;
    GetSize(start, end, step, &size);

    out->Resize(phi::make_ddim({size}));
    out->mutable_data<T>(context.GetPlace());

    std::vector<T> odata;
    T value = start;
    for (int64_t i = 0; i < size; ++i) {
      odata.push_back(value);
      value += step;
    }

    framework::TensorFromVector(odata, context.device_context(), out);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_MLU_KERNEL(range,
                       paddle::operators::RangeMLUKernel<int>,
                       paddle::operators::RangeMLUKernel<int64_t>,
                       paddle::operators::RangeMLUKernel<float>,
                       paddle::operators::RangeMLUKernel<double>)
