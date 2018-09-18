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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class ScaleKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in_var = ctx.InputVar("X");
    auto* in = ctx.Input<framework::Tensor>("X");

    auto* out_var = ctx.OutputVar("Out");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(in->place());

    PADDLE_ENFORCE_EQ(in->dims(), out->dims(),
                      "in and out should have the same dim");

    auto scale = static_cast<T>(ctx.Attr<float>("scale"));
    auto bias = static_cast<T>(ctx.Attr<float>("bias"));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    if (in_var->IsType<framework::SelectedRows>() && in_var != out_var) {
      auto& in_slr = in_var->Get<framework::SelectedRows>();
      auto* out_slr = out_var->GetMutable<framework::SelectedRows>();
      out_slr->set_rows(in_slr.rows());
      out_slr->set_height(in_slr.height());
    }

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    if (bias_after_scale) {
      eigen_out.device(dev) = scale * eigen_in + bias;
    } else {
      eigen_out.device(dev) = scale * (eigen_in + bias);
    }
  }
};

}  // namespace operators
}  // namespace paddle
