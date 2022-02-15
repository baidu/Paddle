// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PoissonKernel;

template <typename DeviceContext, typename T>
class PoissonGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    pten::funcs::SetConstant<DeviceContext, T> functor;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    functor(dev_ctx, dx, static_cast<T>(0));
  }
};

}  // namespace operators
}  // namespace paddle
