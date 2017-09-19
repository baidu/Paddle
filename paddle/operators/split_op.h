/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <vector>
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class SplitOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    size_t before = 1, after = 1;
    const size_t n = outs.size();
    size_t input_axis_dim = in->dims()[axis];

    for (int64_t i = 0; i < in->dims().size(); ++i) {
      if (i == axis) {
        continue;
      }
      if (i < axis) {
        before *= in->dims()[i];
      } else {
        after *= in->dims()[i];
      }
    }
    size_t input_offset = 0;
    for (size_t i = 0; i < n; i++) {
      auto& out = outs[i];
      out->mutable_data<T>(ctx.GetPlace());
      size_t axis_dim = out->dims()[axis];

      for (size_t j = 0; j < before; j++) {
        T* dst = out->data<T>() + axis_dim * after * j;
        const T* src =
            in->data<T>() + input_offset + input_axis_dim * after * j;
        paddle::memory::Copy<Place, Place>(
            boost::get<Place>(ctx.GetPlace()), static_cast<void*>(dst),
            boost::get<Place>(ctx.GetPlace()), static_cast<const void*>(src),
            axis_dim * after * sizeof(T));
      }
      input_offset += axis_dim * after;
    }
  }
};

}  // namespace operators
}  // namespace paddle
