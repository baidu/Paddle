// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// Manual jvp rules for dygraph forward_autograd
// there are three kind of jvp rules
// 1. elementwise functions, e.g. tanh, we can reuse their vjp rules and just
// replace the out_grad with input_tangent
// 2. linear functions with single input, e.g. scale, we can reuse their forward
// functions and just replace the input with input_tangent
// 3. other case, e.g. concat/stack/batch_norm, we need to implement jvp rules
// manually
//

#include "paddle/fluid/eager/api/manual/eager_manual/forward_grad/manual_jvp.h"

void add_jvp(const paddle::Tensor& x_p,
             const paddle::Tensor& x_t,
             const paddle::Tensor& y_p,
             const paddle::Tensor& y_t,
             paddle::Tensor* out_t) {
  *out_t = x_t + y_t;
}

void scale_jvp(const paddle::Tensor& x_p,
               const paddle::Tensor& x_t,
               paddle::experimental::Scalar scale,
               paddle::experimental::Scalar bias,
               bool bias_after_scale,
               paddle::Tensor* out_t) {
  *out_t = scale_ad_func(x_p, scale, 0, false);
}

void tanh_jvp(const paddle::Tensor& x_p,
              const paddle::Tensor& x_t,
              paddle::Tensor* out_t) {
  paddle::prim::tanh_grad<paddle::Tensor>(x_p, x_t, out_t);
}

void concat_jvp(const std::vector<paddle::Tensor>& x_ts,
                paddle::experimental::Scalar axis,
                paddle::Tensor* out_t) {
  *out_t = concat_ad_func(x_ts, axis);
}
