// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/norm_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void NormKernel(const Context& ctx,
                const DenseTensor& x,
                int axis,
                float epsilon,
                bool is_test,
                DenseTensor* out,
                DenseTensor* norm) {
  const auto& xdim = x.dims();
  T eps = epsilon;
  if (axis < 0) axis = xdim.size() + axis;
  int pre = 0, n = 0, post = 0;
  funcs::GetPrePostNumel(xdim, axis, &pre, &n, &post);

  DenseTensor* out_norm = nullptr;
  DenseTensor out_norm_tmp;
  if (is_test) {
    auto out_dim = x.dims();
    out_dim[axis] = 1;
    out_norm = &out_norm_tmp;
    out_norm->Resize(out_dim);
  } else {
    out_norm = norm;
  }

  ctx.template Alloc<T>(out);
  ctx.template Alloc<T>(out_norm);

  auto* place = ctx.eigen_device();

  Eigen::DSizes<int, 3> shape(pre, n, post);
  Eigen::DSizes<int, 2> norm_shape(pre, post);

  auto x_e = phi::EigenVector<T>::Flatten(x);
  auto y_e = phi::EigenVector<T>::Flatten(*out);
  auto norm_e = phi::EigenVector<T>::Flatten(*out_norm);
  auto x_r = x_e.reshape(shape);
  auto y = y_e.reshape(shape);
  auto norm_reshape = norm_e.reshape(norm_shape);

  Eigen::DSizes<int, 1> rdim(1);
  // y = x / sqrt((sum(x * x) + epsilon))
  // norm = sqrt(sum(x * x) + epsilon)
  auto x2 = x_r * x_r;
  auto sum = x2.sum(rdim) + eps;
  norm_reshape.device(*place) = sum.sqrt();

  // y = x / norm
  Eigen::DSizes<int, 3> rshape(pre, 1, post);
  Eigen::DSizes<int, 3> bcast(1, n, 1);
  y.device(*place) = x_r / norm_reshape.reshape(rshape).broadcast(bcast);
}

}  // namespace phi

PD_REGISTER_KERNEL(norm, CPU, ALL_LAYOUT, phi::NormKernel, float, double) {}
