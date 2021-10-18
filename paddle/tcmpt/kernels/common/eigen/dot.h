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

#pragma once

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/kernels/common/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace pt {
namespace eigen {

template <typename DevCtx, typename T>
void Dot(const DevCtx& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         DenseTensor* out) {
  out->mutable_data();
  if (1 == out->dims().size()) {
    auto eigen_out = pt::EigenScalar<T>::From(*out);
    auto eigen_x = pt::EigenVector<T>::Flatten(x);
    auto eigen_y = pt::EigenVector<T>::Flatten(y);

    auto& dev = *dev_ctx.eigen_device();
    eigen_out.device(dev) = (eigen_x * eigen_y).sum();
  } else {
    auto eigen_out = pt::EigenMatrix<T>::From(*out);
    auto eigen_x = pt::EigenMatrix<T>::From(x);
    auto eigen_y = pt::EigenMatrix<T>::From(y);

    auto& dev = *dev_ctx.eigen_device();
    eigen_out.device(dev) = (eigen_x * eigen_y).sum(Eigen::DSizes<int, 1>(1));
  }
}

}  // namespace eigen
}  // namespace pt
