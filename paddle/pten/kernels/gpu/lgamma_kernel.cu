
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

#include <unsupported/Eigen/SpecialFunctions>
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/lgamma_kernel.h"

namespace pten {
template <typename T>
struct CudaLgammaFunctor {
  __device__ __forceinline__ T operator()(const T x) const {
    return Eigen::numext::lgamma(x);
  }
};
template <typename T, typename Context>
void LgammaKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) {
  // XKTODO( add gpu kernel implementation. )
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  auto functor = CudaLgammaFunctor<T>();
  paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
      dev_ctx, ins, &outs, functor);
}
}  // namespace pten

PT_REGISTER_KERNEL(lgamma, GPU, ALL_LAYOUT, pten::LgammaKernel, float, double) {
}
