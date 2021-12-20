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

#include "paddle/pten/backends/cuda/cuda_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/sharedimpl/scale_kernel_impl.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/float16.h"

namespace pten {

template <typename T>
void Scale(const CUDAContext& dev_ctx,
           const DenseTensor& x,
           const Scalar& scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  ScaleImpl<T, CUDAContext>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

}  // namespace pten

PT_REGISTER_KERNEL(scale,
                   CUDA,
                   ALL_LAYOUT,
                   pten::Scale,
                   float,
                   double,
                   paddle::platform::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
