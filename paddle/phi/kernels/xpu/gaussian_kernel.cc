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

#include "paddle/phi/kernels/gaussian_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GaussianKernel(const Context& ctx,
                    const IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  T* data = ctx.template Alloc<T>(out);

  if (out->numel() == 0) {
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t real_seed = seed != 0 ? seed : ctx.GetGenerator()->Random64();

  // int normal(Context* ctx, T* x, T mean, T std, int64_t len, int64_t seed);
  int r = xpu::normal<XPUType>(ctx.x_context(),
                               reinterpret_cast<XPUType*>(data),
                               mean,
                               std,
                               out->numel(),
                               real_seed);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "normal");
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian,
                   XPU,
                   ALL_LAYOUT,
                   phi::GaussianKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
