//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/flatten_grad_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
<<<<<<< HEAD
  const auto& xshape_dims = xshape.dims();
=======
  // NOTE: [Why not to use x.dims() ?]
  // Because inplace strategy is different between old IR and PIR,
  // we need fix it into x.dims() after cleaning old IR system.
  auto x_dims = x_grad->dims();
>>>>>>> 4c9bc9e3cd7680200be9f244f9a5d374345a6741
  dev_ctx.Alloc(x_grad, out_grad.dtype());
  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  x_grad->Resize(x_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(flatten_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   phi::dtype::bfloat16,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(flatten_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(flatten_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   double,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t,
                   bool) {}

#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(flatten_grad,
                   Custom,
                   ALL_LAYOUT,
                   phi::FlattenGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
#endif
