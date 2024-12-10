// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_tensor_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/onednn/elementwise_kernel.cc"

namespace phi {
template <typename T, typename Context>
void ClipTensorGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& min,
                          const DenseTensor& max,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad) {
  phi::DenseTensor ls_min;
  phi::ElementwiseKernel<T, dnnl::algorithm::binary_ge>(dev_ctx, x, min, -1, &ls_min);
  phi::CastKernel<T, Context>(dev_ctx, ls_min, x.dtype(), &ls_min);
  phi::DenseTensor ls_max;
  phi::ElementwiseKernel<T, dnnl::algorithm::binary_le>(dev_ctx, x, max, -1, &ls_max);
  phi::CastKernel<T, Context>(dev_ctx, ls_max, x.dtype(), &ls_max);
  phi::DenseTensor tem_out;
  phi::ElementwiseKernel<T, dnnl::algorithm::binary_mul>(dev_ctx, ls_max, ls_min, -1, &tem_out);
  phi::ElementwiseKernel<T, dnnl::algorithm::binary_mul>(dev_ctx, tem_out, out_grad, -1, x_grad);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    clip_tensor_grad, OneDNN, ONEDNN, phi::ClipTensorGradKernel, float, phi::dtype::bfloat16) {}
