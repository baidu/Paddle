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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedDropoutAddGradKernel(const Context& dev_ctx,
                               const DenseTensor& seed_offset,
                               const DenseTensor& out_grad,
                               const Scalar& p,
                               bool is_test,
                               const std::string& mode,
                               bool fix_seed,  // unused
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  auto* y_grad_data = dev_ctx.template Alloc<T>(y_grad);

  int r = xpu::SUCCESS;

  bool upscale_in_train = (mode == "upscale_in_train");
  float dropout_rate = p.to<float>();

  const int64_t* seed_data_ptr = seed_offset.data<int64_t>();
  uint32_t seed_data = static_cast<uint32_t>(seed_data_ptr[0]);
  const auto* out_grad_data = out_grad.data<T>();

  if (is_test) {
    float factor = 1.0f - dropout_rate;
    r = xpu::copy(dev_ctx.x_context(),
                  reinterpret_cast<const int8_t*>(out_grad_data),
                  reinterpret_cast<int8_t*>(y_grad_data),
                  out_grad.numel() * phi::SizeOf(out_grad.dtype()));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    if (upscale_in_train) {
      r = xpu::copy(dev_ctx.x_context(),
                    reinterpret_cast<const int8_t*>(out_grad_data),
                    reinterpret_cast<int8_t*>(x_grad_data),
                    out_grad.numel() * phi::SizeOf(out_grad.dtype()));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    } else {
      r = xpu::scale(xpu_ctx,
                     reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                     reinterpret_cast<XPUType*>(x_grad_data),
                     out_grad.numel(),
                     false,
                     factor,
                     0.f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  } else {
    r = xpu::dropout_add_grad_v2(
        xpu_ctx,
        reinterpret_cast<const XPUType*>(out_grad_data),
        reinterpret_cast<XPUType*>(y_grad_data),
        reinterpret_cast<XPUType*>(x_grad_data),
        seed_data,
        out_grad.numel(),
        upscale_in_train,
        dropout_rate);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_grad_v2");
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dropout_add_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDropoutAddGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);  // seed
}
