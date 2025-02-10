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
void FusedDropoutAddKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const paddle::optional<DenseTensor>& seed_tensor,
                           const Scalar& p,
                           bool is_test,
                           const std::string& mode,
                           int seed,
                           bool fix_seed,
                           DenseTensor* out,
                           DenseTensor* seed_offset) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  auto* out_data = dev_ctx.template Alloc<T>(out);
  seed_offset->Resize({2});
  int64_t* seed_offset_data = dev_ctx.template HostAlloc<int64_t>(seed_offset);
  int64_t numel = x.numel();

  const auto* x_data = x.data<T>();
  auto* y_data = y.data<T>();
  bool upscale_in_train = (mode == "upscale_in_train");
  float dropout_rate = p.to<float>();

  int r = xpu::SUCCESS;

  if (!is_test) {  // normally dropout + add
    int seed_data = 0;
    if (seed_tensor.get_ptr() != nullptr) {
      if ((seed_tensor->place()).GetType() == phi::AllocationType::XPU) {
        memory_utils::Copy(phi::CPUPlace(),
                           &seed_data,
                           seed_tensor->place(),
                           seed_tensor->data<int>(),
                           sizeof(int));
      } else {
        seed_data = *(seed_tensor->data<int>());
      }
    } else {
      seed_data = fix_seed ? seed : 0;
    }
    if (seed_data == 0) {
      seed_data = dev_ctx.GetGenerator()->Random64();
    }
    seed_offset_data[0] = static_cast<int64_t>(seed_data);
    r = xpu::dropout_add_v2(xpu_ctx,
                            reinterpret_cast<const XPUType*>(x_data),
                            reinterpret_cast<const XPUType*>(y_data),
                            reinterpret_cast<XPUType*>(out_data),
                            (uint32_t)seed_data,
                            numel,
                            upscale_in_train,
                            dropout_rate);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_v2");

  } else {  // scale + add
    float factor = 1.0f - dropout_rate;
    if (!upscale_in_train) {
      r = xpu::scale(xpu_ctx,
                     reinterpret_cast<const XPUType*>(x_data),
                     reinterpret_cast<XPUType*>(out_data),
                     numel,
                     false,
                     factor,
                     0.f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    } else {
      r = xpu::copy(xpu_ctx,
                    reinterpret_cast<const XPUType*>(x_data),
                    reinterpret_cast<XPUType*>(out_data),
                    numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    }
    r = xpu::add(xpu_ctx,
                 reinterpret_cast<const XPUType*>(out_data),
                 reinterpret_cast<const XPUType*>(y_data),
                 reinterpret_cast<XPUType*>(out_data),
                 numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dropout_add,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDropoutAddKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetBackend(phi::Backend::CPU);
}
