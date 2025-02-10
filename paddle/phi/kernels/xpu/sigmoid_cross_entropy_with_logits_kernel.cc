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

#include <memory>

#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/memory_utils.h"

namespace phi {

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& label,
    const paddle::optional<DenseTensor>& pos_weight,
    bool normalize,
    int ignore_index,
    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  PADDLE_ENFORCE_EQ(x.place().GetType() == phi::AllocationType::XPU,
                    true,
                    errors::Unavailable("This kernel only runs on XPU."));

  dev_ctx.template Alloc<T>(out);
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int* hit = RAII_GUARD.alloc_l3_or_gm<int>(x.numel());
  PADDLE_ENFORCE_NOT_NULL(
      hit, errors::External("XPU alloc_l3_or_gm returns nullptr"));
  auto pos_weight_data =
      (pos_weight.get_ptr() == nullptr ? nullptr
                                       : pos_weight.get_ptr()->data<T>());
  // int paddle_sigmoid_cross_entropy_with_logits(Context* ctx, const T* x,
  // const T* label, const T* pos_weight, T* y, int* hit, int ignore_index,
  // int64_t n);
  int r = xpu::paddle_sigmoid_cross_entropy_with_logits(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      reinterpret_cast<const XPUType*>(label.data<T>()),
      reinterpret_cast<const XPUType*>(pos_weight_data),
      reinterpret_cast<XPUType*>(out->data<T>()),
      hit,
      ignore_index,
      x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sigmoid_cross_entropy_with_logits");
  if (normalize) {
    int* non_zero = RAII_GUARD.alloc_l3_or_gm<int>(1);
    PADDLE_ENFORCE_NOT_NULL(
        non_zero, errors::External("XPU alloc_l3_or_gm returns nullptr"));
    int r = xpu::nonzero_count(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(hit),
                               non_zero,
                               x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "nonzero_count");
    int non_zero_cpu = 0;
    memory_utils::Copy(CPUPlace(),
                       static_cast<void*>(&non_zero_cpu),
                       dev_ctx.GetPlace(),
                       static_cast<void*>(non_zero),
                       sizeof(int));
    if (std::getenv("XPUSIM_SKIP_RUN") &&
        std::strcmp(std::getenv("XPUSIM_SKIP_RUN"), "1") == 0) {
      VLOG(3)
          << "WARNING: In the simulator mode, the variable non_zero_cpu "
             "stores an uninitialized value. To avoid allocating a memory of "
             "random size, we assign numel to non_zero_cpu";
      non_zero_cpu = x.numel();
    }
    r = xpu::scale(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(out->data<T>()),
                   reinterpret_cast<XPUType*>(out->data<T>()),
                   x.numel(),
                   false,
                   1.0f / static_cast<float>(non_zero_cpu),
                   0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits,
                   XPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsKernel,
                   float) {}
