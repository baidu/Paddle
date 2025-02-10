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

#include "paddle/phi/kernels/cast_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename InT, typename OutT, typename Context>
void CastXPUKernelImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  using XPUInT = typename XPUTypeTrait<InT>::Type;
  using XPUOutT = typename XPUTypeTrait<OutT>::Type;

  const auto* in_data = x.data<InT>();
  auto* out_data = dev_ctx.template Alloc<OutT>(out);
  auto numel = x.numel();

  if (numel == 0) {
    return;
  }

  if (std::is_same<InT, OutT>::value) {
    int ret = xpu::copy(dev_ctx.x_context(),
                        reinterpret_cast<const int8_t*>(in_data),
                        reinterpret_cast<int8_t*>(out_data),
                        x.numel() * phi::SizeOf(x.dtype()));
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
    return;
  }

  int r = xpu::cast<XPUInT, XPUOutT>(dev_ctx.x_context(),
                                     reinterpret_cast<const XPUInT*>(in_data),
                                     reinterpret_cast<XPUOutT*>(out_data),
                                     numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
}

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  if (x.dtype() == out_dtype) {
    if (!out->IsSharedWith(x)) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  switch (out_dtype) {
    case DataType::INT32:
      CastXPUKernelImpl<T, int, Context>(dev_ctx, x, out);
      break;
    case DataType::FLOAT32:
      CastXPUKernelImpl<T, float, Context>(dev_ctx, x, out);
      break;
    case DataType::FLOAT16:
      CastXPUKernelImpl<T, dtype::float16, Context>(dev_ctx, x, out);
      break;
    case DataType::BFLOAT16:
      CastXPUKernelImpl<T, dtype::bfloat16, Context>(dev_ctx, x, out);
      break;
    case DataType::INT64:
      CastXPUKernelImpl<T, int64_t, Context>(dev_ctx, x, out);
      break;
    case DataType::BOOL:
      CastXPUKernelImpl<T, bool, Context>(dev_ctx, x, out);
      break;
    case DataType::INT8:
      CastXPUKernelImpl<T, int8_t, Context>(dev_ctx, x, out);
      break;
    case DataType::UINT8:
      CastXPUKernelImpl<T, uint8_t, Context>(dev_ctx, x, out);
      break;
    case DataType::FLOAT64:
      CastXPUKernelImpl<T, double, Context>(dev_ctx, x, out);
      break;
    case DataType::INT16:
      CastXPUKernelImpl<T, int16_t, Context>(dev_ctx, x, out);
      break;
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Not supported cast %d -> %d", x.dtype(), out_dtype));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(cast,
                   XPU,
                   ALL_LAYOUT,
                   phi::CastKernel,
                   int16_t,
                   int32_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   bool,
                   int8_t,
                   uint8_t,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
