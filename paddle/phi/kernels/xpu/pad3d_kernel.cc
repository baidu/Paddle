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

#include "paddle/phi/kernels/pad3d_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void Pad3dKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& paddings,
                 const std::string& mode,
                 float pad_value,
                 const std::string& data_format,
                 DenseTensor* out) {
  std::vector<int64_t> pads = paddings.GetData();

  auto in_dims = x.dims();
  const T* in_data = x.data<T>();

  bool is_ncdhw = true;
  if (data_format == "NCDHW") {
    out->Resize({in_dims[0],
                 in_dims[1],
                 in_dims[2] + pads[4] + pads[5],
                 in_dims[3] + pads[2] + pads[3],
                 in_dims[4] + pads[0] + pads[1]});
  } else {
    // NDHWC
    is_ncdhw = false;
    out->Resize({in_dims[0],
                 in_dims[1] + pads[4] + pads[5],
                 in_dims[2] + pads[2] + pads[3],
                 in_dims[3] + pads[0] + pads[1],
                 in_dims[4]});
  }

  T* out_data = dev_ctx.template Alloc<T>(out);

  const int num = in_dims[0];  // n
  int channels = in_dims[1];   // c
  int in_depth = in_dims[2];   // xd
  int in_height = in_dims[3];  // xh
  int in_width = in_dims[4];   // xw
  if (data_format == "NDHWC") {
    channels = in_dims[4];   // c
    in_depth = in_dims[1];   // xd
    in_height = in_dims[2];  // xh
    in_width = in_dims[3];   // xw
  }

  if (mode == "circular") {
    PADDLE_THROW(common::errors::External(
        "XPU is not support circular padding mode in pad3d"));
  }

  if (mode == "reflect") {
    PADDLE_ENFORCE_GT(
        in_depth,
        pads[4],
        errors::InvalidArgument("The depth of Input(X)'s dimension should be "
                                "greater than pad_front"
                                " in reflect mode"
                                ", but received depth(%d) and pad_front(%d).",
                                in_depth,
                                pads[4]));
    PADDLE_ENFORCE_GT(
        in_depth,
        pads[5],
        errors::InvalidArgument("The depth of Input(X)'s dimension should be "
                                "greater than pad_back"
                                " in reflect mode"
                                ", but received depth(%d) and pad_back(%d).",
                                in_depth,
                                pads[5]));

    PADDLE_ENFORCE_GT(
        in_height,
        pads[2],
        errors::InvalidArgument("The height of Input(X)'s dimension should be "
                                "greater than pad_top"
                                " in reflect mode"
                                ", but received depth(%d) and pad_top(%d).",
                                in_height,
                                pads[2]));
    PADDLE_ENFORCE_GT(
        in_height,
        pads[3],
        errors::InvalidArgument("The height of Input(X)'s dimension should be "
                                "greater than pad_bottom"
                                " in reflect mode"
                                ", but received depth(%d) and pad_bottom(%d).",
                                in_height,
                                pads[3]));

    PADDLE_ENFORCE_GT(
        in_width,
        pads[0],
        errors::InvalidArgument("The width of Input(X)'s dimension should be "
                                "greater than pad_left"
                                " in reflect mode"
                                ", but received depth(%d) and pad_left(%d).",
                                in_width,
                                pads[0]));
    PADDLE_ENFORCE_GT(
        in_width,
        pads[1],
        errors::InvalidArgument("The width of Input(X)'s dimension should be "
                                "greater than pad_right"
                                " in reflect mode"
                                ", but received depth(%d) and pad_right(%d).",
                                in_width,
                                pads[1]));
  } else if (mode == "replicate") {
    PADDLE_ENFORCE_NE(in_depth * in_height * in_width,
                      0,
                      errors::InvalidArgument(
                          "The input tensor size can not be 0 for circular "
                          "or replicate padding mode."));
  }

  std::vector<int> pads_xpu(6);
  pads_xpu[0] = pads[4];  // pf
  pads_xpu[1] = pads[5];  // pb
  pads_xpu[2] = pads[2];  // pt
  pads_xpu[3] = pads[3];  // pd
  pads_xpu[4] = pads[0];  // pl
  pads_xpu[5] = pads[1];  // pr

  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
  using XPUTypeBF16 = typename XPUTypeTrait<phi::dtype::bfloat16>::Type;
  // Because the xpu api do not support pad3d with bf16 type, we use fp16
  // temporarily. This would not cause problem because it is a memcpy-only
  // operator.
  using XPURealType = std::
      conditional_t<std::is_same_v<XPUType, XPUTypeBF16>, XPUTypeFP16, XPUType>;

  if (mode == "reflect") {
    int r = xpu::reflection_pad3d<XPURealType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPURealType*>(in_data),
        reinterpret_cast<XPURealType*>(out_data),
        num,
        channels,
        in_depth,
        in_height,
        in_width,
        pads_xpu,
        is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reflection_pad3d");
  } else if (mode == "replicate") {
    int r = xpu::replication_pad3d<XPURealType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPURealType*>(in_data),
        reinterpret_cast<XPURealType*>(out_data),
        num,
        channels,
        in_depth,
        in_height,
        in_width,
        pads_xpu,
        is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "replication_pad3d");
  } else if (mode == "constant") {
    XPUType value = static_cast<XPUType>(pad_value);
    XPURealType real_value;
    std::memcpy(&real_value, &value, sizeof(XPURealType));
    int r = xpu::constant_pad3d<XPURealType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPURealType*>(in_data),
        reinterpret_cast<XPURealType*>(out_data),
        num,
        channels,
        in_depth,
        in_height,
        in_width,
        pads_xpu,
        real_value,
        is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant_pad3d");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(pad3d,
                   XPU,
                   ALL_LAYOUT,
                   phi::Pad3dKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
