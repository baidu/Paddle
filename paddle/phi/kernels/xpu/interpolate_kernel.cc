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

#include "paddle/phi/kernels/interpolate_kernel.h"
#include <cstring>  // For std::memcpy
#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"

namespace phi {

template <typename T>
DenseTensor Reshape5DTo4D(const DenseTensor& tensor,
                          const DataLayout& data_layout) {
  auto dims = tensor.dims();
  int N, C, D, H, W;
  if (data_layout == DataLayout::kNCDHW) {
    N = dims[0];
    C = dims[1];
    D = dims[2];
    H = dims[3];
    W = dims[4];
    phi::DDim new_dims = {N * C * D, 1, H, W};

    DenseTensor reshaped_tensor;
    reshaped_tensor.Resize(new_dims);
    reshaped_tensor.mutable_data<T>(tensor.place());

    const T* src_data = tensor.data<T>();
    T* dst_data = reshaped_tensor.data<T>();

    int total_elements = N * C * D * H * W;
    std::memcpy(dst_data, src_data, total_elements * sizeof(T));
    return reshaped_tensor;
  } else if (data_layout == DataLayout::kNDHWC) {
    N = dims[0];
    D = dims[1];
    H = dims[2];
    W = dims[3];
    C = dims[4];
    phi::DDim new_dims = {N * D * C, 1, H, W};

    DenseTensor reshaped_tensor;
    reshaped_tensor.Resize(new_dims);
    reshaped_tensor.mutable_data<T>(tensor.place());

    const T* src_data = tensor.data<T>();
    T* dst_data = reshaped_tensor.data<T>();

    int total_elements = N * D * C * H * W;
    std::memcpy(dst_data, src_data, total_elements * sizeof(T));
    return reshaped_tensor;
  } else {
    DenseTensor empty;
    return empty;  // Unsupported data layout; return empty tensor
  }
}

template <typename T>
DenseTensor Reshape4DTo5D(const DenseTensor& tensor,
                          int N,
                          int C,
                          int D,
                          const DataLayout& data_layout) {
  auto dims = tensor.dims();  // [N*C*D, 1, H', W'] or [N*D*C, 1, H', W']
  int H_new = dims[2];
  int W_new = dims[3];

  DenseTensor reshaped_tensor;
  if (data_layout == DataLayout::kNCDHW) {
    phi::DDim new_dims = {N, C, D, H_new, W_new};
    reshaped_tensor.Resize(new_dims);
    reshaped_tensor.mutable_data<T>(tensor.place());

    const T* src_data = tensor.data<T>();
    T* dst_data = reshaped_tensor.data<T>();

    int total_elements = N * C * D * H_new * W_new;
    std::memcpy(dst_data, src_data, total_elements * sizeof(T));
    return reshaped_tensor;
  } else if (data_layout == DataLayout::kNDHWC) {
    phi::DDim new_dims = {N, D, H_new, W_new, C};
    reshaped_tensor.Resize(new_dims);
    reshaped_tensor.mutable_data<T>(tensor.place());

    const T* src_data = tensor.data<T>();
    T* dst_data = reshaped_tensor.data<T>();

    int total_elements = N * D * C * H_new * W_new;
    std::memcpy(dst_data, src_data, total_elements * sizeof(T));
    return reshaped_tensor;
  } else {
    DenseTensor empty;
    return empty;  // Unsupported data layout; return empty tensor
  }
}

template <typename T, typename Context>
void InterpolateKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int num_dims = static_cast<int>(x.dims().size());

  // Original logic for scale and size calculation
  float scale_h = -1;
  float scale_w = -1;

  if (size_tensor && size_tensor->size() > 0) {
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    if (scale_tensor) {
      auto scale_data =
          funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor should be > 0, but got %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor should be > 0, but got %d.",
              scale_h));
    } else {
      if (scale.size() > 1) {
        scale_h = scale[0];
        scale_w = scale[1];

        PADDLE_ENFORCE_EQ(
            scale_w > 0,
            true,
            errors::InvalidArgument(
                "The scale_w in Attr(scale) should be > 0, got %d.", scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0,
            true,
            errors::InvalidArgument(
                "The scale_h in Attr(scale) should be > 0, got %d.", scale_h));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      int N, C, H, W, D_dim;
      if (num_dims == 4) {
        if (data_layout == DataLayout::kNCHW) {
          N = x.dims()[0];
          C = x.dims()[1];
          H = x.dims()[2];
          W = x.dims()[3];
        } else {
          N = x.dims()[0];
          H = x.dims()[1];
          W = x.dims()[2];
          C = x.dims()[3];
        }
        out_h = static_cast<int>(H * scale_h);
        out_w = static_cast<int>(W * scale_w);
      } else if (num_dims == 5) {
        if (data_layout == DataLayout::kNCDHW) {
          N = x.dims()[0];
          C = x.dims()[1];
          D_dim = x.dims()[2];
          H = x.dims()[3];
          W = x.dims()[4];
        } else {  // NDHWC
          N = x.dims()[0];
          D_dim = x.dims()[1];
          H = x.dims()[2];
          W = x.dims()[3];
          C = x.dims()[4];
        }
        out_h = static_cast<int>(H * scale_h);
        out_w = static_cast<int>(W * scale_w);
      }
    }
    if (out_size) {
      auto out_size_data =
          funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }

  PADDLE_ENFORCE_GT(out_h, 0, errors::InvalidArgument("out_h should be > 0."));
  PADDLE_ENFORCE_GT(out_w, 0, errors::InvalidArgument("out_w should be > 0."));

  // Distinguish between 4D and 5D
  if (num_dims == 5) {
    // Handle 5D tensor
    int N, C, D, H, W;
    if (data_layout == DataLayout::kNCDHW) {
      N = x.dims()[0];
      C = x.dims()[1];
      D = x.dims()[2];
      H = x.dims()[3];
      W = x.dims()[4];
    } else if (data_layout == DataLayout::kNDHWC) {
      N = x.dims()[0];
      D = x.dims()[1];
      H = x.dims()[2];
      W = x.dims()[3];
      C = x.dims()[4];
    } else {
      PADDLE_THROW(
          errors::InvalidArgument("Unsupported data layout for 5D tensor."));
    }

    // Reshape to 4D
    DenseTensor reshaped_x = Reshape5DTo4D<T>(x, data_layout);
    PADDLE_ENFORCE_EQ(
        reshaped_x.numel() > 0,
        true,
        errors::InvalidArgument("Reshaping to 4D tensor failed."));

    using XPUType = typename XPUTypeTrait<T>::Type;
    bool nearest = (interp_method == "nearest");
    int trans_mode = (align_corners) ? 0 : ((align_mode == 0) ? 1 : 2);

    // Prepare temp_output
    DenseTensor temp_output;
    temp_output.Resize(
        {reshaped_x.dims()[0], reshaped_x.dims()[1], out_h, out_w});
    temp_output.mutable_data<T>(ctx.GetPlace());

    int r = xpu::interpolate2d<XPUType>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(reshaped_x.data<T>()),
        reinterpret_cast<XPUType*>(temp_output.data<T>()),
        reshaped_x.dims()[0],
        reshaped_x.dims()[1],  // 1 after reshape
        reshaped_x.dims()[2],
        reshaped_x.dims()[3],
        out_h,
        out_w,
        nearest,
        trans_mode,
        true);

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "interpolate2d");

    DenseTensor reshaped_back =
        Reshape4DTo5D<T>(temp_output, N, C, D, data_layout);
    PADDLE_ENFORCE_EQ(
        reshaped_back.numel() > 0,
        true,
        errors::InvalidArgument("Reshaping back to 5D tensor failed."));
    *output = reshaped_back;

  } else if (num_dims == 4) {
    // Handle 4D tensor
    int N, C, H, W;
    if (data_layout == DataLayout::kNCHW) {
      N = x.dims()[0];
      C = x.dims()[1];
      H = x.dims()[2];
      W = x.dims()[3];
      phi::DDim dim_out = {N, C, out_h, out_w};
      output->Resize(dim_out);
    } else if (data_layout == DataLayout::kNHWC) {
      N = x.dims()[0];
      H = x.dims()[1];
      W = x.dims()[2];
      C = x.dims()[3];
      phi::DDim dim_out = {N, out_h, out_w, C};
      output->Resize(dim_out);
    } else {
      PADDLE_THROW(
          errors::InvalidArgument("Unsupported data layout for 4D tensor."));
    }
    ctx.template Alloc<T>(output);

    if (H == out_h && W == out_w) {
      phi::Copy<Context>(ctx, x, ctx.GetPlace(), false, output);
      return;
    }

    using XPUType = typename XPUTypeTrait<T>::Type;
    bool nearest = (interp_method == "nearest");
    int trans_mode = (align_corners) ? 0 : ((align_mode == 0) ? 1 : 2);
    if (nearest) {
      trans_mode = (align_corners == true) ? 0 : 2;
      PADDLE_ENFORCE_EQ(
          (data_layout == DataLayout::kNCHW),
          true,
          errors::InvalidArgument("XPU nearest is only support NCHW"));
    }

    int r = xpu::interpolate2d<XPUType>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<XPUType*>(output->data<T>()),
        N,
        C,
        H,
        W,
        out_h,
        out_w,
        nearest,
        trans_mode,
        (data_layout == DataLayout::kNCHW));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "interpolate2d");

  } else {
    PADDLE_THROW(
        errors::InvalidArgument("interpolate supports only 4D or 5D tensors."));
  }
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

}  // namespace phi

PD_REGISTER_KERNEL(bilinear_interp,
                   XPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   phi::dtype::float16,
                   float) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(nearest_interp,
                   XPU,
                   ALL_LAYOUT,
                   phi::NearestInterpKernel,
                   phi::dtype::float16,
                   float,
                   int64_t) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
