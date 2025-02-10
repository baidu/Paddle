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

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/conv_utils_xpu.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#ifdef PADDLE_WITH_XPU_XRE5
#include "xpudnn/xpudnn.h"
namespace xpudnn = baidu::xpu::xpudnn;
#endif

namespace phi {

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                const std::vector<int>& strides_t,
                const std::vector<int>& paddings_t,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations_t,
                int groups,
                const std::string& data_format,
                DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int64_t> paddings(paddings_t.begin(), paddings_t.end());
  std::vector<int64_t> dilations(dilations_t.begin(), dilations_t.end());
  std::vector<int64_t> strides(strides_t.begin(), strides_t.end());
  // The filter will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  dev_ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      data_format == "NDHWC",
      false,
      common::errors::InvalidArgument(
          ("XPU does not support data_format is NDHWC in conv op.")));

  phi::DDim in_data_dims =
      common::slice_ddim(input.dims(), 2, input.dims().size());
  phi::DDim filter_data_dims =
      common::slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int64_t> ksize = common::vectorize<int64_t>(filter_data_dims);
  UpdatePaddingAndDilation<int64_t>(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int64_t batch_size = input.dims()[0];
  int64_t img_c = input.dims()[1];
  int64_t img_h = input.dims()[2];
  int64_t img_w = input.dims()[3];
  int64_t f = filter.dims()[0];
  bool is_nchw = true;
  if (data_format == "NHWC") {
    img_c = input.dims()[3];
    img_h = input.dims()[1];
    img_w = input.dims()[2];
    is_nchw = false;
  }

  const XPUType* input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  const XPUType* filter_data =
      reinterpret_cast<const XPUType*>(filter.data<T>());
  XPUType* output_data = reinterpret_cast<XPUType*>(out->data<T>());

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  XPUType* filter_data_tmp;
  const XPUType* filter_data_ptr = filter_data;
  if (data_format == "NHWC") {
    filter_data_tmp = RAII_GUARD.alloc<XPUType>(filter.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
    std::vector<int> filter_shape = common::vectorize<int>(filter.dims());
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    filter_data,
                                    filter_data_tmp,
                                    filter_shape,
                                    {0, 2, 3, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    filter_data_ptr = reinterpret_cast<const XPUType*>(filter_data_tmp);
  }

  int fc_calc_type = GetConvCalcType<XPUType>();
  PD_VISIT_XPU_CONV_TYPES(XPUType, fc_calc_type, "conv2d", [&] {
#ifdef PADDLE_WITH_XPU_XRE5
    int ret = xpudnn::conv2d_fusion<XPUType, XPUType, XPUType, TGEMM>(
        dev_ctx.x_context(),
        input_data,
        filter_data_ptr,
        output_data,
        batch_size,
        img_c,
        img_h,
        img_w,
        f,
        ksize,
        strides,
        paddings,
        dilations,
        groups,
        nullptr,
        nullptr,
        nullptr,
        is_nchw,
        nullptr,
        nullptr,
        xpu::Activation_t::LINEAR);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "conv2d_fusion");
#else
  int r = xpu::conv2d<XPUType, XPUType, XPUType, TGEMM>(dev_ctx.x_context(),
                                                        input_data,
                                                        filter_data_ptr,
                                                        output_data,
                                                        batch_size,
                                                        img_c,
                                                        img_h,
                                                        img_w,
                                                        f,
                                                        ksize,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        is_nchw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d");
#endif
  });
}

template <typename T, typename Context>
void DepthwiseConvKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations,
                         const std::string& data_format,
                         DenseTensor* out) {
  ConvKernel<T, Context>(dev_ctx,
                         input,
                         filter,
                         strides,
                         paddings,
                         padding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         out);
}

template <typename T, typename Context>
void Conv3DKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations_t,
                  const std::string& data_format,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int64_t> paddings(paddings_t.begin(), paddings_t.end());
  std::vector<int64_t> dilations(dilations_t.begin(), dilations_t.end());
  std::vector<int64_t> strides(strides_t.begin(), strides_t.end());
  // The filter will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  dev_ctx.template Alloc<T>(out);

  phi::DDim in_data_dims =
      common::slice_ddim(input.dims(), 2, input.dims().size());
  phi::DDim filter_data_dims =
      common::slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int64_t> ksize = common::vectorize<int64_t>(filter_data_dims);
  UpdatePaddingAndDilation<int64_t>(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int64_t batch_size = input.dims()[0];
  int64_t img_c = input.dims()[1];
  int64_t img_d = input.dims()[2];
  int64_t img_h = input.dims()[3];
  int64_t img_w = input.dims()[4];
  int64_t f = filter.dims()[0];
  bool is_ncdhw = true;
  if (data_format == "NDHWC") {
    img_c = input.dims()[4];
    img_d = input.dims()[1];
    img_h = input.dims()[2];
    img_w = input.dims()[3];
    is_ncdhw = false;
  }

  XPUType* output_data = reinterpret_cast<XPUType*>(out->data<T>());
  const XPUType* filter_data =
      reinterpret_cast<const XPUType*>(filter.data<T>());
  const XPUType* input_data = reinterpret_cast<const XPUType*>(input.data<T>());

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  XPUType* filter_data_tmp;
  const XPUType* filter_data_ptr = filter_data;
  if (data_format == "NDHWC") {
    filter_data_tmp = RAII_GUARD.alloc<XPUType>(filter.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
    std::vector<int> filter_shape = common::vectorize<int>(filter.dims());
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    filter_data,
                                    filter_data_tmp,
                                    filter_shape,
                                    {0, 2, 3, 4, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    filter_data_ptr = reinterpret_cast<const XPUType*>(filter_data_tmp);
  }

  int fc_calc_type = GetConvCalcType<XPUType>();
  PD_VISIT_XPU_CONV_TYPES(XPUType, fc_calc_type, "conv3d", [&] {
#ifdef PADDLE_WITH_XPU_XRE5
    using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
    using RealTGEMM = std::conditional_t<std::is_same_v<XPUType, XPUTypeFP16> &&
                                             std::is_same_v<TGEMM, float>,
                                         XPUTypeFP16,
                                         TGEMM>;
    int ret = xpudnn::conv3d_fusion<XPUType, XPUType, XPUType, RealTGEMM>(
        dev_ctx.x_context(),
        input_data,
        filter_data_ptr,
        output_data,
        batch_size,
        img_c,
        img_d,
        img_h,
        img_w,
        f,
        ksize,
        strides,
        paddings,
        dilations,
        groups,
        nullptr,
        nullptr,
        nullptr,
        is_ncdhw,
        nullptr,
        nullptr,
        xpu::Activation_t::LINEAR,
        nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "conv3d_fusion");
#else
  int r = xpu::conv3d<XPUType, XPUType, XPUType, int>(dev_ctx.x_context(),
                                                        input_data,
                                                        filter_data_ptr,
                                                        output_data,
                                                        batch_size,
                                                        img_c,
                                                        img_d,
                                                        img_h,
                                                        img_w,
                                                        f,
                                                        ksize,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        is_ncdhw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d");
#endif
  });
}

}  // namespace phi

PD_REGISTER_KERNEL(conv2d,
                   XPU,
                   ALL_LAYOUT,
                   phi::ConvKernel,
                   float,
#ifdef PADDLE_WITH_XPU_XRE5
                   phi::dtype::bfloat16,
#endif
                   phi::dtype::float16) {
}
PD_REGISTER_KERNEL(depthwise_conv2d,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(conv3d,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv3DKernel,
                   float,
#ifdef PADDLE_WITH_XPU_XRE5
                   phi::dtype::bfloat16,
#endif
                   phi::dtype::float16) {
}
