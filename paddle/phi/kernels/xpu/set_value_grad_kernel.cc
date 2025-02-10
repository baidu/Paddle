// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/set_value_grad_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"

namespace phi {

inline void GetOffsets(const DDim& big_dim,
                       const DDim& small_dim,
                       DDim start_offset,
                       int cur_dim,
                       std::vector<DDim>* offsets) {
  if (cur_dim == big_dim.size()) {
    offsets->push_back(start_offset);
    return;
  }
  if (small_dim[cur_dim] == big_dim[cur_dim]) {
    GetOffsets(big_dim, small_dim, start_offset, cur_dim + 1, offsets);
  } else {
    for (int i = 0; i < big_dim[cur_dim]; i++) {
      GetOffsets(big_dim, small_dim, start_offset, cur_dim + 1, offsets);
      start_offset[cur_dim] += 1;
    }
  }
}

template <typename T, typename Context, size_t RANK>
void SetValueGradImpl(const Context& dev_ctx,
                      const DenseTensor& out_grad,
                      const IntArray& starts,
                      const IntArray& ends,
                      const IntArray& steps,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& decrease_axes,
                      const std::vector<int64_t>& none_axes,
                      DenseTensor* x_grad,
                      DenseTensor* value_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  PADDLE_ENFORCE_EQ(
      out_grad.IsInitialized(),
      true,
      errors::PermissionDenied(
          "The input of `set_value_grad`(out_grad) has not been initialized"));

  auto in_dims = out_grad.dims();
  auto in_dims_vector = common::vectorize<int64_t>(in_dims);

  std::vector<int> decrease_axis_int32(decrease_axes.begin(),
                                       decrease_axes.end());
  std::vector<int> axes_int32(axes.begin(), axes.end());
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  funcs::StridedSliceOutDims(starts_local,
                             ends_local,
                             steps_local,
                             axes_int32,
                             infer_flags,
                             in_dims,
                             decrease_axis_int32,
                             out_dims_vector.data(),
                             axes.size(),
                             false);

  DDim out_dims(common::make_ddim(out_dims_vector));

  std::vector<int> reverse_vector(starts_local.size(), 0);
  funcs::StridedSliceFunctor(starts_local.data(),
                             ends_local.data(),
                             steps_local.data(),
                             axes_int32.data(),
                             reverse_vector.data(),
                             in_dims,
                             infer_flags,
                             decrease_axis_int32,
                             starts_local.size());

  std::vector<int64_t> starts_indices(RANK, 0);
  std::vector<int64_t> ends_indices(RANK, 0);
  std::vector<int64_t> steps_indices(RANK, 0);
  std::vector<bool> reverse_axis(RANK, 0);
  std::vector<int64_t> flip_axis;

  for (size_t axis = 0; axis < RANK; axis++) {
    starts_indices[axis] = 0;
    ends_indices[axis] = out_dims[axis];
    steps_indices[axis] = 1;
    reverse_axis[axis] = false;
  }

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts_local[axis];
    ends_indices[axis_index] = ends_local[axis];
    steps_indices[axis_index] = steps_local[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  for (size_t axis = 0; axis < RANK; axis++) {
    if (reverse_axis[axis]) {
      flip_axis.push_back(axis);
    }
    if (ends_indices[axis] > in_dims[axis]) {
      ends_indices[axis] = in_dims[axis];
    }
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  phi::funcs::SetConstant<Context, T> set_zero;
  int r = XPU_SUCCESS;

  if (x_grad) {
    // Set gradient of `Input`
    x_grad->Resize(out_grad.dims());
    dev_ctx.template Alloc<T>(x_grad);
    r = xpu::copy(dev_ctx.x_context(),
                  reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                  reinterpret_cast<XPUType*>(x_grad->data<T>()),
                  out_grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

    DenseTensor tmp = Full<T>(dev_ctx, out_dims_vector, static_cast<T>(0));

    r = xpu::strided_slice_view_update(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(tmp.data<T>()),
        reinterpret_cast<XPUType*>(x_grad->data<T>()),
        out_dims_vector,
        common::vectorize<int64_t>(x_grad->dims()),
        starts_indices,
        ends_indices,
        steps_indices);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice_view_update");
  }
  if (value_grad) {
    dev_ctx.template Alloc<T>(value_grad);
    set_zero(dev_ctx, value_grad, static_cast<T>(0));

    if (value_grad->dims() == out_dims) {
      if (need_reverse) {
        r = xpu::strided_slice(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(out_grad.data<T>()),
            reinterpret_cast<XPUType*>(value_grad->data<T>()),
            in_dims_vector,
            starts_indices,
            ends_indices,
            steps_indices);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");

        r = xpu::flip(dev_ctx.x_context(),
                      reinterpret_cast<const XPUType*>(value_grad->data<T>()),
                      reinterpret_cast<XPUType*>(value_grad->data<T>()),
                      out_dims_vector,
                      flip_axis);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "flip");
      } else {
        r = xpu::strided_slice(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(out_grad.data<T>()),
            reinterpret_cast<XPUType*>(value_grad->data<T>()),
            in_dims_vector,
            starts_indices,
            ends_indices,
            steps_indices);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");
      }
    } else {
      int out_dims_size = out_dims.size();
      auto value_grad_dims = value_grad->dims();
      auto fake_value_grad_dims = out_dims;

      // Create an extended shape according to the rules of broadcast.
      auto value_grad_dims_size = value_grad_dims.size();

      int num_decrease = 0;

      int decrease_axis_size = decrease_axes.size();
      for (int i = 0; i < out_dims_size; i++) {
        if (decrease_axes.end() !=
            std::find(decrease_axes.begin(), decrease_axes.end(), i)) {
          fake_value_grad_dims[i] = 1;
          num_decrease++;
        } else if (i < out_dims_size - (value_grad_dims_size +
                                        decrease_axis_size - num_decrease)) {
          fake_value_grad_dims[i] = 1;
        } else {
          auto index_grad =
              i - (out_dims_size -
                   (value_grad_dims_size + decrease_axis_size - num_decrease));
          fake_value_grad_dims[i] = value_grad_dims[index_grad];

          PADDLE_ENFORCE_EQ(
              (out_dims[i] == value_grad_dims[index_grad]) ||
                  (value_grad_dims[index_grad] == 1),
              true,
              errors::InvalidArgument("An error occurred while calculating %s: "
                                      "[%s] can not be accumulated into [%s].",
                                      "ValueTensor@GRAD",
                                      out_dims,
                                      value_grad_dims));
        }
      }

      VLOG(3) << "Dimensions of "
              << "ValueTensor@GRAD"
              << "([" << value_grad_dims << "])is broadcasted into ["
              << fake_value_grad_dims << "].";

      std::vector<int64_t> slice_end(RANK, 0);
      auto offset = out_dims;
      for (int i = 0; i < out_dims_size; i++) {
        offset[i] = 0;
      }
      std::vector<DDim> offsets;
      GetOffsets(out_dims, fake_value_grad_dims, offset, 0, &offsets);

      DenseTensor tmp = Full<T>(dev_ctx, out_dims_vector, static_cast<T>(0));

      r = xpu::strided_slice(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(out_grad.data<T>()),
          reinterpret_cast<XPUType*>(tmp.data<T>()),
          in_dims_vector,
          starts_indices,
          ends_indices,
          steps_indices);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");

      // accumulate gradient
      DenseTensor tmp2 =
          Full<T>(dev_ctx,
                  {fake_value_grad_dims.Get(), fake_value_grad_dims.size()},
                  static_cast<T>(0));
      auto value_grad_dims_vec = common::vectorize<int64_t>(value_grad_dims);
      // for value is a 0-D Tensor
      if (value_grad_dims.size() == 0) {
        value_grad_dims_vec = common::vectorize<int64_t>(
            common::make_ddim(std::vector<int>({1})));
      }
      for (auto offset : offsets) {
        for (int i = 0; i < out_dims_size; i++) {
          slice_end[i] = offset[i] + fake_value_grad_dims[i];
        }
        r = xpu::slice(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType*>(tmp.data<T>()),
                       reinterpret_cast<XPUType*>(tmp2.data<T>()),
                       out_dims_vector,
                       common::vectorize<int64_t>(offset),
                       slice_end);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
        r = xpu::broadcast_add(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(value_grad->data<T>()),
            reinterpret_cast<const XPUType*>(tmp2.data<T>()),
            reinterpret_cast<XPUType*>(value_grad->data<T>()),
            value_grad_dims_vec,
            value_grad_dims_vec);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
      }
      if (need_reverse) {
        r = xpu::flip(dev_ctx.x_context(),
                      reinterpret_cast<const XPUType*>(value_grad->data<T>()),
                      reinterpret_cast<XPUType*>(value_grad->data<T>()),
                      value_grad_dims_vec,
                      flip_axis);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "flip");
      }
    }
  }
}

template <typename T, typename Context>
void SetValueGradKernel(const Context& dev_ctx,
                        const DenseTensor& out_grad,
                        const IntArray& starts,
                        const IntArray& ends,
                        const IntArray& steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad) {
  const int rank = out_grad.dims().size();

  switch (rank) {
    case 1:
      SetValueGradImpl<T, Context, 1>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 2:
      SetValueGradImpl<T, Context, 2>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 3:
      SetValueGradImpl<T, Context, 3>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 4:
      SetValueGradImpl<T, Context, 4>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 5:
      SetValueGradImpl<T, Context, 5>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 6:
      SetValueGradImpl<T, Context, 6>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of set_value_grad's input should be less than 7, but "
          "received %d.",
          rank));
  }
}

template <typename T, typename Context>
void SetValueWithScalarGradKernel(const Context& dev_ctx,
                                  const DenseTensor& out_grad,
                                  const IntArray& starts,
                                  const IntArray& ends,
                                  const IntArray& steps,
                                  const std::vector<int64_t>& axes,
                                  const std::vector<int64_t>& decrease_axes,
                                  const std::vector<int64_t>& none_axes,
                                  DenseTensor* x_grad) {
  SetValueGradKernel<T, Context>(dev_ctx,
                                 out_grad,
                                 starts,
                                 ends,
                                 steps,
                                 axes,
                                 decrease_axes,
                                 none_axes,
                                 x_grad,
                                 nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(set_value_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetValueGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(set_value_with_scalar_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetValueWithScalarGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
