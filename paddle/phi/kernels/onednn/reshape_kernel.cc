/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static DDim ValidateShape(const std::vector<int64_t>& shape,
                          const DDim& in_dims) {
  const int64_t in_size = product(in_dims);
  auto in_dims_vec = common::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                  in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // inferred
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              common::make_ddim(shape),
              i));
      unk_dim_idx = static_cast<int>(i);
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(
          static_cast<int>(i),
          in_dims.size(),
          errors::InvalidArgument(
              "The index of 0 in `shape` must be less than "
              "the input tensor X's dimensions. "
              "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
              "X's dimensions = %d.",
              common::make_ddim(shape),
              i,
              in_dims,
              in_dims.size()));
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              common::make_ddim(shape),
              i,
              shape[i]));
    }

    capacity *= (shape[i] ? shape[i] : in_dims[i]);  // NOLINT
    output_shape[i] =
        (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);  // NOLINT
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          -in_size,
          errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              common::make_ddim(shape),
              capacity));
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              common::make_ddim(shape),
              capacity));
    }
  }
  return common::make_ddim(output_shape);
}

template <typename T, typename Context>
void ExecuteReshape(const Context& dev_ctx,
                    const DenseTensor& x,
                    const IntArray& shape,
                    const DDim& x_dims,
                    DenseTensor* out) {
  auto out_dims = ValidateShape(shape.GetData(), x_dims);
  auto x_vec_dims = x.mem_desc().get_dims();

  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims,
      x.dtype(),
      funcs::ToOneDNNDataType(x.dtype()),
      dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));
  out->Resize(x_dims);  // to match x numel, format is changed later
  // reorder is done into a plain tag to allow usage with blocked formats
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      out, funcs::GetPlainOneDNNFormat(x_dims.size()), dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);

  astream.wait();

  out->Resize(out_dims);
  const auto reshape_dims = out_dims.size() != 0 ? common::vectorize(out_dims)
                                                 : std::vector<int64_t>{1};
  out->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(reshape_dims));
}

template <typename T, typename Context>
void ReshapeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& shape,
                   DenseTensor* out) {
  auto x_dims = x.dims();
  ExecuteReshape<T, Context>(dev_ctx, x, shape, x_dims, out);
}

template <typename T, typename Context>
void ReshapeWithXShapeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& shape,
                             DenseTensor* out,
                             DenseTensor* xshape) {
  auto x_dims = slice_ddim(xshape->dims(), 1, xshape->dims().size());
  ExecuteReshape<T, Context>(dev_ctx, x, shape, x_dims, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    reshape, OneDNN, ONEDNN, phi::ReshapeKernel, float, phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(reshape_with_xshape,
                   OneDNN,
                   ONEDNN,
                   phi::ReshapeWithXShapeKernel,
                   float,
                   phi::dtype::bfloat16) {}
