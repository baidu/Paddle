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

#include "paddle/phi/kernels/array_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/stack_kernel.h"

namespace phi {
template <typename T, typename Context>
void CreateArrayKernel(const Context& dev_ctx,
                       DataType dtype,
                       TensorArray* out) {}

template <typename T, typename Context>
void CreateArrayLikeKernel(const Context& dev_ctx,
                           const TensorArray& input,
                           float val,
                           TensorArray* out) {
  out->resize(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    DenseTensor input_i = input[i];
    out->at(i).Resize(input_i.dims());
    FullLikeKernel<T, Context>(
        dev_ctx, input_i, val, input_i.dtype(), &out->at(i));
  }
}

template <typename T, typename Context>
void ArrayLengthKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       DenseTensor* out) {
  out->Resize({1});
  dev_ctx.template Alloc<int64_t>(out);
  *out->data<int64_t>() = static_cast<int64_t>(x.size());
}

template <typename T, typename Context>
void ArrayReadKernel(const Context& dev_ctx,
                     const TensorArray& array,
                     const Scalar& i,
                     DenseTensor* out) {
  size_t offset = i.to<int64_t>();
  PADDLE_ENFORCE_EQ(
      offset < array.size(),
      true,
      errors::InvalidArgument(
          "index %d exceed array size %d.", offset, array.size()));
  phi::Copy(dev_ctx, array[offset], dev_ctx.GetPlace(), false, out);
  out->set_lod(array[offset].lod());
}

template <typename T, typename Context>
void ArrayWriteKernel(const Context& dev_ctx,
                      const TensorArray& array,
                      const DenseTensor& x,
                      const Scalar& i,
                      TensorArray* out) {
  size_t offset = i.to<int64_t>();
  if (offset >= out->size()) {
    out->resize(offset + 1);
  }
  auto* out_tensor = &out->at(offset);
  out_tensor->set_lod(x.lod());
  if (x.memory_size() > 0) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out_tensor);
  }
}

template <typename T, typename Context>
void ArrayToTensorKernel(const Context& dev_ctx,
                         const TensorArray& x,
                         int axis,
                         bool use_stack,
                         DenseTensor* out,
                         DenseTensor* out_index) {
  const size_t n = x.size();
  PADDLE_ENFORCE_GT(
      n,
      0,
      common::errors::InvalidArgument("Input tensor array size should > 0,"
                                      "but the received is %d",
                                      n));

  std::vector<DenseTensor> tmp_inputs(x.size());
  std::vector<const DenseTensor*> inputs;

  std::vector<DenseTensor> tmp_indices(x.size());
  std::vector<const DenseTensor*> indices;

  for (size_t i = 0; i < x.size(); i++) {
    tmp_inputs[i].ShareDataWith(x[i]);
    inputs.push_back(&tmp_inputs[i]);
    FullKernel<int, Context>(
        dev_ctx, {1}, x[i].dims()[axis], DataType::INT32, &tmp_indices[i]);
    indices.push_back(&tmp_indices[i]);
  }

  if (use_stack) {
    auto vec = common::vectorize<int>(x[0].dims());
    vec.insert(vec.begin() + axis, x.size());  // NOLINT
    out->Resize(common::make_ddim(vec));
    StackKernel<T, Context>(dev_ctx, inputs, axis, out);
  } else {
    auto out_dims = x[0].dims();
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == static_cast<size_t>(axis)) {
          out_dims[axis] += x[i].dims()[static_cast<int>(j)];
        }
      }
    }
    auto vec = common::vectorize<int>(out_dims);
    out->Resize(common::make_ddim(vec));
    ConcatKernel<T, Context>(dev_ctx, inputs, axis, out);
  }

  out_index->Resize(common::make_ddim({static_cast<int>(x.size())}));
  StackKernel<int, Context>(dev_ctx, indices, 0, out_index);
}

template <typename T, typename Context>
void ArrayPopKernel(const Context& dev_ctx,
                    const TensorArray& array,
                    int index,
                    TensorArray* array_out,
                    DenseTensor* out) {
  PADDLE_ENFORCE_GT(
      array.size(),
      0,
      common::errors::InvalidArgument("Input tensorarray size should > 0,"
                                      "but the received is %d",
                                      array.size()));
  if (index < 0) {
    index += array.size();
  }

  out->ShareDataWith(array[index]);
  array_out->pop(static_cast<size_t>(index));
}

}  // namespace phi
PD_REGISTER_KERNEL(create_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(create_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(create_array,
                   XPU,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

PD_REGISTER_KERNEL(create_array_like,
                   CPU,
                   ALL_LAYOUT,
                   phi::CreateArrayLikeKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(create_array_like,
                   GPU,
                   ALL_LAYOUT,
                   phi::CreateArrayLikeKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(create_array_like,
                   XPU,
                   ALL_LAYOUT,
                   phi::CreateArrayLikeKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

PD_REGISTER_KERNEL(array_length,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayLengthKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(array_read,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayReadKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_read,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayReadKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(array_read,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArrayReadKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

PD_REGISTER_KERNEL(array_write,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayWriteKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_write,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayWriteKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(array_write,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArrayWriteKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

PD_REGISTER_KERNEL(array_to_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayToTensorKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_to_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayToTensorKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(array_to_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArrayToTensorKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

PD_REGISTER_KERNEL(array_pop,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayPopKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_pop,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayPopKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(array_pop,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArrayPopKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
