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

#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void AddFunctor(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                int axis,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.dims() == y.dims()) {
    SameDimsElementwiseCompute<SameDimsAddFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      funcs::ElementwiseCompute<funcs::AddFunctor<T>, T>(
          dev_ctx, x, y, funcs::AddFunctor<T>(), out, axis);
    } else {
      funcs::ElementwiseCompute<funcs::InverseAddFunctor<T>, T>(
          dev_ctx, x, y, funcs::InverseAddFunctor<T>(), out, axis);
    }
  }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  AddFunctor<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  AddFunctor<T>(dev_ctx, x, y, -1, out);
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::phi::dtype::bfloat16;

PD_REGISTER_KERNEL(add,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(grad_add,
                   CPU,
                   ALL_LAYOUT,
                   phi::GradAddKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   complex64,
                   complex128) {}
