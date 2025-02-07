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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::MaximumFunctor<T>, T>(
      dev_ctx, x, y, funcs::MaximumFunctor<T>(), out, axis);
}

template <typename T, typename Context>
void MinimumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::MinimumFunctor<T>, T>(
      dev_ctx, x, y, funcs::MinimumFunctor<T>(), out, axis);
}

template <typename T, typename Context>
void RemainderRawKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {  // NOLINT
    funcs::ElementwiseCompute<funcs::RemainderFunctor<T>, T>(
        dev_ctx, x, y, funcs::RemainderFunctor<T>(), out, axis);
  } else {
    funcs::ElementwiseCompute<funcs::InverseRemainderFunctor<T>, T>(
        dev_ctx, x, y, funcs::InverseRemainderFunctor<T>(), out, axis);
  }
}

template <typename T, typename Context>
void FloorDivideRawKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {  // NOLINT
    funcs::ElementwiseCompute<funcs::FloorDivideFunctor<T>, T>(
        dev_ctx, x, y, funcs::FloorDivideFunctor<T>(), out, axis);
  } else {
    funcs::ElementwiseCompute<funcs::InverseFloorDivideFunctor<T>, T>(
        dev_ctx, x, y, funcs::InverseFloorDivideFunctor<T>(), out, axis);
  }
}

template <typename T, typename Context>
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<funcs::ElementwisePowFunctor<T>, T>(
        dev_ctx, x, y, funcs::ElementwisePowFunctor<T>(), out, axis);
  } else {
    funcs::ElementwiseCompute<funcs::ElementwiseInversePowFunctor<T>, T>(
        dev_ctx, x, y, funcs::ElementwiseInversePowFunctor<T>(), out, axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(maximum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::RemainderRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
