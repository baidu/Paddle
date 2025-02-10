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

#include "paddle/phi/backends/gpu/gpu_context.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#endif
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/legacy/elementwise_add_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_subtract_kernel.h"

namespace phi {

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  phi::SubtractRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  phi::MultiplyRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
  phi::DivideRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MultiPrecisionAddKernelImpl(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  if (y.dtype() == phi::DataType::BFLOAT16) {
    funcs::BroadcastKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        funcs::MultiPrecisionAddFunctor<T, phi::bfloat16>(),
        -1);
  } else if (y.dtype() == phi::DataType::FLOAT16) {
    funcs::BroadcastKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        funcs::MultiPrecisionAddFunctor<T, phi::float16>(),
        -1);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported x dtype:%s, y dtype:%s for add(x, y) operation",
        phi::DataTypeToString(x.type()),
        phi::DataTypeToString(y.type())));
  }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
#ifdef PADDLE_WITH_CUDA
  if (x.dtype() == phi::DataType::FLOAT32 &&
      (y.dtype() == phi::DataType::BFLOAT16 ||
       y.dtype() == phi::DataType::FLOAT16)) {
    MultiPrecisionAddKernelImpl<float, Context>(dev_ctx, x, y, out);
  } else {
#endif
    phi::AddRawKernel<T, Context>(dev_ctx, x, y, -1, out);
#ifdef PADDLE_WITH_CUDA
  }
#endif
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  phi::AddRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  int axis = -1;
  RemainderRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

// Create the definition of Heaviside
template <typename T, typename Context>
void HeavisideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::ElementwiseHeavisideFunctor<T>());
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::CopySignFunctor<T>());
}

}  // namespace phi

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_REGISTER_KERNEL(maximum,
                   KPS,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum,
                   KPS,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder,
                   GPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(floor_divide,
                   KPS,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   KPS,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(copysign,
                   GPU,
                   ALL_LAYOUT,
                   phi::CopySignKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#endif

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(maximum, KPS, ALL_LAYOUT, phi::MaximumKernel, float) {}
PD_REGISTER_KERNEL(minimum, KPS, ALL_LAYOUT, phi::MinimumKernel, float) {}
PD_REGISTER_KERNEL(divide, KPS, ALL_LAYOUT, phi::DivideKernel, float) {}
PD_REGISTER_KERNEL(multiply, KPS, ALL_LAYOUT, phi::MultiplyKernel, float) {}
PD_REGISTER_KERNEL(add, KPS, ALL_LAYOUT, phi::AddKernel, float) {}
PD_REGISTER_KERNEL(subtract, KPS, ALL_LAYOUT, phi::SubtractKernel, float) {}
PD_REGISTER_KERNEL(floor_divide, KPS, ALL_LAYOUT, phi::FloorDivideKernel, int) {
}
PD_REGISTER_KERNEL(
    elementwise_pow, KPS, ALL_LAYOUT, phi::ElementwisePowKernel, float) {}

#else
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(fmax,
                   KPS,
                   ALL_LAYOUT,
                   phi::FMaxKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(fmin,
                   KPS,
                   ALL_LAYOUT,
                   phi::FMinKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(heaviside,
                   KPS,
                   ALL_LAYOUT,
                   phi::HeavisideKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(add,
                   KPS,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(grad_add,
                   KPS,
                   ALL_LAYOUT,
                   phi::GradAddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(divide,
                   KPS,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(multiply,
                   KPS,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   float16,
                   complex64,
                   complex128,
                   bfloat16) {}

PD_REGISTER_KERNEL(subtract,
                   KPS,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

#endif
