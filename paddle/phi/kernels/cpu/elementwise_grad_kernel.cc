//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cpu/elementwise_grad.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  int axis = -1;
  phi::funcs::ElemwiseGradCompute<Context, T, MaxGradDx<T>, MaxGradDy<T>>(
      dev_ctx, x, y, dout, dout, axis, dx, dy, MaxGradDx<T>(), MaxGradDy<T>());
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  int axis = -1;
  phi::funcs::ElemwiseGradCompute<Context, T, MinGradDx<T>, MinGradDy<T>>(
      dev_ctx, x, y, dout, dout, axis, dx, dy, MinGradDx<T>(), MinGradDy<T>());
}

template <typename T, typename Context>
void RemainderGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         DenseTensor* dx,
                         DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  int axis = -1;
  phi::funcs::
      ElemwiseGradCompute<Context, T, RemainderGradDx<T>, RemainderGradDy<T>>(
          dev_ctx,
          x,
          y,
          dout,
          dout,
          axis,
          dx,
          dy,
          RemainderGradDx<T>(),
          RemainderGradDy<T>());
}

template <typename T, typename Context>
void CopySignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad,
                        DenseTensor* y_grad) {
  funcs::ElementwiseGradPreProcess(out_grad, x_grad);
  int axis = -1;
  phi::funcs::
      ElemwiseGradCompute<Context, T, CopySignGradDX<T>, CopySignGradDY<T>>(
          dev_ctx,
          x,
          y,
          out_grad,
          out_grad,
          axis,
          x_grad,
          y_grad,
          CopySignGradDX<T>(),
          CopySignGradDY<T>());
}
}  // namespace phi

PD_REGISTER_KERNEL(fmax_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(fmin_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMinGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(maximum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(minimum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(remainder_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RemainderGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(heaviside_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::HeavisideGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(elementwise_pow_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(copysign_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CopySignGradKernel,
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
