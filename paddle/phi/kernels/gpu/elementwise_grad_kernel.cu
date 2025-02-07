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
#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"
#include "paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  // skip out
  auto* out = &dout;
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    elementwise_sub_grad<T>(dev_ctx, x, y, *out, dout, dx, dy);
  } else {
    default_elementwise_sub_grad<T>(dev_ctx, x, y, *out, dout, dx, dy, axis);
  }
}

template <typename T, typename Context>
void SubtractDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              int axis,
                              DenseTensor* ddout) {
  phi::SubtractDoubleGradImpl<T>(dev_ctx, y, ddx, ddy, dout, axis, ddout);
}

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  ElementwiseMulGrad<T>(dev_ctx, x, y, dout, dx, dy, axis);
}

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      int axis,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  const auto place = dev_ctx.GetPlace();
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor*> ins = {&dout, &x, &y};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::DivGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor*> ins = {&dout, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::DivGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor*> ins = {&dout, &x, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::DivGradYFunctor<T>());
  }
}

template <typename T>
void AddGradFunc(const GPUContext& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const DenseTensor& out,
                 const DenseTensor& dout,
                 DenseTensor* dx,
                 DenseTensor* dy,
                 int axis = -1) {
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    ElementwiseAddGrad<T>(dev_ctx, x, y, out, dout, dx, dy);
  } else {
    DefaultElementwiseAddGrad<T>(dev_ctx, x, y, out, dout, dx, dy, axis);
  }
}

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& dout,
                   int axis,
                   DenseTensor* dx,
                   DenseTensor* dy) {
  phi::AddGradImpl<T>(dev_ctx, x, y, dout, axis, dx, dy, AddGradFunc<T>);
}

template <typename T, typename Context>
void AddDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const paddle::optional<DenseTensor>& ddy,
                         int axis,
                         DenseTensor* ddout) {
  phi::AddDoubleGradImpl<T>(dev_ctx, y, ddx, ddy, dout, axis, ddout);
}

template <typename T, typename Context>
void AddTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& d_ddout,
                         int axis,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy) {
  phi::AddGradImpl<T>(
      dev_ctx, ddx, ddy, d_ddout, axis, d_ddx, d_ddy, AddGradFunc<T>);
}

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  const auto place = dev_ctx.GetPlace();
  int axis = -1;
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::MaxGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::MaxGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::MaxGradYFunctor<T>());
  }
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  const auto place = dev_ctx.GetPlace();
  int axis = -1;
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::MinGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::MinGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::MinGradYFunctor<T>());
  }
}

template <typename T, typename Context>
void RemainderGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         DenseTensor* dx,
                         DenseTensor* dy) {
  const auto place = dev_ctx.GetPlace();
  int axis = -1;
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::RemainderGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::RemainderGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::RemainderGradYFunctor<T>());
  }
}

template <typename T, typename Context>
void CopySignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad,
                        DenseTensor* y_grad) {
  const auto place = dev_ctx.GetPlace();
  int axis = -1;
  if (x_grad != nullptr && y_grad != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &out_grad};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       out_grad,
                       x_grad,
                       y_grad,
                       funcs::CopySignGradXYFunctor<T, T>());
  } else if (x_grad != nullptr && y_grad == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &out_grad};
    GetGradXOrYOut<T>(dev_ctx,
                      place,
                      axis,
                      ins,
                      out_grad,
                      x_grad,
                      funcs::CopySignGradXFunctor<T>());
  } else if (y_grad != nullptr && x_grad == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &out_grad};
    GetGradXOrYOut<T>(dev_ctx,
                      place,
                      axis,
                      ins,
                      out_grad,
                      y_grad,
                      funcs::CopySignGradYFunctor<T>());
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(fmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMaxGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(fmin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMinGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(maximum_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaximumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(minimum_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MinimumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(remainder_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RemainderGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(heaviside_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::HeavisideGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(elementwise_pow_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(add_triple_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddTripleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(divide_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DivideGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(divide_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DivideDoubleGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(multiply_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiplyGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(multiply_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiplyDoubleGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(multiply_triple_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiplyTripleGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(subtract_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SubtractGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(subtract_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SubtractDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(copysign_grad,
                   GPU,
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
