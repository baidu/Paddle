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

#pragma once

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/as_complex_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGradImpl(const Context& dev_ctx,
                        const DenseTensor* X,
                        const DenseTensor* Out,
                        const DenseTensor* dOut,
                        DenseTensor* dX,
                        const Functor& functor) {
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    PADDLE_ENFORCE_NOT_NULL(
        Out, errors::NotFound("The input DenseTensor Out can not be nullptr"));
  }
  PADDLE_ENFORCE_NOT_NULL(
      dOut, errors::NotFound("The input DenseTensor dOut can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      dX, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (!Out) {
    Out = dOut;  // fake out
  }
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    PADDLE_ENFORCE_NOT_NULL(
        X, errors::NotFound("The input DenseTensor X can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    X = dX;
  }

  dev_ctx.template Alloc<T>(dX);
  auto dout = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "ActivationGrad"));
  auto out = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(Out, "Input", "Out", "ActivationGrad"));
  auto dx = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(dX, "Input", "X@GRAD", "ActivationGrad"));
  auto x = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(X, "Input", "X", "ActivationGrad"));
  auto* place = dev_ctx.eigen_device();
  // use 32bit index to speed up computation
  bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  if (use_32bit_index && is_gpu_place) {
    functor(*place,
            To32BitIndex(x),
            To32BitIndex(out),
            To32BitIndex(dout),
            To32BitIndex(dx));
  } else {
    functor(*place, x, out, dout, dx);
  }
}

template <typename T, typename Context, typename Functor>
void ActivationDoubleGradImpl(const Context& dev_ctx,
                              const DenseTensor* X,
                              const DenseTensor* Out,
                              const DenseTensor* ddX,
                              DenseTensor* dX,
                              DenseTensor* dOut,
                              DenseTensor* ddOut,
                              const Functor& functor) {
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    PADDLE_ENFORCE_NOT_NULL(
        X, errors::NotFound("The input DenseTensor X can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    X = ddX;
  }
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    PADDLE_ENFORCE_NOT_NULL(
        Out, errors::NotFound("The input DenseTensor Out can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    Out = ddX;
  }

  if (ddOut) {
    dev_ctx.template Alloc<T>(ddOut);
  }
  if (dOut) {
    dev_ctx.template Alloc<T>(dOut);
  }
  if (dX) {
    dX->Resize(Out->dims());
    dev_ctx.template Alloc<T>(dX);
  }

  functor(dev_ctx, X, Out, ddX, ddOut, dOut, dX);
}

template <typename T, typename Context>
void ReluDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          DenseTensor* ddout) {
  funcs::ReluGradGradFunctor<T> relu_double_grad_functor;
  ActivationDoubleGradImpl<T, Context, funcs::ReluGradGradFunctor<T>>(
      dev_ctx,
      nullptr,
      &out,
      &ddx,
      nullptr,
      nullptr,
      ddout,
      relu_double_grad_functor);
}

template <typename T, typename Context>
void LeakyReluDoubleGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& ddx,
                               float alpha,
                               DenseTensor* ddout) {
  funcs::LeakyReluGradGradFunctor<T> leaky_relu_double_grad_functor;
  leaky_relu_double_grad_functor.alpha = alpha;
  ActivationDoubleGradImpl<T, Context, funcs::LeakyReluGradGradFunctor<T>>(
      dev_ctx,
      &x,
      nullptr,
      &ddx,
      nullptr,
      nullptr,
      ddout,
      leaky_relu_double_grad_functor);
}

template <typename T, typename Context>
void TanhDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dout,
                          const DenseTensor& ddx,
                          DenseTensor* dout_new,
                          DenseTensor* ddout) {
  if (dout_new) {
    dout_new->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout_new);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::TanhGradGradFunctor<T> functor;
  functor(dev_ctx, &out, &ddx, &dout, dout_new, ddout);
}

template <typename T, typename Context>
void TanhTripleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dout,
                          const DenseTensor& ddx,
                          const paddle::optional<DenseTensor>& d_dout_new,
                          const paddle::optional<DenseTensor>& d_ddout,
                          DenseTensor* d_out_new,
                          DenseTensor* d_dout,
                          DenseTensor* d_ddx) {
  if (d_dout) {
    d_dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_out_new) {
    d_out_new->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_out_new);
  }
  if (d_ddx) {
    d_ddx->Resize(ddx.dims());
    dev_ctx.template Alloc<T>(d_ddx);
  }
  funcs::TanhTripleGradFunctor<T> functor;
  functor(dev_ctx,
          &out,
          &ddx,
          &dout,
          d_ddout.get_ptr(),
          d_dout_new.get_ptr(),  // input
          d_dout,
          d_out_new,
          d_ddx);  // output
}

template <typename T, typename Context>
void EluDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         float alpha,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::ELUGradGradFunctor<T> functor;
  functor.alpha = alpha;
  functor(dev_ctx, &x, &ddx, ddout, &dout, dx);
}

template <typename T, typename Context>
void LogitGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     float eps,
                     DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  auto eigen_x = EigenVector<T>::Flatten(x);
  auto eigen_dout = EigenVector<T>::Flatten(out_grad);
  auto eigen_dx = EigenVector<T>::Flatten(*x_grad);
  auto& place = *dev_ctx.eigen_device();
  auto eigen_p = EigenVector<T>::Flatten(x);

  funcs::LogitGradFunctor<T> functor;
  functor(place, eigen_x, eigen_dout, eigen_dx, eigen_p, eps);
}

template <typename T, typename Context>
void SigmoidDoubleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             const DenseTensor& ddx,
                             DenseTensor* dout_new,
                             DenseTensor* ddout) {
  if (dout_new) {
    dout_new->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout_new);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::SigmoidGradGradFunctor<T> functor;
  functor(dev_ctx, &out, &ddx, &dout, dout_new, ddout);
}

template <typename T, typename Context>
void SigmoidTripleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             const DenseTensor& ddx,
                             const DenseTensor& d_dout_new,
                             const paddle::optional<DenseTensor>& d_ddout,
                             DenseTensor* d_out_new,
                             DenseTensor* d_dout,
                             DenseTensor* d_ddx) {
  if (d_dout) {
    d_dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_out_new) {
    d_out_new->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_out_new);
  }
  if (d_ddx) {
    d_ddx->Resize(ddx.dims());
    dev_ctx.template Alloc<T>(d_ddx);
  }
  funcs::SigmoidTripleGradFunctor<T> functor;
  functor(dev_ctx,
          &out,
          &ddx,
          &dout,
          d_ddout.get_ptr(),
          &d_dout_new,
          d_dout,
          d_out_new,
          d_ddx);
}

template <typename T, typename Context>
void LogDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::LogGradGradFunctor<T> functor;
  functor(dev_ctx, &x, &ddx, ddout, &dout, dx);
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   const Scalar& factor,
                   DenseTensor* dx) {
  PADDLE_ENFORCE_NOT_NULL(
      dx, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  auto dout_flatten = EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&dout, "Input", "Out@GRAD", "PowGrad"));
  auto dx_flatten = EigenVector<T>::Flatten(
      GET_DATA_SAFELY(dx, "Output", "X@GRAD", "PowGrad"));
  auto x_flatten =
      EigenVector<T>::Flatten(GET_DATA_SAFELY(&x, "Input", "X", "PowGrad"));
  auto* place = dev_ctx.eigen_device();
  phi::funcs::PowGradFunctor<T> functor;
  if (IsComplexType(x.dtype()) || IsComplexType(factor.dtype())) {
    auto attrs = functor.GetComplexAttrs();
    *(attrs[0].second) = factor.to<T>();
  } else {
    auto attrs = functor.GetAttrs();
    *(attrs[0].second) = factor.to<float>();
  }
  functor(*place, x_flatten, nullptr, dout_flatten, dx_flatten);
}

template <typename T, typename Context>
void PowDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         const Scalar& factor,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  PADDLE_ENFORCE_NOT_NULL(
      dx, errors::NotFound("The output DenseTensor DX can not be nullptr"));
  float exponent = factor.to<float>();
  if (dx) {
    if (exponent == 1) {
      *dx = phi::FullLike<T, Context>(dev_ctx, x, static_cast<T>(0));
    } else {
      DenseTensor dx_tmp1 = phi::Multiply<T, Context>(dev_ctx, dout, ddx);
      DenseTensor dx_tmp2 = phi::Multiply<T, Context>(
          dev_ctx, dx_tmp1, phi::Pow<T, Context>(dev_ctx, x, exponent - 2));
      *dx = phi::Scale<T, Context>(
          dev_ctx, dx_tmp2, exponent * (exponent - 1), 0.0, true);
    }
  }
  if (ddout) {
    DenseTensor ddout_tmp = phi::Multiply<T, Context>(
        dev_ctx, ddx, phi::Pow<T, Context>(dev_ctx, x, exponent - 1));
    *ddout = phi::Scale<T, Context>(dev_ctx, ddout_tmp, exponent, 0.0, true);
  }
}

template <typename T, typename Context>
void PowTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         const DenseTensor& d_dx,
                         const paddle::optional<DenseTensor>& d_ddout,
                         const Scalar& factor,
                         DenseTensor* out_d_x,
                         DenseTensor* out_d_dout,
                         DenseTensor* out_d_ddx) {
  PADDLE_ENFORCE_NOT_NULL(
      out_d_x,
      errors::NotFound("The output DenseTensor D_X can not be nullptr"));
  float exponent = factor.to<float>();
  if (exponent != 2 && exponent != 1) {
    // case1: b != 2 and b != 1
    // D_X = D_DX * DDX * DOut * b * (b-1) * (b-2) * X^(b-3)
    //       + D_DDOut * DDX * b * (b-1) * X^(b-2)
    if (out_d_x) {
      DenseTensor out_d_x_tmp1 = phi::Multiply<T, Context>(dev_ctx, d_dx, ddx);
      DenseTensor out_d_x_tmp2 =
          phi::Scale<T, Context>(dev_ctx,
                                 phi::Pow<T, Context>(dev_ctx, x, exponent - 3),
                                 exponent * (exponent - 1) * (exponent - 2),
                                 0.0,
                                 true);
      DenseTensor out_d_x_part1 = phi::Multiply<T, Context>(
          dev_ctx,
          phi::Multiply<T, Context>(dev_ctx, out_d_x_tmp1, dout),
          out_d_x_tmp2);

      if (d_ddout.get_ptr()) {
        DenseTensor out_d_x_tmp3 =
            phi::Multiply<T, Context>(dev_ctx, d_ddout.get(), ddx);
        DenseTensor out_d_x_tmp4 = phi::Scale<T, Context>(
            dev_ctx,
            phi::Pow<T, Context>(dev_ctx, x, exponent - 2),
            exponent * (exponent - 1),
            0.0,
            true);
        DenseTensor out_d_x_part2 =
            phi::Multiply<T, Context>(dev_ctx, out_d_x_tmp3, out_d_x_tmp4);
        *out_d_x = phi::Add<T, Context>(dev_ctx, out_d_x_part1, out_d_x_part2);
      } else {
        *out_d_x = out_d_x_part1;
      }
    }
    // D_DOut = D_DX * DDX * b * (b-1) * X^(b-2)
    if (out_d_dout) {
      DenseTensor out_d_x_tmp = phi::Multiply<T, Context>(dev_ctx, d_dx, ddx);
      DenseTensor out_d_dout_tmp =
          phi::Scale<T, Context>(dev_ctx,
                                 phi::Pow<T, Context>(dev_ctx, x, exponent - 2),
                                 exponent * (exponent - 1),
                                 0.0,
                                 true);

      *out_d_dout =
          phi::Multiply<T, Context>(dev_ctx, out_d_x_tmp, out_d_dout_tmp);
    }
    // D_DDX = D_DX * DOut * b * (b-1) * X^(b-2) + D_DDOut * b * X^(b-1)
    if (out_d_ddx) {
      DenseTensor out_d_ddx_tmp1 =
          phi::Multiply<T, Context>(dev_ctx, d_dx, dout);
      DenseTensor out_d_dout_tmp =
          phi::Scale<T, Context>(dev_ctx,
                                 phi::Pow<T, Context>(dev_ctx, x, exponent - 2),
                                 exponent * (exponent - 1),
                                 0.0,
                                 true);
      DenseTensor out_d_ddx_part1 =
          phi::Multiply<T, Context>(dev_ctx, out_d_ddx_tmp1, out_d_dout_tmp);
      if (d_ddout.get_ptr()) {
        DenseTensor out_d_ddx_tmp2 = phi::Scale<T, Context>(
            dev_ctx,
            phi::Pow<T, Context>(dev_ctx, x, exponent - 1),
            exponent,
            0.0,
            true);
        DenseTensor out_d_ddx_part2 =
            phi::Multiply<T, Context>(dev_ctx, d_ddout.get(), out_d_ddx_tmp2);
        *out_d_ddx =
            phi::Add<T, Context>(dev_ctx, out_d_ddx_part1, out_d_ddx_part2);
      } else {
        *out_d_ddx = out_d_ddx_part1;
      }
    }
  } else if (exponent == 2) {
    // case2: b = 2
    // D_X = D_DDOut * DDX * b * (b-1) * X^(b-2)
    if (out_d_x) {
      if (d_ddout.get_ptr()) {
        DenseTensor out_d_x_tmp1 =
            phi::Multiply<T, Context>(dev_ctx, d_ddout.get(), ddx);
        DenseTensor out_d_x_tmp2 = phi::Scale<T, Context>(
            dev_ctx,
            phi::Pow<T, Context>(dev_ctx, x, exponent - 2),
            exponent * (exponent - 1),
            0.0,
            true);
        *out_d_x =
            phi::Multiply<T, Context>(dev_ctx, out_d_x_tmp1, out_d_x_tmp2);
      } else {
        *out_d_x = phi::FullLike<T, Context>(dev_ctx, x, static_cast<T>(0));
      }
    }
    // D_DOut = D_DX * DDX * b * (b-1) * X^(b-2)
    if (out_d_dout) {
      DenseTensor out_d_dout_tmp1 =
          phi::Multiply<T, Context>(dev_ctx, d_dx, ddx);
      DenseTensor out_d_dout_tmp2 =
          phi::Scale<T, Context>(dev_ctx,
                                 phi::Pow<T, Context>(dev_ctx, x, exponent - 2),
                                 exponent * (exponent - 1),
                                 0.0,
                                 true);

      *out_d_dout =
          phi::Multiply<T, Context>(dev_ctx, out_d_dout_tmp1, out_d_dout_tmp2);
    }
    // D_DDX = D_DX * DOut * b * (b-1) * X^(b-2) + D_DDOut * b * X^(b-1)
    if (out_d_ddx) {
      DenseTensor out_d_ddx_tmp1 =
          phi::Multiply<T, Context>(dev_ctx, d_dx, dout);
      DenseTensor out_d_dout_tmp2 =
          phi::Scale<T, Context>(dev_ctx,
                                 phi::Pow<T, Context>(dev_ctx, x, exponent - 2),
                                 exponent * (exponent - 1),
                                 0.0,
                                 true);
      DenseTensor out_d_ddx_part1 =
          phi::Multiply<T, Context>(dev_ctx, out_d_ddx_tmp1, out_d_dout_tmp2);

      if (d_ddout.get_ptr()) {
        DenseTensor out_d_ddx_tmp2 = phi::Scale<T, Context>(
            dev_ctx,
            phi::Pow<T, Context>(dev_ctx, x, exponent - 1),
            exponent,
            0.0,
            true);
        DenseTensor out_d_ddx_part2 =
            phi::Multiply<T, Context>(dev_ctx, d_ddout.get(), out_d_ddx_tmp2);
        *out_d_ddx =
            phi::Add<T, Context>(dev_ctx, out_d_ddx_part1, out_d_ddx_part2);
      } else {
        *out_d_ddx = out_d_ddx_part1;
      }
    }
  } else {
    // case3: b = 1
    // D_X = D_DX * DDX * DOut * b * (b-1) * (b-2) * X^(b-3)
    if (out_d_x) {
      DenseTensor out_d_x_tmp1 = phi::Multiply<T, Context>(dev_ctx, d_dx, ddx);
      DenseTensor out_d_x_tmp2 =
          phi::Scale<T, Context>(dev_ctx,
                                 phi::Pow<T, Context>(dev_ctx, x, exponent - 3),
                                 exponent * (exponent - 1) * (exponent - 2),
                                 0.0,
                                 true);

      *out_d_x = phi::Multiply<T, Context>(
          dev_ctx,
          phi::Multiply<T, Context>(dev_ctx, out_d_x_tmp1, dout),
          out_d_x_tmp2);
    }
    // D_DOut = 0
    if (out_d_dout) {
      *out_d_dout = phi::FullLike<T, Context>(dev_ctx, dout, static_cast<T>(0));
    }
    // D_DDX = D_DDOut * b * X^(b-1)
    if (out_d_ddx) {
      if (d_ddout.get_ptr()) {
        DenseTensor out_d_ddx_tmp = phi::Scale<T, Context>(
            dev_ctx,
            phi::Pow<T, Context>(dev_ctx, x, exponent - 1),
            exponent,
            0.0,
            true);

        *out_d_ddx =
            phi::Multiply<T, Context>(dev_ctx, d_ddout.get(), out_d_ddx_tmp);
      } else {
        *out_d_ddx = phi::FullLike<T, Context>(dev_ctx, ddx, static_cast<T>(0));
      }
    }
  }
}

template <typename T, typename Context>
void SqrtDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dx,
                          const DenseTensor& ddx,
                          DenseTensor* dout,
                          DenseTensor* ddout) {
  if (dout) {
    dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }

  phi::funcs::SqrtGradGradFunctor<T> functor;
  functor(dev_ctx, &out, &dx, &ddx, dout, ddout);
}

// rsqrt Grad: dx = -0.5 * dy * y * y * y
// rsqrt GradGrad: ddy = -0.5 * ddx * y * y * y, dy = (3 / y) * dx * ddx
template <typename T, typename Context>
void RsqrtDoubleGradKernel(const Context& dev_ctx,
                           const DenseTensor& out,
                           const DenseTensor& dx,
                           const DenseTensor& ddx,
                           DenseTensor* dout,
                           DenseTensor* ddout) {
  if (dout) {
    dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }

  phi::funcs::RsqrtGradGradFunctor<T> functor;
  functor(dev_ctx, &out, &dx, &ddx, dout, ddout);
}

template <typename T, typename Context>
void CeluDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& dout,
                          const DenseTensor& ddx,
                          float alpha,
                          DenseTensor* dx,
                          DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }

  phi::funcs::CELUGradGradFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = alpha;
  functor(dev_ctx, &x, &dout, &ddx, dx, ddout);
}

template <typename T, typename Context>
void SoftplusDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& dout,
                              const DenseTensor& ddx,
                              float beta,
                              float threshold,
                              DenseTensor* dx,
                              DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }

  phi::funcs::SoftplusDoubleGradFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = beta;
  *(attrs[1].second) = threshold;
  functor(dev_ctx, &x, &dout, &ddx, dx, ddout);
}

template <typename T, typename Context>
void SquareDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& dout,
                            const DenseTensor& ddx,
                            DenseTensor* dx,
                            DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }

  phi::funcs::SquareGradGradFunctor<T> functor;
  functor(dev_ctx, &x, &dout, &ddx, dx, ddout);
}

template <typename T, typename Context>
void SinDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  phi::funcs::SinDoubleGradFunctor<T> functor;
  functor(dev_ctx, &x, &dout, &ddx, dx, ddout);
}

template <typename T, typename Context>
void SinTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const DenseTensor& d_dx_new,
                         const paddle::optional<DenseTensor>& d_ddout,
                         DenseTensor* d_x_new,
                         DenseTensor* d_dout,
                         DenseTensor* d_ddx) {
  if (d_dout) {
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_x_new) {
    dev_ctx.template Alloc<T>(d_x_new);
  }
  if (d_ddx) {
    dev_ctx.template Alloc<T>(d_ddx);
  }
  funcs::SinTripleGradFunctor<T> functor;
  functor(dev_ctx,
          &x,
          ddx.get_ptr(),
          dout.get_ptr(),
          d_ddout.get_ptr(),
          &d_dx_new,  // input
          d_dout,
          d_x_new,
          d_ddx);  // output
}

template <typename T, typename Context>
void CosDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  phi::funcs::CosDoubleGradFunctor<T> functor;
  functor(dev_ctx, &x, &dout, &ddx, dx, ddout);
}

template <typename T, typename Context>
void CosTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const DenseTensor& d_dx_new,
                         const paddle::optional<DenseTensor>& d_ddout,
                         DenseTensor* d_x_new,
                         DenseTensor* d_dout,
                         DenseTensor* d_ddx) {
  if (d_dout) {
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_x_new) {
    dev_ctx.template Alloc<T>(d_x_new);
  }
  if (d_ddx) {
    dev_ctx.template Alloc<T>(d_ddx);
  }
  funcs::CosTripleGradFunctor<T> functor;
  functor(dev_ctx,
          &x,
          ddx.get_ptr(),
          dout.get_ptr(),
          d_ddout.get_ptr(),
          &d_dx_new,  // input
          d_dout,
          d_x_new,
          d_ddx);  // output
}

}  // namespace phi
