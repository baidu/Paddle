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

#include "paddle/phi/kernels/activation_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"

namespace phi {

#define DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(name, functor_class) \
  template <typename T, typename Context>                           \
  void name##GradKernel(const Context& dev_ctx,                     \
                        const DenseTensor& x,                       \
                        const DenseTensor& dout,                    \
                        DenseTensor* dx) {                          \
    funcs::functor_class<T> functor;                                \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>(        \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(      \
    name, functor_class, attr)                               \
  template <typename T, typename Context>                    \
  void name##GradKernel(const Context& dev_ctx,              \
                        const DenseTensor& x,                \
                        const DenseTensor& dout,             \
                        float attr,                          \
                        DenseTensor* dx) {                   \
    funcs::functor_class<T> functor;                         \
    auto attrs = functor.GetAttrs();                         \
    *(attrs[0].second) = attr;                               \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, &x, nullptr, &dout, dx, functor);           \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(      \
    name, functor_class, attr1, attr2)                       \
  template <typename T, typename Context>                    \
  void name##GradKernel(const Context& dev_ctx,              \
                        const DenseTensor& x,                \
                        const DenseTensor& dout,             \
                        float attr1,                         \
                        float attr2,                         \
                        DenseTensor* dx) {                   \
    funcs::functor_class<T> functor;                         \
    auto attrs = functor.GetAttrs();                         \
    *(attrs[0].second) = attr1;                              \
    *(attrs[1].second) = attr2;                              \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, &x, nullptr, &dout, dx, functor);           \
  }

#define DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    funcs::functor_class<T> functor;                                  \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>(          \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPOUT(    \
    name, functor_class, attr)                               \
  template <typename T, typename Context>                    \
  void name##GradKernel(const Context& dev_ctx,              \
                        const DenseTensor& out,              \
                        const DenseTensor& dout,             \
                        float attr,                          \
                        DenseTensor* dx) {                   \
    funcs::functor_class<T> functor;                         \
    auto attrs = functor.GetAttrs();                         \
    *(attrs[0].second) = attr;                               \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, nullptr, &out, &dout, dx, functor);         \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(    \
    name, functor_class, attr1, attr2)                       \
  template <typename T, typename Context>                    \
  void name##GradKernel(const Context& dev_ctx,              \
                        const DenseTensor& out,              \
                        const DenseTensor& dout,             \
                        float attr1,                         \
                        float attr2,                         \
                        DenseTensor* dx) {                   \
    funcs::functor_class<T> functor;                         \
    auto attrs = functor.GetAttrs();                         \
    *(attrs[0].second) = attr1;                              \
    *(attrs[1].second) = attr2;                              \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, nullptr, &out, &dout, dx, functor);         \
  }

#define DEFINE_CPU_ACTIVATION_GRAD_KERNEL_NODEP(name, functor_class)      \
  template <typename T, typename Context>                                 \
  void name##GradKernel(                                                  \
      const Context& dev_ctx, const DenseTensor& dout, DenseTensor* dx) { \
    funcs::functor_class<T> functor;                                      \
    ActivationGradImpl<T, Context, funcs::functor_class<T>>(              \
        dev_ctx, nullptr, nullptr, &dout, dx, functor);                   \
  }

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Cos, CosGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Tan, TanGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Acos, AcosGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Sin, SinGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Asin, AsinGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Atan, AtanGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Sinh, SinhGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Cosh, CoshGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Asinh, AsinhGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Acosh, AcoshGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Atanh, AtanhGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(TanhShrink, TanhShrinkGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Square, SquareGradFunctor);

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Exp, ExpGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Expm1, Expm1GradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Reciprocal, ReciprocalGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Sqrt, SqrtGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Rsqrt, RsqrtGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu6, Relu6GradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Softsign, SoftsignGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(LogSigmoid, LogSigmoidGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Log, LogGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Log2, Log2GradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Log10, Log10GradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Log1p, Log1pGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPX(Swish, SwishGradFunctor);

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu, ReluGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Tanh, TanhGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Sigmoid, SigmoidGradFunctor);

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_NODEP(Round, ZeroGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_NODEP(Floor, ZeroGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_NODEP(Ceil, ZeroGradFunctor);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(LeakyRelu,
                                               LeakyReluGradFunctor,
                                               alpha);
DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(SoftShrink,
                                               SoftShrinkGradFunctor,
                                               lambda);
DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(HardShrink,
                                               HardShrinkGradFunctor,
                                               threshold);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Mish,
                                               MishGradFunctor,
                                               threshold);
DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Celu, CELUGradFunctor, alpha);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(HardTanh,
                                               HardTanhGradFunctor,
                                               t_min,
                                               t_max);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(STanh,
                                               STanhGradFunctor,
                                               scale_a,
                                               scale_b);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(Softplus,
                                               SoftplusGradFunctor,
                                               beta,
                                               threshold);
DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(HardSigmoid,
                                                 HardSigmoidGradFunctor,
                                                 slope,
                                                 offset);
DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(ThresholdedRelu,
                                               ThresholdedReluGradFunctor,
                                               threshold,
                                               value);

template <typename T, typename Context>
void SiluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  funcs::SiluGradFunctor<T> functor;
  ActivationGradImpl<T, Context, funcs::SiluGradFunctor<T>>(
      dev_ctx, &x, &out, &dout, dx, functor);
}

template <typename T, typename Context>
void EluGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   float alpha,
                   DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  auto x_flatten =
      EigenVector<T>::Flatten(GET_DATA_SAFELY(&x, "Input", "X", "elu_grad"));
  auto out_flatten = EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&out, "Input", "Out", "elu_grad"));
  auto dout_flatten = EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&dout, "Input", "dOut", "elu_grad"));
  auto dx_flatten =
      EigenVector<T>::Flatten(GET_DATA_SAFELY(dx, "Output", "dX", "elu_grad"));
  auto* place = dev_ctx.eigen_device();

  if (alpha > 0) {
    funcs::ELUGradFunctor<T> functor;
    functor.alpha = alpha;
    functor(*place, x_flatten, out_flatten, dout_flatten, dx_flatten);
  } else {
    funcs::ELUGradNegativeAlphaFunctor<T> functor;
    functor.alpha = alpha;
    functor(*place, x_flatten, out_flatten, dout_flatten, dx_flatten);
  }
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         DenseTensor* dx) {
  funcs::HardSwishGradFunctor<T> functor;
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = threshold;
  *(attrs[1].second) = scale;
  *(attrs[2].second) = offset;
  ActivationGradImpl<T, Context, funcs::HardSwishGradFunctor<T>>(
      dev_ctx, &x, nullptr, &dout, dx, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    relu_grad, CPU, ALL_LAYOUT, phi::ReluGradKernel, float, double) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name, CPU, ALL_LAYOUT, phi::func, float, double) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(name, func) \
  PD_REGISTER_KERNEL(name,                                          \
                     CPU,                                           \
                     ALL_LAYOUT,                                    \
                     phi::func,                                     \
                     float,                                         \
                     double,                                        \
                     phi::dtype::complex<float>,                    \
                     phi::dtype::complex<double>) {}

#define PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(                                         \
      name, CPU, ALL_LAYOUT, phi::func, float, double, phi::dtype::float16) {}

#define PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL_WITH_COMPLEX(name, func) \
  PD_REGISTER_KERNEL(name,                                                 \
                     CPU,                                                  \
                     ALL_LAYOUT,                                           \
                     phi::func,                                            \
                     float,                                                \
                     double,                                               \
                     phi::dtype::float16,                                  \
                     phi::dtype::complex<float>,                           \
                     phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sin_grad, SinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(cos_grad, CosGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(tan_grad, TanGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(acos_grad, AcosGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(asin_grad, AsinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(atan_grad, AtanGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sinh_grad, SinhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(cosh_grad, CoshGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(asinh_grad, AsinhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(acosh_grad, AcoshGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(atanh_grad, AtanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(tanh_grad, TanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hardtanh_grad, HardTanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(thresholded_relu_grad,
                                   ThresholdedReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(relu6_grad, Relu6GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(softshrink_grad, SoftShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hard_shrink_grad, HardShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_shrink_grad, TanhShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(elu_grad, EluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(silu_grad, SiluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(mish_grad, MishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(stanh_grad, STanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(reciprocal_grad,
                                                ReciprocalGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sqrt_grad, SqrtGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(rsqrt_grad, RsqrtGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(softplus_grad,
                                                SoftplusGradKernel)

PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(relu_double_grad,
                                          ReluDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL_WITH_COMPLEX(tanh_double_grad,
                                                       TanhDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(leaky_relu_double_grad,
                                          LeakyReluDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(elu_double_grad, EluDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(sqrt_double_grad,
                                          SqrtDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(rsqrt_double_grad,
                                          RsqrtDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL_WITH_COMPLEX(softplus_double_grad,
                                                       SoftplusDoubleGradKernel)

PD_REGISTER_KERNEL(tanh_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::TanhTripleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(exp_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ExpGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(expm1_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::Expm1GradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(
    logit_grad, CPU, ALL_LAYOUT, phi::LogitGradKernel, float, double) {}
PD_REGISTER_KERNEL(square_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SquareGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(square_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SquareDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(sin_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SinDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(sin_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SinTripleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(cos_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CosDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(cos_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CosTripleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(softsign_grad,
                                                SoftsignGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sigmoid_grad, SigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sigmoid_double_grad,
                                                SigmoidDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sigmoid_triple_grad,
                                                SigmoidTripleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hardsigmoid_grad, HardSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(logsigmoid_grad,
                                                LogSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log_grad, LogGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log2_grad, Log2GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log10_grad, Log10GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log1p_grad, Log1pGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL_WITH_COMPLEX(log_double_grad,
                                                       LogDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(hardswish_grad,
                                                HardSwishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(swish_grad, SwishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(round_grad, RoundGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(floor_grad, FloorGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(ceil_grad, CeilGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(celu_grad, CeluGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(celu_double_grad,
                                          CeluDoubleGradKernel)

PD_REGISTER_KERNEL(pow_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::PowGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(pow_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::PowDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(pow_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::PowTripleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
