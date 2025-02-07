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

#include "paddle/phi/kernels/activation_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGPUImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out,
                       const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

#define DEFINE_GPU_ACTIVATION_KERNEL(name, functor_class)               \
  template <typename T, typename Context>                               \
  void name##Kernel(                                                    \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) { \
    funcs::functor_class<T> functor;                                    \
    ActivationGPUImpl<T, Context, funcs::functor_class<T>>(             \
        dev_ctx, x, out, functor);                                      \
  }

#define DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(name,           \
                                                           functor_class)  \
  template <typename T, typename Context>                                  \
  void name##Kernel(                                                       \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {    \
    funcs::functor_class<T> functor;                                       \
    using U =                                                              \
        typename std::conditional_t<std::is_integral<T>::value, float, T>; \
    ActivationGPUImpl<U, Context, funcs::functor_class<T>>(                \
        dev_ctx, x, out, functor);                                         \
  }

#define DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(name, functor_class, attr) \
  template <typename T, typename Context>                               \
  void name##Kernel(const Context& dev_ctx,                             \
                    const DenseTensor& x,                               \
                    float attr,                                         \
                    DenseTensor* out) {                                 \
    funcs::functor_class<T> functor;                                    \
    auto attrs = functor.GetAttrs();                                    \
    *(attrs[0].second) = attr;                                          \
    ActivationGPUImpl<T, Context, funcs::functor_class<T>>(             \
        dev_ctx, x, out, functor);                                      \
  }

#define DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(               \
    name, functor_class, attr1, attr2)                      \
  template <typename T, typename Context>                   \
  void name##Kernel(const Context& dev_ctx,                 \
                    const DenseTensor& x,                   \
                    float attr1,                            \
                    float attr2,                            \
                    DenseTensor* out) {                     \
    funcs::functor_class<T> functor;                        \
    auto attrs = functor.GetAttrs();                        \
    *(attrs[0].second) = attr1;                             \
    *(attrs[1].second) = attr2;                             \
    ActivationGPUImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, x, out, functor);                          \
  }

DEFINE_GPU_ACTIVATION_KERNEL(Cos, CudaCosFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Tan, CudaTanFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Acos, CudaAcosFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sin, CudaSinFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Asin, CudaAsinFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Atan, CudaAtanFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sinh, CudaSinhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Cosh, CudaCoshFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Asinh, CudaAsinhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Acosh, CudaAcoshFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Atanh, CudaAtanhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Relu, CudaReluFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Tanh, CudaTanhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(TanhShrink, CudaTanhShrinkFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Silu, CudaSiluFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Reciprocal, CudaReciprocalFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Square, CudaSquareFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sqrt, CudaSqrtFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Rsqrt, CudaRsqrtFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Softsign, CudaSoftsignFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sigmoid, CudaSigmoidFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(LogSigmoid, CudaLogSigmoidFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Floor, CudaFloorFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Ceil, CudaCeilFunctor)

DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(Log, CudaLogFunctor)
DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(Log2, CudaLog2Functor)
DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(Log10, CudaLog10Functor)
DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(Log1p, CudaLog1pFunctor)
DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(Exp, CudaExpFunctor)
DEFINE_GPU_ACTIVATION_KERNEL_WITH_INT_IN_FLOAT_OUT(Expm1, CudaExpm1Functor)

DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, CudaLeakyReluFunctor, alpha)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(LogitCUDA, CudaLogitFunctor, eps)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(HardShrink,
                                     CudaHardShrinkFunctor,
                                     threshold)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(SoftShrink, CudaSoftShrinkFunctor, lambda)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(Elu, CudaELUFunctor, alpha)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(Mish, CudaMishFunctor, threshold)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(Celu, CudaCELUFunctor, alpha)

DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(HardTanh,
                                     CudaHardTanhFunctor,
                                     t_min,
                                     t_max)
DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(Stanh, CudaSTanhFunctor, scale_a, scale_b)
DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(Softplus,
                                     CudaSoftplusFunctor,
                                     beta,
                                     threshold)
DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(HardSigmoid,
                                     CudaHardSigmoidFunctor,
                                     slope,
                                     offset)
DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(Selu, CudaSeluFunctor, scale, alpha)
DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(ThresholdedRelu,
                                     CudaThresholdedReluFunctor,
                                     threshold,
                                     value)

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  funcs::CudaHardSwishFunctor<T> functor;
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = threshold;
  *(attrs[1].second) = scale;
  *(attrs[2].second) = offset;
  ActivationGPUImpl<T, Context, funcs::CudaHardSwishFunctor<T>>(
      dev_ctx, x, out, functor);
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  funcs::CudaSwishFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = 1.0;
  ActivationGPUImpl<T, Context, funcs::CudaSwishFunctor<T>>(
      dev_ctx, x, out, functor);
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  funcs::CudaRelu6Functor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = 6.0;
  ActivationGPUImpl<T, Context, funcs::CudaRelu6Functor<T>>(
      dev_ctx, x, out, functor);
}

template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const int decimals,
                 DenseTensor* out) {
  funcs::CudaRoundFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = decimals;
  ActivationGPUImpl<T, Context, funcs::CudaRoundFunctor<T>>(
      dev_ctx, x, out, functor);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name,                        \
                     GPU,                         \
                     ALL_LAYOUT,                  \
                     phi::func,                   \
                     float,                       \
                     double,                      \
                     phi::dtype::float16,         \
                     phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(name, func) \
  PD_REGISTER_KERNEL(name,                                     \
                     GPU,                                      \
                     ALL_LAYOUT,                               \
                     phi::func,                                \
                     float,                                    \
                     double,                                   \
                     phi::dtype::float16,                      \
                     phi::dtype::bfloat16,                     \
                     phi::dtype::complex<float>,               \
                     phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sin, SinKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(cos, CosKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(tan, TanKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(acos, AcosKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(asin, AsinKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(atan, AtanKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sinh, SinhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(cosh, CoshKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(asinh, AsinhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(acosh, AcoshKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(atanh, AtanhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(tanh, TanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(hardtanh, HardTanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(thresholded_relu, ThresholdedReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(relu6, Relu6Kernel)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(mish, MishKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(stanh, StanhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(reciprocal, ReciprocalKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sqrt, SqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL(rsqrt, RsqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(softplus, SoftplusKernel)

PD_REGISTER_KERNEL(exp,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(expm1,
                   GPU,
                   ALL_LAYOUT,
                   phi::Expm1Kernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(square,
                   GPU,
                   ALL_LAYOUT,
                   phi::SquareKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_KERNEL(hard_shrink, HardShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(softshrink, SoftShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh_shrink, TanhShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(silu, SiluKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(softsign, SoftsignKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sigmoid, SigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(logsigmoid, LogSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(hardsigmoid, HardSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(hardswish, HardSwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(swish, SwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(round, RoundKernel)
PD_REGISTER_ACTIVATION_KERNEL(floor, FloorKernel)
PD_REGISTER_ACTIVATION_KERNEL(ceil, CeilKernel)
PD_REGISTER_ACTIVATION_KERNEL(celu, CeluKernel)
PD_REGISTER_ACTIVATION_KERNEL(logit, LogitCUDAKernel)

PD_REGISTER_KERNEL(log,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(log2,
                   GPU,
                   ALL_LAYOUT,
                   phi::Log2Kernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(log10,
                   GPU,
                   ALL_LAYOUT,
                   phi::Log10Kernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(log1p,
                   GPU,
                   ALL_LAYOUT,
                   phi::Log1pKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(pow,
                   GPU,
                   ALL_LAYOUT,
                   phi::PowKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(selu,
                   GPU,
                   ALL_LAYOUT,
                   phi::SeluKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
