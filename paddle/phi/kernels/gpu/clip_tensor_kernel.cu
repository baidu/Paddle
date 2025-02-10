// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

template <typename T>
struct ClipTensorFunctor {
  inline HOSTDEVICE T operator()(const T x, const T min_, const T max_) const {
    T x_ = x < min_ ? min_ : x;
    T x__ = x_ > max_ ? max_ : x_;
    return x__;
  }
};

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  DenseTensor tem_min;
  MetaTensor meta_tem_min(&tem_min);
  CastInferMeta(min, x.dtype(), &meta_tem_min);
  CastKernel<T, Context>(dev_ctx, min, x.dtype(), &tem_min);
  DenseTensor tem_max;
  MetaTensor meta_tem_max(&tem_max);
  CastInferMeta(max, x.dtype(), &meta_tem_max);
  CastKernel<T, Context>(dev_ctx, max, x.dtype(), &tem_max);

  std::vector<const DenseTensor*> ins = {&x, &tem_min, &tem_max};
  std::vector<DenseTensor*> outs = {out};
  dev_ctx.template Alloc<T>(out);

  ClipTensorFunctor<T> func;
  funcs::ElementwiseKernel<T, ClipTensorFunctor<T>, 1>(
      dev_ctx, ins, &outs, func);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ClipTensorKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
