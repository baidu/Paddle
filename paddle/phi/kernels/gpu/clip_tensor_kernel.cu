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
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

template <typename T>
struct ClipTensorFunctor {
  inline HOSTDEVICE T operator()(const T x, const T min_, const T max_) const {
    return x < min_ ? min_ : x > max_ ? max_ : x;
  }
};

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  DenseTensor ex_min;
  DenseTensor ex_max;
  DenseTensor ex_x;
  std::vector<int> real_target_shape = common::vectorize<int>(out->dims());
  if (x.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, x, real_target_shape, &ex_x);
  } else {
    ex_x = x;
  }
  if (min.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, min, real_target_shape, &ex_min);
  } else {
    ex_min = min;
  }
  if (max.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, max, real_target_shape, &ex_max);
  } else {
    ex_max = max;
  }
  phi::CastKernel<T, Context>(dev_ctx, ex_min, ex_x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, ex_max, ex_x.dtype(), &ex_max);

  std::vector<const DenseTensor*> ins = {&ex_x, &ex_min, &ex_max};
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
