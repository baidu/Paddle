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

#include "paddle/phi/kernels/clip_tensor_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename T>
__global__ void ClipTensorGradFunctor(const int N,
                                      const T* out_grad,
                                      const T* x,
                                      const T* min,
                                      const T* max,
                                      T* x_grad) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    x_grad[idx] = (x[idx] > min[idx]) && (x[idx] < max[idx])
                      ? out_grad[idx]
                      : static_cast<T>(0);
  }
};

template <typename T, typename Context>
void ClipTensorGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& min,
                          const DenseTensor& max,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad) {
  DenseTensor ex_min;
  DenseTensor ex_max;
  DenseTensor ex_x;
  std::vector<int> real_target_shape = common::vectorize<int>(x_grad->dims());
  if (x.dims() != x_grad->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, x, real_target_shape, &ex_x);
  } else {
    ex_x = x;
  }
  if (min.dims() != x_grad->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, min, real_target_shape, &ex_min);
  } else {
    ex_min = min;
  }
  if (max.dims() != x_grad->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, max, real_target_shape, &ex_max);
  } else {
    ex_max = max;
  }
  phi::CastKernel<T, Context>(dev_ctx, ex_min, ex_x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, ex_max, ex_x.dtype(), &ex_max);

  const T* x_data = ex_x.data<T>();
  auto numel = ex_x.numel();
  const T* min_data = ex_min.data<T>();
  const T* max_data = ex_max.data<T>();
  const T* out_grad_data = out_grad.data<T>();

  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  auto stream = dev_ctx.stream();
  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  ClipTensorGradFunctor<T>
      <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
          numel, out_grad_data, x_data, min_data, max_data, x_grad_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ClipTensorGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
