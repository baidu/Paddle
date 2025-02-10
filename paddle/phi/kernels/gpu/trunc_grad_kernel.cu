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

#include "paddle/phi/kernels/trunc_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void TruncGrad(T* dx, int64_t N) {
  CUDA_KERNEL_LOOP(index, N) { dx[index] = static_cast<T>(0.0); }
}

template <typename T, typename Context>
void TruncGradKernel(const Context& dev_ctx,
                     const DenseTensor& out_grad,
                     DenseTensor* in_grad) {
  const auto* out_grad_data = out_grad.data<T>();
  T* in_grad_data = dev_ctx.template Alloc<T>(in_grad);

  int64_t numel = out_grad.numel();

  int threads = PADDLE_CUDA_NUM_THREADS;
  int blocks = (numel + threads - 1) / threads;

  TruncGrad<<<blocks, threads>>>(in_grad_data, numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(trunc_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TruncGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
