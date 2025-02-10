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


#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/lu_solve_grad_kernel.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"

namespace phi {

template <typename T, typename Context>
void LuSolveGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& lu,
                       const DenseTensor& pivots,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       const std::string& trans,
                       DenseTensor* x_grad,
                       DenseTensor* lu_grad) {
  // Allocate memory for x_grad
  dev_ctx.template Alloc<T>(x_grad);

  // Use the forward kernel to compute the gradient
  LuSolveKernel<T, Context>(dev_ctx, out_grad, lu, pivots, trans, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    lu_solve_grad, GPU, ALL_LAYOUT, phi::LuSolveGradKernel, float, double) {}
