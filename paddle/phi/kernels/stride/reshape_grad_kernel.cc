// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/reshape_grad_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reshape_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void ReshapeGradStridedKernel(const Context& dev_ctx,
                              const DenseTensor& out_grad,
                              DenseTensor* x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(
        phi::errors::Fatal("FLAGS_use_stride_kernel is closed. Strided kernel "
                           "be called, something wrong has happened!"));
  }
  ReshapeStridedKernel<Context>(
      dev_ctx,
      out_grad,
      IntArray(common::vectorize<int64_t>(x_grad->dims())),
      x_grad);
}

template <typename Context>
void ReshapeDoubleGradStridedKernel(const Context& dev_ctx,
                                    const DenseTensor& out_grad UNUSED,
                                    const DenseTensor& x_grad_grad,
                                    DenseTensor* out_grad_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(
        phi::errors::Fatal("FLAGS_use_stride_kernel is closed. Strided kernel "
                           "be called, something wrong has happened!"));
  }
  ReshapeGradStridedKernel<Context>(dev_ctx, x_grad_grad, out_grad_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape_grad,
                                         STRIDED,
                                         phi::ReshapeGradStridedKernel) {}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape_double_grad,
                                         STRIDED,
                                         phi::ReshapeDoubleGradStridedKernel) {}
