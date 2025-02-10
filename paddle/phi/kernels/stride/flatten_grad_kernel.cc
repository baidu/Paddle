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
#include "paddle/phi/kernels/flatten_grad_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reshape_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void FlattenGradStridedKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& out_grad,
                              DenseTensor* x_grad) {
<<<<<<< HEAD
  const auto& xshape_dims = xshape.dims();
=======
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
<<<<<<< HEAD
  auto xshape_dims = xshape.dims();
>>>>>>> 8b808f1ca1f3b066d25661279d07b83806836d58
  auto x_dims = common::slice_ddim(xshape_dims, 1, xshape_dims.size());
  ReshapeStridedKernel<Context>(dev_ctx,
                                out_grad,
                                IntArray(common::vectorize<int64_t>(x_dims)),
                                x_grad,
                                nullptr);
=======
  ReshapeStridedKernel<Context>(
      dev_ctx,
      out_grad,
      IntArray(common::vectorize<int64_t>(x_grad->dims())),
      x_grad);
>>>>>>> 4c9bc9e3cd7680200be9f244f9a5d374345a6741
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(flatten_grad,
                                         STRIDED,
                                         phi::FlattenGradStridedKernel) {}
