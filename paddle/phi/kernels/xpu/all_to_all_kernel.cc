// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/all_to_all_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllToAllKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
#if defined(PADDLE_WITH_XPU_BKCL)
  auto x_dims = x.dims();
  out->Resize(x_dims);
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::BKCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "BKCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));

  XPUStream stream = comm_ctx->GetStream();
  int nranks = comm_ctx->GetSize();
  PADDLE_ENFORCE_EQ(
      x_dims[0] % nranks,
      0,
      errors::InvalidArgument(
          "The first dimension size (%d) of the input tensor must be "
          "divisible by the number of ranks (%d).",
          x_dims[0],
          nranks));

  comm_ctx->AllToAll(out, x, stream);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should be compiled with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(all_to_all,
                   XPU,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   int,
                   int64_t,
                   bool,
                   uint8_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
