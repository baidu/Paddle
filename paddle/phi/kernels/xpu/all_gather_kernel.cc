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

#include "paddle/phi/kernels/all_gather_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllGatherKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int nranks,
                     DenseTensor* out) {
#if defined(PADDLE_WITH_XPU_BKCL)
  auto out_dims = x.dims();
  out_dims[0] *= nranks;
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::BKCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "BKCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));

  PADDLE_ENFORCE_EQ(
      nranks,
      comm_ctx->GetSize(),
      errors::InvalidArgument(
          "nranks: %s should equal to %s", nranks, comm_ctx->GetSize()));

  XPUStream stream = comm_ctx->GetStream();
  comm_ctx->AllGather(out, x, stream);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should be compiled with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(all_gather,
                   XPU,
                   ALL_LAYOUT,
                   phi::AllGatherKernel,
                   int,
                   int64_t,
                   bool,
                   uint8_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
