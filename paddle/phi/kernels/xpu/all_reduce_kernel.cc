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

#include "paddle/phi/kernels/all_reduce_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllReduceKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int reduce_type,
                     DenseTensor* out) {
#if defined(PADDLE_WITH_XPU_BKCL)
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  XPUStream stream = nullptr;
  auto comm_ctx =
      static_cast<distributed::BKCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "BKCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  stream = comm_ctx->GetStream();

  BKCLOp bkcl_reduce_type = BKCL_ADD;
  switch (static_cast<ReduceType>(reduce_type)) {
    case ReduceType::kRedSum:
      bkcl_reduce_type = BKCL_ADD;
      break;

    case ReduceType::kRedMax:
      bkcl_reduce_type = BKCL_MAX;
      break;

    case ReduceType::kRedMin:
      bkcl_reduce_type = BKCL_MIN;
      break;

    case ReduceType::kRedProd:
      bkcl_reduce_type = BKCL_PRODUCT;
      break;

    default:
      PADDLE_THROW(common::errors::InvalidArgument("Invalid reduce type: %d",
                                                   reduce_type));
  }

  comm_ctx->AllReduce(out, x, bkcl_reduce_type, stream);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should be compiled with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(all_reduce,
                   XPU,
                   ALL_LAYOUT,
                   phi::AllReduceKernel,
                   int,
                   int64_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
