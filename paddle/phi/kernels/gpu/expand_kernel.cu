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

#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename T, typename Context>
void ExpandKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& shape,
                  DenseTensor* out) {
  auto expand_shape = shape.GetData();
  auto diff = expand_shape.size() - x.dims().size();
  auto out_shape = common::vectorize<int64_t>(x.dims());
  out_shape.insert(out_shape.begin(), diff, 1);
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (expand_shape[i] == 0) {
      out_shape[i] = 0;
    } else if (expand_shape[i] > 0) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
          true,
          common::errors::InvalidArgument(
              "The %d-th dimension of input tensor (%d) must match or be "
              "broadcastable to the corresponding dimension (%d) in shape.",
              i,
              vec_in_dims[i],
              expand_shape[i]));
      out_shape[i] = expand_shape[i];
    } else if (expand_shape[i] == -1) {
      out_shape[i] = vec_in_dims[i];
    }
  }

  out->Resize(common::make_ddim(out_shape));
  ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  phi::funcs::BroadcastKernel<T>(ctx, ins, &outs, kps::IdentityFunctor<T>());
}

}  // namespace phi

PD_REGISTER_KERNEL(expand,
                   GPU,
                   ALL_LAYOUT,
                   phi::ExpandKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
