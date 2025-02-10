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

#include "paddle/phi/kernels/eigh_kernel.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/values_vectors_functor.h"

namespace phi {

template <typename T, typename Context>
void EighKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::string& uplo,
                DenseTensor* out_w,
                DenseTensor* out_v) {
  if (x.numel() == 0) {
    auto x_dim = x.dims();
    auto w_dim = slice_ddim(x_dim, 0, x_dim.size() - 1);
    out_w->Resize(w_dim);
    out_v->Resize(x_dim);
    dev_ctx.template Alloc<T>(out_w);
    dev_ctx.template Alloc<T>(out_v);
    return;
  }
  bool is_lower = (uplo == "L");
  phi::funcs::MatrixEighFunctor<Context, T> functor;
  functor(dev_ctx, x, out_w, out_v, is_lower, true);
}

}  // namespace phi
#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(eigh, GPU, ALL_LAYOUT, phi::EighKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
#else
PD_REGISTER_KERNEL(eigh,
                   GPU,
                   ALL_LAYOUT,
                   phi::EighKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
#endif
