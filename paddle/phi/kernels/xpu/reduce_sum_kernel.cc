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

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              XPUType* y,
              const std::vector<int>& xdims,
              const std::vector<int>& reduce_dims) {
    return xpu::reduce_sum<XPUType>(ctx, x, y, xdims, reduce_dims);
  };
  int r = XPUReduce<Context, XPUType>(
      dev_ctx, x, dims.GetData(), keep_dim, reduce_all, out, f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    sum_raw, XPU, ALL_LAYOUT, phi::SumRawKernel, float, int8_t, int64_t) {}
