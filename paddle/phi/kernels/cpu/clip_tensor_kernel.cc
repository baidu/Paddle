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

#include "paddle/phi/kernels/clip_tensor_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  const T* x_data = x.data<T>();
  const T* min_data = min.data<T>();
  const T* max_data = max.data<T>();

  auto x_numel = x.numel();

  T* out_data = dev_ctx.template Alloc<T>(out);

  for (int i = 0; i < x_numel; i++) {
    out_data[i] = x_data[i] < min_data[i] ? min_data[i] : x_data[i];
    out_data[i] = out_data[i] > max_data[i] ? max_data[i] : out_data[i];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::ClipTensorKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
