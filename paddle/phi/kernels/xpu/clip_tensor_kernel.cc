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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  DenseTensor tem_min;
  MetaTensor meta_tem_min(&tem_min);
  CastInferMeta(min, x.dtype(), &meta_tem_min);
  CastKernel<T, Context>(dev_ctx, min, x.dtype(), &tem_min);
  DenseTensor tem_max;
  MetaTensor meta_tem_max(&tem_max);
  CastInferMeta(max, x.dtype(), &meta_tem_max);
  CastKernel<T, Context>(dev_ctx, max, x.dtype(), &tem_max);

  DenseTensor x_ls_min;
  MetaTensor meta_x_ls_min(&x_ls_min);
  UnchangedInferMeta(x, &meta_x_ls_min);
  phi::LessThanKernel<T, Context>(dev_ctx, x, tem_min, &x_ls_min);
  DenseTensor tem_out;
  MetaTensor meta_tem_out(&tem_out);
  UnchangedInferMeta(x, &meta_tem_out);
  phi::WhereKernel<T, Context>(dev_ctx, x_ls_min, tem_min, x, &tem_out);

  DenseTensor x_gt_max;
  MetaTensor meta_x_gt_max(&x_gt_max);
  UnchangedInferMeta(x, &meta_x_gt_max);
  phi::GreaterThanKernel<T, Context>(dev_ctx, x, tem_max, &x_gt_max);
  phi::WhereKernel<T, Context>(dev_ctx, x_gt_max, tem_max, tem_out, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipTensorKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}
