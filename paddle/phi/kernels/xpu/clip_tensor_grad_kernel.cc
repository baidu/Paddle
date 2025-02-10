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

#include "paddle/phi/kernels/clip_tensor_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/logical_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& min,
                          const DenseTensor& max,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad) {
  phi::DenseTensor x_ls_min;
  MetaTensor meta_x_ls_min(&x_ls_min);
  UnchangedExceptDtypeInferMeta(x, &meta_x_ls_min);
  meta_x_ls_min.set_dtype(phi::DataType::BOOL);
  phi::LessThanKernel<T, Context>(dev_ctx, min, x, &x_ls_min);

  phi::DenseTensor x_ls_max;
  MetaTensor meta_x_ls_max(&x_ls_max);
  UnchangedExceptDtypeInferMeta(x, &meta_x_ls_max);
  meta_x_ls_max.set_dtype(phi::DataType::BOOL);
  phi::LessThanKernel<T, Context>(dev_ctx, x, max, &x_ls_max);

  phi::DenseTensor out;
  MetaTensor meta_out(&out);
  UnchangedExceptDtypeInferMeta(x, &meta_out);
  meta_out.set_dtype(phi::DataType::BOOL);
  phi::LogicalAndKernel<bool, Context>(dev_ctx, x_ls_min, x_ls_max, &out);

  phi::DenseTensor zero_tensor;
  MetaTensor meta_zero(&zero_tensor);
  UnchangedInferMeta(x_grad, &meta_zero);
  phi::FullKernel<T, Context>(dev_ctx,
                         common::vectorize(x_grad->dims()),
                         0.0f,
                         zero_tensor.dtype(),
                         &zero_tensor);
  phi::WhereKernel<T, Context>(dev_ctx, out, out_grad, zero_tensor, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipTensorGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}
