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

#pragma once
#include "glog/logging.h"
#include "paddle/phi/kernels/clip_by_norm_kernel.h"

namespace phi {

template <typename T, typename Context>
void DGCClipByNormKernel(const Context& dev_ctx,
                         const DenseTensor& x_in,
                         const DenseTensor& current_step_in,
                         float max_norm,
                         float rampup_begin_step,
                         DenseTensor* out) {
  if (static_cast<int>(rampup_begin_step) < 0) {
    return;
  }

  auto current_step_tensor = &current_step_in;
  auto* current_step = current_step_tensor->data<T>();

  VLOG(10) << "current_step:" << *current_step
           << ", rampup_begin_step:" << rampup_begin_step;

  if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
    VLOG(10) << "current_step:" << *current_step
             << " < rampup_begin_step:" << rampup_begin_step
             << " so doesn't use dgc_clip_by_norm";
    return;
  }

  auto* x = &x_in;
  auto* y = out;
  return phi::ClipByNormKernel<T>(dev_ctx, *x, max_norm, y);
}
}  // namespace phi
