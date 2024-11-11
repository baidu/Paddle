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
#include "paddle/phi/kernels/tensor_slice_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void TensorSliceKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       int64_t begin_idx,
                       int64_t end_idx,
                       DenseTensor* out) {
  *out = input.Slice(begin_idx, end_idx);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(tensor_slice,
                                         ALL_LAYOUT,
                                         phi::TensorSliceKernel) {}
