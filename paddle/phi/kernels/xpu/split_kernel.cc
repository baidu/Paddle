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

#include "paddle/phi/kernels/split_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& sections,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int axis = axis_scalar.to<int>();
  auto in_dims = x.dims();
  auto input_shape = common::vectorize<int>(in_dims);
  std::vector<XPUType*> out_ptrs;
  std::vector<int> split_lists;

  // Vectors to keep track of zero-sized and non-zero-sized outputs
  std::vector<XPUType*> non_zero_out_ptrs;
  std::vector<int> non_zero_split_lists;

  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    out_ptrs.push_back(reinterpret_cast<XPUType*>(outs[j]->data<T>()));
    int section_size =
        axis < outs[j]->dims().size() ? outs[j]->dims()[axis] : 1;
    split_lists.push_back(section_size);

    if (section_size > 0) {
      non_zero_out_ptrs.push_back(
          reinterpret_cast<XPUType*>(outs[j]->data<T>()));
      non_zero_split_lists.push_back(section_size);
    } else {
      auto zero_dims = in_dims;
      zero_dims[axis] = 0;
      outs[j]->Resize(zero_dims);
    }
  }

  if (x.numel() == 0) {
    return;
  }

  // Perform the split operation only on non-zero sections
  if (!non_zero_split_lists.empty()) {
    int r = xpu::split<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x.data<T>()),
                                non_zero_out_ptrs,
                                input_shape,
                                non_zero_split_lists,
                                axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "split");
  }
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int num,
                        const Scalar& axis_scalar,
                        std::vector<DenseTensor*> outs) {
  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  IntArray sections(sections_vec);
  SplitKernel<T, Context>(dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(split,
                   XPU,
                   ALL_LAYOUT,
                   phi::SplitKernel,
                   float,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(split_with_num,
                   XPU,
                   ALL_LAYOUT,
                   phi::SplitWithNumKernel,
                   float,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
