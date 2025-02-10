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
#pragma once

#include "paddle/phi/kernels/concat_grad_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"

namespace phi {

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const DenseTensor*>& x,
                      const DenseTensor& out_grad,
                      const Scalar& axis_scalar,
                      std::vector<DenseTensor*> x_grad) {
  auto outs = x_grad;
  {
    auto dx = x_grad;
    for (size_t i = 0; i < dx.size(); ++i) {
      if (dx[i] != nullptr) {
        dx[i]->set_lod(x[i]->lod());
      }
    }
  }
  PADDLE_ENFORCE_NOT_NULL(
      x[0],
      common::errors::NotFound("The first input tensor is not initialized."));

  auto axis = axis_scalar.to<int>();
  axis = funcs::ComputeAxis(static_cast<int64_t>(axis),
                            static_cast<int64_t>(x[0]->dims().size()));
  // get output tensor that the name is not kEmptyVarName
  std::vector<DenseTensor*> outputs;
  for (size_t j = 0; j < outs.size(); ++j) {
    if (outs[j] && outs[j]->numel() != 0UL) {
      dev_ctx.template Alloc<T>(outs[j]);

      outputs.push_back(outs[j]);
    } else {
      outputs.push_back(nullptr);
    }
  }

  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && outs.size() < 10) {
    std::vector<const DenseTensor*> ref_shape;
    ref_shape.insert(ref_shape.begin(), x.begin(), x.end());
    phi::funcs::StridedMemcpyWithAxis0<T, Context>(
        dev_ctx, out_grad, ref_shape, &outputs);
  } else {
    phi::funcs::SplitFunctor<Context, T> split_functor;
    split_functor(dev_ctx, out_grad, x, static_cast<int>(axis), &outputs);
  }
}

}  // namespace phi
