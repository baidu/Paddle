/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/reduce_kernel_impl.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// oneDNN's reduction kernel is optimized only for reducing throughout the
// most outer dims, so in case of another type of reduction, it would be
// better to fallback to native implementation
inline bool HasOptimizedOneDNNKernel(const KernelContext* ctx) {
  const DenseTensor& x = ctx->InputAt<phi::DenseTensor>(0);
  IntArray dims_array;
  const TensorRef& dims_tmp = ctx->AttrAt<TensorRef>(0);
  dims_array = IntArray(*dims_tmp.Get());
  int ndims = x.dims().size();
  const bool reduce_all = recompute_reduce_all(x, dims_array);
  auto dims = dims_array.GetData();

  // native reduce kernels don't support bf16
  // so oneDNN kernel is enforced in that case
  if (x.dtype() == phi::DataType::BFLOAT16) return true;

  if (reduce_all) {
    return true;
  }

  for (auto& dim : dims) {
    if (dim < 0) {
      dim += ndims;
    }
  }

  sort(dims.begin(), dims.end());

  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[dims.size() - i - 1] != static_cast<int>(ndims - i - 1)) {
      return false;
    }
  }

  return true;
}

bool ReduceCheckIfOneDNNSupport(const KernelContext* ctx) {
  if (ctx->InputAt<phi::DenseTensor>(0).dims().size() > 5 ||
      !HasOptimizedOneDNNKernel(ctx)) {
    return false;
  }
  return true;
}

bool ReduceMeanCheckIfOneDNNSupport(const KernelContext* ctx) {
  std::cout << "ReduceMeanCheckIfOneDNNSupport" << std::endl;
  if (ctx->InputAt<phi::DenseTensor>(0).dims().size() > 5 ||
      !HasOptimizedOneDNNKernel(ctx)) {
    return false;
  }
  return true;
}

bool ReduceGradCheckIfOneDNNSupport(const KernelContext* ctx) {
  if (ctx->InputAt<phi::DenseTensor>(0).dims().size() > 5) {
    return false;
  }
  return true;
}

}  // namespace phi
