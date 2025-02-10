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

#include "paddle/phi/kernels/arange_kernel.h"

#include "paddle/common/errors.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename OUT_TYPE>
__global__ void Range(T start, T step, int64_t size, OUT_TYPE* out) {
  CUDA_KERNEL_LOOP(index, size) {
    out[index] = static_cast<OUT_TYPE>(start + step * index);
  }
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const DenseTensor& start,
                        const DenseTensor& end,
                        const DenseTensor& step,
                        DenseTensor* out) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType start_value =
      static_cast<MPType>(GetValue<T, Context>(dev_ctx, start));
  MPType end_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, end));
  MPType step_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, step));

  int64_t size = 0;
  phi::funcs::GetSize(start_value, end_value, step_value, &size);
  PADDLE_ENFORCE_GT(
      size, 0u, phi::errors::PreconditionNotMet("size should large than 0"));

  out->Resize(common::make_ddim({size}));
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  int64_t block = std::min(size, static_cast<int64_t>(256));
  if (block == 0) {
    return;
  }
  int64_t grid = (size + block - 1) / block;
  Range<MPType, T>
      <<<grid, block, 0, stream>>>(start_value, step_value, size, out_data);
}

template <typename T, typename Context>
void ArangeNullaryKernel(const Context& dev_ctx,
                         const T start_value,
                         const T end_value,
                         const T step_value,
                         DenseTensor* out) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType start_value_mpt = static_cast<MPType>(start_value);
  MPType end_value_mpt = static_cast<MPType>(end_value);
  MPType step_value_mpt = static_cast<MPType>(step_value);
  int64_t size = 0;
  phi::funcs::GetSize(start_value_mpt, end_value_mpt, step_value_mpt, &size);
  PADDLE_ENFORCE_GT(
      size, 0u, phi::errors::PreconditionNotMet("size should large than 0"));

  out->Resize(common::make_ddim({size}));
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  int64_t block = std::min(size, static_cast<int64_t>(256));
  if (block == 0) {
    return;
  }
  int64_t grid = (size + block - 1) / block;
  Range<MPType, T><<<grid, block, 0, stream>>>(
      start_value_mpt, step_value_mpt, size, out_data);
}

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const Scalar& start,
                  const Scalar& end,
                  const Scalar& step,
                  DenseTensor* out) {
  T start_value = start.to<T>();
  T end_value = end.to<T>();
  T step_value = step.to<T>();
  ArangeNullaryKernel<T, Context>(
      dev_ctx, start_value, end_value, step_value, out);
}

template decltype(ArangeNullaryKernel<int64_t, phi::GPUContext>)
    ArangeNullaryKernel;
template decltype(ArangeNullaryKernel<int, phi::GPUContext>)
    ArangeNullaryKernel;
}  // namespace phi

PD_REGISTER_KERNEL(arange_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArangeTensorKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(arange,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArangeKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
