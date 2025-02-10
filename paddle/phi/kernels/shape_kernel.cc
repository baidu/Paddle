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

#include "paddle/phi/kernels/shape_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ShapeKernel(const Context& ctx,
                 const DenseTensor& input,
                 DenseTensor* out) {
  auto& in_dims = input.dims();
  out->Resize({in_dims.size()});
  auto out_data = ctx.template HostAlloc<int32_t>(out);
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = in_dims[i];
  }
}

template <typename T, typename Context>
void Shape64Kernel(const Context& ctx,
                   const DenseTensor& input,
                   DenseTensor* out) {
  auto& in_dims = input.dims();
  out->Resize({in_dims.size()});
  auto out_data = ctx.template HostAlloc<int64_t>(out);
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = in_dims[i];
  }
}

template <typename T, typename Context>
void GenerateShapeKernel(const Context& ctx,
                         const std::vector<const DenseTensor*>& x,
                         const std::vector<int>& tensor_idx,
                         const std::vector<int>& dim_idx,
                         DenseTensor* out) {
  // only need shape
  ctx.template HostAlloc<int64_t>(out);
}

}  // namespace phi

PD_REGISTER_KERNEL(shape,
                   CPU,
                   ALL_LAYOUT,
                   phi::ShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(shape,
                   GPU,
                   ALL_LAYOUT,
                   phi::ShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(shape,
                   XPU,
                   ALL_LAYOUT,
                   phi::ShapeKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(shape,
                   Custom,
                   ALL_LAYOUT,
                   phi::ShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
#endif

PD_REGISTER_KERNEL(shape64,
                   CPU,
                   ALL_LAYOUT,
                   phi::Shape64Kernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

PD_REGISTER_KERNEL(generate_shape,
                   CPU,
                   ALL_LAYOUT,
                   phi::GenerateShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(shape64,
                   GPU,
                   ALL_LAYOUT,
                   phi::Shape64Kernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

PD_REGISTER_KERNEL(generate_shape,
                   GPU,
                   ALL_LAYOUT,
                   phi::GenerateShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(shape64,
                   XPU,
                   ALL_LAYOUT,
                   phi::Shape64Kernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(shape64,
                   Custom,
                   ALL_LAYOUT,
                   phi::Shape64Kernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif
