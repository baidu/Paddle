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

#include "paddle/phi/kernels/weight_only_mat_mul_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace phi {

template <typename T, typename Context>
void WeightOnlyMatMulKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const DenseTensor& weight_scale,
                            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();

  int k = w_dims[0];
  int n = w_dims[1];
  int m = x.numel() / k;
  auto mixed_gemm_runner =
      CutlassFpAIntBGemmRunner<typename PDDataTypeTraits<T>::DataType,
                               uint8_t>();
  int mixgemm_max_size = std::max(n, k);
  DenseTensor mixgemm_workspace;
  int64_t mixgemm_workspace_size_bytes =
      mixed_gemm_runner.getWorkspaceSize(m, mixgemm_max_size, mixgemm_max_size);

  mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
  dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
  char* mixgemm_workspace_data =
      reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
  mixed_gemm_runner.gemm(
      reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
          x.data<T>()),
      reinterpret_cast<const uint8_t*>(weight.data<int8_t>()),
      reinterpret_cast<const float*>(weight_scale.data<float>()),
      reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out->data<T>()),
      m,
      n,
      k,
      mixgemm_workspace_data,
      mixgemm_workspace_size_bytes,
      dev_ctx.stream());
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_only_mat_mul,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyMatMulKernel,
                   phi::dtype::float16) {}
