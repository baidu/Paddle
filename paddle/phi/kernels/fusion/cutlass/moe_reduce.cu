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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/numeric_conversion.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_cutlass_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"
#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_helper.h"

#pragma GCC diagnostic pop

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeReduceKernel(const Context& ctx,
                     const DenseTensor& ffn_out,
                     const DenseTensor& expert_scales_float,
                     const DenseTensor& permute_indices_per_token,
                     const DenseTensor& top_k_indices,
                     const paddle::optional<DenseTensor>& ffn2_bias,
                     const bool norm_topk_prob,
                     const float routed_scaling_factor,
                     DenseTensor* output) {
  const int topk = top_k_indices.dims()[1];
  const int num_rows = ffn_out.dims()[0] / topk;
  const int hidden_size = ffn_out.dims()[1];
  output->Resize({num_rows, hidden_size});

  finalize_moe_routing_kernelLauncher(
      ffn_out.data<T>(),
      ctx.template Alloc<T>(output),
      ffn2_bias ? ffn2_bias->data<T>() : nullptr,
      expert_scales_float.data<float>(),
      permute_indices_per_token.data<int32_t>(),
      top_k_indices.data<int>(),
      num_rows,
      hidden_size,
      topk,
      static_cast<int>(1),
      norm_topk_prob,
      routed_scaling_factor,
      ctx.stream());
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(moe_reduce,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeReduceKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(moe_reduce,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeReduceKernel,
                   phi::dtype::float16) {}
#endif
