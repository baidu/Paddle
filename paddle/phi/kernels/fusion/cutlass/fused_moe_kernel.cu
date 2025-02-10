// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

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
void FusedMoeKernel(const Context& ctx,
                    const DenseTensor& X,
                    const DenseTensor& gate_weight,
                    const DenseTensor& ffn1_weight,
                    const paddle::optional<DenseTensor>& ffn1_scale,
                    const paddle::optional<DenseTensor>& ffn1_bias,
                    const DenseTensor& ffn2_weight,
                    const paddle::optional<DenseTensor>& ffn2_scale,
                    const paddle::optional<DenseTensor>& ffn2_bias,
                    const std::string& quant_method,
                    const int moe_topk,
                    const bool group_moe,
                    const bool norm_topk_prob,
                    DenseTensor* out) {
  out->Resize(X.dims());
  auto* output_data = ctx.template Alloc<T>(out);

  auto fp16_moe_gemm_runner =
      MoeGemmRunner<typename phi::PDDataTypeTraits<T>::DataType,
                    typename phi::PDDataTypeTraits<T>::DataType>();
  auto int8_moe_gemm_runner =
      MoeGemmRunner<typename phi::PDDataTypeTraits<T>::DataType, uint8_t>();
  auto int4_moe_gemm_runner =
      MoeGemmRunner<typename phi::PDDataTypeTraits<T>::DataType,
                    cutlass::uint4b_t>();

  auto moe_compute = MoeHelper<T>(ctx,
                                  quant_method,
                                  &fp16_moe_gemm_runner,
                                  &int8_moe_gemm_runner,
                                  &int4_moe_gemm_runner);

  moe_compute.ComputeFFN(&X,
                         &gate_weight,
                         &ffn1_weight,
                         ffn1_scale ? ffn1_scale.get_ptr() : nullptr,
                         ffn1_bias ? ffn1_bias.get_ptr() : nullptr,
                         &ffn2_weight,
                         ffn2_scale ? ffn2_scale.get_ptr() : nullptr,
                         ffn2_bias ? ffn2_bias.get_ptr() : nullptr,
                         nullptr,
                         moe_topk,
                         group_moe,
                         norm_topk_prob,
                         1.0,
                         "ffn",
                         out);
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(fused_moe,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMoeKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(fused_moe,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMoeKernel,
                   phi::dtype::float16) {}
#endif
