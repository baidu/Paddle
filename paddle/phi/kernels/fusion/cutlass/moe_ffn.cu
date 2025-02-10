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
void MoeFFNKernel(const Context& ctx,
                  const DenseTensor& permute_input,
                  const DenseTensor& token_nums_per_expert,
                  const DenseTensor& ffn1_weight,
                  const DenseTensor& ffn2_weight,
                  const paddle::optional<DenseTensor>& ffn1_bias,
                  const paddle::optional<DenseTensor>& ffn1_scale,
                  const paddle::optional<DenseTensor>& ffn2_scale,
                  const std::string& quant_method,
                  DenseTensor* ffn_out) {
  ffn_out->Resize(permute_input.dims());
  auto* ffn_out_data = ctx.template Alloc<T>(ffn_out);
  auto permuted_data = permute_input.data<T>();

  auto fp16_moe_gemm_runner =
      MoeGemmRunner<typename phi::PDDataTypeTraits<T>::DataType,
                    typename phi::PDDataTypeTraits<T>::DataType>();
  auto int8_moe_gemm_runner =
      MoeGemmRunner<typename phi::PDDataTypeTraits<T>::DataType, uint8_t>();
  auto int4_moe_gemm_runner =
      MoeGemmRunner<typename phi::PDDataTypeTraits<T>::DataType,
                    cutlass::uint4b_t>();

  const int64_t expanded_active_expert_rows = permute_input.dims()[0];
  const int num_experts = ffn1_weight.dims()[0];
  const int hidden_size = ffn1_weight.dims()[1];
  int inter_dim = ffn1_weight.dims()[2];

  if (quant_method == "weight_only_int4") {
    inter_dim = inter_dim * 2;
  }

  const int64_t inter_size = inter_dim;

  DenseTensor fc1_out_tensor =
      Empty<T>(ctx, {expanded_active_expert_rows, inter_size});
  T* fc1_out = fc1_out_tensor.data<T>();
  using NvType = typename phi::PDDataTypeTraits<T>::DataType;

  const T* fc1_expert_biases = ffn1_bias ? ffn1_bias->data<T>() : nullptr;

  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const uint8_t*>(ffn1_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(ffn1_scale->data<T>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        ctx.stream());
  } else if (quant_method == "weight_only_int4") {
    int4_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const cutlass::uint4b_t*>(ffn1_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(ffn1_scale->data<T>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        ctx.stream());
  } else {
    fp16_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const NvType*>(ffn1_weight.data<T>()),
        nullptr,
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        ctx.stream());
  }

  const int num_rows = expanded_active_expert_rows;

  DenseTensor act_out_tensor = Empty<T>(ctx, {num_rows, inter_size / 2});
  T* act_out = act_out_tensor.data<T>();

  const std::string act_type = "swiglu";
  auto bias_act_helper = BiasActHelper<T>(ctx, act_type, num_rows, inter_size);

  bias_act_helper.Compute(&fc1_out_tensor, nullptr, &act_out_tensor);

  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const uint8_t*>(ffn2_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(ffn2_scale->data<T>()),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        ctx.stream());

  } else if (quant_method == "weight_only_int4") {
    int4_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const cutlass::uint4b_t*>(ffn2_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(ffn2_scale->data<T>()),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        ctx.stream());
  } else {
    fp16_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const NvType*>(ffn2_weight.data<T>()),
        nullptr,
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        ctx.stream());
  }
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(moe_ffn,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeFFNKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(
    moe_ffn, GPU, ALL_LAYOUT, phi::fusion::MoeFFNKernel, phi::dtype::float16) {}
#endif
