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

#include "paddle/phi/backends/gpu/gpu_info.h"

#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_helper.h"

#pragma GCC diagnostic pop

namespace phi {

namespace fusion {

template <typename T, typename Context>
void MoeDispatchKernel(const Context& ctx,
                       const DenseTensor& X,
                       const DenseTensor& gating_output,
                       const int moe_topk,
                       const bool group_moe,
                       const bool topk_only_mode,
                       DenseTensor* permute_input,
                       DenseTensor* token_nums_per_expert,
                       DenseTensor* permute_indices_per_token,
                       DenseTensor* expert_scales_float,
                       DenseTensor* top_k_indices) {
  int token_rows = 0;
  auto input_dims = X.dims();
  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = X.dims()[input_dims.size() - 1];
  auto gating_dims = gating_output.dims();
  const int expert_num = gating_dims[gating_dims.size() - 1];

  if (group_moe) {
    // Check if expert_num is divisible by moe_topk, else throw an error
    PADDLE_ENFORCE_EQ(expert_num % moe_topk,
                      0,
                      common::errors::InvalidArgument(
                          "The number of experts (expert_num) "
                          "must be divisible by moe_topk. "
                          "Got expert_num = %d and moe_topk = %d.",
                          expert_num,
                          moe_topk));
  }

  // correspond to the weighted coefficients of the results from each expert.
  expert_scales_float->Resize({num_rows, moe_topk});

  DenseTensor finished_tensor = Empty<bool>(ctx, {num_rows});
  bool* finished = finished_tensor.data<bool>();
  // set false
  funcs::SetConstant<GPUContext, bool> zero;
  zero(ctx, &finished_tensor, false);

  const int num_moe_inputs = AlignTo16(num_rows * moe_topk);
  const int bytes = num_moe_inputs * sizeof(int);

  CubKeyValueSorter sorter_;
  sorter_.update_num_experts(expert_num);

  const int sorter_ws_size_bytes =
      AlignTo16(sorter_.getWorkspaceSize(moe_topk * num_rows));
  const int sort_tmp_in_out_size = num_moe_inputs * 2 * sizeof(int);

  DenseTensor ws_ptr_tensor =
      Empty<int8_t>(ctx, {bytes + sorter_ws_size_bytes + sort_tmp_in_out_size});

  int8_t* ws_ptr = ws_ptr_tensor.data<int8_t>();
  int* source_rows_ = reinterpret_cast<int*>(ws_ptr);
  int8_t* sorter_ws_ptr = reinterpret_cast<int8_t*>(ws_ptr + bytes);
  int* permuted_experts_ =
      reinterpret_cast<int*>(sorter_ws_ptr + sorter_ws_size_bytes);
  int* permuted_rows_ = permuted_experts_ + num_moe_inputs;

  top_k_indices->Resize({num_rows, moe_topk});
  int* expert_for_source_row = ctx.template Alloc<int>(top_k_indices);

  float* softmax_max_prob = nullptr;
  if (group_moe) {
    DenseTensor softmax_max_prob_tensor =
        Empty<float>(ctx, {num_rows, moe_topk});
    softmax_max_prob = softmax_max_prob_tensor.data<float>();
    funcs::SetConstant<GPUContext, float> zero_float;
    zero_float(ctx, &softmax_max_prob_tensor, false);
  }

  float* softmax_out_;

  const bool is_pow_2 =
      (expert_num != 0) && ((expert_num & (expert_num - 1)) == 0);

  DenseTensor softmax_buffer;

  if (!is_pow_2 || expert_num > 256 || group_moe) {
    softmax_buffer = Empty<float>(ctx, {num_rows * expert_num});
    softmax_out_ = softmax_buffer.data<float>();
  } else {
    softmax_out_ = nullptr;
  }

  VLOG(4) << "[MoE Info] "
          << "num_rows: " << num_rows << ", "
          << "hidden_size: " << hidden_size << ", "
          << "num_experts: " << expert_num << ", "
          << "k: " << moe_topk << ", "
          << "group_moe: " << std::boolalpha << group_moe;

  topk_gating_softmax_kernelLauncher<float>(
      gating_output.data<float>(),
      finished,
      ctx.template Alloc<float>(expert_scales_float),
      softmax_out_,
      expert_for_source_row,
      source_rows_,
      softmax_max_prob,
      num_rows,
      expert_num,
      moe_topk,
      group_moe,
      ctx.stream(),
      topk_only_mode);

  sorter_.run(reinterpret_cast<void*>(sorter_ws_ptr),
              sorter_ws_size_bytes,
              expert_for_source_row,
              permuted_experts_,
              source_rows_,
              permuted_rows_,
              moe_topk * num_rows,
              false,
              ctx.stream());

  permute_input->Resize({moe_topk * num_rows, hidden_size});
  permute_indices_per_token->Resize({moe_topk, num_rows});

  initialize_moe_routing_kernelLauncher(
      X.data<T>(),
      ctx.template Alloc<T>(permute_input),
      permuted_rows_,
      ctx.template Alloc<int32_t>(permute_indices_per_token),
      num_rows,
      num_rows,
      hidden_size,
      moe_topk,
      ctx.stream());

  token_nums_per_expert->Resize({expert_num});

  compute_total_rows_before_expert<T>(
      permuted_experts_,
      X.data<T>(),
      moe_topk * num_rows,
      expert_num,
      ctx.template Alloc<int64_t>(token_nums_per_expert),
      ctx.stream());
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_CUDA_BF16
PD_REGISTER_KERNEL(moe_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(moe_dispatch,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MoeDispatchKernel,
                   phi::dtype::float16) {}
#endif
