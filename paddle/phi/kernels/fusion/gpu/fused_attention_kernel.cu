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

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/fusion/fused_attention_kernel.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"

namespace phi {

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const phi::GPUContext &dev_ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup *pg = map->get(ring_id);
    auto pg_nccl = static_cast<paddle::distributed::ProcessGroupNCCL *>(pg);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = paddle::distributed::ReduceOp::SUM;
    auto task = pg_nccl->AllReduce(&tensor, tensor, opts, true, true);
    task->Wait();
  } else {
    auto dtype = paddle::platform::ToNCCLDataType(
        paddle::framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    const void *sendbuff = tensor.data<T>();
    auto place = dev_ctx.GetPlace();
    void *recvbuff =
        dev_ctx.template Alloc<T>(&tensor, tensor.numel() * sizeof(T));
    auto comm =
        paddle::platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto stream = dev_ctx.stream();
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
  }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T, typename Context>
void FusedAttentionKernel(const Context &dev_ctx,
                          const DenseTensor *x,
                          const DenseTensor *ln_scale,
                          const DenseTensor *ln_bias,
                          const DenseTensor *qkv_weight,
                          const DenseTensor *qkv_bias,
                          const DenseTensor *cache_kv,
                          const DenseTensor *src_mask,
                          const DenseTensor *out_linear_weight,
                          const DenseTensor *out_linear_bias,
                          const DenseTensor *ln_scale_2,
                          const DenseTensor *ln_bias_2,
                          int num_heads,
                          bool transpose_qkv_wb,
                          bool pre_layer_norm,
                          float epsilon,
                          float attn_dropout_rate,
                          bool is_test,
                          bool attn_dropout_fix_seed,
                          int attn_dropout_seed,
                          std::string attn_dropout_implementation,
                          float dropout_rate,
                          bool dropout_fix_seed,
                          int dropout_seed,
                          std::string dropout_implementation,
                          float ln_epsilon,
                          bool add_residual,
                          int ring_id,
                          DenseTensor *ln_mean,
                          DenseTensor *ln_var,
                          DenseTensor *ln_out,
                          DenseTensor *qkv_out,
                          DenseTensor *qkv_bias_out,
                          DenseTensor *transpose_out_2,
                          DenseTensor *qk_out,
                          DenseTensor *qktv_out,
                          DenseTensor *softmax_out,
                          DenseTensor *attn_dropout_mask_out,
                          DenseTensor *attn_dropout_out,
                          DenseTensor *src_mask_out,
                          DenseTensor *fmha_out,
                          DenseTensor *out_linear_out,
                          DenseTensor *dropout_mask_out,
                          DenseTensor *ln_mean_2,
                          DenseTensor *ln_var_2,
                          DenseTensor *bias_dropout_residual_out,
                          DenseTensor *cache_kv_out,
                          DenseTensor *out) {
  using U = phi::funcs::LayerNormParamType<T>;

  // x: qkv's input [batch_size, seq_len, dim_embed]
  // if transpose_qkv_wb is False
  // y: qkv's weight: [3, num_head, dim_head, dim_embed]
  // if transpose_qkv_wb is True
  // y: qkv's weight: [dim_embed, 3 * dim_embed]

  const bool has_attn_dropout = (attn_dropout_rate != 0.0f);

  phi::fusion::DropoutParam dropout_param2(dropout_rate,
                                           dropout_implementation,
                                           is_test,
                                           dropout_fix_seed,
                                           nullptr,
                                           dropout_seed);
  const bool has_dropout = (dropout_param2.dropout_prob != 0.0f);

  auto &dropout_implementation_1 = attn_dropout_implementation;
  bool is_upscale_in_train_1 = (dropout_implementation_1 == "upscale_in_train");
  phi::DenseTensor *seed_1 = nullptr;

  // get data ptr for qkv part.
  const auto input_x_dims = x->dims();
  const auto qkv_w_dims = qkv_weight->dims();

  auto *x_data = x->data<T>();
  auto *qkv_weight_data = qkv_weight->data<T>();
  auto *qkv_bias_data = (qkv_bias == nullptr) ? nullptr : qkv_bias->data<T>();
  auto *qkv_out_data =
      dev_ctx.template Alloc<T>(qkv_out, qkv_out->numel() * sizeof(T));
  auto *qkv_bias_out_data =
      (qkv_bias == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(qkv_bias_out,
                                      qkv_bias_out->numel() * sizeof(T));

  // get data ptr for FMHA.
  auto *transpose_out_2_data = dev_ctx.template Alloc<T>(
      transpose_out_2, transpose_out_2->numel() * sizeof(T));
  auto *cache_kv_out_data =
      (cache_kv_out == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(cache_kv_out,
                                      cache_kv_out->numel() * sizeof(T));
  auto *qk_out_data =
      dev_ctx.template Alloc<T>(qk_out, qk_out->numel() * sizeof(T));
  auto *qktv_out_data =
      dev_ctx.template Alloc<T>(qktv_out, qktv_out->numel() * sizeof(T));
  auto *src_mask_out_data =
      (src_mask == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(src_mask_out,
                                      src_mask_out->numel() * sizeof(T));
  auto *softmax_out_data =
      dev_ctx.template Alloc<T>(softmax_out, softmax_out->numel() * sizeof(T));
  auto *attn_dropout_mask_out_data =
      has_attn_dropout ? dev_ctx.template Alloc<uint8_t>(
                             attn_dropout_mask_out,
                             attn_dropout_mask_out->numel() * sizeof(uint8_t))
                       : nullptr;
  auto *attn_dropout_out_data =
      has_attn_dropout
          ? dev_ctx.template Alloc<T>(attn_dropout_out,
                                      attn_dropout_out->numel() * sizeof(T))
          : nullptr;
  auto *fmha_out_data =
      dev_ctx.template Alloc<T>(fmha_out, fmha_out->numel() * sizeof(T));

  // get data ptr for out_linear.
  auto *out_linear_weight_data = out_linear_weight->data<T>();
  auto *out_linear_bias_data =
      (out_linear_bias == nullptr) ? nullptr : out_linear_bias->data<T>();
  auto *out_linear_out_data = dev_ctx.template Alloc<T>(
      out_linear_out, out_linear_out->numel() * sizeof(T));

  // get data ptr for bias+dropout+residual+layernorm
  auto *dropout_mask_out_data =
      has_dropout
          ? dev_ctx.template Alloc<uint8_t>(
                dropout_mask_out, dropout_mask_out->numel() * sizeof(uint8_t))
          : nullptr;
  auto *final_out_data =
      dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

  int batch_size = input_x_dims[0];
  int max_seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];

  int num_head;
  int dim_head;
  int nranks = 1;
  // get num_head and dim_head in two different ways
  if (!transpose_qkv_wb) {
    num_head = qkv_w_dims[1];
    dim_head = qkv_w_dims[2];
  } else {
    nranks = (qkv_w_dims[0] * 3) / qkv_w_dims[1];
    num_head = num_heads;
    dim_head = dim_embed / (num_head * nranks);
  }

  int bsz_seq = batch_size * max_seq_len;
  int hidden_size = num_head * dim_head;
  int output_size = 3 * hidden_size;
  int input_size = dim_embed;

  auto layer_norm_compute =
      phi::fusion::AttnLayerNorm<T>(dev_ctx, epsilon, bsz_seq, dim_embed);

  bool compute_bias = true;
  if (qkv_bias == nullptr) {
    compute_bias = false;
  }
  // (transA, transB, compute_bias) = (false, true, true)
  bool transB = transpose_qkv_wb ? false : true;
  auto qkv_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, transB, bsz_seq, output_size, input_size, compute_bias);

  phi::fusion::AttnDropoutParam attn_dropout_param(is_test,
                                                   dropout_implementation_1,
                                                   attn_dropout_rate,
                                                   is_upscale_in_train_1,
                                                   attn_dropout_fix_seed,
                                                   attn_dropout_seed,
                                                   seed_1);
  auto fmha_ref_compute = phi::fusion::FMHARef<T>(
      dev_ctx, batch_size, max_seq_len, num_head, dim_head, attn_dropout_param);

  output_size = hidden_size;
  // (transA, transB, compute_bias) = (false, false, false)
  // NOTE(Yuang Liu): For general input size == output size, change the
  // position won't have effects. For mp, the output size is mp_head * dkey
  // which is actually the input size. While the input size is hidden size,
  // which is actually the output size. So for out linear, switch the
  // input size and output size.
  auto out_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, false, false, bsz_seq, input_size, output_size, false);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper(
          dev_ctx, bsz_seq, dim_embed, dropout_param2, ln_epsilon);

  if (pre_layer_norm) {
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
    auto *ln_mean_data =
        dev_ctx.template Alloc<U>(ln_mean, ln_mean->numel() * sizeof(U));
    auto *ln_var_data =
        dev_ctx.template Alloc<U>(ln_var, ln_var->numel() * sizeof(U));
    auto *ln_out_data =
        dev_ctx.template Alloc<T>(ln_out, ln_out->numel() * sizeof(T));

    layer_norm_compute.ComputeForward(x_data,
                                      ln_scale_data,
                                      ln_bias_data,
                                      ln_out_data,
                                      ln_mean_data,
                                      ln_var_data);
    qkv_compute.ComputeForward(
        qkv_weight, ln_out, qkv_bias, qkv_out, qkv_bias_out);
  } else {
    qkv_compute.ComputeForward(qkv_weight, x, qkv_bias, qkv_out, qkv_bias_out);
  }

  if (transpose_qkv_wb) {
    // resize the output for fmha compute
    qkv_out->Resize({batch_size, max_seq_len, 3, num_head, dim_head});
    qkv_bias_out->Resize({batch_size, max_seq_len, 3, num_head, dim_head});
  }

  if (qkv_bias == nullptr) {
    fmha_ref_compute.ComputeForward(*qkv_out,
                                    cache_kv,
                                    src_mask,
                                    transpose_out_2,
                                    cache_kv_out,
                                    qk_out,
                                    src_mask_out,
                                    softmax_out,
                                    attn_dropout_mask_out,
                                    attn_dropout_out,
                                    qktv_out,
                                    fmha_out);
  } else {
    fmha_ref_compute.ComputeForward(*qkv_bias_out,
                                    cache_kv,
                                    src_mask,
                                    transpose_out_2,
                                    cache_kv_out,
                                    qk_out,
                                    src_mask_out,
                                    softmax_out,
                                    attn_dropout_mask_out,
                                    attn_dropout_out,
                                    qktv_out,
                                    fmha_out);
  }

  if (transpose_qkv_wb) {
    // resize the output back to make the shape compatible with infer shape
    qkv_out->Resize({batch_size, max_seq_len, 3 * hidden_size});
    qkv_bias_out->Resize({batch_size, max_seq_len, 3 * hidden_size});
  }

  // fmha_out: [batch_size, seq_len, num_head, head_dim]
  // weight:   [embed_dim, embed_dim]
  // out_linear_out: [batch_size, seq_len, embed_dim]
  out_linear_compute.ComputeForward(
      out_linear_weight, fmha_out, nullptr, out_linear_out, nullptr);
  // tensor model parallel
  AllReduce<T>(*out_linear_out, ring_id, dev_ctx);

  const T *residual_ptr = add_residual ? x_data : nullptr;
  if (pre_layer_norm) {
    // output = (residual + dropout(input + bias))
    fused_dropout_layernorm_helper.ResidualDropoutBias(dev_ctx,
                                                       out_linear_out_data,
                                                       residual_ptr,
                                                       out_linear_bias_data,
                                                       final_out_data,
                                                       dropout_mask_out_data);
  } else {
    // TODO(Xreki): support post layer_norm case when add_residual is false.
    PADDLE_ENFORCE_EQ(
        add_residual,
        true,
        errors::InvalidArgument("Attribute add_residual is expected to be true "
                                "when pre_layer_norm is false."));

    const U *ln_scale_2_ptr = ln_scale_2 ? ln_scale_2->data<U>() : nullptr;
    const U *ln_bias_2_ptr = ln_bias_2 ? ln_bias_2->data<U>() : nullptr;
    T *bias_dropout_residual_out_ptr = dev_ctx.template Alloc<T>(
        bias_dropout_residual_out,
        bias_dropout_residual_out->numel() * sizeof(T));
    U *ln_mean_2_ptr =
        dev_ctx.template Alloc<U>(ln_mean_2, ln_mean_2->numel() * sizeof(U));
    U *ln_var_2_ptr =
        dev_ctx.template Alloc<U>(ln_var_2, ln_var_2->numel() * sizeof(U));
    // output = layernorm(residual + dropout(input + bias))
    fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
        dev_ctx,
        out_linear_out_data,
        residual_ptr,
        out_linear_bias_data,
        ln_scale_2_ptr,
        ln_bias_2_ptr,
        bias_dropout_residual_out_ptr,
        dropout_mask_out_data,
        final_out_data,
        ln_mean_2_ptr,
        ln_var_2_ptr);
  }
}

}  // namespace phi

// PD_REGISTER_KERNEL(trace,
//                    GPU,
//                    ALL_LAYOUT,
//                    phi::FusedAttention,
//                    float,
//                    double,
//                    int,
//                    int64_t,
//                    phi::dtype::float16,
//                    phi::dtype::complex<float>,
//                    phi::dtype::complex<double>) {}
