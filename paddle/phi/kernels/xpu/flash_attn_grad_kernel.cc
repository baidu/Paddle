// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/flash_attn_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_XPU_XRE5
#include "paddle/phi/kernels/xpu/flash_attn_utils.h"
#include "xfa/flash_api.h"
#endif
namespace phi {
#ifdef PADDLE_WITH_XPU_XRE5
template <typename T, typename Context>
void FlashAttnGradKernelBase(const Context& ctx,
                             const DenseTensor& q,
                             const DenseTensor& k,
                             const DenseTensor& v,
                             const api::VectorParam<int>& lod_seqlen_q,
                             const api::VectorParam<int>& lod_seqlen_k,
                             const DenseTensor& out,
                             const DenseTensor& softmax_lse,
                             const DenseTensor& seed_offset,
                             const paddle::optional<DenseTensor>& attn_mask,
                             const DenseTensor& dout,
                             const int batch_size,
                             const int64_t max_seqlen_q,
                             const int64_t max_seqlen_k,
                             const int num_heads,
                             const int num_heads_k,
                             const int head_size,
                             const int head_size_v,
                             float scale,
                             float dropout,
                             bool causal,
                             DenseTensor* dq,
                             DenseTensor* dk,
                             DenseTensor* dv) {
  xpu::ctx_guard RAII_GUARD(ctx.x_context());

  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
  const XPUType* q_data = reinterpret_cast<const XPUType*>(q.data<T>());
  const XPUType* k_data = reinterpret_cast<const XPUType*>(k.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(v.data<T>());
  const XPUType* out_data = reinterpret_cast<const XPUType*>(out.data<T>());
  const float* softmax_lse_data = softmax_lse.data<float>();
  const XPUType* dout_data = reinterpret_cast<const XPUType*>(dout.data<T>());

  float real_scale = scale == 0.0f ? 1.0f / std::sqrt(head_size) : scale;

  const float* bias_data = nullptr;
  int64_t fa_layout = AttnQKVLayout_t::ATTN_BLHD;
  if (attn_mask.get_ptr() != nullptr) {
    const auto& mask_dims = attn_mask->dims();
    if (mask_dims.size() == 3 || (mask_dims[1] == 1 && mask_dims.size() == 4)) {
      fa_layout |= AttnQKVLayout_t::BIAS_BLL;
    } else {
      PADDLE_ENFORCE_EQ(mask_dims.size(),
                        4,
                        common::errors::InvalidArgument(
                            "flash_attn_bwd requires mask's shape "
                            "like [b,l,l] or [b, h, l, l]"));
    }
    if (attn_mask->dtype() == phi::DataType::FLOAT32) {
      bias_data = attn_mask->data<float>();
    } else if (attn_mask->dtype() == phi::DataType::FLOAT16 ||
               attn_mask->dtype() == phi::DataType::BFLOAT16) {
      float* bias_tmp = RAII_GUARD.alloc_l3_or_gm<float>(attn_mask->numel());
      int r = xpu::cast<XPUType, float>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(attn_mask->data<T>()),
          bias_tmp,
          attn_mask->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      bias_data = bias_tmp;
    } else {
      errors::Unimplemented(
          "Unsupported dtype for attention_mask in xpu flash attention, only "
          "float32, float16 and "
          "bfloat16 are supported.");
    }
  }
  // output
  XPUType* dq_data = reinterpret_cast<XPUType*>(dq->data<T>());
  XPUType* dk_data = reinterpret_cast<XPUType*>(dk->data<T>());
  XPUType* dv_data = reinterpret_cast<XPUType*>(dv->data<T>());

  // get seed offset
  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  int fa_tgemm = get_flash_attn_tgemm<XPUType>();
  auto flash_attention_grad_kernel =
      baidu::xpu::xfa::mha_varlen_bwd<XPUType, float, tfloat32, int>;
  if (fa_tgemm == XPU_FA_TGEMM::FA_FLOAT) {
    flash_attention_grad_kernel =
        baidu::xpu::xfa::mha_varlen_bwd<XPUType, float, float, int>;
  } else if (fa_tgemm == XPU_FA_TGEMM::FA_FLOAT16) {
    flash_attention_grad_kernel =
        baidu::xpu::xfa::mha_varlen_bwd<XPUType, float, XPUTypeFP16, int>;
  }
  // template<typename T, typename TACCUM, typename TGEMM, typename TID = int>
  // int mha_varlen_bwd(xdnn::Context* ctx, const T* dout, const T* q, const T*
  // k, const T* v, const T* out, const TACCUM* softmax_lse, T* dq, T* dk, T*
  // dv, const xdnn::VectorParam<TID>& lod_seqlens_q, const
  // xdnn::VectorParam<TID>& lod_seqlens_k, int64_t max_seqlen_q, int64_t
  // max_seqlen_k, int64_t head_num, int64_t head_num_k, int64_t head_dim, const
  // float softmax_scale = 0.0f, const float p_dropout = 0.0f, int seed =
  // 0x45678901, const bool is_causal = true, const TACCUM* attn_mask = nullptr,
  // const TACCUM* bias = nullptr, const float* q_maxptr = nullptr, const float*
  // k_maxptr = nullptr, const float* v_maxptr = nullptr, const float* o_maxptr
  // = nullptr, float* dq_maxptr = nullptr, float* dk_maxptr = nullptr, float*
  // dv_maxptr = nullptr, const float* do_maxptr = nullptr, const bool
  // is_qkv_fusion = false, const bool is_dqkv_fusion = false, const int64_t
  // qkv_layout = AttnQKVLayout_t::ATTN_BLHD, const float* alibi_slopes =
  // nullptr, const std::vector<int64_t>& alibi_slopes_shape = {}, int
  // window_size_left = -1, int window_size_right = -1, int64_t v_head_dim =
  // -1);
  int r = flash_attention_grad_kernel(
      ctx.x_context(),
      dout_data,                                  // dout
      q_data,                                     // q
      k_data,                                     // k
      v_data,                                     // v
      out_data,                                   // out
      softmax_lse_data,                           // softmax_lse
      dq_data,                                    // dq
      dk_data,                                    // dk
      dv_data,                                    // dv
      lod_seqlen_q,                               // lod_seqlens_q
      lod_seqlen_k,                               // lod_seqlens_k
      max_seqlen_q,                               // max_seqlen_q
      max_seqlen_k,                               // max_seqlen_k
      num_heads,                                  // head_num
      num_heads_k,                                // head_num_k
      head_size,                                  // head_dim
      real_scale,                                 // softmax_scale
      dropout,                                    // p_dropout
      static_cast<int32_t>(seed_offset_data[0]),  // seed
      causal,                                     // is_causal
      nullptr,                                    // attn_mask
      bias_data,                                  // bias
      nullptr,                                    // q_maxptr
      nullptr,                                    // k_maxptr
      nullptr,                                    // v_maxptr
      nullptr,                                    // o_maxptr
      nullptr,                                    // dq_maxptr
      nullptr,                                    // dk_maxptr
      nullptr,                                    // dv_maxptr
      nullptr,                                    // do_maxptr
      false,                                      // is_qkv_fusion
      false,                                      // is_dqkv_fusion
      fa_layout,                                  // qkv_layout
      nullptr,                                    // alibi_slopes
      {},                                         // alibi_slopes_shape
      -1,                                         // window_size_left
      -1,                                         // window_size_right
      head_size_v                                 // v_head_dim
  );
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_varlen_bwd");
}
#endif

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& v,
                                 const DenseTensor& cu_seqlens_q,
                                 const DenseTensor& cu_seqlens_k,
                                 const DenseTensor& out,
                                 const DenseTensor& softmax_lse,
                                 const DenseTensor& seed_offset,
                                 const paddle::optional<DenseTensor>& attn_mask,
                                 const DenseTensor& dout,
                                 int64_t max_seqlen_q,
                                 int64_t max_seqlen_k,
                                 float scale,
                                 float dropout,
                                 bool causal,
                                 DenseTensor* dq,
                                 DenseTensor* dk,
                                 DenseTensor* dv) {
#ifdef PADDLE_WITH_XPU_XRE5
  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);
  auto dims = q.dims();

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t head_size_v = v.dims()[2];
  const int64_t num_heads_k = k.dims()[1];

  api::VectorParam<int> qlod{cu_seqlens_q.data<int>(),
                             static_cast<int64_t>(cu_seqlens_q.numel()),
                             nullptr};
  api::VectorParam<int> kvlod{cu_seqlens_k.data<int>(),
                              static_cast<int64_t>(cu_seqlens_k.numel()),
                              nullptr};

  FlashAttnGradKernelBase<T>(ctx,
                             q,
                             k,
                             v,
                             qlod,
                             kvlod,
                             out,
                             softmax_lse,
                             seed_offset,
                             attn_mask,
                             dout,
                             batch_size,
                             max_seqlen_q,
                             max_seqlen_k,
                             num_heads,
                             num_heads_k,
                             head_size,
                             head_size_v,
                             scale,
                             dropout,
                             causal,
                             dq,
                             dk,
                             dv);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "re-compile using -DWITH_XPU_XRE5=ON to use FlashAttnGradKernel"));
#endif
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
#ifdef PADDLE_WITH_XPU_XRE5
  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t head_size_v = v.dims()[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size_v,
      common::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size_v"));

  // lod info
  std::vector<int> qlod_vec = {0};
  std::vector<int> kvlod_vec = {0};
  for (int batch_idx = 1; batch_idx <= batch_size; ++batch_idx) {
    qlod_vec.push_back(seqlen_q * batch_idx);
    kvlod_vec.push_back(seqlen_k * batch_idx);
  }
  api::VectorParam<int> qlod{
      qlod_vec.data(), static_cast<int64_t>(qlod_vec.size()), nullptr};
  api::VectorParam<int> kvlod{
      kvlod_vec.data(), static_cast<int64_t>(kvlod_vec.size()), nullptr};

  FlashAttnGradKernelBase<T>(ctx,
                             q,
                             k,
                             v,
                             qlod,
                             kvlod,
                             out,
                             softmax_lse,
                             seed_offset,
                             attn_mask,
                             dout,
                             batch_size,
                             seqlen_q,
                             seqlen_k,
                             num_heads,
                             num_heads_k,
                             head_size,
                             head_size_v,
                             0.0,
                             dropout,
                             causal,
                             dq,
                             dk,
                             dv);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "re-compile using -DWITH_XPU_XRE5=ON to use FlashAttnGradKernel"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(phi::Backend::CPU);          // cu_seqlens_q
  kernel->InputAt(4).SetBackend(phi::Backend::CPU);          // cu_seqlens_k
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::bfloat16,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
