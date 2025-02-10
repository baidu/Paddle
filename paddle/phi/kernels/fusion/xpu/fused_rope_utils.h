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

#pragma once
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"

namespace phi {
namespace fusion {
template <typename XPUType, typename XPUSCType, typename Context>
void GetSinCosByPassValue(const Context& dev_ctx,
                          const paddle::optional<DenseTensor>& sin,
                          const paddle::optional<DenseTensor>& cos,
                          const paddle::optional<DenseTensor>& position_ids,
                          XPUSCType* sin_data,
                          XPUSCType* cos_data,
                          int64_t batch_size,
                          int64_t seq_len,
                          int64_t head_dim) {
  PADDLE_ENFORCE_EQ((std::is_same<XPUType, XPUSCType>::value),
                    true,
                    common::errors::Unimplemented(
                        "The xpu get_sin_cos_by_pass_value only supports "
                        "sin/cos with the same type as inputs now."));
  auto sin_cos_dims = sin->dims();
  int64_t dims_size = sin_cos_dims.size();
  int ret = xpu::SUCCESS;
  PADDLE_ENFORCE_EQ(
      (dims_size == 2 || dims_size == 4),
      true,
      common::errors::InvalidArgument("The dims of sin and cos is expected to "
                                      "be 2 or 4, but received %d.",
                                      dims_size));
  if (dims_size == 4) {
    // sin.shape: [1, seq_len, 1, head_dim]
    PADDLE_ENFORCE_EQ((sin_cos_dims[2] == 1),
                      true,
                      common::errors::InvalidArgument(
                          "The num_heads of sin and cos must be 1."));
  }
  int sin_seq_len_dim = (dims_size) == 4 ? 1 : 0;
  if (position_ids) {
    PADDLE_ENFORCE_EQ(
        (sin_cos_dims[dims_size - 1] == head_dim &&
         sin_cos_dims[sin_seq_len_dim] >= seq_len),
        true,
        common::errors::InvalidArgument(
            "The seq_len of sin and cos must be greater than or equal to "
            "this of q. The head_dim of sin and cos must be the same as this "
            "of q."));

    auto position_ids_dims = position_ids->dims();
    PADDLE_ENFORCE_EQ(position_ids_dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The dims of position_ids is expected to "
                          "be 2, but received %d.",
                          position_ids_dims.size()));

    PADDLE_ENFORCE_EQ(
        (position_ids_dims[0] == batch_size && position_ids_dims[1] == seq_len),
        true,
        common::errors::InvalidArgument(
            "The batch_size and seq_len of position_ids must be the same as "
            "those of q."));

    ret = xpu::paddle_gather<XPUSCType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUSCType*>(sin->data()),
        position_ids->data<int64_t>(),
        sin_data,
        {seq_len, head_dim},
        batch_size * seq_len,
        0);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_gather");
    ret = xpu::paddle_gather<XPUSCType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUSCType*>(cos->data()),
        position_ids->data<int64_t>(),
        cos_data,
        {seq_len, head_dim},
        batch_size * seq_len,
        0);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_gather");
  } else {
    int sin_cos_batch_size = (dims_size) == 4 ? sin_cos_dims[0] : 1;
    ret = xpu::broadcast<XPUSCType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUSCType*>(sin->data()),
        sin_data,
        {sin_cos_batch_size, seq_len, head_dim},
        {batch_size, seq_len, head_dim});
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
    ret = xpu::broadcast<XPUSCType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUSCType*>(cos->data()),
        cos_data,
        {sin_cos_batch_size, seq_len, head_dim},
        {batch_size, seq_len, head_dim});
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
  }
}

template <typename XPUType, typename XPUSCType, typename Context>
void GetSinCosByRotaryBase(const Context& dev_ctx,
                           XPUSCType* sin_data,
                           XPUSCType* cos_data,
                           int64_t batch_size,
                           int64_t seq_len,
                           int64_t head_dim,
                           float rotary_emb_base) {
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  float* pos_seq_data = RAII_GUARD.alloc_l3_or_gm<float>(seq_len);
  PADDLE_ENFORCE_XDNN_NOT_NULL(pos_seq_data);
  int ret =
      xpu::range<float>(dev_ctx.x_context(), pos_seq_data, 0.0f, 1.0f, seq_len);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "range");
  float* freqs_half_data = RAII_GUARD.alloc_l3_or_gm<float>(head_dim / 2);
  PADDLE_ENFORCE_XDNN_NOT_NULL(freqs_half_data);
  ret = xpu::range<float>(dev_ctx.x_context(),
                          freqs_half_data,
                          0.0f,
                          2.0f / head_dim,
                          head_dim / 2);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "range");

  float* rotary_base_xpu_data = RAII_GUARD.alloc_l3_or_gm<float>(1);
  PADDLE_ENFORCE_XDNN_NOT_NULL(rotary_base_xpu_data);
  ret = xpu::constant<float>(
      dev_ctx.x_context(), rotary_base_xpu_data, 1, rotary_emb_base);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
  ret = xpu::broadcast_pow<float>(dev_ctx.x_context(),
                                  rotary_base_xpu_data,
                                  freqs_half_data,
                                  freqs_half_data,
                                  {1},
                                  {head_dim / 2});
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_pow");
  ret = xpu::reciprocal<float>(
      dev_ctx.x_context(), freqs_half_data, freqs_half_data, head_dim / 2);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reciprocal");
  float* freqs_data = RAII_GUARD.alloc_l3_or_gm<float>(head_dim);
  ret = xpu::broadcast<float>(dev_ctx.x_context(),
                              freqs_half_data,
                              freqs_data,
                              {head_dim / 2, 1},
                              {head_dim / 2, 2});
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
  int64_t rotary_len = seq_len * head_dim;
  float* indices_data = RAII_GUARD.alloc_l3_or_gm<float>(rotary_len);
  PADDLE_ENFORCE_XDNN_NOT_NULL(indices_data);

  ret = xpu::broadcast_mul<float>(dev_ctx.x_context(),
                                  pos_seq_data,
                                  freqs_data,
                                  indices_data,
                                  {seq_len, 1},
                                  {1, head_dim});
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_mul");

  float* sin_fp32_data = nullptr;
  float* cos_fp32_data = nullptr;
  XPUSCType* sin_part_data = nullptr;
  XPUSCType* cos_part_data = nullptr;
  bool need_cast = !std::is_same<XPUSCType, float>::value;
  bool need_broadcast = batch_size > 1;
  if (need_broadcast) {
    sin_part_data = RAII_GUARD.alloc_l3_or_gm<XPUSCType>(rotary_len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(sin_part_data);
    cos_part_data = RAII_GUARD.alloc_l3_or_gm<XPUSCType>(rotary_len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(cos_part_data);
  } else {
    sin_part_data = sin_data;
    cos_part_data = cos_data;
  }
  if (need_cast) {
    sin_fp32_data = RAII_GUARD.alloc_l3_or_gm<float>(rotary_len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(sin_fp32_data);
    cos_fp32_data = RAII_GUARD.alloc_l3_or_gm<float>(rotary_len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(cos_fp32_data);
  } else {
    sin_fp32_data = reinterpret_cast<float*>(sin_part_data);
    cos_fp32_data = reinterpret_cast<float*>(cos_part_data);
  }
  ret = xpu::sin<float>(
      dev_ctx.x_context(), indices_data, sin_fp32_data, rotary_len);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "sin");
  ret = xpu::cos<float>(
      dev_ctx.x_context(), indices_data, cos_fp32_data, rotary_len);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cos");

  if (need_cast) {
    ret = xpu::cast<float, XPUSCType>(
        dev_ctx.x_context(), sin_fp32_data, sin_part_data, rotary_len);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cast");
    ret = xpu::cast<float, XPUSCType>(
        dev_ctx.x_context(), cos_fp32_data, cos_part_data, rotary_len);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "cast");
  }

  if (need_broadcast) {
    ret = xpu::broadcast<XPUSCType>(dev_ctx.x_context(),
                                    sin_part_data,
                                    sin_data,
                                    {1, seq_len, head_dim},
                                    {batch_size, seq_len, head_dim});
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
    ret = xpu::broadcast<XPUSCType>(dev_ctx.x_context(),
                                    cos_part_data,
                                    cos_data,
                                    {1, seq_len, head_dim},
                                    {batch_size, seq_len, head_dim});
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
  }
}

template <typename XPUType, typename XPUSCType, typename Context>
void XPUGetSinCosData(const Context& dev_ctx,
                      const paddle::optional<DenseTensor>& sin,
                      const paddle::optional<DenseTensor>& cos,
                      const paddle::optional<DenseTensor>& position_ids,
                      XPUSCType* sin_data,
                      XPUSCType* cos_data,
                      int64_t batch_size,
                      int64_t seq_len,
                      int64_t head_dim,
                      float rotary_emb_base) {
  if (sin && cos) {
    GetSinCosByPassValue<XPUType, XPUSCType, Context>(dev_ctx,
                                                      sin,
                                                      cos,
                                                      position_ids,
                                                      sin_data,
                                                      cos_data,
                                                      batch_size,
                                                      seq_len,
                                                      head_dim);
  } else {
    GetSinCosByRotaryBase<XPUType, XPUSCType, Context>(dev_ctx,
                                                       sin_data,
                                                       cos_data,
                                                       batch_size,
                                                       seq_len,
                                                       head_dim,
                                                       rotary_emb_base);
  }
}

template <typename XPUType, typename XPUSCType, typename Context>
void XPUFusedRotaryEveryTwo(const Context& dev_ctx,
                            const DenseTensor& in_q,
                            const paddle::optional<DenseTensor>& in_k,
                            const paddle::optional<DenseTensor>& in_v,
                            const XPUSCType* sin_data,
                            const XPUSCType* cos_data,
                            int64_t batch_size,
                            int64_t seq_len,
                            int64_t num_heads,
                            int64_t head_dim,
                            bool time_major,
                            bool is_bwd,
                            DenseTensor* out_q,
                            DenseTensor* out_k,
                            DenseTensor* out_v) {
  auto everytwo_func = &xpu::rotary_embedding_everytwo<XPUType, XPUSCType>;
  const char* single_func_name = "rotary_embedding_everytwo_unary_freqs";
  const char* fusion_func_name = "rotary_embedding_everytwo_binary_freqs";
  if (is_bwd) {
    everytwo_func = &xpu::rotary_embedding_everytwo_grad<XPUType, XPUSCType>;
    single_func_name = "rotary_embedding_everytwo_unary_freqs_grad";
    fusion_func_name = "rotary_embedding_everytwo_binary_freqs_grad";
  }
  if (!in_k) {
    int ret = everytwo_func(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(in_q.data()),
                            nullptr,
                            sin_data,
                            cos_data,
                            reinterpret_cast<XPUType*>(out_q->data()),
                            nullptr,
                            {batch_size, seq_len, num_heads, head_dim},
                            {batch_size, seq_len, 1, head_dim},
                            -1,
                            10000.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, single_func_name);
  } else {
    int64_t num_heads_k = in_k->dims()[2];
    int ret = everytwo_func(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(in_q.data()),
                            reinterpret_cast<const XPUType*>(in_k->data()),
                            sin_data,
                            cos_data,
                            reinterpret_cast<XPUType*>(out_q->data()),
                            reinterpret_cast<XPUType*>(out_k->data()),
                            {batch_size, seq_len, num_heads, head_dim},
                            {batch_size, seq_len, 1, head_dim},
                            num_heads_k,
                            10000.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, fusion_func_name);
  }
  if (in_v) {
    int64_t num_heads_v = in_v->dims()[2];
    int ret = everytwo_func(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(in_v->data()),
                            nullptr,
                            sin_data,
                            cos_data,
                            reinterpret_cast<XPUType*>(out_v->data()),
                            nullptr,
                            {batch_size, seq_len, num_heads_v, head_dim},
                            {batch_size, seq_len, 1, head_dim},
                            -1,
                            10000.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, single_func_name);
  }
}

template <typename XPUType, typename XPUSCType, typename Context>
void XPUFusedRotaryHalf(const Context& dev_ctx,
                        const DenseTensor& in_q,
                        const paddle::optional<DenseTensor>& in_k,
                        const paddle::optional<DenseTensor>& in_v,
                        const XPUSCType* sin_data,
                        const XPUSCType* cos_data,
                        int64_t batch_size,
                        int64_t seq_len,
                        int64_t num_heads,
                        int64_t head_dim,
                        bool time_major,
                        bool is_bwd,
                        DenseTensor* out_q,
                        DenseTensor* out_k,
                        DenseTensor* out_v) {
  auto half_func = &xpu::rotary_embedding_half<XPUType, XPUSCType>;
  const char* single_func_name = "rotary_embedding_half_unary_freqs";
  const char* fusion_func_name = "xpu::rotary_embedding_half_binary_freqs";
  if (is_bwd) {
    half_func = &xpu::rotary_embedding_half_grad<XPUType, XPUSCType>;
    single_func_name = "rotary_embedding_half_unary_freqs_grad";
    fusion_func_name = "xpu::rotary_embedding_half_binary_freqs_grad";
  }

  if (!in_k) {
    int ret = half_func(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(in_q.data()),
                        nullptr,
                        reinterpret_cast<const XPUSCType*>(sin_data),
                        reinterpret_cast<const XPUSCType*>(cos_data),
                        reinterpret_cast<XPUType*>(out_q->data()),
                        nullptr,
                        {batch_size, seq_len, num_heads, head_dim},
                        {batch_size, seq_len, 1, head_dim},
                        {},
                        "BLHD",
                        -1,
                        10000.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, single_func_name);
  } else {
    int64_t num_heads_k = in_k->dims()[2];
    int ret = half_func(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(in_q.data()),
                        reinterpret_cast<const XPUType*>(in_k->data()),
                        reinterpret_cast<const XPUSCType*>(sin_data),
                        reinterpret_cast<const XPUSCType*>(cos_data),
                        reinterpret_cast<XPUType*>(out_q->data()),
                        reinterpret_cast<XPUType*>(out_k->data()),
                        {batch_size, seq_len, num_heads, head_dim},
                        {batch_size, seq_len, 1, head_dim},
                        {},
                        "BLHD",
                        num_heads_k,
                        10000.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, fusion_func_name);
  }

  if (in_v) {
    int64_t num_heads_v = in_v->dims()[2];
    int ret = half_func(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(in_v->data()),
                        nullptr,
                        reinterpret_cast<const XPUSCType*>(sin_data),
                        reinterpret_cast<const XPUSCType*>(cos_data),
                        reinterpret_cast<XPUType*>(out_v->data()),
                        nullptr,
                        {batch_size, seq_len, num_heads_v, head_dim},
                        {batch_size, seq_len, 1, head_dim},
                        {},
                        "BLHD",
                        -1,
                        10000.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, single_func_name);
  }
}
template <typename T, typename SCT, typename Context>
void XPUFusedRopeImpl(const Context& dev_ctx,
                      const DenseTensor& q,
                      const paddle::optional<DenseTensor>& k,
                      const paddle::optional<DenseTensor>& v,
                      const paddle::optional<DenseTensor>& sin,
                      const paddle::optional<DenseTensor>& cos,
                      const paddle::optional<DenseTensor>& position_ids,
                      bool use_neox_rotary_style,
                      bool time_major,
                      bool is_bwd,
                      float rotary_emb_base,
                      DenseTensor* out_q,
                      DenseTensor* out_k,
                      DenseTensor* out_v) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUSCType = typename XPUTypeTrait<SCT>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (q.numel() <= 0) {
    return;
  }
  int64_t batch_size = q.dims()[0];
  int64_t seq_len = q.dims()[1];
  int64_t num_heads = q.dims()[2];
  int64_t head_dim = q.dims()[3];
  PADDLE_ENFORCE_EQ(head_dim % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 2."));
  PADDLE_ENFORCE_EQ(
      time_major,
      false,
      common::errors::InvalidArgument("time_major is not supported in xpu"));

  int64_t sin_cos_len = batch_size * seq_len * head_dim;
  auto* sin_data = RAII_GUARD.alloc_l3_or_gm<XPUSCType>(sin_cos_len);
  PADDLE_ENFORCE_XDNN_NOT_NULL(sin_data);
  auto* cos_data = RAII_GUARD.alloc_l3_or_gm<XPUSCType>(sin_cos_len);
  PADDLE_ENFORCE_XDNN_NOT_NULL(cos_data);
  XPUGetSinCosData<XPUType, XPUSCType, Context>(dev_ctx,
                                                sin,
                                                cos,
                                                position_ids,
                                                sin_data,
                                                cos_data,
                                                batch_size,
                                                seq_len,
                                                head_dim,
                                                rotary_emb_base);
  if (use_neox_rotary_style) {
    XPUFusedRotaryEveryTwo<XPUType, XPUSCType, Context>(dev_ctx,
                                                        q,
                                                        k,
                                                        v,
                                                        sin_data,
                                                        cos_data,
                                                        batch_size,
                                                        seq_len,
                                                        num_heads,
                                                        head_dim,
                                                        time_major,
                                                        is_bwd,
                                                        out_q,
                                                        out_k,
                                                        out_v);
  } else {
    XPUFusedRotaryHalf<XPUType, XPUSCType, Context>(dev_ctx,
                                                    q,
                                                    k,
                                                    v,
                                                    sin_data,
                                                    cos_data,
                                                    batch_size,
                                                    seq_len,
                                                    num_heads,
                                                    head_dim,
                                                    time_major,
                                                    is_bwd,
                                                    out_q,
                                                    out_k,
                                                    out_v);
  }
}
}  // namespace fusion
}  // namespace phi
