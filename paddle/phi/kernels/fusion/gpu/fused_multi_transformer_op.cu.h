/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// This file has been adapted from FasterTransformer file:
// https://github.com/NVIDIA/FasterTransformer/blob/v4.0/fastertransformer/cuda/masked_multihead_attention.cu
// We add License in the head.

#pragma once

#include <fstream>
#include <iomanip>

#include "paddle/common/flags.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/funcs/load_store_util.h"
#include "paddle/phi/kernels/fusion/gpu/fused_bias_act_utils.h"
#include "paddle/phi/kernels/fusion/gpu/mmha_util.cu.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

COMMON_DECLARE_bool(fused_multi_transformer_op_use_mbfmha);
COMMON_DECLARE_int64(multi_block_attention_min_partition_size);

namespace phi {
namespace fusion {

namespace {  // NOLINT

using float16 = phi::dtype::float16;

#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#define MMHA_USE_FP32_ACUM_FOR_OUT
#define MMHA_USE_FP32_ACUM_FOR_FMA
// #define MMHA_USE_HMMA_FOR_REDUCTION

template <typename T>
struct Masked_multihead_attention_params {
  // output buffer, [B, 1(seq_len), num_head * dim_head]
  T *out;
  // qkv_out, [B, 1(seq_len), 3, num_head * dim_head]
  const T *qkv;
  // bias, [3, num_head, dim_head]
  T *qkv_bias;
  // [bsz, seq_len]
  const int *cum_offsets;
  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;
  int mask_length;
  // whether to broadcast num_heads(2nd) dimension for attn_mask
  // in MMHA, if false, attn_mask shape should be
  // [bsz, num_heads, 1, time_step(cache_seq_length)+1]
  bool mask_broadcast_num_heads;

  // [2, B, num_head, max_seq_len(valid cache_seq_len), dim_head]
  // k [B, num_head, dim_head/x, max_seq_len, x], that is `seq_len` first
  // v [B, num_head, max_seq_len, dim_head]
  T *cache_kv = nullptr;
  // [B, max_seq_len]
  const int *beam_cache_offset = nullptr;

  const int *sequence_lengths{nullptr};

  // The RoPE embedding, [2, B, rotary_seq_len, 1, dim_head]
  // rotary_emb_dims = 1 if pos_ids_extra is null else 2
  const float *rotary_emb;
  int rotary_bsz;
  int rotary_emb_dims;
  int rotary_seq_len = 1;

  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;
  int num_head;
  int timestep;  // cache_seq_length
  int seq_len;
  int max_seq_length;

  int gqa_group_size;
  int gqa_num_per_partitions;

  int max_num_partitions;
  int partition_size;

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;

  bool add_qkv_bias;
  bool neox_rotary_style;

  float *exp_sums;
  float *max_logits;
  T *partial_out;
};

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc>
__global__ void masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params,
    LoadFunc load_func,
    StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int bi = blockIdx.y;
  if (params.sequence_lengths && params.sequence_lengths[bi] == 0) {
    return;
  }

  typedef phi::PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

  constexpr int WARP_SIZE_TMP = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE_TMP;

  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);

  char *logits_smem_ = smem_;
  // fp32 accum for logits
  float *logits_smem = reinterpret_cast<float *>(logits_smem_);

  T *out_smem = reinterpret_cast<T *>(smem_);

  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  using Qk_vec_RoPE = typename Qk_vec_RoPE_<T, float, Dh_MAX>::Type;
  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  // beam id
  const int beami = bi % params.beam_width;
  // real batch id
  const int bbi = bi / params.beam_width;
  const int hi = blockIdx.x;
  const int kv_hi =
      hi / params.gqa_num_per_partitions;  // if no gqa, kv_hi = hi
  const int bhi = bi * params.num_head + hi;
  const int bbhi = bbi * params.beam_width * params.num_head + hi;
  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;
  const int thi = params.cum_offsets ? ti * params.num_head + hi : -1;
  const int tid = threadIdx.x;

  const int bi_seq_len_offset = bi * params.max_seq_length;

  float qk_max = -FLT_MAX;
  float qk = 0;

  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];

  // qkv [B, S=1, num_head + 2 * gqa_group_size, head_dim]
  int qkv_base_offset = bi * (params.num_head + 2 * params.gqa_group_size) * Dh;

  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE_TMP, "");
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

  // cache_k, [B, num_head, head_dim / x, max_seq_len, x]
  // x == 4/8 for FP32/FP16, 128bit, 16Byte
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  // const T *q_base = params.qkv;
  // const T *k_base = params.qkv + params.num_head * Dh;
  T *q_bias_base = nullptr;
  T *k_bias_base = nullptr;

  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + params.num_head * Dh;
  }

  if (tid < QK_VECS_PER_WARP) {
    int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    const int q_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
    const int k_bias_offset = kv_hi * Dh + tid * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset + hi * Dh);
    }

    Qk_vec k;
    zero(k);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(
          k, params.num_head * Dh + qk_offset + kv_hi * Dh);
    }

    if (params.add_qkv_bias) {
      Qk_vec q_bias;
      zero(q_bias);
      Qk_vec k_bias;
      zero(k_bias);

      q_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&q_bias_base[q_bias_offset])
              : q_bias;
      k_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&k_bias_base[k_bias_offset])
              : k_bias;

      q = add(q, q_bias);
      // TODO(wangxi): See this https://github.com/microsoft/unilm/issues/510
      //   we may not require k_bias.
      k = add(k, k_bias);
    }

    if (!params.neox_rotary_style) {
      if (params.rotary_emb_dims != 0) {
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rotary_bsz * Dh;
        Qk_vec_RoPE cos_emb, sin_emb;
        zero(cos_emb);
        zero(sin_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        apply_rotary_embedding(q, k, cos_emb, sin_emb);
      }
    } else {
      /* old rotary pos emb */
      if (params.rotary_emb_dims != 0) {
        int last_dim = Dh / params.rotary_emb_dims;
        int half_lastdim = last_dim / 2;
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rotary_bsz * Dh;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = tid / stride_all_lastdim * stride_all_lastdim +
                       (tid + stride) % (stride_all_lastdim);
        int q_right_offset = qkv_base_offset + hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_offset = qkv_base_offset + params.num_head * Dh +
                             kv_hi * Dh + right_id * QK_VEC_SIZE;
        Qk_vec q_right;
        zero(q_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(q_right, q_right_offset);
        }
        Qk_vec k_right;
        zero(k_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(k_right, k_right_offset);
        }

        Qk_vec_RoPE cos_emb;
        zero(cos_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;

        Qk_vec_RoPE sin_emb;
        zero(sin_emb);
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        float alpha = (tid % stride_all_lastdim) < stride
                          ? static_cast<float>(-1)
                          : static_cast<float>(1);
        q = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            q, q_right, cos_emb, sin_emb, alpha);
        k = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            k, k_right, cos_emb, sin_emb, alpha);
      }
    }

    *reinterpret_cast<Qk_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;

    int co = tid / QK_VECS_IN_16B;
    int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;

    int offset = bi * params.gqa_group_size * params.max_seq_length * Dh +
                 kv_hi * params.max_seq_length * Dh +
                 co * params.max_seq_length * QK_ELTS_IN_16B +
                 act_time_step * QK_ELTS_IN_16B + ci;
    if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
      *reinterpret_cast<Qk_vec *>(&params.cache_kv[offset]) = k;
    }

    qk = dot<Qk_vec, Qk_vec>(q, k);

    if (QK_VECS_PER_WARP <= WARP_SIZE_TMP) {
#pragma unroll
      for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
      }
    }
  }
  if (QK_VECS_PER_WARP > WARP_SIZE_TMP) {
    constexpr int WARPS_PER_RED =
        (QK_VECS_PER_WARP + WARP_SIZE_TMP - 1) / WARP_SIZE_TMP;
    qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
  }
  if (tid == 0) {
    qk *= params.inv_sqrt_dh;
    qk_max = qk;
    qk_smem[act_time_step] = qk;
  }
  __syncthreads();

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("=======q_out=======\n");
  //   for (int i = 0; i < Dh; ++i) printf("%f ",
  //   static_cast<float>(q_smem[i])); printf("\n");
  // }
  // __syncthreads();
#endif

  using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  static_assert(Dh_MAX % K_VEC_SIZE == 0, "");

  constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  int ko = tid / THREADS_PER_KEY;
  int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;

  static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD, "");

  K_vec q[K_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
    q[i] = *reinterpret_cast<const K_vec *>(
        &q_smem[ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
  }

  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  constexpr int K_PER_WARP = WARP_SIZE_TMP / THREADS_PER_KEY;

  T *k_cache =
      &params.cache_kv[bi * params.gqa_group_size * params.max_seq_length * Dh +
                       kv_hi * params.max_seq_length * Dh + ki];

  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + ti;
      // get k from the cache_kv, and dequant k for qk operation
      if (ti < act_time_step) {
        k[ii] =
            (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                ? *reinterpret_cast<const K_vec *>(
                      &k_cache[jj * QK_ELTS_IN_16B])
                : k_vec_zero;
      }
    }

    // NOTE(liyurui): We should multiple q with inv_sqrt_dh first, for dot(q, k)
    // may overflow with FP16 in large model.
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

    // bool is_mask = false;
    if (ti < act_time_step && tid % THREADS_PER_KEY == 0) {
      // qk_max = is_mask ? qk_max : fmaxf(qk_max, qk);
      auto mask_bhi = params.mask_broadcast_num_heads ? bi : bhi;
      if (params.attn_mask) {
        T mask = params.attn_mask[mask_bhi * params.mask_length + ti];
        qk += static_cast<float>(mask);
      }
      qk_max = fmaxf(qk_max, qk);

      qk_smem[ti] = qk;
    }
  }

#pragma unroll
  for (int mask = WARP_SIZE_TMP / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  const int warp = tid / WARP_SIZE_TMP;
  const int lane = tid % WARP_SIZE_TMP;

  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  __syncthreads();

  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("=======qk_out=======\n");
  //   for (int i = 0; i <= params.timestep; ++i) printf("%f ", qk_smem[i]);
  //   printf("qk_max=%f\n", qk_max);
  // }
  // __syncthreads();
#endif

  float sum = 0.f;
  for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
    // bool is_mask = false;
    // float logit = is_mask ? 0.f : __expf(qk_smem[ti] - qk_max);
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  // FIXME(wangxi): need add 1.e-6f?
  float inv_sum = __fdividef(1.f, sum + 1.e-6f);
  for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }
  __syncthreads();

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

  T *v_cache =
      &params.cache_kv[params.cache_batch_size * params.gqa_group_size *
                           params.max_seq_length * Dh +
                       bi * params.gqa_group_size * params.max_seq_length * Dh +
                       kv_hi * params.max_seq_length * Dh + vi];

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif

  V_vec_acum out;
  zero(out);

  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if (Dh == Dh_MAX || vi < Dh) {
    for (int ti = vo; ti < act_time_step; ti += V_PER_ITER) {
      V_vec v;
      v = *reinterpret_cast<const V_vec *>(&v_cache[ti * Dh]);
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      float logit = logits_smem[ti];
      out = fma(logit, cast_to_float(v), out);
#else
      DataType_ logit = static_cast<DataType_>(logits_smem[ti]);
      // Update the partial sums.
      out = fma(logit, v, out);
#endif
    }
  }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("======logits_out=====\n");
  //   for (int i = 0; i <= params.timestep; ++i) printf("%f ", logits_smem[i]);
  //   printf("\n");
  // }
  // __syncthreads();
#endif

  V_vec v_bias;
  zero(v_bias);
  if (vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v;
    load_func.template load<V_vec>(v,
                                   params.num_head * Dh +
                                       params.gqa_group_size * Dh +
                                       qkv_base_offset + kv_hi * Dh + vi);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[(params.num_head + params.gqa_group_size) * Dh +
                           kv_hi * Dh + vi]);
      v = add(v, v_bias);
    }

    *reinterpret_cast<V_vec *>(&v_cache[act_time_step * Dh]) = v;

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    out = fma(logits_smem[act_time_step], cast_to_float(v), out);
#else
    out = fma(logits_smem[act_time_step], v, out);
#endif
  }

  __syncthreads();

  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2;
         active_groups /= 2) {
      int midpoint = active_groups / 2;

      if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(
            *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]),
            out);
#else
        *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
      }
      __syncthreads();
      if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
        out =
            add(*reinterpret_cast<const V_vec *>(&out_smem[vo * Dh + vi]), out);
      }
      __syncthreads();
    }
  }

  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    V_vec tmp_out;
    convert_from_float(tmp_out, out);
    store_func.template store<V_vec>(tmp_out,
                                     thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#else
    store_func.template store<V_vec>(out,
                                     thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#endif
  }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // __syncthreads();
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("======fmha_out=====\n");
  //   for (int i = 0; i < Dh; ++i)
  //     printf("%f ", static_cast<float>(params.out[i]));
  //   printf("\n");
  // }
#endif
#else
  assert(false);
#endif
}

template <typename T>
inline size_t smem_size_in_bytes(
    const Masked_multihead_attention_params<T> &params,
    int dim_head,
    int threads_per_value,
    int threads_per_block) {
  size_t qk_sz = div_up(params.timestep + 1, 4) * 16;
  size_t logits_sz = 0;

#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS  // NOLINT
  if (sizeof(T) != 4) {
    logits_sz = div_up(params.max_seq_length, 4) * 4 * sizeof(T);
  }
#endif  // NOLINT
  size_t softmax_sz = qk_sz + logits_sz;

  int rows_per_red = threads_per_block / threads_per_value;
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;

  return max(softmax_sz, red_sz);
}

#define MMHA_LAUNCH_KERNEL(T,                                             \
                           Dh,                                            \
                           Dh_MAX,                                        \
                           THDS_PER_KEY,                                  \
                           THDS_PER_VALUE,                                \
                           THDS_PER_BLOCK,                                \
                           stream,                                        \
                           load_func,                                     \
                           store_func)                                    \
  size_t smem_sz =                                                        \
      smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);  \
  dim3 grid(params.num_head, params.batch_size);                          \
  constexpr auto kernel_fn =                                              \
      masked_multihead_attention_kernel<T,                                \
                                        Dh,                               \
                                        Dh_MAX,                           \
                                        THDS_PER_KEY,                     \
                                        THDS_PER_VALUE,                   \
                                        THDS_PER_BLOCK,                   \
                                        decltype(load_func),              \
                                        decltype(store_func)>;            \
  if (smem_sz > 0xc000) {                                                 \
    cudaFuncSetAttribute(                                                 \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz); \
  }                                                                       \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                   \
      params, load_func, store_func);

template <typename T,
          int Dh,
          int Dh_MAX,
          typename LoadFunc,
          typename StoreFunc,
          bool WITH_INT8 = false>
void fmha_launch_kernel(const Masked_multihead_attention_params<T> &params,
                        const cudaStream_t &stream,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  if (params.timestep < 32) {
    MMHA_LAUNCH_KERNEL(
        T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, stream, load_func, store_func);
  } else if (params.timestep < 2048) {
#if defined(MMHA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
    MMHA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       4,
                       THREADS_PER_VALUE,
                       256,
                       stream,
                       load_func,
                       store_func);
#else
    MMHA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       2,
                       THREADS_PER_VALUE,
                       128,
                       stream,
                       load_func,
                       store_func);
#endif
  } else {
    MMHA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       1,
                       THREADS_PER_VALUE,
                       256,
                       stream,
                       load_func,
                       store_func);
  }
}

template <typename T, typename LoadFunc, typename StoreFunc, bool WITH_INT8>
void fmha_impl(const phi::GPUContext &dev_ctx,
               const Masked_multihead_attention_params<T> &params,
               int dim_head,
               LoadFunc load_func,
               StoreFunc store_func) {
  switch (dim_head) {
    case 10:
      fmha_launch_kernel<T, 10, 32, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 26:
      fmha_launch_kernel<T, 26, 32, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 32:
      fmha_launch_kernel<T, 32, 32, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 64:
      fmha_launch_kernel<T, 64, 64, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 96:
      fmha_launch_kernel<T, 96, 128, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 128:
      fmha_launch_kernel<T, 128, 128, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 192:
      fmha_launch_kernel<T, 192, 256, LoadFunc, StoreFunc, WITH_INT8>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Dim_head = %d is unsupported!", dim_head));
  }
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc>
__global__ void multi_block_masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params,
    LoadFunc load_func,
    StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int bi = blockIdx.y;
  // Each Partition responsible for partial KeyCache and Value Cache Compute.
  const int partition_idx = blockIdx.z;

  const int act_time_step = params.sequence_lengths[bi];

  // There is no work for decoding.
  if (act_time_step == 0) {
    return;
  }

  /*
  Note(zhengzekang):
  If current block processed partition is out of the real sequence length, we
  directly terminate it.

  The reason for why we do not Init Zeros for partial expsum, maxlogits, output
  is: Each sequence need real partition is different, assume partition_size
  is 8. seq0[8] need 1 partition, seq1[26] need 4 partitions. Though we launch 4
  blocks in blockDim.z, for seq0, it only write the `partition_idx == 0`
  position value, other position_idx is random init.

  In rescale operations, it will also compute the real partition num to select
  value, for seq0, it will only select `partition_idx==0` partial value to do
  rescale.
  */
  if (partition_idx * params.partition_size >= act_time_step) {
    return;
  }
  const int num_partitions = div_up(act_time_step, params.partition_size);

  // Each Partition block's start position.
  const auto partition_times_timesteps_per_block =
      partition_idx * params.partition_size;

  typedef phi::PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

  constexpr int WARP_SIZE_TMP = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE_TMP;

  extern __shared__ char smem_[];
  float *qk_smem = reinterpret_cast<float *>(smem_);

  /*
  Here We allocate a shared float variable to store the New SingleQuery matmul
  the New SingleKey results. In previous implementation, we set the result in
  qk_smem[act_time_step] and then iterate KCache to get
  qk_smem[0...act_time_step-1]

  For now, we set the result in qk_current_smem. Final we will get result
  according to the time_now == timestep? if time_now < timestep, we get result
  from `qk_smem`, else from `qk_current_smem`.
  */
  __shared__ float qk_current_smem[1];

  // logits_smem is used to store the result of exp(q*k^T).
  char *logits_smem_ = smem_;

  T *out_smem = reinterpret_cast<T *>(smem_);

  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  using Qk_vec_RoPE = typename Qk_vec_RoPE_<T, float, Dh_MAX>::Type;
  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  const int tid = threadIdx.x;
  const int hi = blockIdx.x;  // head_idx
  const int kv_hi = hi / params.gqa_num_per_partitions;

  const int bhi = bi * params.num_head + hi;
  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;
  const int thi = params.cum_offsets ? ti * params.num_head + hi : -1;

  float qk_max = -FLT_MAX;
  float qk = 0;

  // qkv [B, S=1, 3, num_head, head_dim]
  int qkv_base_offset = bi * (params.num_head + 2 * params.gqa_group_size) *
                        Dh;  // // if no gqa, gqa_group_size = num_head

  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);

  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE_TMP, "");
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

  // cache_k, [B, num_head, head_dim / x, max_seq_len, x]
  // x == 4/8 for FP32/FP16, 128bit, 16Byte
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  const T *q_bias_base = nullptr;
  const T *k_bias_base = nullptr;

  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + params.num_head * Dh;
  }

  if (tid < QK_VECS_PER_WARP) {
    const int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset + hi * Dh);
    }

    Qk_vec k;
    zero(k);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(
          k, params.num_head * Dh + qk_offset + kv_hi * Dh);
    }

    if (params.add_qkv_bias) {
      const int q_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
      const int k_bias_offset = kv_hi * Dh + tid * QK_VEC_SIZE;
      Qk_vec q_bias;
      zero(q_bias);
      Qk_vec k_bias;
      zero(k_bias);

      q_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&q_bias_base[q_bias_offset])
              : q_bias;
      k_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&k_bias_base[k_bias_offset])
              : k_bias;

      q = add(q, q_bias);
      k = add(k, k_bias);
    }

    if (!params.neox_rotary_style) {
      if (params.rotary_emb_dims != 0) {
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rotary_bsz * Dh;
        Qk_vec_RoPE cos_emb, sin_emb;
        zero(cos_emb);
        zero(sin_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        apply_rotary_embedding(q, k, cos_emb, sin_emb);
      }
    } else {
      /* old rotary pos emb */
      if (params.rotary_emb_dims != 0) {
        int last_dim = Dh / params.rotary_emb_dims;
        int half_lastdim = last_dim / 2;
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rotary_bsz * Dh;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = tid / stride_all_lastdim * stride_all_lastdim +
                       (tid + stride) % (stride_all_lastdim);
        int q_right_offset = qkv_base_offset + hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_offset = qkv_base_offset + params.num_head * Dh +
                             kv_hi * Dh + right_id * QK_VEC_SIZE;
        Qk_vec q_right;
        zero(q_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(q_right, q_right_offset);
        }
        Qk_vec k_right;
        zero(k_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(k_right, k_right_offset);
        }

        Qk_vec_RoPE cos_emb;
        zero(cos_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;

        Qk_vec_RoPE sin_emb;
        zero(sin_emb);
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        float alpha = (tid % stride_all_lastdim) < stride
                          ? static_cast<float>(-1)
                          : static_cast<float>(1);
        q = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            q, q_right, cos_emb, sin_emb, alpha);
        k = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            k, k_right, cos_emb, sin_emb, alpha);
      }
    }

    *reinterpret_cast<Qk_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;
    if (partition_idx == num_partitions - 1) {
      if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
        int co = tid / QK_VECS_IN_16B;
        int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;
        int offset = bi * params.gqa_group_size * params.max_seq_length * Dh +
                     kv_hi * params.max_seq_length * Dh +
                     co * params.max_seq_length * QK_ELTS_IN_16B +
                     act_time_step * QK_ELTS_IN_16B + ci;
        *reinterpret_cast<Qk_vec *>(&params.cache_kv[offset]) = k;
      }

      qk = dot<Qk_vec, Qk_vec>(q, k);

      if (QK_VECS_PER_WARP <= WARP_SIZE_TMP) {
#pragma unroll
        for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
          qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
        }
      }
    }
  }

  if (partition_idx == num_partitions - 1) {
    if (QK_VECS_PER_WARP > WARP_SIZE_TMP) {
      constexpr int WARPS_PER_RED =
          (QK_VECS_PER_WARP + WARP_SIZE_TMP - 1) / WARP_SIZE_TMP;
      qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }
    if (tid == 0) {
      qk *= params.inv_sqrt_dh;
      qk_max = qk;
      // The query and new Key matmul result will be stored in `qk_current_smem`
      // not `qk_smem`!.
      qk_current_smem[0] = qk;
    }
  }

  __syncthreads();

  using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
  constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  int ko = tid / THREADS_PER_KEY;
  int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;
  T *k_cache =
      &params.cache_kv[bi * params.gqa_group_size * params.max_seq_length * Dh +
                       kv_hi * params.max_seq_length * Dh + ki];
  static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD, "");

  K_vec q[K_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
    q[i] = *reinterpret_cast<const K_vec *>(
        &q_smem[ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
  }

  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  constexpr int K_PER_WARP = WARP_SIZE_TMP / THREADS_PER_KEY;

  // Each Block iterate PARTITION_SIZE length KVCache.
  int ti_end = div_up(params.partition_size, K_PER_WARP) * K_PER_WARP;

  K_vec k[K_VECS_PER_THREAD];
  K_vec k_vec_zero;
  zero(k_vec_zero);
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    // First, move each block to their start position.
    const int time_now = ti + partition_times_timesteps_per_block;
    const int k_offset =
        bi * params.gqa_group_size * params.max_seq_length * Dh +
        kv_hi * params.max_seq_length * Dh + time_now * Dh + ki;

#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + time_now;
      if (time_now < act_time_step) {
        k[ii] =
            (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                ? *reinterpret_cast<const K_vec *>(
                      &k_cache[jj * QK_ELTS_IN_16B])
                : k_vec_zero;
      } else {
        k[ii] = k_vec_zero;
      }
    }

    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

    if (time_now < act_time_step && tid % THREADS_PER_KEY == 0) {
      qk_max = fmaxf(qk_max, qk);
      qk_smem[ti] = qk;
    }
  }

#pragma unroll
  for (int mask = WARP_SIZE_TMP / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  const int warp = tid / WARP_SIZE_TMP;
  const int lane = tid % WARP_SIZE_TMP;

  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  __syncthreads();

  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  float exp_sum = 0.f;
  for (int ti = tid; ti <= params.partition_size; ti += THREADS_PER_BLOCK) {
    const int time_now = ti + partition_times_timesteps_per_block;
    /*
    Here if we processed the seq_idx < act_time_step, it means we processed the
    KVCache, store in `qk_smem`. if seq_idx == act_time_step, it means the
    thread processed the new Single Query, Key. store in `qk_current_smem`.
    */
    if (time_now < act_time_step && ti != params.partition_size) {
      float logit = expf(qk_smem[ti] - qk_max);
      exp_sum += logit;
      qk_smem[ti] = logit;
    } else if (time_now == act_time_step) {
      float logit = expf(qk_current_smem[0] - qk_max);
      exp_sum += logit;
      qk_current_smem[0] = logit;
    }
  }

  exp_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], exp_sum);
  // Here store the max logit and exp_sum for rescale in reduce kernel.
  if (tid == 0) {
    float *max_logits_ptr = params.max_logits +
                            bi * params.num_head * params.max_num_partitions +
                            hi * params.max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float *exp_sums_ptr = params.exp_sums +
                          bi * params.num_head * params.max_num_partitions +
                          hi * params.max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

  T *v_cache =
      &params.cache_kv[params.cache_batch_size * params.gqa_group_size *
                           params.max_seq_length * Dh +
                       bi * params.gqa_group_size * params.max_seq_length * Dh +
                       kv_hi * params.max_seq_length * Dh + vi];

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif

  V_vec_acum out;
  zero(out);

  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if (Dh == Dh_MAX || vi < Dh) {
    // for (int ti = vo; ti < act_time_step; ti += V_PER_ITER) {
    for (int ti = vo; ti < params.partition_size; ti += V_PER_ITER) {
      // local time idx means the current time step in each threadBlock.
      int local_time_idx = ti + partition_times_timesteps_per_block;

      // Only local_time_idx < act_time_step, do logits matmul CacheV.
      const bool is_mask = local_time_idx >= act_time_step;
      if (!is_mask) {
        V_vec v;
        v = *reinterpret_cast<const V_vec *>(&v_cache[local_time_idx * Dh]);

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        float logit = qk_smem[ti];
        out = fma(logit, cast_to_float(v), out);
#else
        DataType_ logit = static_cast<DataType_>(qk_smem[ti]);
        // Update the partial sums.
        out = fma(logit, v, out);
#endif
      }
    }
  }

  V_vec v_bias;
  zero(v_bias);

  /*
  Note(Zhengzekang): Only the Last Block(in gridDim.z axis) need to add [logits
  x V] result to out. and the new logits will be fetch from
  `logits_current_smem`  not `logits_smem`.

  We should also notice that each sequence use partition num is different.
  For example we have 2 sequence, and partition size is 8.
  - seq0: 8
  - seq1: 26.
  We launch blockZ is determined by the max_seqlen / partition_size ->
  round_up(26 / 8) = 4 For seq0, it only need 1 partition to process, and the
  Last Block for seq0 is blockIdx.z==0. So here we need to compute Last Block
  Index for each sequence like:

  - num_partitions = div_up(context_len, params.partition_size);

  And add a condition: (blockIdx.z == num_partitions - 1)
  */
  if ((blockIdx.z == num_partitions - 1) &&
      vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v;
    load_func.template load<V_vec>(
        v,
        (params.num_head + params.gqa_group_size) * Dh + qkv_base_offset +
            kv_hi * Dh + vi);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[(params.num_head + params.gqa_group_size) * Dh +
                           kv_hi * Dh + vi]);
      v = add(v, v_bias);
    }

    *reinterpret_cast<V_vec *>(&v_cache[act_time_step * Dh]) = v;

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    out = fma(qk_current_smem[0], cast_to_float(v), out);
#else
    out = fma(qk_current_smem[0], v, out);
#endif
  }

  __syncthreads();

  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2;
         active_groups /= 2) {
      int midpoint = active_groups / 2;

      if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(
            *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]),
            out);
#else
        *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
      }
      __syncthreads();
      if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
        out =
            add(*reinterpret_cast<const V_vec *>(&out_smem[vo * Dh + vi]), out);
      }
      __syncthreads();
    }
  }

  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
    // Compute the index to store in `partial_out`.
    const int32_t store_partial_idx =
        bi * params.num_head * params.max_num_partitions * Dh +
        hi * params.max_num_partitions * Dh + partition_idx * Dh + vi;
    // Actually, we do not need the store_func, just use T vectorized type
    // `V_vec` to store in params.partial_out.
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    V_vec tmp_out;
    convert_from_float(tmp_out, out);
    *reinterpret_cast<V_vec *>(params.partial_out + store_partial_idx) =
        tmp_out;
#else
    *reinterpret_cast<V_vec *>(params.partial_out + store_partial_idx) = out;
#endif
  }
#else
  assert(false);
#endif
}

template <typename T, int HEAD_SIZE, int THREADS_PER_BLOCK, typename StoreFunc>
__global__
__launch_bounds__(THREADS_PER_BLOCK) void multi_block_attention_reduce_kernel(
    Masked_multihead_attention_params<T> params, StoreFunc store_func) {
  const int num_heads = params.num_head;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = params.sequence_lengths[seq_idx];
  if (context_len == 0) {
    return;
  }

  const int bhi = seq_idx * params.num_head + head_idx;
  const int ti = params.cum_offsets
                     ? seq_idx * params.seq_len - params.cum_offsets[seq_idx]
                     : -1;
  const int thi = params.cum_offsets ? ti * params.num_head + head_idx : -1;

  const int num_partitions = div_up(context_len, params.partition_size);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    const float *exp_sums_ptr =
        params.exp_sums + seq_idx * num_heads * params.max_num_partitions +
        head_idx * params.max_num_partitions;
    const float inv_global_exp_sum = fdividef(1.0f, exp_sums_ptr[0] + 1e-6f);
    T *tmp_out_ptr =
        params.partial_out +
        seq_idx * num_heads * params.max_num_partitions * HEAD_SIZE +
        head_idx * params.max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      store_func.template store<T>(
          *(tmp_out_ptr + i),
          inv_global_exp_sum,
          thi != -1 ? thi * HEAD_SIZE + i : bhi * HEAD_SIZE + i);
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE_TMP;
  const int warp_idx = threadIdx.x / WARP_SIZE_TMP;
  const int lane = threadIdx.x % WARP_SIZE_TMP;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float *shared_max_logits = reinterpret_cast<float *>(shared_mem);
  const float *max_logits_ptr =
      params.max_logits + seq_idx * num_heads * params.max_num_partitions +
      head_idx * params.max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE_TMP / 2; mask >= 1; mask /= 2) {
    max_logit =
        fmaxf(max_logit, __shfl_xor_sync(uint32_t(-1), max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit =
        fmaxf(max_logit, __shfl_xor_sync(uint32_t(-1), max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = __shfl_sync(uint32_t(-1), max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float *exp_sub_maxes =
      reinterpret_cast<float *>(shared_mem + sizeof(float) * num_partitions);
  const float *exp_sums_ptr = params.exp_sums +
                              seq_idx * num_heads * params.max_num_partitions +
                              head_idx * params.max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float exp_sub_max = expf(l - max_logit);
    float rescaled_exp_sum = exp_sums_ptr[i] * exp_sub_max;
    global_exp_sum += rescaled_exp_sum;
    exp_sub_maxes[i] = exp_sub_max;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  T *tmp_out_ptr = params.partial_out +
                   seq_idx * num_heads * params.max_num_partitions * HEAD_SIZE +
                   head_idx * params.max_num_partitions * HEAD_SIZE;

#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += THREADS_PER_BLOCK) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += static_cast<float>(tmp_out_ptr[j * HEAD_SIZE + i]) *
             exp_sub_maxes[j] * inv_global_exp_sum;
    }
    T acc_val = static_cast<T>(acc);
    store_func.template store<T>(
        acc_val, thi != -1 ? thi * HEAD_SIZE + i : bhi * HEAD_SIZE + i);
  }
}

template <typename T>
inline size_t multi_block_attn_smem_size_in_bytes(
    const Masked_multihead_attention_params<T> &params,
    int dim_head,
    int threads_per_value,
    int threads_per_block) {
  size_t qk_sz = div_up(params.partition_size, 4) * 4 * sizeof(float);

  size_t logits_table_sz = qk_sz;

  int rows_per_red = threads_per_block / threads_per_value;
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;

  return max(logits_table_sz, red_sz);
}

template <typename T>
inline size_t get_reduce_smem_size_in_bytes(
    const Masked_multihead_attention_params<T> &params) {
  const int32_t max_num_partitions =
      div_up(params.timestep, params.partition_size);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
  VLOG(1) << "get_reduce_smem_size_in_bytes, reduce_shared_mem_size: "
          << reduce_shared_mem_size;
  return reduce_shared_mem_size;
}

#define MBMMHA_LAUNCH_KERNEL(T,                                             \
                             Dh,                                            \
                             Dh_MAX,                                        \
                             THDS_PER_KEY,                                  \
                             THDS_PER_VALUE,                                \
                             THDS_PER_BLOCK,                                \
                             stream,                                        \
                             load_func,                                     \
                             reduce_store_func)                             \
  VLOG(1) << "THREADS_PER_VALUE is: " << THREADS_PER_VALUE                  \
          << ", partition_size: " << params.partition_size;                 \
  size_t smem_sz = multi_block_attn_smem_size_in_bytes<T>(                  \
      params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);                          \
  dim3 grid(params.num_head,                                                \
            params.batch_size,                                              \
            div_up(params.timestep, params.partition_size));                \
  constexpr auto kernel_fn = multi_block_masked_multihead_attention_kernel< \
      T,                                                                    \
      Dh,                                                                   \
      Dh_MAX,                                                               \
      THDS_PER_KEY,                                                         \
      THDS_PER_VALUE,                                                       \
      THDS_PER_BLOCK,                                                       \
      decltype(load_func),                                                  \
      decltype(reduce_store_func)>;                                         \
  if (smem_sz > 0xc000) {                                                   \
    cudaFuncSetAttribute(                                                   \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);   \
  }                                                                         \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                     \
      params, load_func, reduce_store_func);                                \
                                                                            \
  dim3 reduce_kernel_grid(params.num_head, params.batch_size, 1);           \
  size_t reduce_smem_sz = get_reduce_smem_size_in_bytes<T>(params);         \
  constexpr int MBLHA_REDUCE_BLOCK_SIZE = 256;                              \
  constexpr auto reduce_kernel_fn =                                         \
      multi_block_attention_reduce_kernel<T,                                \
                                          Dh,                               \
                                          MBLHA_REDUCE_BLOCK_SIZE,          \
                                          decltype(reduce_store_func)>;     \
  reduce_kernel_fn<<<reduce_kernel_grid,                                    \
                     MBLHA_REDUCE_BLOCK_SIZE,                               \
                     reduce_smem_sz,                                        \
                     stream>>>(params, reduce_store_func);

template <typename T,
          int Dh,
          int Dh_MAX,
          typename LoadFunc,
          typename ReduceStoreFunc>
void dispatch_mbmmha_impl(const Masked_multihead_attention_params<T> &params,
                          const cudaStream_t &stream,
                          LoadFunc load_func,
                          ReduceStoreFunc reduce_store_func) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  MBMMHA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       1,
                       THREADS_PER_VALUE,
                       256,
                       stream,
                       load_func,
                       reduce_store_func)
}

template <typename T, typename LoadFunc, typename ReduceStoreFunc>
void dispatch_mbmmha_impl_headsize(
    const phi::GPUContext &dev_ctx,
    const Masked_multihead_attention_params<T> &params,
    int dim_head,
    LoadFunc load_func,
    ReduceStoreFunc reduce_store_func,
    const cudaStream_t &stream) {
  switch (dim_head) {
    case 32:
      dispatch_mbmmha_impl<T, 32, 32>(
          params, stream, load_func, reduce_store_func);
      break;
    case 64:
      dispatch_mbmmha_impl<T, 64, 64>(
          params, stream, load_func, reduce_store_func);
      break;
    case 128:
      dispatch_mbmmha_impl<T, 128, 128>(
          params, stream, load_func, reduce_store_func);
      break;
    default:
      PADDLE_THROW(
          errors::Unimplemented("Dim_head = %d is unsupported!", dim_head));
  }
}

template <typename T>
void DispatchMBMMHA(const phi::GPUContext &dev_ctx,
                    const cudaStream_t &stream,
                    const phi::DenseTensor &qkv_tensor,
                    const Masked_multihead_attention_params<T> &params,
                    int num_head,
                    int dim_head,
                    phi::DenseTensor *out_tensor) {
  // In Multi Block Mode, we store partial out val(type is T) into
  // params.partial_out. Then we do final reduce and postprocess (such quant to
  // int8.) to save in out_tensor.
  MMHALoad<T> load_func(qkv_tensor.data<T>());
  MMHAStore<T> reduce_store_func(out_tensor->data<T>());
  VLOG(1) << "get into dispatch_mblha_impl_headsize";
  dispatch_mbmmha_impl_headsize<T,
                                decltype(load_func),
                                decltype(reduce_store_func)>(
      dev_ctx, params, dim_head, load_func, reduce_store_func, stream);
}

template <typename T>
void mbfmha(const phi::GPUContext &dev_ctx,
            const phi::DenseTensor &qkv_tensor,
            const phi::DenseTensor &qkv_bias_tensor,
            const phi::DenseTensor *src_mask_tensor,
            const phi::DenseTensor *cum_offsets_tensor,
            const phi::DenseTensor *sequence_lengths_tensor,
            const phi::DenseTensor *rotary_tensor,
            phi::DenseTensor *cache_kv_tensor,
            phi::DenseTensor *out_tensor,
            phi::DenseTensor *partial_max_logits_tensor,
            phi::DenseTensor *partial_expsum_tensor,
            phi::DenseTensor *partial_out_tensor,
            int batch_size,
            int cache_batch_size,
            int seq_len,
            int max_seq_length,
            int num_head,
            int dim_head,
            int timestep,
            int rotary_emb_dims,
            float inv_sqrt_dh,
            const bool mask_broadcast_num_heads = true,
            const bool add_qkv_bias = true,
            const bool neox_rotary_style = false,
            const int gqa_group_size = -1) {
  VLOG(1) << "MBFMHA is used in FusedMT.";
  Masked_multihead_attention_params<T> params;
  cudaStream_t stream = dev_ctx.stream();

  if (gqa_group_size > 0) {
    params.gqa_group_size = gqa_group_size;
    params.gqa_num_per_partitions = num_head / gqa_group_size;
  } else {
    params.gqa_group_size = num_head;
    params.gqa_num_per_partitions = 1;
  }
  VLOG(1) << "params.gqa_group_size " << params.gqa_group_size;
  VLOG(1) << "params.gqa_num_per_partitions " << params.gqa_num_per_partitions;

  params.cache_kv = cache_kv_tensor->data<T>();
  params.neox_rotary_style = neox_rotary_style;
  if (src_mask_tensor) {
    params.attn_mask = src_mask_tensor->data<T>();
    params.mask_length = src_mask_tensor->dims()[3];
  } else {
    params.attn_mask = nullptr;
    params.mask_length = -1;
  }

  params.exp_sums = partial_expsum_tensor->data<float>();
  params.max_logits = partial_max_logits_tensor->data<float>();
  params.partial_out = partial_out_tensor->data<T>();

  if (sequence_lengths_tensor) {
    params.sequence_lengths = sequence_lengths_tensor->data<int>();
  }

  if (cum_offsets_tensor) {
    params.cum_offsets = cum_offsets_tensor->data<int>();
  } else {
    params.cum_offsets = nullptr;
  }
  params.seq_len = seq_len;
  if (rotary_emb_dims > 0) {
    params.rotary_emb = rotary_tensor->data<float>();
    params.rotary_bsz = rotary_tensor->dims()[1];
  } else {
    params.rotary_emb = nullptr;
    params.rotary_bsz = 0;
  }

  params.add_qkv_bias = add_qkv_bias;
  if (add_qkv_bias) {
    params.qkv_bias = const_cast<T *>(qkv_bias_tensor.data<T>());
  }

  VLOG(1) << "gqa_group_size " << params.gqa_group_size;
  VLOG(1) << "gqa_num_per_partitions " << params.gqa_num_per_partitions;

  params.add_qkv_bias = add_qkv_bias;
  params.batch_size = batch_size;
  params.cache_batch_size = cache_batch_size;
  params.num_head = num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_length;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;

  /*
  Note(Zhengzekang): We preallocate the partial variable by
  `FLAGS_multi_block_attention_min_partition_size`.

  For align offset, we also compute max_num_partitions by using
  `FLAGS_multi_block_attention_min_partition_size`.
  */
  params.max_num_partitions = div_up(
      params.timestep,
      static_cast<int32_t>(FLAGS_multi_block_attention_min_partition_size));
  params.partition_size = FLAGS_multi_block_attention_min_partition_size;

  DispatchMBMMHA<T>(
      dev_ctx, stream, qkv_tensor, params, num_head, dim_head, out_tensor);
}

template <typename T>
void fmha(const phi::GPUContext &dev_ctx,
          const phi::DenseTensor &qkv_tensor,
          const phi::DenseTensor &qkv_bias_tensor,
          const phi::DenseTensor *src_mask_tensor,
          const phi::DenseTensor *cum_offsets_tensor,
          const phi::DenseTensor *sequence_lengths_tensor,
          const phi::DenseTensor *rotary_tensor,
          const phi::DenseTensor *beam_cache_offset_tensor,
          phi::DenseTensor *cache_kv_tensor,
          phi::DenseTensor *out_tensor,
          int batch_size,
          int cache_batch_size,
          int seq_len,
          int max_seq_length,
          int num_head,
          int dim_head,
          int timestep,
          int rotary_emb_dims,
          float inv_sqrt_dh,
          const bool mask_broadcast_num_heads = true,
          const bool add_qkv_bias = true,
          const bool neox_rotary_style = false,
          const int gqa_group_size = -1) {
  VLOG(1) << "FMHA is used in FusedMT.";
  Masked_multihead_attention_params<T> params;
  // params.out = out_tensor->data<T>();
  // params.qkv = qkv_tensor.data<T>();

  if (add_qkv_bias) {
    // Because we may not add qkv_bias, so here we cast to T*.
    // Author(zhengzekang).
    params.qkv_bias = const_cast<T *>(qkv_bias_tensor.data<T>());
  }
  params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  params.cache_kv = cache_kv_tensor->data<T>();

  params.neox_rotary_style = neox_rotary_style;
  if (src_mask_tensor) {
    params.attn_mask = src_mask_tensor->data<T>();
    params.mask_length = src_mask_tensor->dims()[3];
  } else {
    params.attn_mask = nullptr;
    params.mask_length = -1;
  }

  if (sequence_lengths_tensor) {
    params.sequence_lengths = sequence_lengths_tensor->data<int>();
  }

  if (cum_offsets_tensor) {
    params.cum_offsets = cum_offsets_tensor->data<int>();
  } else {
    params.cum_offsets = nullptr;
  }
  params.seq_len = seq_len;

  if (rotary_emb_dims > 0) {
    params.rotary_emb = rotary_tensor->data<float>();
    params.rotary_bsz = rotary_tensor->dims()[1];
  } else {
    params.rotary_emb = nullptr;
    params.rotary_bsz = 0;
  }

  if (beam_cache_offset_tensor) {
    params.beam_cache_offset = beam_cache_offset_tensor->data<int>();
    params.beam_width = beam_cache_offset_tensor->dims()[1];
  }

  if (gqa_group_size > 0) {
    params.gqa_group_size = gqa_group_size;
    params.gqa_num_per_partitions = num_head / gqa_group_size;
  } else {
    params.gqa_group_size = num_head;
    params.gqa_num_per_partitions = 1;
  }

  VLOG(1) << "gqa_group_size " << params.gqa_group_size;
  VLOG(1) << "gqa_num_per_partitions " << params.gqa_num_per_partitions;

  params.add_qkv_bias = add_qkv_bias;
  params.batch_size = batch_size;
  params.cache_batch_size = cache_batch_size;
  params.num_head = num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_length;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;

  MMHALoad<T> load_func(qkv_tensor.data<T>());
  MMHAStore<T> store_func(out_tensor->data<T>());
  fmha_impl<T, decltype(load_func), decltype(store_func), false>(
      dev_ctx, params, dim_head, load_func, store_func);
}

// NOTE: simd with 16Bytes(128bit), float is 4, float16 is 8
constexpr int VEC_16B = 16;

template <typename T>
__global__ void write_cache_k_kernel(T *cache_k,
                                     const T *k,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto k_src = reinterpret_cast<const uint4 *>(
      k + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  auto k_dst = reinterpret_cast<uint4 *>(
      cache_k + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // vec size
  int dim_head_div_x = dim_head / X_ELEMS;

  // FIXME(wangxi): num_head is not need?
  // if (out_idx >= num_head * dim_head_div_x * max_seq_len) return;
  if (out_idx >= dim_head_div_x * max_seq_len) return;

  int idx = out_idx;
  const int k_seq_len_id = idx % max_seq_len;
  // idx = (idx - k_seq_len_id) / max_seq_len;
  idx = idx / max_seq_len;
  const int k_vec_id = idx % dim_head_div_x;

  if (k_seq_len_id < seq_len) {
    k_dst[out_idx] = k_src[k_seq_len_id * dim_head_div_x + k_vec_id];
  }
}

template <typename T>
__global__ void write_cache_v_kernel(T *cache_v,
                                     const T *v,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int hi = blockIdx.z;

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto v_src = reinterpret_cast<const uint4 *>(
      v + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<uint4 *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * seq_len) return;

  v_dst[idx] = v_src[idx];
}

template <typename T>
void write_cache_kv(const phi::GPUContext &dev_ctx,
                    T *cache_k,
                    T *cache_v,
                    const T *k,
                    const T *v,
                    const int bsz,
                    const int num_head,
                    const int seq_len,
                    const int max_seq_len,
                    const int dim_head) {
  constexpr int block_sz = 128;
  constexpr int x = VEC_16B / sizeof(T);

  assert(dim_head % x == 0);
  PADDLE_ENFORCE_EQ(
      dim_head % x,
      0,
      common::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, x));

  int max_size = max_seq_len * dim_head / x;
  int size = seq_len * dim_head / x;
  dim3 grid(div_up(max_size, block_sz), bsz, num_head);
  dim3 grid_v(div_up(size, block_sz), bsz, num_head);

  // transpose [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  write_cache_k_kernel<<<grid, block_sz, 0, dev_ctx.stream()>>>(
      cache_k, k, num_head, dim_head, seq_len, max_seq_len);

  // copy [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  write_cache_v_kernel<<<grid_v, block_sz, 0, dev_ctx.stream()>>>(
      cache_v, v, num_head, dim_head, seq_len, max_seq_len);
}

template <typename T, int X_ELEMS>
__global__ void gqa_write_cache_k_kernel(T *cache_k,
                                         const T *k,
                                         const int *seq_lens,
                                         const int *padding_offsets,
                                         const int gqa_group_size,
                                         const int max_seq_len,
                                         const int seq_len,
                                         const int dim_head,
                                         const int64_t num_elems) {
  phi::AlignedVector<T, X_ELEMS> in_vec;

  for (int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * X_ELEMS;
       linear_idx < num_elems;
       linear_idx += blockDim.x * gridDim.x * X_ELEMS) {
    const int hidden_size = gqa_group_size * dim_head;
    const int token_idx = linear_idx / hidden_size;
    const int head_idx = (linear_idx % hidden_size) / dim_head;
    const int head_offset = linear_idx % dim_head;
    const int head_vec_id = head_offset / X_ELEMS;
    const int ori_token_id = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_id / seq_len;

    if (seq_lens[ori_bi] == 0) continue;

    const int local_token_id = ori_token_id % seq_len;

    const int tgt_idx = ori_bi * gqa_group_size * max_seq_len * dim_head +
                        head_idx * max_seq_len * dim_head +
                        head_vec_id * max_seq_len * X_ELEMS +
                        local_token_id * X_ELEMS;

    phi::Load(&k[linear_idx], &in_vec);
    phi::Store(in_vec, &cache_k[tgt_idx]);
  }
}

template <typename T, int X_ELEMS>
__global__ void gqa_write_cache_v_kernel(T *cache_v,
                                         const T *v,
                                         const int *seq_lens,
                                         const int *padding_offsets,
                                         const int gqa_group_size,
                                         const int max_seq_len,
                                         const int seq_len,
                                         const int dim_head,
                                         const int64_t num_elems) {
  phi::AlignedVector<T, X_ELEMS> in_vec;

  for (int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * X_ELEMS;
       linear_idx < num_elems;
       linear_idx += blockDim.x * gridDim.x * X_ELEMS) {
    const int hidden_size = gqa_group_size * dim_head;
    const int token_idx = linear_idx / hidden_size;
    const int head_idx = (linear_idx % hidden_size) / dim_head;
    const int head_offset = linear_idx % dim_head;
    const int ori_token_id = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_id / seq_len;

    if (seq_lens[ori_bi] == 0) continue;

    const int local_token_id = ori_token_id % seq_len;

    const int tgt_idx = ori_bi * gqa_group_size * max_seq_len * dim_head +
                        head_idx * max_seq_len * dim_head +
                        local_token_id * dim_head + head_offset;

    phi::Load(&v[linear_idx], &in_vec);
    phi::Store(in_vec, &cache_v[tgt_idx]);
  }
}

template <typename T>
void gqa_write_cachekv(
    const phi::GPUContext &dev_ctx,
    phi::DenseTensor *cache_kv_out,  // [2, cache_bsz, gqa_group_size,
                                     // max_seq_len, dim_head] k need
    const phi::DenseTensor
        &unpadding_k,  // [token_num, gqa_group_size, dim_head]
    const phi::DenseTensor &unpadding_v,
    const phi::DenseTensor &padding_offsets,
    const phi::DenseTensor &seq_lens,
    const int seq_len) {
  constexpr int block_sz = 128;
  constexpr int x = VEC_16B / sizeof(T);

  const int cache_bsz = cache_kv_out->dims()[1];
  const int gqa_group_size = cache_kv_out->dims()[2];
  const int max_seq_len = cache_kv_out->dims()[3];
  const int dim_head = cache_kv_out->dims()[4];

  assert(dim_head % x == 0);
  PADDLE_ENFORCE_EQ(
      dim_head % x,
      0,
      common::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, x));

  const int64_t num_elems = unpadding_k.numel();

  T *cache_k = cache_kv_out->data<T>();
  T *cache_v = cache_k + cache_bsz * gqa_group_size * max_seq_len * dim_head;

  int grid_size;
  GetNumBlocks(num_elems, &grid_size);

  gqa_write_cache_k_kernel<T, x><<<grid_size, block_sz, 0, dev_ctx.stream()>>>(
      cache_k,
      unpadding_k.data<T>(),
      seq_lens.data<int>(),
      padding_offsets.data<int>(),
      gqa_group_size,
      max_seq_len,
      seq_len,
      dim_head,
      num_elems);
  gqa_write_cache_v_kernel<T, x><<<grid_size, block_sz, 0, dev_ctx.stream()>>>(
      cache_v,
      unpadding_v.data<T>(),
      seq_lens.data<int>(),
      padding_offsets.data<int>(),
      gqa_group_size,
      max_seq_len,
      seq_len,
      dim_head,
      num_elems);
}

template <typename T, int VecSize>
__global__ void fusedQKV_transpose_split_kernel(T *q_buf,
                                                T *k_buf,
                                                T *v_buf,
                                                const T *qkv,
                                                const int *padding_offset,
                                                const int *seq_lens,
                                                const int32_t elem_cnt,
                                                const int batch_size,
                                                const int seq_len,
                                                const int token_num,
                                                const int head_num,
                                                const int size_per_head) {
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;

    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    const int32_t write_idx =
        token_idx * hidden_size + head_id * size_per_head + size_id;
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &q_buf[write_idx]);
    } else if (qkv_id == 1) {
      phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
    }
  }
}

template <typename T>
void qkv_transpose_split(const phi::GPUContext &dev_ctx,
                         T *q_buf,
                         T *k_buf,
                         T *v_buf,
                         const T *qkv,
                         const int *padding_offset,
                         const int *seq_lens,
                         const int token_num,
                         const int batch_size,
                         const int head_num,
                         const int seq_len,
                         const int size_per_head) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    common::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                      k_buf,
                                                      v_buf,
                                                      qkv,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head);
}

template <typename T, int VecSize, bool ComputeBias>
__global__ void add_fusedQKV_bias_transpose_split_kernel(
    T *q_buf,
    T *kv_buf,
    const T *qkv,
    const T *qkv_bias,
    const int *padding_offset,
    const int32_t elem_cnt,
    const int batch_size,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int size_per_head) {
  const int32_t offset = batch_size * seq_len * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    if (ComputeBias) {
      phi::Load<T, VecSize>(&qkv_bias[bias_idx], &bias_vec);
#pragma unroll
      for (int32_t unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
        src_vec[unroll_idx] += bias_vec[unroll_idx];
      }
    }
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    const int32_t seq_id = ori_token_idx % seq_len;

    // equal to:
    // const int qkv_id  = (linear_index % fused_hidden_size) / hidden_size;
    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    if (qkv_id == 0) {
      phi::Store<T, VecSize>(
          src_vec,
          &q_buf[target_batch_id * head_num * seq_len * size_per_head +
                 head_id * seq_len * size_per_head + seq_id * size_per_head +
                 size_id]);
    } else {
      const int32_t kv_store_offset = (qkv_id - 1) * offset;
      phi::Store<T, VecSize>(
          src_vec,
          &kv_buf[kv_store_offset +
                  target_batch_id * head_num * seq_len * size_per_head +
                  head_id * seq_len * size_per_head + seq_id * size_per_head +
                  size_id]);
    }
  }
}

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return cudaSuccess;
}

template <typename T>
void qkv_bias_add_transpose_split(const phi::GPUContext &dev_ctx,
                                  T *q_buf,
                                  T *kv_buf,
                                  const T *qkv,
                                  const T *qkv_bias,
                                  const int *padding_offset,
                                  const int token_num,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len,
                                  const int size_per_head,
                                  bool compute_bias) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    common::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (compute_bias) {
    add_fusedQKV_bias_transpose_split_kernel<T, PackSize, true>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                        kv_buf,
                                                        qkv,
                                                        qkv_bias,
                                                        padding_offset,
                                                        elem_cnt,
                                                        batch_size,
                                                        seq_len,
                                                        token_num,
                                                        head_num,
                                                        size_per_head);
  } else {
    add_fusedQKV_bias_transpose_split_kernel<T, PackSize, false>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                        kv_buf,
                                                        qkv,
                                                        qkv_bias,
                                                        padding_offset,
                                                        elem_cnt,
                                                        batch_size,
                                                        seq_len,
                                                        token_num,
                                                        head_num,
                                                        size_per_head);
  }
}

template <typename T, int VecSize>
__global__ void gqa_fusedQKV_transpose_split_kernel(T *q_buf,
                                                    T *k_buf,
                                                    T *v_buf,
                                                    const T *qkv,
                                                    const int *padding_offset,
                                                    const int *seq_lens,
                                                    const int32_t elem_cnt,
                                                    const int batch_size,
                                                    const int seq_len,
                                                    const int token_num,
                                                    const int head_num,
                                                    const int size_per_head,
                                                    const int gqa_group_size) {
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  const int fused_hidden_size = (head_num + 2 * gqa_group_size) * size_per_head;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;

    const int32_t head_id = bias_idx / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    // [token_num, num_head or gqa_group_size, size_per_head]
    if (head_id < head_num) {
      const int32_t write_idx = token_idx * head_num * size_per_head +
                                head_id * size_per_head + size_id;
      phi::Store<T, VecSize>(src_vec, &q_buf[write_idx]);
    } else {
      if (head_id < head_num + gqa_group_size) {
        const int32_t write_idx = token_idx * gqa_group_size * size_per_head +
                                  (head_id - head_num) * size_per_head +
                                  size_id;
        phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
      } else {
        const int32_t write_idx =
            token_idx * gqa_group_size * size_per_head +
            (head_id - head_num - gqa_group_size) * size_per_head + size_id;
        phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
      }
    }
  }
}

template <typename T>
void gqa_qkv_transpose_split(const phi::GPUContext &dev_ctx,
                             T *q_buf,
                             T *k_buf,
                             T *v_buf,
                             const T *qkv,
                             const int *padding_offset,
                             const int *seq_lens,
                             const int token_num,
                             const int batch_size,
                             const int head_num,
                             const int seq_len,
                             const int size_per_head,
                             const int gqa_group_size) {
  const int32_t elem_cnt =
      token_num * (head_num + 2 * gqa_group_size) * size_per_head;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    common::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  gqa_fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                      k_buf,
                                                      v_buf,
                                                      qkv,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head,
                                                      gqa_group_size);
}

/* old rope emb */
template <typename T>
__global__ void NeoXRotaryKernel(const T *input,
                                 const float *cos_emb,
                                 const float *sin_emb,
                                 const int *sequence_lengths,
                                 T *output,
                                 const int rotary_emb_dims,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * head_num * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + ti;
    const int right_idx = base_idx + ti + half_lastdim;
    int emb_idx_left = bi * seq_len * last_dim + si * last_dim + ti;
    int emb_idx_right =
        bi * seq_len * last_dim + si * last_dim + ti + half_lastdim;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);

    float cos_tmp_left = cos_emb[emb_idx_left];
    float sin_tmp_left = sin_emb[emb_idx_left];
    float cos_tmp_right = cos_emb[emb_idx_right];
    float sin_tmp_right = sin_emb[emb_idx_right];

    T res1 =
        static_cast<T>(input_left * cos_tmp_left - input_right * sin_tmp_left);
    T res2 = static_cast<T>(input_right * cos_tmp_right +
                            input_left * sin_tmp_right);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}

template <typename T>
__global__ void RotaryKernel(const T *input,
                             const float *cos_emb,
                             const float *sin_emb,
                             const int *sequence_lengths,
                             T *output,
                             const int rotary_emb_dims,
                             const int batch_size,
                             const int head_num,
                             const int seq_len,
                             const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  // Note(ZhenyuLi): Calculate the relevant data at one time, so that no
  // additional space is required.
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * head_num * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + 2 * ti;
    const int right_idx = base_idx + 2 * ti + 1;
    int emb_idx = bi * seq_len * last_dim + si * last_dim + 2 * ti;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);
    float cos_tmp = cos_emb[emb_idx];
    float sin_tmp = sin_emb[emb_idx];
    T res1 = static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
    T res2 = static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}

template <typename T>
void rotary_qk(const phi::GPUContext &dev_ctx,
               T *q,
               T *k,              // kv
               const T *q_input,  // q
               const T *k_input,  // kv
               const float *rotary_emb,
               const int *sequence_lengths,
               const int rotary_emb_dims,
               const int rope_bsz,
               const int batch_size,
               const int head_num,
               const int seq_len,
               const int dim_head,
               const bool neox_rotary_style) {
  // q_transpose_out_data [bs, head_num, seq_len, dim_head] -> [bs, head_num,
  // seq_len * rotary_emb_dims, dim_head / rotary_emb_dims]
  // kv_transpose_out_data [bs, head_num, seq_len, dim_head] -> [bs, head_num,
  // seq_len * rotary_emb_dims, dim_head / rotary_emb_dims] rotary_emb [2, bs,
  // 1, seq_len, dim_head] -> [2, bs, 1, seq_len * rotary_emb_dims, dim_head /
  // rotary_emb_dims]
  dim3 grid(batch_size, head_num, seq_len * rotary_emb_dims);
  const int last_dim = dim_head / rotary_emb_dims;
  auto getBlockSize = [](int dim) {
    if (dim > 256) {
      return 512;
    } else if (dim > 128) {
      return 256;
    } else if (dim > 64) {
      return 128;
    } else if (dim > 32) {
      return 64;
    } else {
      return 32;
    }
  };
  int BlockSize = getBlockSize(last_dim / 2);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + batch_size * seq_len * dim_head;
  if (!neox_rotary_style) {
    RotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        q_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        q,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
    RotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        k_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        k,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
  } else {
    NeoXRotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        q_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        q,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
    NeoXRotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        k_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        k,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
  }
}

__global__ void GetPaddingOffset(int *d_token_num,
                                 int *padding_offset,
                                 int *cu_seqlens_data,
                                 const int *sequence_lengths,
                                 const int batch_size,
                                 const int max_seq_len) {
  // get padding offset of each batch
  int total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;
  cu_seqlens_data[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    const int seq_len = sequence_lengths[i];
    for (int j = 0; j < seq_len; j++) {
      padding_offset[index] = cum_offset;
      index++;
    }
    cum_offset += max_seq_len - seq_len;
    total_seq_len += seq_len;
    cu_seqlens_data[i + 1] = cu_seqlens_data[i] + seq_len;
  }
  d_token_num[0] = total_seq_len;
}

void InvokeGetPaddingOffset(const phi::GPUContext &dev_ctx,
                            int *h_token_num,
                            int *d_token_num,
                            int *padding_offset,
                            int *cu_seqlens_data,
                            const int *sequence_lengths,
                            const int batch_size,
                            const int max_seq_len) {
  GetPaddingOffset<<<1, 1, 0, dev_ctx.stream()>>>(d_token_num,
                                                  padding_offset,
                                                  cu_seqlens_data,
                                                  sequence_lengths,
                                                  batch_size,
                                                  max_seq_len);
  phi::memory_utils::Copy(phi::CPUPlace(),
                          h_token_num,
                          dev_ctx.GetPlace(),
                          d_token_num,
                          sizeof(int),
                          dev_ctx.stream());
}

template <typename T>
__global__ void RemovePadding(T *output_data,
                              const T *input_data,
                              const int *padding_offset,
                              const int dim_embed) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int src_seq_id = bid + padding_offset[bid];
  const int tgt_seq_id = bid;

  for (int i = tid; i < dim_embed; i += blockDim.x) {
    output_data[tgt_seq_id * dim_embed + i] =
        input_data[src_seq_id * dim_embed + i];
  }
}

template <typename T>
void InvokeRemovePadding(const phi::GPUContext &dev_ctx,
                         T *output_data,
                         const T *input_data,
                         const int *padding_offset,
                         const int token_num,
                         const int dim_embed) {
  RemovePadding<<<token_num, 256, 0, dev_ctx.stream()>>>(
      output_data, input_data, padding_offset, dim_embed);
}

template <typename T>
__global__ void RebuildPadding(T *output_data,
                               const T *input_data,
                               const int *padding_offset,
                               const int dim_embed) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int dst_seq_id = bid + padding_offset[bid];
  const int src_seq_id = bid;

  for (int i = tid; i < dim_embed; i += blockDim.x) {
    output_data[dst_seq_id * dim_embed + i] =
        input_data[src_seq_id * dim_embed + i];
  }
}

template <typename T>
void InvokeRebuildPadding(const phi::GPUContext &dev_ctx,
                          T *output_data,
                          const T *input_data,
                          const int *padding_offset,
                          const int token_num,
                          const int dim_embed) {
  // src: [token_num, dim_embed]
  // dst: [batch_size * max_seq_len, dim_embed]
  RebuildPadding<<<token_num, 256, 0, dev_ctx.stream()>>>(
      output_data, input_data, padding_offset, dim_embed);
}

template <typename T, int VecSize>
__global__ void InitOutValueKernel(T *output_data,
                                   const int64_t numel,
                                   const T init_value) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int64_t global_thread_idx = bid * blockDim.x + tid;

  for (int linear_index = global_thread_idx * VecSize,
           step = gridDim.x * blockDim.x * VecSize;
       linear_index < numel;
       linear_index += step) {
    for (int i = 0; i < VecSize; i++) {
      output_data[linear_index + i] = init_value;
    }
  }
}

template <typename T>
void InitValue(const phi::GPUContext &dev_ctx,
               T *output_data,
               const int64_t numel,
               const T init_value) {
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      numel % PackSize,
      0,
      common::errors::PreconditionNotMet(
          "numel=%d must be divisible by vec_size=%d", numel, PackSize));
  const int pack_num = numel / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  InitOutValueKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          output_data, numel, init_value);
}

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void ActFFNGlu(const T *bias,
                          Functor act_functor,
                          const int token_num,
                          const int hid_dim,
                          const int elem_num,
                          LoadFunc load_func,
                          StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec1;
  LoadT src_vec2;
  LoadT bias_vec1;
  LoadT bias_vec2;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int bi = i / hid_dim;
    int idx = i % hid_dim;
    // const T *input_this_thread = input + bi * hid_dim * 2;
    // T *output_this_thread = output + bi * hid_dim;
    // phi::Load<T, VecSize>(&input_this_thread[idx], &src_vec1);
    // phi::Load<T, VecSize>(&input_this_thread[idx + hid_dim], &src_vec2);

    load_func.template load<VecSize>(&src_vec1, bi * hid_dim * 2 + idx);
    load_func.template load<VecSize>(&src_vec2,
                                     bi * hid_dim * 2 + idx + hid_dim);

    if (bias) {
      phi::Load<T, VecSize>(&bias[idx], &bias_vec1);
      phi::Load<T, VecSize>(&bias[idx + hid_dim], &bias_vec2);
    }
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if (bias) {
        src_vec1[j] += bias_vec1[j];
        src_vec2[j] += bias_vec2[j];
      }
      src_vec1[j] = act_functor(src_vec1[j]);
      src_vec1[j] *= src_vec2[j];
    }
    // phi::Store<T, VecSize>(src_vec1, &output_this_thread[idx]);
    store_func.template store<VecSize>(src_vec1, bi * hid_dim + idx);
  }
}

template <typename T,
          typename Functor,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void LaunchActFFNGlu(const phi::GPUContext &dev_ctx,
                     const T *bias,
                     const int token_num,
                     const int hid_dim,
                     LoadFunc load_func,
                     StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int elem_cnt = token_num * hid_dim;
  const int blocksize = 128;
  int grid_size = 1;
  Functor functor;
  switch (hid_dim % PackSize) {
    case 0:
      GetNumBlocks(elem_cnt / PackSize, &grid_size);
      ActFFNGlu<T, Functor, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(bias,
                                                          functor,
                                                          token_num,
                                                          hid_dim,
                                                          elem_cnt,
                                                          load_func,
                                                          store_func);
      break;
    default:
      GetNumBlocks(elem_cnt, &grid_size);
      ActFFNGlu<T, Functor, 1><<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          bias, functor, token_num, hid_dim, elem_cnt, load_func, store_func);
      break;
  }
}

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void BiasAct(const T *bias,
                        Functor act_functor,
                        const int rows,
                        const int cols,
                        const int elem_num,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

// Zero Initialize BiasVec.
#pragma unroll
  for (int unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
    bias_vec[unroll_idx] = 0;
  }

  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int row_idx = i / cols;
    int col_idx = i % cols;
    int linear_idx = row_idx * cols + col_idx;
    // phi::Load<T, VecSize>(&input[linear_idx], &src_vec);
    load_func.template load<VecSize>(&src_vec, linear_idx);
    if (bias) {
      phi::Load<T, VecSize>(&bias[col_idx], &bias_vec);
    }
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if (bias) {
        src_vec[j] += bias_vec[j];
      }
      src_vec[j] = act_functor(src_vec[j]);
    }
    // phi::Store<T, VecSize>(src_vec, &output[linear_idx]);
    store_func.template store<VecSize>(src_vec, linear_idx);
  }
}

template <typename T,
          typename Functor,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void LaunchBiasAct(const phi::GPUContext &dev_ctx,
                   const T *bias,
                   const int token_num,
                   const int hid_dim,
                   LoadFunc load_func,
                   StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int elem_cnt = token_num * hid_dim;
  const int blocksize = 128;
  int grid_size = 1;
  Functor functor;
  switch (hid_dim % PackSize) {
    case 0:
      GetNumBlocks(elem_cnt / PackSize, &grid_size);
      BiasAct<T, Functor, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(bias,
                                                          functor,
                                                          token_num,
                                                          hid_dim,
                                                          elem_cnt,
                                                          load_func,
                                                          store_func);
      break;
    default:
      GetNumBlocks(elem_cnt, &grid_size);
      BiasAct<T, Functor, 1><<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          bias, functor, token_num, hid_dim, elem_cnt, load_func, store_func);
      break;
  }
}

template <typename T, int VecSize>
__global__ void fused_transpose_split_kernel(
    T *q_out,           // [total, num_head, head_dim]
    T *k_out,           // [total, num_head, head_dim]
    T *v_out,           // [total, num_head, head_dim]
    const T *q_input,   // [bsz, num_head, seq_len, head_dim]
    const T *kv_input,  // [2, bsz, num_head, seq_len, head_dim]
    const int *padding_offset,
    const int *seq_lens,
    const int32_t elem_cnt,
    const int batch_size,
    const int max_len_this_time,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int size_per_head) {
  const int32_t offset =
      batch_size * max_len_this_time * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

  int q_size = token_num * hidden_size;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    int32_t bias_idx = linear_index % fused_hidden_size;
    int32_t current_token = linear_index / fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;
    const int32_t seq_id = ori_token_idx % seq_len;

    // equal to:
    // const int qkv_id  = (linear_index % fused_hidden_size) / hidden_size;
    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    if (qkv_id == 0) {  // read q
      phi::Load<T, VecSize>(
          &q_input[target_batch_id * head_num * max_len_this_time *
                       size_per_head +
                   head_id * max_len_this_time * size_per_head +
                   seq_id * size_per_head + size_id],
          &src_vec);
    } else {  // read k/v
      const int32_t kv_store_offset = (qkv_id - 1) * offset;
      phi::Load<T, VecSize>(
          &kv_input[kv_store_offset +
                    target_batch_id * head_num * max_len_this_time *
                        size_per_head +
                    head_id * max_len_this_time * size_per_head +
                    seq_id * size_per_head + size_id],
          &src_vec);
    }
    int32_t write_index =
        linear_index - (qkv_id + 2 * current_token) * hidden_size;
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &q_out[write_index]);
    } else if (qkv_id == 1) {
      phi::Store<T, VecSize>(src_vec, &k_out[write_index]);
    } else if (qkv_id == 2) {
      phi::Store<T, VecSize>(src_vec, &v_out[write_index]);
    }
  }
}

template <typename T>
void TransposeSplit(const phi::GPUContext &dev_ctx,
                    T *q_out,
                    T *k_out,
                    T *v_out,
                    const T *q_input,
                    const T *kv_input,
                    const int *padding_offset,
                    const int *seq_lens,
                    const int token_num,
                    const int batch_size,
                    const int head_num,
                    const int max_len_this_time,
                    const int seq_len,
                    const int size_per_head) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    common::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  fused_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_out,
                                                      k_out,
                                                      v_out,
                                                      q_input,
                                                      kv_input,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      max_len_this_time,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head);
}

template <typename T>
void TransposeSplit(const phi::GPUContext &dev_ctx,
                    T *q_out,
                    T *k_out,
                    T *v_out,
                    const T *q_input,
                    const T *kv_input,
                    const int *padding_offset,
                    const int *seq_lens,
                    const int token_num,
                    const int batch_size,
                    const int head_num,
                    const int seq_len,
                    const int size_per_head) {
  TransposeSplit<T>(dev_ctx,
                    q_out,
                    k_out,
                    v_out,
                    q_input,
                    kv_input,
                    padding_offset,
                    seq_lens,
                    token_num,
                    batch_size,
                    head_num,
                    seq_len,
                    seq_len,
                    size_per_head);
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx =
        ori_bi * seq_len * last_dim + ori_seq_id * last_dim + h_bias;
    const int64_t bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[2 * i];
        const float sin_tmp = sin_emb_vec[2 * i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, bs, 1, seq_len, dim_head]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int rope_bsz) {
  const int elem_nums = token_num * 3 * head_num * dim_head;  // just q and k
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + rope_bsz * input_output_len * dim_head;

  VariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head);
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const int gqa_group_size) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  // const int hidden_size = num_head * last_dim;
  const int offset = (num_head + 2 * gqa_group_size) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx =
        ori_bi * seq_len * last_dim + ori_seq_id * last_dim + h_bias;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);

      if (hi < num_head + gqa_group_size) {  // qk rope
        const float cos_tmp = cos_emb_vec[2 * i];
        const float sin_tmp = sin_emb_vec[2 * i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int gqa_group_size,
    const int rope_bsz) {
  const int elem_nums =
      token_num * (head_num + 2 * gqa_group_size) * dim_head;  // for all q k v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + rope_bsz * input_output_len * dim_head;
  GQAVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head,
                                                      gqa_group_size);
}

}  // namespace

}  // namespace fusion
}  // namespace phi
