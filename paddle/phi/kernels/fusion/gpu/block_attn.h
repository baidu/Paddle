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

#include "paddle/common/flags.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/quant_dequant.h"
#include "paddle/phi/kernels/fusion/gpu/mmha_util.cu.h"

COMMON_DECLARE_bool(use_xqa_optim);

#ifdef PADDLE_WITH_HIP
#define GPU(str) hip##str
#define GPUMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#else
#define GPU(str) cuda##str
#define GPUMultiProcessorCount cudaDevAttrMultiProcessorCount
#endif

namespace phi {
namespace fusion {

constexpr int VEC_16B = 16;

template <typename T>
struct Block_AttN_params {
  // output buffer, [B, 1(seq_len), q_num_head * dim_head]
  T *out;
  // qkv_out, [B, 1(seq_len), (2 * kv_num_head + q_num_head) * dim_head]
  const T *qkv;
  // bias, [(2 * kv_num_head + q_num_head), dim_head]
  const T *qkv_bias;
  // [bsz, seq_len]
  const int *cum_offsets;
  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;

  // mask_length is the 3th dimension of attn_mask.
  int mask_length;
  bool mask_broadcast_num_heads;

  // k_cache [max_block_num, num_head, block_size, head_size]
  // v_cache [max_block_num, num_head, block_size, head_size]
  T *k_cache;
  T *v_cache;

  uint8_t *k_cache_I;
  uint8_t *v_cache_I;

  const int *block_tables;

  const int *sequence_lengths{nullptr};

  const float *rotary_emb = nullptr;
  int rotary_emb_dims;
  int rope_stride;
  float inv_compression_ratio = 1.0f;  // Default as 1.0

  int batch_size;  // batch * beam
  int beam_width = 1;
  int max_num_blocks_per_seq;
  int block_size;
  int q_num_head;
  int kv_num_head;
  int gqa_num_per_partitions;
  int timestep;
  int seq_len;

  int pre_cache_length = 0;

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;

  bool add_qkv_bias;
  bool neox_rotary_style;

  const float *cache_k_quant_scales = nullptr;
  const float *cache_v_quant_scales = nullptr;
  const float *cache_k_dequant_scales = nullptr;
  const float *cache_v_dequant_scales = nullptr;

  float rope_theta = 10000.0f;
};

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          int BLOCK_SIZE,
          CacheType CACHE_TYPE,
          typename LoadFunc,
          typename StoreFunc>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void block_attention_kernel(
    Block_AttN_params<T> params, LoadFunc load_func, StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__) || defined(PADDLE_WITH_HIP)
  const int bi = blockIdx.y;
  int act_time_step = params.sequence_lengths[bi];
  if (act_time_step == 0) {
    return;
  }

  act_time_step += params.pre_cache_length;

  const int *block_table =
      params.block_tables + bi * params.max_num_blocks_per_seq;

  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  extern __shared__ char smem_[];

  int block_smem_offset =
      div_up(params.max_num_blocks_per_seq, 4) * 4 * sizeof(int);
  float *qk_smem = reinterpret_cast<float *>(smem_ + block_smem_offset);

  float *logits_smem = reinterpret_cast<float *>(smem_ + block_smem_offset);

  /*
  Note: This shared memory is used in Final Reduce, so we do not
  need to add offset. Since in Final Reduce Phase, the block tables is not used.
  */
  T *out_smem = reinterpret_cast<T *>(smem_);

  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  using Qk_vec_RoPE = typename Qk_vec_RoPE_<T, float, Dh_MAX>::Type;
  using QK_Packed_Int8_t = typename Packed_Int8_<Qk_vec, CACHE_TYPE>::Type;

  // 每个 block 有一个 head 的 q 值
  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  const int tid = threadIdx.x;
  const int hi = blockIdx.x;  // head index
  const int kv_hi = hi / params.gqa_num_per_partitions;

  float k_quant_scale;
  float v_quant_scale;
  float k_dequant_scale;
  float v_dequant_scale;
  if (CACHE_TYPE == CacheType::INT8) {
    k_quant_scale = static_cast<float>(params.cache_k_quant_scales[kv_hi]);
    v_quant_scale = static_cast<float>(params.cache_v_quant_scales[kv_hi]);
    k_dequant_scale = static_cast<float>(params.cache_k_dequant_scales[kv_hi]);
    v_dequant_scale = static_cast<float>(params.cache_v_dequant_scales[kv_hi]);
  }

  const int bhi = bi * params.q_num_head + hi;
  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;
  const int thi = params.cum_offsets ? ti * params.q_num_head + hi : -1;
  int *block_table_smem = reinterpret_cast<int *>(smem_);

  for (int local_id = tid; local_id < params.max_num_blocks_per_seq;
       local_id += blockDim.x) {
    block_table_smem[local_id] = block_table[local_id];
  }
  __syncthreads();

  float qk_max = -FLT_MAX;
  float qk = 0;

  const int block_idx = act_time_step / BLOCK_SIZE;
  const int block_offset = act_time_step % BLOCK_SIZE;
  const int physical_block_number = block_table_smem[block_idx];

  // cache offset of current token
  const int base_cache_offset =
      physical_block_number * params.kv_num_head * BLOCK_SIZE * Dh +
      kv_hi * BLOCK_SIZE * Dh + block_offset * Dh;

  // qkv [B, S=1, num_head + 2 * (kv_num_head), head_dim]
  int qkv_base_offset = bi * (params.q_num_head + params.kv_num_head * 2) * Dh;

  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

  // cache_k, [B, num_head, head_dim / x, max_seq_len, x]
  // x == 4/8 for FP32/FP16, 128bit, 16Byte
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  const T *q_bias_base = nullptr;
  const T *k_bias_base = nullptr;

  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + params.q_num_head * Dh;
  }
  // Load the current timestep's Q and K, then compute q*k,
  // with each block computing one head.
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
          k, params.q_num_head * Dh + qk_offset + kv_hi * Dh);
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

    if (params.rotary_emb_dims != 0) {
      if (!params.neox_rotary_style) {
        apply_rotary_embedding(q,
                               k,
                               tid,
                               Dh,
                               act_time_step - params.pre_cache_length,
                               params.inv_compression_ratio,
                               params.rope_theta);
      } else {
        int last_dim = Dh / params.rotary_emb_dims;
        int half_lastdim = last_dim / 2;
        int rotary_offset = act_time_step * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rope_stride;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = tid / stride_all_lastdim * stride_all_lastdim +
                       (tid + stride) % (stride_all_lastdim);
        int q_right_offset = qkv_base_offset + hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_offset = qkv_base_offset + params.q_num_head * Dh +
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
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      if (CACHE_TYPE == CacheType::INT8) {
        const int offset = base_cache_offset + tid * QK_VEC_SIZE;
        QK_Packed_Int8_t k_tmp =
            round_tmp<QK_Packed_Int8_t, Qk_vec, CACHE_TYPE>(
                mul<Qk_vec, float, Qk_vec>(k_quant_scale, k));
        *reinterpret_cast<QK_Packed_Int8_t *>(&params.k_cache_I[offset]) =
            k_tmp;
      } else {
        const int offset = base_cache_offset + tid * QK_VEC_SIZE;
        *reinterpret_cast<Qk_vec *>(&params.k_cache[offset]) = k;
      }
    }

    qk = dot<Qk_vec, Qk_vec>(q, k);

    if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
      for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
#ifdef PADDLE_WITH_HIP
        qk += __shfl_xor(qk, mask);
#else
        qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
#endif
      }
    }
  }
  if (QK_VECS_PER_WARP > WARP_SIZE) {
    constexpr int WARPS_PER_RED =
        (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
    qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
  }
  if (tid == 0) {
    qk *= params.inv_sqrt_dh;
    if (params.attn_mask) {
      auto mask_bhi = bhi;
      if (params.mask_broadcast_num_heads) {
        mask_bhi = bi;
      }
      T mask = params.attn_mask[mask_bhi * params.mask_length + act_time_step];
      qk += static_cast<float>(mask);
    }
    qk_max = qk;
    qk_smem[act_time_step] = qk;
  }
  __syncthreads();
  using K_vec = typename K_vec_bttn_<T>::Type;
  using K_vec_I = typename K_vec_I_bttn_<T, CACHE_TYPE>::Type;
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

  // threads in one block can process 'K_PER_ITER' keys
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  // threads in one warp can process 'K_PER_ITER' keys
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  K_vec k[K_VECS_PER_THREAD];
  K_vec k_vec_zero;
  zero(k_vec_zero);
  // ti is the timestep index
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    const int physical_block_number = block_table_smem[ti / BLOCK_SIZE];
    const int block_offset = ti % BLOCK_SIZE;
    const int k_offset =
        physical_block_number * params.kv_num_head * BLOCK_SIZE * Dh +
        kv_hi * BLOCK_SIZE * Dh + block_offset * Dh + ki;
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      if (ti < act_time_step) {
        if (CACHE_TYPE == CacheType::INT8) {
          mul_pointer_v2<K_vec, float, K_vec_I, CACHE_TYPE>(
              &k[ii],
              k_dequant_scale,
              reinterpret_cast<K_vec_I *>(params.k_cache_I + k_offset +
                                          ii * THREADS_PER_KEY * K_VEC_SIZE));
        } else {
          k[ii] = (Dh == Dh_MAX || ii * THREADS_PER_KEY * K_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const K_vec *>(
                            params.k_cache + k_offset +
                            ii * THREADS_PER_KEY * K_VEC_SIZE)
                      : k_vec_zero;
        }
      } else {
        k[ii] = k_vec_zero;
      }
    }

    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

    if (params.attn_mask) {
      auto mask_bhi = bhi;
      if (params.mask_broadcast_num_heads) {
        mask_bhi = bi;
      }
      T mask = params.attn_mask[mask_bhi * params.mask_length + ti];
      qk += static_cast<float>(mask);
    }

    if (ti < act_time_step && tid % THREADS_PER_KEY == 0) {
      qk_max = fmaxf(qk_max, qk);
      qk_smem[ti] = qk;
    }
  }

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
#ifdef PADDLE_WITH_HIP
    qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
#else
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
#endif
  }

  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;

  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  __syncthreads();

#ifdef PADDLE_WITH_HIP
  qk_max = -FLT_MAX;
  qk_max = red_smem[lane];
#else
  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#endif

#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
#ifdef PADDLE_WITH_HIP
    qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
#else
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
#endif
  }

#ifdef PADDLE_WITH_HIP
  qk_max = __shfl(qk_max, 0);
#else
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);
#endif

  float sum = 0.f;
  for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  float inv_sum = __fdividef(1.f, sum + 1.e-6f);

  for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }
  __syncthreads();

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;
  using V_Packed_Int8_t = typename Packed_Int8_<V_vec, CACHE_TYPE>::Type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

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
      int physical_block_number = block_table_smem[ti / BLOCK_SIZE];

      const int block_offset = ti % BLOCK_SIZE;
      const int v_offset =
          physical_block_number * params.kv_num_head * BLOCK_SIZE * Dh +
          kv_hi * BLOCK_SIZE * Dh + block_offset * Dh + vi;
      V_vec v;
      if (CACHE_TYPE == CacheType::INT8) {
        mul_pointer_v2<V_vec, float, V_Packed_Int8_t, CACHE_TYPE>(
            &v,
            v_dequant_scale,
            reinterpret_cast<V_Packed_Int8_t *>(params.v_cache_I + v_offset));
      } else {
        v = *reinterpret_cast<const V_vec *>(params.v_cache + v_offset);
      }
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

  V_vec v_bias;
  zero(v_bias);
  if (vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v;
    load_func.template load<V_vec>(
        v,
        (params.q_num_head + params.kv_num_head) * Dh + qkv_base_offset +
            kv_hi * Dh + vi);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[(params.q_num_head + params.kv_num_head) * Dh +
                           kv_hi * Dh + vi]);
      v = add(v, v_bias);
    }

    if (CACHE_TYPE == CacheType::INT8) {
      V_Packed_Int8_t v_tmp = round_tmp<V_Packed_Int8_t, V_vec, CACHE_TYPE>(
          mul<V_vec, float, V_vec>(v_quant_scale, v));
      *reinterpret_cast<V_Packed_Int8_t *>(params.v_cache_I +
                                           base_cache_offset + vi) = v_tmp;
    } else {
      *reinterpret_cast<V_vec *>(params.v_cache + base_cache_offset + vi) = v;
    }

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
#else
  assert(false);
#endif
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          int BLOCK_SIZE,
          CacheType CACHE_TYPE,
          int GQA_PARTITION_SIZE,
          int GQA_SUB_PARTITION_SIZE,
          int GQA_NUM_SUB_PARTITIONS,
          typename LoadFunc,
          typename StoreFunc>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void gqa_block_attention_kernel(
    Block_AttN_params<T> params, LoadFunc load_func, StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__) || defined(PADDLE_WITH_HIP)
  const int bi = blockIdx.y;
  const int act_time_step = params.sequence_lengths[bi];
  if (act_time_step == 0) {
    return;
  }

  const int *block_table =
      params.block_tables + bi * params.max_num_blocks_per_seq;

  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  extern __shared__ char smem_[];

  int block_smem_offset =
      div_up(params.max_num_blocks_per_seq, 4) * 4 * sizeof(int);
  int q_smem_offset =
      div_up(Dh_MAX * GQA_SUB_PARTITION_SIZE, 4) * 4 * sizeof(T);

  T *q_smem = reinterpret_cast<T *>(smem_ + block_smem_offset);
  float *qk_smem =
      reinterpret_cast<float *>(smem_ + block_smem_offset + q_smem_offset);
  float *logits_smem =
      reinterpret_cast<float *>(smem_ + block_smem_offset + q_smem_offset);

  /*
  Note: This shared memory is used in Final Reduce, so we do not
  need to add offset. Since in Final Reduce Phase, the block tables is not used.
  */
  T *out_smem = reinterpret_cast<T *>(smem_);

  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  using Qk_vec_RoPE = typename Qk_vec_RoPE_<T, float, Dh_MAX>::Type;
  using QK_Packed_Int8_t = typename Packed_Int8_<Qk_vec, CACHE_TYPE>::Type;

  const int tid = threadIdx.x;
  const int kv_hi = blockIdx.x / GQA_NUM_SUB_PARTITIONS;
  const int gqa_sub_partition_id = blockIdx.x % GQA_NUM_SUB_PARTITIONS;

  float k_quant_scale;
  float v_quant_scale;
  float k_dequant_scale;
  float v_dequant_scale;
  if (CACHE_TYPE == CacheType::INT8) {
    k_quant_scale = static_cast<float>(params.cache_k_quant_scales[kv_hi]);
    v_quant_scale = static_cast<float>(params.cache_v_quant_scales[kv_hi]);
    k_dequant_scale = static_cast<float>(params.cache_k_dequant_scales[kv_hi]);
    v_dequant_scale = static_cast<float>(params.cache_v_dequant_scales[kv_hi]);
  }

  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;

  int *block_table_smem = reinterpret_cast<int *>(smem_);

  for (int local_id = tid; local_id < params.max_num_blocks_per_seq;
       local_id += blockDim.x) {
    block_table_smem[local_id] = block_table[local_id];
  }

  __syncthreads();

  float qk_max = -FLT_MAX;
  float qk = 0;

  const int block_idx = act_time_step / BLOCK_SIZE;
  const int block_offset = act_time_step % BLOCK_SIZE;
  const int physical_block_number = block_table_smem[block_idx];

  // cache offset of current token
  const int base_cache_offset =
      physical_block_number * params.kv_num_head * BLOCK_SIZE * Dh +
      kv_hi * BLOCK_SIZE * Dh + block_offset * Dh;

  // qkv [B, S=1, num_head + 2 * (kv_num_head), head_dim]
  int qkv_base_offset = bi * (params.q_num_head + params.kv_num_head * 2) * Dh;

  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

  // cache_k, [B, num_head, head_dim / x, max_seq_len, x]
  // x == 4/8 for FP32/FP16, 128bit, 16Byte
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  const T *q_bias_base = nullptr;
  const T *k_bias_base = nullptr;

  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + params.q_num_head * Dh;
  }
  // Load the current timestep's Q and K, then compute q*k,
  // GQA use one warp calculate one head
  assert(QK_VECS_PER_WARP < WARP_SIZE);
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  if (warp_id < GQA_SUB_PARTITION_SIZE && lane_id < QK_VECS_PER_WARP) {
    const int hi = kv_hi * GQA_PARTITION_SIZE +
                   gqa_sub_partition_id * GQA_SUB_PARTITION_SIZE + warp_id;
    const int qk_offset = qkv_base_offset + lane_id * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    if (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset + hi * Dh);
    }

    Qk_vec k;
    zero(k);
    if (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(
          k, params.q_num_head * Dh + qk_offset + kv_hi * Dh);
    }

    if (params.add_qkv_bias) {
      const int q_bias_offset = hi * Dh + lane_id * QK_VEC_SIZE;
      const int k_bias_offset = kv_hi * Dh + lane_id * QK_VEC_SIZE;
      Qk_vec q_bias;
      zero(q_bias);
      Qk_vec k_bias;
      zero(k_bias);

      q_bias =
          (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&q_bias_base[q_bias_offset])
              : q_bias;
      k_bias =
          (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&k_bias_base[k_bias_offset])
              : k_bias;

      q = add(q, q_bias);
      k = add(k, k_bias);
    }

    if (params.rotary_emb_dims != 0) {
      if (!params.neox_rotary_style) {
        apply_rotary_embedding(q,
                               k,
                               lane_id,
                               Dh,
                               act_time_step,
                               params.inv_compression_ratio,
                               params.rope_theta);
      } else {
        int last_dim = Dh / params.rotary_emb_dims;
        int half_lastdim = last_dim / 2;
        int rotary_offset = act_time_step * Dh + lane_id * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rope_stride;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = lane_id / stride_all_lastdim * stride_all_lastdim +
                       (lane_id + stride) % (stride_all_lastdim);
        int q_right_offset = qkv_base_offset + hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_offset = qkv_base_offset + params.q_num_head * Dh +
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
        cos_emb = (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;

        Qk_vec_RoPE sin_emb;
        zero(sin_emb);
        sin_emb = (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        float alpha = (lane_id % stride_all_lastdim) < stride
                          ? static_cast<float>(-1)
                          : static_cast<float>(1);
        q = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            q, q_right, cos_emb, sin_emb, alpha);
        k = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            k, k_right, cos_emb, sin_emb, alpha);
      }
    }
    *reinterpret_cast<Qk_vec *>(
        &q_smem[warp_id * Dh_MAX + lane_id * QK_VEC_SIZE]) = q;

    if (Dh == Dh_MAX || lane_id * QK_VEC_SIZE < Dh) {
      if (CACHE_TYPE == CacheType::INT8) {
        const int offset = base_cache_offset + lane_id * QK_VEC_SIZE;
        QK_Packed_Int8_t k_tmp =
            round_tmp<QK_Packed_Int8_t, Qk_vec, CACHE_TYPE>(
                mul<Qk_vec, float, Qk_vec>(k_quant_scale, k));
        *reinterpret_cast<QK_Packed_Int8_t *>(&params.k_cache_I[offset]) =
            k_tmp;
      } else {
        const int offset = base_cache_offset + lane_id * QK_VEC_SIZE;
        *reinterpret_cast<Qk_vec *>(&params.k_cache[offset]) = k;
      }
    }

    qk = dot<Qk_vec, Qk_vec>(q, k);

    if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
      for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
#ifdef PADDLE_WITH_HIP
        qk += __shfl_xor(qk, mask);
#else
        qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
#endif
      }
    }
  }

  if (lane_id == 0) {
    qk *= params.inv_sqrt_dh;
    const int hi = kv_hi * GQA_PARTITION_SIZE +
                   gqa_sub_partition_id * GQA_SUB_PARTITION_SIZE + warp_id;
    const int bhi = bi * params.q_num_head + hi;
    if (params.attn_mask) {
      auto mask_bhi = bhi;
      if (params.mask_broadcast_num_heads) {
        mask_bhi = bi;
      }
      T mask = params.attn_mask[mask_bhi * params.mask_length + act_time_step];
      qk += static_cast<float>(mask);
    }

    qk_max = qk;
    qk_smem[act_time_step * GQA_SUB_PARTITION_SIZE + warp_id] = qk;
  }
  __syncthreads();
  using K_vec = typename K_vec_bttn_<T>::Type;
  using K_vec_I = typename K_vec_I_bttn_<T, CACHE_TYPE>::Type;
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
  constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  int ko = tid / THREADS_PER_KEY;
  int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;

  static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD, "");

  //   K_vec q[K_VECS_PER_THREAD];
  // #pragma unroll
  //   for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
  //     q[i] = *reinterpret_cast<const K_vec *>(
  //         &q_smem[ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
  //   }

  float qk_maxs[GQA_SUB_PARTITION_SIZE];
#pragma unroll
  for (int i = 0; i < GQA_SUB_PARTITION_SIZE; i++) {
    // qk_maxs[i] = -FLT_MAX;
    // initialize qk_maxs!!!
    qk_maxs[i] = qk_smem[act_time_step * GQA_SUB_PARTITION_SIZE + i];
  }

  // threads in one block can process 'K_PER_ITER' keys
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  // threads in one warp can process 'K_PER_ITER' keys
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  K_vec k[K_VECS_PER_THREAD];
  K_vec k_vec_zero;
  zero(k_vec_zero);
  // ti is the timestep index
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    int physical_block_number = block_table_smem[ti / BLOCK_SIZE];

    const int block_offset = ti % BLOCK_SIZE;
    const int k_offset =
        physical_block_number * params.kv_num_head * BLOCK_SIZE * Dh +
        kv_hi * BLOCK_SIZE * Dh + block_offset * Dh + ki;
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      if (ti < act_time_step) {
        if (CACHE_TYPE == CacheType::INT8) {
          mul_pointer_v2<K_vec, float, K_vec_I, CACHE_TYPE>(
              &k[ii],
              k_dequant_scale,
              reinterpret_cast<K_vec_I *>(params.k_cache_I + k_offset +
                                          ii * THREADS_PER_KEY * K_VEC_SIZE));
        } else {
          k[ii] = (Dh == Dh_MAX || ii * THREADS_PER_KEY * K_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const K_vec *>(
                            params.k_cache + k_offset +
                            ii * THREADS_PER_KEY * K_VEC_SIZE)
                      : k_vec_zero;
        }
      } else {
        k[ii] = k_vec_zero;
      }
    }
#pragma unroll
    for (int local_hi = 0; local_hi < GQA_SUB_PARTITION_SIZE; local_hi++) {
      const int hi = kv_hi * GQA_PARTITION_SIZE +
                     gqa_sub_partition_id * GQA_SUB_PARTITION_SIZE + local_hi;
      K_vec q[K_VECS_PER_THREAD];
#pragma unroll
      for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
        q[i] = *reinterpret_cast<const K_vec *>(
            &q_smem[local_hi * Dh_MAX + ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
      }

      float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

      if (params.attn_mask) {
        const int bhi = bi * params.q_num_head + hi;
        auto mask_bhi = bhi;
        if (params.mask_broadcast_num_heads) {
          mask_bhi = bi;
        }
        T mask = params.attn_mask[mask_bhi * params.mask_length + ti];
        qk += static_cast<float>(mask);
      }

      if (ti < act_time_step && tid % THREADS_PER_KEY == 0) {
        qk_maxs[local_hi] = fmaxf(qk_maxs[local_hi], qk);
        qk_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi] = qk;
      }
    }
  }

#pragma unroll
  for (int local_hi = 0; local_hi < GQA_SUB_PARTITION_SIZE; local_hi++) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
      qk_maxs[local_hi] = fmaxf(qk_maxs[local_hi],
#ifdef PADDLE_WITH_HIP
                                __shfl_xor(qk_maxs[local_hi], mask));
#else
                                __shfl_xor_sync(
                                    uint32_t(-1), qk_maxs[local_hi], mask));

#endif
    }

    if (lane_id == 0) {
      red_smem[warp_id] = qk_maxs[local_hi];
    }

    __syncthreads();

#ifdef PADDLE_WITH_HIP
    qk_maxs[local_hi] = -FLT_MAX;
    qk_maxs[local_hi] = red_smem[lane_id];
#else
    qk_maxs[local_hi] =
        lane_id < WARPS_PER_BLOCK ? red_smem[lane_id] : -FLT_MAX;
#endif

#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
      qk_maxs[local_hi] = fmaxf(qk_maxs[local_hi],
#ifdef PADDLE_WITH_HIP
                                __shfl_xor(qk_maxs[local_hi], mask));
#else
                                __shfl_xor_sync(
                                    uint32_t(-1), qk_maxs[local_hi], mask));
#endif
    }

#ifdef PADDLE_WITH_HIP
    qk_maxs[local_hi] = __shfl(qk_maxs[local_hi], 0);
#else
    qk_maxs[local_hi] = __shfl_sync(uint32_t(-1), qk_maxs[local_hi], 0);
#endif

    float sum = 0.f;
    for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
      float logit = __expf(qk_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi] -
                           qk_maxs[local_hi]);
      sum += logit;
      qk_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi] = logit;
    }

    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    float inv_sum = __fdividef(1.f, sum + 1.e-6f);

    for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
      convert_from_float(
          logits_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi],
          qk_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi] * inv_sum);
    }
  }  // for (int local_hi
  __syncthreads();

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;
  using V_Packed_Int8_t = typename Packed_Int8_<V_vec, CACHE_TYPE>::Type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif

  V_vec_acum out[GQA_SUB_PARTITION_SIZE];
#pragma unroll
  for (int i = 0; i < GQA_SUB_PARTITION_SIZE; i++) {
    zero(out[i]);
  }

  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if (Dh == Dh_MAX || vi < Dh) {
    for (int ti = vo; ti < act_time_step; ti += V_PER_ITER) {
      int physical_block_number = block_table_smem[ti / BLOCK_SIZE];

      const int block_offset = ti % BLOCK_SIZE;
      const int v_offset =
          physical_block_number * params.kv_num_head * BLOCK_SIZE * Dh +
          kv_hi * BLOCK_SIZE * Dh + block_offset * Dh + vi;
      V_vec v;
      if (CACHE_TYPE == CacheType::INT8) {
        mul_pointer_v2<V_vec, float, V_Packed_Int8_t, CACHE_TYPE>(
            &v,
            v_dequant_scale,
            reinterpret_cast<V_Packed_Int8_t *>(params.v_cache_I + v_offset));
      } else {
        v = *reinterpret_cast<const V_vec *>(params.v_cache + v_offset);
      }
#pragma unroll
      for (int local_hi = 0; local_hi < GQA_SUB_PARTITION_SIZE; local_hi++) {
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        float logit = logits_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi];
        out[local_hi] = fma(logit, cast_to_float(v), out[local_hi]);
#else
        DataType_ logit = static_cast<DataType_>(
            logits_smem[ti * GQA_SUB_PARTITION_SIZE + local_hi]);
        // Update the partial sums.
        out[local_hi] = fma(logit, v, out[local_hi]);
#endif
      }
    }
  }

  V_vec v_bias;
  zero(v_bias);
  if (vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v;
    // 读取当前的 v 到 v cache 中
    load_func.template load<V_vec>(
        v,
        (params.q_num_head + params.kv_num_head) * Dh + qkv_base_offset +
            kv_hi * Dh + vi);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[(params.q_num_head + params.kv_num_head) * Dh +
                           kv_hi * Dh + vi]);
      v = add(v, v_bias);
    }

    if (CACHE_TYPE == CacheType::INT8) {
      V_Packed_Int8_t v_tmp = round_tmp<V_Packed_Int8_t, V_vec, CACHE_TYPE>(
          mul<V_vec, float, V_vec>(v_quant_scale, v));
      *reinterpret_cast<V_Packed_Int8_t *>(params.v_cache_I +
                                           base_cache_offset + vi) = v_tmp;
    } else {
      *reinterpret_cast<V_vec *>(params.v_cache + base_cache_offset + vi) = v;
    }

#pragma unroll
    for (int local_hi = 0; local_hi < GQA_SUB_PARTITION_SIZE; local_hi++) {
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      out[local_hi] =
          fma(logits_smem[act_time_step * GQA_SUB_PARTITION_SIZE + local_hi],
              cast_to_float(v),
              out[local_hi]);
#else
      out[local_hi] =
          fma(logits_smem[act_time_step * GQA_SUB_PARTITION_SIZE + local_hi],
              v,
              out[local_hi]);
#endif
    }
  }

  __syncthreads();

  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int local_hi = 0; local_hi < GQA_SUB_PARTITION_SIZE; local_hi++) {
#pragma unroll
      for (int active_groups = V_PER_ITER; active_groups >= 2;
           active_groups /= 2) {
        int midpoint = active_groups / 2;

        if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
          convert_from_float(
              *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]),
              out[local_hi]);
#else
          *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]) =
              out[local_hi];
#endif
        }
        __syncthreads();
        if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
          out[local_hi] =
              add(*reinterpret_cast<const V_vec *>(&out_smem[vo * Dh + vi]),
                  out[local_hi]);
        }
        __syncthreads();
      }
    }  // for (int local_hi = 0
  }

  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#pragma unroll
    for (int local_hi = 0; local_hi < GQA_SUB_PARTITION_SIZE; local_hi++) {
      const int hi = kv_hi * GQA_PARTITION_SIZE +
                     gqa_sub_partition_id * GQA_SUB_PARTITION_SIZE + local_hi;
      const int ti = params.cum_offsets
                         ? bi * params.seq_len - params.cum_offsets[bi]
                         : -1;
      const int thi = params.cum_offsets ? ti * params.q_num_head + hi : -1;
      const int bhi = bi * params.q_num_head + hi;
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      V_vec tmp_out;
      convert_from_float(tmp_out, out[local_hi]);
      store_func.template store<V_vec>(
          tmp_out, thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#else
      store_func.template store<V_vec>(
          out[local_hi], thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#endif
    }
  }
#else
  assert(false);
#endif
}

template <typename T>
inline size_t smem_size_in_bytes(const Block_AttN_params<T> &params,
                                 int dim_head,
                                 int threads_per_value,
                                 int threads_per_block) {
  size_t qk_sz = div_up(params.timestep + 1, 4) * 4 * sizeof(int);

  size_t block_table_sz =
      div_up(params.max_num_blocks_per_seq, 4) * 4 * sizeof(int);
  size_t cache_scale_sz = 0;

  size_t logits_table_sz = qk_sz + block_table_sz + cache_scale_sz;

  int rows_per_red = threads_per_block / threads_per_value;

  // Note: Because half threads in ThreadBlock use out_smem to do reduce.
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;

  return max(logits_table_sz, red_sz);
}

template <typename T, int GQA_SUB_PARTITION_SIZE>
inline size_t gqa_smem_size_in_bytes(const Block_AttN_params<T> &params,
                                     int dim_head,
                                     int threads_per_value,
                                     int threads_per_block) {
  size_t qk_sz = div_up(dim_head * GQA_SUB_PARTITION_SIZE +
                            params.timestep * GQA_SUB_PARTITION_SIZE + 1,
                        4) *
                 4 * sizeof(int);
  size_t block_table_sz =
      div_up(params.max_num_blocks_per_seq, 4) * 4 * sizeof(int);
  size_t logits_table_sz = qk_sz + block_table_sz;

  int rows_per_red = threads_per_block / threads_per_value;
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;

  return max(logits_table_sz, red_sz);
}

#ifdef PADDLE_WITH_HIP
#define BLHAG_LAUNCH_KERNEL(T,                                             \
                            Dh,                                            \
                            Dh_MAX,                                        \
                            THDS_PER_KEY,                                  \
                            THDS_PER_VALUE,                                \
                            THDS_PER_BLOCK,                                \
                            BLOCK_SIZE,                                    \
                            CACHE_TYPE,                                    \
                            stream,                                        \
                            load_func,                                     \
                            store_func)                                    \
  size_t smem_sz =                                                         \
      smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);   \
  constexpr auto kernel_fn = block_attention_kernel<T,                     \
                                                    Dh,                    \
                                                    Dh_MAX,                \
                                                    THDS_PER_KEY,          \
                                                    THDS_PER_VALUE,        \
                                                    THDS_PER_BLOCK,        \
                                                    BLOCK_SIZE,            \
                                                    CACHE_TYPE,            \
                                                    decltype(load_func),   \
                                                    decltype(store_func)>; \
  if (smem_sz > 0xc000) {                                                  \
    hipFuncSetAttribute((const void *)kernel_fn,                           \
                        hipFuncAttributeMaxDynamicSharedMemorySize,        \
                        smem_sz);                                          \
  }                                                                        \
  dim3 grid(params.q_num_head, params.batch_size);                         \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                    \
      params, load_func, store_func);

// For GQA specific optimization (use xqa)
#define BLHA_LAUNCH_GQA_KERNEL(T,                                           \
                               Dh,                                          \
                               Dh_MAX,                                      \
                               THDS_PER_KEY,                                \
                               THDS_PER_VALUE,                              \
                               THDS_PER_BLOCK,                              \
                               BLOCK_SIZE,                                  \
                               CACHE_TYPE,                                  \
                               GQA_PARTITION_SIZE,                          \
                               GQA_SUB_PARTITION_SIZE,                      \
                               stream,                                      \
                               load_func,                                   \
                               store_func)                                  \
  size_t smem_sz = gqa_smem_size_in_bytes<T, GQA_SUB_PARTITION_SIZE>(       \
      params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);                          \
  constexpr int GQA_NUM_SUB_PARTITIONS =                                    \
      GQA_PARTITION_SIZE / GQA_SUB_PARTITION_SIZE;                          \
  constexpr auto kernel_fn =                                                \
      gqa_block_attention_kernel<T,                                         \
                                 Dh,                                        \
                                 Dh_MAX,                                    \
                                 THDS_PER_KEY,                              \
                                 THDS_PER_VALUE,                            \
                                 THDS_PER_BLOCK,                            \
                                 BLOCK_SIZE,                                \
                                 CACHE_TYPE,                                \
                                 GQA_PARTITION_SIZE,                        \
                                 GQA_SUB_PARTITION_SIZE,                    \
                                 GQA_NUM_SUB_PARTITIONS,                    \
                                 decltype(load_func),                       \
                                 decltype(store_func)>;                     \
  if (smem_sz > 0xc000) {                                                   \
    hipFuncSetAttribute((const void *)kernel_fn,                            \
                        hipFuncAttributeMaxDynamicSharedMemorySize,         \
                        smem_sz);                                           \
  }                                                                         \
  dim3 grid(params.kv_num_head *GQA_NUM_SUB_PARTITIONS, params.batch_size); \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                     \
      params, load_func, store_func);
#else
#define BLHAG_LAUNCH_KERNEL(T,                                             \
                            Dh,                                            \
                            Dh_MAX,                                        \
                            THDS_PER_KEY,                                  \
                            THDS_PER_VALUE,                                \
                            THDS_PER_BLOCK,                                \
                            BLOCK_SIZE,                                    \
                            CACHE_TYPE,                                    \
                            stream,                                        \
                            load_func,                                     \
                            store_func)                                    \
  size_t smem_sz =                                                         \
      smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);   \
  constexpr auto kernel_fn = block_attention_kernel<T,                     \
                                                    Dh,                    \
                                                    Dh_MAX,                \
                                                    THDS_PER_KEY,          \
                                                    THDS_PER_VALUE,        \
                                                    THDS_PER_BLOCK,        \
                                                    BLOCK_SIZE,            \
                                                    CACHE_TYPE,            \
                                                    decltype(load_func),   \
                                                    decltype(store_func)>; \
  if (smem_sz > 0xc000) {                                                  \
    cudaFuncSetAttribute(                                                  \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);  \
  }                                                                        \
  dim3 grid(params.q_num_head, params.batch_size);                         \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                    \
      params, load_func, store_func);

// For GQA specific optimization (use xqa)
#define BLHA_LAUNCH_GQA_KERNEL(T,                                           \
                               Dh,                                          \
                               Dh_MAX,                                      \
                               THDS_PER_KEY,                                \
                               THDS_PER_VALUE,                              \
                               THDS_PER_BLOCK,                              \
                               BLOCK_SIZE,                                  \
                               CACHE_TYPE,                                  \
                               GQA_PARTITION_SIZE,                          \
                               GQA_SUB_PARTITION_SIZE,                      \
                               stream,                                      \
                               load_func,                                   \
                               store_func)                                  \
  size_t smem_sz = gqa_smem_size_in_bytes<T, GQA_SUB_PARTITION_SIZE>(       \
      params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);                          \
  constexpr int GQA_NUM_SUB_PARTITIONS =                                    \
      GQA_PARTITION_SIZE / GQA_SUB_PARTITION_SIZE;                          \
  constexpr auto kernel_fn =                                                \
      gqa_block_attention_kernel<T,                                         \
                                 Dh,                                        \
                                 Dh_MAX,                                    \
                                 THDS_PER_KEY,                              \
                                 THDS_PER_VALUE,                            \
                                 THDS_PER_BLOCK,                            \
                                 BLOCK_SIZE,                                \
                                 CACHE_TYPE,                                \
                                 GQA_PARTITION_SIZE,                        \
                                 GQA_SUB_PARTITION_SIZE,                    \
                                 GQA_NUM_SUB_PARTITIONS,                    \
                                 decltype(load_func),                       \
                                 decltype(store_func)>;                     \
  if (smem_sz > 0xc000) {                                                   \
    cudaFuncSetAttribute(                                                   \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);   \
  }                                                                         \
  dim3 grid(params.kv_num_head *GQA_NUM_SUB_PARTITIONS, params.batch_size); \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                     \
      params, load_func, store_func);
#endif

template <typename T,
          int Dh,
          int Dh_MAX,
          int BlockSize,
          int THREADS_PER_VALUE,
          int THREADS_PER_KEY,
          int THREADS_PER_BLOCK,
          CacheType CACHE_TYPE,
          typename LoadFunc,
          typename StoreFunc>
void dispatch_blha_impl_kernel(const Block_AttN_params<T> &params,
                               const GPU(Stream_t) & stream,
                               LoadFunc load_func,
                               StoreFunc store_func) {
  VLOG(1) << "group wise";
  BLHAG_LAUNCH_KERNEL(T,
                      Dh,
                      Dh_MAX,
                      THREADS_PER_KEY,
                      THREADS_PER_VALUE,
                      THREADS_PER_BLOCK,
                      BlockSize,
                      CACHE_TYPE,
                      stream,
                      load_func,
                      store_func)
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int BlockSize,
          int THREADS_PER_VALUE,
          int THREADS_PER_KEY,
          int THREADS_PER_BLOCK,
          CacheType CACHE_TYPE,
          typename LoadFunc,
          typename StoreFunc>
void dispatch_blha_gqa_kernel(const Block_AttN_params<T> &params,
                              const GPU(Stream_t) & stream,
                              LoadFunc load_func,
                              StoreFunc store_func) {
  if (params.gqa_num_per_partitions == 1 || !FLAGS_use_xqa_optim) {
    dispatch_blha_impl_kernel<T,
                              Dh,
                              Dh_MAX,
                              BlockSize,
                              THREADS_PER_VALUE,
                              THREADS_PER_KEY,
                              THREADS_PER_BLOCK,
                              CACHE_TYPE>(
        params, stream, load_func, store_func);
  } else if (params.gqa_num_per_partitions == 2) {
    constexpr int THDS_PER_BLOCK = 1024;
    BLHA_LAUNCH_GQA_KERNEL(T,
                           Dh,
                           Dh_MAX,
                           THREADS_PER_KEY,
                           THREADS_PER_VALUE,
                           THDS_PER_BLOCK,
                           BlockSize,
                           CACHE_TYPE,
                           2,
                           2,
                           stream,
                           load_func,
                           store_func)
  } else if (params.gqa_num_per_partitions == 4) {
    constexpr int THDS_PER_BLOCK = 1024;
    BLHA_LAUNCH_GQA_KERNEL(T,
                           Dh,
                           Dh_MAX,
                           THREADS_PER_KEY,
                           THREADS_PER_VALUE,
                           THDS_PER_BLOCK,
                           BlockSize,
                           CACHE_TYPE,
                           4,
                           4,
                           stream,
                           load_func,
                           store_func)
  } else if (params.gqa_num_per_partitions == 6) {
    constexpr int THDS_PER_BLOCK = 1024;
    BLHA_LAUNCH_GQA_KERNEL(T,
                           Dh,
                           Dh_MAX,
                           THREADS_PER_KEY,
                           THREADS_PER_VALUE,
                           THDS_PER_BLOCK,
                           BlockSize,
                           CACHE_TYPE,
                           6,
                           2,
                           stream,
                           load_func,
                           store_func)
  } else if (params.gqa_num_per_partitions == 7) {
    constexpr int THDS_PER_BLOCK = 1024;
    BLHA_LAUNCH_GQA_KERNEL(T,
                           Dh,
                           Dh_MAX,
                           THREADS_PER_KEY,
                           THREADS_PER_VALUE,
                           THDS_PER_BLOCK,
                           BlockSize,
                           CACHE_TYPE,
                           7,
                           1,
                           stream,
                           load_func,
                           store_func)
  } else if (params.gqa_num_per_partitions == 8) {
    constexpr int THDS_PER_BLOCK = 1024;
    BLHA_LAUNCH_GQA_KERNEL(T,
                           Dh,
                           Dh_MAX,
                           THREADS_PER_KEY,
                           THREADS_PER_VALUE,
                           THDS_PER_BLOCK,
                           BlockSize,
                           CACHE_TYPE,
                           8,
                           4,
                           stream,
                           load_func,
                           store_func)
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "gqa_num_per_partitions = %d is unsupported!",
        params.gqa_num_per_partitions));
  }
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int BlockSize,
          int THREADS_PER_KEY,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc>
void dispatch_blha_impl(const Block_AttN_params<T> &params,
                        const GPU(Stream_t) & stream,
                        LoadFunc load_func,
                        StoreFunc store_func,
                        const int use_cachekv_int8) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;

  if (use_cachekv_int8 > 0) {
    VLOG(1) << "cache_int8";
    dispatch_blha_gqa_kernel<T,
                             Dh,
                             Dh_MAX,
                             BlockSize,
                             THREADS_PER_VALUE,
                             THREADS_PER_KEY,
                             THREADS_PER_BLOCK,
                             CacheType::INT8>(
        params, stream, load_func, store_func);
  } else {
    VLOG(1) << "normal cache";
    dispatch_blha_gqa_kernel<T,
                             Dh,
                             Dh_MAX,
                             BlockSize,
                             THREADS_PER_VALUE,
                             THREADS_PER_KEY,
                             THREADS_PER_BLOCK,
                             CacheType::NORMAL>(
        params, stream, load_func, store_func);
  }
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int BlockSize,
          typename LoadFunc,
          typename StoreFunc>
void dispatch_blha_impl_key_and_thread(const Block_AttN_params<T> &params,
                                       const GPU(Stream_t) & stream,
                                       LoadFunc load_func,
                                       StoreFunc store_func,
                                       const int use_cachekv_int8) {
  dispatch_blha_impl<T, Dh, Dh_MAX, BlockSize, 4, 256>(
      params, stream, load_func, store_func, use_cachekv_int8);
}

template <typename T, int Dh, int Dh_MAX, typename LoadFunc, typename StoreFunc>
void dispatch_blha_impl_blocksize(const Block_AttN_params<T> &params,
                                  const GPU(Stream_t) & stream,
                                  LoadFunc load_func,
                                  StoreFunc store_func,
                                  const int use_cachekv_int8) {
  switch (params.block_size) {
    case 1:
      dispatch_blha_impl_key_and_thread<T, Dh, Dh_MAX, 1>(
          params, stream, load_func, store_func, use_cachekv_int8);
      break;
    case 32:
      dispatch_blha_impl_key_and_thread<T, Dh, Dh_MAX, 32>(
          params, stream, load_func, store_func, use_cachekv_int8);
      break;
    case 64:
      dispatch_blha_impl_key_and_thread<T, Dh, Dh_MAX, 64>(
          params, stream, load_func, store_func, use_cachekv_int8);
      break;
    case 128:
      dispatch_blha_impl_key_and_thread<T, Dh, Dh_MAX, 128>(
          params, stream, load_func, store_func, use_cachekv_int8);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "block_size = %d is unsupported!", params.block_size));
  }
}

template <typename T, typename LoadFunc, typename StoreFunc>
void dispatch_blha_impl_headsize(const phi::GPUContext &dev_ctx,
                                 const Block_AttN_params<T> &params,
                                 int dim_head,
                                 LoadFunc load_func,
                                 StoreFunc store_func,
                                 const int use_cachekv_int8) {
  switch (dim_head) {
    case 32:
      dispatch_blha_impl_blocksize<T, 32, 32>(
          params, dev_ctx.stream(), load_func, store_func, use_cachekv_int8);
      break;
    case 64:
      dispatch_blha_impl_blocksize<T, 64, 64>(
          params, dev_ctx.stream(), load_func, store_func, use_cachekv_int8);
      break;
    case 96:
      dispatch_blha_impl_blocksize<T, 96, 128>(
          params, dev_ctx.stream(), load_func, store_func, use_cachekv_int8);
      break;
    case 128:
      dispatch_blha_impl_blocksize<T, 128, 128>(
          params, dev_ctx.stream(), load_func, store_func, use_cachekv_int8);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Dim_head = %d is unsupported!", dim_head));
  }
}

template <typename T>
void DispatchBLHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const Block_AttN_params<T> &params,
                  int q_num_head,
                  int kv_num_head,
                  int dim_head,
                  int use_cachekv_int8,
                  phi::DenseTensor *out_tensor) {
  MMHALoad<T> load_func(qkv_tensor.data<T>());
  MMHAStore<T> store_func(out_tensor->data<T>());
  dispatch_blha_impl_headsize(
      dev_ctx, params, dim_head, load_func, store_func, use_cachekv_int8);
}

template <typename T>
void blha(const phi::GPUContext &dev_ctx,
          const phi::DenseTensor &qkv_tensor,
          const phi::DenseTensor *qkv_bias_tensor,
          const phi::DenseTensor *block_tables,
          const phi::DenseTensor *src_mask_tensor,
          const phi::DenseTensor *cum_offsets_tensor,
          const phi::DenseTensor *sequence_lengths_tensor,
          const phi::DenseTensor *rotary_tensor,
          phi::DenseTensor *k_cache,
          phi::DenseTensor *v_cache,
          phi::DenseTensor *out_tensor,
          const int batch_size,
          const int max_num_blocks_per_seq,
          const int block_size,
          const int seq_len,
          const int pre_cache_length,
          const int q_num_head,
          const int kv_num_head,
          const int dim_head,
          const int timestep,
          const int rotary_emb_dims,
          float inv_sqrt_dh,
          const float rope_theta,
          const bool add_qkv_bias = true,
          const bool neox_rotary_style = false,
          const int quant_round_type = 1,
          const float quant_max_bound = 127.0f,
          const float quant_min_bound = -127.0f,
          const phi::DenseTensor *cache_k_quant_scales = nullptr,
          const phi::DenseTensor *cache_v_quant_scales = nullptr,
          const phi::DenseTensor *cache_k_dequant_scales = nullptr,
          const phi::DenseTensor *cache_v_dequant_scales = nullptr,
          const phi::DenseTensor *dequant_qkv_scales = nullptr,
          const phi::DenseTensor *shift = nullptr,
          const phi::DenseTensor *smooth = nullptr,
          const float quant_fmha_out_scale = -1,
          int use_cachekv_int8 = 0) {
  Block_AttN_params<T> params;

  if (cache_k_quant_scales) {
    VLOG(1) << "blha quant cachekv";
    params.k_cache_I = k_cache->data<uint8_t>();
    params.v_cache_I = v_cache->data<uint8_t>();
    params.cache_k_quant_scales = cache_k_quant_scales->data<float>();
    params.cache_v_quant_scales = cache_v_quant_scales->data<float>();
    params.cache_k_dequant_scales = cache_k_dequant_scales->data<float>();
    params.cache_v_dequant_scales = cache_v_dequant_scales->data<float>();
  } else {
    VLOG(1) << "blha not quant cachekv";
    params.k_cache = k_cache->data<T>();
    params.v_cache = v_cache->data<T>();
  }

  params.max_num_blocks_per_seq = max_num_blocks_per_seq;
  params.neox_rotary_style = neox_rotary_style;
  params.attn_mask = nullptr;
  bool mask_broadcast_num_heads = false;
  if (src_mask_tensor) {
    if (src_mask_tensor->dims()[1] == 1) {
      // all head share a mask.
      mask_broadcast_num_heads = true;
    } else if (src_mask_tensor->dims()[1] == q_num_head) {
      mask_broadcast_num_heads = false;
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Unknown dimension for attn_mask, the q_num_head(2nd) "
          "dimension is invalid, it should be 1 or q_num_head(%d), "
          "but got %d",
          q_num_head,
          src_mask_tensor->dims()[1]));
    }
    params.attn_mask = src_mask_tensor->data<T>();
    params.mask_broadcast_num_heads = mask_broadcast_num_heads;
    params.mask_length = src_mask_tensor->dims()[3];
  } else {
    params.attn_mask = nullptr;
  }
  params.block_tables = block_tables->data<int>();

  if (sequence_lengths_tensor) {
    params.sequence_lengths = sequence_lengths_tensor->data<int>();
  }

  if (cum_offsets_tensor) {
    params.cum_offsets = cum_offsets_tensor->data<int>();
  } else {
    params.cum_offsets = nullptr;
  }
  params.seq_len = seq_len;
  params.pre_cache_length = pre_cache_length;

  if (rotary_tensor) {
    params.rotary_emb = rotary_tensor->data<float>();
    params.rope_stride = rotary_tensor->dims()[2] * rotary_tensor->dims()[4];
  } else {
    params.rotary_emb = nullptr;
  }

  params.add_qkv_bias = add_qkv_bias;
  if (add_qkv_bias) {
    params.qkv_bias = qkv_bias_tensor->data<T>();
  }

  params.batch_size = batch_size;
  params.block_size = block_size;
  params.q_num_head = q_num_head;
  params.kv_num_head = kv_num_head;
  params.gqa_num_per_partitions = q_num_head / kv_num_head;

  params.timestep = timestep + pre_cache_length;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;
  params.rope_theta = rope_theta;

  VLOG(3) << "batch_size: " << batch_size << " q_num_head: " << q_num_head
          << " kv_num_head: " << kv_num_head << " block_size: " << block_size
          << " timestep: " << timestep;

  DispatchBLHA<T>(dev_ctx,
                  qkv_tensor,
                  params,
                  q_num_head,
                  kv_num_head,
                  dim_head,
                  use_cachekv_int8,
                  out_tensor);
}

inline GPU(Error_t) GetNumBlocks(int64_t n, int *num_blocks) {
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
  return GPU(Success);
}

template <class Func>
inline GPU(Error_t) GetNumBlocks(Func func,
                                 int64_t block_size,
                                 size_t dynamic_smem_size,
                                 int64_t max_blocks,
                                 int64_t waves,
                                 int *num_blocks) {
  int dev;
  {
    GPU(Error_t) err = GPU(GetDevice)(&dev);
    if (err != GPU(Success)) {
      return err;
    }
  }
  int sm_count;
  {
    GPU(Error_t)
    err = GPU(DeviceGetAttribute)(&sm_count, GPUMultiProcessorCount, dev);
    if (err != GPU(Success)) {
      return err;
    }
  }
  int max_active_blocks;
  {
    GPU(Error_t)
    err = GPU(OccupancyMaxActiveBlocksPerMultiprocessor)(
        &max_active_blocks, func, block_size, dynamic_smem_size);
  }
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return GPU(Success);
}

template <typename T, int VecSize = 1>
__global__ void cache_int8_kernel(
    const T *__restrict__ qkv,  // [num_tokens, num_heads + 2 * kv_num_heads,
                                // head_size]
    uint8_t *__restrict__ key_cache,          // [num_blocks, kv_num_heads,
                                              // block_size, head_size]
    uint8_t *__restrict__ value_cache,        // [num_blocks, kv_num_heads,
                                              // block_size, head_size]
    const int *__restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,  // [num_tokens]
    const int *__restrict__ seq_lens,         // [bsz]
    const float *cache_k_scales,
    const float *cache_v_scales,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int q_num_heads,
    const int kv_num_heads,
    const int head_size,
    const int block_size,
    const int pre_cache_length,
    const int elem_cnt,
    const int round_type,
    const float max_bound,
    const float min_bound) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadKVT = phi::AlignedVector<uint8_t, VecSize>;
  LoadT src_vec;
  LoadKVT cache_vec;

  uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t hidden_size = kv_num_heads * head_size;
  const uint32_t offset = 2 * hidden_size;
  for (uint32_t linear_index = global_thread_idx * VecSize,
                step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const uint32_t token_idx = linear_index / offset;
    const uint32_t bias = linear_index % offset;
    const uint32_t qkv_id = bias / hidden_size;  // skip q
    const uint32_t qkv_bias = bias % hidden_size;
    const uint32_t hi = qkv_bias / head_size;
    const uint32_t h_bias = qkv_bias % head_size;
    const uint32_t ori_token_idx = token_idx + padding_offsets[token_idx];
    const uint32_t ori_bi = ori_token_idx / max_seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const uint32_t ori_seq_id = ori_token_idx % max_seq_len + pre_cache_length;

    const int32_t *block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * head_size +
                             hi * block_size * head_size +
                             block_offset * head_size + h_bias;
    const uint32_t ori_idx =
        token_idx * (q_num_heads + 2 * kv_num_heads) * head_size +
        q_num_heads * head_size + qkv_id * hidden_size + hi * head_size +
        h_bias;
    phi::Load<T, VecSize>(&qkv[ori_idx], &src_vec);

    const uint32_t cache_idx = hi;
#ifdef PADDLE_WITH_HIP
    T scale;
    if constexpr (kernel_dtype_is_same<T, half>::value) {
      scale = qkv_id == 0 ? __float2half(cache_k_scales[cache_idx])
                          : __float2half(cache_v_scales[cache_idx]);
    } else {
      scale = qkv_id == 0 ? static_cast<T>(cache_k_scales[cache_idx])
                          : static_cast<T>(cache_v_scales[cache_idx]);
    }
#else
    const T scale =
        qkv_id == 0 ? cache_k_scales[cache_idx] : cache_v_scales[cache_idx];
#endif
#pragma unroll
    for (uint32_t i = 0; i < VecSize; i++) {
#ifdef PADDLE_WITH_HIP
      float quant_value;
      if constexpr (kernel_dtype_is_same<T, half>::value) {
        quant_value = __half2float(scale * src_vec[i]);
      } else {
        quant_value = static_cast<float>(scale * src_vec[i]);
      }
#else
      float quant_value = static_cast<float>(scale * src_vec[i]);
#endif
      if (round_type == 0) {
        quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
      } else {
        quant_value = static_cast<float>(round(quant_value));
      }
      quant_value = quant_value > max_bound ? max_bound : quant_value;
      quant_value = quant_value < min_bound ? min_bound : quant_value;
      cache_vec[i] = static_cast<uint8_t>(quant_value + 128.0f);
    }

    if (qkv_id == 0) {
      phi::Store<uint8_t, VecSize>(cache_vec, &key_cache[tgt_idx]);
    } else {
      phi::Store<uint8_t, VecSize>(cache_vec, &value_cache[tgt_idx]);
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void cache_kernel(
    const T *__restrict__ qkv,    // [num_tokens, num_heads + 2 * kv_num_heads,
                                  // head_size]
    T *__restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_size]
    T *__restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_size]
    const int *__restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,  // [num_tokens]
    const int *__restrict__ seq_lens,         // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int q_num_heads,
    const int kv_num_heads,
    const int head_size,
    const int block_size,
    const int pre_cache_length,
    const int elem_cnt) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t hidden_size = kv_num_heads * head_size;
  const uint32_t offset = 2 * hidden_size;
  for (uint32_t linear_index = global_thread_idx * VecSize,
                step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const uint32_t token_idx = linear_index / offset;
    const uint32_t bias = linear_index % offset;
    const uint32_t qkv_id = bias / hidden_size;  // skip q
    const uint32_t qkv_bias = bias % hidden_size;
    const uint32_t hi = qkv_bias / head_size;
    const uint32_t h_bias = qkv_bias % head_size;
    const uint32_t ori_token_idx = token_idx + padding_offsets[token_idx];
    const uint32_t ori_bi = ori_token_idx / max_seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const uint32_t ori_seq_id = ori_token_idx % max_seq_len + pre_cache_length;

    const int32_t *block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * head_size +
                             hi * block_size * head_size +
                             block_offset * head_size + h_bias;
    const uint32_t ori_idx =
        token_idx * (q_num_heads + 2 * kv_num_heads) * head_size +
        q_num_heads * head_size + qkv_id * hidden_size + hi * head_size +
        h_bias;
    phi::Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &key_cache[tgt_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &value_cache[tgt_idx]);
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void write_pre_cache_int8_to_cache(
    uint8_t *__restrict__ key_cache,  // [num_blocks, num_heads, block_size,
                                      // head_size]
    uint8_t *__restrict__ value_cache,
    const T *__restrict__ pre_key_cache,  // [bsz, pre_cache_len, num_head,
                                          // head_dim]
    const T *__restrict__ pre_value_cache,
    const int *__restrict__ block_tables,  // [bsz, max_blocks_per_seq]
    const int *__restrict__ seq_lens,
    const float *cache_k_scales,
    const float *cache_v_scales,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int pre_cache_length,
    const int elem_cnt,
    const int round_type,
    const float max_bound,
    const float min_bound) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadKVT = phi::AlignedVector<uint8_t, VecSize>;
  LoadT src_vec;
  LoadKVT cache_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int hidden_size = pre_cache_length * head_size;
  const int cache_hidden_size = num_heads * hidden_size;
  const int offset = 2 * cache_hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int batch_id = linear_index / offset;
    if (seq_lens[batch_id] == 0) continue;
    const int *block_table_now = block_tables + batch_id * max_blocks_per_seq;

    const int32_t cache_seq_id = (linear_index % hidden_size) / head_size;
    const int32_t head_id = (linear_index % cache_hidden_size) / hidden_size;
    const int32_t size_id = linear_index % head_size;

    const int32_t kv_id = (linear_index % offset) / cache_hidden_size;
    const int32_t read_id = batch_id * cache_hidden_size +
                            head_id * hidden_size + cache_seq_id * head_size +
                            size_id;
    if (kv_id == 0) {
      phi::Load<T, VecSize>(&pre_key_cache[read_id], &src_vec);
    } else {
      phi::Load<T, VecSize>(&pre_value_cache[read_id], &src_vec);
    }

    const int block_idx = block_table_now[cache_seq_id / block_size];
    const int block_offset = cache_seq_id % block_size;

    const int tgt_idx = block_idx * num_heads * block_size * head_size +
                        head_id * block_size * head_size +
                        block_offset * head_size + size_id;

    const float scale =
        kv_id == 0 ? cache_k_scales[head_id] : cache_v_scales[head_id];

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
#ifdef PADDLE_WITH_HIP
      float quant_value;
      if constexpr (kernel_dtype_is_same<T, half>::value) {
        quant_value = scale * __half2float(src_vec[i]);
      } else {
        quant_value = scale * static_cast<float>(src_vec[i]);
      }
#else
      float quant_value = scale * static_cast<float>(src_vec[i]);
#endif
      if (round_type == 0) {
        quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
      } else {
        quant_value = static_cast<float>(round(quant_value));
      }
      quant_value = quant_value > max_bound ? max_bound : quant_value;
      quant_value = quant_value < min_bound ? min_bound : quant_value;
      cache_vec[i] = static_cast<uint8_t>(quant_value + 128.0f);
    }

    if (kv_id == 0) {
      phi::Store<uint8_t, VecSize>(cache_vec, &key_cache[tgt_idx]);
    } else {
      phi::Store<uint8_t, VecSize>(cache_vec, &value_cache[tgt_idx]);
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void write_pre_cache_to_cache(
    T *__restrict__ key_cache,  // [num_blocks, num_heads, block_size,
                                // head_size]
    T *__restrict__ value_cache,
    const T *__restrict__ pre_key_cache,  // [bsz, num_heads, pre_cache_len,
                                          // head_dim]
    const T *__restrict__ pre_value_cache,
    const int *__restrict__ block_tables,  // [bsz, max_blocks_per_seq]
    const int *__restrict__ seq_lens,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int pre_cache_length,
    const int elem_cnt) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int hidden_size = pre_cache_length * head_size;
  const int cache_hidden_size = num_heads * hidden_size;
  const int offset = 2 * cache_hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int batch_id = linear_index / offset;
    if (seq_lens[batch_id] == 0) continue;
    const int *block_table_now = block_tables + batch_id * max_blocks_per_seq;

    const int32_t cache_seq_id = (linear_index % hidden_size) / head_size;
    const int32_t head_id = (linear_index % cache_hidden_size) / hidden_size;
    const int32_t size_id = linear_index % head_size;

    const int32_t kv_id = (linear_index % offset) / cache_hidden_size;
    const int32_t read_id = batch_id * cache_hidden_size +
                            head_id * hidden_size + cache_seq_id * head_size +
                            size_id;
    if (kv_id == 0) {
      phi::Load<T, VecSize>(&pre_key_cache[read_id], &src_vec);
    } else {
      phi::Load<T, VecSize>(&pre_value_cache[read_id], &src_vec);
    }

    const int block_idx = block_table_now[cache_seq_id / block_size];
    const int block_offset = cache_seq_id % block_size;

    const int tgt_idx = block_idx * num_heads * block_size * head_size +
                        head_id * block_size * head_size +
                        block_offset * head_size + size_id;

    if (kv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &key_cache[tgt_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &value_cache[tgt_idx]);
    }
  }
}

template <typename T>
void CacheKernel(const phi::GPUContext &dev_ctx,
                 const phi::DenseTensor &qkv,  // [token_num, (2 * kv_num_head +
                                               // q_num_head), head_dim]
                 const phi::DenseTensor &block_tables,
                 const phi::DenseTensor &padding_offsets,
                 const phi::DenseTensor &seq_lens,
                 const paddle::optional<DenseTensor> &pre_key_cache,
                 const paddle::optional<DenseTensor> &pre_value_cache,
                 const paddle::optional<DenseTensor> &cache_k_scales,
                 const paddle::optional<DenseTensor> &cache_v_scales,
                 const int batch_size,
                 const int num_tokens,
                 const int q_num_heads,
                 const int kv_num_heads,
                 const int head_size,
                 const int max_seq_len,
                 const int pre_cache_length,
                 phi::DenseTensor *key_cache_out,
                 phi::DenseTensor *value_cache_out,
                 const int round_type = 0,
                 const float max_bound = 127.0,
                 const float min_bound = -127.0) {
  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  const int max_blocks_per_seq = block_tables.dims()[1];
  const int32_t block_size = key_cache_out->dims()[2];

  // stage 1: write qkv to cache [pre_cache_length:]
  int elem_nums = num_tokens * 2 * kv_num_heads * head_size;  // just k and v
  constexpr int PackSize = 16 / sizeof(T);
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);

  if (cache_k_scales) {
    VLOG(1) << "cache kv quant";
    cache_int8_kernel<DataType_, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            reinterpret_cast<DataType_ *>(const_cast<T *>(qkv.data<T>())),
            key_cache_out->data<uint8_t>(),
            value_cache_out->data<uint8_t>(),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            seq_lens.data<int>(),
            cache_k_scales.get().data<float>(),
            cache_v_scales.get().data<float>(),
            max_seq_len,
            max_blocks_per_seq,
            q_num_heads,
            kv_num_heads,
            head_size,
            block_size,
            pre_cache_length,
            elem_nums,
            round_type,
            max_bound,
            min_bound);
  } else {
    VLOG(1) << "cache kv not quant";
    cache_kernel<DataType_, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            reinterpret_cast<DataType_ *>(const_cast<T *>(qkv.data<T>())),
            reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            seq_lens.data<int>(),
            max_seq_len,
            max_blocks_per_seq,
            q_num_heads,
            kv_num_heads,
            head_size,
            block_size,
            pre_cache_length,
            elem_nums);
  }

  if (pre_key_cache) {
    // NOTE: not support gqa
    // stage 2: write pre_cache to cache [:pre_cache_length]
    elem_nums = batch_size * kv_num_heads * pre_cache_length * head_size * 2;
    pack_num = elem_nums / PackSize;
    GetNumBlocks(pack_num, &grid_size);
    if (cache_k_scales) {
      write_pre_cache_int8_to_cache<DataType_, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
              key_cache_out->data<uint8_t>(),
              value_cache_out->data<uint8_t>(),
              reinterpret_cast<const DataType_ *>(
                  pre_key_cache.get().data<T>()),
              reinterpret_cast<const DataType_ *>(
                  pre_value_cache.get().data<T>()),
              block_tables.data<int>(),
              seq_lens.data<int>(),
              cache_k_scales->data<float>(),
              cache_v_scales->data<float>(),
              max_seq_len,
              max_blocks_per_seq,
              kv_num_heads,
              head_size,
              block_size,
              pre_cache_length,
              elem_nums,
              round_type,
              max_bound,
              min_bound);
    } else {
      write_pre_cache_to_cache<DataType_, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
              reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
              reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
              reinterpret_cast<const DataType_ *>(
                  pre_key_cache.get().data<T>()),
              reinterpret_cast<const DataType_ *>(
                  pre_value_cache.get().data<T>()),
              block_tables.data<int>(),
              seq_lens.data<int>(),
              max_seq_len,
              max_blocks_per_seq,
              kv_num_heads,
              head_size,
              block_size,
              pre_cache_length,
              elem_nums);
    }
  }
}

template <typename T, int VecSize>
__global__ void quant_write_cache_int8_kernel(
    const T *__restrict__ qkv,  // [num_tokens, 2 * kv_num_heads + q_num_heads,
                                // head_size]
    uint8_t *__restrict__ key_cache,  // [num_blocks, kv_num_heads, block_size,
                                      // head_size]
    uint8_t *__restrict__ value_cache,        // [num_blocks, kv_num_heads,
                                              // block_size, head_size]
    const int *__restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,  // [num_tokens]
    const int *__restrict__ seq_lens,         // [bsz]
    const int max_seq_len,
    const int pre_cache_length,
    const int max_blocks_per_seq,
    const int num_tokens,
    const int q_num_heads,
    const int kv_num_heads,
    const int head_size,
    const int block_size,
    float *k_quant_scales,
    float *v_quant_scales,
    float *k_dequant_scales,
    float *v_dequant_scales) {
  const int hi = blockIdx.x;
  const int b_id = blockIdx.y;
  if (seq_lens[b_id] <= 0) return;
  const int qkv_id = blockIdx.z;
  const int head_group_size = q_num_heads / kv_num_heads;

  using InVec = phi::AlignedVector<T, VecSize>;
  using OutVec = phi::AlignedVector<uint8_t, VecSize>;

  InVec in_vec;
  OutVec out_vec;
  InVec abs_max_vec;
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
#ifdef PADDLE_WITH_HIP
    if constexpr (kernel_dtype_is_same<T, half>::value) {
      abs_max_vec[i] = __float2half(0.0f);
    } else {
      abs_max_vec[i] = static_cast<T>(0.0f);
    }
#else
    abs_max_vec[i] = 0.0f;
#endif
  }

  uint8_t *dst_ptr;
  float *quant_scales;
  float *dequant_scales;
  if (qkv_id == 0) {
    dst_ptr = key_cache;
    quant_scales = k_quant_scales;
    dequant_scales = k_dequant_scales;
  } else {
    dst_ptr = value_cache;
    quant_scales = v_quant_scales;
    dequant_scales = v_dequant_scales;
  }

  T local_abs_max;

  for (int idx = threadIdx.x * VecSize; idx < num_tokens * head_size;
       idx += blockDim.x * VecSize) {
    int token_idx = idx / head_size;
    int h_offset = idx % head_size;
    int linear_idx = token_idx * (2 * kv_num_heads + q_num_heads) * head_size +
                     (qkv_id + head_group_size) * kv_num_heads * head_size +
                     hi * head_size + h_offset;

    Load<T, VecSize>(qkv + linear_idx, &in_vec);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      abs_max_vec[i] = MaxFunc<T>()(abs_max_vec[i], AbsFunc<T>()(in_vec[i]));
    }
  }

  local_abs_max = LocalReduceMax<T, InVec, VecSize>(abs_max_vec);
  T abs_max_val = BlockReduceAbsMax<T>(local_abs_max, 0xffffffff);

  __shared__ float quant_scale;
  if (threadIdx.x == 0) {
#ifdef PADDLE_WITH_HIP
    if constexpr (kernel_dtype_is_same<T, half>::value) {
      quant_scale = 127.0f / __half2float(abs_max_val);
    } else {
      quant_scale = 127.0f / static_cast<float>(abs_max_val);
    }
#else
    quant_scale = 127.0f / static_cast<float>(abs_max_val);
#endif
  }

  __syncthreads();
  for (int idx = threadIdx.x * VecSize; idx < num_tokens * head_size;
       idx += blockDim.x * VecSize) {
    int token_idx = idx / head_size;
    int h_offset = idx % head_size;
    int linear_idx = token_idx * (2 * kv_num_heads + q_num_heads) * head_size +
                     (qkv_id + head_group_size) * kv_num_heads * head_size +
                     hi * head_size + h_offset;

    Load<T, VecSize>(qkv + linear_idx, &in_vec);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] = QuantFunc<T>()(in_vec[i], quant_scale);
    }

    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / max_seq_len;
    if (ori_bi != b_id) continue;
    const int ori_seq_id = ori_token_idx % max_seq_len + pre_cache_length;

    const int *block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[ori_seq_id / block_size];
    const int block_offset = ori_seq_id % block_size;
    // [max_block_num, num_head, block_size, head_dim/x, x]
    Store<uint8_t>(out_vec,
                   dst_ptr + block_idx * kv_num_heads * block_size * head_size +
                       hi * block_size * head_size + block_offset * head_size +
                       h_offset);
  }

  if (threadIdx.x == 0) {
    quant_scales[b_id * kv_num_heads + hi] = quant_scale;
    dequant_scales[b_id * kv_num_heads + hi] = 1.0f / quant_scale;
  }
}

template <typename T>
void DynamicQuantCacheKernel(
    const phi::GPUContext &dev_ctx,
    const phi::DenseTensor
        &qkv,  // [token_num, 2 * q_num_head + kv_num_head, head_dim]
    const phi::DenseTensor &block_tables,
    const phi::DenseTensor &padding_offsets,
    const phi::DenseTensor &seq_lens,
    const phi::DenseTensor &k_quant_scales,
    const phi::DenseTensor &v_quant_scales,
    const phi::DenseTensor &k_dequant_scales,
    const phi::DenseTensor &v_dequant_scales,
    const paddle::optional<DenseTensor> &pre_key_cache,
    const paddle::optional<DenseTensor> &pre_value_cache,
    const int batch_size,
    const int q_num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const int pre_cache_length,
    phi::DenseTensor *key_cache_out,
    phi::DenseTensor *value_cache_out) {
  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  const int num_tokens = padding_offsets.dims()[0];
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int32_t block_size = key_cache_out->dims()[2];
  constexpr int PackSize = 16 / sizeof(T);

  assert(head_size % PackSize == 0);

  const DataType_ *qkv_ptr = reinterpret_cast<const DataType_ *>(qkv.data<T>());

  //  [max_block_num, num_head, block_size, head_dim]

  uint8_t *cache_k_ptr = key_cache_out->data<uint8_t>();
  uint8_t *cache_v_ptr = value_cache_out->data<uint8_t>();

  float *k_quant_scales_data =
      const_cast<float *>(k_quant_scales.data<float>());
  float *k_dequant_scales_data =
      const_cast<float *>(k_dequant_scales.data<float>());

  float *v_quant_scales_data =
      const_cast<float *>(v_quant_scales.data<float>());
  float *v_dequant_scales_data =
      const_cast<float *>(v_dequant_scales.data<float>());

  constexpr int block_sz = 1024;

  const int bsz = seq_lens.dims()[0];

  dim3 grid(kv_num_heads, bsz, 2);

  // [token_num, (2 * kv_num_head + q_num_head), head_dim/x, x]->[max_block_num,
  // kv_num_head, block_size, head_dim/x, x] Quant and Write kv
  quant_write_cache_int8_kernel<DataType_, PackSize>
      <<<grid, block_sz, 0, dev_ctx.stream()>>>(qkv_ptr,
                                                cache_k_ptr,
                                                cache_v_ptr,
                                                block_tables.data<int>(),
                                                padding_offsets.data<int>(),
                                                seq_lens.data<int>(),
                                                max_seq_len,
                                                pre_cache_length,
                                                max_blocks_per_seq,
                                                num_tokens,
                                                q_num_heads,
                                                kv_num_heads,
                                                head_size,
                                                block_size,
                                                k_quant_scales_data,
                                                v_quant_scales_data,
                                                k_dequant_scales_data,
                                                v_dequant_scales_data);

  if (pre_key_cache) {
    // stage 2: write pre_cache to cache [:pre_cache_length]
    const int elem_nums =
        batch_size * kv_num_heads * pre_cache_length * head_size * 2;
    const int pack_num = elem_nums / PackSize;
    const int blocksize = 128;
    int grid_size = 1;
    GetNumBlocks(pack_num, &grid_size);
    write_pre_cache_int8_to_cache<DataType_, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            key_cache_out->data<uint8_t>(),
            value_cache_out->data<uint8_t>(),
            reinterpret_cast<const DataType_ *>(pre_key_cache.get().data<T>()),
            reinterpret_cast<const DataType_ *>(
                pre_value_cache.get().data<T>()),
            block_tables.data<int>(),
            seq_lens.data<int>(),
            k_quant_scales.data<float>(),
            v_quant_scales.data<float>(),
            max_seq_len,
            max_blocks_per_seq,
            kv_num_heads,
            head_size,
            block_size,
            pre_cache_length,
            elem_nums,
            1,
            127.0f,
            -127.0f);
  }
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t base_idx = token_idx * 3 * hidden_size +
                             qkv_id * hidden_size + hi * last_dim + h_bias;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  // [token_num, 2, num_head, dim_head / 2]
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int base_idx_left = token_idx * 3 * full_hidden_size +
                              qkv_id * full_hidden_size + hi * last_dim +
                              h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;

    phi::Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                   // [token_num, 3, num_head, dim_head]
    const T *qkv_input,       // qkv
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    bool use_neox_style = false) {
  int elem_nums = token_num * 2 * head_num * dim_head;  // just q and k
  if (use_neox_style) {
    elem_nums = token_num * head_num * dim_head;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    VariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv,
                                                        elem_nums,
                                                        head_num,
                                                        seq_len,
                                                        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    NeoxVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv,
                                                        elem_nums,
                                                        head_num,
                                                        seq_len,
                                                        dim_head);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + kv_num_head) * last_dim;
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

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    // [token_num, q_num_head + 2 * kv_num_head, last_dim]
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  // [token_num, 2, q_num_head, dim_head / 2]
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + kv_num_head) * half_lastdim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int base_idx_left =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;

    phi::Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                   // [token_num, 3, q_num_head, dim_head]
    const T *qkv_input,       // qkv
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int q_head_num,
    const int kv_head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    bool use_neox_style = false) {
  int elem_nums =
      token_num * (q_head_num + kv_head_num) * dim_head;  // just q and k
  if (use_neox_style) {
    elem_nums /= 2;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    GQAVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv,
                                                        elem_nums,
                                                        q_head_num,
                                                        kv_head_num,
                                                        seq_len,
                                                        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    GQANeoxVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv,
                                                        elem_nums,
                                                        q_head_num,
                                                        kv_head_num,
                                                        seq_len,
                                                        dim_head);
  }
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<int, VecSize>;
  using LoadBiasT = phi::AlignedVector<T, VecSize>;
  using LoadScaleT = phi::AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
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
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    phi::Load<int, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (qkv_id < 2) {
      phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = input_left * out_scale_vec[2 * i] +
                   static_cast<float>(bias_vec[2 * i]);
      input_right = input_right * out_scale_vec[2 * i + 1] +
                    static_cast<float>(bias_vec[2 * i + 1]);
      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<int, VecSize>;
  using LoadBiasT = phi::AlignedVector<T, VecSize>;
  using LoadScaleT = phi::AlignedVector<float, VecSize>;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadBiasT left_bias_vec;
  LoadBiasT right_bias_vec;
  LoadScaleT left_out_scale_vec;
  LoadScaleT right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left =
        qkv_id * full_hidden_size + hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * 3 * full_hidden_size + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    phi::Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx_left],
                              &left_out_scale_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx_right],
                              &right_out_scale_vec);
    if (qkv_id < 2) {
      phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      // dequant + bias_add
      input_left = input_left * left_out_scale_vec[i] +
                   static_cast<float>(left_bias_vec[i]);
      input_right = input_right * right_out_scale_vec[i] +
                    static_cast<float>(right_bias_vec[i]);
      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(left_bias_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_bias_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                       // [token_num, 3, num_head, dim_head]
    const int *qkv_input,         // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    bool use_neox_style = false) {
  int elem_nums = token_num * 3 * head_num * dim_head;  // just q and k
  if (use_neox_style) {
    elem_nums = token_num * 3 * head_num * dim_head / 2;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    VariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv_out_scales,
                                                        qkv_bias,
                                                        qkv,
                                                        elem_nums,
                                                        head_num,
                                                        seq_len,
                                                        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    NeoxVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv_out_scales,
                                                        qkv_bias,
                                                        qkv,
                                                        elem_nums,
                                                        head_num,
                                                        seq_len,
                                                        dim_head);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_biases,          // [3, q_num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<int, VecSize>;
  using LoadBiasT = phi::AlignedVector<T, VecSize>;
  using LoadScaleT = phi::AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
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

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    phi::Load<int, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (hi < q_num_head + kv_num_head) {
      phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = input_left * out_scale_vec[2 * i] +
                   static_cast<float>(bias_vec[2 * i]);
      input_right = input_right * out_scale_vec[2 * i + 1] +
                    static_cast<float>(bias_vec[2 * i + 1]);
      if (hi < q_num_head + kv_num_head) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_biases,          // [3, q_num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<int, VecSize>;
  using LoadBiasT = phi::AlignedVector<T, VecSize>;
  using LoadScaleT = phi::AlignedVector<float, VecSize>;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadBiasT left_bias_vec;
  LoadBiasT right_bias_vec;
  LoadScaleT left_out_scale_vec;
  LoadScaleT right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * half_lastdim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left = hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * offset + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    phi::Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx_left],
                              &left_out_scale_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx_right],
                              &right_out_scale_vec);
    if (hi < (q_num_head + kv_num_head)) {
      phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      // dequant + bias_add
      input_left = input_left * left_out_scale_vec[i] +
                   static_cast<float>(left_bias_vec[i]);
      input_right = input_right * right_out_scale_vec[i] +
                    static_cast<float>(right_bias_vec[i]);
      if (hi < (q_num_head + kv_num_head)) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(left_bias_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_bias_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                       // [token_num, 3, q_num_head, dim_head]
    const int *qkv_input,         // qkv
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int q_head_num,
    const int kv_head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    bool use_neox_style = false) {
  int elem_nums =
      token_num * (q_head_num + 2 * kv_head_num) * dim_head;  // for all q k v
  if (use_neox_style) {
    elem_nums /= 2;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    GQAVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv_out_scales,
                                                        qkv_bias,
                                                        qkv,
                                                        elem_nums,
                                                        q_head_num,
                                                        kv_head_num,
                                                        seq_len,
                                                        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    GQANeoxVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv_out_scales,
                                                        qkv_bias,
                                                        qkv,
                                                        elem_nums,
                                                        q_head_num,
                                                        kv_head_num,
                                                        seq_len,
                                                        dim_head);
  }
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
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
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
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
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

template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotaryKernel(
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
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadT left_bias_vec;
  LoadT right_bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left =
        qkv_id * full_hidden_size + hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * 3 * full_hidden_size + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    phi::Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left =
          static_cast<float>(left_vec[i] + left_bias_vec[i]);
      const float input_right =
          static_cast<float>(right_vec[i] + right_bias_vec[i]);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_vec[i] = static_cast<T>(input_left);
        right_vec[i] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T>
void rotary_qk_variable(
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
    bool use_neox_style = false) {
  int elem_nums = token_num * 3 * head_num * dim_head;  // just q and k
  if (use_neox_style) {
    elem_nums = token_num * 3 * head_num * dim_head / 2;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
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
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    NeoxVariableLengthRotaryKernel<T, PackSize>
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
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
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

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);

      if (hi < q_num_head + kv_num_head) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
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

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadT left_bias_vec;
  LoadT right_bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * half_lastdim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left = hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * offset + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    phi::Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left =
          static_cast<float>(left_vec[i] + left_bias_vec[i]);
      const float input_right =
          static_cast<float>(right_vec[i] + right_bias_vec[i]);

      if (hi < (q_num_head + kv_num_head)) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_vec[i] = static_cast<T>(input_left);
        right_vec[i] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, q_num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int q_head_num,
    const int kv_head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    bool use_neox_style = false) {
  int elem_nums =
      token_num * (q_head_num + 2 * kv_head_num) * dim_head;  // for all q k v
  if (use_neox_style) {
    elem_nums /= 2;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    GQAVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv_bias,
                                                        qkv,
                                                        elem_nums,
                                                        q_head_num,
                                                        kv_head_num,
                                                        seq_len,
                                                        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    GQANeoxVariableLengthRotaryKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv_bias,
                                                        qkv,
                                                        elem_nums,
                                                        q_head_num,
                                                        kv_head_num,
                                                        seq_len,
                                                        dim_head);
  }
}

template <typename T, int VecSize, int RoundType>
__global__ void ShiftSmoothQuant(const T *input,
                                 const T *shift,
                                 const T *smooth,
                                 float scale,
                                 int8_t *out,
                                 int num,
                                 int cols,
                                 float quant_max_bound,
                                 float quant_min_bound) {
  phi::AlignedVector<T, VecSize> in_vec;
  phi::AlignedVector<T, VecSize> shift_vec;
  phi::AlignedVector<T, VecSize> smooth_vec;
  phi::AlignedVector<int8_t, VecSize> out_vec;

  for (int linear_id = blockIdx.x * blockDim.x + threadIdx.x;
       linear_id * VecSize < num;
       linear_id += gridDim.x * blockDim.x) {
    int idx = linear_id * VecSize;
    phi::Load<T, VecSize>(input + idx, &in_vec);
    phi::Load<T, VecSize>(shift + (idx % cols), &shift_vec);
    phi::Load<T, VecSize>(smooth + (idx % cols), &smooth_vec);

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float quant_value =
          quant_max_bound *
          static_cast<float>((in_vec[i] + shift_vec[i]) * smooth_vec[i]) *
          scale;
      quant_value = static_cast<float>(RoundType == 1 ? round(quant_value)
                                                      : rintf(quant_value));
      quant_value =
          quant_value > quant_max_bound ? quant_max_bound : quant_value;
      quant_value =
          quant_value < quant_min_bound ? quant_min_bound : quant_value;
      out_vec[i] = static_cast<int8_t>(quant_value);
    }
    phi::Store<int8_t, VecSize>(out_vec, out + idx);
  }
}

template <typename T, int VecSize, int RoundType>
__global__ void ShiftSmooth(const T *input,
                            const T *shift,
                            const T *smooth,
                            T *out,
                            int num,
                            int cols) {
  phi::AlignedVector<T, VecSize> in_vec;
  phi::AlignedVector<T, VecSize> shift_vec;
  phi::AlignedVector<T, VecSize> smooth_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  for (int linear_id = blockIdx.x * blockDim.x + threadIdx.x;
       linear_id * VecSize < num;
       linear_id += gridDim.x * blockDim.x) {
    int idx = linear_id * VecSize;
    phi::Load<T, VecSize>(input + idx, &in_vec);
    phi::Load<T, VecSize>(shift + (idx % cols), &shift_vec);
    phi::Load<T, VecSize>(smooth + (idx % cols), &smooth_vec);

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      out_vec[i] = (in_vec[i] + shift_vec[i]) * smooth_vec[i];
    }
    phi::Store<T, VecSize>(out_vec, out + idx);
  }
}

template <typename T>
void shift_smooth_quant(const phi::GPUContext &dev_ctx,
                        phi::DenseTensor *fmha_out,
                        const phi::DenseTensor &fmha_in,
                        const phi::DenseTensor &out_linear_shift,
                        const phi::DenseTensor &out_linear_smooth,
                        float out_linear_in_scale,
                        const int num_head,
                        const int dim_head,
                        const int quant_round_type,
                        const float quant_max_bound,
                        const float quant_min_bound) {
  constexpr int block_size = 512;
  constexpr int waves = 32;
  constexpr int vec_size = 16 / sizeof(T);

  int max_blocks = fmha_out->numel() / vec_size;
  int num_blocks = 0;
  if (out_linear_in_scale > 0) {
    if (quant_round_type == 0) {
      GetNumBlocks(ShiftSmoothQuant<T, vec_size, 0>,
                   block_size,
                   0,
                   max_blocks,
                   waves,
                   &num_blocks);
      ShiftSmoothQuant<T, vec_size, 0>
          <<<num_blocks, block_size, 0, dev_ctx.stream()>>>(
              fmha_in.data<T>(),
              out_linear_shift.data<T>(),
              out_linear_smooth.data<T>(),
              out_linear_in_scale,
              fmha_out->data<int8_t>(),
              fmha_out->numel(),
              num_head * dim_head,
              quant_max_bound,
              quant_min_bound);
    } else {
      GetNumBlocks(ShiftSmoothQuant<T, vec_size, 1>,
                   block_size,
                   0,
                   max_blocks,
                   waves,
                   &num_blocks);
      ShiftSmoothQuant<T, vec_size, 1>
          <<<num_blocks, block_size, 0, dev_ctx.stream()>>>(
              fmha_in.data<T>(),
              out_linear_shift.data<T>(),
              out_linear_smooth.data<T>(),
              out_linear_in_scale,
              fmha_out->data<int8_t>(),
              fmha_out->numel(),
              num_head * dim_head,
              quant_max_bound,
              quant_min_bound);
    }
  } else {
    if (quant_round_type == 0) {
      GetNumBlocks(ShiftSmooth<T, vec_size, 0>,
                   block_size,
                   0,
                   max_blocks,
                   waves,
                   &num_blocks);
      ShiftSmooth<T, vec_size, 0>
          <<<num_blocks, block_size, 0, dev_ctx.stream()>>>(
              fmha_in.data<T>(),
              out_linear_shift.data<T>(),
              out_linear_smooth.data<T>(),
              fmha_out->data<T>(),
              fmha_out->numel(),
              num_head * dim_head);
    } else {
      GetNumBlocks(ShiftSmooth<T, vec_size, 1>,
                   block_size,
                   0,
                   max_blocks,
                   waves,
                   &num_blocks);
      ShiftSmooth<T, vec_size, 1>
          <<<num_blocks, block_size, 0, dev_ctx.stream()>>>(
              fmha_in.data<T>(),
              out_linear_shift.data<T>(),
              out_linear_smooth.data<T>(),
              fmha_out->data<T>(),
              fmha_out->numel(),
              num_head * dim_head);
    }
  }
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
                                                const int q_head_num,
                                                const int kv_head_num,
                                                const int size_per_head) {
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  const int fused_hidden_size = (q_head_num + 2 * kv_head_num) * size_per_head;

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

    // [token_num, q_head_num or kv_head_num, size_per_head]
    if (head_id < q_head_num) {
      const int32_t write_idx = token_idx * q_head_num * size_per_head +
                                head_id * size_per_head + size_id;
      phi::Store<T, VecSize>(src_vec, &q_buf[write_idx]);
    } else {
      if (head_id < q_head_num + kv_head_num) {
        const int32_t write_idx = token_idx * kv_head_num * size_per_head +
                                  (head_id - q_head_num) * size_per_head +
                                  size_id;
        phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
      } else {
        const int32_t write_idx =
            token_idx * kv_head_num * size_per_head +
            (head_id - q_head_num - kv_head_num) * size_per_head + size_id;
        phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
      }
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
                         const int q_head_num,
                         const int kv_head_num,
                         const int seq_len,
                         const int size_per_head) {
  const int32_t elem_cnt =
      token_num * (q_head_num + kv_head_num * 2) * size_per_head;
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
                                                      q_head_num,
                                                      kv_head_num,
                                                      size_per_head);
}

template <typename T, int VecSize>
__global__ void write_pre_cache_to_kv_buffer(
    T *k_buf,  // [bsz, num_head, seq_len + pre_cache_length, head_dim]
    T *v_buf,
    const T *pre_key_cache,  // [bsz, num_head, pre_cache_length, head_dim]
    const T *pre_value_cache,
    const int *seq_lens,
    const int batch_size,
    const int pre_cache_length,
    const int num_head,
    const int head_dim,
    const int max_len_this_time,
    const int elem_cnt) {
  const int32_t hidden_size = pre_cache_length * head_dim;
  const int32_t cache_hidden_size = num_head * hidden_size;
  const int32_t fused_hidden_size = 2 * cache_hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int32_t batch_id = linear_index / fused_hidden_size;
    if (seq_lens[batch_id] == 0) continue;

    const int32_t cache_seq_id = (linear_index % hidden_size) / head_dim;
    const int32_t head_id = (linear_index % cache_hidden_size) / hidden_size;
    const int32_t size_id = linear_index % head_dim;
    const int32_t kv_id =
        (linear_index % fused_hidden_size) / cache_hidden_size;

    const int32_t read_id = batch_id * cache_hidden_size +
                            head_id * hidden_size + cache_seq_id * head_dim +
                            size_id;
    if (kv_id == 0) {
      phi::Load<T, VecSize>(&pre_key_cache[read_id], &src_vec);
    } else {
      phi::Load<T, VecSize>(&pre_value_cache[read_id], &src_vec);
    }

    const int tmp_max_len_this_time = max_len_this_time + pre_cache_length;
    const int32_t write_idx =
        batch_id * num_head * tmp_max_len_this_time * head_dim +
        head_id * tmp_max_len_this_time * head_dim + cache_seq_id * head_dim +
        size_id;
    if (kv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
    }
  }
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
                                                const int max_len_this_time,
                                                const int seq_len,
                                                const int pre_cache_length,
                                                const int token_num,
                                                const int q_head_num,
                                                const int kv_head_num,
                                                const int size_per_head) {
  const int fused_hidden_size = (q_head_num + 2 * kv_head_num) * size_per_head;
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
    const int32_t seq_id = ori_token_idx % seq_len;

    const int32_t head_id = bias_idx / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    const int tmp_max_len_this_time =
        max_len_this_time + (head_id < q_head_num ? 0 : pre_cache_length);
    const int tmp_seq_id =
        head_id < q_head_num ? seq_id : seq_id + pre_cache_length;

    if (head_id < q_head_num) {
      const int write_idx =
          target_batch_id * q_head_num * tmp_max_len_this_time * size_per_head +
          head_id * tmp_max_len_this_time * size_per_head +
          tmp_seq_id * size_per_head + size_id;
      phi::Store<T, VecSize>(src_vec, &q_buf[write_idx]);
    } else if (head_id < q_head_num + kv_head_num) {
      const int write_idx =
          target_batch_id * kv_head_num * tmp_max_len_this_time *
              size_per_head +
          (head_id - q_head_num) * tmp_max_len_this_time * size_per_head +
          tmp_seq_id * size_per_head + size_id;
      phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
    } else {
      const int write_idx = target_batch_id * kv_head_num *
                                tmp_max_len_this_time * size_per_head +
                            (head_id - q_head_num - kv_head_num) *
                                tmp_max_len_this_time * size_per_head +
                            tmp_seq_id * size_per_head + size_id;
      phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
    }
  }
}

template <typename T>
void qkv_transpose_split(
    const phi::GPUContext &dev_ctx,
    T *q_buf,
    T *k_buf,
    T *v_buf,
    const T *qkv,
    const T *pre_key_cache,  // [bsz, num_head, pre_cache_length, head_dim]
    const T *pre_value_cache,
    const int *padding_offset,
    const int *seq_lens,
    const int token_num,
    const int batch_size,
    const int q_head_num,
    const int kv_head_num,
    const int max_len_this_time,
    const int seq_len,
    const int pre_cache_length,
    const int size_per_head) {
  int32_t elem_cnt = token_num * (q_head_num + kv_head_num * 2) * size_per_head;

  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    common::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  int32_t pack_num = elem_cnt / PackSize;
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
                                                      max_len_this_time,
                                                      seq_len,
                                                      pre_cache_length,
                                                      token_num,
                                                      q_head_num,
                                                      kv_head_num,
                                                      size_per_head);
  if (pre_key_cache) {
    // stage 2: write pre_cache to kv_buf
    elem_cnt = batch_size * q_head_num * pre_cache_length * size_per_head * 2;
    pack_num = elem_cnt / PackSize;
    GetNumBlocks(pack_num, &grid_size);
    write_pre_cache_to_kv_buffer<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(k_buf,
                                                        v_buf,
                                                        pre_key_cache,
                                                        pre_value_cache,
                                                        seq_lens,
                                                        batch_size,
                                                        pre_cache_length,
                                                        kv_head_num,
                                                        size_per_head,
                                                        max_len_this_time,
                                                        elem_cnt);
  }
}

template <typename T, int VecSize>
__global__ void GetDecoderTensorKernel(const T *qkv_out,
                                       const int *cum_offsets,
                                       T *qkv_out_decoder,
                                       const int token_num,
                                       const int batch_size,
                                       const int q_head_num,
                                       const int kv_head_num,
                                       const int seq_len,
                                       const int dim_head,
                                       const int elem_nums,
                                       const int qkv_out_nums) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int32_t fused_hidden_size = (q_head_num + 2 * kv_head_num) * dim_head;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / fused_hidden_size;
    const int bias_idx = i % fused_hidden_size;
    const int ori_token_idx = bi * seq_len - cum_offsets[bi];
    const int src_offset = ori_token_idx * fused_hidden_size + bias_idx;
    if (src_offset >= qkv_out_nums) continue;
    phi::Load<T, VecSize>(&qkv_out[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &qkv_out_decoder[i]);
  }
}

template <typename T, int VecSize>
__global__ void GetDecoderRoPEKernel(const T *rope_emb,
                                     T *rope_out_emb,
                                     const int rope_bsz,
                                     const int batch_size,
                                     const int seq_len,
                                     const int dim_head,
                                     const int elem_nums) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const T *rope_cos_emb = rope_emb;
  const T *rope_sin_emb = rope_emb + rope_bsz * seq_len * dim_head;
  T *cos_emb = rope_out_emb;
  T *sin_emb = rope_out_emb + batch_size * dim_head;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / dim_head;
    const int src_offset = bi * seq_len * dim_head + i % dim_head;
    phi::Load<T, VecSize>(&rope_cos_emb[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &cos_emb[i]);
    phi::Load<T, VecSize>(&rope_sin_emb[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &sin_emb[i]);
  }
}

template <typename T>
void GetDecoderTensor(const phi::GPUContext &dev_ctx,
                      const phi::DenseTensor &qkv_out,
                      const phi::DenseTensor *rope_emb,
                      const int *cum_offsets,
                      phi::DenseTensor *qkv_out_decoder,
                      phi::DenseTensor *rope_out_emb,
                      const int token_num,
                      const int batch_size,
                      const int q_num_head,
                      const int kv_num_head,
                      const int seq_len,
                      const int dim_head) {
  // qkv_out: [token_num, 2 * kv_num_head + q_num_head, dim_head] -> [bs, 1, 2 *
  // kv_num_head + q_num_head, dim_head] rope: [2, bsz, 1, seq_len, dim_head] ->
  // [2, bsz, 1, 1, dim_head]
  int elem_nums = qkv_out_decoder->numel();
  int qkv_out_nums = qkv_out.numel();
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      dim_head % PackSize,
      0,
      common::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, PackSize));
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  GetDecoderTensorKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          qkv_out.data<T>(),
          cum_offsets,
          qkv_out_decoder->data<T>(),
          token_num,
          batch_size,
          q_num_head,
          kv_num_head,
          seq_len,
          dim_head,
          elem_nums,
          qkv_out_nums);
  if (rope_out_emb) {
    elem_nums = rope_out_emb->numel() / 2;
    pack_num = elem_nums / PackSize;
    GetNumBlocks(pack_num, &grid_size);
    GetDecoderRoPEKernel<float, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            rope_emb->data<float>(),
            rope_out_emb->data<float>(),
            rope_emb->dims()[1],
            batch_size,
            seq_len,
            dim_head,
            elem_nums);
  }
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return max(a, b);
  }
};

template <int THREADBLOCK_SIZE>
__global__ void GetMaxLenKernel(const int *seq_lens,
                                int *max_len,
                                const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    max_len_this_thread = max(seq_lens[i], max_len_this_thread);
  }
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  if (tid == 0) {
    *max_len = total;
  }
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

template <typename T, int VecSize>
__global__ void TransposeRemovingPadding(const T *input_data,
                                         const int *seq_lens,
                                         T *output_data,
                                         const int batch_size,
                                         const int num_head,
                                         const int max_len_this_time,
                                         const int seq_len,
                                         const int head_dim,
                                         const int token_num,
                                         const int elem_cnt,
                                         const int *padding_offset) {
  // transpose and remove padding
  // [batch_size, num_head, max_len_this_time, head_dim] -> [token_num,
  // num_head, head_dim]
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int dim_embed = num_head * head_dim;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / dim_embed;
    const int ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int ori_batch_id = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_batch_id] == 0) continue;
    const int ori_seq_id = ori_token_idx % seq_len;
    const int ori_head_id = (linear_index % dim_embed) / head_dim;
    const int ori_head_lane = (linear_index % dim_embed) % head_dim;
    const int ori_idx = ori_batch_id * num_head * max_len_this_time * head_dim +
                        ori_head_id * max_len_this_time * head_dim +
                        ori_seq_id * head_dim + ori_head_lane;
    phi::Load<T, VecSize>(&input_data[ori_idx], &src_vec);
    phi::Store<T, VecSize>(src_vec, &output_data[linear_index]);
  }
}

template <typename T>
void InvokeTransposeRemovePadding(const phi::GPUContext &dev_ctx,
                                  const T *input_data,
                                  const int *seq_lens,
                                  T *output_data,
                                  const int batch_size,
                                  const int num_head,
                                  const int max_len_this_time,
                                  const int seq_len,
                                  const int head_dim,
                                  const int token_num,
                                  const int *padding_offset) {
  // [batch_size, num_head, max_len_this_time, head_dim] -> [token_num,
  // num_head, head_dim]
  constexpr int VEC_16B = 16;
  const int elem_cnt = token_num * num_head * head_dim;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      head_dim % PackSize,
      0,
      common::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", head_dim, PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t block_size = 128;
  int32_t grid_size = (pack_num + block_size - 1) / block_size;
  TransposeRemovingPadding<T, PackSize>
      <<<grid_size, block_size, 0, dev_ctx.stream()>>>(input_data,
                                                       seq_lens,
                                                       output_data,
                                                       batch_size,
                                                       num_head,
                                                       max_len_this_time,
                                                       seq_len,
                                                       head_dim,
                                                       token_num,
                                                       elem_cnt,
                                                       padding_offset);
}

template <typename T>
static void VLOGMatrix(const T *mat_d,
                       int num,
                       std::string name,
                       int max_num = 10) {
  num = num < max_num ? num : max_num;
  std::vector<T> tmp(num);
  GPU(Memcpy)(tmp.data(), mat_d, sizeof(T) * num, GPU(MemcpyDeviceToHost));

  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    if (std::is_same<T, int8_t>::value) {
      ss << static_cast<int>(tmp[i]) << ", ";
    } else {
      ss << std::setprecision(8) << (float)(tmp[i]) << ", ";  // NOLINT
    }
  }
  LOG(INFO) << "===>: " << name;
  LOG(INFO) << ss.str();
}

}  // namespace fusion
}  // namespace phi
