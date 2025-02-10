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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/mmha_util.cu.h"

namespace phi {
namespace fusion {

#ifndef PADDLE_WITH_HIP

constexpr unsigned int str2int(const char *str, int h = 0) {
  return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

template <typename T>
struct Masked_multihead_attention_params {
  float *qk_sum_max_split_seq;
  float *split_out;
  // qkv_out, [B, 1(seq_len), 3, num_head * dim_head]
  const T *qkv;
  // bias, [3, num_head, dim_head]
  T *qkv_bias;
  // [2, B, num_head, max_seq_len(valid cache_seq_len), dim_head]
  // k [B, num_head, dim_head/x, max_seq_len, x], that is `seq_len` first
  // v [B, num_head, max_seq_len, dim_head]
  T *cache_kv;
  // [B, max_seq_len]
  const int *beam_cache_offset = nullptr;

  const int *sequence_lengths{nullptr};

  // The RoPE embedding, [2, B, rotary_seq_len, 1, dim_head]
  // rotary_emb_dims = 1 if pos_ids_extra is null else 2
  const float *rotary_emb;

  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;

  int rotary_emb_dims;
  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;
  int num_head;
  // k_num_head and v_num_head must be equal, we unify them.
  // kv_num_head = k_num_head && kv_num_head == v_num_head
  int kv_num_head;
  int timestep;  // cache_seq_length
  int max_seq_length;

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;

  int steps_per_block;
  int split_seq = 1;

  bool add_qkv_bias;
  bool neox_rotary_style;
  // whether to broadcast num_heads(2nd) dimension for attn_mask
  // in MMHA, if false, attn_mask shape should be
  // [bsz, num_heads, 1, time_step(cache_seq_length)+1]
  bool mask_broadcast_num_heads;
};

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc,
          bool SPLIT>
__global__ void masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params,
    LoadFunc load_func,
    StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int bi = blockIdx.z;
  // params.sequence_lengths[bi] means how many k and v we have cached in
  // cache_kv.
  if (params.sequence_lengths && params.sequence_lengths[bi] < 0) {
    return;
  }

  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

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
  const int hi = blockIdx.y;
  const int bhi = bi * params.num_head + hi;

  const int kv_num_head = params.kv_num_head;
  const int num_head_per_group = params.num_head / kv_num_head;
  // hi means the head index in query processed by this cuda thread.
  // kv_bhi means the merged batch and head index in key and value processed by
  // this cuda thread.
  const int kv_bhi = bi * kv_num_head + hi / num_head_per_group;

  const int bbhi = bbi * params.beam_width * params.num_head + hi;
  const int tid = threadIdx.x;

  const int bi_seq_len_offset = bi * params.max_seq_length;

  float qk_max = -FLT_MAX;
  float qk = 0;

  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];

  // with SPLIT, The last single q*k*v is computed by the last threadBlock of
  // split_index
  const int split_index = blockIdx.x;
  int start_seq = 0;
  int end_seq = act_time_step;
  bool is_last_block = (SPLIT == false);
  int real_split_each_batch = (act_time_step - 1) / params.steps_per_block + 1;
  if constexpr (SPLIT) {
    if (split_index >= real_split_each_batch) return;

    start_seq = split_index * params.steps_per_block;
    end_seq = start_seq + params.steps_per_block;
    if (split_index == real_split_each_batch - 1) {
      is_last_block = true;
      end_seq = act_time_step;
    }
  }
  int curr_seq_section = end_seq - start_seq;

  // qkv [B, S=1, num_head + 2 * kv_num_head, head_dim]
  // this hi means the head index in query!
  int qkv_base_offset = bi * (params.num_head + 2 * kv_num_head) * Dh + hi * Dh;

  // QK_VEC_SIZE is only used for compute q dot k .
  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
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

  // q and k have only head_dim scalar elements.
  // below only compute q dot k = 1 element.
  // q has QK_VECS_PER_WARP elements, [Qk_vec, Qk_vec, ..., Qk_vec]
  // k has QK_VECS_PER_WARP elements: [Qk_vec, Qk_vec, ..., Qk_vec]
  // per cuda thread read a Qk_vec of q and k and compute q dot k.
  if (tid < QK_VECS_PER_WARP) {
    int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    int q_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
    int k_bias_offset = hi / num_head_per_group * Dh + tid * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    // q = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
    //         ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_offset])
    //         : q;
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset);
    }

    Qk_vec k;
    zero(k);
    // k = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
    //         ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_offset])
    //         : k;
    if ((Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) && is_last_block) {
      load_func.template load<Qk_vec>(k,
                                      params.num_head * Dh + qk_offset -
                                          hi * Dh +
                                          hi / num_head_per_group * Dh);
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
        const float *sin_base = params.rotary_emb + params.batch_size * Dh;
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
        const float *sin_base = params.rotary_emb + params.batch_size * Dh;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = tid / stride_all_lastdim * stride_all_lastdim +
                       (tid + stride) % (stride_all_lastdim);
        int qk_right_offset = qkv_base_offset + right_id * QK_VEC_SIZE;
        int q_right_bias_offset = hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_bias_offset =
            hi / num_head_per_group * Dh + right_id * QK_VEC_SIZE;
        Qk_vec q_right;
        zero(q_right);
        // q_right =
        //     (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //         ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_right_offset])
        //         : q_right;
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(q_right, qk_right_offset);
        }
        Qk_vec k_right;
        zero(k_right);
        // k_right =
        //     (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //         ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_right_offset])
        //         : k_right;
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(k_right,
                                          params.num_head * Dh +
                                              qk_right_offset - hi * Dh +
                                              hi / num_head_per_group * Dh);
        }

        if (params.add_qkv_bias) {
          Qk_vec q_right_bias;
          zero(q_right_bias);
          q_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
                             ? *reinterpret_cast<const Qk_vec *>(
                                   &q_bias_base[q_right_bias_offset])
                             : q_right_bias;
          Qk_vec k_right_bias;
          zero(k_right_bias);
          k_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
                             ? *reinterpret_cast<const Qk_vec *>(
                                   &k_bias_base[k_right_bias_offset])
                             : k_right_bias;

          q_right = add(q_right, q_right_bias);
          k_right = add(k_right, k_right_bias);
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

    if (is_last_block) {
      int co = tid / QK_VECS_IN_16B;
      int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;
      int offset = kv_bhi * params.max_seq_length * Dh +
                   co * params.max_seq_length * QK_ELTS_IN_16B +
                   act_time_step * QK_ELTS_IN_16B + ci;
      if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
        *reinterpret_cast<Qk_vec *>(&params.cache_kv[offset]) = k;
      }

      qk = dot<Qk_vec, Qk_vec>(q, k);
      // QK_VECS_PER_WARP is <= WARP_SIZE, reduce it within a warp!
      if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
        for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
          qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
        }
      }
    }
  }

  // when QK_VECS_PER_WARP > WARP_SIZE, we need to reduce the qk in smem!
  if (QK_VECS_PER_WARP > WARP_SIZE && is_last_block) {
    constexpr int WARPS_PER_RED =
        (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
    qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
  }

  // Let only the last cuda ThreadBlock compute the final q*k.
  if (tid == 0 && is_last_block) {
    // NOTE(wangxi): mask must be 0.0
    // T mask = params.attn_mask[
    //    bi * (params.timestep + 1) + params.timestep];
    // qk += static_cast<float>(mask);
    qk *= params.inv_sqrt_dh;
    if (params.attn_mask) {
      auto mask_bhi = params.mask_broadcast_num_heads ? bi : bhi;
      T mask =
          params.attn_mask[mask_bhi * (params.timestep + 1) + act_time_step];
      qk += static_cast<float>(mask);
    }
    qk_max = qk;
    qk_smem[act_time_step - start_seq] = qk;
  }
  __syncthreads();

  using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
  constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  int ko = tid / THREADS_PER_KEY + start_seq;
  int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;

  static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD, "");

  K_vec q[K_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
    q[i] = *reinterpret_cast<const K_vec *>(
        &q_smem[ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
  }

  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  T *k_cache = &params.cache_kv[kv_bhi * params.max_seq_length * Dh + ki];
  T *k_cache_batch = &params.cache_kv[bbhi * params.max_seq_length * Dh + ki];
  int ti_end = div_up(curr_seq_section, K_PER_WARP) * K_PER_WARP + start_seq;

  const int *beam_offsets = params.beam_cache_offset
                                ? &params.beam_cache_offset[bi_seq_len_offset]
                                : nullptr;

#pragma unroll
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    const int beam_offset = beam_offsets ? beam_offsets[ti] * params.num_head *
                                               params.max_seq_length * Dh
                                         : 0;
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + ti;
      if (ti < end_seq) {
        if (beam_offset) {
          k[ii] =
              (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                  ? *reinterpret_cast<const K_vec *>(
                        &k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B])
                  : k_vec_zero;
        } else {
          k[ii] =
              (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                  ? *reinterpret_cast<const K_vec *>(
                        &k_cache[jj * QK_ELTS_IN_16B])
                  : k_vec_zero;
        }
      }
    }

    // NOTE(liyurui): We should multiple q with inv_sqrt_dh first, for dot(q, k)
    // may overflow with FP16 in large model.
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

    // bool is_mask = false;
    if (ti < end_seq && tid % THREADS_PER_KEY == 0) {
      // qk_max = is_mask ? qk_max : fmaxf(qk_max, qk);
      auto mask_bhi = params.mask_broadcast_num_heads ? bi : bhi;
      // T mask = params.attn_mask[mask_bhi * (params.timestep + 1) + ti];
      if (params.attn_mask) {
        T mask = params.attn_mask[mask_bhi * (params.timestep + 1) + ti];
        qk += static_cast<float>(mask);
      }
      qk_max = fmaxf(qk_max, qk);

      qk_smem[ti - start_seq] = qk;
    }
  }

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;

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

  int useful_smem_index =
      is_last_block ? curr_seq_section : curr_seq_section - 1;
  float sum = 0.f;
  for (int ti = tid; ti <= useful_smem_index; ti += THREADS_PER_BLOCK) {
    // bool is_mask = false;
    // float logit = is_mask ? 0.f : __expf(qk_smem[ti] - qk_max);
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  int bhsi = bhi * params.split_seq;
  if (SPLIT && tid == 0) {
    float2 sum_max = {sum, qk_max};
    *reinterpret_cast<float2 *>(
        &params.qk_sum_max_split_seq[(bhsi + split_index) * 2]) = sum_max;
  }

  // FIXME(wangxi): need add 1.e-6f?
  float inv_sum = __fdividef(1.f, sum + 1.e-6f);

  for (int ti = tid; ti <= useful_smem_index; ti += THREADS_PER_BLOCK) {
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }

  __syncthreads();

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

  // now we have got [1, seq] ，distributed in logits_smem.
  // next we compute [1, seq] * [seq, head_dim] = [1, head_dim]
  // THREADS_PER_VALUE means num of threads per value's head_dim.
  // we split the seq dimension for more cuda threads to compute.
  // vo means the first seq index processed by this cuda thread in the value.
  // vi means the head_dim index processed by this cuda thread in the value.
  // so this cuda thread compute [1, k] * [k, vi:vi+V_VEC_SIZE] and k starts
  // from vo and increases by a step V_PER_ITER.
  int vo = tid / THREADS_PER_VALUE + start_seq;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

  T *v_cache = &params.cache_kv[params.cache_batch_size * kv_num_head *
                                    params.max_seq_length * Dh +
                                kv_bhi * params.max_seq_length * Dh + vi];
  T *v_cache_batch = &params.cache_kv[params.batch_size * params.num_head *
                                          params.max_seq_length * Dh +
                                      bbhi * params.max_seq_length * Dh + vi];

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif

  V_vec_acum out;
  zero(out);
  // V_PER_ITER is used to strip-mined the seq dimension.
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int ti = vo; ti < end_seq; ti += V_PER_ITER) {
      const int beam_offset =
          beam_offsets
              ? beam_offsets[ti] * params.num_head * params.max_seq_length * Dh
              : 0;
      V_vec v;
      if (beam_offset) {
        v = *reinterpret_cast<const V_vec *>(
            &v_cache_batch[beam_offset + ti * Dh]);
      } else {
        v = *reinterpret_cast<const V_vec *>(&v_cache[ti * Dh]);
      }
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      float logit = logits_smem[ti - start_seq];
      out = fma(logit, cast_to_float(v), out);
#else
      DataType_ logit = static_cast<DataType_>(logits_smem[ti - start_seq]);
      // Update the partial sums.
      out = fma(logit, v, out);
#endif
    }
  }

  V_vec v_bias;
  zero(v_bias);
  // now we process the last v.
  if (vo == (act_time_step % V_PER_ITER + start_seq) &&
      (Dh == Dh_MAX || vi < Dh) && is_last_block) {
    // V_vec v = *reinterpret_cast<const V_vec *>(
    //     &params.qkv[2 * params.num_head * Dh + qkv_base_offset + vi]);
    V_vec v;
    load_func.template load<V_vec>(v,
                                   qkv_base_offset + vi - hi * Dh +
                                       params.num_head * Dh + kv_num_head * Dh +
                                       hi / num_head_per_group * Dh);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[(kv_num_head + params.num_head) * Dh +
                           hi / num_head_per_group * Dh + vi]);
      v = add(v, v_bias);
    }

    *reinterpret_cast<V_vec *>(&v_cache[act_time_step * Dh]) = v;

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    out = fma(logits_smem[act_time_step - start_seq], cast_to_float(v), out);
#else
    out = fma(logits_smem[act_time_step - start_seq], v, out);
#endif
  }

  __syncthreads();

  // now we do the reduction in the seq dimension to get [1, head_dim].
  if (Dh == Dh_MAX || vi < Dh) {
    int vo_blk = vo - start_seq;  // vo id of current block
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2;
         active_groups /= 2) {
      int midpoint = active_groups / 2;
      if (vo_blk >= midpoint && vo_blk < active_groups &&
          (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(*reinterpret_cast<V_vec *>(
                               &out_smem[(vo_blk - midpoint) * Dh + vi]),
                           out);
#else
        *reinterpret_cast<V_vec *>(&out_smem[(vo_blk - midpoint) * Dh + vi]) =
            out;
#endif
      }
      __syncthreads();
      if (vo_blk < midpoint && (Dh == Dh_MAX || vi < Dh)) {
        out = add(*reinterpret_cast<const V_vec *>(&out_smem[vo_blk * Dh + vi]),
                  out);
      }
      __syncthreads();
    }
  }

  if (vo == start_seq && (Dh == Dh_MAX || vi < Dh)) {
    if (SPLIT && real_split_each_batch > 1) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      *(reinterpret_cast<V_vec_acum *>(
          &params.split_out[(bhsi + split_index) * Dh + vi])) = out;
#else
      *(reinterpret_cast<V_vec_acum_fp32_<V_vec>::Type *>(
          &params.split_out[(bhsi + split_index) * Dh + vi])) =
          cast_to_float(out);
#endif
    } else {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      V_vec tmp_out;
      convert_from_float(tmp_out, out);
      store_func.template store<V_vec>(tmp_out, bhi * Dh + vi);
#else
      store_func.template store<V_vec>(out, bhi * Dh + vi);
#endif
    }
  }

#else
  assert(false);
#endif
}

template <typename T, int Dh, int Dh_MAX, typename StoreFunc>
__global__ void post_process_kernel(Masked_multihead_attention_params<T> params,
                                    StoreFunc store_func) {
  const int bi = blockIdx.y;
  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];
  int real_split_each_batch = (act_time_step - 1) / params.steps_per_block + 1;
  if (real_split_each_batch <= 1) {
    return;
  }

  const int tid = threadIdx.x;
  const int hi = blockIdx.x;
  const int bhi = (bi * params.num_head + hi);
  const int bhsi = (bi * params.num_head + hi) * params.split_seq;
  extern __shared__ float2 qk_sum_max_smem[];

  for (int i = tid; i < real_split_each_batch; i += blockDim.x) {
    qk_sum_max_smem[i] = *reinterpret_cast<float2 *>(
        &params.qk_sum_max_split_seq[(bhsi + i) * 2]);
  }
  __syncthreads();

  float max = -FLT_MAX;
  float sum = 0;
  float v = 0;
  if (tid < Dh) {
#pragma unroll
    for (int i = 0; i < real_split_each_batch; ++i) {
      float2 sum_max = qk_sum_max_smem[i];
      float tmp_max = sum_max.y;
      max = tmp_max > max ? tmp_max : max;
    }
#pragma unroll
    for (int i = 0; i < real_split_each_batch; ++i) {
      float2 sum_max = qk_sum_max_smem[i];
      // split_out:[bsz , num_head, split_seq, dim_head]
      float this_v = params.split_out[(bhsi + i) * Dh + tid];

      float real_this_sum = sum_max.x * __expf(sum_max.y - max);
      v += real_this_sum * this_v;
      sum += real_this_sum;
    }

    v /= sum;
    T tmp_v = (T)v;
    store_func.template store<T>(tmp_v, bhi * Dh + tid);
    // params.out[bhi * Dh + tid] = (T)(v);
  }
}

template <typename T, bool SPLIT>
inline size_t smem_size_in_bytes(
    const Masked_multihead_attention_params<T> &params,
    int dim_head,
    int threads_per_value,
    int threads_per_block) {
  // for qk_smem and logits_smem(both float)
  size_t qk_sz = div_up(params.timestep, 4) * 16;
  if (SPLIT) {
    qk_sz = div_up(params.steps_per_block, 4) * 16;
  }
  // for reduce (logits dot V) result
  int rows_per_red = threads_per_block / threads_per_value;
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;
  return max(qk_sz, red_sz);
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
  size_t smem_sz = smem_size_in_bytes<T, SPLIT>(                          \
      params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);                        \
  constexpr auto kernel_fn =                                              \
      masked_multihead_attention_kernel<T,                                \
                                        Dh,                               \
                                        Dh_MAX,                           \
                                        THDS_PER_KEY,                     \
                                        THDS_PER_VALUE,                   \
                                        THDS_PER_BLOCK,                   \
                                        decltype(load_func),              \
                                        decltype(store_func),             \
                                        SPLIT>;                           \
  if (smem_sz > 0xc000) {                                                 \
    cudaFuncSetAttribute(                                                 \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz); \
  }                                                                       \
  dim3 grid(params.split_seq, params.num_head, params.batch_size);        \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                   \
      params, load_func, store_func)

template <typename T,
          int Dh,
          int Dh_MAX,
          typename LoadFunc,
          typename StoreFunc,
          bool SPLIT>
void fmha_launch_kernel(const Masked_multihead_attention_params<T> &params,
                        const cudaStream_t &stream,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  // If try adjusting the hyperparam, THDS_PER_KEY can try [1, 2, 4]
  // for LLM: multiBatch(8)/longSeq(>2048) case, reduce THDS_PER_KEY may work
  // for super longSeq(>3072) case, larger steps_per_block（256） may work
  if constexpr (SPLIT) {
    MMHA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       4,
                       THREADS_PER_VALUE,
                       128,
                       stream,
                       load_func,
                       store_func);
  } else {
    if (params.timestep < 32) {
      MMHA_LAUNCH_KERNEL(T,
                         Dh,
                         Dh_MAX,
                         4,
                         THREADS_PER_VALUE,
                         64,
                         stream,
                         load_func,
                         store_func);
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
}

#define FMHA_LAUNCH_KERNEL(dim_head_, dim_head_max_, stream)              \
  case dim_head_:                                                         \
    fmha_launch_kernel<T,                                                 \
                       dim_head_,                                         \
                       dim_head_max_,                                     \
                       decltype(load_func),                               \
                       decltype(store_func),                              \
                       SPLIT>(params, stream, load_func, store_func);     \
    if (SPLIT) {                                                          \
      post_process_kernel<T, dim_head_, dim_head_max_>                    \
          <<<grid, dim_head_max_, smem_sz, stream>>>(params, store_func); \
    }                                                                     \
    break;

template <typename T, typename LoadFunc, typename StoreFunc, bool SPLIT>
void fmha_impl(const phi::GPUContext &dev_ctx,
               const Masked_multihead_attention_params<T> &params,
               int dim_head,
               LoadFunc load_func,
               StoreFunc store_func) {
  dim3 grid(params.num_head, params.batch_size);
  int smem_sz = params.split_seq * sizeof(float2);
  auto stream = dev_ctx.stream();
  switch (dim_head) {
    FMHA_LAUNCH_KERNEL(16, 32, stream)
    FMHA_LAUNCH_KERNEL(32, 32, stream)
    FMHA_LAUNCH_KERNEL(64, 64, stream)
    FMHA_LAUNCH_KERNEL(80, 128, stream)
    FMHA_LAUNCH_KERNEL(96, 128, stream)
    FMHA_LAUNCH_KERNEL(128, 128, stream)
    FMHA_LAUNCH_KERNEL(192, 256, stream)
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Dim_head = %d is unsupported!", dim_head));
  }
}

template <typename T, bool SPLIT = false>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const Masked_multihead_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const phi::DenseTensor *dequant_qkv_scales = nullptr,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore<T, int8_t> store_func(out_tensor->data<int8_t>(),
                                    quant_round_type,
                                    quant_fmha_out_scale,
                                    quant_max_bound,
                                    quant_min_bound);
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, int8_t> store_func(out_tensor->data<int8_t>(),
                                    quant_round_type,
                                    quant_fmha_out_scale,
                                    quant_max_bound,
                                    quant_min_bound);
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore<T> store_func(out_tensor->data<T>());
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T> store_func(out_tensor->data<T>());
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  }
}

template <typename T, bool SPLIT = false>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const phi::DenseTensor &shift,
                  const phi::DenseTensor &smooth,
                  const Masked_multihead_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const phi::DenseTensor *dequant_qkv_scales = nullptr,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          num_head * dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          num_head * dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     num_head * dim_head);
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     num_head * dim_head);
    fmha_impl<T, decltype(load_func), decltype(store_func), SPLIT>(
        dev_ctx, params, dim_head, load_func, store_func);
  }
}

struct NormalVersion {};
struct UnusedVersion {};

template <typename T>
struct DispatchDtypeTrait {
  using FuncVersion = NormalVersion;
};

template <>
struct DispatchDtypeTrait<int32_t> {
  using FuncVersion = UnusedVersion;
};

template <typename T, typename Context>
void DispatchWithDtype(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &cache_kv,
                       const paddle::optional<DenseTensor> &bias,
                       const paddle::optional<DenseTensor> &src_mask,
                       const paddle::optional<DenseTensor> &cum_offsets,
                       const paddle::optional<DenseTensor> &sequence_lengths,
                       const paddle::optional<DenseTensor> &rotary_tensor,
                       const paddle::optional<DenseTensor> &beam_cache_offset,
                       const paddle::optional<DenseTensor> &qkv_out_scale,
                       const paddle::optional<DenseTensor> &out_shift,
                       const paddle::optional<DenseTensor> &out_smooth,
                       int seq_len,
                       int rotary_emb_dims,
                       const bool use_neox_rotary_style,
                       const float out_scale,
                       const int quant_round_type,
                       const float quant_max_bound,
                       const float quant_min_bound,
                       DenseTensor *out,
                       DenseTensor *cache_kv_out,
                       DenseTensor *beam_cache_offset_out,
                       NormalVersion) {
  const auto &x_dims = x.dims();
  int bsz = x_dims[0];
  int cache_bsz = cache_kv.dims()[1];
  int max_seq_len = cache_kv.dims()[3];
  int dim_head = cache_kv.dims()[4];
  int timestep = max_seq_len;
  float inv_sqrt_dh = 1. / sqrt(dim_head);

  int k_num_head = cache_kv.dims()[2];
  int v_num_head = k_num_head;
  // this num_head means query's head
  int num_head =
      x.dims()[x.dims().size() - 1] / dim_head - k_num_head - v_num_head;

  Masked_multihead_attention_params<T> params;

  bool mask_broadcast_num_heads = true;

  params.add_qkv_bias = false;
  if (bias) {
    params.add_qkv_bias = true;
    params.qkv_bias = const_cast<T *>(bias->data<T>());
  }

  if (src_mask) {
    if (src_mask->dims()[1] == 1) {
      mask_broadcast_num_heads = true;
    } else if (src_mask->dims()[1] == num_head) {
      mask_broadcast_num_heads = false;
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Unknown dimension for attn_mask, the num_head(2nd) "
          "dimension is invalid, it should be 1 or num_head(%d), "
          "but got %d",
          num_head,
          src_mask->dims()[1]));
    }
    params.attn_mask = src_mask->data<T>();
    timestep = src_mask->dims()[3] - 1;
  }

  if (out_scale > 0) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  if (sequence_lengths) {
    params.sequence_lengths = sequence_lengths->data<int>();
  }

  if (cum_offsets) {
    PADDLE_THROW(common::errors::PermissionDenied(
        "Current mmha kernel does not support cum_offsets param."));
  }

  if (rotary_emb_dims > 0) {
    params.rotary_emb = rotary_tensor->data<float>();
  } else {
    params.rotary_emb = nullptr;
  }

  if (beam_cache_offset) {
    params.beam_cache_offset = beam_cache_offset->data<int>();
    params.beam_width = beam_cache_offset->dims()[1];
  }

  params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  params.cache_kv = const_cast<T *>(cache_kv_out->data<T>());
  params.neox_rotary_style = use_neox_rotary_style;
  params.batch_size = bsz;
  params.cache_batch_size = cache_bsz;
  params.num_head = num_head;
  params.kv_num_head = k_num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_len;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;

  params.steps_per_block = timestep;  // if not SPLIT, this is unuseful.
  params.split_seq = 1;               // if not SPLIT, grid.x==1

  bool SPLIT = false;
  if (bsz <= 4 && timestep >= 512) {
    SPLIT = true;
  }
  if (SPLIT) {
    const int steps_per_block = 128;
    params.steps_per_block = steps_per_block;
    params.split_seq = (timestep - 1) / steps_per_block + 1;
    int split_seq = params.split_seq;

    phi::DenseTensor qk_sum_max_split_seq;
    // 2 means sum and max.
    qk_sum_max_split_seq.Resize({{bsz, num_head, split_seq, 2}});
    dev_ctx.template Alloc<float>(&qk_sum_max_split_seq,
                                  qk_sum_max_split_seq.numel() * sizeof(float));
    params.qk_sum_max_split_seq = qk_sum_max_split_seq.data<float>();

    phi::DenseTensor split_out;
    split_out.Resize({{bsz, num_head, split_seq, dim_head}});
    dev_ctx.template Alloc<float>(&split_out,
                                  split_out.numel() * sizeof(float));
    params.split_out = split_out.data<float>();

    if (out_shift) {
      DispatchFMHA<T, true>(dev_ctx,
                            x,
                            *(out_shift.get_ptr()),
                            *(out_smooth.get_ptr()),
                            params,
                            num_head,
                            dim_head,
                            out,
                            qkv_out_scale.get_ptr(),
                            out_scale,
                            quant_round_type,
                            quant_max_bound,
                            quant_min_bound);
    } else {
      DispatchFMHA<T, true>(dev_ctx,
                            x,
                            params,
                            num_head,
                            dim_head,
                            out,
                            qkv_out_scale.get_ptr(),
                            out_scale,
                            quant_round_type,
                            quant_max_bound,
                            quant_min_bound);
    }
  } else {
    if (out_shift) {
      DispatchFMHA<T, false>(dev_ctx,
                             x,
                             *(out_shift.get_ptr()),
                             *(out_smooth.get_ptr()),
                             params,
                             num_head,
                             dim_head,
                             out,
                             qkv_out_scale.get_ptr(),
                             out_scale,
                             quant_round_type,
                             quant_max_bound,
                             quant_min_bound);
    } else {
      DispatchFMHA<T, false>(dev_ctx,
                             x,
                             params,
                             num_head,
                             dim_head,
                             out,
                             qkv_out_scale.get_ptr(),
                             out_scale,
                             quant_round_type,
                             quant_max_bound,
                             quant_min_bound);
    }
  }
}

template <typename T, typename Context>
void DispatchWithDtype(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &cache_kv,
                       const paddle::optional<DenseTensor> &bias,
                       const paddle::optional<DenseTensor> &src_mask,
                       const paddle::optional<DenseTensor> &cum_offsets,
                       const paddle::optional<DenseTensor> &sequence_lengths,
                       const paddle::optional<DenseTensor> &rotary_tensor,
                       const paddle::optional<DenseTensor> &beam_cache_offset,
                       const paddle::optional<DenseTensor> &qkv_out_scale,
                       const paddle::optional<DenseTensor> &out_shift,
                       const paddle::optional<DenseTensor> &out_smooth,
                       int seq_len,
                       int rotary_emb_dims,
                       const bool use_neox_rotary_style,
                       const float out_scale,
                       const int quant_round_type,
                       const float quant_max_bound,
                       const float quant_min_bound,
                       DenseTensor *out,
                       DenseTensor *cache_kv_out,
                       DenseTensor *beam_cache_offset_out,
                       UnusedVersion) {}

#endif  // PADDLE_WITH_HIP

template <typename T, typename Context>
void MMHAKernel(const Context &dev_ctx,
                const DenseTensor &x,
                const DenseTensor &cache_kv,
                const paddle::optional<DenseTensor> &bias,
                const paddle::optional<DenseTensor> &src_mask,
                const paddle::optional<DenseTensor> &cum_offsets,
                const paddle::optional<DenseTensor> &sequence_lengths,
                const paddle::optional<DenseTensor> &rotary_tensor,
                const paddle::optional<DenseTensor> &beam_cache_offset,
                const paddle::optional<DenseTensor> &qkv_out_scale,
                const paddle::optional<DenseTensor> &out_shift,
                const paddle::optional<DenseTensor> &out_smooth,
                int seq_len,
                int rotary_emb_dims,
                const bool use_neox_rotary_style,
                const std::string &compute_dtype,
                const float out_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                DenseTensor *out,
                DenseTensor *cache_kv_out,
                DenseTensor *beam_cache_offset_out) {
#ifndef PADDLE_WITH_HIP
  if (x.dtype() == phi::DataType::INT32) {
    switch (str2int(compute_dtype.c_str())) {
      case str2int("fp16"):
        DispatchWithDtype<phi::dtype::float16, Context>(
            dev_ctx,
            x,
            cache_kv,
            bias,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out,
            cache_kv_out,
            beam_cache_offset_out,
            typename DispatchDtypeTrait<phi::dtype::float16>::FuncVersion{});
        break;
#if CUDA_VERSION >= 11000
      case str2int("bf16"):
        DispatchWithDtype<phi::dtype::bfloat16, Context>(
            dev_ctx,
            x,
            cache_kv,
            bias,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out,
            cache_kv_out,
            beam_cache_offset_out,
            typename DispatchDtypeTrait<phi::dtype::bfloat16>::FuncVersion{});
        break;
#endif
      case str2int("fp32"):
        DispatchWithDtype<float, Context>(
            dev_ctx,
            x,
            cache_kv,
            bias,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out,
            cache_kv_out,
            beam_cache_offset_out,
            typename DispatchDtypeTrait<float>::FuncVersion{});
        break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "In the case of quantization enabled with Input(x) INT32, "
            "Attr(compute_dtype) must be set in (bf16, fp16, fp32), "
            "but get compute_dtype (%s)",
            compute_dtype));
    }
  } else {
    DispatchWithDtype<T, Context>(
        dev_ctx,
        x,
        cache_kv,
        bias,
        src_mask,
        cum_offsets,
        sequence_lengths,
        rotary_tensor,
        beam_cache_offset,
        qkv_out_scale,
        out_shift,
        out_smooth,
        seq_len,
        rotary_emb_dims,
        use_neox_rotary_style,
        out_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        out,
        cache_kv_out,
        beam_cache_offset_out,
        typename DispatchDtypeTrait<T>::FuncVersion{});
  }
#endif  // PADDLE_WITH_HIP
}

}  // namespace fusion
}  // namespace phi

#if CUDA_VERSION >= 11000
PD_REGISTER_KERNEL(masked_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMHAKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int32_t) {}
#else
PD_REGISTER_KERNEL(masked_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMHAKernel,
                   float,
                   phi::dtype::float16,
                   int32_t) {}
#endif
