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
struct QkvUnpackMhaParams {
  const T *q;  // B, 1, num_head * dim_head
  const T *k;  // B, seq_len), num_head * dim_head
  const T *v;  // B, seq_len, num_head * dim_head
  T *out;

  const T *attn_mask;
  int mask_length;

  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;
  int num_head;
  // k_num_head and v_num_head must be equal, we unify them.
  // kv_num_head = k_num_head && kv_num_head == v_num_head
  int kv_num_head;
  int timestep;  // cache_seq_length

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;
};

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc>
__global__ void qkv_attention_kernel(QkvUnpackMhaParams<T> params,
                                     LoadFunc load_func,
                                     StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int bi = blockIdx.y;

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
  const int hi = blockIdx.x;
  const int bhi = bi * params.num_head + hi;

  const int kv_num_head = params.kv_num_head;
  const int num_head_per_group = params.num_head / kv_num_head;

  const int kv_bhi = bi * kv_num_head + hi / num_head_per_group;

  const int bbhi = bbi * params.beam_width * params.num_head + hi;

  const int tid = threadIdx.x;

  float qk_max = -FLT_MAX;
  float qk = 0;

  int act_time_step = params.timestep;

  int qkv_base_offset = bi * (params.num_head) * Dh + hi * Dh;

  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  // load q element to q smem
  if (tid < QK_VECS_PER_WARP) {
    int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    int q_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
    int k_bias_offset = hi / num_head_per_group * Dh + tid * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset);
    }

    *reinterpret_cast<Qk_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;
  }

  __syncthreads();

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
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  // each thread process act_time_step
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);

#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      if (ti < act_time_step) {
        k[ii] = *reinterpret_cast<const K_vec *>(
            &params.k[bi * params.timestep * params.num_head * Dh +
                      ti * params.num_head * Dh + ki +
                      ii * THREADS_PER_KEY * K_VEC_SIZE + hi * Dh]);
      }
    }

    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);
    const T *q_ptr = reinterpret_cast<const T *>(q);
    const T *k_ptr = reinterpret_cast<const T *>(k);

    if (ti < act_time_step && tid % THREADS_PER_KEY == 0) {
      qk_max = fmaxf(qk_max, qk);

      qk_smem[ti] = qk;
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

  float sum = 0.f;
  for (int ti = tid; ti < act_time_step; ti += THREADS_PER_BLOCK) {
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  float inv_sum = __fdividef(1.f, sum + 1.e-6f);

  for (int ti = tid; ti < act_time_step; ti += THREADS_PER_BLOCK) {
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }
  __syncthreads();

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;  // 128 / 16 = 8
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif

  V_vec_acum out;
  zero(out);
  // V_PER_ITER is used to strip-mined the seq dimension.
  constexpr int V_PER_ITER =
      THREADS_PER_BLOCK / THREADS_PER_VALUE;  // 128 / 16 == 8?
  if (Dh == Dh_MAX || vi < Dh) {
    for (int ti = vo; ti < act_time_step; ti += V_PER_ITER) {
      // 8 x float16
      V_vec v;

      // update here
      v = *reinterpret_cast<const V_vec *>(
          &params.v[bi * params.timestep * params.num_head * Dh +
                    ti * params.num_head * Dh + vi + hi * Dh]);

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

  __syncthreads();

  // now we do the reduction in the seq dimension to get [1, head_dim].
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

  // write the [1, head_dim] result back to global memory.
  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    V_vec tmp_out;
    convert_from_float(tmp_out, out);
    store_func.template store<V_vec>(
        tmp_out, bi * (params.num_head) * Dh + vi + hi * Dh);
#else

    store_func.template store<V_vec>(out, vi + hi * Dh);
#endif
  }

#else
  assert(false);
#endif
}

template <typename T>
inline size_t smem_size_in_bytes(const QkvUnpackMhaParams<T> &params,
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
  constexpr auto kernel_fn = qkv_attention_kernel<T,                      \
                                                  Dh,                     \
                                                  Dh_MAX,                 \
                                                  THDS_PER_KEY,           \
                                                  THDS_PER_VALUE,         \
                                                  THDS_PER_BLOCK,         \
                                                  decltype(load_func),    \
                                                  decltype(store_func)>;  \
  if (smem_sz > 0xc000) {                                                 \
    cudaFuncSetAttribute(                                                 \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz); \
  }                                                                       \
  dim3 grid(params.num_head, params.batch_size);                          \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                   \
      params, load_func, store_func)

template <typename T, int Dh, int Dh_MAX, typename LoadFunc, typename StoreFunc>
void q_kv_fmha_launch_kernel(const QkvUnpackMhaParams<T> &params,
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

template <typename T, typename LoadFunc, typename StoreFunc>
void fmha_impl_qkv(const phi::GPUContext &dev_ctx,
                   const QkvUnpackMhaParams<T> &params,
                   int dim_head,
                   LoadFunc load_func,
                   StoreFunc store_func) {
  switch (dim_head) {
    case 16:
      q_kv_fmha_launch_kernel<T, 16, 32>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 32:
      q_kv_fmha_launch_kernel<T, 32, 32>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 64:
      q_kv_fmha_launch_kernel<T, 64, 64>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 80:
      q_kv_fmha_launch_kernel<T, 80, 128>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 96:
      q_kv_fmha_launch_kernel<T, 96, 128>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 128:
      q_kv_fmha_launch_kernel<T, 128, 128>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 192:
      q_kv_fmha_launch_kernel<T, 192, 256>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Dim_head = %d is unsupported!", dim_head));
  }
}

template <typename T>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &q,
                  const QkvUnpackMhaParams<T> &params,
                  int dim_head,
                  phi::DenseTensor *out_tensor) {
  MMHALoad<T> load_func(q.data<T>());
  MMHAStore<T> store_func(out_tensor->data<T>());
  fmha_impl_qkv(dev_ctx, params, dim_head, load_func, store_func);
}

template <typename T, typename Context>
void QKVDispatchWithDtype(const Context &dev_ctx,
                          const DenseTensor &q,
                          const DenseTensor &k,
                          const DenseTensor &v,
                          const paddle::optional<DenseTensor> &src_mask,
                          DenseTensor *out) {
  const auto &q_dims = q.dims();
  int bsz = q_dims[0];
  int cache_bsz = q.dims()[0];
  int max_seq_len = v.dims()[1];
  int dim_head = v.dims()[3];
  int timestep = max_seq_len;
  float inv_sqrt_dh = 1. / sqrt(dim_head);

  int k_num_head = k.dims()[2];
  int v_num_head = k_num_head;
  // this num_head means query's head
  int num_head = q.dims()[2];

  QkvUnpackMhaParams<T> params;

  dev_ctx.template Alloc<T>(out);

  params.q = q.data<T>();
  params.k = k.data<T>();
  params.v = v.data<T>();

  params.batch_size = bsz;
  params.cache_batch_size = cache_bsz;
  params.num_head = num_head;
  params.kv_num_head = k_num_head;
  params.timestep = timestep;
  params.inv_sqrt_dh = inv_sqrt_dh;

  DispatchFMHA<T>(dev_ctx, q, params, dim_head, out);
}

#endif  // PADDLE_WITH_HIP

template <typename T, typename Context>
void QKVMMHAKernel(const Context &dev_ctx,
                   const DenseTensor &q,
                   const DenseTensor &k,
                   const DenseTensor &v,
                   const paddle::optional<DenseTensor> &src_mask,
                   DenseTensor *out) {
#ifndef PADDLE_WITH_HIP
  QKVDispatchWithDtype<T, Context>(dev_ctx, q, k, v, src_mask, out);
#endif  // PADDLE_WITH_HIP
}

}  // namespace fusion
}  // namespace phi

#if CUDA_VERSION >= 11000
PD_REGISTER_KERNEL(qkv_unpack_mha,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::QKVMMHAKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(qkv_unpack_mha,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::QKVMMHAKernel,
                   float,
                   phi::dtype::float16) {}
#endif
