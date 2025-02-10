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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>  // NOLINT
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>

#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/kernels/funcs/multihead_matmul_functor.h"

#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {
namespace funcs {

template <typename T>
struct CUDATypeTraits;

template <>
struct CUDATypeTraits<half> {
  typedef phi::dtype::float16 TYPE;
};

template <>
struct CUDATypeTraits<float> {
  typedef float TYPE;
};

using phi::funcs::operator+;

template <typename T>
__global__ void SoftmaxKernelWithEltadd(T *qk_buf_,
                                        const T *bias_qk_,
                                        const int batch_size,
                                        const int head_num,
                                        const int seq_len,
                                        const phi::funcs::warp_mask_t mask) {
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % WARP_SIZE == 0);

  float tmp = threadIdx.x < seq_len
                  ? static_cast<float>(qk_buf_[threadIdx.x + qk_offset] +
                                       bias_qk_[threadIdx.x + qk_offset])
                  : -1e20f;
  float max_val = phi::funcs::BlockReduceMax<float>(tmp, mask);

  float qk_tmp = threadIdx.x < seq_len ? __expf(tmp - max_val) : 0.0f;
  float sum_val = phi::funcs::BlockReduceSum<float>(qk_tmp, mask);

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / sum_val);
}

template <>
__global__ void SoftmaxKernelWithEltadd<half>(
    half *qk_buf_,
    const half *bias_qk_,
    const int batch_size,
    const int head_num,
    const int seq_len,
    const phi::funcs::warp_mask_t mask) {
#if defined(PADDLE_WITH_CUDA) && CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % WARP_SIZE == 0);

  float tmp = threadIdx.x < seq_len
                  ? static_cast<float>(qk_buf_[threadIdx.x + qk_offset] +
                                       bias_qk_[threadIdx.x + qk_offset])
                  : -1e20f;
  float max_val = phi::funcs::BlockReduceMax<float>(tmp, mask);

  float qk_tmp = threadIdx.x < seq_len ? __expf(tmp - max_val) : 0.0f;
  float sum_val = phi::funcs::BlockReduceSum<float>(qk_tmp, mask);

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (half)(qk_tmp / sum_val);
#endif
}

template <typename T>
__global__ void SoftmaxKernelWithEltadd2(T *qk_buf_,
                                         const T *bias_qk_,
                                         const int batch_size,
                                         const int head_num,
                                         const int seq_len,
                                         const phi::funcs::warp_mask_t mask) {
  int qk_offset = blockIdx.x * seq_len;
  int idx = threadIdx.x;
  assert(blockDim.x % WARP_SIZE == 0);

  float2 tmp = idx < seq_len
                   ? phi::funcs::ToFloat2<T>(qk_buf_[idx + qk_offset] +
                                             bias_qk_[idx + qk_offset])
                   : make_float2(-1e20f, -1e20f);
  float max_val = phi::funcs::BlockReduceMax<float>(max(tmp.x, tmp.y), mask);
  float2 qk_tmp = idx < seq_len ? make_float2(__expf(tmp.x - max_val),
                                              __expf(tmp.y - max_val))
                                : make_float2(0.f, 0.f);
  float sum_val =
      phi::funcs::BlockReduceSum<float>(qk_tmp.x + qk_tmp.y, mask) + 1e-6f;

  if (idx < seq_len) {
    qk_buf_[idx + qk_offset] =
        phi::funcs::FloatsToPair<T>(qk_tmp.x / sum_val, qk_tmp.y / sum_val);
  }
}

template <>
__global__ void SoftmaxKernelWithEltadd2<half2>(
    half2 *qk_buf_,
    const half2 *bias_qk_,
    const int batch_size,
    const int head_num,
    const int seq_len,
    const phi::funcs::warp_mask_t mask) {
// operator "+" of half only suppotted after cuda version 10.0
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#if defined(PADDLE_WITH_CUDA) && CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int qk_offset = blockIdx.x * seq_len;
  int idx = threadIdx.x;
  assert(blockDim.x % WARP_SIZE == 0);

  float2 tmp = idx < seq_len
                   ? phi::funcs::ToFloat2<half2>(qk_buf_[idx + qk_offset] +
                                                 bias_qk_[idx + qk_offset])
                   : make_float2(-1e20f, -1e20f);
  float max_val = phi::funcs::BlockReduceMax<float>(max(tmp.x, tmp.y), mask);
  float2 qk_tmp = idx < seq_len ? make_float2(__expf(tmp.x - max_val),
                                              __expf(tmp.y - max_val))
                                : make_float2(0.f, 0.f);
  float sum_val =
      phi::funcs::BlockReduceSum<float>(qk_tmp.x + qk_tmp.y, mask) + 1e-6f;

  if (idx < seq_len) {
    qk_buf_[idx + qk_offset] =
        phi::funcs::FloatsToPair<half2>(qk_tmp.x / sum_val, qk_tmp.y / sum_val);
  }
#endif
}

template <typename T>
__global__ void SoftmaxKernelWithEltaddForLarge(
    T *qk_buf,
    const T *bias_qk,
    const int batch_size,
    const int head_num,
    const int seq_len,
    const phi::funcs::warp_mask_t mask) {
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % WARP_SIZE == 0);

  T stride_max = -1e20f;
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    stride_max = qk_buf[threadIdx.x + i + qk_offset] +
                             bias_qk[threadIdx.x + i + qk_offset] >
                         stride_max
                     ? qk_buf[threadIdx.x + i + qk_offset] +
                           bias_qk[threadIdx.x + i + qk_offset]
                     : stride_max;
  }
  T max_val = phi::funcs::BlockReduceMax<T>(stride_max, mask);

  T stride_sum = 0.f;
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    stride_sum += __expf(qk_buf[threadIdx.x + i + qk_offset] +
                         bias_qk[threadIdx.x + i + qk_offset] - max_val);
  }
  T sum_val = phi::funcs::BlockReduceSum<T>(stride_sum, mask);

  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    qk_buf[threadIdx.x + i + qk_offset] =
        (T)(__expf(qk_buf[threadIdx.x + i + qk_offset] +
                   bias_qk[threadIdx.x + i + qk_offset] - max_val) /
            sum_val);
  }
}

template <>
__global__ void SoftmaxKernelWithEltaddForLarge(
    half *qk_buf,
    const half *bias_qk,
    const int batch_size,
    const int head_num,
    const int seq_len,
    const phi::funcs::warp_mask_t mask) {
#if defined(PADDLE_WITH_CUDA) && \
    (CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__) && CUDA_VERSION >= 10000)
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % WARP_SIZE == 0);

  float stride_max = -1e20f;
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float tmp = static_cast<float>(qk_buf[threadIdx.x + i + qk_offset] +
                                   bias_qk[threadIdx.x + i + qk_offset]);
    stride_max = tmp > stride_max ? tmp : stride_max;
  }
  float max_val = phi::funcs::BlockReduceMax<float>(stride_max, mask);

  float stride_sum = 0.f;
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float tmp = static_cast<float>(qk_buf[threadIdx.x + i + qk_offset] +
                                   bias_qk[threadIdx.x + i + qk_offset]);
    stride_sum += __expf(tmp - max_val);
  }
  float sum_val = phi::funcs::BlockReduceSum<float>(stride_sum, mask);

  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float tmp =
        __expf(static_cast<float>(qk_buf[threadIdx.x + i + qk_offset] +
                                  bias_qk[threadIdx.x + i + qk_offset]) -
               max_val);
    qk_buf[threadIdx.x + i + qk_offset] = (half)(tmp / sum_val);
  }
#endif
}

template <typename T>
__global__ void SoftmaxKernelWithEltaddForLarge2(
    T *qk_buf_,
    const T *bias_qk_,
    const int batch_size,
    const int head_num,
    const int seq_len,
    const phi::funcs::warp_mask_t mask) {
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % WARP_SIZE == 0);

  float2 stride_max = make_float2(-1e20f, -1e20f);
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float2 cur = phi::funcs::ToFloat2<T>(qk_buf_[threadIdx.x + i + qk_offset] +
                                         bias_qk_[threadIdx.x + i + qk_offset]);
    stride_max.x = max(stride_max.x, cur.x);
    stride_max.y = max(stride_max.y, cur.y);
  }
  float max_val =
      phi::funcs::BlockReduceMax<float>(max(stride_max.x, stride_max.y), mask);

  float2 stride_sum = make_float2(0.f, 0.f);
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float2 cur = phi::funcs::ToFloat2<T>(qk_buf_[threadIdx.x + i + qk_offset] +
                                         bias_qk_[threadIdx.x + i + qk_offset]);
    stride_sum.x += __expf(cur.x - max_val);
    stride_sum.y += __expf(cur.y - max_val);
  }

  float sum_val =
      phi::funcs::BlockReduceSum<float>(stride_sum.x + stride_sum.y, mask) +
      1e-6f;

  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float2 cur = phi::funcs::ToFloat2<T>(qk_buf_[threadIdx.x + i + qk_offset] +
                                         bias_qk_[threadIdx.x + i + qk_offset]);
    qk_buf_[threadIdx.x + i + qk_offset] = phi::funcs::FloatsToPair<T>(
        __expf(cur.x - max_val) / sum_val, __expf(cur.y - max_val) / sum_val);
  }
}

template <>
__global__ void SoftmaxKernelWithEltaddForLarge2(
    half2 *qk_buf_,
    const half2 *bias_qk_,
    const int batch_size,
    const int head_num,
    const int seq_len,
    const phi::funcs::warp_mask_t mask) {
// operator "+" of half only suppotted after cuda version 10.0
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#if defined(PADDLE_WITH_CUDA) && \
    (CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__) && CUDA_VERSION >= 10000)

  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % WARP_SIZE == 0);

  float2 stride_max = make_float2(-1e20f, -1e20f);
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float2 cur =
        phi::funcs::ToFloat2<half2>(qk_buf_[threadIdx.x + i + qk_offset] +
                                    bias_qk_[threadIdx.x + i + qk_offset]);
    stride_max.x = max(stride_max.x, cur.x);
    stride_max.y = max(stride_max.y, cur.y);
  }
  float max_val =
      phi::funcs::BlockReduceMax<float>(max(stride_max.x, stride_max.y), mask);

  float2 stride_sum = make_float2(0.f, 0.f);
  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float2 cur =
        phi::funcs::ToFloat2<half2>(qk_buf_[threadIdx.x + i + qk_offset] +
                                    bias_qk_[threadIdx.x + i + qk_offset]);
    stride_sum.x += __expf(cur.x - max_val);
    stride_sum.y += __expf(cur.y - max_val);
  }

  float sum_val =
      phi::funcs::BlockReduceSum<float>(stride_sum.x + stride_sum.y, mask) +
      1e-6f;

  for (int i = 0; threadIdx.x + i < seq_len; i += blockDim.x) {
    float2 cur =
        phi::funcs::ToFloat2<half2>(qk_buf_[threadIdx.x + i + qk_offset] +
                                    bias_qk_[threadIdx.x + i + qk_offset]);
    qk_buf_[threadIdx.x + i + qk_offset] = phi::funcs::FloatsToPair<half2>(
        __expf(cur.x - max_val) / sum_val, __expf(cur.y - max_val) / sum_val);
  }
#endif
}

template <typename T>
inline __device__ T ldg(const T *val) {
  return __ldg(val);
}

template <typename T>
inline __device__ T hexp2(T a) {
  return h2exp(a);
}

template <typename T_IN, typename T_OUT>
inline __device__ T_OUT type2type2(T_IN a);

template <>
inline __device__ half2 type2type2(half a) {
  return __half2half2(a);
}

template <typename T>
inline __device__ T float2type2(float a);

template <>
inline __device__ half2 float2type2(float a) {
  return __float2half2_rn(a);
}

template <typename T>
inline __device__ T hmul2(T a, T b) {
  return __hmul2(a, b);
}

template <typename T>
inline __device__ T hsub2(T a, T b) {
  return __hsub2(a, b);
}

template <typename T>
inline __device__ T hadd2(T a, T b) {
  return __hadd2(a, b);
}

template <typename T, int ITEMS_PER_THREAD, int NUM>
__global__ void softmax_kernel_with_mask(T *qk_buf_,
                                         const T *attr_mask,
                                         const int batch_size,
                                         const int head_num,
                                         const int seq_len) {
  using T2 = half2;
  T2 *qk_buf_half2 = reinterpret_cast<T2 *>(qk_buf_);
  const T2 *attr_mask_half2 = (const T2 *)attr_mask;

  for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x * NUM) {
    T2 data[NUM][ITEMS_PER_THREAD];

    int qk_offset[NUM];

    __shared__ float s_sum[NUM], s_max[NUM];
    float local_max[NUM];
#pragma unroll
    for (int j = 0; j < NUM; j++) {
      local_max[j] = -1e20f;
    }

    for (int i = 0;
         blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
         i++) {
      int mask_offset[NUM];
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len +
                        seq_id + j * gridDim.x) *
                           (seq_len / 2) +
                       blockDim.x * i + threadIdx.x;
        mask_offset[j] =
            (blockIdx.y * seq_len + seq_id + j * gridDim.x) * (seq_len / 2) +
            blockDim.x * i + threadIdx.x;
      }

      T2 mask_val[NUM];
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        mask_val[j] = ldg(&attr_mask_half2[mask_offset[j]]);
      }

      T2 qk[NUM];
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        qk[j] = qk_buf_half2[qk_offset[j]];
      }

#pragma unroll
      for (int j = 0; j < NUM; j++) {
        mask_val[j] = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val[j]),
                                float2type2<T2>(-10000.0f));
      }

#pragma unroll
      for (int j = 0; j < NUM; j++) {
        data[j][i] = hadd2<T2>(qk[j], mask_val[j]);
        local_max[j] = fmax(local_max[j],
                            fmax(static_cast<float>(data[j][i].x),
                                 static_cast<float>(data[j][i].y)));
      }
    }

    if (blockDim.x <= WARP_SIZE) {
      phi::funcs::WarpReduceMaxV2<float, NUM>(local_max);
    } else {
      phi::funcs::BlockReduceMaxV2<float, NUM>(local_max);
    }

    if (threadIdx.x == 0) {
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        s_max[j] = local_max[j];
      }
    }
    __syncthreads();

    float local_sum[NUM];
#pragma unroll
    for (int j = 0; j < NUM; j++) {
      local_sum[j] = {0.f};
    }

    for (int i = 0;
         blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
         i++) {
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        data[j][i] =
            hexp2<T2>(hsub2<T2>(data[j][i], float2type2<T2>(s_max[j])));
      }

#pragma unroll
      for (int j = 0; j < NUM; j++) {
        local_sum[j] += static_cast<float>(data[j][i].x + data[j][i].y);
      }
    }

    if (blockDim.x <= WARP_SIZE) {
      phi::funcs::WarpReduceSumV2<float, NUM>(local_sum);
    } else {
      phi::funcs::BlockReduceSumV2<float, NUM>(local_sum);
    }

    if (threadIdx.x == 0) {
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
      }
    }
    __syncthreads();

    for (int i = 0;
         blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
         i++) {
#pragma unroll
      for (int j = 0; j < NUM; j++) {
        qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len +
                        seq_id + j * gridDim.x) *
                           (seq_len / 2) +
                       blockDim.x * i + threadIdx.x;
      }

#pragma unroll
      for (int j = 0; j < NUM; j++) {
        qk_buf_half2[qk_offset[j]] =
            hmul2<T2>(data[j][i], float2type2<T2>(s_sum[j]));
      }
    }
  }
}

#define SOFTMAX_KERNEL_WITH_MASK(REPEAT_THREAD)                         \
  do {                                                                  \
    block.x /= REPEAT_THREAD;                                           \
    grid.x /= 4;                                                        \
    constexpr int NUM = 4;                                              \
    softmax_kernel_with_mask<half, REPEAT_THREAD, NUM>                  \
        <<<grid, block, 0, stream>>>(reinterpret_cast<half *>(qk_buf_), \
                                     (const half *)bias_qk,             \
                                     batch_size,                        \
                                     head_num,                          \
                                     seq_len);                          \
  } while (0)

template <typename T>
inline void MatmulWithHeadQK(const phi::GPUContext &context,
                             int head_num,
                             int seq_len,
                             int size_per_head,
                             int batch_size,
                             bool q_trans,
                             bool k_trans,
                             T *q_buf_,
                             T *k_buf_,
                             T *qk_buf_,
                             const T *bias_qk,
                             bool bias_is_mask,
                             T alpha,
                             T beta) {
  CBLAS_TRANSPOSE transA = !q_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !k_trans ? CblasNoTrans : CblasTrans;

  typedef typename CUDATypeTraits<T>::TYPE run_type;
  auto blas = phi::funcs::GetBlas<phi::GPUContext, run_type>(context);
  auto stream = context.stream();

  blas.BatchedGEMM(transA,
                   transB,
                   seq_len,
                   seq_len,
                   size_per_head,
                   static_cast<run_type>(alpha),
                   reinterpret_cast<run_type *>(q_buf_),
                   reinterpret_cast<run_type *>(k_buf_),
                   static_cast<run_type>(beta),
                   reinterpret_cast<run_type *>(qk_buf_),
                   batch_size * head_num,
                   seq_len * size_per_head,
                   seq_len * size_per_head);

  if (seq_len <= 1024) {
    int grid = batch_size * head_num * seq_len;
    int block = seq_len;

    // Align block to 32, also limit seq_len to max block size.
    if (seq_len % 2 == 0) {
      block =
          (seq_len <= (2 * WARP_SIZE))
              ? WARP_SIZE
              : ((seq_len + (2 * WARP_SIZE - 1)) / (2 * WARP_SIZE)) * WARP_SIZE;
      if (std::is_same<T, float>::value) {
        SoftmaxKernelWithEltadd2<float2><<<grid, block, 0, stream>>>(
            reinterpret_cast<float2 *>(qk_buf_),
            reinterpret_cast<const float2 *>(bias_qk),
            batch_size,
            head_num,
            seq_len / 2,
            FINAL_MASK);
      } else {
        if (bias_is_mask) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700)
          PADDLE_ENFORCE_EQ(bias_is_mask,
                            false,
                            common::errors::InvalidArgument(
                                "QK_bias is mask can't be supported on rocm or "
                                "cuda_arch<700"));
#else
          dim3 grid(seq_len, batch_size, head_num);
          dim3 block((seq_len / 2 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
          SOFTMAX_KERNEL_WITH_MASK(1);
#endif
        } else {
          SoftmaxKernelWithEltadd2<__half2><<<grid, block, 0, stream>>>(
              reinterpret_cast<__half2 *>(qk_buf_),
              reinterpret_cast<const __half2 *>(bias_qk),
              batch_size,
              head_num,
              seq_len / 2,
              FINAL_MASK);
        }
      }
    } else {
      block = (seq_len <= WARP_SIZE)
                  ? WARP_SIZE
                  : ((seq_len + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
      SoftmaxKernelWithEltadd<T><<<grid, block, 0, stream>>>(
          qk_buf_, bias_qk, batch_size, head_num, seq_len, FINAL_MASK);
    }
  } else {
    int grid = batch_size * head_num * seq_len;
    int block = 512;
    if (seq_len % 2 == 0) {
      if (std::is_same<T, float>::value) {
        SoftmaxKernelWithEltaddForLarge2<float2><<<grid, block, 0, stream>>>(
            reinterpret_cast<float2 *>(qk_buf_),
            reinterpret_cast<const float2 *>(bias_qk),
            batch_size,
            head_num,
            seq_len / 2,
            FINAL_MASK);
      } else {
        if (bias_is_mask) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700)
          PADDLE_ENFORCE_EQ(bias_is_mask,
                            false,
                            common::errors::InvalidArgument(
                                "QK_bias is mask can't be supported on rocm or "
                                "cuda_arch<700"));
#else
          dim3 grid(seq_len, batch_size, head_num);
          dim3 block((seq_len / 2 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
          if (block.x > 0 && block.x <= 1024) {
            SOFTMAX_KERNEL_WITH_MASK(1);
          } else if (block.x <= 2048) {
            SOFTMAX_KERNEL_WITH_MASK(2);
          } else if (block.x <= 4096) {
            SOFTMAX_KERNEL_WITH_MASK(4);
          } else {
            PADDLE_THROW(common::errors::InvalidArgument(
                "Cannot support the length of attention > 8192."));
          }
#endif
        } else {
          SoftmaxKernelWithEltaddForLarge2<__half2><<<grid, block, 0, stream>>>(
              reinterpret_cast<__half2 *>(qk_buf_),
              reinterpret_cast<const __half2 *>(bias_qk),
              batch_size,
              head_num,
              seq_len / 2,
              FINAL_MASK);
        }
      }
    } else {
      SoftmaxKernelWithEltaddForLarge<T><<<grid, block, 0, stream>>>(
          qk_buf_, bias_qk, batch_size, head_num, seq_len, FINAL_MASK);
    }
  }
}

template <typename T>
inline void MatmulWithHeadQKV(const phi::GPUContext &context,
                              int head_num,
                              int seq_len,
                              int size_per_head,
                              int batch_size,
                              bool qk_trans,
                              bool v_trans,
                              T *v_buf_,
                              const T *qk_buf_,
                              T *dst,
                              T alpha,
                              T beta) {
  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  typedef typename CUDATypeTraits<T>::TYPE run_type;
  auto blas = phi::funcs::GetBlas<phi::GPUContext, run_type>(context);
  auto stream = context.stream();
  CBLAS_TRANSPOSE transA = !qk_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !v_trans ? CblasNoTrans : CblasTrans;

  blas.BatchedGEMM(transA,
                   transB,
                   seq_len,
                   size_per_head,
                   seq_len,
                   static_cast<run_type>(alpha),
                   reinterpret_cast<const run_type *>(qk_buf_),
                   reinterpret_cast<run_type *>(v_buf_),
                   static_cast<run_type>(beta),
                   reinterpret_cast<run_type *>(dst),
                   batch_size * head_num,
                   seq_len * seq_len,
                   seq_len * size_per_head);
}

template <typename T>
void MultiheadGPUComputeFunctor<T>::operator()(const phi::GPUContext &dev_ctx,
                                               int batch,
                                               int seq_len,
                                               int head_num,
                                               int head_size,
                                               T *qkptr,
                                               const T *bias_qk_ptr,
                                               bool bias_is_mask,
                                               T *tptr,
                                               T alpha,
                                               T beta) {
  auto stream = dev_ctx.stream();
  const int tsize = batch * head_num * seq_len * head_size;

  T *qptr = tptr;
  T *kptr = qptr + tsize;
  T *vptr = kptr + tsize;
  // batch gemm stride, softmaxwithscale.
  MatmulWithHeadQK<T>(dev_ctx,
                      head_num,
                      seq_len,
                      head_size,
                      batch,
                      false,
                      true,
                      qptr,
                      kptr,
                      qkptr,
                      bias_qk_ptr,
                      bias_is_mask,
                      alpha,
                      beta);
  // batch gemm stride, transpose.
  MatmulWithHeadQKV<T>(dev_ctx,
                       head_num,
                       seq_len,
                       head_size,
                       batch,
                       false,
                       false,
                       vptr,
                       qkptr,
                       tptr,
                       T(1.0),
                       beta);
}

template class MultiheadGPUComputeFunctor<float>;

// device function 'operator()' is not supported until cuda 10.0
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
template class MultiheadGPUComputeFunctor<half>;
#endif

}  // namespace funcs
}  // namespace phi
