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

#include "paddle/phi/kernels/funcs/emb_eltwise_layer_norm_functor.h"

#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {
namespace funcs {

template <typename T>
__device__ inline T rsqrt(const T& x);

template <>
__device__ inline float rsqrt(const float& x) {
  return rsqrtf(x);
}

template <typename T>
__device__ __forceinline__ T local_rsqrt(T num) {
  return rsqrt(static_cast<float>(num));
}
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
__device__ __forceinline__ half local_rsqrt(half num) { return hrsqrt(num); }
#endif

template <typename T, int TPB>
__device__ inline void LayerNorm(const phi::funcs::kvp<T>& thread_data,
                                 const int ld,
                                 const int offset,
                                 const T* bias,
                                 const T* scale,
                                 T* output,
                                 T eps) {
  using BlockReduce = cub::BlockReduce<phi::funcs::kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = local_rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(scale[i]);
    const T b(bias[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, unsigned TPB>
__global__ void EmbEltwiseLayernormKernel(int hidden,
                                          const int64_t* ids,
                                          const T* scale,
                                          const T* bias,
                                          const int64_t* embs,
                                          T* output,
                                          T eps,
                                          int input_num) {
  cub::Sum pair_sum;
  // blockIdx.x: position in the sequence
  // blockIdx.y: batch
  // gridDim.x: Seq
  // gridDim.y: Batch

  extern __shared__ int64_t array_id[];

  const T rhidden = T(1.f) / T(hidden);
  const int64_t seq_pos = blockIdx.y + blockIdx.x * gridDim.y;
  if (threadIdx.x == 0) {
    for (int i = 0; i < input_num; ++i) {
      const int64_t* ids_p = reinterpret_cast<const int64_t*>(ids[i]);
      array_id[i] = ids_p[seq_pos];
    }
  }
  __syncthreads();

  const int64_t out_offset = seq_pos * hidden;

  phi::funcs::kvp<T> thread_data(0, 0);

#pragma unroll
  for (int it = threadIdx.x; it < hidden; it += TPB) {
    T val = 0;
    for (int i = 0; i < input_num; ++i) {
      val += reinterpret_cast<const T*>(embs[i])[array_id[i] * hidden + it];
    }

    output[out_offset + it] = val;
    const T rhiddenval = rhidden * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<T>(rhiddenval, rhiddenval * val));
  }
  LayerNorm<T, TPB>(thread_data, hidden, out_offset, bias, scale, output, eps);
}

// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#ifndef __HIPCC__  // @{ Half kernel: EmbEltwiseLayernormKernel
template <>
__global__ void EmbEltwiseLayernormKernel<half, 256>(int hidden,
                                                     const int64_t* ids,
                                                     const half* scale,
                                                     const half* bias,
                                                     const int64_t* embs,
                                                     half* output,
                                                     half eps,
                                                     int input_num) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  cub::Sum pair_sum;
  // blockIdx.x: position in the sequence
  // blockIdx.y: batch
  // gridDim.x: Seq
  // gridDim.y: Batch

  extern __shared__ int64_t array_id[];

  const half rhidden = half(1.f) / half(hidden);
  const int64_t seq_pos = blockIdx.y + blockIdx.x * gridDim.y;
  if (threadIdx.x == 0) {
    for (int i = 0; i < input_num; ++i) {
      const int64_t* ids_p = reinterpret_cast<const int64_t*>(ids[i]);
      array_id[i] = ids_p[seq_pos];
    }
  }
  __syncthreads();

  const int64_t out_offset = seq_pos * hidden;

  phi::funcs::kvp<half> thread_data(0, 0);

#pragma unroll
  for (int it = threadIdx.x; it < hidden; it += 256) {
    half val = 0;
    for (int i = 0; i < input_num; ++i) {
      val += reinterpret_cast<const half*>(embs[i])[array_id[i] * hidden + it];
    }

    output[out_offset + it] = val;
    const half rhiddenval = rhidden * val;
    thread_data = pair_sum(thread_data,
                           phi::funcs::kvp<half>(rhiddenval, rhiddenval * val));
  }
  LayerNorm<half, 256>(
      thread_data, hidden, out_offset, bias, scale, output, eps);
#endif
}
#endif  // @} End Half kernel: EmbEltwiseLayernormKernel

template <typename T>
void EmbEltwiseLayerNormFunctor<T>::operator()(int batch,
                                               int seq_len,
                                               int hidden,
                                               const int64_t* ids,
                                               const T* scale,
                                               const T* bias,
                                               const int64_t* embs,
                                               T* output,
                                               float eps,
                                               int input_num,
                                               gpuStream_t stream) {
  const unsigned tpb = 256;
  const dim3 grid(seq_len, batch, 1);
  const dim3 block(tpb, 1, 1);
  int shared_bytes = input_num * sizeof(int64_t);
  EmbEltwiseLayernormKernel<T, tpb><<<grid, block, shared_bytes, stream>>>(
      hidden, ids, scale, bias, embs, output, eps, input_num);
}

template class EmbEltwiseLayerNormFunctor<float>;

// device function 'operator()' is not supported until cuda 10.0
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
template class EmbEltwiseLayerNormFunctor<half>;
#endif

}  // namespace funcs
}  // namespace phi
