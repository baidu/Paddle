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

#include "paddle/phi/kernels/funcs/math/bert_encoder_functor.h"
#include <algorithm>
#include <type_traits>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {
namespace math {

// NOTE(chenfeiyu): explicitly use operator+ for float2
// since float2 is not in namespace phi::funcs, ADL won't help
using phi::funcs::operator+;

template <typename T>
__device__ __forceinline__ T local_rsqrt(T num) {
  return rsqrt(static_cast<float>(num));
}
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
__device__ __forceinline__ half local_rsqrt(half num) { return hrsqrt(num); }
#endif

template <typename T, int TPB>
__device__ inline void LayerNormSmall(T val,
                                      const phi::funcs::kvp<T> &thread_data,
                                      const int ld,
                                      const int idx,
                                      const T *bias,
                                      const T *scale,
                                      T *output,
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

  if (threadIdx.x < ld) {
    const T g(scale[threadIdx.x]);
    const T b(bias[threadIdx.x]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, int TPB>
__device__ inline void LayerNorm(const phi::funcs::kvp<T> &thread_data,
                                 const int ld,
                                 const int offset,
                                 const T *bias,
                                 const T *scale,
                                 T *output,
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

template <typename T, typename T2, int TPB>
__device__ inline void LayerNorm2(const phi::funcs::kvp<T> &thread_data,
                                  const int ld,
                                  const int offset,
                                  const T2 *bias,
                                  const T2 *scale,
                                  T2 *output,
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
    T2 val = output[idx];
    const T2 g = scale[i];
    const T2 b = bias[i];
    val.x = T(g.x) * (val.x - mu) * rsigma + T(b.x);
    val.y = T(g.y) * (val.y - mu) * rsigma + T(b.y);
    output[idx] = val;
  }
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormSmallKernel(int num,
                                         int hidden,
                                         const T *input1,
                                         const T *input2,
                                         T *output,
                                         const T *scale,
                                         const T *bias,
                                         T eps) {
  const T rld = T(1) / T(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;
  if (threadIdx.x < hidden) {
    val = input1[idx] + input2[idx];
    const T rldval = rld * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<T>(rldval, rldval * val));
  }
  LayerNormSmall<T, TPB>(
      val, thread_data, hidden, idx, bias, scale, output, eps);
}

// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#ifndef __HIPCC__  // @{ Half kernel: SkipLayerNormSmallKernel
template <>
__global__ void SkipLayerNormSmallKernel<half, 32>(int num,
                                                   int hidden,
                                                   const half *input1,
                                                   const half *input2,
                                                   half *output,
                                                   const half *scale,
                                                   const half *bias,
                                                   half eps) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const half rld = half(1) / half(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<half> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  half val = 0;
  if (threadIdx.x < hidden) {
    val = input1[idx] + input2[idx];
    const half rldval = rld * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<half>(rldval, rldval * val));
  }
  LayerNormSmall<half, 32>(
      val, thread_data, hidden, idx, bias, scale, output, eps);
#endif
}

template <>
__global__ void SkipLayerNormSmallKernel<half, 128>(int num,
                                                    int hidden,
                                                    const half *input1,
                                                    const half *input2,
                                                    half *output,
                                                    const half *scale,
                                                    const half *bias,
                                                    half eps) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const half rld = half(1) / half(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<half> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  half val = 0;
  if (threadIdx.x < hidden) {
    val = input1[idx] + input2[idx];
    const half rldval = rld * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<half>(rldval, rldval * val));
  }
  LayerNormSmall<half, 128>(
      val, thread_data, hidden, idx, bias, scale, output, eps);
#endif
}

template <>
__global__ void SkipLayerNormSmallKernel<half, 384>(int num,
                                                    int hidden,
                                                    const half *input1,
                                                    const half *input2,
                                                    half *output,
                                                    const half *scale,
                                                    const half *bias,
                                                    half eps) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const half rld = half(1) / half(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<half> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  half val = 0;
  if (threadIdx.x < hidden) {
    val = input1[idx] + input2[idx];
    const half rldval = rld * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<half>(rldval, rldval * val));
  }
  LayerNormSmall<half, 384>(
      val, thread_data, hidden, idx, bias, scale, output, eps);
#endif
}
#endif  // @} End Half kernel: SkipLayerNormSmallKernel

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(int num,
                                    int hidden,
                                    const T *input1,
                                    const T *input2,
                                    T *output,
                                    const T *scale,
                                    const T *bias,
                                    T eps) {
  const T rld = T(1) / T(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const int idx = offset + it;
    const T val = input1[idx] + input2[idx];
    const T rldval = rld * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<T>(rldval, rldval * val));
    output[idx] = val;
  }
  LayerNorm<T, TPB>(thread_data, hidden, offset, bias, scale, output, eps);
}

// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#ifndef __HIPCC__  // @{ Half kernel: SkipLayerNormKernel
template <>
__global__ void SkipLayerNormKernel<half, 256>(int num,
                                               int hidden,
                                               const half *input1,
                                               const half *input2,
                                               half *output,
                                               const half *scale,
                                               const half *bias,
                                               half eps) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const half rld = half(1) / half(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<half> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += 256) {
    const int idx = offset + it;
    const half val = input1[idx] + input2[idx];
    const half rldval = rld * val;
    thread_data =
        pair_sum(thread_data, phi::funcs::kvp<half>(rldval, rldval * val));
    output[idx] = val;
  }
  LayerNorm<half, 256>(thread_data, hidden, offset, bias, scale, output, eps);
#endif
}
#endif  // @} End Half kernel: SkipLayerNormKernel

template <typename T, typename T2, unsigned TPB>
__global__ void SkipLayerNormKernel2(int num,
                                     int hidden,
                                     const T2 *input1,
                                     const T2 *input2,
                                     T2 *output,
                                     const T2 *scale,
                                     const T2 *bias,
                                     float eps) {
  const T rld = T(0.5f / hidden);  // because hidden is hidden/2
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const int idx = offset + it;
    const T2 val2 = input1[idx] + input2[idx];
    thread_data = pair_sum(
        thread_data,
        phi::funcs::kvp<T>(rld * (val2.x + val2.y),
                           rld * val2.x * val2.x + rld * val2.y * val2.y));
    output[idx] = val2;
  }
  LayerNorm2<T, T2, TPB>(thread_data, hidden, offset, bias, scale, output, eps);
}

// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#ifndef __HIPCC__  // @{ Half kernel: SkipLayerNormKernel2
template <>
__global__ void SkipLayerNormKernel2<half, half2, 256>(int num,
                                                       int hidden,
                                                       const half2 *input1,
                                                       const half2 *input2,
                                                       half2 *output,
                                                       const half2 *scale,
                                                       const half2 *bias,
                                                       float eps) {
// operator "+" of half only suppotted after cuda version 10.0
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__) && CUDA_VERSION >= 10000
  const half rld = half(0.5f / hidden);  // because hidden is hidden/2
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  phi::funcs::kvp<half> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += 256) {
    const int idx = offset + it;
    const half2 val2 = input1[idx] + input2[idx];
    thread_data = pair_sum(
        thread_data,
        phi::funcs::kvp<half>(rld * (val2.x + val2.y),
                              rld * val2.x * val2.x + rld * val2.y * val2.y));
    output[idx] = val2;
  }
  LayerNorm2<half, half2, 256>(
      thread_data, hidden, offset, bias, scale, output, eps);
#endif
}
#endif  // @} End Half kernel: SkipLayerNormKernel2

template <typename T>
void SkipLayerNormFunctor<T>::operator()(const int num,
                                         const int hidden,
                                         const T *input1,
                                         const T *input2,
                                         const T *scale,
                                         const T *bias,
                                         T *output,
                                         float eps,
                                         gpuStream_t stream) {
  int block = num / hidden;
  if (hidden <= WARP_SIZE) {
    const int threads = WARP_SIZE;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden <= 128) {
    const int threads = 128;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden == 384) {
    const int threads = 384;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else {
    const int threads = 256;
    if (hidden % 2 == 0) {
      if (std::is_same<T, float>::value) {
        SkipLayerNormKernel2<float, float2, threads>
            <<<block, threads, 0, stream>>>(
                num,
                hidden / 2,
                reinterpret_cast<const float2 *>(input1),
                reinterpret_cast<const float2 *>(input2),
                reinterpret_cast<float2 *>(output),
                reinterpret_cast<const float2 *>(scale),
                reinterpret_cast<const float2 *>(bias),
                eps);
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#ifndef __HIPCC__
      } else if (std::is_same<T, __half>::value) {
        SkipLayerNormKernel2<__half, __half2, threads>
            <<<block, threads, 0, stream>>>(
                num,
                hidden / 2,
                reinterpret_cast<const __half2 *>(input1),
                reinterpret_cast<const __half2 *>(input2),
                reinterpret_cast<__half2 *>(output),
                reinterpret_cast<const __half2 *>(scale),
                reinterpret_cast<const __half2 *>(bias),
                eps);
#endif
      } else {
        assert(false);
        // should not be here
      }
    } else {
      SkipLayerNormKernel<T, threads><<<block, threads, 0, stream>>>(
          num, hidden, input1, input2, output, scale, bias, eps);
    }
  }
}

template class SkipLayerNormFunctor<float>;

// device function 'operator()' is not supported until cuda 10.0
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
template class SkipLayerNormFunctor<half>;
#endif

}  // namespace math
}  // namespace phi
