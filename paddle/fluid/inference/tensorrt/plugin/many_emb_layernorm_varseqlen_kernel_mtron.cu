// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "NvInfer.h"
#include "paddle/fluid/inference/tensorrt/plugin/common/common.cuh"
#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_varseqlen_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T, unsigned TPB>
__global__ void embLayerNormKernelMTron_2(int32_t ld,
                                          int32_t const* inputIds0,
                                          int32_t const* inputIds1,
                                          float const* beta,
                                          float const* gamma,
                                          T const* mIdsEmbDev0,
                                          T const* mIdsEmbDev1,
                                          int32_t IdsSize0,
                                          int32_t IdsSize1,
                                          T* output,
                                          T* skip) {
  cub::Sum pairSum;
  int32_t const s = blockIdx.x;
  int32_t const b = blockIdx.y;
  int32_t const sumS = inputIds0[b];
  int32_t const s_b = inputIds0[b + 1] - sumS;
  if (s >= s_b) {
    return;  // This CTA has nothing to do
  }
  T const rld = T(1.f) / T(ld);
  const int32_t seqPos = sumS + s;
  extern __shared__ int32_t word_id[];

  if (threadIdx.x == 0) {
    if (static_cast<int32_t const*>(inputIds1)[seqPos] < 0 ||
        static_cast<int32_t const*>(inputIds1)[seqPos] >= IdsSize1) {
      printf(
          "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
          "table: ID < 0 or ID > max ");
      return;
    } else {
      word_id[0] = static_cast<int32_t const*>(inputIds1)[seqPos];
    }
  }
  __syncthreads();

  // 2. load pos/tok/word embeddings and add them together
  // offset into embeddings is given by wordId * hidden_size
  const int32_t poffset = blockIdx.x * ld;
  const int32_t outOffset = seqPos * ld;
  // the output offset is given by b * (S*hidden_size) + s * hidden_size
  kvp<T> threadData(0, 0);

  for (int32_t it = threadIdx.x; it < ld; it += TPB) {
    T p(mIdsEmbDev0[poffset + it]);  // pos id
    T val = p;
    const int32_t offset = word_id[0] * ld;
    val += mIdsEmbDev1[offset + it];
    output[outOffset + it] = val;
    skip[outOffset + it] = val;

    const T rldval = rld * val;
    threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
  }

  // 3. layer norm on the sum
  layerNorm<T, T, float, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T, unsigned TPB>
__global__ void embLayerNormKernelMTron_3(int32_t ld,
                                          int32_t const* inputIds0,
                                          int32_t const* inputIds1,
                                          int32_t const* inputIds2,
                                          float const* beta,
                                          float const* gamma,
                                          T const* mIdsEmbDev0,
                                          T const* mIdsEmbDev1,
                                          T const* mIdsEmbDev2,
                                          int32_t IdsSize0,
                                          int32_t IdsSize1,
                                          int32_t IdsSize2,
                                          T* output,
                                          T* skip) {
  cub::Sum pairSum;
  const int32_t s = blockIdx.x;
  const int32_t b = blockIdx.y;
  const int32_t sumS = inputIds0[b];
  const int32_t s_b = inputIds0[b + 1] - sumS;
  if (s >= s_b) {
    return;  // This CTA has nothing to do
  }
  const T rld = T(1.f) / T(ld);
  const int32_t seqPos = sumS + s;
  extern __shared__ int32_t word_id[];

  if (threadIdx.x == 0) {
    if (static_cast<int32_t const*>(inputIds1)[seqPos] < 0 ||
        static_cast<int32_t const*>(inputIds1)[seqPos] >= IdsSize1) {
      printf(
          "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
          "table: ID < 0 or ID > max ");
      return;
    } else {
      word_id[0] = static_cast<int32_t const*>(inputIds1)[seqPos];
    }

    if (static_cast<int32_t const*>(inputIds2)[seqPos] < 0 ||
        static_cast<int32_t const*>(inputIds2)[seqPos] >= IdsSize2) {
      printf(
          "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
          "table: ID < 0 or ID > max ");
      return;
    } else {
      word_id[1] = static_cast<int32_t const*>(inputIds2)[seqPos];
    }
  }
  __syncthreads();

  // 2. load pos/tok/word embeddings and add them together
  // offset into embeddings is given by wordId * hidden_size
  const int32_t poffset = blockIdx.x * ld;
  const int32_t outOffset = seqPos * ld;
  // the output offset is given by b * (S*hidden_size) + s * hidden_size
  kvp<T> threadData(0, 0);

  for (int32_t it = threadIdx.x; it < ld; it += TPB) {
    T p(mIdsEmbDev0[poffset + it]);  // pos id
    T val = p;
    const int32_t offset0 = word_id[0] * ld;
    val += mIdsEmbDev1[offset0 + it];
    const int32_t offset1 = word_id[1] * ld;
    val += mIdsEmbDev2[offset1 + it];
    output[outOffset + it] = val;
    skip[outOffset + it] = val;

    const T rldval = rld * val;
    threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
  }

  // 3. layer norm on the sum
  layerNorm<T, T, float, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T, unsigned TPB>
__global__ void embLayerNormKernelMTron_4(int32_t ld,
                                          int32_t const* inputIds0,
                                          int32_t const* inputIds1,
                                          int32_t const* inputIds2,
                                          int32_t const* inputIds3,
                                          float const* beta,
                                          float const* gamma,
                                          T const* mIdsEmbDev0,
                                          T const* mIdsEmbDev1,
                                          T const* mIdsEmbDev2,
                                          T const* mIdsEmbDev3,
                                          int32_t IdsSize0,
                                          int32_t IdsSize1,
                                          int32_t IdsSize2,
                                          int32_t IdsSize3,
                                          T* output,
                                          T* skip) {
  cub::Sum pairSum;
  const int32_t s = blockIdx.x;
  const int32_t b = blockIdx.y;
  const int32_t sumS = inputIds0[b];
  const int32_t s_b = inputIds0[b + 1] - sumS;
  if (s >= s_b) {
    return;  // This CTA has nothing to do
  }
  const T rld = T(1.f) / T(ld);
  const int32_t seqPos = sumS + s;
  extern __shared__ int32_t word_id[];

  if (threadIdx.x == 0) {
    if (static_cast<int32_t const*>(inputIds1)[seqPos] < 0 ||
        static_cast<int32_t const*>(inputIds1)[seqPos] >= IdsSize1) {
      printf(
          "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
          "table: ID < 0 or ID > max ");
      return;
    } else {
      word_id[0] = static_cast<int32_t const*>(inputIds1)[seqPos];
    }

    if (static_cast<int32_t const*>(inputIds2)[seqPos] < 0 ||
        static_cast<int32_t const*>(inputIds2)[seqPos] >= IdsSize2) {
      printf(
          "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
          "table: ID < 0 or ID > max ");
      return;
    } else {
      word_id[1] = static_cast<int32_t const*>(inputIds2)[seqPos];
    }

    if (static_cast<int32_t const*>(inputIds3)[seqPos] < 0 ||
        static_cast<int32_t const*>(inputIds3)[seqPos] >= IdsSize3) {
      printf(
          "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
          "table: ID < 0 or ID > max ");
      return;
    } else {
      word_id[2] = static_cast<int32_t const*>(inputIds3)[seqPos];
    }
  }
  __syncthreads();

  // 2. load pos/tok/word embeddings and add them together
  // offset into embeddings is given by wordId * hidden_size
  const int32_t poffset = blockIdx.x * ld;
  const int32_t outOffset = seqPos * ld;
  // the output offset is given by b * (S*hidden_size) + s * hidden_size
  kvp<T> threadData(0, 0);

  for (int32_t it = threadIdx.x; it < ld; it += TPB) {
    T p(mIdsEmbDev0[poffset + it]);  // pos id
    T val = p;
    const int32_t offset0 = word_id[0] * ld;
    val += mIdsEmbDev1[offset0 + it];
    const int32_t offset1 = word_id[1] * ld;
    val += mIdsEmbDev2[offset1 + it];
    const int32_t offset2 = word_id[2] * ld;
    val += mIdsEmbDev3[offset2 + it];
    output[outOffset + it] = val;
    skip[outOffset + it] = val;

    const T rldval = rld * val;
    threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
  }

  // 3. layer norm on the sum
  layerNorm<T, T, float, TPB>(threadData, ld, outOffset, beta, gamma, output);
}
template <typename T>
int32_t embSkipLayerNormMTron_2(cudaStream_t stream,
                                int32_t ld,
                                int32_t B,
                                int32_t S,
                                int32_t const* inputIds0,
                                int32_t const* inputIds1,
                                int32_t nbLookupTables,
                                float const* beta,
                                float const* gamma,
                                T const* mIdsEmbDev0,
                                T const* mIdsEmbDev1,
                                int32_t IdsSize0,
                                int32_t IdsSize1,
                                T* output,
                                T* skip) {
  constexpr int32_t tpb = 256;
  dim3 const grid(S, B, 1);
  dim3 const block(tpb, 1, 1);
  size_t cache_size = sizeof(int32_t) * (nbLookupTables - 1);
  embLayerNormKernelMTron_2<T, tpb>
      <<<grid, block, cache_size, stream>>>(ld,
                                            inputIds0,
                                            inputIds1,
                                            beta,
                                            gamma,
                                            mIdsEmbDev0,
                                            mIdsEmbDev1,
                                            IdsSize0,
                                            IdsSize1,
                                            output,
                                            skip);
  return cudaPeekAtLastError();
}

template <typename T>
int32_t embSkipLayerNormMTron_3(cudaStream_t stream,
                                int32_t ld,
                                int32_t B,
                                int32_t S,
                                int32_t const* inputIds0,
                                int32_t const* inputIds1,
                                int32_t const* inputIds2,
                                int32_t nbLookupTables,
                                float const* beta,
                                float const* gamma,
                                T const* mIdsEmbDev0,
                                T const* mIdsEmbDev1,
                                T const* mIdsEmbDev2,
                                int32_t IdsSize0,
                                int32_t IdsSize1,
                                int32_t IdsSize2,
                                T* output,
                                T* skip) {
  constexpr int32_t tpb = 256;
  dim3 const grid(S, B, 1);
  dim3 const block(tpb, 1, 1);
  size_t cache_size = sizeof(int32_t) * (nbLookupTables - 1);
  embLayerNormKernelMTron_3<T, tpb>
      <<<grid, block, cache_size, stream>>>(ld,
                                            inputIds0,
                                            inputIds1,
                                            inputIds2,
                                            beta,
                                            gamma,
                                            mIdsEmbDev0,
                                            mIdsEmbDev1,
                                            mIdsEmbDev2,
                                            IdsSize0,
                                            IdsSize1,
                                            IdsSize2,
                                            output,
                                            skip);
  return cudaPeekAtLastError();
}

template <typename T>
int32_t embSkipLayerNormMTron_4(cudaStream_t stream,
                                int32_t ld,
                                int32_t B,
                                int32_t S,
                                int32_t const* inputIds0,
                                int32_t const* inputIds1,
                                int32_t const* inputIds2,
                                int32_t const* inputIds3,
                                int32_t nbLookupTables,
                                float const* beta,
                                float const* gamma,
                                T const* mIdsEmbDev0,
                                T const* mIdsEmbDev1,
                                T const* mIdsEmbDev2,
                                T const* mIdsEmbDev3,
                                int32_t IdsSize0,
                                int32_t IdsSize1,
                                int32_t IdsSize2,
                                int32_t IdsSize3,
                                T* output,
                                T* skip) {
  constexpr int32_t tpb = 256;
  dim3 const grid(S, B, 1);
  dim3 const block(tpb, 1, 1);
  size_t cache_size = sizeof(int32_t) * (nbLookupTables - 1);
  embLayerNormKernelMTron_4<T, tpb>
      <<<grid, block, cache_size, stream>>>(ld,
                                            inputIds0,
                                            inputIds1,
                                            inputIds2,
                                            inputIds3,
                                            beta,
                                            gamma,
                                            mIdsEmbDev0,
                                            mIdsEmbDev1,
                                            mIdsEmbDev2,
                                            mIdsEmbDev3,
                                            IdsSize0,
                                            IdsSize1,
                                            IdsSize2,
                                            IdsSize3,
                                            output,
                                            skip);
  return cudaPeekAtLastError();
}

template int32_t embSkipLayerNormMTron_2<float>(cudaStream_t,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                int32_t const*,
                                                int32_t const*,
                                                int32_t,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                int32_t,
                                                int32_t,
                                                float*,
                                                float*);

template int32_t embSkipLayerNormMTron_3<float>(cudaStream_t,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                int32_t const*,
                                                int32_t const*,
                                                int32_t const*,
                                                int32_t,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                float*,
                                                float*);

template int32_t embSkipLayerNormMTron_4<float>(cudaStream_t,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                int32_t const*,
                                                int32_t const*,
                                                int32_t const*,
                                                int32_t const*,
                                                int32_t,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                int32_t,
                                                float*,
                                                float*);

template int32_t embSkipLayerNormMTron_2<half>(cudaStream_t,
                                               int32_t,
                                               int32_t,
                                               int32_t,
                                               int32_t const*,
                                               int32_t const*,
                                               int32_t,
                                               float const*,
                                               float const*,
                                               half const*,
                                               half const*,
                                               int32_t,
                                               int32_t,
                                               half*,
                                               half*);

template int32_t embSkipLayerNormMTron_3<half>(cudaStream_t,
                                               int32_t,
                                               int32_t,
                                               int32_t,
                                               int32_t const*,
                                               int32_t const*,
                                               int32_t const*,
                                               int32_t,
                                               float const*,
                                               float const*,
                                               half const*,
                                               half const*,
                                               half const*,
                                               int32_t,
                                               int32_t,
                                               int32_t,
                                               half*,
                                               half*);

template int32_t embSkipLayerNormMTron_4<half>(cudaStream_t,
                                               int32_t,
                                               int32_t,
                                               int32_t,
                                               int32_t const*,
                                               int32_t const*,
                                               int32_t const*,
                                               int32_t const*,
                                               int32_t,
                                               float const*,
                                               float const*,
                                               half const*,
                                               half const*,
                                               half const*,
                                               half const*,
                                               int32_t,
                                               int32_t,
                                               int32_t,
                                               int32_t,
                                               half*,
                                               half*);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
