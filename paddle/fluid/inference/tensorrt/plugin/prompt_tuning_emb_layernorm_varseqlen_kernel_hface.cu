// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
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
#include "paddle/fluid/inference/tensorrt/plugin/prompt_tuning_emb_layernorm_varseqlen_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T, unsigned TPB>
__global__ void prompt_tuning_embKernel(int32_t B,
                                        int32_t ld,
                                        int32_t const* inputIds0,
                                        int32_t const* inputIds1,
                                        int32_t const* inputIds2,
                                        T const* dense_vector,
                                        float const* beta,
                                        float const* gamma,
                                        T const* mIdsEmbDev0,
                                        T const* mIdsEmbDev1,
                                        T const* mIdsEmbDev2,
                                        int32_t IdsSize0,
                                        int32_t IdsSize1,
                                        int32_t IdsSize2,
                                        T* output,
                                        int32_t* new_pos_id) {
  cub::Sum pairSum;
  int32_t const s = blockIdx.x;
  int32_t const b = blockIdx.y;
  int32_t const sumS = inputIds0[b];
  int32_t const s_b = inputIds0[b + 1] - inputIds0[b];

  int32_t const new_sumS = sumS + b;

  // new pos_id: Add an id to each sentence
  new_pos_id[b] = new_sumS;

  // last id
  if (b == B - 1) {
    new_pos_id[B] = inputIds0[B] + B;
  }

  T const rld = T(1.f) / T(ld);
  int32_t const seqPos = sumS + s;
  int32_t const out_seqPos = new_sumS + s + 1;
  int32_t const new_out_seqPos = new_sumS + s;

  kvp<T> threadData(0, 0);

  int32_t const new_outoffset = new_out_seqPos * ld;
  int32_t const prompt_tuning_offset = new_sumS * ld;
  int32_t const dense_vector_offset = b * ld;

  if (s < s_b) {
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
    int32_t const poffset = blockIdx.x * ld;
    int32_t const outoffset = out_seqPos * ld;

    // the output offset is given by b * (S*hidden_size) + s * hidden_size

    for (int32_t it = threadIdx.x; it < ld; it += TPB) {
      T p(mIdsEmbDev0[poffset + it]);  // pos id
      T val = p;
      int32_t const offset0 = word_id[0] * ld;
      val += mIdsEmbDev1[offset0 + it];
      int32_t const offset1 = word_id[1] * ld;
      val += mIdsEmbDev2[offset1 + it];
      output[outoffset + it] = val;
      T const rldval = rld * val;
      threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }
    // 3. layer norm on the sum
    layerNorm<T, T, float, TPB>(threadData, ld, outoffset, beta, gamma, output);
  } else if (s == s_b) {
    for (int32_t it = threadIdx.x; it < ld; it += TPB) {
      T val = dense_vector[dense_vector_offset + it];
      output[prompt_tuning_offset + it] = val;
      T const rldval = rld * val;
      threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
      // 3. layer norm on the sum
    }
    layerNorm<T, T, float, TPB>(
        threadData, ld, prompt_tuning_offset, beta, gamma, output);

  } else {
    return;  // This CTA has nothing to do
  }
}

template <typename T>
int32_t prompt_tuning_emb(cudaStream_t stream,
                          int32_t ld,
                          int32_t B,
                          int32_t S,
                          int const* inputIds0,
                          int const* inputIds1,
                          int const* inputIds2,
                          T const* dense_vector,
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
                          int32_t* new_pos_id) {
  constexpr int32_t tpb = 256;
  dim3 const grid(S, B, 1);
  dim3 const block(tpb, 1, 1);
  size_t cache_size = sizeof(int32_t) * (nbLookupTables - 1);
  prompt_tuning_embKernel<T, tpb>
      <<<grid, block, cache_size, stream>>>(B,
                                            ld,
                                            inputIds0,
                                            inputIds1,
                                            inputIds2,
                                            dense_vector,
                                            beta,
                                            gamma,
                                            mIdsEmbDev0,
                                            mIdsEmbDev1,
                                            mIdsEmbDev2,
                                            IdsSize0,
                                            IdsSize1,
                                            IdsSize2,
                                            output,
                                            new_pos_id);
  return cudaPeekAtLastError();
}

template int32_t prompt_tuning_emb<half>(cudaStream_t,
                                         int32_t,
                                         int32_t,
                                         int32_t,
                                         int32_t const*,
                                         int32_t const*,
                                         int32_t const*,
                                         half const*,
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
                                         int32_t*);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
