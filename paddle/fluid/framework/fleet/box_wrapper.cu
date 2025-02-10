// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_BOX_PS
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>

#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
__global__ void PullCopy(
    float** dest,
    const boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>* src,
    const int64_t* len,
    int hidden,
    int expand_dim,
    int slot_num,
    int total_len,
    uint64_t** keys) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * hidden) = 0;
      *(dest[x] + y * hidden + 1) = 0;
      *(dest[x] + y * hidden + 2) = 0;
    } else {
      *(dest[x] + y * hidden) = (src + i)->show;
      *(dest[x] + y * hidden + 1) = (src + i)->clk;
      *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
    }
    if ((src + i)->embedding_size == 0 || *(keys[x] + y) == 0) {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = 0;
      }
    } else {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = (src + i)->embedx[1 + j];
      }
    }
    // process embed_expand
    if (expand_dim > 0) {
      int z = x + slot_num;
      if ((src + i)->embed_expand_size[0] == 0 || *(keys[x] + y) == 0) {
        for (int j = 0; j < expand_dim; j++) {
          *(dest[z] + y * expand_dim + j) = 0;
        }
      } else {
        for (int j = 0; j < expand_dim; j++) {
          *(dest[z] + y * expand_dim + j) = (src + i)->embed_expand[1 + j];
        }
      }
    }
  }  // end kernel loop
}

__global__ void CopyKeysKernel(uint64_t** src_keys,
                               uint64_t* dest_total_keys,
                               const int64_t* len,
                               int slot_num,
                               int total_len) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    dest_total_keys[i] = src_keys[x][y];
  }
}

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
__global__ void PushCopy(
    boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>* dest,
    float** src,
    int64_t* len,
    int hidden,
    int expand_dim,
    int slot_num,
    int total_len,
    int bs,
    int* slot_vector) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    (dest + i)->slot = slot_vector[x];
    (dest + i)->show = *(src[x] + y * hidden);
    (dest + i)->clk = *(src[x] + y * hidden + 1);
    (dest + i)->embed_g = *(src[x] + y * hidden + 2) * -1. * bs;
    for (int j = 0; j < hidden - 3; j++) {
      (dest + i)->embedx_g[j] = *(src[x] + y * hidden + 3 + j) * -1. * bs;
    }
    if (expand_dim > 0) {
      int z = x + slot_num;
      for (int j = 0; j < expand_dim; j++) {
        (dest + i)->embed_expand_g[j] =
            *(src[z] + y * expand_dim + j) * -1. * bs;
      }
    }
  }
}

void BoxWrapper::CopyForPull(const phi::Place& place,
                             uint64_t** gpu_keys,
                             const std::vector<float*>& values,
                             void* total_values_gpu,
                             const int64_t* gpu_len,
                             const int slot_num,
                             const int hidden_size,
                             const int expand_embed_dim,
                             const int64_t total_length) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  auto buf_value = memory::Alloc(place, values.size() * sizeof(float*));
  float** gpu_values = reinterpret_cast<float**>(buf_value->ptr());
#ifdef PADDLE_WITH_HIP
  hipMemcpy(gpu_values,
            values.data(),
            values.size() * sizeof(float*),
            hipMemcpyHostToDevice);
#else
  cudaMemcpy(gpu_values,
             values.data(),
             values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
#endif
#define EMBEDX_CASE(i, ...)                                                    \
  case i: {                                                                    \
    constexpr size_t EmbedxDim = i;                                            \
    switch (expand_embed_dim) {                                                \
      __VA_ARGS__                                                              \
      default:                                                                 \
        PADDLE_THROW(common::errors::InvalidArgument(                          \
            "Unsupported this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                          \
  } break

#ifdef PADDLE_WITH_HIP
#define EXPAND_EMBED_PUSH_CASE(i, ...)                                   \
  case i: {                                                              \
    constexpr size_t ExpandDim = i;                                      \
    hipLaunchKernelGGL(                                                  \
        PushCopy<EmbedxDim, ExpandDim>,                                  \
        dim3((total_length + 512 - 1) / 512),                            \
        dim3(512),                                                       \
        0,                                                               \
        stream,                                                          \
        gpu_values,                                                      \
        reinterpret_cast<boxps::FeatureValueGpu<EmbedxDim, ExpandDim>*>( \
            total_values_gpu),                                           \
        gpu_len,                                                         \
        hidden_size,                                                     \
        expand_embed_dim,                                                \
        slot_num,                                                        \
        total_length,                                                    \
        gpu_keys);                                                       \
  } break
#else
#define EXPAND_EMBED_PULL_CASE(i, ...)                                       \
  case i: {                                                                  \
    constexpr size_t ExpandDim = i;                                          \
    PullCopy<EmbedxDim, ExpandDim>                                           \
        <<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(                \
            gpu_values,                                                      \
            reinterpret_cast<boxps::FeatureValueGpu<EmbedxDim, ExpandDim>*>( \
                total_values_gpu),                                           \
            gpu_len,                                                         \
            hidden_size,                                                     \
            expand_embed_dim,                                                \
            slot_num,                                                        \
            total_length,                                                    \
            gpu_keys);                                                       \
  } break
#endif

  switch (hidden_size - 3) {
    EMBEDX_CASE(8, EXPAND_EMBED_PULL_CASE(0); EXPAND_EMBED_PULL_CASE(8);
                EXPAND_EMBED_PULL_CASE(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PULL_CASE(0););
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported this embedding size [%d]", hidden_size - 3));
  }
  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PULL_CASE
#undef EMBEDX_CASE
}

void BoxWrapper::CopyKeys(const phi::Place& place,
                          uint64_t** origin_keys,
                          uint64_t* total_keys,
                          const int64_t* gpu_len,
                          int slot_num,
                          int total_len) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(CopyKeysKernel,
                     dim3((total_len + 512 - 1) / 512),
                     dim3(512),
                     0,
                     stream,
                     origin_keys,
                     total_keys,
                     gpu_len,
                     slot_num,
                     total_len);
  hipStreamSynchronize(stream);
#else
  CopyKeysKernel<<<(total_len + 512 - 1) / 512, 512, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
#endif
}

void BoxWrapper::CopyForPush(const phi::Place& place,
                             const std::vector<const float*>& grad_values,
                             void* total_grad_values_gpu,
                             const std::vector<int64_t>& slot_lengths,
                             const int hidden_size,
                             const int expand_embed_dim,
                             const int64_t total_length,
                             const int batch_size) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  auto slot_lengths_lod = slot_lengths;
  for (int i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  auto buf_grad_value =
      memory::Alloc(place, grad_values.size() * sizeof(float*));
  auto buf_length = memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
  auto buf_slot_vector =
      memory::Alloc(place, slot_lengths_lod.size() * sizeof(int));

  float** gpu_values = reinterpret_cast<float**>(buf_grad_value->ptr());
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector->ptr());

#ifdef PADDLE_WITH_HIP
  hipMemcpy(gpu_values,
            grad_values.data(),
            grad_values.size() * sizeof(float*),
            hipMemcpyHostToDevice);
  hipMemcpy(gpu_len,
            slot_lengths_lod.data(),
            slot_lengths.size() * sizeof(int64_t),
            hipMemcpyHostToDevice);
  hipMemcpy(d_slot_vector,
            slot_vector_.data(),
            slot_lengths_lod.size() * sizeof(int),
            hipMemcpyHostToDevice);
#else
  cudaMemcpy(gpu_values,
             grad_values.data(),
             grad_values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_len,
             slot_lengths_lod.data(),
             slot_lengths.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_slot_vector,
             slot_vector_.data(),
             slot_lengths_lod.size() * sizeof(int),
             cudaMemcpyHostToDevice);
#endif

#define EMBEDX_CASE(i, ...)                                                    \
  case i: {                                                                    \
    constexpr size_t EmbedxDim = i;                                            \
    switch (expand_embed_dim) {                                                \
      __VA_ARGS__                                                              \
      default:                                                                 \
        PADDLE_THROW(common::errors::InvalidArgument(                          \
            "Unsupported this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                          \
  } break

#ifdef PADDLE_WITH_HIP
#define EXPAND_EMBED_PUSH_CASE(i, ...)                                       \
  case i: {                                                                  \
    constexpr size_t ExpandDim = i;                                          \
    hipLaunchKernelGGL(PushCopy<EmbedxDim, ExpandDim>,                       \
        dim3(total_length + 512 - 1) / 512), dim3(512), 0, stream,           \
        reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
            total_grad_values_gpu),                                          \
        gpu_values, gpu_len, hidden_size, expand_embed_dim,                  \
        slot_lengths.size(), total_length, batch_size, d_slot_vector);       \
  } break
#else
#define EXPAND_EMBED_PUSH_CASE(i, ...)                                       \
  case i: {                                                                  \
    constexpr size_t ExpandDim = i;                                          \
    PushCopy<EmbedxDim,                                                      \
             ExpandDim><<<(total_length + 512 - 1) / 512, 512, 0, stream>>>( \
        reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
            total_grad_values_gpu),                                          \
        gpu_values,                                                          \
        gpu_len,                                                             \
        hidden_size,                                                         \
        expand_embed_dim,                                                    \
        slot_lengths.size(),                                                 \
        total_length,                                                        \
        batch_size,                                                          \
        d_slot_vector);                                                      \
  } break
#endif

  switch (hidden_size - 3) {
    EMBEDX_CASE(8, EXPAND_EMBED_PUSH_CASE(0); EXPAND_EMBED_PUSH_CASE(8);
                EXPAND_EMBED_PUSH_CASE(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PUSH_CASE(0););
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported this embedding size [%d]", hidden_size - 3));
  }

  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PUSH_CASE
#undef EMBEDX_CASE
}

}  // namespace framework
}  // namespace paddle
#endif
