// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "cub/cub.cuh"

#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/platform/device_context.h"

#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"
#include "paddle/phi/kernels/funcs/fused_token_prune_utils.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

__global__ void compute_token_length(const int32_t* src,
                                     int32_t* dst,
                                     float scale) {
  int32_t it = threadIdx.x;
  dst[it] = max(static_cast<int>((src[it + 1] - src[it]) * scale), 1);
}

template <typename T>
__global__ void fill_index_padding_score(int32_t* token_index,
                                         const T* scores,
                                         int32_t sequence_length,
                                         T* padding_scores) {
  int padding_scores_it = threadIdx.x + blockIdx.x * blockDim.x;
  int scores_it = threadIdx.x + blockIdx.x * sequence_length;
  token_index[padding_scores_it] = threadIdx.x;
  if (threadIdx.x < sequence_length) {
    padding_scores[padding_scores_it] = scores[scores_it];
  } else {
    padding_scores[padding_scores_it] = 0;
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void general_topk_pair_sort(T* in_keys, int32_t* in_out_values) {
  typedef cub::BlockRadixSort<T, BLOCK_THREADS, ITEMS_PER_THREAD, int>
      BlockRadixSort;
  typedef cub::
      BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>
          BlockLoadKey;
  typedef cub::
      BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>
          BlockLoadValue;
  typedef cub::
      BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE>
          BlockStoreKey;
  typedef cub::BlockStore<int,
                          BLOCK_THREADS,
                          ITEMS_PER_THREAD,
                          cub::BLOCK_STORE_TRANSPOSE>
      BlockStoreValue;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKey::TempStorage loadkey;
    typename BlockLoadValue::TempStorage loadvalue;
    typename BlockStoreKey::TempStorage storekey;
    typename BlockStoreValue::TempStorage storevalue;
  } temp_storage;

  int block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;

  T thread_keys[ITEMS_PER_THREAD];
  int thread_values[ITEMS_PER_THREAD];
  BlockLoadKey(temp_storage.loadkey).Load(in_keys + block_offset, thread_keys);
  BlockLoadValue(temp_storage.loadvalue)
      .Load(in_out_values + block_offset, thread_values);
  __syncthreads();

  BlockRadixSort(temp_storage.sort).SortDescending(thread_keys, thread_values);
  __syncthreads();

  BlockStoreValue(temp_storage.storevalue)
      .Store(in_out_values + block_offset, thread_values);
}

__global__ void varlen_prune_token_change_order(
    const half* tokens,
    const int32_t* token_pos,
    const int32_t padding_token_length,
    const int32_t* token_index,
    half* output) {
  int batch = blockIdx.x;
  int token_it = batch * gridDim.y + blockIdx.y;
  int pre_value_it =
      token_it * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
  int token_index_it = batch * padding_token_length + blockIdx.y;

  if (token_index[token_index_it] < token_pos[batch + 1] - token_pos[batch]) {
    output[(token_index[token_index_it] + token_pos[batch]) * gridDim.z *
               blockDim.x +
           blockIdx.z * blockDim.x + threadIdx.x] = tokens[pre_value_it];
  }
}

template <typename T>
__global__ void prune_token_change_order(const T* tokens,
                                         int32_t new_sequence_length,
                                         const int32_t padding_token_length,
                                         const int32_t* token_index,
                                         T* output) {
  int batch = blockIdx.x;
  int token_it = batch * gridDim.y + blockIdx.y;
  int pre_value_it =
      token_it * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
  int token_index_it = batch * padding_token_length + blockIdx.y;

  if (token_index[token_index_it] < new_sequence_length) {
    output[(batch * new_sequence_length + token_index[token_index_it]) *
               gridDim.z * blockDim.x +
           blockIdx.z * blockDim.x + threadIdx.x] = tokens[pre_value_it];
  }
}

template <typename T>
__global__ void prune_token_keep_order(const T* tokens,
                                       int32_t pre_sequence_length,
                                       int32_t new_sequence_length,
                                       const int32_t padding_token_length,
                                       const int32_t* token_index,
                                       T* output0,
                                       int32_t* output1) {
  int batch = blockIdx.x;
  int index = 0;
  for (int i = 0; i < pre_sequence_length; ++i) {
    if (token_index[batch * padding_token_length + i] < new_sequence_length) {
      output0[(batch * new_sequence_length + index) * gridDim.y * blockDim.x +
              blockIdx.y * blockDim.x + threadIdx.x] =
          tokens[(batch * pre_sequence_length + i) * gridDim.y * blockDim.x +
                 blockIdx.y * blockDim.x + threadIdx.x];
      output1[batch * new_sequence_length + index] = i;
      index++;
    }
  }
}

nvinfer1::DimsExprs FusedTokenPrunePluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  auto x_dims = inputs[1], new_mask_dims = inputs[3];
  if (flag_varseqlen_) {
    // max sum of seqlen: ceil(sum / scale) + n -1 >= for(i=0;i<n;i++) {sum +=
    // floor(num(i) / scale)} auto
    // pruned_sum_length=std::ceil(inputs[4].d[0]*new_mask_dims.d[2]/inputs[6].d[1])+
    // inputs[1].d[0] - 1;
    auto pruned_sum_length = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUB,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kSUM,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kCEIL_DIV,
                *expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                        *inputs[4].d[0],
                                        *new_mask_dims.d[2]),
                *inputs[6].d[1]),
            *inputs[1].d[0]),
        *expr_builder.constant(1));
    if (output_index == 0) {
      nvinfer1::DimsExprs ret;
      ret.nbDims = 4;
      ret.d[0] = pruned_sum_length;
      ret.d[1] = x_dims.d[2];
      ret.d[2] = expr_builder.constant(1);
      ret.d[3] = expr_builder.constant(1);
      return ret;
    } else if (output_index == 1) {
      nvinfer1::DimsExprs ret;
      ret.nbDims = 2;
      ret.d[0] = new_mask_dims.d[0];
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    } else if (output_index == 2) {
      // word id
      nvinfer1::DimsExprs ret;
      ret.nbDims = 1;
      ret.d[0] = pruned_sum_length;
      return ret;
    } else if (output_index == 3) {
      // pos id
      nvinfer1::DimsExprs ret = inputs[5];
      return ret;
    } else if (output_index == 4) {
      // mask id
      nvinfer1::DimsExprs ret;
      ret.nbDims = 2;
      ret.d[0] = inputs[6].d[0];
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    }
  } else {
    if (output_index == 0) {
      nvinfer1::DimsExprs ret = x_dims;
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    } else {
      nvinfer1::DimsExprs ret;
      ret.nbDims = 2;
      ret.d[0] = new_mask_dims.d[0];
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    }
  }
}

bool FusedTokenPrunePluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (flag_varseqlen_) {
    if (pos <= 3 || pos == 7) {
      if (with_fp16_) {
        return (in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
      } else {
        PADDLE_THROW(
            common::errors::Fatal("The FusedTokenPrune TRT Plugin's input type "
                                  "should be half for varseqlen."));
      }
    } else if (pos == 6 || pos == 11) {  // mask_id, mask_id_out
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    } else {
      return in.type == nvinfer1::DataType::kINT32 &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  } else {
    if (pos == 0) {
      if (with_fp16_) {
        return (in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
      } else {
        return (in.type == nvinfer1::DataType::kFLOAT) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
      }
    } else if (pos <= 4) {
      const nvinfer1::PluginTensorDesc& prev = in_out[0];
      return in.type == prev.type && in.format == prev.format;
    } else {
      return in.type == nvinfer1::DataType::kINT32 &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
}

nvinfer1::DataType FusedTokenPrunePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (flag_varseqlen_) {
    if (index == 0) {
      return nvinfer1::DataType::kHALF;
    } else if (index == 4) {  // mask id
      return input_types[6];
    } else {
      // index = 1,2,3
      return nvinfer1::DataType::kINT32;
    }
  } else {
    if (index == 0) {
      return input_types[1];
    } else {
      // index = 1
      return nvinfer1::DataType::kINT32;
    }
  }
}

size_t FusedTokenPrunePluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nb_outputs) const TRT_NOEXCEPT {
  auto attn_dims = inputs[0].dims;
  auto x_dims = inputs[1].dims;
  auto new_mask_dims = inputs[3].dims;
  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];

  int slimmed_x_len = new_mask_dims.d[2];
  int total = bsz * nb_head * max_seq_len * max_seq_len;
  size_t size = total * sizeof(float);
  size += bsz * max_seq_len * sizeof(float);
  size += bsz * max_seq_len * sizeof(int32_t);
  size += bsz * max_seq_len * sizeof(float);
  size += bsz * max_seq_len * sizeof(int32_t);
  size += (bsz + 1) * sizeof(int);
  size += bsz * slimmed_x_len * sizeof(int32_t);
  return size;
}

int FusedTokenPrunePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  if (flag_varseqlen_) {
    if (!(input_desc[0].type == nvinfer1::DataType::kHALF &&
          input_desc[1].type == nvinfer1::DataType::kHALF)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Token_prune'type must half for varseqlen"));
    }
    float scale =
        static_cast<float>(input_desc[3].dims.d[2]) / input_desc[2].dims.d[2];
    const int32_t* input5 =
        static_cast<const int32_t*>(inputs[5]);            // pre pos id
    int32_t* output3 = static_cast<int32_t*>(outputs[3]);  // new pos id
    half* output0 = static_cast<half*>(outputs[0]);
    const int32_t B = input_desc[1].dims.d[0];  // batches
    const int32_t max_sequence_length =
        input_desc[1].dims.d[1];                     // max sequence length
    const int32_t length = input_desc[1].dims.d[2];  // hidden size
    const half* scores = static_cast<const half*>(inputs[0]);  // reduce sum
    const half* tokens = static_cast<const half*>(inputs[1]);
    int32_t padding_token_length;
    if (max_sequence_length <= 64) {
      padding_token_length = 64;
    } else if (max_sequence_length <= 128) {
      padding_token_length = 128;
    } else if (max_sequence_length <= 256) {
      padding_token_length = 256;
    } else if (max_sequence_length <= 384) {
      padding_token_length = 384;
    } else if (max_sequence_length <= 512) {
      padding_token_length = 512;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Token_prune'token_length must <= 512"));
    }

    // 1. Compute the token length after pruning.
    compute_token_length<<<1, B, 0, stream>>>(
        input5, pruned_token_lengths_, scale);

    // 2. Padding scores
    fill_index_padding_score<half><<<B, padding_token_length, 0, stream>>>(
        token_index_,
        scores,
        max_sequence_length,
        static_cast<half*>(padding_scores_));

    // 3. compute new pos id
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        NULL, temp_storage_bytes, pruned_token_lengths_, output3, B + 1);
    // Allocate temporary storage

    phi::GPUPlace place(platform::GetCurrentDeviceId());
    auto d_temp_storage = phi::memory_utils::Alloc(place, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                  temp_storage_bytes,
                                  pruned_token_lengths_,
                                  output3,
                                  B + 1);

    // 4. sort scores
    if (padding_token_length == 64) {
      general_topk_pair_sort<half, 32, 2><<<B, 32, 0, stream>>>(
          static_cast<half*>(padding_scores_), token_index_);  // 64
    } else if (padding_token_length == 128) {
      general_topk_pair_sort<half, 32, 4><<<B, 32, 0, stream>>>(
          static_cast<half*>(padding_scores_), token_index_);  // 128
    } else if (padding_token_length == 256) {
      general_topk_pair_sort<half, 64, 4><<<B, 64, 0, stream>>>(
          static_cast<half*>(padding_scores_), token_index_);  // 256
    } else if (padding_token_length == 384) {
      general_topk_pair_sort<half, 96, 4><<<B, 96, 0, stream>>>(
          static_cast<half*>(padding_scores_), token_index_);  // 384
    } else {
      general_topk_pair_sort<half, 128, 4><<<B, 128, 0, stream>>>(
          static_cast<half*>(padding_scores_), token_index_);  // 512
    }

    // 5. compute output
    int32_t num_threads;
    if (length < 1024) {
      num_threads = length;
    } else {
      if (length % 512 == 0) {
        num_threads = 512;
      } else if (length % 256 == 0) {
        num_threads = 256;
      } else if (length % 128 == 0) {
        num_threads = 128;
      } else if (length % 64 == 0) {
        num_threads = 64;
      } else if (length % 32 == 0) {
        num_threads = 32;
      } else if (length % 16 == 0) {
        num_threads = 16;
      } else if (length % 8 == 0) {
        num_threads = 8;
      } else if (length % 4 == 0) {
        num_threads = 4;
      } else if (length % 2 == 0) {
        num_threads = 2;
      } else {
        num_threads = 1;
      }
    }
    const dim3 num_blocks(
        B,
        max_sequence_length,
        length / num_threads);  // batches, max_sequence_length, vector_length
    varlen_prune_token_change_order<<<num_blocks, num_threads, 0, stream>>>(
        tokens, output3, padding_token_length, token_index_, output0);
  } else {
    auto input_type = input_desc[0].type;
    const int32_t B = input_desc[1].dims.d[0];  // batches
    const int32_t pre_sequence_length = input_desc[1].dims.d[1];
    const int32_t new_sequence_length = input_desc[3].dims.d[2];  // new mask
    const int32_t length = input_desc[1].dims.d[2];               // hidden size
    if (input_type == nvinfer1::DataType::kFLOAT) {
      VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp32";
      const float* scores = static_cast<const float*>(inputs[0]);  // reduce sum
      const float* tokens = static_cast<const float*>(inputs[1]);  // X
      float* output0 = static_cast<float*>(outputs[0]);
      int32_t* output1 = static_cast<int32_t*>(outputs[1]);
      int32_t padding_token_length;
      if (pre_sequence_length <= 64) {
        padding_token_length = 64;
      } else if (pre_sequence_length <= 128) {
        padding_token_length = 128;
      } else if (pre_sequence_length <= 256) {
        padding_token_length = 256;
      } else if (pre_sequence_length <= 384) {
        padding_token_length = 384;
      } else if (pre_sequence_length <= 512) {
        padding_token_length = 512;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Token_prune'token_length must <= 512"));
      }

      // 1. Padding scores
      fill_index_padding_score<float><<<B, padding_token_length, 0, stream>>>(
          token_index_,
          scores,
          pre_sequence_length,
          static_cast<float*>(padding_scores_));

      // 2. sort scores
      if (padding_token_length == 64) {
        general_topk_pair_sort<float, 32, 2><<<B, 32, 0, stream>>>(
            static_cast<float*>(padding_scores_), token_index_);  // 64
      } else if (padding_token_length == 128) {
        general_topk_pair_sort<float, 32, 4><<<B, 32, 0, stream>>>(
            static_cast<float*>(padding_scores_), token_index_);  // 128
      } else if (padding_token_length == 256) {
        general_topk_pair_sort<float, 64, 4><<<B, 64, 0, stream>>>(
            static_cast<float*>(padding_scores_), token_index_);  // 256
      } else if (padding_token_length == 384) {
        general_topk_pair_sort<float, 96, 4><<<B, 96, 0, stream>>>(
            static_cast<float*>(padding_scores_), token_index_);  // 384
      } else {
        general_topk_pair_sort<float, 128, 4><<<B, 128, 0, stream>>>(
            static_cast<float*>(padding_scores_), token_index_);  // 512
      }

      // 3. compute output
      int32_t num_threads;
      if (length < 1024) {
        num_threads = length;
      } else {
        if (length % 512 == 0) {
          num_threads = 512;
        } else if (length % 256 == 0) {
          num_threads = 256;
        } else if (length % 128 == 0) {
          num_threads = 128;
        } else if (length % 64 == 0) {
          num_threads = 64;
        } else if (length % 32 == 0) {
          num_threads = 32;
        } else if (length % 16 == 0) {
          num_threads = 16;
        } else if (length % 8 == 0) {
          num_threads = 8;
        } else if (length % 4 == 0) {
          num_threads = 4;
        } else if (length % 2 == 0) {
          num_threads = 2;
        } else {
          num_threads = 1;
        }
      }
      if (keep_order_) {
        const dim3 num_blocks(B, length / num_threads);
        prune_token_keep_order<float>
            <<<num_blocks, num_threads, 0, stream>>>(tokens,
                                                     pre_sequence_length,
                                                     new_sequence_length,
                                                     padding_token_length,
                                                     token_index_,
                                                     output0,
                                                     output1);
      } else {
        const dim3 num_blocks(B, pre_sequence_length, length / num_threads);
        prune_token_change_order<float>
            <<<num_blocks, num_threads, 0, stream>>>(tokens,
                                                     new_sequence_length,
                                                     padding_token_length,
                                                     token_index_,
                                                     output0);
      }
    } else if (input_type == nvinfer1::DataType::kHALF) {
      VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp16";
      const half* scores = static_cast<const half*>(inputs[0]);  // reduce sum
      const half* tokens = static_cast<const half*>(inputs[1]);  // X
      half* output0 = static_cast<half*>(outputs[0]);
      int32_t* output1 = static_cast<int32_t*>(outputs[1]);
      int32_t padding_token_length;
      if (pre_sequence_length <= 64) {
        padding_token_length = 64;
      } else if (pre_sequence_length <= 128) {
        padding_token_length = 128;
      } else if (pre_sequence_length <= 256) {
        padding_token_length = 256;
      } else if (pre_sequence_length <= 384) {
        padding_token_length = 384;
      } else if (pre_sequence_length <= 512) {
        padding_token_length = 512;
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Token_prune'token_length must <= 512"));
      }

      // 1. Padding scores
      fill_index_padding_score<half><<<B, padding_token_length, 0, stream>>>(
          token_index_,
          scores,
          pre_sequence_length,
          static_cast<half*>(padding_scores_));

      // 2. sort scores
      if (padding_token_length == 64) {
        general_topk_pair_sort<half, 32, 2><<<B, 32, 0, stream>>>(
            static_cast<half*>(padding_scores_), token_index_);  // 64
      } else if (padding_token_length == 128) {
        general_topk_pair_sort<half, 32, 4><<<B, 32, 0, stream>>>(
            static_cast<half*>(padding_scores_), token_index_);  // 128
      } else if (padding_token_length == 256) {
        general_topk_pair_sort<half, 64, 4><<<B, 64, 0, stream>>>(
            static_cast<half*>(padding_scores_), token_index_);  // 256
      } else if (padding_token_length == 384) {
        general_topk_pair_sort<half, 96, 4><<<B, 96, 0, stream>>>(
            static_cast<half*>(padding_scores_), token_index_);  // 384
      } else {
        general_topk_pair_sort<half, 128, 4><<<B, 128, 0, stream>>>(
            static_cast<half*>(padding_scores_), token_index_);  // 512
      }

      // 3. compute output
      int32_t num_threads;
      if (length < 1024) {
        num_threads = length;
      } else {
        if (length % 512 == 0) {
          num_threads = 512;
        } else if (length % 256 == 0) {
          num_threads = 256;
        } else if (length % 128 == 0) {
          num_threads = 128;
        } else if (length % 64 == 0) {
          num_threads = 64;
        } else if (length % 32 == 0) {
          num_threads = 32;
        } else if (length % 16 == 0) {
          num_threads = 16;
        } else if (length % 8 == 0) {
          num_threads = 8;
        } else if (length % 4 == 0) {
          num_threads = 4;
        } else if (length % 2 == 0) {
          num_threads = 2;
        } else {
          num_threads = 1;
        }
      }
      if (keep_order_) {
        const dim3 num_blocks(B, length / num_threads);
        prune_token_keep_order<half>
            <<<num_blocks, num_threads, 0, stream>>>(tokens,
                                                     pre_sequence_length,
                                                     new_sequence_length,
                                                     padding_token_length,
                                                     token_index_,
                                                     output0,
                                                     output1);
      } else {
        const dim3 num_blocks(B, pre_sequence_length, length / num_threads);
        prune_token_change_order<half>
            <<<num_blocks, num_threads, 0, stream>>>(tokens,
                                                     new_sequence_length,
                                                     padding_token_length,
                                                     token_index_,
                                                     output0);
      }
    } else {
      PADDLE_THROW(
          common::errors::Fatal("The FusedTokenPrune TRT Plugin's input type "
                                "should be float or half."));
    }
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
