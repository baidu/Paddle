/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)

#include "paddle/fluid/framework/data_feed.h"
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <sstream>
#include "cub/cub.cuh"
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/kernels/funcs/shuffle_batch.cu.h"
#include "paddle/phi/kernels/gpu/graph_reindex_funcs.h"
#include "paddle/phi/kernels/graph_reindex_kernel.h"

COMMON_DECLARE_bool(enable_opt_get_features);
COMMON_DECLARE_bool(graph_metapath_split_opt);
COMMON_DECLARE_int32(gpugraph_storage_mode);
COMMON_DECLARE_double(gpugraph_hbm_table_load_factor);
COMMON_DECLARE_bool(enable_graph_multi_node_sampling);
COMMON_DECLARE_bool(query_dest_rank_by_multi_node);
COMMON_DECLARE_string(graph_edges_split_mode);

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define DEBUG_STATE(state)                                             \
  VLOG(2) << "left: " << state->left << " right: " << state->right     \
          << " central_word: " << state->central_word                  \
          << " step: " << state->step << " cursor: " << state->cursor  \
          << " len: " << state->len << " row_num: " << state->row_num; \
// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void fill_idx(T *idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

/**
 * @brief sort cub
 */
template <typename K, typename V>
void cub_sort_pairs(const int gpu_id,
                    int len,
                    const K *in_keys,
                    K *out_keys,
                    const V *in_vals,
                    V *out_vals,
                    cudaStream_t stream,
                    std::shared_ptr<phi::Allocation> &d_buf,  // NOLINT
                    const phi::Place &place) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL,
                                             temp_storage_bytes,
                                             in_keys,
                                             out_keys,
                                             in_vals,
                                             out_vals,
                                             len,
                                             0,
                                             8 * sizeof(K),
                                             stream,
                                             false));
  if (d_buf == NULL || d_buf->size() < temp_storage_bytes) {
    d_buf = memory::AllocShared(
        place,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  }
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_buf->ptr(),
                                             temp_storage_bytes,
                                             in_keys,
                                             out_keys,
                                             in_vals,
                                             out_vals,
                                             len,
                                             0,
                                             8 * sizeof(K),
                                             stream,
                                             false));
}

/**
 * @Brief cub run length encode
 */
template <typename K, typename V, typename TNum>
void cub_runlength_encode(int N,
                          const K *in_keys,
                          K *out_keys,
                          V *out_sizes,
                          TNum *d_out_len,
                          cudaStream_t stream,
                          std::shared_ptr<phi::Allocation> &d_buf,  // NOLINT
                          const phi::Place &place) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(NULL,
                                                temp_storage_bytes,
                                                in_keys,
                                                out_keys,
                                                out_sizes,
                                                d_out_len,
                                                N,
                                                stream));
  if (d_buf == NULL || d_buf->size() < temp_storage_bytes) {
    d_buf = memory::AllocShared(
        place,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  }
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_buf->ptr(),
                                                temp_storage_bytes,
                                                in_keys,
                                                out_keys,
                                                out_sizes,
                                                d_out_len,
                                                N,
                                                stream));
}

/**
 * @brief exclusive sum
 */
template <typename K>
void cub_exclusivesum(const int gpu_id,
                      int N,
                      const K *in,
                      K *out,
                      cudaStream_t stream,
                      std::shared_ptr<phi::Allocation> &d_buf,  // NOLINT
                      const phi::Place &place) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, in, out, N, stream));
  if (d_buf == NULL || d_buf->size() < temp_storage_bytes) {
    d_buf = memory::AllocShared(
        place,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  }
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      d_buf->ptr(), temp_storage_bytes, in, out, N, stream));
}

template <typename T>
__global__ void kernel_fill_restore_idx(size_t N,
                                        const T *d_sorted_idx,
                                        const T *d_offset,
                                        const T *d_merged_cnts,
                                        T *d_restore_idx) {
  CUDA_KERNEL_LOOP(i, N) {
    const T &off = d_offset[i];
    const T &num = d_merged_cnts[i];
    for (size_t k = 0; k < num; k++) {
      d_restore_idx[d_sorted_idx[off + k]] = i;
    }
  }
}

template <typename T>
__global__ void kernel_fill_restore_idx_by_search(size_t N,
                                                  const T *d_sorted_idx,
                                                  size_t merge_num,
                                                  const T *d_offset,
                                                  T *d_restore_idx) {
  CUDA_KERNEL_LOOP(i, N) {
    if (i < d_offset[1]) {
      d_restore_idx[d_sorted_idx[i]] = 0;
      continue;
    }
    int high = merge_num - 1;
    int low = 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < d_offset[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    d_restore_idx[d_sorted_idx[i]] = low;
  }
}

// For unique node and inverse id.
int dedup_keys_and_fillidx(const int gpu_id,
                           const uint64_t *d_keys,
                           int total_nodes_num,
                           uint64_t *d_merged_keys,  // input
                           uint32_t *d_restore_idx,  // inverse
                           const phi::Place &place,
                           cudaStream_t stream) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto d_sorted_keys =
      memory::AllocShared(place,
                          total_nodes_num * sizeof(uint64_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t *d_sorted_keys_ptr =
      reinterpret_cast<uint64_t *>(d_sorted_keys->ptr());
  auto d_sorted_idx =
      memory::AllocShared(place,
                          total_nodes_num * sizeof(uint32_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *d_sorted_idx_ptr =
      reinterpret_cast<uint32_t *>(d_sorted_idx->ptr());
  auto d_offset =
      memory::AllocShared(place,
                          total_nodes_num * sizeof(uint32_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *d_offset_ptr = reinterpret_cast<uint32_t *>(d_offset->ptr());
  auto d_merged_cnts =
      memory::AllocShared(place,
                          total_nodes_num * sizeof(uint32_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *d_merged_cnts_ptr =
      reinterpret_cast<uint32_t *>(d_merged_cnts->ptr());
  std::shared_ptr<phi::Allocation> d_buf = NULL;

  int merged_size = 0;  // Final num
  auto d_index_in =
      memory::Alloc(place,
                    sizeof(uint32_t) * (total_nodes_num + 1),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *d_index_in_ptr = reinterpret_cast<uint32_t *>(d_index_in->ptr());
  int *d_merged_size =
      reinterpret_cast<int *>(&d_index_in_ptr[total_nodes_num]);
  fill_idx<<<GET_BLOCKS(total_nodes_num), CUDA_NUM_THREADS, 0, stream>>>(
      d_index_in_ptr, total_nodes_num);
  cub_sort_pairs(gpu_id,
                 total_nodes_num,
                 d_keys,
                 d_sorted_keys_ptr,
                 d_index_in_ptr,
                 d_sorted_idx_ptr,
                 stream,
                 d_buf,
                 place);
  cub_runlength_encode(total_nodes_num,
                       d_sorted_keys_ptr,
                       d_merged_keys,
                       d_merged_cnts_ptr,
                       d_merged_size,
                       stream,
                       d_buf,
                       place);
  CUDA_CHECK(cudaMemcpyAsync(&merged_size,
                             d_merged_size,
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cub_exclusivesum(gpu_id,
                   merged_size,
                   d_merged_cnts_ptr,
                   d_offset_ptr,
                   stream,
                   d_buf,
                   place);

  if (total_nodes_num < merged_size * 2) {
    kernel_fill_restore_idx<<<GET_BLOCKS(merged_size),
                              CUDA_NUM_THREADS,
                              0,
                              stream>>>(merged_size,
                                        d_sorted_idx_ptr,
                                        d_offset_ptr,
                                        d_merged_cnts_ptr,
                                        d_restore_idx);
  } else {
    // used mid search fill idx when high dedup rate
    kernel_fill_restore_idx_by_search<<<GET_BLOCKS(total_nodes_num),
                                        CUDA_NUM_THREADS,
                                        0,
                                        stream>>>(total_nodes_num,
                                                  d_sorted_idx_ptr,
                                                  merged_size,
                                                  d_offset_ptr,
                                                  d_restore_idx);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return merged_size;
}

// fill slot values
__global__ void FillSlotValueOffsetKernel(const int ins_num,
                                          const int used_slot_num,
                                          size_t *slot_value_offsets,
                                          const int *uint64_offsets,
                                          const int uint64_slot_size,
                                          const int *float_offsets,
                                          const int float_slot_size,
                                          const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(slot_idx, used_slot_num) {
    int value_off = slot_idx * col_num;
    slot_value_offsets[value_off] = 0;

    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * uint64_cols + info.slot_value_idx;
        int num = uint64_offsets[pos + 1] - uint64_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    } else {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * float_cols + info.slot_value_idx;
        int num = float_offsets[pos + 1] - float_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    }
  }
}

struct RandInt {
  int low, high;

  __host__ __device__ RandInt(int low, int high) : low(low), high(high) {}

  __host__ __device__ int operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(low, high);
    rng.discard(n);

    return dist(rng);
  }
};

void SlotRecordInMemoryDataFeed::FillSlotValueOffset(
    const int ins_num,
    const int used_slot_num,
    size_t *slot_value_offsets,
    const int *uint64_offsets,
    const int uint64_slot_size,
    const int *float_offsets,
    const int float_slot_size,
    const UsedSlotGpuType *used_slots,
    cudaStream_t stream) {
  FillSlotValueOffsetKernel<<<GET_BLOCKS(used_slot_num),
                              CUDA_NUM_THREADS,
                              0,
                              stream>>>(ins_num,
                                        used_slot_num,
                                        slot_value_offsets,
                                        uint64_offsets,
                                        uint64_slot_size,
                                        float_offsets,
                                        float_slot_size,
                                        used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void CopyForTensorKernel(const int used_slot_num,
                                    const int ins_num,
                                    void **dest,
                                    const size_t *slot_value_offsets,
                                    const uint64_t *uint64_feas,
                                    const int *uint64_offsets,
                                    const int *uint64_ins_lens,
                                    const int uint64_slot_size,
                                    const float *float_feas,
                                    const int *float_offsets,
                                    const int *float_ins_lens,
                                    const int float_slot_size,
                                    const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(i, ins_num * used_slot_num) {
    int slot_idx = i / ins_num;
    int ins_idx = i % ins_num;

    uint32_t value_offset = slot_value_offsets[slot_idx * col_num + ins_idx];
    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      uint64_t *up = reinterpret_cast<uint64_t *>(dest[slot_idx]);
      int index = info.slot_value_idx + uint64_cols * ins_idx;
      int old_off = uint64_offsets[index];
      int num = uint64_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int uint64_value_offset = uint64_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        up[k + value_offset] = uint64_feas[k + old_off + uint64_value_offset];
      }
    } else {
      float *fp = reinterpret_cast<float *>(dest[slot_idx]);
      int index = info.slot_value_idx + float_cols * ins_idx;
      int old_off = float_offsets[index];
      int num = float_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int float_value_offset = float_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        fp[k + value_offset] = float_feas[k + old_off + float_value_offset];
      }
    }
  }
}

void SlotRecordInMemoryDataFeed::CopyForTensor(
    const int ins_num,
    const int used_slot_num,
    void **dest,
    const size_t *slot_value_offsets,
    const uint64_t *uint64_feas,
    const int *uint64_offsets,
    const int *uint64_ins_lens,
    const int uint64_slot_size,
    const float *float_feas,
    const int *float_offsets,
    const int *float_ins_lens,
    const int float_slot_size,
    const UsedSlotGpuType *used_slots,
    cudaStream_t stream) {
  CopyForTensorKernel<<<GET_BLOCKS(used_slot_num * ins_num),
                        CUDA_NUM_THREADS,
                        0,
                        stream>>>(used_slot_num,
                                  ins_num,
                                  dest,
                                  slot_value_offsets,
                                  uint64_feas,
                                  uint64_offsets,
                                  uint64_ins_lens,
                                  uint64_slot_size,
                                  float_feas,
                                  float_offsets,
                                  float_ins_lens,
                                  float_slot_size,
                                  used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void GraphFillCVMKernel(int64_t *tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { tensor[idx] = 1; }
}

__global__ void CopyDuplicateKeys(int64_t *dist_tensor,
                                  uint64_t *src_tensor,
                                  int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    dist_tensor[idx * 2] = src_tensor[idx];
    dist_tensor[idx * 2 + 1] = src_tensor[idx];
  }
}

#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
int AcquireInstance(BufState *state) {
  if (state->GetNextStep()) {
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextCentralWord()) {
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextBatch()) {
    DEBUG_STATE(state);
    return state->len;
  }
  return 0;
}

__global__ void GraphFillIdKernel(uint64_t *id_tensor,
                                  int32_t *pair_label_tensor,
                                  int *fill_ins_num,
                                  uint64_t *walk,
                                  uint8_t *walk_ntype,
                                  int *row,
                                  int *row_col_shift,
                                  int central_word,
                                  int step,
                                  int len,
                                  int col_num,
                                  uint8_t *excluded_train_pair,
                                  int excluded_train_pair_len,
                                  int32_t *pair_label_conf,
                                  int node_type_num) {
  __shared__ uint64_t local_key[CUDA_NUM_THREADS * 2];
  __shared__ uint8_t local_pair_label[CUDA_NUM_THREADS];
  __shared__ int local_num;
  __shared__ int global_num;
  bool need_filter = false;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  // int dst = idx * 2;
  // id_tensor[dst] = walk[src];
  // id_tensor[dst + 1] = walk[src + step];
  if (idx < len) {
    int col_idx = (central_word + row_col_shift[idx]) % col_num;
    int src = row[idx] * col_num + col_idx;
    int last_row = row[idx] * col_num;
    int next_row = last_row + col_num;

    if ((src + step) >= last_row && (src + step) < next_row && walk[src] != 0 &&
        walk[src + step] != 0) {
      for (int i = 0; i < excluded_train_pair_len; i += 2) {
        if (walk_ntype[src] == excluded_train_pair[i] &&
            walk_ntype[src + step] == excluded_train_pair[i + 1]) {
          // filter this pair
          need_filter = true;
          break;
        }
      }
      if (!need_filter) {
        size_t dst = atomicAdd(&local_num, 1);
        local_key[dst * 2] = walk[src];
        local_key[dst * 2 + 1] = walk[src + step];
        if (pair_label_conf != NULL) {
          int pair_label_conf_index =
              walk_ntype[src] * node_type_num + walk_ntype[src + step];
          local_pair_label[dst] = pair_label_conf[pair_label_conf_index];
        }
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(fill_ins_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    id_tensor[global_num * 2 + 2 * threadIdx.x] = local_key[2 * threadIdx.x];
    id_tensor[global_num * 2 + 2 * threadIdx.x + 1] =
        local_key[2 * threadIdx.x + 1];
    if (pair_label_conf != NULL) {
      pair_label_tensor[global_num + threadIdx.x] =
          local_pair_label[threadIdx.x];
    }
  }
}

__global__ void GraphZeroIdKernel(uint64_t *id_tensor, int len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t zerokey = 0;
  if (idx < len) {
    int dst = idx * 2;
    id_tensor[dst] = zerokey;
    id_tensor[dst + 1] = zerokey;
  }
}

__global__ void GraphFillSlotKernel(uint64_t *id_tensor,
                                    uint64_t *feature_buf,
                                    int len,
                                    int total_ins,
                                    int slot_num,
                                    int *slot_feature_num_map,
                                    int fea_num_per_node,
                                    int *actual_slot_id_map,
                                    int *fea_offset_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int fea_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    int actual_slot_id = actual_slot_id_map[fea_idx];
    int fea_offset = fea_offset_map[fea_idx];
    reinterpret_cast<uint64_t *>(id_tensor[actual_slot_id])
        [ins_idx * slot_feature_num_map[actual_slot_id] + fea_offset] =
            feature_buf[ins_idx * fea_num_per_node + fea_idx];
  }
}

__global__ void GraphFillSlotLodKernelOpt(uint64_t *id_tensor,
                                          int len,
                                          int total_ins,
                                          int *slot_feature_num_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int slot_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    (reinterpret_cast<uint64_t *>(id_tensor[slot_idx]))[ins_idx] =
        ins_idx * slot_feature_num_map[slot_idx];
  }
}

__global__ void GraphFillSlotLodKernel(int64_t *id_tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { id_tensor[idx] = idx; }
}

// fill sage neighbor results
__global__ void FillActualNeighbors(int64_t *vals,
                                    int64_t *actual_vals,
                                    int64_t *actual_vals_dst,
                                    int *actual_sample_size,
                                    int *cumsum_actual_sample_size,
                                    float *weights,
                                    float *actual_weights,
                                    int sample_size,
                                    int len,
                                    int mod,
                                    bool return_weight) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    int offset1 = cumsum_actual_sample_size[i];
    int offset2 = sample_size * i;
    int dst_id = i % mod;
    for (int j = 0; j < actual_sample_size[i]; j++) {
      actual_vals[offset1 + j] = vals[offset2 + j];
      actual_vals_dst[offset1 + j] = dst_id;
      if (return_weight) {
        actual_weights[offset1 + j] = weights[offset2 + j];
      }
    }
  }
}

int GraphDataGenerator::FillIdShowClkTensor(int total_instance,
                                            bool gpu_graph_training) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  show_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({total_instance}, this->place_);
  GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(show_tensor_ptr_, total_instance);
  clk_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
  GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(clk_tensor_ptr_, total_instance);

  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    int feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
    id_tensor_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int64_t>(
        {total_instance, 1}, this->place_);
    if (gpu_graph_training) {
      uint64_t *ins_buf =
          reinterpret_cast<uint64_t *>(d_ins_buf_[tensor_pair_idx]->ptr());
      uint64_t *ins_cursor =
          ins_buf + ins_buf_pair_len_[tensor_pair_idx] * 2 - total_instance;
      cudaMemcpyAsync(id_tensor_ptr_,
                      ins_cursor,
                      sizeof(uint64_t) * total_instance,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);

      if (conf_.enable_pair_label) {
        pair_label_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int32_t>(
            {total_instance / 2}, this->place_);
        int32_t *pair_label_buf = reinterpret_cast<int32_t *>(
            d_pair_label_buf_[tensor_pair_idx]->ptr());
        int32_t *pair_label_cursor = pair_label_buf +
                                     ins_buf_pair_len_[tensor_pair_idx] -
                                     total_instance / 2;
        cudaMemcpyAsync(pair_label_ptr_,
                        pair_label_cursor,
                        sizeof(int32_t) * total_instance / 2,
                        cudaMemcpyDeviceToDevice,
                        train_stream_);
      }
    } else {  // infer
      uint64_t *d_type_keys = reinterpret_cast<uint64_t *>(
          d_device_keys_[tensor_pair_idx][infer_cursor_[tensor_pair_idx]]
              ->ptr());
      d_type_keys += infer_node_start_[tensor_pair_idx];
      infer_node_start_[tensor_pair_idx] += total_instance / 2;
      CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                          CUDA_NUM_THREADS,
                          0,
                          train_stream_>>>(
          id_tensor_ptr_, d_type_keys, total_instance / 2);
    }
  }  // end for (int tensor_pair_idx = 0; tensor_pair_idx <
     // conf_.tensor_pair_num;

  return 0;
}

int GraphDataGenerator::FillGraphIdShowClkTensor(int uniq_instance,
                                                 int total_instance,
                                                 int index) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  show_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({uniq_instance}, this->place_);
  GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(show_tensor_ptr_, uniq_instance);
  clk_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({uniq_instance}, this->place_);
  GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(clk_tensor_ptr_, uniq_instance);

  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    int feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
    id_tensor_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int64_t>(
        {uniq_instance, 1}, this->place_);
    cudaMemcpyAsync(id_tensor_ptr_,
                    final_sage_nodes_vec_[index]->ptr(),
                    sizeof(int64_t) * uniq_instance,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);

    if (conf_.enable_pair_label) {
      pair_label_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int32_t>(
          {total_instance / 2}, this->place_);
      int32_t *pair_label_buf = reinterpret_cast<int32_t *>(
          d_pair_label_buf_[tensor_pair_idx]->ptr());
      int32_t *pair_label_cursor = pair_label_buf +
                                   ins_buf_pair_len_[tensor_pair_idx] -
                                   total_instance / 2;
      cudaMemcpyAsync(pair_label_ptr_,
                      pair_label_vec_[index]->ptr(),
                      sizeof(int32_t) * total_instance / 2,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);
    }

    feed_vec_idx += conf_.slot_num * 2;

    int len_samples = conf_.samples.size();
    int *num_nodes_tensor_ptr_[len_samples];
    int *next_num_nodes_tensor_ptr_[len_samples];
    int64_t *edges_src_tensor_ptr_[len_samples];
    int64_t *edges_dst_tensor_ptr_[len_samples];
    int *edges_split_tensor_ptr_[len_samples];
    float *edges_weight_tensor_ptr_[len_samples];
    std::vector<std::vector<int>> edges_split_num_for_graph =
        edges_split_num_vec_[index];
    std::vector<std::shared_ptr<phi::Allocation>> graph_edges =
        graph_edges_vec_[index];
    int graph_edges_index = 0;
    for (int i = 0; i < len_samples; i++) {
      std::vector<int> edges_split_num = edges_split_num_for_graph[i];

      int neighbor_len = edges_split_num[conf_.edge_to_id_len + 2];
      num_nodes_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int>({1}, this->place_);
      next_num_nodes_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int>({1}, this->place_);
      edges_src_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int64_t>({neighbor_len, 1},
                                                           this->place_);
      edges_dst_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int64_t>({neighbor_len, 1},
                                                           this->place_);
      edges_split_tensor_ptr_[i] = feed_vec_[feed_vec_idx++]->mutable_data<int>(
          {conf_.edge_to_id_len}, this->place_);
      if (conf_.return_weight) {
        edges_weight_tensor_ptr_[i] =
            feed_vec_[feed_vec_idx++]->mutable_data<float>({neighbor_len, 1},
                                                           this->place_);
      }

      // [edges_split_num, next_num_nodes, num_nodes, neighbor_len]
      cudaMemcpyAsync(next_num_nodes_tensor_ptr_[i],
                      edges_split_num.data() + conf_.edge_to_id_len,
                      sizeof(int),
                      cudaMemcpyHostToDevice,
                      train_stream_);
      cudaMemcpyAsync(num_nodes_tensor_ptr_[i],
                      edges_split_num.data() + conf_.edge_to_id_len + 1,
                      sizeof(int),
                      cudaMemcpyHostToDevice,
                      train_stream_);
      cudaMemcpyAsync(edges_split_tensor_ptr_[i],
                      edges_split_num.data(),
                      sizeof(int) * conf_.edge_to_id_len,
                      cudaMemcpyHostToDevice,
                      train_stream_);
      cudaMemcpyAsync(edges_src_tensor_ptr_[i],
                      graph_edges[graph_edges_index++]->ptr(),
                      sizeof(int64_t) * neighbor_len,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);
      cudaMemcpyAsync(edges_dst_tensor_ptr_[i],
                      graph_edges[graph_edges_index++]->ptr(),
                      sizeof(int64_t) * neighbor_len,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);
      if (conf_.return_weight) {
        cudaMemcpyAsync(edges_weight_tensor_ptr_[i],
                        graph_edges[graph_edges_index++]->ptr(),
                        sizeof(float) * neighbor_len,
                        cudaMemcpyDeviceToDevice,
                        train_stream_);
      }
    }  // end for (int i = 0; i < len_samples; i++) {

    index_tensor_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int>(
        {total_instance}, this->place_);
    cudaMemcpyAsync(index_tensor_ptr_,
                    inverse_vec_[index]->ptr(),
                    sizeof(int) * total_instance,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);

    if (conf_.get_degree) {
      degree_tensor_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int>(
          {uniq_instance * conf_.edge_to_id_len}, this->place_);
      cudaMemcpyAsync(degree_tensor_ptr_,
                      node_degree_vec_[index]->ptr(),
                      sizeof(int) * uniq_instance * conf_.edge_to_id_len,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);
    }
  }  // end for (int tensor_pair_idx = 0; tensor_pair_idx <
     // conf_.tensor_pair_num;

  return 0;
}

int GraphDataGenerator::FillGraphIdShowClkTensorAccum(int index) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  VLOG(1) << conf_.gpuid << ": Begin fill in show, cvm and id";
  int fake_accumulate_num = 2;
  for (int accum = 0; accum < fake_accumulate_num; accum++) {
    int uniq_instance = uniq_instance_vec_[2 * index + 1 - accum];
    show_tensor_ptr_ = feed_vec_[3 * accum]->mutable_data<int64_t>(
        {uniq_instance}, this->place_);
    GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance),
                         CUDA_NUM_THREADS,
                         0,
                         train_stream_>>>(show_tensor_ptr_, uniq_instance);
    clk_tensor_ptr_ = feed_vec_[3 * accum + 1]->mutable_data<int64_t>(
        {uniq_instance}, this->place_);
    GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance),
                         CUDA_NUM_THREADS,
                         0,
                         train_stream_>>>(clk_tensor_ptr_, uniq_instance);

    id_tensor_ptr_ = feed_vec_[3 * accum + 2]->mutable_data<int64_t>(
        {uniq_instance, 1}, this->place_);
    cudaMemcpyAsync(id_tensor_ptr_,
                    final_sage_nodes_vec_[2 * index + 1 - accum]->ptr(),
                    sizeof(int64_t) * uniq_instance,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
    cudaStreamSynchronize(train_stream_);

    VLOG(1) << conf_.gpuid << ": finish show, cvm and id for " << accum
            << " uniq_instance: " << uniq_instance;
  }

  VLOG(1) << conf_.gpuid << ": Begin copy graph data";
  for (int accum = 0; accum < fake_accumulate_num; accum++) {
    int feed_vec_idx = 3 * fake_accumulate_num + conf_.slot_num * 2 +
                       accum * conf_.tensor_num_of_one_subgraph;
    int new_index = index * 2 + 1 - accum;
    int uniq_instance = uniq_instance_vec_[new_index];
    int total_instance = total_instance_vec_[new_index];
    VLOG(1) << conf_.gpuid << " accum: " << accum
            << " uniq_instance: " << uniq_instance
            << " total_instance: " << total_instance;
    int len_samples = conf_.samples.size();
    int *num_nodes_tensor_ptr_[len_samples];
    int *next_num_nodes_tensor_ptr_[len_samples];
    int64_t *edges_src_tensor_ptr_[len_samples];
    int64_t *edges_dst_tensor_ptr_[len_samples];
    int *edges_split_tensor_ptr_[len_samples];
    float *edges_weight_tensor_ptr_[len_samples];
    std::vector<std::vector<int>> edges_split_num_for_graph =
        edges_split_num_vec_[new_index];
    std::vector<std::shared_ptr<phi::Allocation>> graph_edges =
        graph_edges_vec_[new_index];
    int graph_edges_index = 0;
    for (int i = 0; i < len_samples; i++) {
      std::vector<int> edges_split_num = edges_split_num_for_graph[i];

      int neighbor_len = edges_split_num[conf_.edge_to_id_len + 2];
      num_nodes_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int>({1}, this->place_);
      next_num_nodes_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int>({1}, this->place_);
      edges_src_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int64_t>({neighbor_len, 1},
                                                           this->place_);
      edges_dst_tensor_ptr_[i] =
          feed_vec_[feed_vec_idx++]->mutable_data<int64_t>({neighbor_len, 1},
                                                           this->place_);
      edges_split_tensor_ptr_[i] = feed_vec_[feed_vec_idx++]->mutable_data<int>(
          {conf_.edge_to_id_len}, this->place_);
      if (conf_.return_weight) {
        edges_weight_tensor_ptr_[i] =
            feed_vec_[feed_vec_idx++]->mutable_data<float>({neighbor_len, 1},
                                                           this->place_);
      }

      // [edges_split_num, next_num_nodes, num_nodes, neighbor_len]
      cudaMemcpyAsync(next_num_nodes_tensor_ptr_[i],
                      edges_split_num.data() + conf_.edge_to_id_len,
                      sizeof(int),
                      cudaMemcpyHostToDevice,
                      train_stream_);
      cudaMemcpyAsync(num_nodes_tensor_ptr_[i],
                      edges_split_num.data() + conf_.edge_to_id_len + 1,
                      sizeof(int),
                      cudaMemcpyHostToDevice,
                      train_stream_);
      cudaMemcpyAsync(edges_split_tensor_ptr_[i],
                      edges_split_num.data(),
                      sizeof(int) * conf_.edge_to_id_len,
                      cudaMemcpyHostToDevice,
                      train_stream_);
      cudaMemcpyAsync(edges_src_tensor_ptr_[i],
                      graph_edges[graph_edges_index++]->ptr(),
                      sizeof(int64_t) * neighbor_len,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);
      cudaMemcpyAsync(edges_dst_tensor_ptr_[i],
                      graph_edges[graph_edges_index++]->ptr(),
                      sizeof(int64_t) * neighbor_len,
                      cudaMemcpyDeviceToDevice,
                      train_stream_);
      if (conf_.return_weight) {
        cudaMemcpyAsync(edges_weight_tensor_ptr_[i],
                        graph_edges[graph_edges_index++]->ptr(),
                        sizeof(float) * neighbor_len,
                        cudaMemcpyDeviceToDevice,
                        train_stream_);
      }
    }  // end for (int i = 0; i < len_samples; i++) {

    VLOG(1) << conf_.gpuid << ": Fill index_tensor";
    index_tensor_ptr_ = feed_vec_[feed_vec_idx++]->mutable_data<int>(
        {total_instance}, this->place_);
    cudaMemcpyAsync(index_tensor_ptr_,
                    inverse_vec_[new_index]->ptr(),
                    sizeof(int) * total_instance,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
    cudaStreamSynchronize(train_stream_);
  }  // end for (int accum = 0; accum < fake_accumulate_num;

  VLOG(1) << conf_.gpuid << ": Finish FillGraphIdAccum";
  return 0;
}

int GraphDataGenerator::FillGraphSlotFeature(
    int total_instance,
    bool gpu_graph_training,
    std::shared_ptr<phi::Allocation> final_sage_nodes) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  int ret = 0;
  uint64_t *ins_cursor, *ins_buf;

  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    int feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
    if (gpu_graph_training) {
      ins_buf =
          reinterpret_cast<uint64_t *>(d_ins_buf_[tensor_pair_idx]->ptr());
      ins_cursor =
          ins_buf + ins_buf_pair_len_[tensor_pair_idx] * 2 - total_instance;
    } else {
      id_tensor_ptr_ = feed_vec_[feed_vec_idx]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      ins_cursor = reinterpret_cast<uint64_t *>(id_tensor_ptr_);
    }
    if (!conf_.sage_mode) {
      if (uint_slot_num_ > 0) {
        ret += FillSlotFeature(ins_cursor, total_instance, tensor_pair_idx);
      }
      if (float_slot_num_ > 0) {
        ret += FillFloatFeature(ins_cursor, total_instance, tensor_pair_idx);
      }
    } else {
      uint64_t *sage_nodes_ptr =
          reinterpret_cast<uint64_t *>(final_sage_nodes->ptr());
      if (uint_slot_num_ > 0) {
        ret += FillSlotFeature(sage_nodes_ptr, total_instance, tensor_pair_idx);
      }
      if (float_slot_num_ > 0) {
        ret +=
            FillFloatFeature(sage_nodes_ptr, total_instance, tensor_pair_idx);
      }
    }
  }

  return ret;
}

int GraphDataGenerator::FillGraphSlotFeatureAccum(bool gpu_graph_training,
                                                  int index) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  int ret = 0;
  // only use for gpu_graph_training is True and sage_mode is True.
  int fake_accumulate_num = 2;
  for (int accum = 0; accum < fake_accumulate_num; accum++) {
    int uniq_instance = uniq_instance_vec_[2 * index + 1 - accum];
    uint64_t *sage_nodes_ptr = reinterpret_cast<uint64_t *>(
        final_sage_nodes_vec_[2 * index + 1 - accum]->ptr());
    if (uint_slot_num_ > 0) {
      ret += FillSlotFeature(sage_nodes_ptr, uniq_instance, 0, accum);
    }
  }
  return ret;
}

int MakeInsPair(const std::shared_ptr<phi::Allocation> &d_walk,        // input
                const std::shared_ptr<phi::Allocation> &d_walk_ntype,  // input
                const GraphDataGeneratorConfig &conf,
                const std::shared_ptr<phi::Allocation> &d_random_row,
                const std::shared_ptr<phi::Allocation> &d_random_row_col_shift,
                BufState *buf_state,
                uint64_t *ins_buf,        // output
                int32_t *pair_label_buf,  // output
                int *d_pair_num,          // output
                int *ins_buf_pair_len_ptr,
                cudaStream_t stream) {
  platform::CUDADeviceGuard guard(conf.gpuid);
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk->ptr());
  uint8_t *walk_ntype = NULL;
  if (conf.need_walk_ntype) {
    walk_ntype = reinterpret_cast<uint8_t *>(d_walk_ntype->ptr());
  }
  uint8_t *excluded_train_pair = NULL;
  if (conf.excluded_train_pair_len > 0) {
    excluded_train_pair =
        reinterpret_cast<uint8_t *>(conf.d_excluded_train_pair->ptr());
  }
  int32_t *pair_label_conf = NULL;
  if (conf.enable_pair_label) {
    pair_label_conf =
        reinterpret_cast<int32_t *>(conf.d_pair_label_conf->ptr());
  }
  int *random_row = reinterpret_cast<int *>(d_random_row->ptr());
  int *random_row_col_shift =
      reinterpret_cast<int *>(d_random_row_col_shift->ptr());
  cudaMemsetAsync(d_pair_num, 0, sizeof(int), stream);
  int len = buf_state->len;
  int &ins_buf_pair_len = *ins_buf_pair_len_ptr;

  // make pair
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream>>>(
      ins_buf + ins_buf_pair_len * 2,
      pair_label_buf + ins_buf_pair_len,
      d_pair_num,
      walk,
      walk_ntype,
      random_row + buf_state->cursor,
      random_row_col_shift + buf_state->cursor,
      buf_state->central_word,
      conf.window_step[buf_state->step],
      len,
      conf.walk_len,
      excluded_train_pair,
      conf.excluded_train_pair_len,
      pair_label_conf,
      conf.node_type_num);
  int h_pair_num;
  cudaMemcpyAsync(
      &h_pair_num, d_pair_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  ins_buf_pair_len += h_pair_num;

  if (conf.debug_mode) {
    uint64_t h_ins_buf[ins_buf_pair_len * 2];  // NOLINT
    cudaMemcpy(h_ins_buf,
               ins_buf,
               2 * ins_buf_pair_len * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    VLOG(2) << "h_pair_num = " << h_pair_num
            << ", ins_buf_pair_len = " << ins_buf_pair_len;
    for (int xx = 0; xx < ins_buf_pair_len; xx++) {
      VLOG(2) << "h_ins_buf: " << h_ins_buf[xx * 2] << ", "
              << h_ins_buf[xx * 2 + 1];
    }
  }
  return ins_buf_pair_len;
}

int FillInsBuf(const std::shared_ptr<phi::Allocation> &d_walk,        // input
               const std::shared_ptr<phi::Allocation> &d_walk_ntype,  // input
               const GraphDataGeneratorConfig &conf,
               const std::shared_ptr<phi::Allocation> &d_random_row,
               const std::shared_ptr<phi::Allocation> &d_random_row_col_shift,
               BufState *buf_state,
               uint64_t *ins_buf,        // output
               int32_t *pair_label_buf,  // output
               int *pair_num_ptr,        // output
               int *ins_buf_pair_len_ptr,
               cudaStream_t stream) {
  int &ins_buf_pair_len = *ins_buf_pair_len_ptr;
  if (ins_buf_pair_len >= conf.batch_size) {
    return conf.batch_size;
  }
  int total_instance = AcquireInstance(buf_state);

  VLOG(2) << "total_ins: " << total_instance;
  buf_state->Debug();

  if (total_instance == 0) {
    return -1;
  }
  return MakeInsPair(d_walk,
                     d_walk_ntype,
                     conf,
                     d_random_row,
                     d_random_row_col_shift,
                     buf_state,
                     ins_buf,
                     pair_label_buf,
                     pair_num_ptr,
                     ins_buf_pair_len_ptr,
                     stream);
}

int GraphDataGenerator::GenerateBatch() {
  int total_instance = conf_.batch_size;
  platform::CUDADeviceGuard guard(conf_.gpuid);
  int res = 0;
  if (!conf_.gpu_graph_training) {
    // infer
    if (!conf_.sage_mode) {
      for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
           ++tensor_pair_idx) {
        int size = infer_node_end_[tensor_pair_idx] -
                   infer_node_start_[tensor_pair_idx];
        if (size < total_instance) {
          total_instance = size;
        }
      }
      total_instance *= 2;
      if (total_instance == 0) {
        return 0;
      }
      FillIdShowClkTensor(total_instance, conf_.gpu_graph_training);
    } else {
      if (sage_batch_count_ == sage_batch_num_) {
        return 0;
      }
      FillGraphIdShowClkTensor(uniq_instance_vec_[sage_batch_count_],
                               total_instance_vec_[sage_batch_count_],
                               sage_batch_count_);
    }
  } else {  // train
    if (!conf_.sage_mode) {
      int fill_zero_num = 10;
      for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
           ++tensor_pair_idx) {
        while (ins_buf_pair_len_[tensor_pair_idx] < conf_.batch_size) {
          int32_t *pair_label_buf = NULL;
          if (conf_.enable_pair_label) {
            pair_label_buf = reinterpret_cast<int32_t *>(
                d_pair_label_buf_[tensor_pair_idx]->ptr());
          }
          res = FillInsBuf(
              d_walk_[tensor_pair_idx],
              d_walk_ntype_[tensor_pair_idx],
              conf_,
              d_random_row_[tensor_pair_idx],
              d_random_row_col_shift_[tensor_pair_idx],
              &buf_state_[tensor_pair_idx],
              reinterpret_cast<uint64_t *>(d_ins_buf_[tensor_pair_idx]->ptr()),
              pair_label_buf,
              reinterpret_cast<int *>(d_pair_num_[tensor_pair_idx]->ptr()),
              &ins_buf_pair_len_[tensor_pair_idx],
              train_stream_);
          if (res == -1) {
            if (ins_buf_pair_len_[tensor_pair_idx] == 0) {
              if (conf_.is_multi_node) {
                pass_end_ = 1;
                if (total_row_[tensor_pair_idx] != 0) {
                  buf_state_[tensor_pair_idx].Reset(
                      total_row_[tensor_pair_idx]);
                  VLOG(1) << "reset buf state to make batch num equal in multi "
                             "node";
                } else {
                  VLOG(1) << "total row in buf state is 0";
                  // Fill 0 ins kernel
                  GraphZeroIdKernel<<<GET_BLOCKS(fill_zero_num),
                                      CUDA_NUM_THREADS,
                                      0,
                                      train_stream_>>>(
                      reinterpret_cast<uint64_t *>(
                          d_ins_buf_[tensor_pair_idx]->ptr()),
                      fill_zero_num);
                  break;
                }
              } else {
                return 0;
              }
            } else {
              break;
            }
          }
        }  // end while (ins_buf_pair_len_ < conf_.batch_size)
      }    // end for (int tensor_pair_idx = 0; tensor_pair_idx <
           // conf_.tensor_pair_num;

      int min_ins_buf_pair_len =
          *std::min_element(ins_buf_pair_len_.begin(), ins_buf_pair_len_.end());
      total_instance = min_ins_buf_pair_len < conf_.batch_size
                           ? min_ins_buf_pair_len
                           : conf_.batch_size;
      if (conf_.is_multi_node && total_row_[0] == 0) {
        total_instance = fill_zero_num;
        ins_buf_pair_len_[0] = fill_zero_num;
        VLOG(1) << "gpu id: " << conf_.gpuid
                << "set total ins num: " << fill_zero_num;
      }

      total_instance *= 2;
      VLOG(2) << "total_instance: " << total_instance
              << ", ins_buf_pair_len = " << min_ins_buf_pair_len;
      FillIdShowClkTensor(total_instance, conf_.gpu_graph_training);
    } else {  // sage
      if (sage_batch_count_ == sage_batch_num_) {
        return 0;
      }
      if (conf_.accumulate_num == 1) {
        FillGraphIdShowClkTensor(uniq_instance_vec_[sage_batch_count_],
                                 total_instance_vec_[sage_batch_count_],
                                 sage_batch_count_);
      } else {
        FillGraphIdShowClkTensorAccum(sage_batch_count_);
      }
    }
  }

  if (conf_.slot_num > 0) {
    if (!conf_.sage_mode) {
      FillGraphSlotFeature(total_instance, conf_.gpu_graph_training);
    } else {
      if (conf_.accumulate_num == 1) {
        FillGraphSlotFeature(uniq_instance_vec_[sage_batch_count_],
                             conf_.gpu_graph_training,
                             final_sage_nodes_vec_[sage_batch_count_]);
      } else {
        FillGraphSlotFeatureAccum(conf_.gpu_graph_training, sage_batch_count_);
      }
    }
  }
  offset_.clear();
  offset_.push_back(0);
  if (!conf_.sage_mode) {
    offset_.push_back(total_instance);
  } else {
    if (conf_.accumulate_num == 1) {
      offset_.push_back(uniq_instance_vec_[sage_batch_count_]);
    } else {
      offset_.push_back(uniq_instance_vec_[sage_batch_count_ * 2 + 1]);
    }
    sage_batch_count_ += 1;
  }
  LegacyLoD lod{offset_};

  if (conf_.accumulate_num >= 2) {
    offset_.clear();
    offset_.push_back(0);
    offset_.push_back(uniq_instance_vec_[sage_batch_count_ * 2]);
  }
  LegacyLoD lod2{offset_};

  if (conf_.accumulate_num == 1) {
    for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
         ++tensor_pair_idx) {
      int feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
      feed_vec_[feed_vec_idx++]->set_lod(lod);
      if (conf_.enable_pair_label) {
        feed_vec_idx++;
      }
      // adapt for float feature
      if (conf_.slot_num > 0) {
        for (int i = 0; i < conf_.slot_num; ++i) {
          if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
            feed_vec_[feed_vec_idx + 2 * i]->set_lod(lod);
          }
        }
      }
    }
  } else {
    feed_vec_[2]->set_lod(lod);   // id0
    feed_vec_[5]->set_lod(lod2);  // id1
    if (conf_.slot_num > 0) {
      int feed_vec_idx = 6;
      for (int i = 0; i < conf_.slot_num / 2; i++) {
        if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
          feed_vec_[feed_vec_idx + 2 * i]->set_lod(lod);
        }
      }
      feed_vec_idx += conf_.slot_num;
      for (int i = 0; i < conf_.slot_num / 2; i++) {
        if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
          feed_vec_[feed_vec_idx + 2 * i]->set_lod(lod2);
        }
      }
    }
  }

  cudaStreamSynchronize(train_stream_);
  if (!conf_.gpu_graph_training) return 1;
  if (!conf_.sage_mode) {
    ins_buf_pair_len_[0] -= total_instance / 2;
  }
  return 1;
}

__global__ void GraphFillSampleKeysKernel(int *prefix_sum,
                                          int *sampleidx2row,
                                          int *tmp_sampleidx2row,
                                          int *actual_sample_size,
                                          int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t offset = prefix_sum[idx] + k;
      tmp_sampleidx2row[offset] = sampleidx2row[idx] + k;
    }
  }
}

__global__ void GraphDoWalkKernel(uint64_t *neighbors,
                                  uint64_t *walk,
                                  uint8_t *walk_ntype,
                                  int *d_prefix_sum,
                                  int *actual_sample_size,
                                  int cur_degree,
                                  int step,
                                  int len,
                                  int *sampleidx2row,
                                  int col_size,
                                  uint8_t edge_dst_id) {
  CUDA_KERNEL_LOOP(i, len) {
    for (int k = 0; k < actual_sample_size[i]; k++) {
      // int idx = sampleidx2row[i];
      size_t row = sampleidx2row[k + d_prefix_sum[i]];
      // size_t row = idx * cur_degree + k;
      size_t col = step;
      size_t offset = (row * col_size + col);
      walk[offset] = neighbors[i * cur_degree + k];
      if (walk_ntype != NULL) {
        walk_ntype[offset] = edge_dst_id;
      }
    }
  }
}

// Fill keys to the first column of walk
__global__ void GraphFillFirstStepKernel(int *prefix_sum,
                                         int *sampleidx2row,
                                         uint64_t *walk,
                                         uint8_t *walk_ntype,
                                         uint64_t *keys,
                                         uint8_t edge_src_id,
                                         uint8_t edge_dst_id,
                                         int len,
                                         int walk_degree,
                                         int col_size,
                                         int *actual_sample_size,
                                         uint64_t *neighbors) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t row = prefix_sum[idx] + k;
      sampleidx2row[row] = row;

      size_t offset = col_size * row;
      walk[offset] = keys[idx];
      walk[offset + 1] = neighbors[idx * walk_degree + k];
      if (walk_ntype != NULL) {
        walk_ntype[offset] = edge_src_id;
        walk_ntype[offset + 1] = edge_dst_id;
      }
    }
  }
}

__global__ void get_each_ins_info(uint8_t *slot_list,
                                  uint32_t *slot_size_list,
                                  uint32_t *slot_size_prefix,
                                  uint32_t *each_ins_slot_num,
                                  uint32_t *each_ins_slot_num_inner_prefix,
                                  size_t key_num,
                                  int slot_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    uint32_t slot_index = slot_size_prefix[i];
    size_t each_ins_slot_index = i * slot_num;
    for (int j = 0; j < slot_size_list[i]; j++) {
      each_ins_slot_num[each_ins_slot_index + slot_list[slot_index + j]] += 1;
    }
    each_ins_slot_num_inner_prefix[each_ins_slot_index] = 0;
    for (int j = 1; j < slot_num; j++) {
      each_ins_slot_num_inner_prefix[each_ins_slot_index + j] =
          each_ins_slot_num[each_ins_slot_index + j - 1] +
          each_ins_slot_num_inner_prefix[each_ins_slot_index + j - 1];
    }
  }
}

__global__ void get_each_ins_float_info(
    uint8_t *slot_list,
    uint32_t *slot_size_list,
    uint32_t *slot_size_prefix,
    uint32_t *each_ins_slot_num,
    uint32_t *each_ins_slot_num_inner_prefix,
    size_t key_num,
    int slot_num) {  // offset of float slot
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    uint32_t slot_index = slot_size_prefix[i];
    size_t each_ins_slot_index = i * slot_num;
    for (int j = 0; j < slot_size_list[i]; j++) {
      each_ins_slot_num[each_ins_slot_index + slot_list[slot_index + j]] += 1;
    }
    each_ins_slot_num_inner_prefix[each_ins_slot_index] = 0;
    for (int j = 1; j < slot_num; j++) {
      each_ins_slot_num_inner_prefix[each_ins_slot_index + j] =
          each_ins_slot_num[each_ins_slot_index + j - 1] +
          each_ins_slot_num_inner_prefix[each_ins_slot_index + j - 1];
    }
  }
}

__global__ void fill_slot_num(uint32_t *d_each_ins_slot_num_ptr,
                              uint64_t **d_ins_slot_num_vector_ptr,
                              size_t key_num,
                              int slot_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    size_t d_each_index = i * slot_num;
    for (int j = 0; j < slot_num; j++) {
      d_ins_slot_num_vector_ptr[j][i] =
          d_each_ins_slot_num_ptr[d_each_index + j];
    }
  }
}
// 可以搞成模板
__global__ void fill_slot_tensor(uint64_t *feature_list,
                                 uint32_t *feature_size_prefixsum,
                                 uint32_t *each_ins_slot_num_inner_prefix,
                                 uint64_t *ins_slot_num,
                                 int64_t *slot_lod_tensor,
                                 int64_t *slot_tensor,
                                 int slot,
                                 int slot_num,
                                 size_t node_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < node_num) {
    size_t dst_index = slot_lod_tensor[i];
    size_t src_index = feature_size_prefixsum[i] +
                       each_ins_slot_num_inner_prefix[slot_num * i + slot];
    for (uint64_t j = 0; j < ins_slot_num[i]; j++) {
      slot_tensor[dst_index + j] = feature_list[src_index + j];
    }
  }
}

__global__ void fill_float_tensor(float *feature_list,
                                  uint32_t *feature_size_prefixsum,
                                  uint32_t *each_ins_slot_num_inner_prefix,
                                  uint64_t *ins_slot_num,
                                  int64_t *slot_lod_tensor,
                                  float *slot_tensor,
                                  int slot,
                                  int slot_num,
                                  size_t node_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < node_num) {
    size_t dst_index = slot_lod_tensor[i];
    size_t src_index = feature_size_prefixsum[i] +
                       each_ins_slot_num_inner_prefix[slot_num * i + slot];
    for (uint64_t j = 0; j < ins_slot_num[i]; j++) {
      slot_tensor[dst_index + j] = feature_list[src_index + j];
    }
  }
}

__global__ void GetUniqueFeaNum(uint64_t *d_in,
                                uint64_t *unique_num,
                                size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t local_num;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len - 1) {
    if (d_in[i] != d_in[i + 1]) {
      atomicAdd(&local_num, 1);
    }
  }
  if (i == len - 1) {
    atomicAdd(&local_num, 1);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(unique_num, local_num);
  }
}

__global__ void UniqueFeature(uint64_t *d_in,
                              uint64_t *d_out,
                              uint64_t *unique_num,
                              size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t local_key[CUDA_NUM_THREADS];
  __shared__ uint64_t local_num;
  __shared__ uint64_t global_num;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len - 1) {
    if (d_in[i] != d_in[i + 1]) {
      size_t dst = atomicAdd(&local_num, 1);
      local_key[dst] = d_in[i];
    }
  }
  if (i == len - 1) {
    size_t dst = atomicAdd(&local_num, 1);
    local_key[dst] = d_in[i];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(unique_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    d_out[global_num + threadIdx.x] = local_key[threadIdx.x];
  }
}
// Fill sample_res to the stepth column of walk
void FillOneStep(
    uint64_t *d_start_ids,
    int etype_id,
    uint64_t *walk,
    uint8_t *walk_ntype,
    int len,
    NeighborSampleResult *sample_res,
    int cur_degree,
    int step,
    const GraphDataGeneratorConfig &conf,
    std::vector<std::shared_ptr<phi::Allocation>> *d_sampleidx2rows,
    int *cur_sampleidx2row,
    const phi::Place &place,
    cudaStream_t stream) {
  platform::CUDADeviceGuard guard(conf.gpuid);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  uint64_t node_id = gpu_graph_ptr->edge_to_node_map_[etype_id];
  uint8_t edge_src_id = node_id >> 32;
  uint8_t edge_dst_id = node_id;
  size_t temp_storage_bytes = 0;
  int *d_actual_sample_size = sample_res->actual_sample_size;
  uint64_t *d_neighbors = sample_res->val;
  auto d_prefix_sum_ptr =
      memory::AllocShared(place,
                          (conf.once_max_sample_keynum + 1) * sizeof(int),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *d_prefix_sum = reinterpret_cast<int *>(d_prefix_sum_ptr->ptr());
  cudaMemsetAsync(
      d_prefix_sum, 0, (conf.once_max_sample_keynum + 1) * sizeof(int), stream);
  int *d_sampleidx2row =
      reinterpret_cast<int *>((*d_sampleidx2rows)[*cur_sampleidx2row]->ptr());
  int *d_tmp_sampleidx2row = reinterpret_cast<int *>(
      (*d_sampleidx2rows)[1 - *cur_sampleidx2row]->ptr());

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           stream));
  auto d_temp_storage =
      memory::Alloc(place,
                    temp_storage_bytes,
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           stream));

  cudaStreamSynchronize(stream);

  if (step == 1) {
    GraphFillFirstStepKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream>>>(
        d_prefix_sum,
        d_tmp_sampleidx2row,
        walk,
        walk_ntype,
        d_start_ids,
        edge_src_id,
        edge_dst_id,
        len,
        conf.walk_degree,
        conf.walk_len,
        d_actual_sample_size,
        d_neighbors);

  } else {
    GraphFillSampleKeysKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream>>>(
        d_prefix_sum,
        d_sampleidx2row,
        d_tmp_sampleidx2row,
        d_actual_sample_size,
        len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream>>>(
        d_neighbors,
        walk,
        walk_ntype,
        d_prefix_sum,
        d_actual_sample_size,
        cur_degree,
        step,
        len,
        d_tmp_sampleidx2row,
        conf.walk_len,
        edge_dst_id);
  }
  if (conf.debug_mode) {
    int *h_prefix_sum = new int[len + 1];
    int *h_actual_size = new int[len];
    int *h_offset2idx = new int[conf.once_max_sample_keynum];
    cudaMemcpy(h_offset2idx,
               d_tmp_sampleidx2row,
               conf.once_max_sample_keynum * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_prefix_sum,
               d_prefix_sum,
               (len + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < conf.once_max_sample_keynum; xx++) {
      VLOG(2) << "h_offset2idx[" << xx << "]: " << h_offset2idx[xx];
    }
    for (int xx = 0; xx < len + 1; xx++) {
      VLOG(2) << "h_prefix_sum[" << xx << "]: " << h_prefix_sum[xx];
    }
    delete[] h_prefix_sum;
    delete[] h_actual_size;
    delete[] h_offset2idx;
  }
  cudaStreamSynchronize(stream);
  *cur_sampleidx2row = 1 - *cur_sampleidx2row;
}

int GraphDataGenerator::FillSlotFeature(uint64_t *d_walk,
                                        size_t key_num,
                                        int tensor_pair_idx,
                                        int accum) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  std::shared_ptr<phi::Allocation> d_feature_list;
  std::shared_ptr<phi::Allocation> d_slot_list;

  size_t temp_bytes = (key_num + 1) * sizeof(uint32_t);
  if (d_feature_size_list_buf_ == NULL ||
      d_feature_size_list_buf_->size() < temp_bytes) {
    d_feature_size_list_buf_ = memory::AllocShared(this->place_, temp_bytes);
  }
  if (d_feature_size_prefixsum_buf_ == NULL ||
      d_feature_size_prefixsum_buf_->size() < temp_bytes) {
    d_feature_size_prefixsum_buf_ =
        memory::AllocShared(this->place_, temp_bytes);
  }
  int fea_num =
      gpu_graph_ptr->get_feature_info_of_nodes(conf_.gpuid,
                                               d_walk,
                                               key_num,
                                               d_feature_size_list_buf_,
                                               d_feature_size_prefixsum_buf_,
                                               d_feature_list,
                                               d_slot_list,
                                               conf_.sage_mode);
  // num of slot feature
  int slot_num = conf_.slot_num - float_slot_num_;
  int conf_slot_num = slot_num;
  int fake_accumulate_num = 1;
  if (conf_.accumulate_num >= 2) {
    slot_num /= 2;
    conf_slot_num /= 2;
    fake_accumulate_num = 2;
  }
  int64_t *slot_tensor_ptr_[slot_num];
  int64_t *slot_lod_tensor_ptr_[slot_num];

  int feed_vec_idx = 0;
  if (fake_accumulate_num == 1) {
    feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
    ++feed_vec_idx;
    if (conf_.enable_pair_label) {
      ++feed_vec_idx;
    }
  } else {
    // set tensor_pair_idx = 0
    feed_vec_idx = fake_accumulate_num * 3 + accum * conf_.slot_num;
  }

  if (fea_num == 0) {
    int64_t default_lod = 1;
    int ii = 0;

    for (int i = 0; i < conf_slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
        slot_lod_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i + 1]->mutable_data<int64_t>(
                {(long)key_num + 1}, this->place_);  // NOLINT
        slot_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i]->mutable_data<int64_t>(
                {1, 1}, this->place_);
        CUDA_CHECK(cudaMemsetAsync(
            slot_tensor_ptr_[ii], 0, sizeof(int64_t), train_stream_));
        CUDA_CHECK(cudaMemsetAsync(slot_lod_tensor_ptr_[ii],
                                   0,
                                   sizeof(int64_t) * key_num,
                                   train_stream_));
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<char *>(slot_lod_tensor_ptr_[ii] + key_num),
            &default_lod,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            train_stream_));
        ii++;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    return 0;
  }

  uint64_t *d_feature_list_ptr =
      reinterpret_cast<uint64_t *>(d_feature_list->ptr());
  uint8_t *d_slot_list_ptr = reinterpret_cast<uint8_t *>(d_slot_list->ptr());
  uint32_t *d_feature_size_list_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_list_buf_->ptr());
  uint32_t *d_feature_size_prefixsum_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_prefixsum_buf_->ptr());
  VLOG(2) << "end trans feature list and slot list";

  CUDA_CHECK(cudaStreamSynchronize(train_stream_));

  std::shared_ptr<phi::Allocation> d_each_ins_slot_num_inner_prefix =
      memory::AllocShared(place_, (conf_slot_num * key_num) * sizeof(uint32_t));
  std::shared_ptr<phi::Allocation> d_each_ins_slot_num =
      memory::AllocShared(place_, (conf_slot_num * key_num) * sizeof(uint32_t));
  uint32_t *d_each_ins_slot_num_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num->ptr());
  uint32_t *d_each_ins_slot_num_inner_prefix_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num_inner_prefix->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_each_ins_slot_num_ptr,
                             0,
                             conf_slot_num * key_num * sizeof(uint32_t),
                             train_stream_));

  dim3 grid((key_num - 1) / 256 + 1);
  dim3 block(1, 256);
  get_each_ins_info<<<grid, block, 0, train_stream_>>>(
      d_slot_list_ptr,
      d_feature_size_list_ptr,
      d_feature_size_prefixsum_ptr,
      d_each_ins_slot_num_ptr,
      d_each_ins_slot_num_inner_prefix_ptr,
      key_num,
      slot_num);

  std::vector<std::shared_ptr<phi::Allocation>> ins_slot_num(slot_num, nullptr);
  std::vector<uint64_t *> ins_slot_num_vector(slot_num, NULL);
  std::shared_ptr<phi::Allocation> d_ins_slot_num_vector =
      memory::AllocShared(place_, (slot_num) * sizeof(uint64_t *));
  uint64_t **d_ins_slot_num_vector_ptr =
      reinterpret_cast<uint64_t **>(d_ins_slot_num_vector->ptr());

  int ii = 0;
  for (int i = 0; i < conf_slot_num; i++) {
    if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
      ins_slot_num[ii] =
          memory::AllocShared(place_, key_num * sizeof(uint64_t));
      ins_slot_num_vector[ii] =
          reinterpret_cast<uint64_t *>(ins_slot_num[ii]->ptr());
      ii++;
    }
  }
  if (slot_num > 0) {
    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char *>(d_ins_slot_num_vector_ptr),
                        ins_slot_num_vector.data(),
                        sizeof(uint64_t *) * slot_num,
                        cudaMemcpyHostToDevice,
                        train_stream_));
    fill_slot_num<<<grid, block, 0, train_stream_>>>(
        d_each_ins_slot_num_ptr, d_ins_slot_num_vector_ptr, key_num, slot_num);
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));

    ii = 0;
    for (int i = 0; i < conf_slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
        slot_lod_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i + 1]->mutable_data<int64_t>(
                {(long)key_num + 1}, this->place_);  // NOLINT
        ii++;
      }
    }

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                             temp_storage_bytes,
                                             ins_slot_num_vector[0],
                                             slot_lod_tensor_ptr_[0] + 1,
                                             key_num,
                                             train_stream_));
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));

    auto d_temp_storage = memory::Alloc(
        this->place_,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(train_stream_)));

    ii = 0;
    std::vector<int64_t> each_slot_fea_num(slot_num, 0);
    for (int i = 0; i < conf_slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
        CUDA_CHECK(cudaMemsetAsync(
            slot_lod_tensor_ptr_[ii], 0, sizeof(uint64_t), train_stream_));
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                                 temp_storage_bytes,
                                                 ins_slot_num_vector[ii],
                                                 slot_lod_tensor_ptr_[ii] + 1,
                                                 key_num,
                                                 train_stream_));
        CUDA_CHECK(cudaMemcpyAsync(&each_slot_fea_num[ii],
                                   slot_lod_tensor_ptr_[ii] + key_num,
                                   sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost,
                                   train_stream_));
        ii++;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    ii = 0;
    for (int i = 0; i < conf_slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
        slot_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i]->mutable_data<int64_t>(
                {each_slot_fea_num[ii], 1}, this->place_);
        ii++;
      }
    }
    ii = 0;
    int64_t default_lod = 1;
    for (int i = 0; i < conf_slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'u') {
        fill_slot_tensor<<<grid, block, 0, train_stream_>>>(
            d_feature_list_ptr,
            d_feature_size_prefixsum_ptr,
            d_each_ins_slot_num_inner_prefix_ptr,
            ins_slot_num_vector[ii],
            slot_lod_tensor_ptr_[ii],
            slot_tensor_ptr_[ii],
            ii,
            slot_num,
            key_num);

        // trick for empty tensor
        if (each_slot_fea_num[ii] == 0) {
          slot_tensor_ptr_[ii] =
              feed_vec_[feed_vec_idx + 2 * i]->mutable_data<int64_t>(
                  {1, 1}, this->place_);
          CUDA_CHECK(cudaMemsetAsync(
              slot_tensor_ptr_[ii], 0, sizeof(uint64_t), train_stream_));
          CUDA_CHECK(cudaMemcpyAsync(
              reinterpret_cast<char *>(slot_lod_tensor_ptr_[ii] + key_num),
              &default_lod,
              sizeof(int64_t),
              cudaMemcpyHostToDevice,
              train_stream_));
        }
        ii++;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
  }
  if (conf_.debug_mode) {
    std::vector<uint32_t> h_feature_size_list(key_num, 0);
    std::vector<uint32_t> h_feature_size_list_prefixsum(key_num, 0);
    std::vector<uint64_t> node_list(key_num, 0);
    std::vector<uint64_t> h_feature_list(fea_num, 0);
    std::vector<uint8_t> h_slot_list(fea_num, 0);

    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_size_list.data()),
                        d_feature_size_list_ptr,
                        sizeof(uint32_t) * key_num,
                        cudaMemcpyDeviceToHost,
                        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char *>(h_feature_size_list_prefixsum.data()),
        d_feature_size_prefixsum_ptr,
        sizeof(uint32_t) * key_num,
        cudaMemcpyDeviceToHost,
        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(node_list.data()),
                               d_walk,
                               sizeof(uint64_t) * key_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_list.data()),
                               d_feature_list_ptr,
                               sizeof(uint64_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_list.data()),
                               d_slot_list_ptr,
                               sizeof(uint8_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    for (size_t i = 0; i < key_num; i++) {
      std::stringstream ss;
      ss << "node_id: " << node_list[i]
         << " fea_num: " << h_feature_size_list[i] << " offset "
         << h_feature_size_list_prefixsum[i] << " slot: ";
      for (uint32_t j = 0; j < h_feature_size_list[i]; j++) {
        ss << int(h_slot_list[h_feature_size_list_prefixsum[i] + j]) << " : "
           << h_feature_list[h_feature_size_list_prefixsum[i] + j] << "  ";
      }
      VLOG(0) << ss.str();
    }
    VLOG(0) << "all fea_num is " << fea_num << " calc fea_num is "
            << h_feature_size_list[key_num - 1] +
                   h_feature_size_list_prefixsum[key_num - 1];

    for (int i = 0; i < slot_num; ++i) {
      std::vector<int64_t> h_slot_lod_tensor(key_num + 1, 0);
      CUDA_CHECK(
          cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_lod_tensor.data()),
                          slot_lod_tensor_ptr_[i],
                          sizeof(int64_t) * (key_num + 1),
                          cudaMemcpyDeviceToHost,
                          train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));
      std::stringstream ss_lod;
      std::stringstream ss_tensor;
      ss_lod << " slot " << i << " lod is [";
      for (size_t j = 0; j < key_num + 1; j++) {
        ss_lod << h_slot_lod_tensor[j] << ",";
      }
      ss_lod << "]";
      std::vector<int64_t> h_slot_tensor(h_slot_lod_tensor[key_num], 0);
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_tensor.data()),
                                 slot_tensor_ptr_[i],
                                 sizeof(int64_t) * h_slot_lod_tensor[key_num],
                                 cudaMemcpyDeviceToHost,
                                 train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));

      ss_tensor << " tensor is [ ";
      for (size_t j = 0; j < h_slot_lod_tensor[key_num]; j++) {
        ss_tensor << h_slot_tensor[j] << ",";
      }
      ss_tensor << "]";
      VLOG(0) << ss_lod.str() << "  " << ss_tensor.str();
    }
  }

  return 0;
}

int GraphDataGenerator::FillFloatFeature(uint64_t *d_walk,
                                         size_t key_num,
                                         int tensor_pair_idx) {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  std::shared_ptr<phi::Allocation> d_feature_list;
  std::shared_ptr<phi::Allocation> d_slot_list;

  size_t temp_storage_bytes = (key_num + 1) * sizeof(uint32_t);
  if (d_feature_size_list_buf_ == NULL ||
      d_feature_size_list_buf_->size() < temp_storage_bytes) {
    d_feature_size_list_buf_ =
        memory::AllocShared(this->place_, temp_storage_bytes);
  }
  if (d_feature_size_prefixsum_buf_ == NULL ||
      d_feature_size_prefixsum_buf_->size() < temp_storage_bytes) {
    d_feature_size_prefixsum_buf_ =
        memory::AllocShared(this->place_, temp_storage_bytes);
  }

  uint32_t *d_feature_size_list_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_list_buf_->ptr());
  uint32_t *d_feature_size_prefixsum_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_prefixsum_buf_->ptr());

  int fea_num = gpu_graph_ptr->get_float_feature_info_of_nodes(
      conf_.gpuid,
      d_walk,
      key_num,
      d_feature_size_list_buf_,
      d_feature_size_prefixsum_buf_,
      d_feature_list,
      d_slot_list,
      conf_.sage_mode);
  int feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
  ++feed_vec_idx;
  if (conf_.enable_pair_label) {
    ++feed_vec_idx;
  }

  float *slot_tensor_ptr_[float_slot_num_];
  int64_t *slot_lod_tensor_ptr_[float_slot_num_];
  if (fea_num == 0) {
    int64_t default_lod = 1;
    int ii = 0;
    for (int i = 0; i < conf_.slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'f') {
        slot_lod_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i + 1]->mutable_data<int64_t>(
                {(long)key_num + 1}, this->place_);  // NOLINT
        slot_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i]->mutable_data<float>({1, 1},
                                                                 this->place_);
        CUDA_CHECK(cudaMemsetAsync(
            slot_tensor_ptr_[ii], 0, sizeof(float), train_stream_));
        CUDA_CHECK(cudaMemsetAsync(slot_lod_tensor_ptr_[ii],
                                   0,
                                   sizeof(int64_t) * key_num,
                                   train_stream_));
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<char *>(slot_lod_tensor_ptr_[ii] + key_num),
            &default_lod,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            train_stream_));
        ii++;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    return 0;
  }

  float *d_feature_list_ptr = reinterpret_cast<float *>(d_feature_list->ptr());
  uint8_t *d_slot_list_ptr = reinterpret_cast<uint8_t *>(d_slot_list->ptr());

  std::shared_ptr<phi::Allocation> d_each_ins_slot_num_inner_prefix =
      memory::AllocShared(place_,
                          (float_slot_num_ * key_num) * sizeof(uint32_t));
  std::shared_ptr<phi::Allocation> d_each_ins_slot_num = memory::AllocShared(
      place_, (float_slot_num_ * key_num) * sizeof(uint32_t));
  uint32_t *d_each_ins_slot_num_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num->ptr());
  uint32_t *d_each_ins_slot_num_inner_prefix_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num_inner_prefix->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_each_ins_slot_num_ptr,
                             0,
                             float_slot_num_ * key_num * sizeof(uint32_t),
                             train_stream_));

  dim3 grid((key_num - 1) / 256 + 1);
  dim3 block(1, 256);
  get_each_ins_float_info<<<grid, block, 0, train_stream_>>>(
      d_slot_list_ptr,
      d_feature_size_list_ptr,
      d_feature_size_prefixsum_ptr,
      d_each_ins_slot_num_ptr,
      d_each_ins_slot_num_inner_prefix_ptr,
      key_num,
      float_slot_num_);

  std::vector<std::shared_ptr<phi::Allocation>> ins_slot_num(float_slot_num_,
                                                             nullptr);
  std::vector<uint64_t *> ins_slot_num_vector(float_slot_num_, NULL);
  std::shared_ptr<phi::Allocation> d_ins_slot_num_vector =
      memory::AllocShared(place_, (float_slot_num_) * sizeof(uint64_t *));
  uint64_t **d_ins_slot_num_vector_ptr =
      reinterpret_cast<uint64_t **>(d_ins_slot_num_vector->ptr());

  int ii = 0;
  for (int i = 0; i < conf_.slot_num; i++) {
    if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'f') {
      ins_slot_num[ii] =
          memory::AllocShared(place_, key_num * sizeof(uint64_t));
      ins_slot_num_vector[ii] =
          reinterpret_cast<uint64_t *>(ins_slot_num[ii]->ptr());
      ii++;
    }
  }

  if (float_slot_num_ > 0) {
    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char *>(d_ins_slot_num_vector_ptr),
                        ins_slot_num_vector.data(),
                        sizeof(uint64_t *) * float_slot_num_,
                        cudaMemcpyHostToDevice,
                        train_stream_));

    fill_slot_num<<<grid, block, 0, train_stream_>>>(d_each_ins_slot_num_ptr,
                                                     d_ins_slot_num_vector_ptr,
                                                     key_num,
                                                     float_slot_num_);
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));

    ii = 0;
    for (int i = 0; i < conf_.slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'f') {
        slot_lod_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i + 1]->mutable_data<int64_t>(
                {(long)key_num + 1}, this->place_);  // NOLINT
        ii++;
      }
    }
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                             temp_storage_bytes,
                                             ins_slot_num_vector[0],
                                             slot_lod_tensor_ptr_[0] + 1,
                                             key_num,
                                             train_stream_));
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    auto d_temp_storage = memory::Alloc(
        this->place_,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(train_stream_)));
    std::vector<int64_t> each_slot_fea_num(float_slot_num_, 0);

    ii = 0;
    for (int i = 0; i < conf_.slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'f') {
        CUDA_CHECK(cudaMemsetAsync(
            slot_lod_tensor_ptr_[ii], 0, sizeof(uint64_t), train_stream_));
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                                 temp_storage_bytes,
                                                 ins_slot_num_vector[ii],
                                                 slot_lod_tensor_ptr_[ii] + 1,
                                                 key_num,
                                                 train_stream_));
        CUDA_CHECK(cudaMemcpyAsync(&each_slot_fea_num[ii],
                                   slot_lod_tensor_ptr_[ii] + key_num,
                                   sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost,
                                   train_stream_));
        ii++;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    ii = 0;
    for (int i = 0; i < conf_.slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'f') {
        slot_tensor_ptr_[ii] =
            feed_vec_[feed_vec_idx + 2 * i]->mutable_data<float>(
                {each_slot_fea_num[ii], 1}, this->place_);
        ii++;
      }
    }
    ii = 0;
    int64_t default_lod = 1;
    for (int i = 0; i < conf_.slot_num; ++i) {
      if ((*feed_info_)[feed_vec_idx + 2 * i].type[0] == 'f') {
        fill_float_tensor<<<grid, block, 0, train_stream_>>>(
            d_feature_list_ptr,
            d_feature_size_prefixsum_ptr,
            d_each_ins_slot_num_inner_prefix_ptr,
            ins_slot_num_vector[ii],
            slot_lod_tensor_ptr_[ii],
            slot_tensor_ptr_[ii],
            ii,
            float_slot_num_,
            key_num);
        // trick for empty tensor
        if (each_slot_fea_num[ii] == 0) {
          slot_tensor_ptr_[ii] =
              feed_vec_[feed_vec_idx + 2 * i]->mutable_data<float>(
                  {1, 1}, this->place_);
          CUDA_CHECK(cudaMemsetAsync(
              slot_tensor_ptr_[ii], 0, sizeof(float), train_stream_));
          CUDA_CHECK(cudaMemcpyAsync(
              reinterpret_cast<char *>(slot_lod_tensor_ptr_[ii] + key_num),
              &default_lod,
              sizeof(int64_t),
              cudaMemcpyHostToDevice,
              train_stream_));
        }
        ii++;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
  }
  if (conf_.debug_mode) {
    std::vector<uint32_t> h_feature_size_list(key_num, 0);
    std::vector<uint32_t> h_feature_size_list_prefixsum(key_num, 0);
    std::vector<uint64_t> node_list(key_num, 0);
    std::vector<float> h_feature_list(fea_num, 0);
    std::vector<uint8_t> h_slot_list(fea_num, 0);

    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_size_list.data()),
                        d_feature_size_list_ptr,
                        sizeof(uint32_t) * key_num,
                        cudaMemcpyDeviceToHost,
                        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char *>(h_feature_size_list_prefixsum.data()),
        d_feature_size_prefixsum_ptr,
        sizeof(uint32_t) * key_num,
        cudaMemcpyDeviceToHost,
        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(node_list.data()),
                               d_walk,
                               sizeof(uint64_t) * key_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_list.data()),
                               d_feature_list_ptr,
                               sizeof(float) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_list.data()),
                               d_slot_list_ptr,
                               sizeof(uint8_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    for (size_t i = 0; i < key_num; i++) {
      std::stringstream ss;
      ss << "node_id: " << node_list[i]
         << " fea_num: " << h_feature_size_list[i] << " offset "
         << h_feature_size_list_prefixsum[i] << " slot: ";
      for (uint32_t j = 0; j < h_feature_size_list[i]; j++) {
        ss << int(h_slot_list[h_feature_size_list_prefixsum[i] + j]) << " : "
           << h_feature_list[h_feature_size_list_prefixsum[i] + j] << "  ";
      }
      VLOG(0) << ss.str();
    }
    VLOG(0) << "all float fea_num is " << fea_num << " calc float fea_num is "
            << h_feature_size_list[key_num - 1] +
                   h_feature_size_list_prefixsum[key_num - 1];
    for (int i = 0; i < float_slot_num_; ++i) {
      std::vector<int64_t> h_slot_lod_tensor(key_num + 1, 0);
      CUDA_CHECK(
          cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_lod_tensor.data()),
                          slot_lod_tensor_ptr_[i],
                          sizeof(int64_t) * (key_num + 1),
                          cudaMemcpyDeviceToHost,
                          train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));
      std::stringstream ss_lod;
      std::stringstream ss_tensor;
      ss_lod << " slot " << i << " lod is [";
      for (size_t j = 0; j < key_num + 1; j++) {
        ss_lod << h_slot_lod_tensor[j] << ",";
      }
      ss_lod << "]";
      std::vector<float> h_slot_tensor(h_slot_lod_tensor[key_num], 0);
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_tensor.data()),
                                 slot_tensor_ptr_[i],
                                 sizeof(float) * h_slot_lod_tensor[key_num],
                                 cudaMemcpyDeviceToHost,
                                 train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));

      ss_tensor << " tensor is [ ";
      for (size_t j = 0; j < h_slot_lod_tensor[key_num]; j++) {
        ss_tensor << h_slot_tensor[j] << ",";
      }
      ss_tensor << "]";
      VLOG(0) << ss_lod.str() << "  " << ss_tensor.str();
    }
  }
  return 0;
}

uint64_t CopyUniqueNodes(
    int gpu_id,
    std::shared_ptr<HashTable<uint64_t, uint32_t>> table,
    uint64_t copy_unique_len,
    const phi::Place &place,
    const std::shared_ptr<phi::Allocation> &d_uniq_node_num_ptr,
    std::vector<uint64_t> *host_vec_ptr,    // output
    std::vector<uint32_t> *host_ranks_ptr,  // output
    cudaStream_t stream) {
  platform::CUDADeviceGuard guard(gpu_id);
  if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
    uint64_t h_uniq_node_num = 0;
    uint64_t *d_uniq_node_num =
        reinterpret_cast<uint64_t *>(d_uniq_node_num_ptr->ptr());
    cudaMemcpyAsync(&h_uniq_node_num,
                    d_uniq_node_num,
                    sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    // prepare temp hbm resource
    auto d_uniq_node = memory::AllocShared(
        place,
        h_uniq_node_num * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    uint64_t *d_uniq_node_ptr =
        reinterpret_cast<uint64_t *>(d_uniq_node->ptr());

    auto d_node_ranks = memory::AllocShared(
        place,
        h_uniq_node_num * sizeof(uint32_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    uint32_t *d_node_ranks_ptr =
        reinterpret_cast<uint32_t *>(d_node_ranks->ptr());

    auto d_node_cursor = memory::AllocShared(
        place,
        sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    uint64_t *d_node_cursor_ptr =
        reinterpret_cast<uint64_t *>(d_node_cursor->ptr());
    cudaMemsetAsync(d_node_cursor_ptr, 0, sizeof(uint64_t), stream);

    // get keys and ranks
    if (FLAGS_graph_edges_split_mode == "fennel" ||
        FLAGS_query_dest_rank_by_multi_node) {
      table->get_key_values(
          d_uniq_node_ptr, d_node_ranks_ptr, d_node_cursor_ptr, stream);
    } else {
      table->get_keys(d_uniq_node_ptr, d_node_cursor_ptr, stream);
    }
    cudaStreamSynchronize(stream);

    // copy keys to host_vec for pass pull
    host_vec_ptr->resize(h_uniq_node_num + copy_unique_len);
    cudaMemcpyAsync(host_vec_ptr->data() + copy_unique_len,
                    d_uniq_node_ptr,
                    sizeof(uint64_t) * h_uniq_node_num,
                    cudaMemcpyDeviceToHost,
                    stream);
    host_ranks_ptr->resize(h_uniq_node_num + copy_unique_len);
    cudaMemcpyAsync(host_ranks_ptr->data() + copy_unique_len,
                    d_node_ranks_ptr,
                    sizeof(uint32_t) * h_uniq_node_num,
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    return h_uniq_node_num;
  }
  return 0;
}

// 对于deepwalk模式，尝试插入table，0表示插入成功，1表示插入失败；
// 对于sage模式，尝试插入table，table数量不够则清空table重新插入，返回值无影响。
int InsertTable(int gpu_id,
                const uint64_t *d_keys,   // Input
                const uint32_t *d_ranks,  // Input
                uint64_t len,             // Input
                std::shared_ptr<phi::Allocation> *d_uniq_node_num,
                const GraphDataGeneratorConfig &conf,
                uint64_t *copy_unique_len_ptr,
                const phi::Place &place,
                std::vector<uint64_t> *host_vec_ptr,                   // Output
                std::vector<uint32_t> *host_ranks_ptr,                 // output
                std::shared_ptr<HashTable<uint64_t, uint32_t>> table,  // Output
                cudaStream_t stream) {
  if (FLAGS_gpugraph_storage_mode == GpuGraphStorageMode::WHOLE_HBM) {
    return 0;
  }
  platform::CUDADeviceGuard guard(conf.gpuid);
  // Used under NOT WHOLE_HBM.
  uint64_t h_uniq_node_num = 0;
  uint64_t *d_uniq_node_num_ptr =
      reinterpret_cast<uint64_t *>((*d_uniq_node_num)->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&h_uniq_node_num,
                                             d_uniq_node_num_ptr,
                                             sizeof(uint64_t),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  if (conf.gpu_graph_training) {
    VLOG(2) << "table capacity: " << conf.train_table_cap << ", "
            << h_uniq_node_num << " used";
    if (h_uniq_node_num + len >= conf.train_table_cap) {
      if (!conf.sage_mode) {
        return 1;
      } else {
        // Copy unique nodes first.
        uint64_t copy_len = CopyUniqueNodes(gpu_id,
                                            table,
                                            *copy_unique_len_ptr,
                                            place,
                                            *d_uniq_node_num,
                                            host_vec_ptr,
                                            host_ranks_ptr,
                                            stream);
        *copy_unique_len_ptr += copy_len;
        table->clear(stream);
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemsetAsync(d_uniq_node_num_ptr, 0, sizeof(uint64_t), stream));
      }
    }
  } else {
    // used only for sage_mode.
    if (h_uniq_node_num + len >= conf.infer_table_cap) {
      uint64_t copy_len = CopyUniqueNodes(gpu_id,
                                          table,
                                          *copy_unique_len_ptr,
                                          place,
                                          *d_uniq_node_num,
                                          host_vec_ptr,
                                          host_ranks_ptr,
                                          stream);
      *copy_unique_len_ptr += copy_len;
      table->clear(stream);
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(d_uniq_node_num_ptr, 0, sizeof(uint64_t), stream));
    }
  }

  if (d_ranks) {
    table->insert(d_keys, d_ranks, len, d_uniq_node_num_ptr, stream);
  } else {
    table->insert(d_keys, len, d_uniq_node_num_ptr, 0 /*useless*/, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

std::vector<std::shared_ptr<phi::Allocation>> SampleNeighbors(
    const uint64_t *d_uniq_nodes,
    int len,
    int sample_size,
    const GraphDataGeneratorConfig &conf,
    std::vector<int> *edges_split_num_ptr,
    int64_t *neighbor_len,
    std::vector<std::shared_ptr<phi::Allocation>> *edge_type_graph_ptr,
    std::shared_ptr<phi::Allocation> *d_uniq_node_num,
    uint64_t *copy_unique_len_ptr,
    std::vector<uint64_t> *host_vec_ptr,                             // Output
    std::vector<uint32_t> *host_ranks_ptr,                           // output
    std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank_table,  // Output
    const phi::Place &place,
    cudaStream_t stream) {
  platform::CUDADeviceGuard guard(conf.gpuid);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();

  // Obtain the dest machine for sample node through cross machine queries
  // Actually, its just filling in d_ranks_ptr.
  auto d_ranks =
      memory::Alloc(place,
                    len * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *d_ranks_ptr = nullptr;
  if (FLAGS_graph_edges_split_mode == "fennel" ||
      FLAGS_query_dest_rank_by_multi_node) {
    d_ranks_ptr = reinterpret_cast<uint32_t *>(d_ranks->ptr());
    gpu_graph_ptr->seek_keys_rank(conf.gpuid, d_uniq_nodes, len, d_ranks_ptr);
  }

  if (InsertTable(conf.gpuid,
                  d_uniq_nodes,
                  d_ranks_ptr,
                  len,
                  d_uniq_node_num,
                  conf,
                  copy_unique_len_ptr,
                  place,
                  host_vec_ptr,
                  host_ranks_ptr,
                  keys2rank_table,
                  stream) != 0) {
    VLOG(0) << "gpu:" << conf.gpuid
            << " in SampleNeighbor, insert key stage, table is full";
    assert(false);
  }

  auto sample_res =
      gpu_graph_ptr->graph_neighbor_sample_sage(conf.gpuid,
                                                conf.edge_to_id_len,
                                                d_uniq_nodes,
                                                sample_size,
                                                len,
                                                *edge_type_graph_ptr,
                                                conf.weighted_sample,
                                                conf.return_weight);

  guard.SetDeviceIndex(conf.gpuid);
  int *all_sample_count_ptr =
      reinterpret_cast<int *>(sample_res.actual_sample_size_mem->ptr());

  auto cumsum_actual_sample_size =
      memory::Alloc(place,
                    (len * conf.edge_to_id_len + 1) * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *cumsum_actual_sample_size_ptr =
      reinterpret_cast<int *>(cumsum_actual_sample_size->ptr());
  cudaMemsetAsync(cumsum_actual_sample_size_ptr,
                  0,
                  (len * conf.edge_to_id_len + 1) * sizeof(int),
                  stream);

  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           all_sample_count_ptr,
                                           cumsum_actual_sample_size_ptr + 1,
                                           len * conf.edge_to_id_len,
                                           stream));
  auto d_temp_storage =
      memory::Alloc(place,
                    temp_storage_bytes,
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           all_sample_count_ptr,
                                           cumsum_actual_sample_size_ptr + 1,
                                           len * conf.edge_to_id_len,
                                           stream));
  cudaStreamSynchronize(stream);

  edges_split_num_ptr->resize(conf.edge_to_id_len);
  for (int i = 0; i < conf.edge_to_id_len; i++) {
    cudaMemcpyAsync(edges_split_num_ptr->data() + i,
                    cumsum_actual_sample_size_ptr + (i + 1) * len,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    stream);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  int all_sample_size = (*edges_split_num_ptr)[conf.edge_to_id_len - 1];
  auto final_sample_val =
      memory::AllocShared(place,
                          all_sample_size * sizeof(int64_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto final_sample_val_dst =
      memory::AllocShared(place,
                          all_sample_size * sizeof(int64_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int64_t *final_sample_val_ptr =
      reinterpret_cast<int64_t *>(final_sample_val->ptr());
  int64_t *final_sample_val_dst_ptr =
      reinterpret_cast<int64_t *>(final_sample_val_dst->ptr());
  int64_t *all_sample_val_ptr =
      reinterpret_cast<int64_t *>(sample_res.val_mem->ptr());

  std::shared_ptr<phi::Allocation> final_sample_weight;
  float *final_sample_weight_ptr = nullptr, *all_sample_weight_ptr = nullptr;
  if (conf.return_weight) {
    final_sample_weight = memory::AllocShared(
        place,
        all_sample_size * sizeof(float),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    final_sample_weight_ptr =
        reinterpret_cast<float *>(final_sample_weight->ptr());
    all_sample_weight_ptr =
        reinterpret_cast<float *>(sample_res.weight_mem->ptr());
  }
  FillActualNeighbors<<<GET_BLOCKS(len * conf.edge_to_id_len),
                        CUDA_NUM_THREADS,
                        0,
                        stream>>>(all_sample_val_ptr,
                                  final_sample_val_ptr,
                                  final_sample_val_dst_ptr,
                                  all_sample_count_ptr,
                                  cumsum_actual_sample_size_ptr,
                                  all_sample_weight_ptr,
                                  final_sample_weight_ptr,
                                  sample_size,
                                  len * conf.edge_to_id_len,
                                  len,
                                  conf.return_weight);
  *neighbor_len = all_sample_size;
  cudaStreamSynchronize(stream);

  std::vector<std::shared_ptr<phi::Allocation>> sample_results;
  sample_results.emplace_back(final_sample_val);
  sample_results.emplace_back(final_sample_val_dst);
  if (conf.return_weight) {
    sample_results.emplace_back(final_sample_weight);
  }
  return sample_results;
}

std::shared_ptr<phi::Allocation> FillReindexHashTable(int64_t *input,
                                                      int num_input,
                                                      int64_t len_hashtable,
                                                      int64_t *keys,
                                                      int *values,
                                                      int *key_index,
                                                      int *final_nodes_len,
                                                      const phi::Place &place,
                                                      const int gpu_id,
                                                      cudaStream_t stream) {
  platform::CUDADeviceGuard guard(gpu_id);
  phi::BuildHashTable<int64_t>
      <<<GET_BLOCKS(num_input), CUDA_NUM_THREADS, 0, stream>>>(
          input, num_input, len_hashtable, keys, key_index);

  // Get item index count.
  auto item_count =
      memory::Alloc(place,
                    (num_input + 1) * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *item_count_ptr = reinterpret_cast<int *>(item_count->ptr());
  cudaMemsetAsync(item_count_ptr, 0, sizeof(int) * (num_input + 1), stream);
  phi::GetItemIndexCount<int64_t>
      <<<GET_BLOCKS(num_input), CUDA_NUM_THREADS, 0, stream>>>(
          input, item_count_ptr, num_input, len_hashtable, keys, key_index);

  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(NULL,
                                temp_storage_bytes,
                                item_count_ptr,
                                item_count_ptr,
                                num_input + 1,
                                stream);
  auto d_temp_storage =
      memory::Alloc(place,
                    temp_storage_bytes,
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                temp_storage_bytes,
                                item_count_ptr,
                                item_count_ptr,
                                num_input + 1,
                                stream);

  int total_unique_items = 0;
  cudaMemcpyAsync(&total_unique_items,
                  item_count_ptr + num_input,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  auto unique_items =
      memory::AllocShared(place,
                          total_unique_items * sizeof(int64_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int64_t *unique_items_ptr = reinterpret_cast<int64_t *>(unique_items->ptr());
  *final_nodes_len = total_unique_items;

  // Get unique items
  phi::FillUniqueItems<int64_t>
      <<<GET_BLOCKS(num_input), CUDA_NUM_THREADS, 0, stream>>>(input,
                                                               num_input,
                                                               len_hashtable,
                                                               unique_items_ptr,
                                                               item_count_ptr,
                                                               keys,
                                                               values,
                                                               key_index);
  cudaStreamSynchronize(stream);
  return unique_items;
}

std::shared_ptr<phi::Allocation> GetReindexResult(int64_t *reindex_src_data,
                                                  int64_t *center_nodes,
                                                  int *final_nodes_len,
                                                  int reindex_table_size,
                                                  int node_len,
                                                  int64_t neighbor_len,
                                                  const phi::Place &place,
                                                  const int gpu_id,
                                                  cudaStream_t stream) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto d_reindex_table_key =
      memory::AllocShared(place,
                          reindex_table_size * sizeof(int64_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int64_t *d_reindex_table_key_ptr =
      reinterpret_cast<int64_t *>(d_reindex_table_key->ptr());
  auto d_reindex_table_value =
      memory::AllocShared(place,
                          reindex_table_size * sizeof(int),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *d_reindex_table_value_ptr =
      reinterpret_cast<int *>(d_reindex_table_value->ptr());
  auto d_reindex_table_index =
      memory::AllocShared(place,
                          reindex_table_size * sizeof(int),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *d_reindex_table_index_ptr =
      reinterpret_cast<int *>(d_reindex_table_index->ptr());

  // Fill table with -1.
  cudaMemsetAsync(d_reindex_table_key_ptr,
                  -1,
                  reindex_table_size * sizeof(int64_t),
                  stream);
  cudaMemsetAsync(
      d_reindex_table_value_ptr, -1, reindex_table_size * sizeof(int), stream);
  cudaMemsetAsync(
      d_reindex_table_index_ptr, -1, reindex_table_size * sizeof(int), stream);

  auto all_nodes =
      memory::AllocShared(place,
                          (node_len + neighbor_len) * sizeof(int64_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int64_t *all_nodes_data = reinterpret_cast<int64_t *>(all_nodes->ptr());

  cudaMemcpyAsync(all_nodes_data,
                  center_nodes,
                  sizeof(int64_t) * node_len,
                  cudaMemcpyDeviceToDevice,
                  stream);
  cudaMemcpyAsync(all_nodes_data + node_len,
                  reindex_src_data,
                  sizeof(int64_t) * neighbor_len,
                  cudaMemcpyDeviceToDevice,
                  stream);

  cudaStreamSynchronize(stream);

  auto final_nodes = FillReindexHashTable(all_nodes_data,
                                          node_len + neighbor_len,
                                          reindex_table_size,
                                          d_reindex_table_key_ptr,
                                          d_reindex_table_value_ptr,
                                          d_reindex_table_index_ptr,
                                          final_nodes_len,
                                          place,
                                          gpu_id,
                                          stream);

  phi::ReindexSrcOutput<int64_t>
      <<<GET_BLOCKS(neighbor_len), CUDA_NUM_THREADS, 0, stream>>>(
          reindex_src_data,
          neighbor_len,
          reindex_table_size,
          d_reindex_table_key_ptr,
          d_reindex_table_value_ptr);

  cudaStreamSynchronize(stream);
  return final_nodes;
}

std::shared_ptr<phi::Allocation> GenerateSampleGraph(
    uint64_t *node_ids,  // input
    int len,             // input
    int *final_len,
    const GraphDataGeneratorConfig &conf,
    std::vector<std::shared_ptr<phi::Allocation>> *inverse_vec_ptr,
    std::vector<std::vector<std::shared_ptr<phi::Allocation>>>
        *graph_edges_vec_ptr,  // output
    std::vector<std::vector<std::vector<int>>>
        *edges_split_num_vec_ptr,  // output
    std::vector<std::shared_ptr<phi::Allocation>> *edge_type_graph_ptr,
    std::shared_ptr<phi::Allocation> *d_uniq_node_num,
    uint64_t *copy_unique_len_ptr,
    std::vector<uint64_t> *host_vec_ptr,                   // Output
    std::vector<uint32_t> *host_ranks_ptr,                 // output
    std::shared_ptr<HashTable<uint64_t, uint32_t>> table,  // Output
    const phi::Place &place,
    cudaStream_t stream) {
  VLOG(1) << conf.gpuid << " Get Unique Nodes";
  platform::CUDADeviceGuard guard(conf.gpuid);

  auto inverse =
      memory::AllocShared(place,
                          len * sizeof(uint32_t),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *inverse_ptr = reinterpret_cast<uint32_t *>(inverse->ptr());
  auto uniq_nodes =
      memory::Alloc(place,
                    len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t *uniq_nodes_data = reinterpret_cast<uint64_t *>(uniq_nodes->ptr());
  int uniq_len = dedup_keys_and_fillidx(
      conf.gpuid, node_ids, len, uniq_nodes_data, inverse_ptr, place, stream);
  int len_samples = conf.samples.size();

  VLOG(1) << conf.gpuid << " Sample Neighbors and Reindex";
  std::vector<int> edges_split_num;
  std::vector<std::shared_ptr<phi::Allocation>> final_nodes_vec;
  std::vector<std::shared_ptr<phi::Allocation>> graph_edges;
  std::vector<std::vector<int>> edges_split_num_for_graph;
  std::vector<int> final_nodes_len_vec;

  for (int i = 0; i < len_samples; i++) {
    edges_split_num.clear();
    std::shared_ptr<phi::Allocation> neighbors, reindex_dst, weights;
    int64_t neighbors_len = 0;
    if (i == 0) {
      auto sample_results = SampleNeighbors(uniq_nodes_data,
                                            uniq_len,
                                            conf.samples[i],
                                            conf,
                                            &edges_split_num,
                                            &neighbors_len,
                                            edge_type_graph_ptr,
                                            d_uniq_node_num,
                                            copy_unique_len_ptr,
                                            host_vec_ptr,
                                            host_ranks_ptr,
                                            table,
                                            place,
                                            stream);
      neighbors = sample_results[0];
      reindex_dst = sample_results[1];
      if (conf.return_weight) {
        weights = sample_results[2];
      }
      edges_split_num.push_back(uniq_len);
    } else {
      uint64_t *final_nodes_data =
          reinterpret_cast<uint64_t *>(final_nodes_vec[i - 1]->ptr());
      auto sample_results = SampleNeighbors(final_nodes_data,
                                            final_nodes_len_vec[i - 1],
                                            conf.samples[i],
                                            conf,
                                            &edges_split_num,
                                            &neighbors_len,
                                            edge_type_graph_ptr,
                                            d_uniq_node_num,
                                            copy_unique_len_ptr,
                                            host_vec_ptr,
                                            host_ranks_ptr,
                                            table,
                                            place,
                                            stream);
      neighbors = sample_results[0];
      reindex_dst = sample_results[1];
      if (conf.return_weight) {
        weights = sample_results[2];
      }
      edges_split_num.push_back(final_nodes_len_vec[i - 1]);
    }

    int64_t *reindex_src_data = reinterpret_cast<int64_t *>(neighbors->ptr());
    int final_nodes_len = 0;
    if (i == 0) {
      auto tmp_final_nodes =
          GetReindexResult(reindex_src_data,
                           reinterpret_cast<int64_t *>(uniq_nodes_data),
                           &final_nodes_len,
                           conf.reindex_table_size,
                           uniq_len,
                           neighbors_len,
                           place,
                           conf.gpuid,
                           stream);
      final_nodes_vec.emplace_back(tmp_final_nodes);
      final_nodes_len_vec.emplace_back(final_nodes_len);
    } else {
      int64_t *final_nodes_data =
          reinterpret_cast<int64_t *>(final_nodes_vec[i - 1]->ptr());
      auto tmp_final_nodes = GetReindexResult(reindex_src_data,
                                              final_nodes_data,
                                              &final_nodes_len,
                                              conf.reindex_table_size,
                                              final_nodes_len_vec[i - 1],
                                              neighbors_len,
                                              place,
                                              conf.gpuid,
                                              stream);
      final_nodes_vec.emplace_back(tmp_final_nodes);
      final_nodes_len_vec.emplace_back(final_nodes_len);
    }
    edges_split_num.emplace_back(
        final_nodes_len_vec[i]);  // [edges_split_num, next_num_nodes,
                                  // num_nodes]
    edges_split_num.emplace_back(neighbors_len);
    graph_edges.emplace_back(neighbors);
    graph_edges.emplace_back(reindex_dst);
    if (conf.return_weight) {
      graph_edges.emplace_back(weights);
    }
    edges_split_num_for_graph.emplace_back(edges_split_num);
  }

  // Obtain the dest machine for sample node through cross machine queries
  // Actually, its just filling in d_ranks_ptr.
  if (!final_nodes_vec.empty() && !final_nodes_len_vec.empty()) {
    auto d_final_keys =
        reinterpret_cast<uint64_t *>((*final_nodes_vec.rbegin())->ptr());
    auto tmp_len = *(final_nodes_len_vec.rbegin());
    if (tmp_len > 0) {
      auto d_ranks =
          memory::Alloc(place,
                        tmp_len * sizeof(uint32_t),
                        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      uint32_t *d_ranks_ptr = nullptr;
      if (FLAGS_graph_edges_split_mode == "fennel" ||
          FLAGS_query_dest_rank_by_multi_node) {
        auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
        d_ranks_ptr = reinterpret_cast<uint32_t *>(d_ranks->ptr());
        gpu_graph_ptr->seek_keys_rank(
            conf.gpuid, d_final_keys, tmp_len, d_ranks_ptr);
      }

      if (InsertTable(conf.gpuid,
                      d_final_keys,
                      d_ranks_ptr,
                      tmp_len,
                      d_uniq_node_num,
                      conf,
                      copy_unique_len_ptr,
                      place,
                      host_vec_ptr,
                      host_ranks_ptr,
                      table,
                      stream) != 0) {
        VLOG(0) << "gpu:" << conf.gpuid
                << " in GenerateSampleGraph, insert key stage, table is full";
        assert(false);
      }
    }
  }

  graph_edges_vec_ptr->emplace_back(graph_edges);
  edges_split_num_vec_ptr->emplace_back(edges_split_num_for_graph);
  inverse_vec_ptr->emplace_back(inverse);

  *final_len = final_nodes_len_vec[len_samples - 1];
  return final_nodes_vec[len_samples - 1];
}

std::shared_ptr<phi::Allocation> GetNodeDegree(
    uint64_t *node_ids,
    int len,
    const GraphDataGeneratorConfig &conf,
    const phi::Place &place,
    cudaStream_t stream) {
  platform::CUDADeviceGuard guard(conf.gpuid);
  auto node_degree =
      memory::AllocShared(place,
                          len * conf.edge_to_id_len * sizeof(int),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  int *node_degree_ptr = reinterpret_cast<int *>(node_degree->ptr());
  for (auto &iter : edge_to_id) {
    int edge_idx = iter.second;
    auto sub_node_degree =
        gpu_graph_ptr->get_node_degree(conf.gpuid, edge_idx, node_ids, len);
    int *sub_node_degree_ptr = reinterpret_cast<int *>(sub_node_degree->ptr());
    cudaMemcpy(node_degree_ptr + edge_idx * len,
               sub_node_degree_ptr,
               sizeof(int) * len,
               cudaMemcpyDeviceToDevice);
  }
  return node_degree;
}

int multi_node_sync_sample(int flag,
                           const ncclRedOp_t &op,
                           const phi::Place &place,
                           const int gpu_id,
                           phi::DenseTensor *multi_node_sync_stat_ptr) {
  if (flag < 0 && flag > 2) {
    VLOG(0) << "invalid flag! " << flag;
    assert(false);
    return -1;
  }

  int ret = 0;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  platform::CUDADeviceGuard guard(gpu_id);
  int *stat_ptr = multi_node_sync_stat_ptr->data<int>();
  auto comm = platform::NCCLCommContext::Instance().Get(0, place.GetDeviceId());
  auto stream = comm->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
      &stat_ptr[flag], &stat_ptr[3], 1, ncclInt, op, comm->comm(), stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&ret,  // output
                                             &stat_ptr[3],
                                             sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
  return ret;
}

int get_multi_node_global_flag(int local_flag,
                               const ncclRedOp_t &op,
                               const phi::Place &place,
                               const int gpu_id,
                               cudaStream_t stream) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto send_buff =
      memory::Alloc(place,
                    2 * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *send_buff_ptr = reinterpret_cast<int *>(send_buff->ptr());
  cudaMemcpyAsync(
      send_buff_ptr, &local_flag, sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  auto comm = platform::NCCLCommContext::Instance().Get(0, place.GetDeviceId());
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(&send_buff_ptr[0],
                                                         &send_buff_ptr[1],
                                                         1,
                                                         ncclInt,
                                                         op,
                                                         comm->comm(),
                                                         stream));
  int global_flag = 0;
  cudaMemcpyAsync(&global_flag,
                  &send_buff_ptr[1],
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  return global_flag;
}

int FillWalkBuf(const std::vector<uint64_t> &h_device_keys_len,
                const std::vector<std::shared_ptr<phi::Allocation>>
                    &d_device_keys,                              // input
                const std::vector<std::vector<int>> &meta_path,  // input
                const GraphDataGeneratorConfig &conf,
                bool *epoch_finish_ptr,
                uint64_t *copy_unique_len_ptr,
                const phi::Place &place,
                const std::vector<int> &first_node_type,
                std::unordered_map<int, size_t> *node_type_start_ptr,
                std::set<int> *finish_node_type_ptr,
                uint64_t *walk,  // output
                uint8_t *walk_ntype,
                std::shared_ptr<phi::Allocation> *d_uniq_node_num,
                int *d_random_row,
                int *d_random_row_col_shift,
                phi::DenseTensor *multi_node_sync_stat_ptr,
                std::vector<uint64_t> *host_vec_ptr,    // output
                std::vector<uint32_t> *host_ranks_ptr,  // output
                int *total_row_ptr,
                size_t *jump_rows_ptr,
                int *shuffle_seed_ptr,
                std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank_table,
                BufState *buf_state,
                cudaStream_t stream) {
  platform::CUDADeviceGuard guard(conf.gpuid);

  ////////
  uint64_t *h_walk;
  if (conf.debug_mode) {
    h_walk = new uint64_t[conf.buf_size];
  }
  ///////
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(walk, 0, conf.buf_size * sizeof(uint64_t), stream));
  if (conf.need_walk_ntype) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
        walk_ntype, 0, conf.buf_size * sizeof(uint8_t), stream));
  }
  int sample_times = 0;
  int i = 0;
  *total_row_ptr = 0;

  std::vector<std::shared_ptr<phi::Allocation>> d_sampleidx2rows;
  d_sampleidx2rows.push_back(memory::AllocShared(
      place,
      conf.once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(stream))));
  d_sampleidx2rows.push_back(memory::AllocShared(
      place,
      conf.once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(stream))));
  int cur_sampleidx2row = 0;

  // 获取全局采样状态
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  auto &edge_neighbor_size_limit = gpu_graph_ptr->get_type_to_neighbor_limit();
  auto &cursor = gpu_graph_ptr->cursor_[conf.thread_id];
  size_t node_type_len = first_node_type.size();
  int remain_size = conf.buf_size - conf.walk_degree *
                                        conf.once_sample_startid_len *
                                        conf.walk_len;
  int total_samples = 0;

  // Definition of variables related to multi machine sampling
  int switch_flag = EVENT_NOT_SWITCH;  // Mark whether the local machine needs
                                       // to switch metapath
  int switch_command = EVENT_NOT_SWITCH;    // Mark whether to switch metapath,
                                            // after multi node sync
  int sample_flag = EVENT_CONTINUE_SAMPLE;  // Mark whether the local machine
                                            // needs to continue sampling
  int sample_command =
      EVENT_CONTINUE_SAMPLE;  // Mark whether to continue sampling, after multi
                              // node sync

  // In the case of a single machine, for scenarios where the d_walk buffer is
  // full, epoch sampling ends, and metapath switching occurs, direct decisions
  // are made to end the current card sampling or perform metapath switching.
  // However, in the case of multiple machines, further decisions can only be
  // made after waiting for the multiple machines to synchronize and exchange
  // information.
  while (1) {
    if (i > remain_size) {
      // scenarios 1: d_walk is full
      if (FLAGS_enable_graph_multi_node_sampling) {
        sample_flag = EVENT_WALKBUF_FULL;
      } else {
        break;
      }
    }

    int cur_node_idx = cursor % node_type_len;
    int node_type = first_node_type[cur_node_idx];
    auto &path = meta_path[cur_node_idx];
    size_t start = (*node_type_start_ptr)[node_type];
    int type_index = type_to_index[node_type];
    size_t device_key_size = h_device_keys_len[type_index];
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys[type_index]->ptr());
    int tmp_len = start + conf.once_sample_startid_len > device_key_size
                      ? device_key_size - start
                      : conf.once_sample_startid_len;
    VLOG(2) << "gpu=" << conf.gpuid << " choose node_type: " << node_type
            << " cur_node_idx: " << cur_node_idx
            << " meta_path.size: " << meta_path.size()
            << " key_size: " << device_key_size << " start: " << start
            << " tmp_len: " << tmp_len;
    if (tmp_len == 0) {
      finish_node_type_ptr->insert(node_type);
      if (finish_node_type_ptr->size() == node_type_start_ptr->size()) {
        // scenarios 2: epoch finish
        if (FLAGS_enable_graph_multi_node_sampling) {
          sample_flag = EVENT_FINISH_EPOCH;
        } else {
          cursor = 0;
          *epoch_finish_ptr = true;
          break;
        }
      }

      // scenarios 3: switch metapath
      if (FLAGS_enable_graph_multi_node_sampling) {
        if (sample_flag == EVENT_CONTINUE_SAMPLE) {
          // Switching only occurs when multi machine sampling continues
          switch_flag = EVENT_SWITCH_METAPATH;
        }
      } else {
        cursor += 1;
        continue;
      }
    }

    // Perform synchronous information exchange between multiple machines
    // to decide whether to continue sampling
    if (FLAGS_enable_graph_multi_node_sampling) {
      switch_command = multi_node_sync_sample(
          switch_flag, ncclProd, place, conf.gpuid, multi_node_sync_stat_ptr);
      VLOG(2) << "gpuid:" << conf.gpuid << " multi node sample sync"
              << " switch_flag:" << switch_flag << "," << switch_command;
      if (switch_command) {
        cursor += 1;
        switch_flag = EVENT_NOT_SWITCH;
        continue;
      }

      sample_command = multi_node_sync_sample(
          sample_flag, ncclMax, place, conf.gpuid, multi_node_sync_stat_ptr);
      VLOG(2) << "gpuid:" << conf.gpuid << " multi node sample sync"
              << " sample_flag:" << sample_flag << "," << sample_command;
      if (sample_command == EVENT_FINISH_EPOCH) {
        // end sampling current epoch
        cursor = 0;
        *epoch_finish_ptr = true;
        VLOG(1) << "sample epoch finish!";
        break;
      } else if (sample_command == EVENT_WALKBUF_FULL) {
        // end sampling current pass
        VLOG(1) << "sample pass finish!";
        break;
      } else if (sample_command == EVENT_CONTINUE_SAMPLE) {
        // continue sampling
      } else {
        // shouldn't come here
        VLOG(0) << "should not come here, sample_command:" << sample_command;
        assert(false);
      }
    }

    int step = 1;
    bool update = true;
    uint64_t *cur_walk = walk + i;
    uint8_t *cur_walk_ntype = NULL;
    if (conf.need_walk_ntype) {
      cur_walk_ntype = walk_ntype + i;
    }

    NeighborSampleQuery q;
    if (edge_neighbor_size_limit.find(path[0]) ==
        edge_neighbor_size_limit.end()) {
      VLOG(0) << "Fail to find edge[" << path[0]
              << "] in edge_neighbor_size_limit";
      assert(false);
      break;
    }
    auto neighbor_size_limit = edge_neighbor_size_limit[path[0]];
    q.initialize(conf.gpuid,
                 path[0],
                 (uint64_t)(d_type_keys + start),
                 conf.walk_degree,
                 tmp_len,
                 neighbor_size_limit,
                 step);

    // Obtain the dest machine for sample node through cross machine queries
    // Actually, its just filling in d_ranks_ptr.
    auto d_ranks =
        memory::Alloc(place,
                      tmp_len * conf.walk_degree * sizeof(uint32_t),
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    uint32_t *d_ranks_ptr = nullptr;
    if (FLAGS_graph_edges_split_mode == "fennel" ||
        FLAGS_query_dest_rank_by_multi_node) {
      d_ranks_ptr = reinterpret_cast<uint32_t *>(d_ranks->ptr());
      gpu_graph_ptr->seek_keys_rank(
          conf.gpuid, q.src_nodes, q.len, d_ranks_ptr);
    }

    // fill three data structure, as follows:
    // 1. host_vec_ptr.
    // 2. host_ranks_ptr
    // 3. keys2rank_table
    if (InsertTable(conf.gpuid,
                    d_type_keys + start,
                    d_ranks_ptr,
                    tmp_len,
                    d_uniq_node_num,
                    conf,
                    copy_unique_len_ptr,
                    place,
                    host_vec_ptr,
                    host_ranks_ptr,
                    keys2rank_table,
                    stream) != 0) {
      VLOG(2) << "gpu:" << conf.gpuid
              << " in step 0, insert key stage, table is full";
      update = false;
      assert(false);
      break;
    }

    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(
        q, false, true, conf.weighted_sample);

    *jump_rows_ptr = sample_res.total_sample_size;
    total_samples += sample_res.total_sample_size;

    if (FLAGS_enable_graph_multi_node_sampling) {
      int flag = *jump_rows_ptr > 0 ? 1 : 0;
      int command = multi_node_sync_sample(
          flag, ncclMax, place, conf.gpuid, multi_node_sync_stat_ptr);
      VLOG(2) << "gpuid:" << conf.gpuid << " multi node step sync"
              << " step:" << step << " step_sample:" << flag << "," << command;
      if (command <= 0) {
        (*node_type_start_ptr)[node_type] = tmp_len + start;
        cursor += 1;
        continue;
      }
    } else if (*jump_rows_ptr == 0) {
      (*node_type_start_ptr)[node_type] = tmp_len + start;
      cursor += 1;
      continue;
    }

    FillOneStep(d_type_keys + start,
                path[0],
                cur_walk,
                cur_walk_ntype,
                tmp_len,
                &sample_res,
                conf.walk_degree,
                step,
                conf,
                &d_sampleidx2rows,
                &cur_sampleidx2row,
                place,
                stream);
    /////////
    if (conf.debug_mode) {
      cudaMemcpy(h_walk,
                 walk,
                 conf.buf_size * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < conf.buf_size; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }

    /////////
    step++;
    size_t path_len = path.size();
    for (; step < conf.walk_len; step++) {
      if (FLAGS_enable_graph_multi_node_sampling) {
        // Step synchronization for multi-step sampling in multi node
        int flag = sample_res.total_sample_size > 0 ? 1 : 0;
        int command = multi_node_sync_sample(
            flag, ncclMax, place, conf.gpuid, multi_node_sync_stat_ptr);
        VLOG(2) << "gpuid:" << conf.gpuid << " multi node step sync"
                << " step:" << step << " step_sample:" << flag << ","
                << command;
        if (command <= 0) {
          break;
        }
      } else {
        // Finish multi-step sampling in single node
        if (sample_res.total_sample_size == 0) {
          VLOG(2) << "sample finish, step=" << step;
          break;
        }
      }

      auto sample_key_mem = sample_res.actual_val_mem;
      uint64_t *sample_keys_ptr = nullptr;
      if (sample_key_mem) {
        sample_keys_ptr = reinterpret_cast<uint64_t *>(sample_key_mem->ptr());
      }
      int edge_type_id = path[(step - 1) % path_len];
      if (edge_neighbor_size_limit.find(edge_type_id) ==
          edge_neighbor_size_limit.end()) {
        VLOG(0) << "Fail to find edge[" << path[0]
                << "] in edge_neighbor_size_limit";
        assert(false);
        break;
      }
      neighbor_size_limit = edge_neighbor_size_limit[edge_type_id];
      q.initialize(conf.gpuid,
                   edge_type_id,
                   (uint64_t)sample_keys_ptr,
                   1,
                   sample_res.total_sample_size,
                   neighbor_size_limit,
                   step);
      if (FLAGS_graph_edges_split_mode == "fennel" ||
          FLAGS_query_dest_rank_by_multi_node) {
        gpu_graph_ptr->seek_keys_rank(
            conf.gpuid, q.src_nodes, q.len, d_ranks_ptr);
      }

      if (InsertTable(conf.gpuid,
                      sample_res.actual_val,
                      d_ranks_ptr,
                      sample_res.total_sample_size,
                      d_uniq_node_num,
                      conf,
                      copy_unique_len_ptr,
                      place,
                      host_vec_ptr,
                      host_ranks_ptr,
                      keys2rank_table,
                      stream) != 0) {
        VLOG(0) << "gpu:" << conf.gpuid << " in step: " << step
                << ", table is full";
        update = false;
        assert(false);
        break;
      }

      int sample_key_len = sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(
          q, false, true, conf.weighted_sample);
      total_samples += sample_res.total_sample_size;
      FillOneStep(d_type_keys + start,
                  edge_type_id,
                  cur_walk,
                  cur_walk_ntype,
                  sample_key_len,
                  &sample_res,
                  1,
                  step,
                  conf,
                  &d_sampleidx2rows,
                  &cur_sampleidx2row,
                  place,
                  stream);
      if (conf.debug_mode) {
        cudaMemcpy(h_walk,
                   walk,
                   conf.buf_size * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int xx = 0; xx < conf.buf_size; xx++) {
          VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
        }
      }
    }

    // Process the result of last sampling
    if (sample_res.total_sample_size >= 0) {
      if (FLAGS_graph_edges_split_mode == "fennel" ||
          FLAGS_query_dest_rank_by_multi_node) {
        gpu_graph_ptr->seek_keys_rank(conf.gpuid,
                                      sample_res.actual_val,
                                      sample_res.total_sample_size,
                                      d_ranks_ptr);
      }

      if (InsertTable(conf.gpuid,
                      sample_res.actual_val,
                      d_ranks_ptr,
                      sample_res.total_sample_size,
                      d_uniq_node_num,
                      conf,
                      copy_unique_len_ptr,
                      place,
                      host_vec_ptr,
                      host_ranks_ptr,
                      keys2rank_table,
                      stream) != 0) {
        VLOG(0) << "gpu:" << conf.gpuid << " in step: " << step
                << ", table is full";
        update = false;
        assert(false);
        break;
      }
    }

    // 此时更新全局采样状态
    if (update == true) {
      (*node_type_start_ptr)[node_type] = tmp_len + start;
      i += *jump_rows_ptr * conf.walk_len;
      *total_row_ptr += *jump_rows_ptr;
      cursor += 1;
      sample_times++;
    } else {
      VLOG(0) << "table is full, multi node will suspend!";
      assert(false);
      break;
    }

    VLOG(2) << "gpu=" << conf.gpuid << ", sample " << sample_times
            << " finish, node_type=" << node_type << ", path:[" << path[0]
            << "," << path[1] << "]"
            << ", start:" << start << ", len:" << tmp_len
            << ", row:" << *jump_rows_ptr << ", total_step:" << step
            << ", total_row:" << *total_row_ptr << ", i:" << i
            << ", remain_size:" << remain_size
            << ", device_key_size:" << device_key_size;
  }

  // insert 0key to table
  {
    auto d_buf = memory::AllocShared(
        place,
        sizeof(uint64_t) * 2,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    uint64_t *d_buf_ptr = reinterpret_cast<uint64_t *>(d_buf->ptr());
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemsetAsync(d_buf_ptr, 0, sizeof(uint64_t) * 2, stream));
    uint64_t *d_zero_key_ptr = d_buf_ptr;
    uint32_t *d_zero_rank_ptr = reinterpret_cast<uint32_t *>(d_buf_ptr + 1);
    uint64_t key_num = 1;
    if (InsertTable(conf.gpuid,
                    d_zero_key_ptr,
                    d_zero_rank_ptr,
                    key_num,
                    d_uniq_node_num,
                    conf,
                    copy_unique_len_ptr,
                    place,
                    host_vec_ptr,
                    host_ranks_ptr,
                    keys2rank_table,
                    stream) != 0) {
      VLOG(0) << "gpu:" << conf.gpuid << " insert 0key failed";
      assert(false);
    }
  }

  platform::CUDADeviceGuard guard2(conf.gpuid);
  buf_state->Reset(*total_row_ptr);
  paddle::memory::ThrustAllocator<cudaStream_t> allocator(place, stream);
  thrust::random::default_random_engine engine(*shuffle_seed_ptr);
  const auto &exec_policy = thrust::cuda::par(allocator).on(stream);
  thrust::counting_iterator<int> cnt_iter(0);
#if defined(PADDLE_WITH_CUDA)
  phi::funcs::shuffle_copy_fixed(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec_policy)),
#else
  thrust::shuffle_copy(exec_policy,
#endif
      cnt_iter,
      cnt_iter + *total_row_ptr,
      thrust::device_pointer_cast(d_random_row),
      engine);

  thrust::transform(exec_policy,
                    cnt_iter,
                    cnt_iter + *total_row_ptr,
                    thrust::device_pointer_cast(d_random_row_col_shift),
                    RandInt(0, conf.walk_len));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  *shuffle_seed_ptr = engine();

  if (conf.debug_mode) {
    int *h_random_row = new int[*total_row_ptr + 10];
    cudaMemcpy(h_random_row,
               d_random_row,
               *total_row_ptr * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < *total_row_ptr; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
  }

  return *total_row_ptr != 0;
}

int FillWalkBufMultiPath(
    const std::vector<uint64_t> &h_device_keys_len,
    const std::vector<std::vector<int>> &meta_path,
    const GraphDataGeneratorConfig &conf,
    bool *epoch_finish_ptr,
    uint64_t *copy_unique_len_ptr,
    const phi::Place &place,
    const std::vector<int> &first_node_type,
    std::unordered_map<int, size_t> *node_type_start_ptr,
    uint64_t *walk,  // output
    uint8_t *walk_ntype,
    std::shared_ptr<phi::Allocation> *d_uniq_node_num,
    int *d_random_row,
    int *d_random_row_col_shift,
    std::vector<uint64_t> *host_vec_ptr,
    int *total_row_ptr,
    size_t *jump_rows_ptr,
    int *shuffle_seed_ptr,
    uint64_t *d_train_metapath_keys,
    uint64_t *h_train_metapath_keys_len_ptr,
    std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank_table,
    BufState *buf_state,
    cudaStream_t stream) {
  platform::CUDADeviceGuard guard(conf.gpuid);

  ////////
  uint64_t *h_walk;
  if (conf.debug_mode) {
    h_walk = new uint64_t[conf.buf_size];
  }
  ///////
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  cudaMemsetAsync(walk, 0, conf.buf_size * sizeof(uint64_t), stream);
  if (conf.need_walk_ntype) {
    cudaMemsetAsync(walk_ntype, 0, conf.buf_size * sizeof(uint8_t), stream);
  }
  int sample_times = 0;
  int i = 0;
  *total_row_ptr = 0;

  std::vector<std::shared_ptr<phi::Allocation>> d_sampleidx2rows;
  d_sampleidx2rows.push_back(memory::AllocShared(
      place,
      conf.once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(stream))));
  d_sampleidx2rows.push_back(memory::AllocShared(
      place,
      conf.once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(stream))));
  int cur_sampleidx2row = 0;

  // 获取全局采样状态
  auto &cur_metapath = gpu_graph_ptr->cur_metapath_;
  auto &edge_neighbor_size_limit = gpu_graph_ptr->get_type_to_neighbor_limit();
  auto &path = gpu_graph_ptr->cur_parse_metapath_;
  auto &cur_metapath_start = gpu_graph_ptr->cur_metapath_start_[conf.gpuid];
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  size_t node_type_len = first_node_type.size();
  std::string first_node =
      paddle::string::split_string<std::string>(cur_metapath, "2")[0];
  auto it = gpu_graph_ptr->node_to_id.find(first_node);
  auto node_type = it->second;

  int remain_size = conf.buf_size - conf.walk_degree *
                                        conf.once_sample_startid_len *
                                        conf.walk_len;
  int total_samples = 0;

  while (i <= remain_size) {
    size_t start = cur_metapath_start;
    size_t device_key_size = *h_train_metapath_keys_len_ptr;
    VLOG(2) << "type: " << node_type << " size: " << device_key_size
            << " start: " << start;
    uint64_t *d_type_keys = d_train_metapath_keys;
    int tmp_len = start + conf.once_sample_startid_len > device_key_size
                      ? device_key_size - start
                      : conf.once_sample_startid_len;
    bool update = true;
    if (tmp_len == 0) {
      *epoch_finish_ptr = true;
      break;
    }

    VLOG(2) << "gpuid = " << conf.gpuid << " path[0] = " << path[0];
    uint64_t *cur_walk = walk + i;
    uint8_t *cur_walk_ntype = NULL;
    if (conf.need_walk_ntype) {
      cur_walk_ntype = walk_ntype + i;
    }

    int step = 1;
    VLOG(2) << "sample edge type: " << path[0] << " step: " << 1;

    NeighborSampleQuery q;
    auto neighbor_size_limit = edge_neighbor_size_limit[path[0]];
    q.initialize(conf.gpuid,
                 path[0],
                 (uint64_t)(d_type_keys + start),
                 conf.walk_degree,
                 tmp_len,
                 neighbor_size_limit,
                 step);
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(
        q, false, true, conf.weighted_sample);

    *jump_rows_ptr = sample_res.total_sample_size;
    total_samples += sample_res.total_sample_size;
    VLOG(2) << "i = " << i << " start = " << start << " tmp_len = " << tmp_len
            << "jump row: " << *jump_rows_ptr;
    if (*jump_rows_ptr == 0) {
      cur_metapath_start = tmp_len + start;
      continue;
    }

    if (!conf.sage_mode) {
      if (InsertTable(conf.gpuid,
                      d_type_keys + start,
                      nullptr,
                      tmp_len,
                      d_uniq_node_num,
                      conf,
                      copy_unique_len_ptr,
                      place,
                      host_vec_ptr,
                      nullptr,
                      keys2rank_table,
                      stream) != 0) {
        VLOG(2) << "in step 0, insert key stage, table is full";
        update = false;
        break;
      }
      if (InsertTable(conf.gpuid,
                      sample_res.actual_val,
                      nullptr,
                      sample_res.total_sample_size,
                      d_uniq_node_num,
                      conf,
                      copy_unique_len_ptr,
                      place,
                      host_vec_ptr,
                      nullptr,
                      keys2rank_table,
                      stream) != 0) {
        VLOG(2) << "in step 0, insert sample res stage, table is full";
        update = false;
        break;
      }
    }

    FillOneStep(d_type_keys + start,
                path[0],
                cur_walk,
                cur_walk_ntype,
                tmp_len,
                &sample_res,
                conf.walk_degree,
                step,
                conf,
                &d_sampleidx2rows,
                &cur_sampleidx2row,
                place,
                stream);
    /////////
    if (conf.debug_mode) {
      cudaMemcpy(h_walk,
                 walk,
                 conf.buf_size * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < conf.buf_size; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }

    VLOG(2) << "sample, step=" << step << " sample_keys=" << tmp_len
            << " sample_res_len=" << sample_res.total_sample_size;

    /////////
    step++;
    size_t path_len = path.size();
    for (; step < conf.walk_len; step++) {
      if (sample_res.total_sample_size == 0) {
        VLOG(2) << "sample finish, step=" << step;
        break;
      }
      auto sample_key_mem = sample_res.actual_val_mem;
      uint64_t *sample_keys_ptr =
          reinterpret_cast<uint64_t *>(sample_key_mem->ptr());
      int edge_type_id = path[(step - 1) % path_len];
      VLOG(2) << "sample edge type: " << edge_type_id << " step: " << step;
      neighbor_size_limit = edge_neighbor_size_limit[edge_type_id];
      q.initialize(conf.gpuid,
                   edge_type_id,
                   (uint64_t)sample_keys_ptr,
                   1,
                   sample_res.total_sample_size,
                   neighbor_size_limit,
                   step);
      int sample_key_len = sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(
          q, false, true, conf.weighted_sample);
      total_samples += sample_res.total_sample_size;
      if (!conf.sage_mode) {
        if (InsertTable(conf.gpuid,
                        sample_res.actual_val,
                        nullptr,
                        sample_res.total_sample_size,
                        d_uniq_node_num,
                        conf,
                        copy_unique_len_ptr,
                        place,
                        host_vec_ptr,
                        nullptr,
                        keys2rank_table,
                        stream) != 0) {
          VLOG(2) << "in step: " << step << ", table is full";
          update = false;
          break;
        }
      }
      FillOneStep(d_type_keys + start,
                  edge_type_id,
                  cur_walk,
                  cur_walk_ntype,
                  sample_key_len,
                  &sample_res,
                  1,
                  step,
                  conf,
                  &d_sampleidx2rows,
                  &cur_sampleidx2row,
                  place,
                  stream);
      if (conf.debug_mode) {
        cudaMemcpy(h_walk,
                   walk,
                   conf.buf_size * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int xx = 0; xx < conf.buf_size; xx++) {
          VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
        }
      }

      VLOG(2) << "sample, step=" << step << " sample_keys=" << sample_key_len
              << " sample_res_len=" << sample_res.total_sample_size;
    }
    // 此时更新全局采样状态
    if (update == true) {
      cur_metapath_start = tmp_len + start;
      i += *jump_rows_ptr * conf.walk_len;
      *total_row_ptr += *jump_rows_ptr;
      sample_times++;
    } else {
      VLOG(2) << "table is full, not update stat!";
      break;
    }
  }
  buf_state->Reset(*total_row_ptr);

  paddle::memory::ThrustAllocator<cudaStream_t> allocator(place, stream);
  thrust::random::default_random_engine engine(*shuffle_seed_ptr);
  const auto &exec_policy = thrust::cuda::par(allocator).on(stream);
  thrust::counting_iterator<int> cnt_iter(0);
#if defined(PADDLE_WITH_CUDA)
  phi::funcs::shuffle_copy_fixed(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec_policy)),
#else
  thrust::shuffle_copy(exec_policy,
#endif
      cnt_iter,
      cnt_iter + *total_row_ptr,
      thrust::device_pointer_cast(d_random_row),
      engine);

  thrust::transform(exec_policy,
                    cnt_iter,
                    cnt_iter + *total_row_ptr,
                    thrust::device_pointer_cast(d_random_row_col_shift),
                    RandInt(0, conf.walk_len));

  cudaStreamSynchronize(stream);
  *shuffle_seed_ptr = engine();

  if (conf.debug_mode) {
    int *h_random_row = new int[*total_row_ptr + 10];
    cudaMemcpy(h_random_row,
               d_random_row,
               *total_row_ptr * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < *total_row_ptr; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
  }

  return *total_row_ptr != 0;
}

void GraphDataGenerator::DoWalkandSage() {
  if (FLAGS_graph_edges_split_mode == "fennel" ||
      FLAGS_query_dest_rank_by_multi_node) {
    auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
    gpu_graph_ptr->set_keys2rank(conf_.gpuid, keys2rank_table_);
  }

  int device_id = place_.GetDeviceId();
  debug_gpu_memory_info(device_id, "DoWalkandSage start");
  platform::CUDADeviceGuard guard(conf_.gpuid);
  sage_batch_num_ = 0;
  if (conf_.gpu_graph_training) {
    int local_train_flag = DoWalkForTrain();
    if (!conf_.is_multi_node) {
      if (local_train_flag && conf_.sage_mode) {
        DoSageForTrain();
      }
    } else {
      if (conf_.sage_mode) {
        global_train_flag_ = get_multi_node_global_flag(
            local_train_flag, ncclProd, place_, conf_.gpuid, sample_stream_);
        VLOG(1) << "gpu_id: " << conf_.gpuid
                << ", local_train_flag: " << local_train_flag
                << ", global_train_flag: " << global_train_flag_;
        if (global_train_flag_) {
          // When global_train_flag is true, we need to go ahead in multi-node
          // scenario.
          DoSageForTrain();
        }
      }
    }
  } else {
    bool infer_flag = DoWalkForInfer();
    if (infer_flag && conf_.sage_mode) {
      DoSageForInfer();
    }
  }

  if (conf_.gpu_graph_training || conf_.sage_mode) {
    CopyUniqueNodes(conf_.gpuid,
                    keys2rank_table_,
                    copy_unique_len_,
                    place_,
                    d_uniq_node_num_,
                    &host_vec_,
                    &host_ranks_,
                    sample_stream_);
  }
}

bool GraphDataGenerator::DoWalkForTrain() {
  bool train_flag = true;
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    uint8_t *walk_ntype = NULL;
    if (conf_.need_walk_ntype) {
      walk_ntype =
          reinterpret_cast<uint8_t *>(d_walk_ntype_[tensor_pair_idx]->ptr());
    }

    auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
    if (FLAGS_graph_metapath_split_opt) {
      train_flag &= FillWalkBufMultiPath(
          h_device_keys_len_[tensor_pair_idx],
          gpu_graph_ptr->meta_path_[tensor_pair_idx],
          conf_,
          &epoch_finish_,
          &copy_unique_len_,
          place_,
          gpu_graph_ptr->first_node_type_[tensor_pair_idx],
          &(gpu_graph_ptr->node_type_start_[tensor_pair_idx][conf_.gpuid]),
          reinterpret_cast<uint64_t *>(d_walk_[tensor_pair_idx]->ptr()),
          walk_ntype,
          &d_uniq_node_num_,
          reinterpret_cast<int *>(d_random_row_[tensor_pair_idx]->ptr()),
          reinterpret_cast<int *>(
              d_random_row_col_shift_[tensor_pair_idx]->ptr()),
          &host_vec_,
          &total_row_[tensor_pair_idx],
          &jump_rows_[tensor_pair_idx],
          &shuffle_seed_[tensor_pair_idx],
          reinterpret_cast<uint64_t *>(
              d_train_metapath_keys_[tensor_pair_idx]->ptr()),
          &h_train_metapath_keys_len_[tensor_pair_idx],
          keys2rank_table_,
          &buf_state_[tensor_pair_idx],
          sample_stream_);
    } else {
      train_flag &= FillWalkBuf(
          h_device_keys_len_[tensor_pair_idx],
          d_device_keys_[tensor_pair_idx],
          gpu_graph_ptr->meta_path_[tensor_pair_idx],
          conf_,
          &epoch_finish_,
          &copy_unique_len_,
          place_,
          gpu_graph_ptr->first_node_type_[tensor_pair_idx],
          &(gpu_graph_ptr->node_type_start_[tensor_pair_idx][conf_.gpuid]),
          &(gpu_graph_ptr->finish_node_type_[tensor_pair_idx][conf_.gpuid]),
          reinterpret_cast<uint64_t *>(d_walk_[tensor_pair_idx]->ptr()),
          walk_ntype,
          &d_uniq_node_num_,
          reinterpret_cast<int *>(d_random_row_[tensor_pair_idx]->ptr()),
          reinterpret_cast<int *>(
              d_random_row_col_shift_[tensor_pair_idx]->ptr()),
          &multi_node_sync_stat_,
          &host_vec_,
          &host_ranks_,
          &total_row_[tensor_pair_idx],
          &jump_rows_[tensor_pair_idx],
          &shuffle_seed_[tensor_pair_idx],
          keys2rank_table_,
          &buf_state_[tensor_pair_idx],
          sample_stream_);
    }
  }

  return train_flag;
}

void GraphDataGenerator::DoSageForTrain() {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  int total_instance = 0, uniq_instance = 0;
  bool is_sage_pass_continue = true;
  int sage_pass_end = 0;
  uint64_t *ins_buf, *ins_cursor;
  bool not_empty_batch = 1;
  while (is_sage_pass_continue) {
    int fill_zero_num = 10;
    for (int tensor_pair_idx = 0;
         tensor_pair_idx < conf_.tensor_pair_num && is_sage_pass_continue;
         ++tensor_pair_idx) {
      while (ins_buf_pair_len_[tensor_pair_idx] < conf_.batch_size) {
        int32_t *pair_label_buf = NULL;
        if (conf_.enable_pair_label) {
          pair_label_buf = reinterpret_cast<int32_t *>(
              d_pair_label_buf_[tensor_pair_idx]->ptr());
        }
        int res = FillInsBuf(
            d_walk_[tensor_pair_idx],
            d_walk_ntype_[tensor_pair_idx],
            conf_,
            d_random_row_[tensor_pair_idx],
            d_random_row_col_shift_[tensor_pair_idx],
            &buf_state_[tensor_pair_idx],
            reinterpret_cast<uint64_t *>(d_ins_buf_[tensor_pair_idx]->ptr()),
            pair_label_buf,
            reinterpret_cast<int *>(d_pair_num_[tensor_pair_idx]->ptr()),
            &ins_buf_pair_len_[tensor_pair_idx],
            sample_stream_);

        if (res == -1) {
          if (ins_buf_pair_len_[tensor_pair_idx] == 0) {
            if (conf_.is_multi_node) {
              sage_pass_end = 1;
              if (total_row_[tensor_pair_idx] != 0) {
                buf_state_[tensor_pair_idx].Reset(total_row_[tensor_pair_idx]);
                VLOG(1) << "reset buf state to make batch num equal in "
                           "multi node";
              } else {
                VLOG(1) << conf_.gpuid << ": total row in buf state is 0";
                GraphZeroIdKernel<<<GET_BLOCKS(fill_zero_num),
                                    CUDA_NUM_THREADS,
                                    0,
                                    train_stream_>>>(
                    reinterpret_cast<uint64_t *>(
                        d_ins_buf_[tensor_pair_idx]->ptr()),
                    fill_zero_num);
                VLOG(1) << conf_.gpuid << ": end set seq ins";
                break;
              }
            } else {
              is_sage_pass_continue = false;
              break;
            }
          } else {
            break;
          }
        }
      }  // end while (ins_buf_pair_len_[tensor_pair_idx] < conf_.batch_size)

      // check whether reach sage pass end
      if (conf_.is_multi_node) {
        int res = multi_node_sync_sample(sage_pass_end,
                                         ncclProd,
                                         place_,
                                         conf_.gpuid,
                                         &multi_node_sync_stat_);
        VLOG(1) << conf_.gpuid << " get global sage_pass_end: " << res;
        if (res) {
          VLOG(1) << conf_.gpuid << ": reach sage pass end";
          is_sage_pass_continue = false;
          break;
        }
      }

      total_instance = ins_buf_pair_len_[tensor_pair_idx] < conf_.batch_size
                           ? ins_buf_pair_len_[tensor_pair_idx]
                           : conf_.batch_size;
      if (conf_.is_multi_node && total_row_[0] == 0) {
        total_instance = fill_zero_num;
        ins_buf_pair_len_[0] = fill_zero_num;
        VLOG(1) << "gpu id: " << conf_.gpuid
                << " set total ins num: " << fill_zero_num;
      }
      total_instance *= 2;

      if (total_instance == 0) {
        not_empty_batch = 0;
        break;
      }

      ins_buf =
          reinterpret_cast<uint64_t *>(d_ins_buf_[tensor_pair_idx]->ptr());
      ins_cursor =
          ins_buf + ins_buf_pair_len_[tensor_pair_idx] * 2 - total_instance;
      int mini_batch_size = total_instance / conf_.accumulate_num;
      if (conf_.accumulate_num >= 2) {
        if (mini_batch_size == 0) {
          ins_buf_pair_len_[tensor_pair_idx] -= total_instance / 2;
          not_empty_batch = 0;
          break;
        } else if (mini_batch_size % 2 == 1) {
          mini_batch_size += 1;
        }
        if (total_instance - mini_batch_size == 0) {
          ins_buf_pair_len_[tensor_pair_idx] -= total_instance / 2;
          not_empty_batch = 0;
          break;
        }
      }

      // fill first graph holder
      auto final_sage_nodes = GenerateSampleGraph(ins_cursor,
                                                  mini_batch_size,
                                                  &uniq_instance,
                                                  conf_,
                                                  &inverse_vec_,
                                                  &graph_edges_vec_,
                                                  &edges_split_num_vec_,
                                                  &edge_type_graph_,
                                                  &d_uniq_node_num_,
                                                  &copy_unique_len_,
                                                  &host_vec_,
                                                  &host_ranks_,
                                                  keys2rank_table_,
                                                  place_,
                                                  sample_stream_);
      final_sage_nodes_vec_.emplace_back(final_sage_nodes);
      uniq_instance_vec_.emplace_back(uniq_instance);
      total_instance_vec_.emplace_back(mini_batch_size);

      // fill second graph holder
      if (mini_batch_size != total_instance) {
        auto final_sage_nodes_v2 =
            GenerateSampleGraph(ins_cursor + mini_batch_size,
                                total_instance - mini_batch_size,
                                &uniq_instance,
                                conf_,
                                &inverse_vec_,
                                &graph_edges_vec_,
                                &edges_split_num_vec_,
                                &edge_type_graph_,
                                &d_uniq_node_num_,
                                &copy_unique_len_,
                                &host_vec_,
                                &host_ranks_,
                                keys2rank_table_,
                                place_,
                                sample_stream_);
        final_sage_nodes_vec_.emplace_back(final_sage_nodes_v2);
        uniq_instance_vec_.emplace_back(uniq_instance);
        total_instance_vec_.emplace_back(total_instance - mini_batch_size);
      }

      if (conf_.get_degree) {  // accumulate do not consider about degree
                               // currently.
        auto node_degrees =
            GetNodeDegree(reinterpret_cast<uint64_t *>(final_sage_nodes->ptr()),
                          uniq_instance,
                          conf_,
                          place_,
                          sample_stream_);
        node_degree_vec_.emplace_back(node_degrees);
      }

      if (conf_.enable_pair_label) {  // accumulate do not consider about pair
                                      // label currently.
        auto pair_label = memory::AllocShared(
            place_,
            total_instance / 2 * sizeof(int),
            phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
        int32_t *pair_label_buf = reinterpret_cast<int32_t *>(
            d_pair_label_buf_[tensor_pair_idx]->ptr());
        int32_t *pair_label_cursor = pair_label_buf +
                                     ins_buf_pair_len_[tensor_pair_idx] -
                                     total_instance / 2;
        cudaMemcpyAsync(pair_label->ptr(),
                        pair_label_cursor,
                        sizeof(int32_t) * total_instance / 2,
                        cudaMemcpyDeviceToDevice,
                        sample_stream_);
        pair_label_vec_.emplace_back(pair_label);
      }

      cudaStreamSynchronize(sample_stream_);

      ins_buf_pair_len_[tensor_pair_idx] -= total_instance / 2;
    }  // end for (int tensor_pair_idx = 0;
    if (is_sage_pass_continue && not_empty_batch) {
      sage_batch_num_ += 1;
    }
  }  // end while (is_sage_pass_continue)
  VLOG(1) << "gpuid: " << conf_.gpuid
          << " train_sage_batch_num: " << sage_batch_num_;
}

void GraphDataGenerator::DoSageForInfer() {
  // Set new batch size for multi_node
  platform::CUDADeviceGuard guard(conf_.gpuid);
  if (conf_.is_multi_node) {
    int new_batch_size = dynamic_adjust_batch_num_for_sage();
    conf_.batch_size = new_batch_size;
  }

  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    int total_instance = 0, uniq_instance = 0;
    int global_pass_end = 0;
    total_instance = (infer_node_start_[tensor_pair_idx] + conf_.batch_size <=
                      infer_node_end_[tensor_pair_idx])
                         ? conf_.batch_size
                         : infer_node_end_[tensor_pair_idx] -
                               infer_node_start_[tensor_pair_idx];
    total_instance *= 2;
    while (!global_pass_end) {
      int local_pass_end = total_instance == 0;
      if (conf_.is_multi_node) {
        global_pass_end = get_multi_node_global_flag(
            local_pass_end, ncclProd, place_, conf_.gpuid, sample_stream_);
      } else {
        global_pass_end = local_pass_end;
      }
      if (global_pass_end) {
        break;
      }
      uint64_t *d_type_keys = reinterpret_cast<uint64_t *>(
          d_device_keys_[tensor_pair_idx][infer_cursor_[tensor_pair_idx]]
              ->ptr());
      if (local_pass_end) {
        total_instance = 2;
        d_type_keys = reinterpret_cast<uint64_t *>(
            d_device_keys_[tensor_pair_idx][infer_cursor_[tensor_pair_idx]]
                ->ptr());  // copy from beginning
      } else {
        d_type_keys += infer_node_start_[tensor_pair_idx];
        infer_node_start_[tensor_pair_idx] += total_instance / 2;
      }
      auto node_buf = memory::AllocShared(
          place_,
          total_instance * sizeof(uint64_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
      int64_t *node_buf_ptr = reinterpret_cast<int64_t *>(node_buf->ptr());
      CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                          CUDA_NUM_THREADS,
                          0,
                          sample_stream_>>>(
          node_buf_ptr, d_type_keys, total_instance / 2);
      auto final_sage_nodes =
          GenerateSampleGraph(reinterpret_cast<uint64_t *>(node_buf->ptr()),
                              total_instance,
                              &uniq_instance,
                              conf_,
                              &inverse_vec_,
                              &graph_edges_vec_,
                              &edges_split_num_vec_,
                              &edge_type_graph_,
                              &d_uniq_node_num_,
                              &copy_unique_len_,
                              &host_vec_,
                              &host_ranks_,
                              keys2rank_table_,
                              place_,
                              sample_stream_);
      final_sage_nodes_vec_.emplace_back(final_sage_nodes);
      uniq_instance_vec_.emplace_back(uniq_instance);
      total_instance_vec_.emplace_back(total_instance);

      if (conf_.get_degree) {
        auto node_degrees =
            GetNodeDegree(reinterpret_cast<uint64_t *>(final_sage_nodes->ptr()),
                          uniq_instance,
                          conf_,
                          place_,
                          sample_stream_);
        node_degree_vec_.emplace_back(node_degrees);
      }
      cudaStreamSynchronize(sample_stream_);

      sage_batch_num_ += 1;

      total_instance = (infer_node_start_[tensor_pair_idx] + conf_.batch_size <=
                        infer_node_end_[tensor_pair_idx])
                           ? conf_.batch_size
                           : infer_node_end_[tensor_pair_idx] -
                                 infer_node_start_[tensor_pair_idx];
      total_instance *= 2;
    }  // end while (!global_pass_end)
  }    // end for (int tensor_pair_idx = 0; tensor_pair_idx <
       // conf_.tensor_pair_num;

  sage_batch_num_ /= conf_.tensor_pair_num;
  VLOG(1) << "gpuid: " << conf_.gpuid
          << " infer_sage_batch_num: " << sage_batch_num_;
}

void GraphDataGenerator::clear_gpu_mem() {
  platform::CUDADeviceGuard guard(conf_.gpuid);
}

int dynamic_adjust_total_row_for_infer(int local_reach_end,
                                       const phi::Place &place,
                                       cudaStream_t stream) {
  auto send_buff =
      memory::Alloc(place,
                    2 * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int *send_buff_ptr = reinterpret_cast<int *>(send_buff->ptr());
  cudaMemcpyAsync(send_buff_ptr,
                  &local_reach_end,
                  sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaStreamSynchronize(stream);
  auto comm = platform::NCCLCommContext::Instance().Get(0, place.GetDeviceId());
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(&send_buff_ptr[0],
                                                         &send_buff_ptr[1],
                                                         1,
                                                         ncclInt,
                                                         ncclProd,
                                                         comm->comm(),
                                                         stream));
  int global_reach_end = 0;
  cudaMemcpyAsync(&global_reach_end,
                  &send_buff_ptr[1],
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  return global_reach_end;
}

bool FillInferBuf(
    const std::vector<uint64_t> &h_device_keys_len,  // input
    const std::vector<std::shared_ptr<phi::Allocation>> &d_device_keys,
    const GraphDataGeneratorConfig &conf,
    const std::vector<int> &all_node_type,
    int tensor_pair_idx,
    int *total_row_ptr,
    size_t *infer_node_start_ptr,
    size_t *infer_node_end_ptr,
    size_t *infer_cursor_ptr,
    std::vector<uint64_t> *host_vec_ptr,    // output
    std::vector<uint32_t> *host_ranks_ptr,  // output
    std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank_table,
    const std::shared_ptr<phi::Allocation> &d_uniq_node_num,
    const phi::Place &place,
    cudaStream_t stream) {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto &global_infer_node_type_start =
      gpu_graph_ptr->global_infer_node_type_start_[tensor_pair_idx][conf.gpuid];
  auto &infer_cursor =
      gpu_graph_ptr->infer_cursor_[tensor_pair_idx][conf.thread_id];
  *total_row_ptr = 0;

  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  std::set<int> all_node_type_index_set;
  for (auto &node_type : all_node_type) {
    all_node_type_index_set.insert(type_to_index[node_type]);
    VLOG(1) << "add all_node_type_index_set: "
            << gpu_graph_ptr->node_types_idx_to_node_type_str(node_type);
  }

  if (infer_cursor < h_device_keys_len.size()) {
    while (global_infer_node_type_start[infer_cursor] >=
           h_device_keys_len[infer_cursor]) {
      infer_cursor++;
      if (infer_cursor >= h_device_keys_len.size()) {
        return false;
      }
    }
    while (infer_cursor < h_device_keys_len.size()) {
      if (all_node_type_index_set.find(infer_cursor) ==
          all_node_type_index_set.end()) {
        VLOG(2) << "Skip cursor[" << infer_cursor
                << "], by all_node_type_index_set";
        infer_cursor++;
        if (infer_cursor >= h_device_keys_len.size()) {
          return false;
        }
        continue;
      } else {
        VLOG(2) << "Not skip cursor[" << infer_cursor << "]";
        break;
      }
    }
    if (!conf.infer_node_type_index_set.empty()) {
      while (infer_cursor < h_device_keys_len.size()) {
        if (conf.infer_node_type_index_set.find(infer_cursor) ==
            conf.infer_node_type_index_set.end()) {
          VLOG(2) << "Skip cursor[" << infer_cursor
                  << "], by infer_node_type_index_set";
          infer_cursor++;
          continue;
        } else {
          VLOG(2) << "Not skip cursor[" << infer_cursor << "]";
          break;
        }
      }
      if (infer_cursor >= h_device_keys_len.size()) {
        return false;
      }
    }

    size_t device_key_size = h_device_keys_len[infer_cursor];
    if (conf.is_multi_node || conf.is_thread_sharding) {
      int local_reach_end =
          global_infer_node_type_start[infer_cursor] + conf.buf_size >=
          device_key_size;
      int global_reach_end = get_multi_node_global_flag(
          local_reach_end, ncclProd, place, conf.gpuid, stream);
      int remain = device_key_size - global_infer_node_type_start[infer_cursor];
      if (global_reach_end) {
        *total_row_ptr = remain;
      } else {
        if (local_reach_end) {
          *total_row_ptr = remain / 2;
        } else {
          *total_row_ptr = conf.buf_size;
        }
      }
    } else {
      *total_row_ptr =
          (global_infer_node_type_start[infer_cursor] + conf.buf_size <=
           device_key_size)
              ? conf.buf_size
              : device_key_size - global_infer_node_type_start[infer_cursor];
    }

    platform::CUDADeviceGuard guard(conf.gpuid);
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys[infer_cursor]->ptr());
    if (!conf.sage_mode) {
      host_vec_ptr->resize(*total_row_ptr);
      cudaMemcpyAsync(host_vec_ptr->data(),
                      d_type_keys + global_infer_node_type_start[infer_cursor],
                      sizeof(uint64_t) * *total_row_ptr,
                      cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
    }
    VLOG(1) << "gpuid: " << conf.gpuid << " cursor: " << infer_cursor
            << " start: " << global_infer_node_type_start[infer_cursor]
            << " num: " << *total_row_ptr
            << " device_key_size: " << device_key_size;
    *infer_node_start_ptr = global_infer_node_type_start[infer_cursor];
    global_infer_node_type_start[infer_cursor] += *total_row_ptr;
    *infer_node_end_ptr = global_infer_node_type_start[infer_cursor];
    *infer_cursor_ptr = infer_cursor;
    return true;
  }

  return false;
}

bool GraphDataGenerator::DoWalkForInfer() {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  bool infer_flag = true;
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    infer_flag &= FillInferBuf(h_device_keys_len_[tensor_pair_idx],
                               d_device_keys_[tensor_pair_idx],
                               conf_,
                               gpu_graph_ptr->all_node_type_,
                               tensor_pair_idx,
                               &total_row_[tensor_pair_idx],
                               &infer_node_start_[tensor_pair_idx],
                               &infer_node_end_[tensor_pair_idx],
                               &infer_cursor_[tensor_pair_idx],
                               &host_vec_,
                               &host_ranks_,
                               keys2rank_table_,
                               d_uniq_node_num_,
                               place_,
                               sample_stream_);
    VLOG(1) << "aft FillInferBuf, total_row[" << tensor_pair_idx
            << "] = " << total_row_[tensor_pair_idx];
    cudaStreamSynchronize(sample_stream_);
  }

  return infer_flag;
}

void GraphDataGenerator::ClearSampleState() {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    auto &finish_node_type =
        gpu_graph_ptr->finish_node_type_[tensor_pair_idx][conf_.gpuid];
    auto &node_type_start =
        gpu_graph_ptr->node_type_start_[tensor_pair_idx][conf_.gpuid];
    finish_node_type.clear();
    for (auto iter = node_type_start.begin(); iter != node_type_start.end();
         iter++) {
      iter->second = 0;
    }
  }
}

void GraphDataGenerator::SetFeedVec(std::vector<phi::DenseTensor *> feed_vec) {
  feed_vec_ = feed_vec;
}

void GraphDataGenerator::SetFeedInfo(std::vector<UsedSlotInfo> *feed_info) {
  feed_info_ = feed_info;
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    int feed_vec_idx = 2 + tensor_pair_idx * conf_.tensor_num_of_one_pair;
    ++feed_vec_idx;
    if (conf_.enable_pair_label) {
      ++feed_vec_idx;
    }
    for (int i = 0; i < conf_.slot_num; i++) {
      int offset = feed_vec_idx + 2 * i;
      if ((*feed_info_)[offset].type[0] == 'f') {  // float feature
        // if (first_float_idx_ == -1) {
        //  first_float_idx_ = i;
        // }
        float_slot_num_++;
      } else if ((*feed_info_)[offset].type[0] == 'u') {  // slot feature
        // if (first_slot_idx_ == -1) {
        //   first_slot_idx_ = i;
        // }
        uint_slot_num_++;
      }
    }
  }
}

void GraphDataGenerator::AllocResource(
    int thread_id, std::vector<phi::DenseTensor *> feed_vec) {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  conf_.gpuid = gpu_graph_ptr->device_id_mapping[thread_id];
  conf_.thread_id = thread_id;
  place_ = phi::GPUPlace(conf_.gpuid);
  debug_gpu_memory_info(conf_.gpuid, "AllocResource start");

  platform::CUDADeviceGuard guard(conf_.gpuid);
  sample_stream_ = gpu_graph_ptr->get_local_stream(conf_.gpuid);
  train_stream_ = dynamic_cast<phi::GPUContext *>(
                      phi::DeviceContextPool::Instance().Get(place_))
                      ->stream();
  if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
    if (conf_.gpu_graph_training) {
      keys2rank_table_ = std::shared_ptr<HashTable<uint64_t, uint32_t>>(
          new HashTable<uint64_t, uint32_t>(
              conf_.train_table_cap / FLAGS_gpugraph_hbm_table_load_factor,
              sample_stream_));
    } else {
      keys2rank_table_ = std::shared_ptr<HashTable<uint64_t, uint32_t>>(
          new HashTable<uint64_t, uint32_t>(
              conf_.infer_table_cap / FLAGS_gpugraph_hbm_table_load_factor,
              sample_stream_));
    }
  }
  VLOG(1) << "AllocResource gpuid " << conf_.gpuid
          << " feed_vec.size: " << feed_vec.size()
          << " table cap: " << conf_.train_table_cap;
  conf_.is_multi_node = false;
#if defined(PADDLE_WITH_GLOO)
  auto gloo = paddle::framework::GlooWrapper::GetInstance();
  if (gloo->Size() > 1) {
    conf_.is_multi_node = true;
  }
#endif

  // infer_node_type_start_ = std::vector<int>(h_device_keys_.size(), 0);
  // for (size_t i = 0; i < h_device_keys_.size(); i++) {
  //   for (size_t j = 0; j < h_device_keys_[i]->size(); j++) {
  //     VLOG(3) << "h_device_keys_[" << i << "][" << j
  //             << "] = " << (*(h_device_keys_[i]))[j];
  //   }
  //   auto buf = memory::AllocShared(
  //       place_, h_device_keys_[i]->size() * sizeof(uint64_t));
  //   d_device_keys_.push_back(buf);
  //   CUDA_CHECK(cudaMemcpyAsync(buf->ptr(),
  //                              h_device_keys_[i]->data(),
  //                              h_device_keys_[i]->size() * sizeof(uint64_t),
  //                              cudaMemcpyHostToDevice,
  //                              stream_));
  // }
  if (conf_.gpu_graph_training && FLAGS_graph_metapath_split_opt) {
    d_train_metapath_keys_.resize(conf_.tensor_pair_num);
    h_train_metapath_keys_len_.resize(conf_.tensor_pair_num);
    for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
         ++tensor_pair_idx) {
      d_train_metapath_keys_[tensor_pair_idx] =
          gpu_graph_ptr->d_node_iter_graph_metapath_keys_[thread_id];
      h_train_metapath_keys_len_[tensor_pair_idx] =
          gpu_graph_ptr->h_node_iter_graph_metapath_keys_len_[thread_id];
      VLOG(2) << "h train metapaths key len: "
              << h_train_metapath_keys_len_[tensor_pair_idx];
    }
  } else {
    d_device_keys_.resize(conf_.tensor_pair_num);
    h_device_keys_len_.resize(conf_.tensor_pair_num);
    for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
         ++tensor_pair_idx) {
      auto &d_graph_all_type_keys =
          gpu_graph_ptr->d_node_iter_graph_all_type_keys_;
      auto &h_graph_all_type_keys_len =
          gpu_graph_ptr->h_node_iter_graph_all_type_keys_len_;

      for (size_t i = 0; i < d_graph_all_type_keys.size(); i++) {
        d_device_keys_[tensor_pair_idx].push_back(
            d_graph_all_type_keys[i][thread_id]);
        h_device_keys_len_[tensor_pair_idx].push_back(
            h_graph_all_type_keys_len[i][thread_id]);
      }
      VLOG(2) << "h_device_keys size: "
              << h_device_keys_len_[tensor_pair_idx].size();
    }
  }

  infer_cursor_.assign(conf_.tensor_pair_num, 0);
  jump_rows_.assign(conf_.tensor_pair_num, 0);
  d_uniq_node_num_ = memory::AllocShared(
      place_,
      sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  cudaMemsetAsync(d_uniq_node_num_->ptr(), 0, sizeof(uint64_t), sample_stream_);

  total_row_.assign(conf_.tensor_pair_num, 0);
  infer_node_start_.assign(conf_.tensor_pair_num, 0);
  infer_node_end_.assign(conf_.tensor_pair_num, 0);
  d_walk_.resize(conf_.tensor_pair_num);
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    d_walk_[tensor_pair_idx] = memory::AllocShared(
        place_,
        conf_.buf_size * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    cudaMemsetAsync(d_walk_[tensor_pair_idx]->ptr(),
                    0,
                    conf_.buf_size * sizeof(uint64_t),
                    sample_stream_);
  }

  conf_.excluded_train_pair_len = gpu_graph_ptr->excluded_train_pair_.size();
  if (conf_.excluded_train_pair_len > 0) {
    conf_.d_excluded_train_pair = memory::AllocShared(
        place_,
        conf_.excluded_train_pair_len * sizeof(uint8_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    CUDA_CHECK(cudaMemcpyAsync(conf_.d_excluded_train_pair->ptr(),
                               gpu_graph_ptr->excluded_train_pair_.data(),
                               conf_.excluded_train_pair_len * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               sample_stream_));
  }

  for (int i = -conf_.window; i < 0; i++) {
    conf_.window_step.push_back(i);
  }
  for (int i = 0; i < conf_.window; i++) {
    conf_.window_step.push_back(i + 1);
  }
  buf_state_.resize(conf_.tensor_pair_num);
  d_random_row_.resize(conf_.tensor_pair_num);
  d_random_row_col_shift_.resize(conf_.tensor_pair_num);
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    buf_state_[tensor_pair_idx].Init(
        conf_.batch_size, conf_.walk_len, &conf_.window_step);
    d_random_row_[tensor_pair_idx] = memory::AllocShared(
        place_,
        (conf_.once_sample_startid_len * conf_.walk_degree * repeat_time_) *
            sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

    d_random_row_col_shift_[tensor_pair_idx] = memory::AllocShared(
        place_,
        (conf_.once_sample_startid_len * conf_.walk_degree * repeat_time_) *
            sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  }

  shuffle_seed_.assign(conf_.tensor_pair_num, 0);
  ins_buf_pair_len_.assign(conf_.tensor_pair_num, 0);
  d_ins_buf_.resize(conf_.tensor_pair_num);
  d_pair_num_.resize(conf_.tensor_pair_num);
  for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
       ++tensor_pair_idx) {
    d_ins_buf_[tensor_pair_idx] = memory::AllocShared(
        place_,
        (conf_.batch_size * 2 * 2) * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_pair_num_[tensor_pair_idx] = memory::AllocShared(
        place_,
        sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  }
  conf_.enable_pair_label =
      conf_.gpu_graph_training && gpu_graph_ptr->pair_label_conf_.size() > 0;
  if (conf_.enable_pair_label) {
    conf_.node_type_num = gpu_graph_ptr->id_to_feature.size();
    d_pair_label_buf_.resize(conf_.tensor_pair_num);
    for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
         ++tensor_pair_idx) {
      d_pair_label_buf_[tensor_pair_idx] = memory::AllocShared(
          place_,
          (conf_.batch_size * 2) * sizeof(int32_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    }
    conf_.d_pair_label_conf = memory::AllocShared(
        place_,
        conf_.node_type_num * conf_.node_type_num * sizeof(int32_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    CUDA_CHECK(cudaMemcpyAsync(
        conf_.d_pair_label_conf->ptr(),
        gpu_graph_ptr->pair_label_conf_.data(),
        conf_.node_type_num * conf_.node_type_num * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        sample_stream_));
    id_offset_of_feed_vec_ = 4;
  } else {
    id_offset_of_feed_vec_ = 3;
  }

  conf_.need_walk_ntype =
      conf_.excluded_train_pair_len > 0 || conf_.enable_pair_label;
  if (conf_.need_walk_ntype) {
    d_walk_ntype_.resize(conf_.tensor_pair_num);
    for (int tensor_pair_idx = 0; tensor_pair_idx < conf_.tensor_pair_num;
         ++tensor_pair_idx) {
      d_walk_ntype_[tensor_pair_idx] = memory::AllocShared(
          place_,
          conf_.buf_size * sizeof(uint8_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
      cudaMemsetAsync(d_walk_ntype_[tensor_pair_idx]->ptr(),
                      0,
                      conf_.buf_size * sizeof(uint8_t),
                      sample_stream_);
    }
  }

  if (!conf_.sage_mode) {
    conf_.slot_num = (feed_vec.size() - id_offset_of_feed_vec_) / 2;
  } else {
    VLOG(1) << "feed_vec.size(): " << feed_vec.size();
    conf_.tensor_num_of_one_pair =
        (feed_vec.size() - 2) / conf_.tensor_pair_num;  // 2 means show and clk
    assert((conf_.tensor_num_of_one_pair * conf_.tensor_pair_num + 2) ==
           feed_vec.size());
    VLOG(1) << "feed_vec.size(): " << feed_vec.size()
            << " tensor_num_of_one_pair: " << conf_.tensor_num_of_one_pair
            << " tensor_pair_num: " << conf_.tensor_pair_num;
    uint32_t tensor_num_of_one_sample = 5;
    if (conf_.return_weight) {
      tensor_num_of_one_sample++;
    }

    conf_.tensor_num_of_one_subgraph =
        tensor_num_of_one_sample * conf_.samples.size();
    conf_.tensor_num_of_one_subgraph++;  // final_index
    if (conf_.get_degree) {
      conf_.tensor_num_of_one_subgraph++;  // degree_norm
    }
    VLOG(1) << "conf_.tensor_num_of_one_sample: " << tensor_num_of_one_sample
            << " conf_.sample.size(): " << conf_.samples.size()
            << " conf_.tensor_num_of_one_subgraph: "
            << conf_.tensor_num_of_one_subgraph
            << " conf_.accumulate_num: " << conf_.accumulate_num;

    if (conf_.accumulate_num == 1) {
      conf_.slot_num = (conf_.tensor_num_of_one_pair - 1 -
                        conf_.tensor_num_of_one_subgraph) /
                       2;
      assert((1 + conf_.slot_num * 2 + conf_.tensor_num_of_one_subgraph) ==
             conf_.tensor_num_of_one_pair);

      VLOG(1) << "get conf_.slot_num: " << conf_.slot_num;
    } else {
      conf_.slot_num =
          (feed_vec.size() - 2 * conf_.tensor_num_of_one_subgraph - 6) / 2;
    }
    VLOG(1) << "tensor_num_of_one_pair[" << conf_.tensor_num_of_one_pair
            << "] tensor_num_of_one_sample[" << tensor_num_of_one_sample
            << "] tensor_num_of_one_subgraph["
            << conf_.tensor_num_of_one_subgraph << "]";
  }
  VLOG(1) << "slot_num[" << conf_.slot_num << "]";
  conf_.tensor_num_of_one_pair = 1 + conf_.slot_num * 2;  // id and slot
  if (conf_.sage_mode) {
    if (conf_.accumulate_num >= 2) {
      conf_.tensor_num_of_one_pair += 2 * (5 * conf_.samples.size() + 1);
    } else {
      conf_.tensor_num_of_one_pair += 5 * conf_.samples.size() + 1;
    }
  }
  if (conf_.enable_pair_label) {
    conf_.tensor_num_of_one_pair++;
  }
  if (conf_.get_degree) {
    conf_.tensor_num_of_one_pair++;
  }
  if (conf_.return_weight) {
    if (conf_.accumulate_num >= 2) {
      conf_.tensor_num_of_one_pair += 2 * conf_.samples.size();
    } else {
      conf_.tensor_num_of_one_pair += conf_.samples.size();
    }
  }
  VLOG(1) << "tensor_num_of_one_pair[" << conf_.tensor_num_of_one_pair << "]";

  d_slot_tensor_ptr_ = memory::AllocShared(
      place_,
      conf_.slot_num * sizeof(uint64_t *),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  d_slot_lod_tensor_ptr_ = memory::AllocShared(
      place_,
      conf_.slot_num * sizeof(uint64_t *),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

  if (conf_.sage_mode) {
    conf_.reindex_table_size = conf_.batch_size * 2;
    // get hashtable size
    for (int i = 0; i < conf_.samples.size(); i++) {
      conf_.reindex_table_size *= (conf_.samples[i] * conf_.edge_to_id_len + 1);
    }
    int64_t next_pow2 =
        1 << static_cast<size_t>(1 + std::log2(conf_.reindex_table_size >> 1));
    conf_.reindex_table_size = next_pow2 << 1;

    edge_type_graph_ =
        gpu_graph_ptr->get_edge_type_graph(conf_.gpuid, conf_.edge_to_id_len);
  }

  // parse infer_node_type
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  if (!conf_.gpu_graph_training) {
    auto node_types =
        paddle::string::split_string<std::string>(infer_node_type_, ";");
    auto node_to_id = gpu_graph_ptr->node_to_id;
    for (auto &type : node_types) {
      auto iter = node_to_id.find(type);
      PADDLE_ENFORCE_NE(
          iter,
          node_to_id.end(),
          common::errors::NotFound("(%s) is not found in node_to_id.", type));
      int node_type = iter->second;
      int type_index = type_to_index[node_type];
      VLOG(2) << "add node[" << type
              << "] into infer_node_type, type_index(cursor)[" << type_index
              << "]";
      conf_.infer_node_type_index_set.insert(type_index);
    }
    VLOG(2) << "infer_node_type_index_set_num: "
            << conf_.infer_node_type_index_set.size();
  }

  int *stat_ptr =
      multi_node_sync_stat_.mutable_data<int>(place_, sizeof(int) * 4);
  int flags[4] = {0, 1, 2, 0};
  cudaMemcpyAsync(stat_ptr,  // output
                  &flags,
                  sizeof(int) * 4,
                  cudaMemcpyHostToDevice,
                  sample_stream_);
  cudaStreamSynchronize(sample_stream_);

  debug_gpu_memory_info(conf_.gpuid, "AllocResource end");
}

void GraphDataGenerator::AllocTrainResource(int thread_id) {
  if (conf_.slot_num > 0) {
    platform::CUDADeviceGuard guard(conf_.gpuid);
    d_feature_size_list_buf_ = NULL;
    d_feature_size_prefixsum_buf_ = NULL;
  }
}

void GraphDataGenerator::SetConfig(
    const paddle::framework::DataFeedDesc &data_feed_desc) {
  auto graph_config = data_feed_desc.graph_config();
  conf_.walk_degree = graph_config.walk_degree();
  conf_.walk_len = graph_config.walk_len();
  conf_.window = graph_config.window();
  conf_.once_sample_startid_len = graph_config.once_sample_startid_len();
  conf_.debug_mode = graph_config.debug_mode();
  conf_.gpu_graph_training = graph_config.gpu_graph_training();
  if (conf_.debug_mode || !conf_.gpu_graph_training) {
    conf_.batch_size = graph_config.batch_size();
  } else {
    conf_.batch_size = conf_.once_sample_startid_len;
  }
  repeat_time_ = graph_config.sample_times_one_chunk();
  conf_.buf_size = conf_.once_sample_startid_len * conf_.walk_len *
                   conf_.walk_degree * repeat_time_;
  conf_.train_table_cap = graph_config.train_table_cap();
  conf_.infer_table_cap = graph_config.infer_table_cap();
  conf_.get_degree = graph_config.get_degree();
  conf_.weighted_sample = graph_config.weighted_sample();
  conf_.return_weight = graph_config.return_weight();
  conf_.accumulate_num = graph_config.accumulate_num();

  epoch_finish_ = false;
  VLOG(1) << "Confirm GraphConfig, walk_degree : " << conf_.walk_degree
          << ", walk_len : " << conf_.walk_len << ", window : " << conf_.window
          << ", once_sample_startid_len : " << conf_.once_sample_startid_len
          << ", sample_times_one_chunk : " << repeat_time_
          << ", batch_size: " << conf_.batch_size
          << ", train_table_cap: " << conf_.train_table_cap
          << ", infer_table_cap: " << conf_.infer_table_cap;
  std::string first_node_type = graph_config.first_node_type();
  std::string meta_path = graph_config.meta_path();
  conf_.is_thread_sharding = graph_config.is_thread_sharding();
  conf_.sage_mode = graph_config.sage_mode();
  std::string str_samples = graph_config.samples();
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  debug_gpu_memory_info("init_conf start");
  gpu_graph_ptr->init_conf(first_node_type,
                           meta_path,
                           graph_config.excluded_train_pair(),
                           graph_config.pair_label());
  conf_.tensor_pair_num = gpu_graph_ptr->tensor_pair_num_;
  debug_gpu_memory_info("init_conf end");

  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  conf_.edge_to_id_len = edge_to_id.size();
  sage_batch_count_ = 0;
  auto samples = paddle::string::split_string<std::string>(str_samples, ";");
  for (size_t i = 0; i < samples.size(); i++) {
    int sample_size = std::stoi(samples[i]);
    conf_.samples.emplace_back(sample_size);
  }
  copy_unique_len_ = 0;

  if (!conf_.gpu_graph_training) {
    infer_node_type_ = graph_config.infer_node_type();
  }

  conf_.once_max_sample_keynum =
      conf_.walk_degree * conf_.once_sample_startid_len;
}
#endif

void GraphDataGenerator::DumpWalkPath(std::string dump_path, size_t dump_rate) {
#ifdef _LINUX
  PADDLE_ENFORCE_LE(
      dump_rate,
      10000000,
      common::errors::InvalidArgument(
          "dump_rate can't be large than 10000000. Please check the dump "
          "rate[1, 10000000]"));
  PADDLE_ENFORCE_GE(dump_rate,
                    1,
                    common::errors::InvalidArgument(
                        "dump_rate can't be less than 1. Please check the dump "
                        "rate[1, 10000000]"));
  int err_no = 0;
  std::shared_ptr<FILE> fp = fs_open_append_write(dump_path, &err_no, "");
  platform::CUDADeviceGuard guard(conf_.gpuid);
  uint64_t *h_walk = new uint64_t[conf_.buf_size];
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_[0]->ptr());
  cudaMemcpy(
      h_walk, walk, conf_.buf_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  VLOG(1) << "DumpWalkPath all conf_.buf_size:" << conf_.buf_size;
  std::string ss = "";
  size_t write_count = 0;
  for (int xx = 0; xx < conf_.buf_size / dump_rate; xx += conf_.walk_len) {
    ss = "";
    for (int yy = 0; yy < conf_.walk_len; yy++) {
      ss += std::to_string(h_walk[xx + yy]) + "-";
    }
    write_count = fwrite_unlocked(ss.data(), 1, ss.length(), fp.get());
    if (write_count != ss.length()) {
      VLOG(1) << "dump walk path" << ss << " failed";
    }
    write_count = fwrite_unlocked("\n", 1, 1, fp.get());
  }
  delete[] h_walk;
#endif
}

void GraphDataGenerator::DumpSampleNeighbors(std::string dump_path) {
#ifdef _LINUX
  int err_no = 0;
  int part_num = rand() % 100;  // NOLINT // set 100 part files
  std::string part_path =
      string::format_string("%s-%03d", dump_path.c_str(), part_num);
  std::shared_ptr<FILE> fp = fs_open_append_write(part_path, &err_no, "");
  platform::CUDADeviceGuard guard(conf_.gpuid);
  for (int i = 0; i < sage_batch_num_; i++) {
    int uniq_instance = uniq_instance_vec_[i];
    uint64_t *h_id_tensor = new uint64_t[uniq_instance];
    cudaMemcpy(h_id_tensor,
               final_sage_nodes_vec_[i]->ptr(),
               sizeof(uint64_t) * uniq_instance,
               cudaMemcpyDeviceToHost);
    std::string ss = "id:";
    for (int xx = 0; xx < uniq_instance; xx++) {
      ss += std::to_string(h_id_tensor[xx]) + ",";
    }
    ss += "\t";
    int len_samples = conf_.samples.size();
    std::vector<std::vector<int>> edges_split_num_for_graph =
        edges_split_num_vec_[i];
    std::vector<std::shared_ptr<phi::Allocation>> graph_edges =
        graph_edges_vec_[i];
    int graph_edges_index = 0;
    for (int j = 0; j < len_samples; j++) {
      ss += std::to_string(j) + ":[";
      std::vector<int> edges_split_num = edges_split_num_for_graph[j];
      int neighbor_len = edges_split_num[conf_.edge_to_id_len + 2];
      int64_t *h_edge_src_tensor = new int64_t[neighbor_len];
      int64_t *h_edge_dst_tensor = new int64_t[neighbor_len];
      cudaMemcpy(h_edge_src_tensor,
                 graph_edges[graph_edges_index++]->ptr(),
                 sizeof(int64_t) * neighbor_len,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(h_edge_dst_tensor,
                 graph_edges[graph_edges_index++]->ptr(),
                 sizeof(int64_t) * neighbor_len,
                 cudaMemcpyDeviceToHost);
      ss += "src:";
      for (int yy = 0; yy < neighbor_len; yy++) {
        ss += std::to_string(h_edge_src_tensor[yy]) + ",";
      }
      ss += "\tdst:";
      for (int yy = 0; yy < neighbor_len; yy++) {
        ss += std::to_string(h_edge_dst_tensor[yy]) + ",";
      }
      ss += "\tsplit:";
      for (int yy = 0; yy < conf_.edge_to_id_len; yy++) {
        ss += std::to_string(edges_split_num[yy]) + ",";
      }
      ss += "]\t";

      delete[] h_edge_src_tensor;
      delete[] h_edge_dst_tensor;
    }
    size_t write_count = fwrite_unlocked(ss.data(), 1, ss.length(), fp.get());
    if (write_count != ss.length()) {
      VLOG(1) << "dump sample neighbors: " << ss << " failed!";
    }
    write_count = fwrite_unlocked("\n", 1, 1, fp.get());
    delete[] h_id_tensor;
  }
#endif
}

int GraphDataGenerator::dynamic_adjust_batch_num_for_sage() {
  platform::CUDADeviceGuard guard(conf_.gpuid);
  int batch_num = (total_row_[0] + conf_.batch_size - 1) / conf_.batch_size;
  auto send_buff = memory::Alloc(
      place_,
      2 * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int *send_buff_ptr = reinterpret_cast<int *>(send_buff->ptr());
  cudaMemcpyAsync(send_buff_ptr,
                  &batch_num,
                  sizeof(int),
                  cudaMemcpyHostToDevice,
                  sample_stream_);
  cudaStreamSynchronize(sample_stream_);
  auto comm =
      platform::NCCLCommContext::Instance().Get(0, place_.GetDeviceId());
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(&send_buff_ptr[0],
                                                         &send_buff_ptr[1],
                                                         1,
                                                         ncclInt,
                                                         ncclMax,
                                                         comm->comm(),
                                                         sample_stream_));
  int thread_max_batch_num = 0;
  cudaMemcpyAsync(&thread_max_batch_num,
                  &send_buff_ptr[1],
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  sample_stream_);
  cudaStreamSynchronize(sample_stream_);

  int new_batch_size =
      (total_row_[0] + thread_max_batch_num - 1) / thread_max_batch_num;
  VLOG(1) << "dynamic adjust sage batch num "
          << " max_batch_num: " << thread_max_batch_num
          << " new_batch_size:  " << new_batch_size;
  return new_batch_size;
}

}  // namespace framework
}  // namespace paddle
#endif
