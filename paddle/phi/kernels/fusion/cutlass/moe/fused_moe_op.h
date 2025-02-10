/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#ifndef _FUSED_MOE_OP_H_
#define _FUSED_MOE_OP_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_imp_op.h"
#include "paddle/phi/kernels/impl/llm_int8_matmul_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"

#include "cutlass/numeric_conversion.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/platform/device/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_helper.h"
// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "paddle/phi/backends/gpu/gpu_info.h"
#pragma GCC diagnostic pop

namespace phi {

inline GpuLaunchConfig Get1DBlocksAnd2DGridsMoe(const int64_t cols) {
  int blocks_x = cols;
  int blocks_y = 1;
  int blocks_z = 1;
  if (blocks_x > 1024) {
    blocks_y = 256;
    blocks_x = (blocks_x + blocks_y - 1) / blocks_y;
  }

  GpuLaunchConfig config;
  config.block_per_grid.x = blocks_x;
  config.block_per_grid.y = blocks_y;
  config.block_per_grid.z = blocks_z;
  return config;
}

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <typename T, int TPB>
__launch_bounds__(TPB) __global__
    void group_moe_softmax(const T* input,
                           const bool* finished,
                           T* output,
                           T* softmax_max_prob,
                           const int64_t num_cols,
                           const int64_t softmax_num_rows) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;
  __shared__ float max_out;

  int globalIdx = blockIdx.x + blockIdx.y * gridDim.x;
  if (globalIdx >= softmax_num_rows) {
    return;
  }
  const int64_t thread_row_offset = globalIdx * num_cols;

  cub::Sum sum;
  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  // if ((finished != nullptr) && finished[blockIdx.x]) {
  //     return;
  // }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = max(static_cast<float>(input[idx]), threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((static_cast<float>(input[idx]) - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val =
        exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = T(val);
    threadData = max(static_cast<float>(T(val)), threadData);
  }

  const float maxOut = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    // group max probs
    max_out = 1.f / maxOut;
    softmax_max_prob[globalIdx] = T(max_out);
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    // group softmax normalization
    output[idx] = output[idx] * static_cast<T>(max_out);
  }
}

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T* inputs_after_softmax,
                                                 const bool* finished,
                                                 T* output,
                                                 int* indices,
                                                 int* source_rows,
                                                 T* softmax_max_prob,
                                                 const int64_t num_experts,
                                                 const int64_t k,
                                                 const int64_t num_rows) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x + blockIdx.y * gridDim.x;
  if (block_row >= num_rows) {
    return;
  }

  const bool should_process_row = finished ? !finished[block_row] : true;
  const int thread_read_offset = block_row * num_experts;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      // restore normalized probes
      output[idx] = result_kvp.value / T(softmax_max_prob[idx]);
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();
  }
}

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_softmax(const T* input,
                                                   const bool* finished,
                                                   T* output,
                                                   const int64_t num_cols,
                                                   const int64_t num_rows) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  int globalIdx = blockIdx.x + blockIdx.y * gridDim.x;
  if (globalIdx >= num_rows) {
    return;
  }
  const int64_t thread_row_offset = globalIdx * num_cols;

  cub::Sum sum;
  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[globalIdx]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = max(static_cast<float>(input[idx]), threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((static_cast<float>(input[idx]) - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val =
        exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = T(val);
  }
}

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T* inputs_after_softmax,
                                                 const bool* finished,
                                                 T* output,
                                                 int* indices,
                                                 int* source_rows,
                                                 const int64_t num_experts,
                                                 const int64_t k,
                                                 const int64_t num_rows) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x + blockIdx.y * gridDim.x;
  if (block_row >= num_rows) {
    return;
  }

  const bool should_process_row = finished ? !finished[block_row] : true;
  const int thread_read_offset = block_row * num_experts;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      output[idx] = result_kvp.value;
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();
  }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k.
*/

template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk_gating_softmax(const T* input,
                             const bool* finished,
                             T* output,
                             const int64_t num_rows,
                             int* indices,
                             int* source_rows,
                             const int64_t k) {
  // We begin by enforcing compile time assertions and setting up compile time
  // constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  // Restrictions based on previous section.
  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time
  // variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a
  // block contains WARPS_PER_CTA warps. This, each block processes a chunk of
  // rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows) return;
  const bool should_process_row = finished ? !finished[thread_row] : true;

  // We finally start setting up the read pointers for each thread. First, each
  // thread jumps to the start of the row it will read.
  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the
  // first column to start loads.
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Determine the pointer type to use to read in the data depending on the
  // BYTES_PER_LDG template param. In theory, this can support all powers of 2
  // up to 16.
  using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;

  // Finally, we pull in the data from global mem
  cutlass::Array<T, VPT> row_chunk_input;
  AccessType* row_chunk_vec_ptr =
      reinterpret_cast<AccessType*>(&row_chunk_input);
  const AccessType* vec_thread_read_ptr =
      reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  using ComputeType = float;
  using Converter = cutlass::NumericArrayConverter<ComputeType, T, VPT>;
  Converter compute_type_converter;
  cutlass::Array<ComputeType, VPT> row_chunk =
      compute_type_converter(row_chunk_input);

  // First, we perform a max reduce within the thread. We can do the max in fp16
  // safely (I think) and just convert to float afterwards for the exp + sum
  // reduction.
  ComputeType thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max =
        max(thread_max,
            __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp.
  // We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
  }

  // From this point, all threads have the max and the sum for their rows in the
  // thread_max and thread_sum variables respectively. Finally, we can scale the
  // rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only
  // compute for the top-k values in the row. However, this kernel will likely
  // not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }

  // Now, softmax_res contains the softmax of the row chunk. Now, I want to find
  // the topk elements in each row, along with the max index.​
  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];

        // No check on the experts here since columns with the smallest index
        // are processed first and only updated if > (not >=)
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max =
          __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
      int other_expert =
          __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

      // We want lower indices to "win" in every thread so we break ties this
      // way
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    if (thread_group_idx == 0) {
      // The lead thread from each sub-group will write out the final results to
      // global memory. (This will be a single) thread per row of the
      // input/output matrices.
      const int idx = k * thread_row + k_idx;
      output[idx] = T(max_val);
      indices[idx] = should_process_row ? expert : NUM_EXPERTS;
      source_rows[idx] = k_idx * num_rows + thread_row;
    }

    // Finally, we clear the value in the thread with the current max if there
    // is another iteration to run.
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            ComputeType(-10000.f);
      }
    }
  }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                "");
  static constexpr int VECs_PER_THREAD =
      std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template <typename T, int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_launcher_helper(const T* input,
                                         const bool* finished,
                                         T* output,
                                         int* indices,
                                         int* source_row,
                                         const int64_t num_rows,
                                         const int64_t num_experts,
                                         const int64_t k,
                                         cudaStream_t stream) {
  static constexpr uint64_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG =
      std::min(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
  using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topk_gating_softmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
      <<<num_blocks, block_dim, 0, stream>>>(
          input, finished, output, num_rows, indices, source_row, k);
}

template <typename T>
void topk_gating_softmax_kernelLauncher(const T* input,
                                        const bool* finished,
                                        T* output,
                                        T* softmax,
                                        int* indices,
                                        int* source_row,
                                        T* softmax_max_prob,
                                        const int64_t num_rows,
                                        const int64_t num_experts,
                                        const int64_t k,
                                        const bool group_moe,
                                        cudaStream_t stream,
                                        const bool topk_only_mode = false) {
  if (topk_only_mode) {
    static constexpr int TPB = 256;
    const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
    moe_top_k<T, TPB><<<config_topk.block_per_grid, TPB, 0, stream>>>(
        input, finished, output, indices, source_row, num_experts, k, num_rows);
    return;
  }
  static constexpr int WARPS_PER_TB = 4;

  switch (num_experts) {
    case 2: {
      topk_gating_softmax_launcher_helper<T, 2, WARPS_PER_TB>(input,
                                                              finished,
                                                              output,
                                                              indices,
                                                              source_row,
                                                              num_rows,
                                                              num_experts,
                                                              k,
                                                              stream);
      break;
    }
    case 4: {
      topk_gating_softmax_launcher_helper<T, 4, WARPS_PER_TB>(input,
                                                              finished,
                                                              output,
                                                              indices,
                                                              source_row,
                                                              num_rows,
                                                              num_experts,
                                                              k,
                                                              stream);
      break;
    }
    case 8: {
      topk_gating_softmax_launcher_helper<T, 8, WARPS_PER_TB>(input,
                                                              finished,
                                                              output,
                                                              indices,
                                                              source_row,
                                                              num_rows,
                                                              num_experts,
                                                              k,
                                                              stream);
      break;
    }
    case 16: {
      topk_gating_softmax_launcher_helper<T, 16, WARPS_PER_TB>(input,
                                                               finished,
                                                               output,
                                                               indices,
                                                               source_row,
                                                               num_rows,
                                                               num_experts,
                                                               k,
                                                               stream);
      break;
    }
    case 32: {
      topk_gating_softmax_launcher_helper<T, 32, WARPS_PER_TB>(input,
                                                               finished,
                                                               output,
                                                               indices,
                                                               source_row,
                                                               num_rows,
                                                               num_experts,
                                                               k,
                                                               stream);
      break;
    }
    case 64: {
      topk_gating_softmax_launcher_helper<T, 64, WARPS_PER_TB>(input,
                                                               finished,
                                                               output,
                                                               indices,
                                                               source_row,
                                                               num_rows,
                                                               num_experts,
                                                               k,
                                                               stream);
      break;
    }
    case 128: {
      topk_gating_softmax_launcher_helper<T, 128, WARPS_PER_TB>(input,
                                                                finished,
                                                                output,
                                                                indices,
                                                                source_row,
                                                                num_rows,
                                                                num_experts,
                                                                k,
                                                                stream);
      break;
    }
    case 256: {
      topk_gating_softmax_launcher_helper<T, 256, WARPS_PER_TB>(input,
                                                                finished,
                                                                output,
                                                                indices,
                                                                source_row,
                                                                num_rows,
                                                                num_experts,
                                                                k,
                                                                stream);
      break;
    }
    default: {
      static constexpr int TPB = 256;
      if (group_moe) {
        const int group_experts = num_experts / k;
        const int softmax_num_rows = num_rows * k;
        const auto config_softmax = Get1DBlocksAnd2DGridsMoe(softmax_num_rows);
        group_moe_softmax<T, TPB>
            <<<config_softmax.block_per_grid, TPB, 0, stream>>>(
                input,
                finished,
                softmax,
                softmax_max_prob,
                group_experts,
                softmax_num_rows);
        const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
        moe_top_k<T, TPB>
            <<<config_topk.block_per_grid, TPB, 0, stream>>>(softmax,
                                                             finished,
                                                             output,
                                                             indices,
                                                             source_row,
                                                             softmax_max_prob,
                                                             num_experts,
                                                             k,
                                                             num_rows);
      } else {
        const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
        moe_softmax<T, TPB><<<config_topk.block_per_grid, TPB, 0, stream>>>(
            input, finished, softmax, num_experts, num_rows);
        moe_top_k<T, TPB>
            <<<config_topk.block_per_grid, TPB, 0, stream>>>(softmax,
                                                             finished,
                                                             output,
                                                             indices,
                                                             source_row,
                                                             num_experts,
                                                             k,
                                                             num_rows);
      }
    }
  }
}

// ========================== Permutation things
// =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation
// map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It
// is "expanded" since we will have to duplicate some rows in the input matrix
// to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here
// has indices in the range (0, k*rows_in_input - 1). However, it is set up so
// that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input all map
// to row 0 in the original matrix. Thus, to know where to read in the source
// matrix, we simply take the modulus of the expanded index.

template <typename T, int VecSize>
__global__ void initialize_moe_routing_kernel(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int64_t num_rows,
    const int64_t active_rows,
    const int64_t cols,
    const int64_t num_rows_k) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  const int expanded_dest_row = blockIdx.x + blockIdx.y * gridDim.x;
  if (expanded_dest_row >= num_rows_k) return;
  const int expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  if (threadIdx.x == 0) {
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        expanded_dest_row;
  }

  if ((blockIdx.x + blockIdx.y * gridDim.x) < active_rows) {
    // Duplicate and permute rows
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* dest_row_ptr = permuted_output + expanded_dest_row * cols;

    for (int tid = threadIdx.x * VecSize; tid < cols;
         tid += blockDim.x * VecSize) {
      // dest_row_ptr[tid] = source_row_ptr[tid];
      Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
      Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
  }
}

template <typename T>
void initialize_moe_routing_kernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int64_t num_rows,
    const int64_t active_rows,
    const int64_t cols,
    const int64_t k,
    cudaStream_t stream) {
  const int threads = std::min(cols, int64_t(1024));
  constexpr int max_pack_size = 16 / sizeof(T);
  const auto config_initialize = Get1DBlocksAnd2DGridsMoe(num_rows * k);
  if (cols % max_pack_size == 0) {
    initialize_moe_routing_kernel<T, max_pack_size>
        <<<config_initialize.block_per_grid, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            num_rows,
            k * active_rows,
            cols,
            num_rows * k);
  } else {
    initialize_moe_routing_kernel<T, 1>
        <<<config_initialize.block_per_grid, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            num_rows,
            k * active_rows,
            cols,
            num_rows * k);
  }
}

// ============================== Infer GEMM sizes
// =================================
template <typename T>
__device__ inline int find_total_elts_leq_target(int* sorted_indices,
                                                 const int64_t arr_length,
                                                 const int64_t target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] > target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

template <typename T>
__global__ void compute_total_rows_before_expert_kernel(
    int* sorted_experts,
    const int64_t sorted_experts_len,
    const int64_t num_experts,
    int64_t* total_rows_before_expert) {
  // First, compute the global tid. We only need 1 thread per expert.
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) return;

  // This should construct the last index where each expert occurs.
  total_rows_before_expert[expert] =
      find_total_elts_leq_target<T>(sorted_experts, sorted_experts_len, expert);
}

template <typename T>
void compute_total_rows_before_expert(int* sorted_indices,
                                      const T* kkk,
                                      const int64_t total_indices,
                                      const int64_t num_experts,
                                      int64_t* total_rows_before_expert,
                                      cudaStream_t stream) {
  const int threads = std::min(int64_t(1024), num_experts);
  const int blocks = (num_experts + threads - 1) / threads;

  compute_total_rows_before_expert_kernel<T><<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and
// performs the final skip connection.
template <typename T, int RESIDUAL_NUM>
__global__ void finalize_moe_routing_kernel(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* bias,
    const float* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row,
    const int64_t cols,
    const int64_t k,
    const int64_t compute_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor,
    const int64_t num_rows) {
  const int original_row = blockIdx.x + blockIdx.y * gridDim.x;
  // const int original_row = blockIdx.x;
  // const int num_rows = gridDim.x;
  if (original_row >= num_rows) return;
  T* reduced_row_ptr = reduced_unpermuted_output + original_row * cols;

  for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
    T thread_output{0.f};
    float row_rescale{0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expanded_original_row = original_row + k_idx * num_rows;
      const int expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      const int64_t k_offset = original_row * k + k_idx;
      const float row_scale = scales[k_offset];
      row_rescale = row_rescale + row_scale;

      const T* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows + expanded_permuted_row * cols;

      const int expert_idx = expert_for_source_row[k_offset];
      const T* bias_ptr = bias ? bias + expert_idx * cols : nullptr;
      const T bias_value = bias_ptr ? bias_ptr[tid] : T{0.f};

      thread_output =
          static_cast<float>(thread_output) +
          row_scale * static_cast<float>(
                          expanded_permuted_rows_row_ptr[tid] +
                          bias_value *
                              static_cast<T>(static_cast<float>(compute_bias)));
    }

    thread_output = static_cast<float>(thread_output) /
                    (norm_topk_prob ? row_rescale : 1.0f) *
                    routed_scaling_factor;
    reduced_row_ptr[tid] = thread_output;
  }
}

template <typename T>
void finalize_moe_routing_kernelLauncher(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* bias,
    const float* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row,
    const int64_t num_rows,
    const int64_t cols,
    const int64_t k,
    const int64_t compute_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor,
    cudaStream_t stream) {
  const int threads = std::min(cols, int64_t(1024));
  const auto config_final = Get1DBlocksAnd2DGridsMoe(num_rows);

  finalize_moe_routing_kernel<T, 1>
      <<<config_final.block_per_grid, threads, 0, stream>>>(
          expanded_permuted_rows,
          reduced_unpermuted_output,
          bias,
          scales,
          expanded_source_row_to_expanded_dest_row,
          expert_for_source_row,
          cols,
          k,
          compute_bias,
          norm_topk_prob,
          routed_scaling_factor,
          num_rows);
}

// ========================= TopK Softmax specializations
// ===========================
template void topk_gating_softmax_kernelLauncher(const float*,
                                                 const bool*,
                                                 float*,
                                                 float*,
                                                 int*,
                                                 int*,
                                                 float*,
                                                 const int64_t,
                                                 const int64_t,
                                                 const int64_t,
                                                 const bool,
                                                 cudaStream_t,
                                                 const bool);
template void topk_gating_softmax_kernelLauncher(const half*,
                                                 const bool*,
                                                 half*,
                                                 half*,
                                                 int*,
                                                 int*,
                                                 half*,
                                                 const int64_t,
                                                 const int64_t,
                                                 const int64_t,
                                                 const bool,
                                                 cudaStream_t,
                                                 const bool);
#ifdef PADDLE_CUDA_BF16
template void topk_gating_softmax_kernelLauncher(const __nv_bfloat16*,
                                                 const bool*,
                                                 __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 int*,
                                                 int*,
                                                 __nv_bfloat16*,
                                                 const int64_t,
                                                 const int64_t,
                                                 const int64_t,
                                                 const bool,
                                                 cudaStream_t,
                                                 const bool);
#endif
// ===================== Specializations for init routing
// =========================
template void initialize_moe_routing_kernelLauncher(const float*,
                                                    float*,
                                                    const int*,
                                                    int*,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    cudaStream_t);
template void initialize_moe_routing_kernelLauncher(const half*,
                                                    half*,
                                                    const int*,
                                                    int*,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    cudaStream_t);
#ifdef PADDLE_CUDA_BF16
template void initialize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                    __nv_bfloat16*,
                                                    const int*,
                                                    int*,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    cudaStream_t);
#endif
// ==================== Specializations for final routing
// ===================================
template void finalize_moe_routing_kernelLauncher(const float*,
                                                  float*,
                                                  const float*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const bool,
                                                  const float,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const half*,
                                                  half*,
                                                  const half*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const bool,
                                                  const float,
                                                  cudaStream_t);
#ifdef PADDLE_CUDA_BF16
template void finalize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                  __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const bool,
                                                  const float,
                                                  cudaStream_t);
#endif
template void compute_total_rows_before_expert(int*,
                                               const half*,
                                               const int64_t,
                                               const int64_t,
                                               int64_t*,
                                               cudaStream_t stream);
#ifdef PADDLE_CUDA_BF16
template void compute_total_rows_before_expert(int*,
                                               const __nv_bfloat16*,
                                               const int64_t,
                                               const int64_t,
                                               int64_t*,
                                               cudaStream_t stream);
#endif
}  // namespace phi

#endif
