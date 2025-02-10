// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
//
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
#pragma once

#include <assert.h>
#include "paddle/phi/backends/gpu/gpu_types.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_INF_F __int_as_float(0x7f800000)

namespace phi {
namespace funcs {

constexpr int kWarpSize = 32;

template <typename T>
__inline__ __device__ T Inf();

template <>
__inline__ __device__ float Inf<float>() {
  return CUDART_INF_F;
}

template <>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

template <typename T,
          template <typename>
          class Functor,
          int ThreadGroupWidth = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = ThreadGroupWidth / 2; mask > 0; mask /= 2) {
#if PADDLE_WITH_HIP
    val = Functor<T>()(val, __shfl_xor(val, mask));
#else
    val = Functor<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
#endif
  }
  return val;
}

#if PADDLE_WITH_HIP
inline void GetNumBlocks(int64_t block_size,
                         int64_t max_blocks,
                         int64_t waves,
                         int* num_blocks) {
  int dev;
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetDevice(&dev));
  int sm_count;
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceGetAttribute(
      &sm_count, hipDeviceAttributeMultiprocessorCount, dev));
  int tpm;
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceGetAttribute(
      &tpm, hipDeviceAttributeMaxThreadsPerMultiProcessor, dev));
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
}
#else
inline void GetNumBlocks(int64_t block_size,
                         int64_t max_blocks,
                         int64_t waves,
                         int* num_blocks) {
  int dev;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&dev));
  int sm_count;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int tpm;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceGetAttribute(
      &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
}
#endif

template <typename T,
          typename SourceType,
          typename Context,
          int VecSize,
          int ColsPerThread,
          int RowsPerThread,
          int ThreadGroupWidth,
          bool NeedPadding>
__global__ void LogsumexpWarpImpl(const Context& dev_ctx,
                                  const int64_t num_row,
                                  const int64_t num_col,
                                  const SourceType* in,
                                  SourceType* out) {
  static_assert(ColsPerThread % VecSize == 0, "");
  static_assert(ThreadGroupWidth <= kWarpSize, "");
  static_assert(kWarpSize % ThreadGroupWidth == 0, "");
  constexpr int num_read = ColsPerThread / VecSize;
  assert(num_col <= ColsPerThread * ThreadGroupWidth);
  const int group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_thread_group = gridDim.x * blockDim.y;
  const int thread_id = threadIdx.x;
  const int step = num_thread_group * RowsPerThread;

  using LoadType = phi::AlignedVector<SourceType, VecSize>;
  using StoreType = phi::AlignedVector<SourceType, RowsPerThread>;

  LoadType load_vec;
  StoreType store_vec;

  T buffer[RowsPerThread][ColsPerThread];

  for (int64_t cur_row = group_id * RowsPerThread; cur_row < num_row;
       cur_row += step) {
    T thread_max[RowsPerThread];
// Read data
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      thread_max[row_id] = -Inf<T>();
      T* row_buffer = buffer[row_id];
#pragma unroll
      for (int read_id = 0; read_id < num_read; read_id++) {
        const int offset = read_id * VecSize;
        const int cur_col = (read_id * ThreadGroupWidth + thread_id) * VecSize;
        if (!NeedPadding || cur_col < num_col) {
          int64_t load_offset = ((cur_row + row_id) * num_col + cur_col);
          phi::Load<SourceType, VecSize>(in + load_offset, &load_vec);
#pragma unroll
          for (int i = 0; i < VecSize; i++) {
            row_buffer[offset + i] = static_cast<T>(load_vec[i]);
            thread_max[row_id] =
                max(thread_max[row_id], row_buffer[offset + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < VecSize; i++) {
            row_buffer[offset + i] = -Inf<T>();
          }
        }
      }
    }
    T warp_max[RowsPerThread];
// Get warp max
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      warp_max[row_id] = WarpAllReduce<T, kps::MaxFunctor, ThreadGroupWidth>(
          thread_max[row_id]);
    }
    T thread_sum[RowsPerThread];
// Calculate
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      thread_sum[row_id] = 0;
      T* row_buffer = buffer[row_id];
#pragma unroll
      for (int i = 0; i < ColsPerThread; i++) {
        thread_sum[row_id] += exp(row_buffer[i] - warp_max[row_id]);
      }
    }
// Get warp sum and write
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      T res = log(WarpAllReduce<T, kps::AddFunctor, ThreadGroupWidth>(
          thread_sum[row_id]));
      store_vec[row_id] = static_cast<SourceType>(res + warp_max[row_id]);
    }
    if (thread_id == 0 && cur_row < num_row) {
      phi::Store<SourceType, RowsPerThread>(store_vec,
                                            out + group_id * RowsPerThread);
    }
  }
}

template <typename T,
          typename SourceType,
          typename Context,
          int VecSize,
          int ColsPerThread,
          int RowsPerThread,
          int ThreadGroupWidth,
          bool NeedPadding>
inline gpuError_t LaunchLogsumexpWarp(const Context& dev_ctx,
                                      const int64_t num_row,
                                      const int64_t num_col,
                                      const SourceType* in,
                                      SourceType* out) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % ThreadGroupWidth == 0, "");
  constexpr int thread_groups_per_block = block_size / ThreadGroupWidth;
  dim3 block_dim(ThreadGroupWidth, thread_groups_per_block);
  const int64_t num_blocks =
      (num_row / RowsPerThread + thread_groups_per_block - 1) /
      thread_groups_per_block;
  int grid_dim_x;
  { GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x); }
  LogsumexpWarpImpl<T,
                    SourceType,
                    Context,
                    VecSize,
                    ColsPerThread,
                    RowsPerThread,
                    ThreadGroupWidth,
                    NeedPadding>
      <<<grid_dim_x, block_dim, 0, dev_ctx.stream()>>>(
          dev_ctx, num_row, num_col, in, out);
#if PADDLE_WITH_HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}

template <typename T,
          typename SourceType,
          typename Context,
          int VecSize,
          int ColsPerThread,
          int RowsPerThread,
          int ThreadGroupWidth>
inline gpuError_t DispatchLogsumexpWarpWithPadding(const Context& dev_ctx,
                                                   const int64_t num_row,
                                                   const int64_t num_col,
                                                   const SourceType* in,
                                                   SourceType* out) {
  if (num_col == ColsPerThread * ThreadGroupWidth) {
    return LaunchLogsumexpWarp<T,
                               SourceType,
                               Context,
                               VecSize,
                               ColsPerThread,
                               RowsPerThread,
                               ThreadGroupWidth,
                               false>(dev_ctx, num_row, num_col, in, out);
  } else {
    return LaunchLogsumexpWarp<T,
                               SourceType,
                               Context,
                               VecSize,
                               ColsPerThread,
                               RowsPerThread,
                               ThreadGroupWidth,
                               true>(dev_ctx, num_row, num_col, in, out);
  }
}

template <typename T, typename SourceType, typename Context, int VecSize>
typename std::enable_if<VecSize == 1, gpuError_t>::type
DispatchLogsumexpWarpCols(const Context& dev_ctx,
                          const int64_t num_row,
                          const int64_t num_col,
                          const SourceType* in,
                          SourceType* out) {
  if (num_col <= 0) {
#if PADDLE_WITH_HIP
    return hipErrorInvalidValue;
#else
    return cudaErrorInvalidValue;
#endif
  }
#define HANDLE_THREAD_GROUP(thread_group_width)                    \
  if (num_col <= (thread_group_width)*VecSize) {                   \
    if (num_row % 2 == 0) {                                        \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              VecSize,             \
                                              VecSize,             \
                                              2,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, in, out);                     \
    } else {                                                       \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              VecSize,             \
                                              VecSize,             \
                                              1,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, in, out);                     \
    }                                                              \
  }
  HANDLE_THREAD_GROUP(1)
  HANDLE_THREAD_GROUP(2)
  HANDLE_THREAD_GROUP(4)
  HANDLE_THREAD_GROUP(8)
  HANDLE_THREAD_GROUP(16)
  HANDLE_THREAD_GROUP(32)
#undef HANDLE_ROWS
// if num_col > 32
#define HANDLE_COL(col)                                 \
  if (num_col <= (col)*kWarpSize) {                     \
    return DispatchLogsumexpWarpWithPadding<T,          \
                                            SourceType, \
                                            Context,    \
                                            VecSize,    \
                                            col,        \
                                            1,          \
                                            kWarpSize>( \
        dev_ctx, num_row, num_col, in, out);            \
  }

  HANDLE_COL(2)
  HANDLE_COL(3)
  HANDLE_COL(4)
  HANDLE_COL(5)
  HANDLE_COL(6)
  HANDLE_COL(7)
  HANDLE_COL(8)
  HANDLE_COL(9)
  HANDLE_COL(10)
  HANDLE_COL(11)
  HANDLE_COL(12)
  HANDLE_COL(13)
  HANDLE_COL(14)
  HANDLE_COL(15)
  HANDLE_COL(16)
  HANDLE_COL(17)
  HANDLE_COL(18)
  HANDLE_COL(19)
  HANDLE_COL(20)
  HANDLE_COL(21)
  HANDLE_COL(22)
  HANDLE_COL(23)
  HANDLE_COL(24)
  HANDLE_COL(25)
  HANDLE_COL(26)
  HANDLE_COL(27)
  HANDLE_COL(28)
  HANDLE_COL(29)
  HANDLE_COL(30)
  HANDLE_COL(31)
  HANDLE_COL(32)
#undef HANDLE_COL
#if PADDLE_WITH_HIP
  return hipErrorInvalidValue;
#else
  return cudaErrorInvalidValue;
#endif
}

template <typename T, typename SourceType, typename Context, int VecSize>
typename std::enable_if<VecSize == 2, gpuError_t>::type
DispatchLogsumexpWarpCols(const Context& dev_ctx,
                          const int64_t num_row,
                          const int64_t num_col,
                          const SourceType* in,
                          SourceType* out) {
  if (num_col <= 0) {
#if PADDLE_WITH_HIP
    return hipErrorInvalidValue;
#else
    return cudaErrorInvalidValue;
#endif
  }
#define HANDLE_THREAD_GROUP(thread_group_width)                    \
  if (num_col <= (thread_group_width)*VecSize) {                   \
    if (num_row % 2 == 0) {                                        \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              VecSize,             \
                                              VecSize,             \
                                              2,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, in, out);                     \
    } else {                                                       \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              VecSize,             \
                                              VecSize,             \
                                              1,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, in, out);                     \
    }                                                              \
  }
  HANDLE_THREAD_GROUP(1)
  HANDLE_THREAD_GROUP(2)
  HANDLE_THREAD_GROUP(4)
  HANDLE_THREAD_GROUP(8)
  HANDLE_THREAD_GROUP(16)
  HANDLE_THREAD_GROUP(32)
#undef HANDLE_THREAD_GROUP
// if num_col > 32
#define HANDLE_COL(col)                                 \
  if (num_col <= (col)*kWarpSize) {                     \
    return DispatchLogsumexpWarpWithPadding<T,          \
                                            SourceType, \
                                            Context,    \
                                            VecSize,    \
                                            col,        \
                                            1,          \
                                            kWarpSize>( \
        dev_ctx, num_row, num_col, in, out);            \
  }

  HANDLE_COL(4)
  HANDLE_COL(6)
  HANDLE_COL(8)
  HANDLE_COL(10)
  HANDLE_COL(12)
  HANDLE_COL(14)
  HANDLE_COL(16)
  HANDLE_COL(18)
  HANDLE_COL(20)
  HANDLE_COL(22)
  HANDLE_COL(24)
  HANDLE_COL(26)
  HANDLE_COL(28)
  HANDLE_COL(30)
  HANDLE_COL(32)
#undef HANDLE_COL
#if PADDLE_WITH_HIP
  return hipErrorInvalidValue;
#else
  return cudaErrorInvalidValue;
#endif
}

template <typename T, typename SourceType, typename Context>
inline gpuError_t DispatchLogsumexpWarp(const Context& dev_ctx,
                                        const int64_t num_row,
                                        const int64_t num_col,
                                        const SourceType* in,
                                        SourceType* out) {
  // dispatch logsumexp warp with vecsize
  if (num_col % 2 == 0) {
    return DispatchLogsumexpWarpCols<T, SourceType, Context, 2>(
        dev_ctx, num_row, num_col, in, out);
  } else {
    return DispatchLogsumexpWarpCols<T, SourceType, Context, 1>(
        dev_ctx, num_row, num_col, in, out);
  }
}
}  // namespace funcs
}  // namespace phi
