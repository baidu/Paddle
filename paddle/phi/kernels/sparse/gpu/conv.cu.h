/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <thrust/remove.h>
#include <thrust/unique.h>
#ifdef __NVCC__
#include <cub/block/block_scan.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/phi/kernels/sparse/conv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"

namespace phi {
namespace sparse {

using Dims4D = phi::funcs::sparse::Dims4D;

// Vectorize load and store global memory
// In the scene of 3D point cloud, the slice_size 4,8,16,32,64 are commonly
// used.
template <typename T, typename IndexT = int, int VecSize>
__global__ void GatherKernel(const T* params,
                             const IndexT* indices,
                             T* output,
                             size_t index_size,
                             size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size / VecSize, int64_t) {
    const int vec_slice_size = slice_size / VecSize;
    int indices_i = i / vec_slice_size;
    int slice_i = i - indices_i * vec_slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i * VecSize;
    using LoadT = phi::AlignedVector<T, VecSize>;
    using StoreT = phi::AlignedVector<T, VecSize>;
    LoadT params_vec;
    phi::Load<T, VecSize>(params + params_i, &params_vec);
    phi::Store<T, VecSize>(params_vec, output + i * VecSize);
  }
}

// double sparse, seed GroupIndices
template <typename T, typename IntT, int VecSize>
__global__ void GatherKernelV2(const T* inputs,
                               const int* index_counts,
                               const int* index_groups,
                               const int non_zero_num,
                               const int kernel_size,
                               const int channels,
                               const int buffer_count,
                               T* output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int vec_channels = channels / VecSize;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  for (int i = tid; i < non_zero_num * vec_channels;
       i += gridDim.x * blockDim.x) {
    int indices_i = i / vec_channels;
    int channels_i = i - indices_i * vec_channels;
    LoadT in_vec;
    phi::Load<T, VecSize>(inputs + indices_i * channels + channels_i * VecSize,
                          &in_vec);
#pragma unroll
    for (int it = 0; it < buffer_count; it++) {
      int len = index_counts[indices_i + it * non_zero_num];
      const int group_offset = it * kernel_size * non_zero_num;
#pragma unroll
      for (int j = 0; j < len; j++) {
        int out_i = index_groups[indices_i * kernel_size + j + group_offset];
        phi::Store<T, VecSize>(
            in_vec, output + out_i * channels + channels_i * VecSize);
      }
    }
  }
}

template <typename T, typename IntT>
inline void Gather(const GPUContext& dev_ctx,
                   const T* inputs,
                   const IntT* indices,
                   const int indices_size,
                   const int channels,
                   T* output) {
  const int VecSize = VecBytes / sizeof(T);
  if (channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, indices_size * channels / VecSize, 1);
    GatherKernel<T, IntT, VecSize>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(inputs, indices, output, indices_size, channels);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, indices_size * channels, 1);
    GatherKernel<T, IntT, 1>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(inputs, indices, output, indices_size, channels);
  }
}

template <typename T, typename IntT>
inline void GatherV2(const GPUContext& dev_ctx,
                     const T* inputs,
                     const int* index_counts,
                     const int* index_groups,
                     const int non_zero_num,
                     const int kernel_size,
                     const int channels,
                     const int buffer_count,
                     T* output) {
  const int VecSize = VecBytes / sizeof(T);
  if (channels % VecSize == 0) {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, non_zero_num * channels / VecSize, 1);
    GatherKernelV2<T, IntT, VecSize><<<config.block_per_grid.x,
                                       config.thread_per_block.x,
                                       0,
                                       dev_ctx.stream()>>>(inputs,
                                                           index_counts,
                                                           index_groups,
                                                           non_zero_num,
                                                           kernel_size,
                                                           channels,
                                                           buffer_count,
                                                           output);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, non_zero_num * channels, 1);
    GatherKernelV2<T, IntT, 1><<<config.block_per_grid.x,
                                 config.thread_per_block.x,
                                 0,
                                 dev_ctx.stream()>>>(inputs,
                                                     index_counts,
                                                     index_groups,
                                                     non_zero_num,
                                                     kernel_size,
                                                     channels,
                                                     buffer_count,
                                                     output);
  }
}

// unique the out indices in rulebook
template <typename IntT>
__global__ void UniqueKernel(const IntT* in_indices,
                             const int rulebook_len,
                             int* index_flags,
                             int* out_indices,
                             int* nnz) {
  extern __shared__ int cache[];
  __shared__ int count, start;
  if (threadIdx.x == 0) {
    count = 0;
    start = 0;
  }
  __syncthreads();

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < rulebook_len) {
    // atomicOr only support int
    int index = static_cast<int>(in_indices[i]);
    const bool flag = phi::funcs::sparse::SetBits(index, index_flags);
    if (!flag) {
      int j = atomicAdd(&count, 1);
      cache[j] = index;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    start = atomicAdd(nnz, count);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    out_indices[start + i] = cache[i];
  }
}

inline __device__ uint32_t BitCount(const uint32_t data) {
  uint32_t count = data;
  count = (count & 0x55555555) + ((count >> 1) & 0x55555555);
  count = (count & 0x33333333) + ((count >> 2) & 0x33333333);
  count = (count & 0x0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f);
  count = (count & 0x00ff00ff) + ((count >> 8) & 0x00ff00ff);
  count = (count & 0x0000ffff) + ((count >> 16) & 0x0000ffff);
  return count;
}

static __global__ void GetOutIndicesCounter(const int* flags,
                                            const int n,
                                            int* out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int block_count;
  if (threadIdx.x == 0) {
    block_count = 0;
  }
  __syncthreads();

  if (tid < n) {
    // get the count of 1 in flags[tid]
    uint32_t count = BitCount(static_cast<uint32_t>(flags[tid]));
    // add to block_count
    // TODO(zhangkaihuo): replace with block reduce_sum
    atomicAdd(&block_count, static_cast<int>(count));
  }
  __syncthreads();
  // write to out
  if (threadIdx.x == 0) {
    out[blockIdx.x] = block_count;
  }
}

template <int BS>
__global__ void GetOutIndices(const int* flags,
                              const int n,
                              const int* offsets,
                              const int out_nnz,
                              int* out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int block_counts[BS];
  __shared__ int block_outs[BS * 32];

  int count = 0;

  if (tid < n) {
    // get the count of 1 in flags[tid]
    int flag = flags[tid];
    count = BitCount(static_cast<uint32_t>(flag));
  }

  // call block prefix_sum
  // using namespace cub;
  typedef cub::BlockScan<int, BS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockScan(temp_storage).ExclusiveSum(count, count);
  __syncthreads();

  // write index to out
  if (tid < n) {
    // get the count of 1 in flags[tid]
    int flag = flags[tid];
    // int j = block_counts[threadIdx.x];
    int j = count;
    // TODO(zhangkaihuo): opt the loop
    for (int i = 0; i < 32; ++i) {
      if ((1 & (flag >> i)) == 1) {
        block_outs[j++] = (tid << 5) + i;
      }
    }
  }

  __syncthreads();
  // write to block_outs
  int start = offsets[blockIdx.x];
  int end = blockIdx.x == gridDim.x - 1 ? out_nnz : offsets[blockIdx.x + 1];
  for (int i = threadIdx.x; i < end - start; i += blockDim.x) {
    out[start + i] = block_outs[i];
  }
}

template <typename IntT>
__global__ void GroupIndices(const int* out_index_table,
                             const int n,
                             const int kernel_size,
                             IntT* out_indices,
                             int* out_index_counts,
                             int* out_index_groups) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    IntT index = out_indices[i];
    int real_index = out_index_table[index];
    out_indices[i] = real_index;

    // kernel_size at most
    int j = atomicAdd(out_index_counts + real_index, 1);
    // nnz * kernel_size
    out_index_groups[real_index * kernel_size + j] = i;
  }
}

/**
 * @brief product rulebook
 * for input_i in x_indices:
 *   if input_i participate in the convolution calculation:
 *       infer the output_i by input_i and kernel_i
 *       save output_i
 *
 * x_indices: the indices of input features
 * x_dims: the input dims
 * kernel_dims: the kernel dims
 * out_dims: the output dims
 * non_zero_num: the number of input features
 * rulebook: the rulebook to save the kernel index, input index and output index
 * counter: save the number of times each location in the kernel participates in
 * the calculation
 **/
template <typename T>
__global__ void ProductRuleBookKernel(const T* x_indices,
                                      const Dims4D x_dims,
                                      const Dims4D kernel_dims,
                                      const Dims4D out_dims,
                                      const int64_t non_zero_num,
                                      const Dims4D paddings,
                                      const Dims4D dilations,
                                      const Dims4D strides,
                                      const bool is2D,
                                      T* rulebook,
                                      int* counter) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ int counter_buf[];  // kernel_size
  const int kernel_size = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
  const int offset = kernel_size * non_zero_num;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int kernel_index = 0;
    T batch = x_indices[i];
    T in_z = is2D ? 0 : x_indices[i + non_zero_num];
    T in_y =
        is2D ? x_indices[i + non_zero_num] : x_indices[i + 2 * non_zero_num];
    T in_x = is2D ? x_indices[i + 2 * non_zero_num]
                  : x_indices[i + 3 * non_zero_num];
    for (int kz = 0; kz < kernel_dims[1]; kz++) {
      for (int ky = 0; ky < kernel_dims[2]; ky++) {
        for (int kx = 0; kx < kernel_dims[3]; kx++) {
          int in_i = -1, out_index = -1, kernel_i = -1;
          if (phi::funcs::sparse::Check(x_dims,
                                        kernel_dims,
                                        paddings,
                                        dilations,
                                        strides,
                                        in_x,
                                        in_y,
                                        in_z,
                                        kx,
                                        ky,
                                        kz)) {
            T out_z =
                is2D ? 0
                     : (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            T out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            T out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            in_i = i;
            out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                batch, out_x, out_y, out_z, out_dims);
            atomicAdd(&counter_buf[kernel_index], 1);
            kernel_i = kernel_index;
          }
          rulebook[kernel_index * non_zero_num + i] = in_i;
          rulebook[kernel_index * non_zero_num + offset + i] = out_index;
          ++kernel_index;
        }
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    atomicAdd(&counter[i], counter_buf[i]);
  }
}

template <typename IntT>
__global__ void GetOutIndexTable1(const IntT* indices,
                                  const IntT non_zero_num,
                                  const Dims4D dims,
                                  int* index_flags,
                                  const bool is2D,
                                  int* out_index_table) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num, int64_t) {
    IntT batch = indices[i];
    IntT in_z = is2D ? 0 : indices[i + non_zero_num];
    IntT in_y =
        is2D ? indices[i + non_zero_num] : indices[i + 2 * non_zero_num];
    IntT in_x =
        is2D ? indices[i + 2 * non_zero_num] : indices[i + 3 * non_zero_num];
    IntT index = PointToIndex(batch, in_x, in_y, in_z, dims);
    phi::funcs::sparse::SetBits(index, index_flags);
    out_index_table[index] = i;
  }
}

template <typename IntT>
__global__ void GetOutIndexTable(int* indices,
                                 const int non_zero_num,
                                 const Dims4D out_dims,
                                 const bool is2D,
                                 int* out_index_table,
                                 IntT* out_indices) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num, int64_t) {
    IntT index = static_cast<IntT>(indices[i]);
    out_index_table[index] = i;
    IntT batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<Dims4D>(
        index, out_dims, &batch, &x, &y, &z);
    // get out indices
    out_indices[i] = batch;
    if (is2D) {
      out_indices[i + non_zero_num] = y;
      out_indices[i + non_zero_num * 2] = x;
    } else {
      out_indices[i + non_zero_num] = z;
      out_indices[i + non_zero_num * 2] = y;
      out_indices[i + non_zero_num * 3] = x;
    }
    indices[i] = 0;
  }
}

template <typename IntT>
__global__ void CopyRuleBook(const int* counters,
                             const int* offsets,
                             const IntT* in_rulebook,
                             const int len,
                             const int kernel_size,
                             const int non_zero_num,
                             IntT* out_rulebook) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ int cache_counters[];
  int* cache_offsets = cache_counters + kernel_size;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    cache_counters[i] = counters[i];
    cache_offsets[i] = offsets[i];
  }
  __syncthreads();
  for (int i = tid; i < len; i += gridDim.x * blockDim.x) {
    // get the kernel index
    int kernel_index = 0;
    for (; kernel_index < kernel_size - 1; kernel_index++) {
      if (i >= offsets[kernel_index] && i < offsets[kernel_index + 1]) {
        break;
      }
    }
    int inner_index = i - offsets[kernel_index];
    out_rulebook[i] = in_rulebook[kernel_index * non_zero_num + inner_index];
    out_rulebook[len + i] =
        in_rulebook[kernel_size * non_zero_num + kernel_index * non_zero_num +
                    inner_index];
  }
}

template <typename T>
__global__ void ProductSubmRuleBookKernel(const T* x_indices,
                                          const Dims4D x_dims,
                                          const Dims4D kernel_dims,
                                          const Dims4D out_dims,
                                          const int64_t non_zero_num,
                                          const Dims4D paddings,
                                          const Dims4D dilations,
                                          const Dims4D strides,
                                          const bool is2D,
                                          const int* index_flags,
                                          const int* out_index_table,
                                          T* rulebook,
                                          int* counter) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int kernel_size = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
  extern __shared__ int counter_buf[];  // kernel_size
  int* counter_buf2 = counter_buf + kernel_size;
  int* rulebook_buf = counter_buf + kernel_size * 2;

  const int offset = kernel_size * non_zero_num;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int kernel_index = 0;
    T batch = x_indices[i];
    T in_z = is2D ? 0 : x_indices[i + non_zero_num];
    T in_y =
        is2D ? x_indices[i + non_zero_num] : x_indices[i + 2 * non_zero_num];
    T in_x = is2D ? x_indices[i + 2 * non_zero_num]
                  : x_indices[i + 3 * non_zero_num];
    for (int kz = 0; kz < kernel_dims[1]; kz++) {
      for (int ky = 0; ky < kernel_dims[2]; ky++) {
        for (int kx = 0; kx < kernel_dims[3]; kx++) {
          int in_i = -1, out_index = -1, kernel_i = -1;
          if (phi::funcs::sparse::Check(x_dims,
                                        kernel_dims,
                                        paddings,
                                        dilations,
                                        strides,
                                        in_x,
                                        in_y,
                                        in_z,
                                        kx,
                                        ky,
                                        kz)) {
            T out_z =
                is2D ? 0
                     : (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            T out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            T out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                batch, out_x, out_y, out_z, out_dims);
            const bool flag =
                phi::funcs::sparse::TestBits(out_index, index_flags);
            if (flag) {
              int real_out_index = out_index_table[out_index];
              in_i = i;
              int buf_i = atomicAdd(&counter_buf[kernel_index], 1);
              kernel_i = kernel_index;
              rulebook_buf[kernel_index * blockDim.x + buf_i] = in_i;
              rulebook_buf[kernel_index * blockDim.x +
                           kernel_size * blockDim.x + buf_i] = real_out_index;
            }
          }
          ++kernel_index;
        }
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf2[i] = atomicAdd(&counter[i], counter_buf[i]);
  }
  __syncthreads();
  for (int i = 0; i < kernel_size; i++) {
    if (threadIdx.x < counter_buf[i]) {
      rulebook[i * non_zero_num + counter_buf2[i] + threadIdx.x] =
          rulebook_buf[i * blockDim.x + threadIdx.x];
      rulebook[i * non_zero_num + offset + counter_buf2[i] + threadIdx.x] =
          rulebook_buf[i * blockDim.x + kernel_size * blockDim.x + threadIdx.x];
    }
  }
}

template <typename IntT>
__global__ void GroupIndices(const int n,
                             const int kernel_size,
                             const IntT* indices,
                             int* index_counts,
                             int* index_groups) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    IntT index = indices[i];
    // kernel_size at most
    int j = atomicAdd(index_counts + index, 1);
    // nnz * kernel_size
    index_groups[index * kernel_size + j] = i;
  }
}

// double space to reduce atomicAdd conflict
template <typename IntT>
__global__ void GroupIndicesV2(const int rulebook_len,
                               const int non_zero_num,
                               const int kernel_size,
                               const int half_kernel_offset,
                               const IntT* indices,
                               int* index_counts,
                               int* index_groups) {
  CUDA_KERNEL_LOOP_TYPE(i, rulebook_len, int64_t) {
    IntT index = indices[i];
    int* counts_ptr =
        i < half_kernel_offset ? index_counts : index_counts + non_zero_num;
    int* groups_ptr = i < half_kernel_offset
                          ? index_groups
                          : index_groups + non_zero_num * kernel_size;
    // conflict kernel_size times at most
    int j = atomicAdd(counts_ptr + index, 1);
    // nnz * kernel_size
    groups_ptr[index * kernel_size + j] = i;
  }
}

inline void CallThrustScan(const GPUContext& dev_ctx,
                           const int* counter_ptr,
                           const int kernel_size,
                           int* offsets_ptr,
                           int* h_counter_ptr,
                           int* h_offsets_ptr) {
#ifdef PADDLE_WITH_HIP
  thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         counter_ptr,
                         counter_ptr + kernel_size,
                         offsets_ptr);

  phi::backends::gpu::GpuMemcpyAsync(h_counter_ptr,
                                     counter_ptr,
                                     kernel_size * sizeof(int),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());

  phi::backends::gpu::GpuMemcpyAsync(h_offsets_ptr,
                                     offsets_ptr,
                                     kernel_size * sizeof(int),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());
}

// the basic algorithm can refer to convolution_kernel.cc or
// the second paper
// example:
// 1. the rulebook:
//  the kernel_index:                       0, 0, 0, 1, 1, 1, 2, 2, ....
//  the out_index(key):                     20, 30, 33, 30, 33, 20, 25
// 2. mark the index of out_index(value):   0, 1, 2, 3, 4, 5, 6, ....
// 3. sorted the (key, value)
// 4. unique the (key, value):
//  unique_key:     20, 25, 30, 33
//  unique_values:  0, 2, 3, 5
//  the index of unique_values is: 0, 1, 2, 3
// 5. update the out_index by unique_key, unique_value and the index of
// unique_value:
//  the new out_index: 0, 2, 3, 2, 3, 0, 1
template <typename T, typename Context, typename IntT = int>
int ProductRuleBook(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const std::vector<int>& kernel_sizes,
                    const std::vector<int>& paddings,
                    const std::vector<int>& dilations,
                    const std::vector<int>& strides,
                    const DDim& out_dims,
                    const bool subm,
                    DenseTensor* rulebook,
                    DenseTensor* counter_per_kernel,
                    DenseTensor* offsets_per_kernel,
                    DenseTensor* out_index,
                    DenseTensor* unique_value,
                    SparseCooTensor* out,
                    int* h_counter,
                    int* h_offsets) {
  const bool is2D = out_dims.size() == 4 ? true : false;
  auto indices_dtype = phi::CppTypeToDataType<IntT>::Type();
  const int64_t non_zero_num = x.nnz();
  const auto& indices = x.indices();
  const IntT* indices_ptr = indices.data<IntT>();
  int* counter_ptr = counter_per_kernel->data<int>();
  int* offsets_ptr = offsets_per_kernel->data<int>();
  int kernel_size = is2D ? kernel_sizes[0] * kernel_sizes[1]
                         : kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];

  const auto x_dims = x.dims();

  int xdim0, xdim1, xdim2, xdim3;
  int kdim0, kdim1, kdim2, kdim3;
  int odim0, odim1, odim2, odim3;
  int pdim0, pdim1, pdim2, pdim3;
  int sdim0, sdim1, sdim2, sdim3;
  int ddim0, ddim1, ddim2, ddim3;

  xdim0 = x_dims[0];
  xdim1 = is2D ? x_dims[2] : x_dims[3];
  xdim2 = is2D ? x_dims[1] : x_dims[2];
  xdim3 = is2D ? 1 : x_dims[1];

  kdim0 = 1;
  kdim1 = is2D ? kernel_sizes[1] : kernel_sizes[2];
  kdim2 = is2D ? kernel_sizes[0] : kernel_sizes[1];
  kdim3 = is2D ? 1 : kernel_sizes[0];

  odim0 = out_dims[0];
  odim1 = is2D ? out_dims[2] : out_dims[3];
  odim2 = is2D ? out_dims[1] : out_dims[2];
  odim3 = is2D ? 1 : out_dims[1];

  pdim0 = 1;
  pdim1 = is2D ? paddings[1] : paddings[2];
  pdim2 = is2D ? paddings[0] : paddings[1];
  pdim3 = is2D ? 1 : paddings[0];

  sdim0 = 1;
  sdim1 = is2D ? strides[1] : strides[2];
  sdim2 = is2D ? strides[0] : strides[1];
  sdim3 = is2D ? 1 : strides[0];

  ddim0 = 1;
  ddim1 = is2D ? dilations[1] : dilations[2];
  ddim2 = is2D ? dilations[0] : dilations[1];
  ddim3 = is2D ? 1 : dilations[0];

  const Dims4D d_x_dims(xdim0, xdim1, xdim2, xdim3);
  const Dims4D d_kernel_dims(kdim0, kdim1, kdim2, kdim3);
  const Dims4D d_out_dims(odim0, odim1, odim2, odim3);
  const Dims4D d_paddings(pdim0, pdim1, pdim2, pdim3);
  const Dims4D d_strides(sdim0, sdim1, sdim2, sdim3);
  const Dims4D d_dilations(ddim0, ddim1, ddim2, ddim3);

  // 1. product rule book
  phi::backends::gpu::GpuMemsetAsync(counter_ptr,
                                     0,
                                     sizeof(int) * counter_per_kernel->numel(),
                                     dev_ctx.stream());
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);

  const int rulebook_rows = 2;
  const int rulebook_cols = kernel_size * non_zero_num;
  DenseTensorMeta rulebook_meta(
      indices_dtype, {rulebook_rows, rulebook_cols}, DataLayout::NCHW);

  int table_size = 1;
  for (int i = 0; i < out_dims.size() - 1; i++) {
    table_size *= out_dims[i];
  }
  DenseTensor out_index_table = phi::Empty<int>(dev_ctx, {table_size});
  int* out_index_table_ptr = out_index_table.data<int>();
  // index_flags: flag the indices exist or not
  int index_flags_size = (table_size + 31) / 32;
  DenseTensor index_flags = phi::Empty<int>(dev_ctx, {index_flags_size});
  int* index_flags_ptr = index_flags.data<int>();
  phi::backends::gpu::GpuMemsetAsync(
      index_flags_ptr, 0, sizeof(int) * index_flags.numel(), dev_ctx.stream());

  if (subm) {
    DenseTensor tmp_rulebook = phi::Empty(dev_ctx, std::move(rulebook_meta));
    IntT* rulebook_ptr = tmp_rulebook.data<IntT>();
    DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
    int tmpidx = is2D ? 3 : 4;
    DenseTensor out_values =
        phi::Empty<T>(dev_ctx, {x.nnz(), kernel_sizes[tmpidx]});

    phi::Copy(dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &out_indices);

    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);
    GetOutIndexTable1<IntT><<<config.block_per_grid,
                              config.thread_per_block,
                              0,
                              dev_ctx.stream()>>>(out_indices.data<IntT>(),
                                                  non_zero_num,
                                                  d_x_dims,
                                                  index_flags_ptr,
                                                  is2D,
                                                  out_index_table_ptr);

    size_t cache_size =
        kernel_size * 2 * sizeof(int) +
        kernel_size * config.thread_per_block.x * 2 * sizeof(int);
    const int MAX_CACHE_SIZE = 48 * 1024;
    while (cache_size >= MAX_CACHE_SIZE) {
      config.thread_per_block.x /= 2;
      config.block_per_grid.x *= 2;
      PADDLE_ENFORCE_GE(
          config.thread_per_block.x,
          32,
          common::errors::Fatal("the shared memory is not enough"));
      cache_size = kernel_size * 2 * sizeof(int) +
                   kernel_size * config.thread_per_block.x * 2 * sizeof(int);
    }
    ProductSubmRuleBookKernel<IntT><<<config.block_per_grid.x,
                                      config.thread_per_block.x,
                                      cache_size,
                                      dev_ctx.stream()>>>(indices_ptr,
                                                          d_x_dims,
                                                          d_kernel_dims,
                                                          d_out_dims,
                                                          non_zero_num,
                                                          d_paddings,
                                                          d_dilations,
                                                          d_strides,
                                                          is2D,
                                                          index_flags_ptr,
                                                          out_index_table_ptr,
                                                          rulebook_ptr,
                                                          counter_ptr);

    out->SetMember(out_indices, out_values, out_dims, false);

    CallThrustScan(
        dev_ctx, counter_ptr, kernel_size, offsets_ptr, h_counter, h_offsets);

    dev_ctx.Wait();
    int rulebook_len = h_offsets[kernel_size - 1] + h_counter[kernel_size - 1];
    DenseTensor out_rulebook =
        phi::Empty<IntT>(dev_ctx, {rulebook_rows, rulebook_len});
    IntT* out_rulebook_ptr = out_rulebook.data<IntT>();
    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
    cache_size = kernel_size * 2 * sizeof(int);
    CopyRuleBook<IntT><<<config.block_per_grid,
                         config.thread_per_block,
                         cache_size,
                         dev_ctx.stream()>>>(counter_ptr,
                                             offsets_ptr,
                                             rulebook_ptr,
                                             rulebook_len,
                                             kernel_size,
                                             non_zero_num,
                                             out_rulebook_ptr);
    *rulebook = out_rulebook;

    return rulebook_len;

  } else {
    *rulebook = phi::Empty(dev_ctx, std::move(rulebook_meta));
    IntT* rulebook_ptr = rulebook->data<IntT>();
    ProductRuleBookKernel<IntT><<<config.block_per_grid.x,
                                  config.thread_per_block.x,
                                  kernel_size * sizeof(int),
                                  dev_ctx.stream()>>>(indices_ptr,
                                                      d_x_dims,
                                                      d_kernel_dims,
                                                      d_out_dims,
                                                      non_zero_num,
                                                      d_paddings,
                                                      d_dilations,
                                                      d_strides,
                                                      is2D,
                                                      rulebook_ptr,
                                                      counter_ptr);

    // 2. remove -1
#ifdef PADDLE_WITH_HIP
    IntT* last = thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
    IntT* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                                rulebook_ptr,
                                rulebook_ptr + rulebook_rows * rulebook_cols,
                                -1);

    IntT rulebook_len = (last - rulebook_ptr) / 2;

    CallThrustScan(
        dev_ctx, counter_ptr, kernel_size, offsets_ptr, h_counter, h_offsets);

    rulebook->Resize({rulebook_rows, static_cast<int>(rulebook_len)});
    // 3. sorted or merge the out index
    out_index->ResizeAndAllocate({static_cast<int>(rulebook_len)});
    DenseTensor unique_key =
        phi::Empty<int>(dev_ctx, {static_cast<int>(rulebook_len)});
    int* out_index_ptr = out_index->data<int>();
    int* unique_key_ptr = unique_key.data<int>();

    phi::backends::gpu::GpuMemsetAsync(
        unique_key_ptr, 0, sizeof(int), dev_ctx.stream());

    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
    size_t cache_size = sizeof(int) * config.thread_per_block.x;
    UniqueKernel<IntT><<<config.block_per_grid,
                         config.thread_per_block,
                         cache_size,
                         dev_ctx.stream()>>>(rulebook_ptr + rulebook_len,
                                             rulebook_len,
                                             index_flags_ptr,
                                             out_index_ptr,
                                             unique_key_ptr);

    int out_nnz = 0;
    phi::backends::gpu::GpuMemcpyAsync(&out_nnz,
                                       unique_key_ptr,
                                       sizeof(int),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx.stream());
    dev_ctx.Wait();

    const int threads = 256;
    const int blocks = (index_flags.numel() + threads - 1) / threads;
    GetOutIndicesCounter<<<blocks, threads, 0, dev_ctx.stream()>>>(
        index_flags_ptr, index_flags.numel(), out_index_table_ptr);
#ifdef PADDLE_WITH_HIP
    thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                           out_index_table_ptr,
                           out_index_table_ptr + blocks,
                           out_index_table_ptr);
    GetOutIndices<threads>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(index_flags_ptr,
                                                   index_flags.numel(),
                                                   out_index_table_ptr,
                                                   out_nnz,
                                                   out_index_ptr);

    const int64_t sparse_dim = is2D ? 3 : 4;
    phi::DenseTensor out_indices =
        phi::Empty<IntT>(dev_ctx, {sparse_dim, out_nnz});
    phi::DenseTensor out_values =
        phi::Empty<T>(dev_ctx, {out_nnz, kernel_sizes[sparse_dim]});
    out->SetMember(out_indices, out_values, out_dims, false);

    IntT* out_indices_ptr = out_indices.data<IntT>();

    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz, 1);
    GetOutIndexTable<IntT><<<config.block_per_grid,
                             config.thread_per_block,
                             0,
                             dev_ctx.stream()>>>(out_index_ptr,
                                                 out_nnz,
                                                 d_out_dims,
                                                 is2D,
                                                 out_index_table_ptr,
                                                 out_indices_ptr);
    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
    unique_value->ResizeAndAllocate({static_cast<int>(out_nnz * kernel_size)});
    int* unique_value_ptr = unique_value->data<int>();

    GroupIndices<<<config.block_per_grid,
                   config.thread_per_block,
                   0,
                   dev_ctx.stream()>>>(out_index_table_ptr,
                                       rulebook_len,
                                       kernel_size,
                                       rulebook_ptr + rulebook_len,
                                       out_index_ptr,
                                       unique_value_ptr);

    return rulebook_len;
  }
}

}  // namespace sparse
}  // namespace phi
