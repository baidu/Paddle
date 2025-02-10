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

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"
#include "paddle/phi/kernels/sparse/conv_kernel.h"

namespace phi {
namespace sparse {

using Dims4D = phi::funcs::sparse::Dims4D;

// TODO(zhangkaihuo): After the GatherCUDAKernel is migrated to phi, replace
// this kernel with phi::GatherCUDAKernel;
// Vectorization can be used to improve read and write bandwidth
/**
 * brief: gather data from params according to indices
 * params: the inputs
 * indices: the indices you want to gather
 * output: the outputs
 * index_size: the size of indices
 * slice_size: slice size corresponding to each index, here is the channel size
 **/
template <typename T, typename IndexT = int>
__global__ void GatherKernel(const T* params,
                             const IndexT* indices,
                             T* output,
                             size_t index_size,
                             size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}

template <typename Context, typename IntT = int>
inline IntT* SortedAndUniqueIndex(const Context& dev_ctx,
                                  const IntT* rulebook_ptr,
                                  const int len,
                                  DenseTensor* out_index,
                                  DenseTensor* unique_key,
                                  DenseTensor* unique_value) {
  phi::IndexKernel<int, kps::IdentityFunctor<int>>(
      dev_ctx, out_index, kps::IdentityFunctor<int>());
  phi::IndexKernel<int, kps::IdentityFunctor<int>>(
      dev_ctx, unique_value, kps::IdentityFunctor<int>());

  phi::backends::gpu::GpuMemcpyAsync(unique_key->data<IntT>(),
                                     rulebook_ptr,
                                     sizeof(IntT) * len,
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyDeviceToDevice,
#else
                                     cudaMemcpyDeviceToDevice,
#endif
                                     dev_ctx.stream());
// compared with thrust::sort_by_key, thrust::merge_by_key may achieved higher
// performance, but thrust::merge_by_key limited by data size
#ifdef PADDLE_WITH_HIP
  thrust::sort_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::sort_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                      unique_key->data<IntT>(),
                      unique_key->data<IntT>() + len,
                      out_index->data<int>());

  // 4. unique
  thrust::pair<IntT*, int*> new_end =
#ifdef PADDLE_WITH_HIP
      thrust::unique_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
      thrust::unique_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                            unique_key->data<IntT>(),
                            unique_key->data<IntT>() + len,
                            unique_value->data<int>());
  return new_end.first;
}

/**
 * @brief: update the out index and indices
 * unique_keys: save the index of the output feature list
 * unique_values: indicates the index of key before deduplication
 * out_indexes: indicates the position of the output index in the rulebook
 * rulebook_len: indicates the length of rulebook
 * out_dims: indicates the output dims
 * out_indices: the indices of output, out_indices = IndexToPoint(unique_keys)
 * rulebook_out_indices: the output index in rulebook
 **/
template <typename T>
__global__ void UpdateIndexKernel(const T* unique_keys,
                                  const int* unique_values,
                                  const int* out_indexes,
                                  const int64_t non_zero_num,
                                  const int rulebook_len,
                                  const Dims4D out_dims,
                                  T* out_indices,
                                  T* rulebook_out_indices) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    const T index = unique_keys[i];
    T batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<Dims4D>(
        index, out_dims, &batch, &x, &y, &z);
    // get out indices
    out_indices[i] = batch;
    out_indices[i + non_zero_num] = z;
    out_indices[i + non_zero_num * 2] = y;
    out_indices[i + non_zero_num * 3] = x;

    // update rulebook
    int start = unique_values[i];
    int end = i == non_zero_num - 1 ? rulebook_len : unique_values[i + 1];
    // max(end-start) = kernel_size
    for (T j = start; j < end; j++) {
      rulebook_out_indices[out_indexes[j]] = i;
    }
  }
}

template <typename IntT>
__global__ void UpdateOutIndexAndCounterAfterLowerBound(
    const IntT* x_indices,
    const IntT* bound_out,
    const int rulebook_len,
    const int kernel_size,
    const int64_t non_zero_num,
    IntT* rulebook_ptr,
    IntT* out_indices,
    int* counter_ptr) {
  extern __shared__ int cache_count[];
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    cache_count[i] = 0;
  }
  __syncthreads();

  CUDA_KERNEL_LOOP_TYPE(i, rulebook_len, int64_t) {
    int j = bound_out[i];
    if (j >= 0 && j < non_zero_num && out_indices[i] == x_indices[j]) {
      out_indices[i] = j;
    } else {
      // mask this position will be remove
      int kernel_index = rulebook_ptr[i];
      rulebook_ptr[i + rulebook_len] = -1;
      rulebook_ptr[i + 2 * rulebook_len] = -1;
      rulebook_ptr[i] = -1;
      atomicAdd(&cache_count[kernel_index], 1);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    atomicSub(&counter_ptr[i], cache_count[i]);
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
                                      const bool subm,
                                      T* rulebook,
                                      int* counter,
                                      T* in_indices) {
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
    T in_z = x_indices[i + non_zero_num];
    T in_y = x_indices[i + 2 * non_zero_num];
    T in_x = x_indices[i + 3 * non_zero_num];
    if (subm) {
      in_indices[i] = PointToIndex(batch, in_x, in_y, in_z, x_dims);
    }
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
            T out_z = (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            T out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            T out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            in_i = i;
            out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                batch, out_x, out_y, out_z, out_dims);
            atomicAdd(&counter_buf[kernel_index], 1);
            kernel_i = kernel_index;
          }
          rulebook[kernel_index * non_zero_num + i] = kernel_i;
          rulebook[kernel_index * non_zero_num + offset + i] = in_i;
          rulebook[kernel_index * non_zero_num + offset * 2 + i] = out_index;
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
                    std::vector<int>* h_counter,
                    std::vector<int>* h_offsets) {
  auto indices_dtype = phi::CppTypeToDataType<IntT>::Type();
  const int64_t non_zero_num = x.nnz();
  const auto& indices = x.indices();
  const IntT* indices_ptr = indices.data<IntT>();
  DenseTensor in_indices = phi::Empty<Context>(
      dev_ctx, DenseTensorMeta(indices_dtype, {x.nnz()}, DataLayout::NCHW));
  int* counter_ptr = counter_per_kernel->data<int>();
  int* offsets_ptr = offsets_per_kernel->data<int>();
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  const int rulebook_rows = 3;
  const int rulebook_cols = kernel_size * non_zero_num;
  DenseTensorMeta rulebook_meta(
      indices_dtype, {rulebook_rows, rulebook_cols}, DataLayout::NCHW);
  *rulebook = phi::Empty(dev_ctx, std::move(rulebook_meta));
  IntT* rulebook_ptr = rulebook->data<IntT>();

  const auto x_dims = x.dims();
  Dims4D d_x_dims(x_dims[0], x_dims[3], x_dims[2], x_dims[1]);
  Dims4D d_kernel_dims(1, kernel_sizes[2], kernel_sizes[1], kernel_sizes[0]);
  Dims4D d_out_dims(out_dims[0], out_dims[3], out_dims[2], out_dims[1]);
  Dims4D d_paddings(1, paddings[2], paddings[1], paddings[0]);
  Dims4D d_strides(1, strides[2], strides[1], strides[0]);
  Dims4D d_dilations(1, dilations[2], dilations[1], dilations[0]);
  // 1. product rule book
  phi::funcs::SetConstant<Context, int> set_zero;
  set_zero(dev_ctx, counter_per_kernel, 0);
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);

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
                                                    subm,
                                                    rulebook_ptr,
                                                    counter_ptr,
                                                    in_indices.data<IntT>());

// 2. remove -1
#ifdef PADDLE_WITH_HIP
  IntT* last = thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
  IntT* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                              rulebook_ptr,
                              rulebook_ptr + rulebook_rows * rulebook_cols,
                              -1);

  phi::funcs::sparse::DistanceKernel<IntT><<<1, 1, 0, dev_ctx.stream()>>>(
      rulebook_ptr, last, rulebook_ptr + 3 * kernel_size * non_zero_num - 1);
  IntT rulebook_len = 0;
  phi::backends::gpu::GpuMemcpyAsync(
      &rulebook_len,
      rulebook_ptr + 3 * kernel_size * non_zero_num - 1,
      sizeof(IntT),
#ifdef PADDLE_WITH_HIP
      hipMemcpyDeviceToHost,
#else
      cudaMemcpyDeviceToHost,
#endif
      dev_ctx.stream());
  dev_ctx.Wait();
  rulebook_len /= 3;

  if (subm) {
    // At present, hashtable is not used to map the input and output indexes.
    // At present, the intermediate output index is generated by normal
    // convolution,
    // and then the intermediate output index is subtracted from the input index
    // to obain the rulebook.

    // call lower_bound to get the real index of out_index
    const IntT* in_indices_ptr = in_indices.data<IntT>();
    IntT* out_indices_ptr = rulebook_ptr + 2 * rulebook_len;
    DenseTensor bound = phi::Empty(
        dev_ctx,
        DenseTensorMeta(
            indices_dtype, {static_cast<int>(rulebook_len)}, DataLayout::NCHW));
    IntT* bound_ptr = bound.data<IntT>();
#ifdef PADDLE_WITH_HIP
    thrust::lower_bound(thrust::hip::par.on(dev_ctx.stream()),
#else
    thrust::lower_bound(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                        in_indices_ptr,
                        in_indices_ptr + in_indices.numel(),
                        out_indices_ptr,
                        out_indices_ptr + rulebook_len,
                        bound_ptr);

    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);

    UpdateOutIndexAndCounterAfterLowerBound<<<config.block_per_grid,
                                              config.thread_per_block,
                                              kernel_size * sizeof(int),
                                              dev_ctx.stream()>>>(
        in_indices_ptr,
        bound.data<IntT>(),
        rulebook_len,
        kernel_size,
        x.nnz(),
        rulebook_ptr,
        out_indices_ptr,
        counter_ptr);

// remove -1
#ifdef PADDLE_WITH_HIP
    IntT* last = thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
    IntT* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                                rulebook_ptr,
                                rulebook_ptr + 3 * rulebook_len,
                                -1);
    phi::funcs::sparse::DistanceKernel<IntT>
        <<<1, 1, 0, dev_ctx.stream()>>>(rulebook_ptr, last, bound_ptr);
    phi::backends::gpu::GpuMemcpyAsync(&rulebook_len,
                                       bound_ptr,
                                       sizeof(IntT),
#ifdef PADDLE_WITH_HIP
                                       hipMemcpyDeviceToHost,
#else
                                       cudaMemcpyDeviceToHost,
#endif
                                       dev_ctx.stream());
    dev_ctx.Wait();
    rulebook_len /= 3;
  }

#ifdef PADDLE_WITH_HIP
  thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         counter_ptr,
                         counter_ptr + kernel_size,
                         offsets_ptr);

  phi::backends::gpu::GpuMemcpyAsync(&(*h_counter)[0],
                                     counter_ptr,
                                     kernel_size * sizeof(int),
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyDeviceToHost,
#else
                                     cudaMemcpyDeviceToHost,
#endif
                                     dev_ctx.stream());

  phi::backends::gpu::GpuMemcpyAsync(&(*h_offsets)[0],
                                     offsets_ptr,
                                     kernel_size * sizeof(int),
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyDeviceToHost,
#else
                                     cudaMemcpyDeviceToHost,
#endif
                                     dev_ctx.stream());

  rulebook->Resize({rulebook_rows, static_cast<int>(rulebook_len)});

  if (!subm) {
    // 3. sorted or merge the out index
    out_index->ResizeAndAllocate({static_cast<int>(rulebook_len)});
    unique_value->ResizeAndAllocate({static_cast<int>(rulebook_len)});
    DenseTensor unique_key = phi::Empty(
        dev_ctx,
        DenseTensorMeta(
            indices_dtype, {static_cast<int>(rulebook_len)}, DataLayout::NCHW));
    int* out_index_ptr = out_index->data<int>();
    int* unique_value_ptr = unique_value->data<int>();
    IntT* unique_key_ptr = unique_key.data<IntT>();

    IntT* new_end =
        SortedAndUniqueIndex<Context, IntT>(dev_ctx,
                                            rulebook_ptr + 2 * rulebook_len,
                                            rulebook_len,
                                            out_index,
                                            &unique_key,
                                            unique_value);
    // thrust::distance doesn't support stream parameters
    // const int out_non_zero_num = thrust::distance(unique_key_ptr,
    // new_end.first);
    phi::funcs::sparse::DistanceKernel<IntT><<<1, 1, 0, dev_ctx.stream()>>>(
        unique_key_ptr,
        new_end,
        rulebook_ptr + rulebook_rows * rulebook_cols - 1);
    IntT out_non_zero_num = 0;
#ifdef PADDLE_WITH_HIP
    phi::backends::gpu::GpuMemcpyAsync(
        &out_non_zero_num,
        rulebook_ptr + rulebook_rows * rulebook_cols - 1,
        sizeof(IntT),
        hipMemcpyDeviceToHost,
        dev_ctx.stream());
#else
    phi::backends::gpu::GpuMemcpyAsync(
        &out_non_zero_num,
        rulebook_ptr + rulebook_rows * rulebook_cols - 1,
        sizeof(IntT),
        cudaMemcpyDeviceToHost,
        dev_ctx.stream());
#endif
    dev_ctx.Wait();

    // 5. update out_indices and rulebook by unique_value_ptr
    const int64_t sparse_dim = 4;
    DenseTensorMeta indices_meta(
        indices_dtype, {sparse_dim, out_non_zero_num}, DataLayout::NCHW);
    DenseTensorMeta values_meta(
        x.dtype(), {out_non_zero_num, kernel_sizes[4]}, x.values().layout());
    phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

    IntT* out_indices_ptr = out_indices.data<IntT>();

    config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_non_zero_num, 1);
    UpdateIndexKernel<IntT>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(unique_key_ptr,
                               unique_value_ptr,
                               out_index_ptr,
                               out_non_zero_num,
                               rulebook_len,
                               d_out_dims,
                               out_indices_ptr,
                               rulebook_ptr + 2 * rulebook_len);
    out->SetMember(out_indices, out_values, out_dims, true);
  } else {
    DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
    DenseTensor out_values = phi::Empty(
        dev_ctx,
        DenseTensorMeta(
            x.dtype(), {x.nnz(), kernel_sizes[4]}, x.values().layout()));
    phi::Copy(dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &out_indices);
    out->SetMember(out_indices, out_values, out_dims, true);
  }
  return rulebook_len;
}

}  // namespace sparse
}  // namespace phi
