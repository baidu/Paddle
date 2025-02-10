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

#include <thrust/execution_policy.h>

#include "paddle/phi/kernels/sparse/mask_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.cu.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
__global__ void MaskKernel(const T* x_ptr,
                           const IntT* indices_ptr,
                           const int64_t* sparse_offsets,
                           const int64_t non_zero_num,
                           const int cols,
                           const int sparse_dim,
                           T* out_values_ptr) {
  CUDA_KERNEL_LOOP_TYPE(i, non_zero_num * cols, int64_t) {
    int64_t out_i = i / cols;
    int64_t col_i = i - out_i * cols;
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices_ptr[j * non_zero_num + out_i] * sparse_offsets[j];
    }
    out_values_ptr[out_i * cols + col_i] = x_ptr[index * cols + col_i];
  }
}

template <typename T, typename IntT>
void MaskCooGPUKernel(const GPUContext& dev_ctx,
                      const DenseTensor& x,
                      const SparseCooTensor& mask,
                      SparseCooTensor* out) {
  const DDim& dims = x.dims();
  PADDLE_ENFORCE_EQ(x.dims(),
                    mask.dims(),
                    common::errors::InvalidArgument(
                        "the input x and mask must have the shape"));
  const DenseTensor& indices = mask.indices();
  const DenseTensor& values = mask.values();
  DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, values);
  if (mask.nnz() <= 0) {
    out->SetMember(out_indices, out_values, dims, true);
    return;
  }

  const int sparse_dim = mask.sparse_dim();
  DenseTensor sparse_offsets = phi::Empty<GPUContext>(
      dev_ctx,
      DenseTensorMeta(DataType::INT64, {sparse_dim}, DataLayout::NCHW));
  std::vector<int64_t> h_sparse_offsets(sparse_dim);
  phi::funcs::sparse::CalcOffsetsPerDim(
      dims, sparse_dim, h_sparse_offsets.data());

  phi::backends::gpu::GpuMemcpyAsync(sparse_offsets.data<int64_t>(),
                                     &h_sparse_offsets[0],
                                     sizeof(int64_t) * sparse_dim,
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &out_indices);

  const IntT* indices_ptr = indices.data<IntT>();
  T* out_values_ptr = out_values.data<T>();
  const T* x_ptr = x.data<T>();
  const int64_t non_zero_num = mask.nnz();
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = dims_2d[1];

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num * cols, 1);
  MaskKernel<T, IntT>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_ptr,
          indices_ptr,
          sparse_offsets.data<int64_t>(),
          non_zero_num,
          cols,
          sparse_dim,
          out_values_ptr);

  out->SetMember(out_indices, out_values, dims, true);
}

template <typename IntT>
__global__ void ConvertCsrCrowsToCooRows(const IntT* crows_ptr,
                                         const IntT* crows_offsets,
                                         IntT* rows_ptr,
                                         IntT* batch_ptr,
                                         const int rows) {
  const int b = blockIdx.y;
  const int64_t offset = crows_offsets ? crows_offsets[b] : 0;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    for (int j = crows_ptr[b * (rows + 1) + i];
         j < crows_ptr[b * (rows + 1) + i + 1];
         j++) {
      rows_ptr[offset + j] = i;
      if (batch_ptr) {
        batch_ptr[offset + j] = b;
      }
    }
  }
}

template <typename IntT>
__global__ void GetBatchSizes(const IntT* crows,
                              const int rows,
                              const int batches,
                              IntT* batch_sizes) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < batches) {
    batch_sizes[tid] = crows[tid * (rows + 1) + rows];
  }
}

template <typename T, typename IntT>
void MaskCsr2DGPUKernel(const GPUContext& dev_ctx,
                        const DenseTensor& x,
                        const SparseCsrTensor& mask,
                        SparseCsrTensor* out) {
  const DenseTensor& mask_cols = mask.cols();
  const DenseTensor& mask_crows = mask.crows();
  int64_t num_non_zeros = mask.nnz();

  DenseTensor out_cols = phi::EmptyLike<IntT>(dev_ctx, mask_cols);
  DenseTensor out_crows = phi::EmptyLike<IntT>(dev_ctx, mask_crows);
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {num_non_zeros});

  phi::Copy(dev_ctx, mask_cols, dev_ctx.GetPlace(), false, &out_cols);
  phi::Copy(dev_ctx, mask_crows, dev_ctx.GetPlace(), false, &out_crows);

  const DDim& dims = x.dims();
  const int64_t non_zero_num = mask.nnz();
  int64_t sparse_dim = 2;
  DenseTensor sparse_offsets = phi::Empty<IntT>(dev_ctx, {sparse_dim});
  std::vector<int64_t> h_sparse_offsets(sparse_dim);
  phi::funcs::sparse::CalcOffsetsPerDim(
      dims, sparse_dim, h_sparse_offsets.data());

  phi::backends::gpu::GpuMemcpyAsync(sparse_offsets.data<int64_t>(),
                                     &h_sparse_offsets[0],
                                     sizeof(int64_t) * sparse_dim,
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  const auto& csr_crows = mask.crows();
  const auto& csr_cols = mask.cols();
  const IntT* csr_crows_data = csr_crows.data<IntT>();
  const IntT* csr_cols_data = csr_cols.data<IntT>();

  const int batches = 1;
  const int rows = dims[0];
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = dims_2d[1];

  DenseTensor indices = phi::Empty<IntT>(dev_ctx, {sparse_dim, non_zero_num});
  IntT* coo_indices = indices.data<IntT>();
  IntT* batch_ptr = nullptr;
  IntT* coo_rows_data = coo_indices;
  IntT* coo_cols_data = coo_rows_data + non_zero_num;
  IntT* offsets_ptr = nullptr;

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows, 1);
  config.block_per_grid.y = batches;
  ConvertCsrCrowsToCooRows<IntT>
      <<<config.block_per_grid, config.thread_per_block.x>>>(
          csr_crows_data, offsets_ptr, coo_rows_data, batch_ptr, rows);
  phi::backends::gpu::GpuMemcpyAsync(coo_cols_data,
                                     csr_cols_data,
                                     sizeof(IntT) * non_zero_num,
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());

  const T* x_ptr = x.data<T>();
  const IntT* indices_ptr = coo_indices;
  T* out_values_ptr = out_values.data<T>();

  auto config_mask =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num * cols, 1);
  MaskKernel<T, IntT><<<config_mask.block_per_grid,
                        config_mask.thread_per_block,
                        0,
                        dev_ctx.stream()>>>(x_ptr,
                                            indices_ptr,
                                            sparse_offsets.data<int64_t>(),
                                            non_zero_num,
                                            cols,
                                            sparse_dim,
                                            out_values_ptr);

  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

template <typename T, typename IntT>
void MaskCsr3DGPUKernel(const GPUContext& dev_ctx,
                        const DenseTensor& x,
                        const SparseCsrTensor& mask,
                        SparseCsrTensor* out) {
  const DenseTensor& mask_cols = mask.cols();
  const DenseTensor& mask_crows = mask.crows();
  int64_t num_non_zeros = mask.nnz();

  DenseTensor out_cols = phi::EmptyLike<IntT>(dev_ctx, mask_cols);
  DenseTensor out_crows = phi::EmptyLike<IntT>(dev_ctx, mask_crows);
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {num_non_zeros});

  phi::Copy(dev_ctx, mask_cols, dev_ctx.GetPlace(), false, &out_cols);
  phi::Copy(dev_ctx, mask_crows, dev_ctx.GetPlace(), false, &out_crows);

  const DDim& dims = x.dims();
  const int64_t non_zero_num = mask.nnz();
  int64_t sparse_dim = 3;
  DenseTensor sparse_offsets = phi::Empty<IntT>(dev_ctx, {sparse_dim});
  std::vector<int64_t> h_sparse_offsets(sparse_dim);
  phi::funcs::sparse::CalcOffsetsPerDim(
      dims, sparse_dim, h_sparse_offsets.data());

  phi::backends::gpu::GpuMemcpyAsync(sparse_offsets.data<int64_t>(),
                                     &h_sparse_offsets[0],
                                     sizeof(int64_t) * sparse_dim,
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  const auto& csr_crows = mask.crows();
  const auto& csr_cols = mask.cols();
  const IntT* csr_crows_data = csr_crows.data<IntT>();
  const IntT* csr_cols_data = csr_cols.data<IntT>();

  const int batches = dims[0];
  const int rows = dims[1];
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = dims_2d[1];

  DenseTensor indices = phi::Empty<IntT>(dev_ctx, {sparse_dim, non_zero_num});
  DenseTensor offsets = phi::Empty<IntT>(dev_ctx, {batches});
  IntT* coo_indices = indices.data<IntT>();
  IntT* batch_ptr = coo_indices;
  IntT* coo_rows_data = batch_ptr + non_zero_num;
  IntT* coo_cols_data = coo_rows_data + non_zero_num;
  IntT* offsets_ptr = offsets.data<IntT>();

  auto config_batch =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, batches, 1);
  GetBatchSizes<IntT>
      <<<config_batch.block_per_grid.x, config_batch.thread_per_block.x>>>(
          csr_crows_data, rows, batches, offsets_ptr);

#ifdef PADDLE_WITH_HIP
  thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         offsets_ptr,
                         offsets_ptr + batches,
                         offsets_ptr);

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows, 1);
  config.block_per_grid.y = batches;
  ConvertCsrCrowsToCooRows<IntT>
      <<<config.block_per_grid, config.thread_per_block.x>>>(
          csr_crows_data, offsets_ptr, coo_rows_data, batch_ptr, rows);
  phi::backends::gpu::GpuMemcpyAsync(coo_cols_data,
                                     csr_cols_data,
                                     sizeof(IntT) * non_zero_num,
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());

  const T* x_ptr = x.data<T>();
  const IntT* indices_ptr = coo_indices;
  T* out_values_ptr = out_values.data<T>();

  auto config_mask =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num * cols, 1);
  MaskKernel<T, IntT><<<config_mask.block_per_grid,
                        config_mask.thread_per_block,
                        0,
                        dev_ctx.stream()>>>(x_ptr,
                                            indices_ptr,
                                            sparse_offsets.data<int64_t>(),
                                            non_zero_num,
                                            cols,
                                            sparse_dim,
                                            out_values_ptr);

  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

/**
 * @brief Filter the DenseTensor x by the
 * mask.indices() and output a SparseCooTensor
 * x and mask must have the same shape.
 **/
template <typename T, typename Context>
void MaskAsCooKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const SparseCooTensor& mask,
                     SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      mask.indices().dtype(), "MaskCooGPUKernel", ([&] {
        MaskCooGPUKernel<T, data_t>(dev_ctx, x, mask, out);
      }));
}

/**
 * @brief Filter the DenseTensor x by the
 * mask.crows(), mask.cols() and output a SparseCsrTensor
 * x and mask must have the same shape.
 **/
template <typename T, typename Context>
void MaskAsCsrKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const SparseCsrTensor& mask,
                     SparseCsrTensor* out) {
  const phi::DDim& x_dims = x.dims();
  if (x_dims.size() == 2) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        mask.crows().dtype(), "MaskCsr2DGPUKernel", ([&] {
          MaskCsr2DGPUKernel<T, data_t>(dev_ctx, x, mask, out);
        }));
  } else if (x_dims.size() == 3) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        mask.crows().dtype(), "MaskCsr3DGPUKernel", ([&] {
          MaskCsr3DGPUKernel<T, data_t>(dev_ctx, x, mask, out);
        }));
  } else {
    // throw exception
    common::errors::InvalidArgument(
        "mask_as for Sparse CSR Tensor only support 2-D or 3-D, but got "
        "%d-D.",
        x_dims.size());
  }
}

template <typename IntT>
__global__ void MaskTable(const IntT* x_indices,
                          const int n,
                          int* index_flags,
                          int* table) {
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    int index = x_indices[i];
    phi::funcs::sparse::SetBits(index, index_flags);
    table[index] = i;
  }
}

template <typename T, typename IntT, int VecSize>
__global__ void MaskCopy(const IntT* mask_indices,
                         const int* index_flags,
                         const int* table,
                         const int n,
                         const int stride,
                         const T* x_values,
                         T* out_values) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  CUDA_KERNEL_LOOP_TYPE(i, n, int64_t) {
    const int mask_index = mask_indices[i];
    const bool flag = phi::funcs::sparse::TestBits(mask_index, index_flags);
    if (flag) {
      int j = table[mask_index];
      for (int k = 0; k < stride; k += VecSize) {
        LoadT vec_x;
        phi::Load<T, VecSize>(x_values + j * stride + k, &vec_x);
        phi::Store<T, VecSize>(vec_x, out_values + i * stride + k);
      }
    }
  }
}

template <typename T, typename IntT>
void MaskHelperCooGPUKernel(const GPUContext& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mask_indices,
                            DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      mask_indices.dims().size(),
      2,
      common::errors::InvalidArgument("the mask_indices must be 2-D tensor"));

  const int32_t sparse_dim = x.sparse_dim();
  auto indices_dtype = phi::CppTypeToDataType<IntT>::Type();

  std::vector<IntT> sparse_offsets(sparse_dim);

  DenseTensorMeta x_indices_meta(indices_dtype, {x.nnz()}, DataLayout::NCHW);
  DenseTensorMeta mask_indices_meta(
      indices_dtype, {mask_indices.dims()[1]}, DataLayout::NCHW);
  DenseTensorMeta sparse_offset_meta(
      indices_dtype, {sparse_dim}, DataLayout::NCHW);

  DenseTensor x_indices =
      phi::Empty<GPUContext>(dev_ctx, std::move(x_indices_meta));
  DenseTensor mask_meta_indices =
      phi::Empty<GPUContext>(dev_ctx, std::move(mask_indices_meta));
  DenseTensor bound_out =
      phi::Empty<GPUContext>(dev_ctx, std::move(mask_indices_meta));
  DenseTensor d_sparse_offsets =
      phi::Empty<GPUContext>(dev_ctx, std::move(sparse_offset_meta));
  IntT* x_indices_ptr = x_indices.data<IntT>();
  IntT* mask_indices_ptr = mask_meta_indices.data<IntT>();
  IntT* bound_out_ptr = bound_out.data<IntT>();

  // 1. calc the offsets of per dim
  phi::funcs::sparse::CalcOffsetsPerDim(
      x.dims(), sparse_dim, sparse_offsets.data());
  // 2. copy sparse_offsets to device
  phi::backends::gpu::GpuMemcpyAsync(d_sparse_offsets.data<IntT>(),
                                     sparse_offsets.data(),
                                     sizeof(IntT) * sparse_dim,
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  // 3. flatten x indices and mask indices
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_indices.numel(), 1);
  phi::funcs::sparse::FlattenIndicesKernel<<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
      x.indices().data<IntT>(),
      d_sparse_offsets.data<IntT>(),
      x_indices.numel(),
      sparse_dim,
      x_indices_ptr);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, mask_meta_indices.numel(), 1);
  phi::funcs::sparse::FlattenIndicesKernel<<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
      mask_indices.data<IntT>(),
      d_sparse_offsets.data<IntT>(),
      mask_meta_indices.numel(),
      sparse_dim,
      mask_indices_ptr);

  int table_size = 1;
  auto x_dims = x.dims();
  for (int i = 0; i < sparse_dim; i++) {
    table_size *= x_dims[i];
  }
  DenseTensor table = phi::Empty<int>(dev_ctx, {table_size});
  DenseTensor index_flags = phi::Empty<int>(dev_ctx, {(table_size + 31) / 32});
  phi::backends::gpu::GpuMemsetAsync(index_flags.data<int>(),
                                     0,
                                     index_flags.numel() * sizeof(int),
                                     dev_ctx.stream());
  const int64_t stride =
      x.dims().size() == sparse_dim ? 1 : x.values().dims()[1];
  *out = phi::EmptyLike<T>(dev_ctx, x.values());
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  set_zero(dev_ctx, out, static_cast<T>(0));
  T* out_ptr = out->data<T>();
  config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_indices.numel(), 1);
  MaskTable<<<config.block_per_grid,
              config.thread_per_block,
              0,
              dev_ctx.stream()>>>(x_indices_ptr,
                                  x_indices.numel(),
                                  index_flags.data<int>(),
                                  table.data<int>());
  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, mask_meta_indices.numel(), 1);

  const int VecBytes = 16;
  const int VecSize = VecBytes / sizeof(T);
  if (stride % VecSize == 0) {
    MaskCopy<T, IntT, VecSize><<<config.block_per_grid,
                                 config.thread_per_block,
                                 0,
                                 dev_ctx.stream()>>>(mask_indices_ptr,
                                                     index_flags.data<int>(),
                                                     table.data<int>(),
                                                     mask_meta_indices.numel(),
                                                     stride,
                                                     x.values().data<T>(),
                                                     out_ptr);
  } else {
    MaskCopy<T, IntT, 1><<<config.block_per_grid,
                           config.thread_per_block,
                           0,
                           dev_ctx.stream()>>>(mask_indices_ptr,
                                               index_flags.data<int>(),
                                               table.data<int>(),
                                               mask_meta_indices.numel(),
                                               stride,
                                               x.values().data<T>(),
                                               out_ptr);
  }
}

template <typename T, typename Context>
void MaskHelperCooKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& mask_indices,
                         DenseTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "MaskHelperCooGPUKernel", ([&] {
        MaskHelperCooGPUKernel<T, data_t>(dev_ctx, x, mask_indices, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(mask_helper_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskHelperCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(mask_as_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(mask_as_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
