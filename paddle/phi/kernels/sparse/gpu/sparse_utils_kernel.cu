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

#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/rocsparse.h"
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/common_shape.h"

namespace phi {
namespace sparse {

template <typename T>
inline __device__ bool DevIsZero(const T* data, const int64_t cols) {
  const T zero = static_cast<T>(0);
  // TODO(zhangkaihuo): check the data is zero or not in parallen when cols > 1
  for (int64_t i = 0; i < cols; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

template <typename T>
__global__ void GetNonZeroNums(const T* dense_data,
                               const int rows,
                               const int cols,
                               int* non_zero_num,
                               int* temp_indices) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int counter;
  if (threadIdx.x == 0) counter = 0;
  __syncthreads();

  for (int i = tid; i < rows; i += gridDim.x * blockDim.x) {
    int index = -1;
    // TODO(zhangkaihuo): when cols=1, vectorization can be used
    if (!DevIsZero(dense_data + i * cols, cols)) {
      // use reductions?
      atomicAdd(&counter, 1);
      index = i;
    }
    temp_indices[i] = index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(non_zero_num, counter);
  }
}

template <typename T>
__global__ void GetNonZeroElementsAndIndices(const T* dense_data,
                                             const int64_t sparse_dim,
                                             const int64_t cols,
                                             const int64_t* x_dims,
                                             const int non_zero_num,
                                             const int* sparse_indices,
                                             int64_t* indices,
                                             T* sparse_data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int64_t sparse_index = sparse_indices[i];
    int64_t x_index = sparse_index;
    for (int64_t j = sparse_dim - 1; j >= 0; j--) {
      indices[j * non_zero_num + i] = sparse_index % x_dims[j];
      sparse_index /= x_dims[j];
    }

    for (int j = 0; j < cols; j++) {
      sparse_data[i * cols + j] = dense_data[x_index * cols + j];
    }
  }
}

template <typename T, typename Context>
void DenseToCooKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const int64_t sparse_dim,
                      SparseCooTensor* out) {
  const T* x_data = x.data<T>();
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_LE(sparse_dim,
                    x_dims.size(),
                    common::errors::InvalidArgument(
                        "sparse_dim must be less than the size of x.dims()"));
  PADDLE_ENFORCE_GT(
      sparse_dim, 0, common::errors::InvalidArgument("sparse_dim must be >0"));
  auto dims_2d = flatten_to_2d(x_dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];
  DenseTensor nums = phi::Empty<int32_t>(dev_ctx, {1});
  DenseTensor d_x_dims = phi::Empty<int64_t>(dev_ctx, {x_dims.size()});

  // 1. get numbers of non zero elements, and get the index of non zero elements
  int* nums_ptr = nums.data<int>();
  phi::backends::gpu::GpuMemsetAsync(
      nums_ptr, 0, sizeof(int), dev_ctx.stream());
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows, 1);

  DenseTensor temp_indices = phi::Empty<int32_t>(dev_ctx, {rows});
  int* temp_indices_ptr = temp_indices.data<int>();

  GetNonZeroNums<<<config.block_per_grid.x,
                   config.thread_per_block.x,
                   0,
                   dev_ctx.stream()>>>(
      x_data, rows, cols, nums_ptr, temp_indices_ptr);

#ifdef PADDLE_WITH_HIP
  thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                 temp_indices_ptr,
                 temp_indices_ptr + rows,
                 -1);

  // 2. copy non_zero_num to host, copy x_dims to device
  int non_zero_num = 0;
  phi::backends::gpu::GpuMemcpyAsync(&non_zero_num,
                                     nums_ptr,
                                     sizeof(int),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(d_x_dims.data<int64_t>(),
                                     x_dims.Get(),
                                     x_dims.size() * sizeof(x_dims[0]),
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  dev_ctx.Wait();  // wait the copy

  const auto values_dims =
      phi::funcs::sparse::InferDenseDims(x_dims, sparse_dim, non_zero_num);
  phi::DenseTensor indices = phi::Empty<int64_t>(
      dev_ctx, {sparse_dim, static_cast<int64_t>(non_zero_num)});
  int64_t* indices_data = indices.data<int64_t>();
  phi::DenseTensor values;
  values.Resize(values_dims);
  T* sparse_data = dev_ctx.template Alloc<T>(&values);

  // 3. calc indices by indices and get values by indices
  if (non_zero_num > 0) {
    config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);
    GetNonZeroElementsAndIndices<<<config.block_per_grid.x,
                                   config.thread_per_block.x,
                                   0,
                                   dev_ctx.stream()>>>(x_data,
                                                       sparse_dim,
                                                       cols,
                                                       d_x_dims.data<int64_t>(),
                                                       non_zero_num,
                                                       temp_indices_ptr,
                                                       indices_data,
                                                       sparse_data);
  }

  out->SetMember(indices, values, x_dims, true);
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

template <typename T, typename IntT>
void CsrToCooGPUKernel(const GPUContext& dev_ctx,
                       const SparseCsrTensor& x,
                       SparseCooTensor* out) {
  const DDim& x_dims = x.dims();
  const int64_t non_zero_num = x.cols().numel();
  int64_t sparse_dim = 2;
  if (x_dims.size() == 3) {
    sparse_dim = 3;
  }

  if (x.nnz() <= 0) {
#ifdef PADDLE_WITH_HIP
    DenseTensor indices = phi::Empty<int>(dev_ctx, {sparse_dim, non_zero_num});
#else
    DenseTensor indices = phi::Empty<IntT>(dev_ctx, {sparse_dim, non_zero_num});
#endif
    DenseTensor values = phi::EmptyLike<T, GPUContext>(dev_ctx, x.values());
    out->SetMember(indices, values, x_dims, true);
    return;
  }

// rocsparse_csr2coo only support index with type 'rocsparse_int' (aka 'int')
// now
#ifdef PADDLE_WITH_HIP
  const auto& csr_crows = Cast<IntT>(dev_ctx, x.crows(), DataType::INT32);
  const auto& csr_cols = Cast<IntT>(dev_ctx, x.cols(), DataType::INT32);
  const int* csr_crows_data = csr_crows.template data<int>();
  const int* csr_cols_data = csr_cols.template data<int>();
#else
  const auto& csr_crows = x.crows();
  const auto& csr_cols = x.cols();
  const IntT* csr_crows_data = csr_crows.data<IntT>();
  const IntT* csr_cols_data = csr_cols.data<IntT>();
#endif
  const auto& csr_values = x.values();
  const T* csr_values_data = csr_values.data<T>();

  int batches = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

#ifdef PADDLE_WITH_HIP
  DenseTensor indices = phi::Empty<int>(dev_ctx, {sparse_dim, non_zero_num});
  int* coo_indices = indices.data<int>();
  int* coo_rows_data = coo_indices;
  int* coo_cols_data = coo_rows_data + non_zero_num;
#else
  DenseTensor indices = phi::Empty<IntT>(dev_ctx, {sparse_dim, non_zero_num});
  DenseTensor offsets = phi::Empty<IntT>(dev_ctx, {batches});
  IntT* coo_indices = indices.data<IntT>();
  IntT* batch_ptr = x_dims.size() == 2 ? nullptr : coo_indices;
  IntT* coo_rows_data =
      x_dims.size() == 2 ? coo_indices : batch_ptr + non_zero_num;
  IntT* coo_cols_data = coo_rows_data + non_zero_num;
  IntT* offsets_ptr = batches == 1 ? nullptr : offsets.data<IntT>();
#endif
  DenseTensor values = phi::EmptyLike<T, GPUContext>(dev_ctx, csr_values);
  T* coo_values_data = values.data<T>();

  if (batches > 1) {
#ifdef PADDLE_WITH_HIP
    PADDLE_THROW(common::errors::Unimplemented(
        "'rocsparse_csr2coo' only supports batches "
        "with a value of 1 currently."));
#else
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, batches, 1);
    GetBatchSizes<IntT><<<config.block_per_grid.x, config.thread_per_block.x>>>(
        csr_crows_data, rows, batches, offsets_ptr);

    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
                           offsets_ptr,
                           offsets_ptr + batches,
                           offsets_ptr);
#endif
  }

#ifdef PADDLE_WITH_HIP
  dev_ctx.CusparseCall([&](rocsparse_handle handle) {
    phi::dynload::rocsparse_csr2coo(handle,
                                    csr_crows_data,
                                    non_zero_num,
                                    rows,
                                    coo_rows_data,
                                    rocsparse_index_base_zero);
  });
#else
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows, 1);
  config.block_per_grid.y = batches;
  ConvertCsrCrowsToCooRows<IntT>
      <<<config.block_per_grid, config.thread_per_block.x>>>(
          csr_crows_data, offsets_ptr, coo_rows_data, batch_ptr, rows);
#endif
  phi::backends::gpu::GpuMemcpyAsync(coo_cols_data,
                                     csr_cols_data,
#ifdef PADDLE_WITH_HIP
                                     sizeof(int) * non_zero_num,
#else
                                     sizeof(IntT) * non_zero_num,
#endif
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(coo_values_data,
                                     csr_values_data,
                                     sizeof(T) * non_zero_num,
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());

#ifdef PADDLE_WITH_HIP
  if (std::is_same<IntT, int64_t>::value)
    indices = Cast<int>(dev_ctx, indices, DataType::INT64);
#endif

  out->SetMember(indices, values, x_dims, true);
}

template <typename T, typename Context>
void CsrToCooKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.crows().dtype(), "CsrToCooGPUKernel", ([&] {
                                 CsrToCooGPUKernel<T, data_t>(dev_ctx, x, out);
                               }));
}

template <typename IntT>
__global__ void GetBatchesOffset(const IntT* batches_ptr,
                                 const int batches,
                                 const int non_zero_num,
                                 int* batches_offset) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    if (i == non_zero_num - 1 || batches_ptr[i] != batches_ptr[i + 1]) {
      const int start = batches_ptr[i];
      const int end = i == non_zero_num - 1 ? batches : batches_ptr[i + 1];
      for (int j = start; j < end; j++) {
        batches_offset[j] = i + 1;
      }
    }
  }
}

template <typename IntT>
__global__ void ConvertCooRowsToCsrCrows(
    const int* batches_offset,  // can be null if batches = 1
    const IntT* coo_rows_data,
    IntT* csr_crows_data,
    const int rows,
    const int64_t non_zero_num) {
  const int b = blockIdx.y;
  int batch_non_zero_num =
      batches_offset == nullptr ? non_zero_num : batches_offset[b];
  IntT batch_start = 0;
  if (b > 0) {
    batch_start = batches_offset[b - 1];
    batch_non_zero_num -= batch_start;
  }

  const IntT* coo_rows_ptr = coo_rows_data + batch_start;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < batch_non_zero_num; i += gridDim.x * blockDim.x) {
    if (i == 0) {
      for (IntT j = 0; j <= coo_rows_ptr[0]; j++) {
        csr_crows_data[b * (rows + 1) + j] = 0;
      }
    } else {
      for (IntT j = coo_rows_ptr[i - 1]; j < coo_rows_ptr[i]; j++) {
        csr_crows_data[b * (rows + 1) + j + 1] = i;
      }
    }
    if (i == batch_non_zero_num - 1) {
      for (IntT i = coo_rows_ptr[batch_non_zero_num - 1] + 1; i < rows + 1;
           i++) {
        csr_crows_data[b * (rows + 1) + i] = batch_non_zero_num;
      }
    }
  }
  if (batch_non_zero_num == 0) {
    for (int i = tid; i < rows + 1; i += gridDim.x * blockDim.x) {
      csr_crows_data[b * (rows + 1) + i] = 0;
    }
  }
}

template <typename T, typename IntT>
void CooToCsrGPUKernel(const GPUContext& dev_ctx,
                       const SparseCooTensor& x,
                       SparseCsrTensor* out) {
  const auto& x_dims = x.dims();
  bool valid = x_dims.size() == 2 || x_dims.size() == 3;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    common::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D matrix"));
  const int64_t non_zero_num = x.nnz();

  int batches = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

  phi::DenseTensor crows = phi::Empty<IntT>(dev_ctx, {batches * (rows + 1)});
  phi::DenseTensor cols = phi::Empty<IntT>(dev_ctx, {non_zero_num});
  phi::DenseTensor values = phi::EmptyLike<T, GPUContext>(dev_ctx, x.values());
  if (non_zero_num <= 0) {
    out->SetMember(crows, cols, values, x_dims);
    return;
  }
  IntT* csr_crows_data = crows.data<IntT>();
  IntT* csr_cols_data = cols.data<IntT>();
  T* csr_values_data = values.data<T>();

  const auto& coo_indices = x.indices();
  const auto& coo_values = x.values();
  const IntT* batches_ptr = coo_indices.data<IntT>();
  const IntT* coo_rows_data =
      x_dims.size() == 2 ? batches_ptr : batches_ptr + non_zero_num;
  const IntT* coo_cols_data = coo_rows_data + non_zero_num;
  const T* coo_values_data = coo_values.data<T>();

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, batches, 1);
  if (batches > 1) {
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);
    phi::DenseTensor batches_offset = phi::Empty<int>(dev_ctx, {batches});
    int* batches_offset_ptr = batches_offset.data<int>();
    phi::funcs::SetConstant<GPUContext, int> set_zero;
    // set zero if the nnz=0 of batches[0]
    set_zero(dev_ctx, &batches_offset, static_cast<IntT>(0));
    GetBatchesOffset<IntT><<<config.block_per_grid.x,
                             config.thread_per_block.x,
                             0,
                             dev_ctx.stream()>>>(
        batches_ptr, batches, non_zero_num, batches_offset_ptr);

    config.block_per_grid.y = batches;
    ConvertCooRowsToCsrCrows<IntT><<<config.block_per_grid,
                                     config.thread_per_block.x,
                                     0,
                                     dev_ctx.stream()>>>(
        batches_offset_ptr, coo_rows_data, csr_crows_data, rows, non_zero_num);
  } else {
    ConvertCooRowsToCsrCrows<IntT><<<config.block_per_grid.x,
                                     config.thread_per_block.x,
                                     0,
                                     dev_ctx.stream()>>>(
        nullptr, coo_rows_data, csr_crows_data, rows, non_zero_num);
  }

  phi::backends::gpu::GpuMemcpyAsync(csr_cols_data,
                                     coo_cols_data,
                                     sizeof(IntT) * non_zero_num,
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(csr_values_data,
                                     coo_values_data,
                                     sizeof(T) * non_zero_num,
                                     gpuMemcpyDeviceToDevice,
                                     dev_ctx.stream());
  out->SetMember(crows, cols, values, x_dims);
}

template <typename T, typename Context>
void CooToCsrKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    SparseCsrTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "CooToCsrGPUKernel", ([&] {
                                 CooToCsrGPUKernel<T, data_t>(dev_ctx, x, out);
                               }));
}

template <typename ValueT, typename IndicesT>
__global__ void KernelCooToDense(const IndicesT* indices,
                                 const int64_t* sparse_offsets,
                                 const ValueT* data,
                                 ValueT* dense_data,
                                 const IndicesT non_zero_num,
                                 const int64_t base_offset,
                                 const int64_t sparse_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      dense_data[index * base_offset + j] = data[i * base_offset + j];
    }
  }
}

template <typename T, typename IntT>
void CooToDenseGPUKernel(const GPUContext& dev_ctx,
                         const SparseCooTensor& x,
                         DenseTensor* out) {
  const auto non_zero_num = x.nnz();
  const auto dense_dims = x.dims();
  const auto indices = x.indices();
  const auto values = x.values();
  const auto indices_dims = indices.dims();
  int64_t sparse_dim = indices_dims[0];
  if (indices_dims.size() == 1) {
    sparse_dim = 1;
  }
  const int64_t dense_dim = values.dims().size() - 1;

  const auto place = dev_ctx.GetPlace();
  dev_ctx.template Alloc<T>(out);

  T* out_data = out->data<T>();
  phi::backends::gpu::GpuMemsetAsync(
      out_data, 0, sizeof(T) * out->numel(), dev_ctx.stream());

  if (x.nnz() <= 0) {
    return;
  }

  const T* x_data = values.data<T>();
  int64_t base_offset = 1;
  for (int64_t i = 0; i < dense_dim; i++) {
    base_offset *= dense_dims[sparse_dim + i];
  }
  std::vector<int64_t> sparse_offsets(sparse_dim);
  int64_t offset = 1;
  for (int i = sparse_dim - 1; i >= 0; i--) {
    sparse_offsets[i] = offset;
    offset *= dense_dims[i];
  }

  DenseTensor d_sparse_offsets = Empty<int64_t>(dev_ctx, {sparse_dim});

  phi::backends::gpu::GpuMemcpyAsync(d_sparse_offsets.data<int64_t>(),
                                     sparse_offsets.data(),
                                     sparse_dim * sizeof(int64_t),
                                     gpuMemcpyHostToDevice,
                                     dev_ctx.stream());

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);

  KernelCooToDense<T, IntT>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(indices.data<IntT>(),
                             d_sparse_offsets.data<int64_t>(),
                             x_data,
                             out_data,
                             non_zero_num,
                             base_offset,
                             sparse_dim);
}

template <typename T, typename Context>
void CooToDenseKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      DenseTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "CooToDenseGPUKernel", ([&] {
        CooToDenseGPUKernel<T, data_t>(dev_ctx, x, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(dense_to_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(csr_to_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrToCooKernel,
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
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(coo_to_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CooToCsrKernel,
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
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(dense_to_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(coo_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CooToDenseKernel,
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
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(csr_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrToDenseKernel,
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
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(values_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ValuesCooKernel,
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
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(values_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ValuesCsrKernel,
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
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(indices_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::IndicesCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_coo_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooTensorKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
