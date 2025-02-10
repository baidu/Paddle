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

#include "paddle/phi/kernels/sparse/mask_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.h"

namespace phi::sparse {

template <typename T, typename IntT>
void MaskCooCPUKernel(const CPUContext& dev_ctx,
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
  const int sparse_dim = mask.sparse_dim();

  DenseTensor out_indices = phi::EmptyLike<T>(dev_ctx, indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, values);

  // the out_indices is same as indices of mask
  phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &out_indices);

  T* out_values_ptr = out_values.data<T>();
  const T* x_ptr = x.data<T>();

  const int64_t non_zero_num = mask.nnz();
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = static_cast<int>(dims_2d[1]);
  const IntT* indices_ptr = indices.data<IntT>();

  std::vector<IntT> sparse_offsets(sparse_dim);

  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      dims, sparse_dim, sparse_offsets.data());

  for (int64_t i = 0; i < non_zero_num; i++) {
    int64_t index = phi::funcs::sparse::CoordinateToIndex<IntT>(
        indices_ptr, sparse_offsets.data(), non_zero_num, sparse_dim, i);
    memcpy(out_values_ptr + i * cols, x_ptr + index * cols, cols * sizeof(T));
  }

  out->SetMember(out_indices, out_values, dims, true);
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
      mask.indices().dtype(), "MaskCooCPUKernel", ([&] {
        MaskCooCPUKernel<T, data_t>(dev_ctx, x, mask, out);
      }));
}

template <typename T, typename IntT>
void MaskCsr2DCPUKernel(const CPUContext& dev_ctx,
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

  int64_t numel = 0;
  for (int64_t i = 0; i < mask_crows.numel() - 1; ++i) {
    for (int64_t j = mask_crows.data<IntT>()[i];
         j < mask_crows.data<IntT>()[i + 1];
         ++j) {
      IntT col_idx = mask_cols.data<IntT>()[numel];

      out_values.data<T>()[numel] =
          x.data<T>()[(i / x.dims()[0]) * x.dims()[1] +
                      (i % x.dims()[0]) * x.dims()[1] + col_idx];

      ++numel;
    }
  }

  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

template <typename T, typename IntT>
void MaskCsr3DCPUKernel(const CPUContext& dev_ctx,
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

  int64_t numel = 0;
  for (int64_t i = 0; i < mask_crows.numel() - 1; ++i) {
    for (int64_t j = mask_crows.data<IntT>()[i];
         j < mask_crows.data<IntT>()[i + 1];
         ++j) {
      IntT col_idx = mask_cols.data<IntT>()[numel];

      out_values.data<T>()[numel] =
          x.data<T>()[(i / (mask_crows.numel() / x.dims()[0])) *
                          (x.dims()[1] * x.dims()[2]) +
                      (i % (mask_crows.numel() / x.dims()[0])) * x.dims()[2] +
                      col_idx];

      ++numel;
    }
  }

  out->SetMember(out_crows, out_cols, out_values, x.dims());
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
        mask.crows().dtype(), "MaskCsr2DCPUKernel", ([&] {
          MaskCsr2DCPUKernel<T, data_t>(dev_ctx, x, mask, out);
        }));
  } else if (x_dims.size() == 3) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        mask.crows().dtype(), "MaskCsr3DCPUKernel", ([&] {
          MaskCsr3DCPUKernel<T, data_t>(dev_ctx, x, mask, out);
        }));
  } else {
    // throw exception
    common::errors::InvalidArgument(
        "mask_as for Sparse CSR Tensor only support 2-D or 3-D, but got "
        "%d-D.",
        x_dims.size());
  }
}

template <typename T, typename IntT>
void MaskHelperCooCPUKernel(const CPUContext& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mask_indices,
                            DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      mask_indices.dims().size(),
      2,
      common::errors::InvalidArgument("the mask_indices must be 2-D tensor"));

  const int32_t sparse_dim = x.sparse_dim();

  std::vector<IntT> sparse_offsets(sparse_dim), x_indices(x.nnz()),
      mask_out_indices(mask_indices.dims()[1]);
  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      x.dims(), sparse_dim, sparse_offsets.data());

  phi::funcs::sparse::FlattenIndices(x.indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     x_indices.data());
  phi::funcs::sparse::FlattenIndices(mask_indices.data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     mask_out_indices.data());

  std::unordered_map<IntT, uint64_t> x_indices_map;
  for (uint64_t i = 0; i < x_indices.size(); i++) {
    x_indices_map[x_indices[i]] = i;
  }

  *out = phi::EmptyLike<T>(dev_ctx, x.values());
  phi::funcs::SetConstant<CPUContext, T> set_zero;
  set_zero(dev_ctx, out, static_cast<T>(0));
  T* out_ptr = out->data<T>();
  const int64_t stride =
      x.dims().size() == sparse_dim ? 1 : x.values().dims()[1];
  const T* in_ptr = x.values().data<T>();
  // TODO(zhangkaihuo): multithreading can be used for acceleration
  for (uint64_t i = 0; i < mask_out_indices.size(); i++) {
    auto iter = x_indices_map.find(mask_out_indices[i]);
    if (iter != x_indices_map.end()) {
      memcpy(out_ptr + i * stride,
             in_ptr + iter->second * stride,
             stride * sizeof(T));
    }
  }
}

/**
 * @brief filter values from x.values() using mask_indices
 */
template <typename T, typename Context>
void MaskHelperCooKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& mask_indices,
                         DenseTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "MaskHelperCooCPUKernel", ([&] {
        MaskHelperCooCPUKernel<T, data_t>(dev_ctx, x, mask_indices, out);
      }));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(mask_helper_coo,
                   CPU,
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
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCooKernel,
                   float,
                   double,
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
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCsrKernel,
                   float,
                   double,
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
