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

#include <set>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/sparse/conv_kernel.h"

namespace phi {
namespace sparse {

using Dims4D = phi::funcs::sparse::Dims4D;

// such as: kernel(3, 3, 3), kernel_size = 27
// counter_per_weight: (kernel_size)
// TODO(zhangkaihuo): optimize performance with multithreading
template <typename T, typename Context, typename IntT = int>
void ProductRuleBook(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const DDim& out_dims,
                     const bool subm,
                     DenseTensor* rulebook,
                     int* counter_per_kernel) {
  const bool is2D = out_dims.size() == 4 ? true : false;
  const int64_t non_zero_num = x.nnz();
  const auto& indices = x.indices();
  const IntT* indices_ptr = indices.data<IntT>();
  int kernel_size = is2D ? kernel_sizes[0] * kernel_sizes[1]
                         : kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  memset(counter_per_kernel, 0, kernel_size * sizeof(int));

  int rulebook_len = 0;
  // calc the rulebook_len
  const auto& x_dims = x.dims();

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

  const Dims4D c_x_dims(xdim0, xdim1, xdim2, xdim3);
  const Dims4D c_kernel_dims(kdim0, kdim1, kdim2, kdim3);
  const Dims4D c_out_dims(odim0, odim1, odim2, odim3);
  const Dims4D c_paddings(pdim0, pdim1, pdim2, pdim3);
  const Dims4D c_strides(sdim0, sdim1, sdim2, sdim3);
  const Dims4D c_dilations(ddim0, ddim1, ddim2, ddim3);

  std::set<IntT> hash_in;
  if (subm) {
    for (int i = 0; i < non_zero_num; i++) {
      IntT batch = indices_ptr[i];
      IntT in_z = is2D ? 0 : indices_ptr[i + non_zero_num];
      IntT in_y = is2D ? indices_ptr[i + non_zero_num]
                       : indices_ptr[i + 2 * non_zero_num];
      IntT in_x = is2D ? indices_ptr[i + 2 * non_zero_num]
                       : indices_ptr[i + 3 * non_zero_num];
      IntT index = phi::funcs::sparse::PointToIndex<Dims4D>(
          batch, in_x, in_y, in_z, c_x_dims);
      hash_in.insert(index);
    }
  }

  auto f_calc_rulebook = [&](IntT* rulebook_ptr) {
    int kernel_index = 0, rulebook_index = 0;
    int zceil = is2D ? 1 : kernel_sizes[0];
    int yceil = is2D ? kernel_sizes[0] : kernel_sizes[1];
    int xceil = is2D ? kernel_sizes[1] : kernel_sizes[2];
    for (int kz = 0; kz < zceil; kz++) {
      for (int ky = 0; ky < yceil; ky++) {
        for (int kx = 0; kx < xceil; kx++) {
          ++kernel_index;
          for (int64_t i = 0; i < non_zero_num; i++) {
            IntT batch = indices_ptr[i];
            IntT in_z = is2D ? 0 : indices_ptr[i + non_zero_num];
            IntT in_y = is2D ? indices_ptr[i + non_zero_num]
                             : indices_ptr[i + 2 * non_zero_num];
            IntT in_x = is2D ? indices_ptr[i + 2 * non_zero_num]
                             : indices_ptr[i + 3 * non_zero_num];

            IntT out_z =
                is2D ? 0
                     : (in_z + paddings[0] - kz * dilations[0]) / strides[0];
            IntT out_y =
                (in_y + c_paddings[2] - ky * c_dilations[2]) / c_strides[2];
            IntT out_x =
                (in_x + c_paddings[3] - kx * c_dilations[3]) / c_strides[3];
            if (phi::funcs::sparse::Check(c_x_dims,
                                          c_kernel_dims,
                                          c_paddings,
                                          c_dilations,
                                          c_strides,
                                          in_x,
                                          in_y,
                                          in_z,
                                          kx,
                                          ky,
                                          kz)) {
              if (subm) {
                IntT out_index = phi::funcs::sparse::PointToIndex<Dims4D>(
                    batch, out_x, out_y, out_z, c_out_dims);
                if (hash_in.find(out_index) == hash_in.end()) {
                  continue;
                }
              }

              if (rulebook_ptr == nullptr) {
                counter_per_kernel[kernel_index - 1] += 1;
                ++rulebook_len;
              } else {
                rulebook_ptr[rulebook_index] = kernel_index - 1;
                rulebook_ptr[rulebook_index + rulebook_len] = i;  // in_i
                rulebook_ptr[rulebook_index + rulebook_len * 2] =
                    phi::funcs::sparse::PointToIndex<Dims4D>(
                        batch, out_x, out_y, out_z, c_out_dims);  // out_index
                ++rulebook_index;
              }
            }
          }
        }
      }
    }
  };

  f_calc_rulebook(nullptr);
  // alloc the rulebook
  *rulebook = phi::Empty(dev_ctx,
                         DenseTensorMeta(phi::CppTypeToDataType<IntT>::Type(),
                                         {3, rulebook_len},
                                         DataLayout::NCHW));
  IntT* rulebook_ptr = rulebook->data<IntT>();
  f_calc_rulebook(rulebook_ptr);
}

template <typename T, typename Context, typename IntT = int>
void UpdateRulebookAndOutIndex(const Context& dev_ctx,
                               const SparseCooTensor& x,
                               const int kernel_size UNUSED,
                               const int out_channels,
                               const DDim& out_dims,
                               DenseTensor* rulebook,
                               SparseCooTensor* out) {
  const bool is2D = out_dims.size() == 4 ? true : false;

  std::set<IntT> tmp_indices;
  int n = rulebook->dims()[1];
  IntT* rulebook_ptr = rulebook->data<IntT>();
  for (int i = 0; i < n; i++) {
    tmp_indices.insert(rulebook_ptr[i + n * 2]);
  }

  int out_non_zero_num = tmp_indices.size();
  const int64_t sparse_dim = is2D ? 3 : 4;
  DenseTensorMeta indices_meta(phi::CppTypeToDataType<IntT>::Type(),
                               {sparse_dim, out_non_zero_num},
                               DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {out_non_zero_num, out_channels}, x.values().layout());
  phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));
  IntT* out_indices_ptr = out_indices.data<IntT>();
  int i = 0;

  int odim0, odim1, odim2, odim3;
  odim0 = out_dims[0];
  odim1 = is2D ? out_dims[2] : out_dims[3];
  odim2 = is2D ? out_dims[1] : out_dims[2];
  odim3 = is2D ? 1 : out_dims[1];
  const Dims4D c_out_dims(odim0, odim1, odim2, odim3);

  for (auto it = tmp_indices.begin(); it != tmp_indices.end(); it++, i++) {
    const IntT index = *it;
    IntT batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<Dims4D>(
        index, c_out_dims, &batch, &x, &y, &z);
    out_indices_ptr[i] = batch;
    if (is2D) {
      out_indices_ptr[i + out_non_zero_num] = y;
      out_indices_ptr[i + out_non_zero_num * 2] = x;
    } else {
      out_indices_ptr[i + out_non_zero_num] = z;
      out_indices_ptr[i + out_non_zero_num * 2] = y;
      out_indices_ptr[i + out_non_zero_num * 3] = x;
    }
  }
  for (i = 0; i < n; i++) {
    IntT out_index = rulebook_ptr[i + n * 2];
    rulebook_ptr[i + n * 2] =
        std::distance(tmp_indices.begin(), tmp_indices.find(out_index));
  }

  out->SetMember(out_indices, out_values, out_dims, true);
}

template <typename T, typename IntT = int>
void Gather(
    const T* x, const IntT* indices, const int n, const int channels, T* out) {
  for (int i = 0; i < n; i++) {
    IntT real_i = indices[i];
    memcpy(out + i * channels, x + real_i * channels, channels * sizeof(T));
  }
}

template <typename T, typename IntT = int>
void Scatter(
    const T* x, const IntT* indices, const int n, const int channels, T* out) {
  for (int i = 0; i < n; i++) {
    IntT real_i = indices[i];
    for (int j = 0; j < channels; j++) {
      out[real_i * channels + j] += x[i * channels + j];
    }
  }
}

}  // namespace sparse
}  // namespace phi
