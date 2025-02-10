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

#include "paddle/phi/kernels/sparse/softmax_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/cpu_vec.h"
#include "paddle/phi/kernels/funcs/sparse/softmax.h"
#include "paddle/phi/kernels/softmax_grad_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void SoftmaxCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& out,
                          const SparseCsrTensor& dout,
                          int axis,
                          SparseCsrTensor* dx) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    common::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, dout, dx);
  auto out_dim = out.dims();
  auto out_rank = out_dim.size();

  int batch_size = 1;
  int row_number = 1;
  for (int i = 0; i < out_rank - 1; ++i) {
    if (i < out_rank - 2) {
      batch_size *= static_cast<int>(out_dim[i]);
    } else if (i == out_rank - 2) {
      row_number = static_cast<int>(out_dim[i]);
    }
  }

  const DenseTensor& out_crows = out.non_zero_crows();
  const DenseTensor& out_values = out.non_zero_elements();
  const DenseTensor& dout_values = dout.non_zero_elements();
  DenseTensor* dx_values = dx->mutable_non_zero_elements();

  int row_nnz = 0;
  const T* out_data = out_values.data<T>();
  const T* dout_data = dout_values.data<T>();
  T* dx_data = dx_values->data<T>();

  // dx = (dout - sum(dout * out)) * out
  PD_VISIT_BASE_INTEGRAL_TYPES(
      out.non_zero_crows().dtype(), "SoftmaxCsrGradKernel", ([&] {
        const data_t* out_crows_data = out_crows.data<data_t>();
        for (int i = 0; i < batch_size; ++i) {
          for (int j = 0; j < row_number; ++j) {
            int crow_idx = i * (row_number + 1) + j;
            row_nnz = static_cast<int>(out_crows_data[crow_idx + 1] -
                                       out_crows_data[crow_idx]);

            T sum = 0;
            phi::funcs::vec_mul_reduce<T, backends::cpu::avx>(
                row_nnz, dout_data, out_data, &sum);
            phi::funcs::vec_add_bias<T, backends::cpu::avx>(
                row_nnz, static_cast<T>(-1) * sum, dout_data, dx_data);
            phi::funcs::vec_mul<T, backends::cpu::avx>(
                row_nnz, dx_data, out_data, dx_data);

            out_data = out_data + row_nnz;
            dout_data = dout_data + row_nnz;
            dx_data = dx_data + row_nnz;
          }
        }
      }));
}

template <typename T, typename IntT, typename Context>
void SoftmaxCooGradCPUKernel(const Context& dev_ctx,
                             const SparseCooTensor& out,
                             const SparseCooTensor& dout,
                             int axis,
                             SparseCooTensor* dx) {
  const auto& out_indices = out.indices();
  const auto& out_values = out.values();
  const auto& out_dims = out.dims();
  auto sparse_dim = out.sparse_dim();
  auto sizes = common::vectorize<IntT>(out_dims);
  const auto& grad_indices = dout.indices();
  const auto& grad_values = dout.values();
  auto grad_nnz = dout.nnz();

  *(dx->mutable_indices()) = out_indices;
  DenseTensor* values = dx->mutable_values();
  values->Resize(out_dims);
  values->set_meta(out_values.meta());
  dev_ctx.template Alloc<T>(values);

  auto out_offsets = phi::funcs::sparse::GetOffsets(out_indices, sizes, -1);
  auto grad_offsets = phi::funcs::sparse::GetOffsets(grad_indices, sizes, -1);

  int dim = axis < 0 ? out_dims.size() + axis : axis;

  if (dim >= sparse_dim) {
    bool is_same_offset = out_offsets == grad_offsets;
    PADDLE_ENFORCE_EQ(
        is_same_offset,
        true,
        common::errors::Unimplemented(
            "SparseCooTensor only support same offsets for softmax."));

    SoftmaxGradKernel<T, Context>(
        dev_ctx, out_values, grad_values, dim - sparse_dim + 1, values);
    return;
  }

  auto nnz = out.nnz();
  IntT nvalues = std::accumulate(sizes.begin() + sparse_dim,
                                 sizes.end(),
                                 static_cast<IntT>(1),
                                 std::multiplies<>());

  DenseTensor values_2(*values);
  values_2.Resize(common::make_ddim({nnz, nvalues}));

  DenseTensor out_values_2(out_values);
  out_values_2.Resize(common::make_ddim({nnz, nvalues}));

  DenseTensor grad_values_2(grad_values);
  grad_values_2.Resize(common::make_ddim({nnz, nvalues}));
  std::map<IntT, std::vector<IntT>> pools;
  phi::funcs::sparse::GetPoolsSoftmax(out_indices, sizes, dim, &pools);

  for (size_t p = 0; p < pools.size(); p++) {
    auto pool_indices = pools[p];
    if (pool_indices.empty()) continue;

    std::vector<T> tmp_row(nvalues, 0);

    /* Compute tmp = - sum_j output_j * grad_j */
    for (IntT i : pool_indices) {
      auto out_values_row = out_values_2.data<T>() + i * nvalues;
      auto low = std::lower_bound(
          grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
      auto j = low - grad_offsets.begin();

      if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
        auto grad_values_row = grad_values_2.data<T>() + j * nvalues;
        for (IntT k = 0; k < nvalues; k++) {
          tmp_row[k] -= (*(out_values_row + k)) * (*(grad_values_row + k));
        }
      }
    }

    /* Compute grad_input = output * (grad + tmp)*/
    for (IntT i : pool_indices) {
      auto out_values_row = out_values_2.data<T>() + i * nvalues;
      auto values_row = values_2.data<T>() + i * nvalues;
      auto low = std::lower_bound(
          grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
      auto j = low - grad_offsets.begin();

      if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
        auto grad_values_row = grad_values_2.data<T>() + j * nvalues;
        for (IntT k = 0; k < nvalues; k++) {
          *(values_row + k) =
              (*(out_values_row + k)) * ((*(grad_values_row + k)) + tmp_row[k]);
        }
      } else {
        for (IntT k = 0; k < nvalues; k++) {
          *(values_row + k) = (*out_values_row + k) * (tmp_row[k]);
        }
      }
    }
  }
}

template <typename T, typename Context>
void SoftmaxCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& out,
                          const SparseCooTensor& dout,
                          int axis,
                          SparseCooTensor* dx) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      out.indices().dtype(), "SoftmaxCooGradCPUKernel", ([&] {
        SoftmaxCooGradCPUKernel<T, data_t, Context>(
            dev_ctx, out, dout, axis, dx);
      }));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(softmax_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(softmax_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCooGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
