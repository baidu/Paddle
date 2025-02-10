// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"

namespace phi {

template <typename T, typename Context>
void LuSolveKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const std::string& trans,
                   DenseTensor* out) {
  // Get lu matrix dimensions
  auto lu_dims = lu.dims();
  // Get x matrix dimensions
  auto x_dims = x.dims();

  // Allocate output tensor
  dev_ctx.template Alloc<T>(out);
  // Copy RHS data to output (will be overwritten with solution)
  // phi::Copy(dev_ctx, x, x.place(), false, out);
  *out = Transpose2DTo6D<Context, T>(dev_ctx, x);

  // Prepare LAPACK parameters
  char trans_char = (trans == "N") ? 'N' : ((trans == "T") ? 'T' : 'C');
  int n_int = lu_dims[lu_dims.size() - 1];
  int nrhs_int = x_dims[x_dims.size() - 1];
  int lda = std::max(1, n_int);  // Leading dimension of A (LU matrix)
  int ldb = std::max(1, n_int);  // Leading dimension of B (RHS/solution matrix)
  int info = 0;

  auto outdims = out->dims();
  auto outrank = outdims.size();
  int batchsize = product(common::slice_ddim(outdims, 0, outrank - 2));
  auto out_data = out->data<T>();
  auto lu_data = reinterpret_cast<T*>(const_cast<T*>(lu.data<T>()));
  auto pivots_data = reinterpret_cast<int*>(const_cast<int*>(pivots.data<int>()));

  for (int i = 0; i < batchsize; i++) {
    auto* out_data_item = &out_data[i * lda * n_int];
    auto* lu_data_item = &lu_data[i * ldb * nrhs_int];
    auto* pivots_data_item = &pivots_data[i * n_int];
    phi::funcs::lapackLuSolve<T>(
      trans_char,
      n_int,
      nrhs_int,
      lu_data_item,
      lda,
      pivots_data_item,
      out_data_item,
      ldb,
      &info);
    PADDLE_ENFORCE_EQ(
      info,
      0,
      phi::errors::PreconditionNotMet(
      "LU solve failed with error code %d. Check if matrix is singular.",
      info));
  }
  *out = Transpose2DTo6D<Context, T>(dev_ctx, *out);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    lu_solve, CPU, ALL_LAYOUT, phi::LuSolveKernel, float, double) {}