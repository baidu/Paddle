// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>
#include <algorithm>
#include <complex>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/impl/lstsq_kernel_impl.h"
#include "paddle/phi/kernels/impl/qr_kernel_impl.h"
#include "paddle/phi/kernels/impl/tril_triu_kernel_impl.h"
#include "paddle/phi/kernels/lstsq_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/triangular_solve_kernel.h"

namespace phi {

enum class LapackDriverType : int { Gels, Gelsd, Gelsy, Gelss };

template <typename T, typename Context>
void LstsqKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const Scalar& rcond_scalar,
                 const std::string& driver_string,
                 DenseTensor* solution,
                 DenseTensor* residuals,
                 DenseTensor* rank,
                 DenseTensor* singular_values) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  int dim_size = x_dims.size();
  int m = x_dims[dim_size - 2];
  int n = x_dims[dim_size - 1];
  int nrhs = y_dims[dim_size - 1];
  int min_mn = std::min(m, n);
  int max_mn = std::max(m, n);
  int k = min_mn;

  int x_stride = phi::GetMatrixStride(x_dims);
  int y_stride = phi::GetMatrixStride(y_dims);
  int tau_stride = min_mn;
  int batch_count = phi::GetBatchCount(x_dims);

  T rcond = rcond_scalar.to<T>();

  DenseTensor* new_x = new DenseTensor();
  new_x->Resize(common::make_ddim({batch_count, m, n}));
  dev_ctx.template Alloc<T>(new_x);
  phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), true, new_x);

  DenseTensor* new_y = new DenseTensor();
  new_y->Resize(common::make_ddim({batch_count, m, nrhs}));
  dev_ctx.template Alloc<T>(new_y);
  phi::Copy<Context>(dev_ctx, y, dev_ctx.GetPlace(), true, new_y);

  // Prepare tau
  auto tau_dims_vec = common::vectorize<int>(x_dims);
  tau_dims_vec.pop_back();
  tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;

  DenseTensor* tau = new DenseTensor();
  tau->Resize(common::make_ddim(tau_dims_vec));
  auto tau_data = dev_ctx.template Alloc<T>(tau);

  if (m >= n) {
    DenseTensor tmp_x = phi::TransposeLast2Dim<T>(dev_ctx, *new_x);
    DenseTensor tmp_y = phi::TransposeLast2Dim<T>(dev_ctx, *new_y);
    auto x_data = tmp_x.data<T>();
    auto y_data = tmp_y.data<T>();

    // step 1, compute QR factorization using geqrf
    BatchedGeqrf<Context, T>(
        dev_ctx, batch_count, m, n, x_data, m, tau_data, x_stride, tau_stride);

    // Step 2, Y <- Q^H Y
    BatchedOrmqr<Context, T>(dev_ctx,
                             true,
                             true,
                             batch_count,
                             m,
                             nrhs,
                             k,
                             x_data,
                             x_stride,
                             tau_data,
                             tau_stride,
                             y_data,
                             y_stride);

    DenseTensor trans_r = phi::TransposeLast2Dim<T>(dev_ctx, tmp_x);
    DenseTensor slice_r =
        phi::funcs::Slice<T>(dev_ctx, trans_r, {-2}, {0}, {min_mn});
    DenseTensor* res_r = new DenseTensor();
    res_r->Resize(common::make_ddim({batch_count, min_mn, min_mn}));
    dev_ctx.template Alloc<T>(res_r);
    phi::TrilTriuKernel<T>(dev_ctx, slice_r, 0, false, res_r);

    DenseTensor trans_y = phi::TransposeLast2Dim<T>(dev_ctx, tmp_y);
    DenseTensor slice_y =
        phi::funcs::Slice<T>(dev_ctx, trans_y, {-2}, {0}, {min_mn});

    // Step 3, solve R X = Y
    phi::TriangularSolveKernel<T, Context>(
        dev_ctx, *res_r, slice_y, true, false, false, solution);

  } else {
    auto x_data = dev_ctx.template Alloc<T>(new_x);
    auto y_data = dev_ctx.template Alloc<T>(new_y);

    // step 1, compute QR factorization using geqrf
    BatchedGeqrf<Context, T>(
        dev_ctx, batch_count, n, m, x_data, n, tau_data, x_stride, tau_stride);

    // Step 2, solve R^H Z = Y
    DenseTensor trans_r = phi::TransposeLast2Dim<T>(dev_ctx, *new_x);
    DenseTensor slice_r =
        phi::funcs::Slice<T>(dev_ctx, trans_r, {-2}, {0}, {min_mn});
    DenseTensor* res_r = new DenseTensor();
    res_r->Resize(common::make_ddim({batch_count, min_mn, min_mn}));
    dev_ctx.template Alloc<T>(res_r);
    phi::TrilTriuKernel<T>(dev_ctx, slice_r, 0, false, res_r);

    phi::TriangularSolveKernel<T, Context>(
        dev_ctx, *res_r, *new_y, true, true, false, solution);

    // Step 3, X <- Q Z
    BatchedOrgqr<Context, T>(dev_ctx,
                             batch_count,
                             n,
                             m,
                             min_mn,
                             x_data,
                             n,
                             tau_data,
                             x_stride,
                             tau_stride);

    DenseTensor trans_q = phi::TransposeLast2Dim<T>(dev_ctx, *new_x);
    DenseTensor slice_q =
        phi::funcs::Slice<T>(dev_ctx, trans_q, {-1}, {0}, {m});
    DenseTensor solu_tensor =
        phi::Matmul<T>(dev_ctx, slice_q, *solution, false, false);
    phi::Copy<Context>(
        dev_ctx, solu_tensor, dev_ctx.GetPlace(), true, solution);
  }

  if (batch_count == 1) solution->Resize(common::make_ddim({n, nrhs}));
  GetResidualsTensor<Context, T>(dev_ctx, x, y, solution, residuals);
}

}  // namespace phi

PD_REGISTER_KERNEL(lstsq, GPU, ALL_LAYOUT, phi::LstsqKernel, float, double) {}
