// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/matrix_reduce.h"
#include "paddle/phi/kernels/lu_solve_grad_kernel.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void LUSolveGradKernel(const Context& dev_ctx,
                       const DenseTensor& lu,
                       const DenseTensor& pivot,
                       const DenseTensor& b,
                       const DenseTensor& out,
                       const DenseTensor& dout,
                       bool trans,
                       DenseTensor* dlu,
                       DenseTensor* db) {
  // get broadcast dim
  std::vector<int64_t> lu_bst_dims_vec;
  std::vector<int64_t> b_bst_dims_vec;
  std::tie(lu_bst_dims_vec, b_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(lu, b);
  IntArray lu_bst_dims(lu_bst_dims_vec);
  IntArray b_bst_dims(b_bst_dims_vec);

  // Tensor broadcast to temp 'b_bst'
  DenseTensor b_bst = phi::Empty<T, Context>(dev_ctx, b_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, b, b_bst_dims, &b_bst);

  // Tensor broadcast to temp 'lu_bst'
  DenseTensor lu_bst = phi::Empty<T, Context>(dev_ctx, lu_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, lu, lu_bst_dims, &lu_bst);

  // reuse forward to calculate db_bst using transposed operation
  DenseTensor db_bst = phi::Empty<T, Context>(dev_ctx, b_bst_dims);
  LUSolveKernel<T, Context>(dev_ctx, lu_bst, pivot, dout, !trans, &db_bst);

  // get 'db' according to 'db_bst'
  db->Resize(b.dims());
  dev_ctx.template Alloc<T>(db);
  if (db_bst.dims() == b.dims()) {
    Copy<Context>(dev_ctx, db_bst, dev_ctx.GetPlace(), false, db);
  } else {
    funcs::MatrixReduceSumFunctor<T, Context> functor;
    functor(dev_ctx, db_bst, db);
    db->Resize(b.dims());
  }

  // calculate out's conjugate for complex
  DenseTensor out_conj = Conj<T, Context>(dev_ctx, out);
  out_conj = phi::TransposeLast2Dim<T>(dev_ctx, out_conj);

  DenseTensor commonterm = phi::Empty<T, Context>(dev_ctx, lu_bst_dims);
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  blas.MatMul(db_bst,
              phi::funcs::CreateMatrixDescriptor(db_bst.dims(), 0, false),
              out_conj,
              phi::funcs::CreateMatrixDescriptor(out_conj.dims(), 0, false),
              static_cast<T>(1),
              &commonterm,
              static_cast<T>(0));

  // calculate commonterm's conjugate for complex
  DenseTensor commonterm_conj = Conj<T, Context>(dev_ctx, commonterm);
  commonterm_conj = phi::TransposeLast2Dim<T>(dev_ctx, commonterm_conj);

  phi::AddKernel<T>(dev_ctx, commonterm, commonterm_conj, &commonterm);

  DenseTensor dlu_bst = phi::Empty<T, Context>(dev_ctx, lu_bst_dims);
  if (trans) {
    blas.MatMul(lu_bst,
                phi::funcs::CreateMatrixDescriptor(lu_bst.dims(), 0, false),
                commonterm,
                phi::funcs::CreateMatrixDescriptor(commonterm.dims(), 0, false),
                static_cast<T>(-1),
                &dlu_bst,
                static_cast<T>(0));
  } else {
    blas.MatMul(commonterm,
                phi::funcs::CreateMatrixDescriptor(commonterm.dims(), 0, false),
                lu_bst,
                phi::funcs::CreateMatrixDescriptor(lu_bst.dims(), 0, false),
                static_cast<T>(-1),
                &dlu_bst,
                static_cast<T>(0));
  }

  // get 'dlu' according to 'dlu_bst'
  dlu->Resize(lu.dims());
  dev_ctx.template Alloc<T>(dlu);
  if (dlu_bst.dims() == lu.dims()) {
    Copy<Context>(dev_ctx, dlu_bst, dev_ctx.GetPlace(), false, dlu);
  } else {
    funcs::MatrixReduceSumFunctor<T, Context> functor;
    functor(dev_ctx, dlu_bst, dlu);
    dlu->Resize(lu.dims());
  }
}

}  // namespace phi
