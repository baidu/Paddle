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
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
class LUSolveFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  bool trans,
                  int M,
                  int N,
                  T* Adata,
                  int lda,
                  const int* ipiv,
                  T* Bdata,
                  int* devInfo);
};

template <typename T, typename Context>
void LUSolveKernel(const Context& dev_ctx,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const DenseTensor& b,
                   bool trans,
                   DenseTensor* out) {
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

  // Calculate b_bst's conjugate for complex
  DenseTensor b_bst_conj = Conj<T, Context>(dev_ctx, b_bst);
  b_bst_conj = phi::TransposeLast2Dim<T>(dev_ctx, b_bst_conj);

  // Calculate lu_bst's conjugate for complex
  DenseTensor lu_bst_conj = Conj<T, Context>(dev_ctx, lu_bst);
  lu_bst_conj = phi::TransposeLast2Dim<T>(dev_ctx, lu_bst_conj);
  T* lu_bst_conj_data = lu_bst_conj.data<T>();

  // Copy b_bst's conjugate to 'result'
  DenseTensor result;
  Copy<Context>(dev_ctx, b_bst_conj, dev_ctx.GetPlace(), false, &result);
  T* res_data = result.data<T>();

  int lu_bst_ndim = lu_bst_dims_vec.size();
  int M = static_cast<int>(lu_bst_dims_vec[lu_bst_ndim - 2]);
  int N = static_cast<int>(lu_bst_dims_vec[lu_bst_ndim - 1]);
  int batchsize =
      product(common::slice_ddim(lu_bst.dims(), 0, lu_bst_ndim - 2));

  DenseTensor info = phi::Empty<int, Context>(dev_ctx, IntArray({batchsize}));
  int* info_data = info.data<int>();

  const int* ipiv_data = pivots.data<int>();

  LUSolveFunctor<T, Context> functor;
  for (int i = 0; i < batchsize; ++i) {
    functor(dev_ctx,
            trans,
            M,
            N,
            lu_bst_conj_data + i * M * M,
            std::max(1, M),
            ipiv_data + i * M,
            res_data + i * M * N,
            info_data + i);
  }

  // Calculate out's conjugate for complex
  result = phi::TransposeLast2Dim<T>(dev_ctx, result);
  out->Resize(common::make_ddim(lu_bst_dims_vec));
  ConjKernel<T, Context>(dev_ctx, result, out);
}

}  // namespace phi
