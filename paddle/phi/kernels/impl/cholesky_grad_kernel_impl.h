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

#include "paddle/phi/kernels/cholesky_grad_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename Context, typename T>
inline void TransCompute(const int dim,
                         const Context& dev_ctx,
                         const DenseTensor& in,
                         DenseTensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      funcs::Transpose<Context, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      funcs::Transpose<Context, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      funcs::Transpose<Context, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      funcs::Transpose<Context, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      funcs::Transpose<Context, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      funcs::Transpose<Context, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      funcs::TransposeNormal<Context, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

/*! Use these functors to implement tril, triu, diagonal and other operators */
template <typename T>
struct EyeFunctor {
  EyeFunctor(const int m, const int n, T* output)
      : m_(m), n_(n), output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int global_row = index / n_;
    const int col = index - global_row * n_;
    const int batch = global_row / m_;
    const int row = global_row - batch * m_;
    output_[index] = col == row ? static_cast<T>(1) : static_cast<T>(0);
  }

  const int m_, n_;
  T* output_;
};

template <typename T>
struct MatrixSetDiagFunctor {
  /*! Overwrite specified diagonals of output by the values in diagonal.
   * diagonals can be a central band specified by num_diags and
   * upper_diag_index, where upper_diag_index=0 refers to the main diagonal,
   * positive value means superdiagonal and negative value means subdiagonal.
   * When it is a band, `diag` has a shape [i, j, ..., num_diags, max_diag_len]
   * and the num_diags diagonals has a up to down layout. Otherwise it has a
   * shape [i, j, ..., max_diag_len].
   */
  MatrixSetDiagFunctor(const int m,
                       const int n,
                       const int num_diags,
                       const int max_diag_len,
                       const int upper_diag_index,
                       const T* diag,
                       T* output)
      : m_(m),
        n_(n),
        num_diags_(num_diags),
        max_diag_len_(max_diag_len),
        upper_diag_index_(upper_diag_index),
        diag_(diag),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int batch_and_diag_index = index / max_diag_len_;
    const int index_in_the_diagonal =
        index - batch_and_diag_index * max_diag_len_;
    const int batch = batch_and_diag_index / num_diags_;
    const int diag_index_in_input = batch_and_diag_index - batch * num_diags_;
    // diag_index=0 refers to the main diagonal
    const int diag_index = upper_diag_index_ - diag_index_in_input;
    // shift down for subdiagonal if diag_index < 0
    const int y_index =
        index_in_the_diagonal + (0 > -diag_index ? 0 : -diag_index);
    // shift right for superdiagonal if diag_index > 0
    const int x_index =
        index_in_the_diagonal + (0 > diag_index ? 0 : diag_index);

    // Upper-bound checks for diagonals shorter than max_diag_len.
    // y_index and x_index are nonnegative by construction.
    if (y_index < m_ && x_index < n_) {
      const int out_index = batch * m_ * n_ + y_index * n_ + x_index;
      output_[out_index] = diag_[index];
    }
  }

  const int m_, n_, num_diags_, max_diag_len_, upper_diag_index_;
  const T* diag_;
  T* output_;
};

template <typename T>
struct MatrixDiagPartFunctor {
  /*! Similar to MatrixSetDiagFunctor but return the diagonals. diag_index=0
   * refers to the main diagonal, positive value means superdiagonal and
   * negative value means subdiagonal */
  MatrixDiagPartFunctor(const int m,
                        const int n,
                        const int num_diags,
                        const int max_diag_len,
                        const int upper_diag_index,
                        const T padding,
                        const T* input,
                        T* output)
      : m_(m),
        n_(n),
        num_diags_(num_diags),
        max_diag_len_(max_diag_len),
        upper_diag_index_(upper_diag_index),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int batch_and_mapped_diag_index = index / max_diag_len_;
    const int index_in_the_diagonal =
        index - batch_and_mapped_diag_index * max_diag_len_;
    const int batch = batch_and_mapped_diag_index / num_diags_;
    const int mapped_diag_index =
        batch_and_mapped_diag_index - batch * num_diags_;
    // diag_index=0 refers to the main diagonal
    const int diag_index = upper_diag_index_ - mapped_diag_index;
    // shift down for subdiagonal if diag_index < 0
    const int y_index =
        index_in_the_diagonal + (0 > -diag_index ? 0 : -diag_index);
    // shift right for superdiagonal if diag_index > 0
    const int x_index =
        index_in_the_diagonal + (0 > diag_index ? 0 : diag_index);
    if (y_index < m_ && x_index < n_) {
      output_[index] = input_[batch * m_ * n_ + y_index * m_ + x_index];
    } else {
      output_[index] = padding_;
    }
  }

  const int m_, n_, num_diags_, max_diag_len_, upper_diag_index_;
  const T padding_;
  const T* input_;
  T* output_;
};

template <typename T>
struct MatrixBandPartScaleEndFunctor {
  /*! Compared with MatrixBandPartFunctor, it scale up values at the end of
   * band. It can be used to fuse the following operations, which actually
   * output triangular with diagonal scaled up:
   * 1. dig = matrix_diag_part(middle)
   * 2. middle = matrix_set_diag(middle, diag * scalar)
   * 3. middle = matrix_band_part(middle, -1, 0)
   */
  MatrixBandPartScaleEndFunctor(const int m,
                                const int n,
                                const int num_lower_diags,
                                const int num_upper_diags,
                                const T scale,
                                const T* input,
                                T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        scale_(scale),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = 0;
    } else if (col == band_end - 1) {
      output_[index] = scale_ * input_[index];
    } else {
      output_[index] = input_[index];
    }
  }

  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const T scale_;
  const T* input_;
  T* output_;
};

template <typename T>
struct AddtoScaleFunctor {
  AddtoScaleFunctor(const T scale, const T* input, T* output)
      : scale_(scale), input_(input), output_(output) {}
  HOSTDEVICE void operator()(size_t index) const {
    output_[index] += input_[index];
    output_[index] *= scale_;
  }
  const T scale_;
  const T* input_;
  T* output_;
};

template <typename T, typename Context>
void CholeskyGradKernel(const Context& dev_ctx,
                        const DenseTensor& out,
                        const DenseTensor& out_grad,
                        bool upper,
                        DenseTensor* x_grad) {
  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  auto& dims = out.dims();
  if (out.numel() == 0) {
    x_grad->Resize(dims);
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  int batch_count = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }
  auto m = dims[dims.size() - 1];
  int tensor_size = batch_count * m * m;

  std::vector<int> axis(dims.size() - 2);
  std::iota(axis.begin(), axis.end(), 0);
  axis.insert(axis.end(), {dims.size() - 1, dims.size() - 2});
  DenseTensor l, l_grad;
  if (upper) {
    l.Resize(dims);
    dev_ctx.template Alloc<T>(&l);
    l_grad.Resize(dims);
    dev_ctx.template Alloc<T>(&l_grad);
    TransCompute<Context, T>(dims.size(), dev_ctx, out, &l, axis);
    TransCompute<Context, T>(dims.size(), dev_ctx, out_grad, &l_grad, axis);
  } else {
    l = out;
    l_grad = out_grad;
  }
  auto* l_data = l.data<T>();

  /*！ refer to Iain Murray (2016); arXiv 1602.07527 */
  /*! phi = matmul(L.transpose(-1, -2), grad) */
  DenseTensor middle;
  middle.Resize(dims);
  auto* middle_data = dev_ctx.template Alloc<T>(&middle);
  auto trans_desc = funcs::CreateMatrixDescriptor(dims, 0, true);
  auto no_trans_desc = funcs::CreateMatrixDescriptor(dims, 0, false);
  auto blas = funcs::GetBlas<Context, T>(dev_ctx);
  blas.MatMul(l, trans_desc, l_grad, no_trans_desc, T(1), &middle, T(0));

  /*! phi.tril_().diagonal(0, -2, -1).mul_(0.5) */
  phi::funcs::ForRange<Context> for_range(dev_ctx, tensor_size);
  MatrixBandPartScaleEndFunctor<T> matrix_band_part_scale_end_functor(
      m,
      m,
      /* num_lower_diags */ m,
      /* num_upper_diags */ 0,
      /* scale */ 0.5,
      middle_data,
      middle_data);
  for_range(matrix_band_part_scale_end_functor);

  // Compute inverse by solving the triangular linear system AX = B, where B
  // is the identity matrix. The matrix X would be overwritten on B
  DenseTensor identity;
  identity.Resize(dims);
  auto* identity_data = dev_ctx.template Alloc<T>(&identity);
  EyeFunctor<T> eye_functor(m, m, identity_data);
  for_range(eye_functor);
  // TODO(guosheng): use trsmBatched for GPU
  for (int i = 0; i < batch_count; i++) {
    blas.TRSM(/*side*/ CblasLeft,
              /*uplo*/ CblasLower,
              /*trans*/ CblasNoTrans,
              /*diag*/ CblasNonUnit,
              /*m*/ m,
              /*n*/ m,
              /*alpha*/ T(1),
              l_data + i * m * m,
              /*lda*/ m,
              identity_data + i * m * m,
              /*ldb*/ m);
  }
  DenseTensor& l_inverse = identity;

  /*! x_grad = matmul(matmul(L_inverse.transpose(-1, -2), phi), L_inverse) */
  DenseTensor middle1;
  middle1.Resize(dims);
  dev_ctx.template Alloc<T>(&middle1);
  blas.MatMul(
      l_inverse, trans_desc, middle, no_trans_desc, T(1), &middle1, T(0));
  blas.MatMul(
      middle1, no_trans_desc, l_inverse, no_trans_desc, T(1), x_grad, T(0));

  /*! x_grad.add(x_grad.transpose(-1, -2)).mul_(0.5) */
  DenseTensor x_grad_trans;
  x_grad_trans.Resize(dims);
  auto* x_grad_trans_data = dev_ctx.template Alloc<T>(&x_grad_trans);
  TransCompute<Context, T>(dims.size(), dev_ctx, *x_grad, &x_grad_trans, axis);
  AddtoScaleFunctor<T> addto_scale_functor(0.5, x_grad_trans_data, x_grad_data);
  for_range(addto_scale_functor);
}

}  // namespace phi
