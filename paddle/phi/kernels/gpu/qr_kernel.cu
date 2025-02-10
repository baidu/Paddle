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

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/rocsolver.h"
#else
#include "paddle/phi/backends/dynload/cusolver.h"
#endif
#include <thrust/device_vector.h>
#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/diagonal_kernel.h"
#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/parse_qr_mode.h"
#include "paddle/phi/kernels/impl/qr_kernel_impl.h"
#include "paddle/phi/kernels/qr_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

namespace phi {

template <class T, class Context>
static DenseTensor Fill(const Context& ctx,
                        std::vector<int64_t> shape,
                        T fill_value) {
  DenseTensor ret;
  ret.Resize(common::make_ddim(shape));
  ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(ctx, &ret, fill_value);
  return ret;
}

template <class T, class Context>
static DenseTensor identity_matrix(const Context& ctx, common::DDim shape) {
  DenseTensor M =
      Fill<T, Context>(ctx, common::vectorize<int64_t>(shape), T(0));
  size_t rank = M.dims().size();
  int64_t M_diag_len = std::min(M.dims()[rank - 1], M.dims()[rank - 2]);
  std::vector<int64_t> M_diag_shape;
  for (size_t i = 0; i < rank - 2; ++i) {
    M_diag_shape.push_back(M.dims()[i]);
  }
  M_diag_shape.push_back(M_diag_len);
  DenseTensor M_diag = Fill<T, Context>(
      ctx, common::vectorize<int64_t>(make_ddim(M_diag_shape)), T(1));
  M = FillDiagonalTensor<T, Context>(ctx, M, M_diag, 0, rank - 2, rank - 1);
  return M;
}

template <typename T, typename Context>
struct QrFunctor {
  void operator()(const Context& ctx,
                  const DenseTensor& x,
                  bool compute_q,
                  bool reduced_mode,
                  DenseTensor* q,
                  DenseTensor* r) {
    auto x_dims = x.dims();
    int x_rank = x_dims.size();
    int m = x_dims[x_rank - 2];
    int n = x_dims[x_rank - 1];
    int min_mn = std::min(m, n);
    int k = reduced_mode ? min_mn : m;
    int64_t batch_size = static_cast<int64_t>(x.numel() / (m * n));
    int qr_stride = m * n;
    int tau_stride = min_mn;

    if (compute_q) {
      ctx.template Alloc<phi::dtype::Real<T>>(
          q, batch_size * m * k * sizeof(phi::dtype::Real<T>));
    }
    ctx.template Alloc<phi::dtype::Real<T>>(
        r, batch_size * k * n * sizeof(phi::dtype::Real<T>));

    // Note: allocate temporary tensors because of lacking in-place operatios.
    // Prepare qr
    DenseTensor qr;
    ctx.template Alloc<phi::dtype::Real<T>>(
        &qr, size_t(batch_size * m * n * sizeof(phi::dtype::Real<T>)));
    // BatchedGeqrf performs computation in-place and 'qr' must be a copy of
    // input
    phi::Copy(ctx, x, ctx.GetPlace(), false, &qr);

    // Prepare tau
    auto tau_dims_vec = common::vectorize<int64_t>(x_dims);
    tau_dims_vec.pop_back();
    tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;
    DenseTensor tau = Fill<T, Context>(ctx, tau_dims_vec, T(0));

    // Transpose 'qr' to conform the column-major order
    auto tmp_qr = TransposeLast2Dim<T, Context>(ctx, qr);
    phi::Copy(ctx, tmp_qr, qr.place(), false, &qr);
    auto qr_data = ctx.template Alloc<phi::dtype::Real<T>>(&qr);
    auto tau_data = ctx.template Alloc<phi::dtype::Real<T>>(&tau);

    BatchedGeqrf<Context, T>(
        ctx, batch_size, m, n, qr_data, m, tau_data, qr_stride, tau_stride);

    if (reduced_mode) {
      auto trans_qr = TransposeLast2Dim<T, Context>(ctx, qr);
      auto sliced_qr = Slice<T, Context>(
          ctx, trans_qr, {trans_qr.dims().size() - 2}, {0}, {min_mn});
      auto tmp_r = TrilTriu<T, Context>(ctx, sliced_qr, 0, false);
      // Transpose 'tmp_r' to restore the original row-major order
      phi::Copy(ctx, tmp_r, r->place(), false, r);
    } else {
      auto trans_qr = TransposeLast2Dim<T, Context>(ctx, qr);
      auto tmp_r = TrilTriu<T, Context>(ctx, trans_qr, 0, false);
      // Transpose 'tmp_r' to restore the original row-major order
      phi::Copy(ctx, tmp_r, r->place(), false, r);
    }

    if (compute_q) {
      // Perform QRGQR for Q using the result from GEQRF
      // Transpose 'q' to restore the original row-major order
      if (reduced_mode) {
        BatchedOrgqr<Context, T>(ctx,
                                 batch_size,
                                 m,
                                 min_mn,
                                 min_mn,
                                 qr_data,
                                 m,
                                 tau_data,
                                 qr_stride,
                                 tau_stride);
        auto trans_q = TransposeLast2Dim<T, Context>(ctx, qr);
        auto sliced_q = Slice<T, Context>(
            ctx, trans_q, {trans_q.dims().size() - 1}, {0}, {min_mn});
        phi::Copy(ctx, sliced_q, q->place(), false, q);
      } else {
        if (m > n) {
          auto new_qr_dims_vec = common::vectorize<int64_t>(x_dims);
          new_qr_dims_vec[new_qr_dims_vec.size() - 1] = m;
          DenseTensor new_qr = Fill<T, Context>(ctx, new_qr_dims_vec, T(0));
          auto new_qr_data = ctx.template Alloc<phi::dtype::Real<T>>(&new_qr);
          auto new_qr_stride = m * m;
          for (int i = 0; i < batch_size; ++i) {
            memory_utils::Copy(ctx.GetPlace(),
                               (new_qr_data + i * new_qr_stride),
                               ctx.GetPlace(),
                               (qr_data + i * qr_stride),
                               qr_stride * sizeof(phi::dtype::Real<T>),
                               ctx.stream());
          }
          BatchedOrgqr<Context, T>(ctx,
                                   batch_size,
                                   m,
                                   m,
                                   min_mn,
                                   new_qr_data,
                                   m,
                                   tau_data,
                                   new_qr_stride,
                                   tau_stride);
          auto trans_q = TransposeLast2Dim<T, Context>(ctx, new_qr);
          phi::Copy(ctx, trans_q, q->place(), false, q);
        } else {
          BatchedOrgqr<Context, T>(ctx,
                                   batch_size,
                                   m,
                                   m,
                                   min_mn,
                                   qr_data,
                                   m,
                                   tau_data,
                                   qr_stride,
                                   tau_stride);
          auto trans_q = TransposeLast2Dim<T, Context>(ctx, qr);
          auto sliced_q = Slice<T, Context>(
              ctx, trans_q, {trans_q.dims().size() - 1}, {0}, {m});
          phi::Copy(ctx, sliced_q, q->place(), false, q);
        }
      }
    }
  }
};

template <typename T, typename Context>
struct QrFunctor<phi::dtype::complex<T>, Context> {
  void operator()(const Context& ctx,
                  const DenseTensor& x,
                  bool compute_q,
                  bool reduced_mode,
                  DenseTensor* q,
                  DenseTensor* r) {
    auto x_dims = x.dims();
    int x_rank = x_dims.size();
    int m = x_dims[x_rank - 2];
    int n = x_dims[x_rank - 1];
    int min_mn = std::min(m, n);
    int k = reduced_mode ? min_mn : m;
    int batch_size = x.numel() / (m * n);
    int qr_stride = m * n;
    int tau_stride = min_mn;
    if (compute_q) {
      ctx.template Alloc<phi::dtype::complex<T>>(
          q, batch_size * m * k * sizeof(phi::dtype::complex<T>));
    }
    ctx.template Alloc<phi::dtype::complex<T>>(
        r, batch_size * k * n * sizeof(phi::dtype::complex<T>));
    // Note: allocate temporary tensors because of lacking in-place operatios.
    // Prepare qr
    DenseTensor qr;
    ctx.template Alloc<phi::dtype::complex<T>>(
        &qr, size_t(batch_size * m * n * sizeof(phi::dtype::complex<T>)));
    // BatchedGeqrf performs computation in-place and 'qr' must be a copy of
    // input
    phi::Copy(ctx, x, ctx.GetPlace(), false, &qr);
    // Prepare tau
    auto tau_dims_vec = common::vectorize<int64_t>(x_dims);
    tau_dims_vec.pop_back();
    tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;
    DenseTensor tau =
        Fill<phi::dtype::complex<T>, Context>(ctx, tau_dims_vec, T(0));
    // Transpose 'qr' to conform the column-major order
    auto tmp_qr = TransposeLast2Dim<phi::dtype::complex<T>, Context>(ctx, qr);
    phi::Copy(ctx, tmp_qr, qr.place(), false, &qr);
    auto qr_data = ctx.template Alloc<phi::dtype::complex<T>>(&qr);
    auto tau_data = ctx.template Alloc<phi::dtype::complex<T>>(&tau);
    BatchedGeqrf<Context, phi::dtype::complex<T>>(
        ctx, batch_size, m, n, qr_data, m, tau_data, qr_stride, tau_stride);
    if (reduced_mode) {
      auto trans_qr =
          TransposeLast2Dim<phi::dtype::complex<T>, Context>(ctx, qr);
      auto sliced_qr = Slice<phi::dtype::complex<T>, Context>(
          ctx, trans_qr, {trans_qr.dims().size() - 2}, {0}, {min_mn});
      auto tmp_r =
          TrilTriu<phi::dtype::complex<T>, Context>(ctx, sliced_qr, 0, false);
      // Transpose 'tmp_r' to restore the original row-major order
      phi::Copy(ctx, tmp_r, r->place(), false, r);
    } else {
      auto trans_qr =
          TransposeLast2Dim<phi::dtype::complex<T>, Context>(ctx, qr);
      auto tmp_r =
          TrilTriu<phi::dtype::complex<T>, Context>(ctx, trans_qr, 0, false);
      // Transpose 'tmp_r' to restore the original row-major order
      phi::Copy(ctx, tmp_r, r->place(), false, r);
    }
    if (compute_q) {
      // Perform QRGQR for Q using the result from GEQRF
      // Transpose 'q' to restore the original row-major order
      if (reduced_mode) {
        BatchedOrgqr<Context, phi::dtype::complex<T>>(ctx,
                                                      batch_size,
                                                      m,
                                                      min_mn,
                                                      min_mn,
                                                      qr_data,
                                                      m,
                                                      tau_data,
                                                      qr_stride,
                                                      tau_stride);
        auto trans_q =
            TransposeLast2Dim<phi::dtype::complex<T>, Context>(ctx, qr);
        auto sliced_q = Slice<phi::dtype::complex<T>, Context>(
            ctx, trans_q, {trans_q.dims().size() - 1}, {0}, {min_mn});
        phi::Copy(ctx, sliced_q, q->place(), false, q);
      } else {
        if (m > n) {
          auto new_qr_dims_vec = common::vectorize<int64_t>(x_dims);
          new_qr_dims_vec[new_qr_dims_vec.size() - 1] = m;
          DenseTensor new_qr =
              Fill<phi::dtype::complex<T>, Context>(ctx, new_qr_dims_vec, T(0));
          auto new_qr_data =
              ctx.template Alloc<phi::dtype::complex<T>>(&new_qr);
          auto new_qr_stride = m * m;
          for (int i = 0; i < batch_size; ++i) {
            memory_utils::Copy(ctx.GetPlace(),
                               (new_qr_data + i * new_qr_stride),
                               ctx.GetPlace(),
                               (qr_data + i * qr_stride),
                               qr_stride * sizeof(phi::dtype::complex<T>),
                               ctx.stream());
          }
          BatchedOrgqr<Context, phi::dtype::complex<T>>(ctx,
                                                        batch_size,
                                                        m,
                                                        m,
                                                        min_mn,
                                                        new_qr_data,
                                                        m,
                                                        tau_data,
                                                        new_qr_stride,
                                                        tau_stride);
          auto trans_q =
              TransposeLast2Dim<phi::dtype::complex<T>, Context>(ctx, new_qr);
          phi::Copy(ctx, trans_q, q->place(), false, q);
        } else {
          BatchedOrgqr<Context, phi::dtype::complex<T>>(ctx,
                                                        batch_size,
                                                        m,
                                                        m,
                                                        min_mn,
                                                        qr_data,
                                                        m,
                                                        tau_data,
                                                        qr_stride,
                                                        tau_stride);
          auto trans_q =
              TransposeLast2Dim<phi::dtype::complex<T>, Context>(ctx, qr);
          auto sliced_q = Slice<phi::dtype::complex<T>, Context>(
              ctx, trans_q, {trans_q.dims().size() - 1}, {0}, {m});
          phi::Copy(ctx, sliced_q, q->place(), false, q);
        }
      }
    }
  }
};

template <typename T, typename Context>
void QrKernel(const Context& ctx,
              const DenseTensor& x,
              const std::string& mode,
              DenseTensor* q,
              DenseTensor* r) {
  bool compute_q;
  bool reduced_mode;
  std::tie(compute_q, reduced_mode) = phi::funcs::ParseQrMode(mode);
  if (x.numel() == 0) {
    if (q->numel() == 0) {
      q->Resize(q->dims());
    } else {
      *q = identity_matrix<T, Context>(ctx, q->dims());
    }
    r->Resize(r->dims());
    ctx.template Alloc<T>(q);
    ctx.template Alloc<T>(r);
    return;
  }
  QrFunctor<T, Context>()(ctx, x, compute_q, reduced_mode, q, r);
}

#ifdef PADDLE_WITH_HIP
#define FUNC_WITH_TYPES(m) m(float, s) m(double, d)
#define GEQRF_BATCH_INSTANCE(T, C)                              \
  template <>                                                   \
  void BatchedGeqrf<GPUContext, T>(const GPUContext& dev_ctx,   \
                                   int batch_size,              \
                                   int m,                       \
                                   int n,                       \
                                   T* a,                        \
                                   int lda,                     \
                                   T* tau,                      \
                                   int a_stride,                \
                                   int tau_stride) {            \
    auto handle = dev_ctx.cusolver_dn_handle();                 \
    for (int i = 0; i < batch_size; ++i) {                      \
      T* a_working_ptr = &a[i * a_stride];                      \
      T* tau_working_ptr = &tau[i * tau_stride];                \
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::rocsolver_##C##geqrf( \
          handle, m, n, a_working_ptr, lda, tau_working_ptr));  \
    }                                                           \
  }

FUNC_WITH_TYPES(GEQRF_BATCH_INSTANCE);

#define ORGQR_BATCH_INSTANCE(T, C)                                \
  template <>                                                     \
  void BatchedOrgqr<GPUContext, T>(const GPUContext& dev_ctx,     \
                                   int batch_size,                \
                                   int m,                         \
                                   int n,                         \
                                   int k,                         \
                                   T* a,                          \
                                   int lda,                       \
                                   T* tau,                        \
                                   int a_stride,                  \
                                   int tau_stride) {              \
    auto handle = dev_ctx.cusolver_dn_handle();                   \
    for (int i = 0; i < batch_size; ++i) {                        \
      T* a_working_ptr = &a[i * a_stride];                        \
      T* tau_working_ptr = &tau[i * tau_stride];                  \
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::rocsolver_##C##orgqr(   \
          handle, m, n, k, a_working_ptr, lda, tau_working_ptr)); \
    }                                                             \
  }

FUNC_WITH_TYPES(ORGQR_BATCH_INSTANCE);
#else
template <>
void BatchedGeqrf<GPUContext, float>(const GPUContext& dev_ctx,
                                     int batch_size,
                                     int m,
                                     int n,
                                     float* a,
                                     int lda,
                                     float* tau,
                                     int a_stride,
                                     int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  float* workspace_ptr = dev_ctx.template Alloc<float>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgeqrf(handle,
                                                              m,
                                                              n,
                                                              a_working_ptr,
                                                              lda,
                                                              tau_working_ptr,
                                                              workspace_ptr,
                                                              lwork,
                                                              info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedGeqrf<GPUContext, double>(const GPUContext& dev_ctx,
                                      int batch_size,
                                      int m,
                                      int n,
                                      double* a,
                                      int lda,
                                      double* tau,
                                      int a_stride,
                                      int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  double* workspace_ptr = dev_ctx.template Alloc<double>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgeqrf(handle,
                                                              m,
                                                              n,
                                                              a_working_ptr,
                                                              lda,
                                                              tau_working_ptr,
                                                              workspace_ptr,
                                                              lwork,
                                                              info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedGeqrf<GPUContext, phi::dtype::complex<float>>(
    const GPUContext& dev_ctx,
    int batch_size,
    int m,
    int n,
    phi::dtype::complex<float>* a,
    int lda,
    phi::dtype::complex<float>* tau,
    int a_stride,
    int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnCgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<cuComplex*>(a), lda, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  phi::dtype::complex<float>* workspace_ptr =
      dev_ctx.template Alloc<phi::dtype::complex<float>>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    phi::dtype::complex<float>* a_working_ptr = &a[i * a_stride];
    phi::dtype::complex<float>* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnCgeqrf(
        handle,
        m,
        n,
        reinterpret_cast<cuComplex*>(a_working_ptr),
        lda,
        reinterpret_cast<cuComplex*>(tau_working_ptr),
        reinterpret_cast<cuComplex*>(workspace_ptr),
        lwork,
        info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedGeqrf<GPUContext, phi::dtype::complex<double>>(
    const GPUContext& dev_ctx,
    int batch_size,
    int m,
    int n,
    phi::dtype::complex<double>* a,
    int lda,
    phi::dtype::complex<double>* tau,
    int a_stride,
    int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnZgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<cuDoubleComplex*>(a), lda, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  phi::dtype::complex<double>* workspace_ptr =
      dev_ctx.template Alloc<phi::dtype::complex<double>>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    phi::dtype::complex<double>* a_working_ptr = &a[i * a_stride];
    phi::dtype::complex<double>* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnZgeqrf(
        handle,
        m,
        n,
        reinterpret_cast<cuDoubleComplex*>(a_working_ptr),
        lda,
        reinterpret_cast<cuDoubleComplex*>(tau_working_ptr),
        reinterpret_cast<cuDoubleComplex*>(workspace_ptr),
        lwork,
        info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<GPUContext, float>(const GPUContext& dev_ctx,
                                     int batch_size,
                                     int m,
                                     int n,
                                     int k,
                                     float* a,
                                     int lda,
                                     float* tau,
                                     int a_stride,
                                     int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSorgqr_bufferSize(
      handle, m, n, k, a, lda, tau, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  float* workspace_ptr = dev_ctx.template Alloc<float>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSorgqr(handle,
                                                              m,
                                                              n,
                                                              k,
                                                              a_working_ptr,
                                                              lda,
                                                              tau_working_ptr,
                                                              workspace_ptr,
                                                              lwork,
                                                              info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<GPUContext, double>(const GPUContext& dev_ctx,
                                      int batch_size,
                                      int m,
                                      int n,
                                      int k,
                                      double* a,
                                      int lda,
                                      double* tau,
                                      int a_stride,
                                      int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDorgqr_bufferSize(
      handle, m, n, k, a, lda, tau, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  double* workspace_ptr = dev_ctx.template Alloc<double>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDorgqr(handle,
                                                              m,
                                                              n,
                                                              k,
                                                              a_working_ptr,
                                                              lda,
                                                              tau_working_ptr,
                                                              workspace_ptr,
                                                              lwork,
                                                              info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<GPUContext, phi::dtype::complex<float>>(
    const GPUContext& dev_ctx,
    int batch_size,
    int m,
    int n,
    int k,
    phi::dtype::complex<float>* a,
    int lda,
    phi::dtype::complex<float>* tau,
    int a_stride,
    int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnCungqr_bufferSize(
      handle,
      m,
      n,
      k,
      reinterpret_cast<cuComplex*>(a),
      lda,
      reinterpret_cast<cuComplex*>(tau),
      &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  phi::dtype::complex<float>* workspace_ptr =
      dev_ctx.template Alloc<phi::dtype::complex<float>>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    phi::dtype::complex<float>* a_working_ptr = &a[i * a_stride];
    phi::dtype::complex<float>* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnCungqr(
        handle,
        m,
        n,
        k,
        reinterpret_cast<cuComplex*>(a_working_ptr),
        lda,
        reinterpret_cast<cuComplex*>(tau_working_ptr),
        reinterpret_cast<cuComplex*>(workspace_ptr),
        lwork,
        info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<GPUContext, phi::dtype::complex<double>>(
    const GPUContext& dev_ctx,
    int batch_size,
    int m,
    int n,
    int k,
    phi::dtype::complex<double>* a,
    int lda,
    phi::dtype::complex<double>* tau,
    int a_stride,
    int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnZungqr_bufferSize(
      handle,
      m,
      n,
      k,
      reinterpret_cast<cuDoubleComplex*>(a),
      lda,
      reinterpret_cast<cuDoubleComplex*>(tau),
      &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(common::make_ddim({lwork}));
  phi::dtype::complex<double>* workspace_ptr =
      dev_ctx.template Alloc<phi::dtype::complex<double>>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(common::make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    phi::dtype::complex<double>* a_working_ptr = &a[i * a_stride];
    phi::dtype::complex<double>* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnZungqr(
        handle,
        m,
        n,
        k,
        reinterpret_cast<cuDoubleComplex*>(a_working_ptr),
        lda,
        reinterpret_cast<cuDoubleComplex*>(tau_working_ptr),
        reinterpret_cast<cuDoubleComplex*>(workspace_ptr),
        lwork,
        info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory_utils::Copy(phi::CPUPlace(),
                       &info_h,
                       dev_ctx.GetPlace(),
                       info_d,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}
#endif

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(qr, GPU, ALL_LAYOUT, phi::QrKernel, float, double) {}
#else
PD_REGISTER_KERNEL(qr,
                   GPU,
                   ALL_LAYOUT,
                   phi::QrKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
