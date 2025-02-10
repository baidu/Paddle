/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "glog/logging.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/autotune/cache_base.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.hip.h"
#else
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#endif
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/scale_kernel.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/kernels/funcs/cublaslt.h"
#include "paddle/phi/kernels/gpu/cuda_gemm_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/kernels/funcs/hipblaslt.h"
#endif
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#endif

COMMON_DECLARE_bool(cuda_core_int8_gemm);

namespace phi {

static void GetBroadcastFromDims(const int x_ndim,
                                 const std::int64_t* x_dims,
                                 const int y_ndim,
                                 const std::int64_t* y_dims,
                                 std::int64_t* x_bd_dims,
                                 std::int64_t* y_bd_dims,
                                 std::int64_t* out_bd_dims) {
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::fill(x_bd_dims, x_bd_dims + ndim - x_ndim, 1);
  std::fill(y_bd_dims, y_bd_dims + ndim - y_ndim, 1);
  std::copy(x_dims, x_dims + x_ndim, x_bd_dims + ndim - x_ndim);
  std::copy(y_dims, y_dims + y_ndim, y_bd_dims + ndim - y_ndim);

  for (int i = 0; i < ndim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] <= 1 || y_bd_dims[i] <= 1,
        true,
        common::errors::InvalidArgument(
            "Input(X) and Input(Y) has error dim. "
            "X_broadcast's shape[%s] must be equal to Y_broadcast's shape[%s], "
            "or X_broadcast's shape[%s] <= 1, or Y_broadcast's shape[%s] <= 1, "
            "but received X_broadcast's shape[%s] = [%s]"
            "received Y_broadcast's shape[%s] = [%s].",
            i,
            i,
            i,
            i,
            i,
            x_bd_dims[i],
            i,
            y_bd_dims[i]));
    if (x_bd_dims[i] == 0 || y_bd_dims[i] == 0) {
      out_bd_dims[i] = 0;
    } else {
      out_bd_dims[i] = (std::max)(x_bd_dims[i], y_bd_dims[i]);
    }
  }
}

static int64_t GetIndexMessage(const int n,
                               const int64_t* dims,
                               const int64_t* index) {
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    if (dims[i] > 1) {
      sum = sum * dims[i] + index[i];
    }
  }
  return sum;
}

static void IndexIncreaseFromDims(const int ndim,
                                  const int64_t* dims,
                                  int64_t* index) {
  for (int i = ndim - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

// The general implementation with blas.
template <typename Context, typename T>
void MatMulFunctionImplWithBlas(
    const Context& dev_ctx,
    const DenseTensor& X,
    const DenseTensor& Y,
    const std::vector<std::int64_t>& x_dims,
    const std::vector<std::int64_t>& y_dims,
    DenseTensor* Out,
    bool trans_x,
    bool trans_y,
    bool flag = false,
    phi::funcs::MatmulPlanner* matmul_planner UNUSED = nullptr) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // Get data ptr
  const T* x_data = X.data<T>();
  const T* y_data = Y.data<T>();

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = X.numel();
    const int N = Y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        common::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers, "
            "when X/Y's dims =1. But received X has [%d] elements, "
            "received Y has [%d] elements.",
            M,
            N));
    VLOG(3) << "MatMul's case 1";
    Out->Resize(common::make_ddim({}));
    dev_ctx.template Alloc<T>(Out);
    blas.GEMM(CblasNoTrans,
              CblasTrans,
              1,
              1,
              M,
              static_cast<T>(1),
              y_data,
              x_data,
              static_cast<T>(flag),
              dev_ctx.template Alloc<T>(Out));
    return;
  }

  if (x_ndim == 1) {
    const int N = X.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 1,
                                          N,
                                          y_ndim - 1,
                                          y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 2,
                                          N,
                                          y_ndim - 2,
                                          y_dims[y_ndim - 2]));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    Out->ResizeAndAllocate(common::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);
    if (trans_y) {
      const int M = Y.numel() / N;
      VLOG(3) << "MatMul's case 2";
      blas.GEMV(false,
                M,
                N,
                static_cast<T>(1),
                y_data,
                x_data,
                static_cast<T>(flag),
                dev_ctx.template Alloc<T>(Out));
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 3";
        blas.GEMV(true,
                  N,
                  M,
                  static_cast<T>(1),
                  y_data,
                  x_data,
                  static_cast<T>(flag),
                  dev_ctx.template Alloc<T>(Out));
      } else {
        VLOG(3) << "MatMul's case 4";
        blas.BatchedGEMM(CblasTrans,
                         CblasNoTrans,
                         M,
                         1,
                         N,
                         static_cast<T>(1),
                         y_data,
                         x_data,
                         static_cast<T>(flag),
                         dev_ctx.template Alloc<T>(Out),
                         batch_size,
                         M * N,
                         0);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d"
                                          "But received X'dims[%d] is %d",
                                          x_ndim - 2,
                                          N,
                                          x_ndim - 2,
                                          x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d"
                                          "But received X'dims[%d] is %d",
                                          x_ndim - 1,
                                          N,
                                          x_ndim - 1,
                                          x_dims[x_ndim - 1]));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    Out->ResizeAndAllocate(common::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 5";
        blas.GEMV(true,
                  N,
                  M,
                  static_cast<T>(1),
                  x_data,
                  y_data,
                  static_cast<T>(flag),
                  dev_ctx.template Alloc<T>(Out));
      } else {
        VLOG(3) << "MatMul's case 6";
        blas.BatchedGEMM(CblasTrans,
                         CblasNoTrans,
                         M,
                         1,
                         N,
                         static_cast<T>(1),
                         x_data,
                         y_data,
                         static_cast<T>(flag),
                         dev_ctx.template Alloc<T>(Out),
                         batch_size,
                         M * N,
                         0);
      }
    } else {
      const int M = X.numel() / N;
      VLOG(3) << "MatMul's case 7";
      blas.GEMV(false,
                M,
                N,
                static_cast<T>(1),
                x_data,
                y_data,
                static_cast<T>(flag),
                dev_ctx.template Alloc<T>(Out));
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 1,
                                        K,
                                        y_ndim - 1,
                                        y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 2,
                                        K,
                                        y_ndim - 2,
                                        y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  GetBroadcastFromDims(x_ndim - 2,
                       x_dims.data(),
                       y_ndim - 2,
                       y_dims.data(),
                       x_broadcast_dims.data(),
                       y_broadcast_dims.data(),
                       out_broadcast_dims.data());
  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  Out->ResizeAndAllocate(common::make_ddim(out_broadcast_dims));
  dev_ctx.template Alloc<T>(Out);

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims =
      !std::equal(x_broadcast_dims.cbegin(),
                  x_broadcast_dims.cbegin() + batch_dim,
                  y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size =
      std::accumulate(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size =
      std::accumulate(y_broadcast_dims.cbegin(),
                      y_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size =
      std::accumulate(out_broadcast_dims.cbegin(),
                      out_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) return;
  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(3) << "MatMul's case 8";
    blas.GEMM(trans_x ? CblasTrans : CblasNoTrans,
              trans_y ? CblasTrans : CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              x_data,
              y_data,
              static_cast<T>(flag),
              dev_ctx.template Alloc<T>(Out));
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul's case 9";
      blas.GEMV(false,
                y_batch_size * N,
                K,
                static_cast<T>(1),
                y_data,
                x_data,
                static_cast<T>(flag),
                dev_ctx.template Alloc<T>(Out));
    } else {
      VLOG(3) << "MatMul's case 10";
      blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                       trans_y ? CblasTrans : CblasNoTrans,
                       M,
                       N,
                       K,
                       static_cast<T>(1),
                       x_data,
                       y_data,
                       static_cast<T>(flag),
                       dev_ctx.template Alloc<T>(Out),
                       out_batch_size,
                       0,
                       K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul's case 11";
      blas.GEMM(CblasNoTrans,
                trans_y ? CblasTrans : CblasNoTrans,
                x_batch_size * M,
                N,
                K,
                static_cast<T>(1),
                x_data,
                y_data,
                static_cast<T>(flag),
                dev_ctx.template Alloc<T>(Out));
    } else {
      VLOG(3) << "MatMul's case 12";
      blas.BatchedGEMM(CblasTrans,
                       trans_y ? CblasTrans : CblasNoTrans,
                       M,
                       N,
                       K,
                       static_cast<T>(1),
                       x_data,
                       y_data,
                       static_cast<T>(flag),
                       dev_ctx.template Alloc<T>(Out),
                       out_batch_size,
                       M * K,
                       0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul's case 13";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans,
                     M,
                     N,
                     K,
                     static_cast<T>(1),
                     x_data,
                     y_data,
                     static_cast<T>(flag),
                     dev_ctx.template Alloc<T>(Out),
                     out_batch_size,
                     M * K,
                     K * N);
  } else {
    // in the case, can't use stridedgemm
    std::vector<const T*> x_ptr(out_batch_size);
    std::vector<const T*> y_ptr(out_batch_size);
    std::vector<T*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim, 0);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      // using the index to get offset
      const std::int64_t x_index =
          GetIndexMessage(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexMessage(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = dev_ctx.template Alloc<T>(Out) + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul's case 14";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans,
                     M,
                     N,
                     K,
                     static_cast<T>(1),
                     x_ptr.data(),
                     y_ptr.data(),
                     static_cast<T>(flag),
                     out_ptr.data(),
                     out_batch_size);
  }
}

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
// This is almost a copy from MatMulFunctionImplWithBlas,
// compare cublas with cublasLt kernels when Matmul autotune is on
template <typename Context, typename T>
void MatMulFunctionImplWithCublasLt(
    const Context& dev_ctx,
    const DenseTensor& X,
    const DenseTensor& Y,
    const std::vector<std::int64_t>& x_dims,
    const std::vector<std::int64_t>& y_dims,
    DenseTensor* Out,
    bool trans_x,
    bool trans_y,
    bool flag = false,
    phi::funcs::MatmulPlanner* matmul_planner = nullptr) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();
  const T* x_data = X.data<T>();
  const T* y_data = Y.data<T>();
  using blaslt = phi::funcs::MatmulWithCublasLt<T>;

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = X.numel();
    const int N = Y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        common::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers, "
            "when X/Y's dims =1. But received X has [%d] elements, "
            "received Y has [%d] elements",
            M,
            N));

    // MatMul's case 0  =>  vector * vector
    Out->Resize(common::make_ddim({}));
    dev_ctx.template Alloc<T>(Out);
    VLOG(3) << "MatMul with blaslt case 1";
    blaslt::Run(dev_ctx,
                y_data,
                x_data,
                dev_ctx.template Alloc<T>(Out),
                1,
                1,
                M,
                false,
                true,
                matmul_planner);
    return;
  }

  if (x_ndim == 1) {
    const int N = X.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 1,
                                          N,
                                          y_ndim - 1,
                                          y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 2,
                                          N,
                                          y_ndim - 2,
                                          y_dims[y_ndim - 2]));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    Out->ResizeAndAllocate(common::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);
    if (trans_y) {
      const int M = Y.numel() / N;
      VLOG(3) << "MatMul with blaslt 2";
      blaslt::Run(dev_ctx,
                  y_data,
                  x_data,
                  dev_ctx.template Alloc<T>(Out),
                  M,
                  1,
                  N,
                  false,
                  false,
                  matmul_planner);
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul with blaslt 3";
        blaslt::Run(dev_ctx,
                    y_data,
                    x_data,
                    dev_ctx.template Alloc<T>(Out),
                    M,
                    1,
                    N,
                    true,
                    false,
                    matmul_planner);
      } else {
        VLOG(3) << "MatMul with blaslt 4";
        blaslt::RunWithBatch(dev_ctx,
                             y_data,
                             x_data,
                             dev_ctx.template Alloc<T>(Out),
                             M,
                             1,
                             N,
                             true,
                             false,
                             batch_size,
                             M * N,
                             0,
                             M,
                             matmul_planner);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d"
                                          "But received X'dims[%d] is %d",
                                          x_ndim - 2,
                                          N,
                                          x_ndim - 2,
                                          x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d, "
                                          "but received X'dims[%d] is %d",
                                          x_ndim - 1,
                                          N,
                                          x_ndim - 1,
                                          x_dims[x_ndim - 1]));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    Out->ResizeAndAllocate(common::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul with blaslt 5";
        blaslt::Run(dev_ctx,
                    x_data,
                    y_data,
                    dev_ctx.template Alloc<T>(Out),
                    M,
                    1,
                    N,
                    true,
                    false,
                    matmul_planner);
      } else {
        VLOG(3) << "MatMul with blaslt 6";
        blaslt::RunWithBatch(dev_ctx,
                             x_data,
                             y_data,
                             dev_ctx.template Alloc<T>(Out),
                             M,
                             1,
                             N,
                             true,
                             false,
                             batch_size,
                             M * N,
                             0,
                             M,
                             matmul_planner);
      }
    } else {
      const int M = X.numel() / N;
      VLOG(3) << "MatMul with blaslt 7";
      blaslt::Run(dev_ctx,
                  x_data,
                  y_data,
                  dev_ctx.template Alloc<T>(Out),
                  M,
                  1,
                  N,
                  false,
                  false,
                  matmul_planner);
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 1,
                                        K,
                                        y_ndim - 1,
                                        y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 2,
                                        K,
                                        y_ndim - 2,
                                        y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  GetBroadcastFromDims(x_ndim - 2,
                       x_dims.data(),
                       y_ndim - 2,
                       y_dims.data(),
                       x_broadcast_dims.data(),
                       y_broadcast_dims.data(),
                       out_broadcast_dims.data());
  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  Out->ResizeAndAllocate(common::make_ddim(out_broadcast_dims));
  dev_ctx.template Alloc<T>(Out);

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims =
      !std::equal(x_broadcast_dims.cbegin(),
                  x_broadcast_dims.cbegin() + batch_dim,
                  y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size =
      std::accumulate(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size =
      std::accumulate(y_broadcast_dims.cbegin(),
                      y_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size =
      std::accumulate(out_broadcast_dims.cbegin(),
                      out_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) return;
  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(3) << "MatMul with blaslt 8";
    blaslt::Run(dev_ctx,
                x_data,
                y_data,
                dev_ctx.template Alloc<T>(Out),
                M,
                N,
                K,
                trans_x,
                trans_y,
                matmul_planner);
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul with blaslt 9";
      blaslt::Run(dev_ctx,
                  y_data,
                  x_data,
                  dev_ctx.template Alloc<T>(Out),
                  y_batch_size * N,
                  1,
                  K,
                  false,
                  false,
                  matmul_planner);
    } else {
      VLOG(3) << "MatMul with blaslt 10";
      blaslt::RunWithBatch(dev_ctx,
                           x_data,
                           y_data,
                           dev_ctx.template Alloc<T>(Out),
                           M,
                           N,
                           K,
                           trans_x,
                           trans_y,
                           out_batch_size,
                           0,
                           K * N,
                           M * N,
                           matmul_planner);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul with blaslt 11";
      blaslt::Run(dev_ctx,
                  x_data,
                  y_data,
                  dev_ctx.template Alloc<T>(Out),
                  x_batch_size * M,
                  N,
                  K,
                  false,
                  trans_y,
                  matmul_planner);
    } else {
      VLOG(3) << "MatMul with blaslt 12";
      blaslt::RunWithBatch(dev_ctx,
                           x_data,
                           y_data,
                           dev_ctx.template Alloc<T>(Out),
                           M,
                           N,
                           K,
                           true,
                           trans_y,
                           out_batch_size,
                           M * K,
                           0,
                           M * N,
                           matmul_planner);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul with blaslt 13";
    blaslt::RunWithBatch(dev_ctx,
                         x_data,
                         y_data,
                         dev_ctx.template Alloc<T>(Out),
                         M,
                         N,
                         K,
                         trans_x,
                         trans_y,
                         out_batch_size,
                         M * K,
                         K * N,
                         M * N,
                         matmul_planner);
  } else {
    // in the case, can't use stridedgemm
    std::vector<const T*> x_ptr(out_batch_size);
    std::vector<const T*> y_ptr(out_batch_size);
    std::vector<T*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim, 0);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      // using the index to get offset
      const std::int64_t x_index =
          GetIndexMessage(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexMessage(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = dev_ctx.template Alloc<T>(Out) + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul with blaslt 14";
    blaslt::RunWithBatch(dev_ctx,
                         x_ptr.data(),
                         y_ptr.data(),
                         out_ptr.data(),
                         M,
                         N,
                         K,
                         trans_x,
                         trans_y,
                         out_batch_size,
                         matmul_planner);
  }
}
#endif

template <typename Context, typename T>
struct MatMulDispatcher {
  void operator()(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  const std::vector<std::int64_t>& x_dims,
                  const std::vector<std::int64_t>& y_dims,
                  DenseTensor* out,
                  bool trans_x,
                  bool trans_y,
                  bool flag = false) {
    MatMulFunctionImplWithBlas<Context, T>(
        ctx, x, y, x_dims, y_dims, out, trans_x, trans_y, flag);
  }
};

#ifdef PADDLE_WITH_CUDA
template <typename T>
struct MatMulDispatcher<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  const std::vector<std::int64_t>& x_dims,
                  const std::vector<std::int64_t>& y_dims,
                  DenseTensor* out,
                  bool trans_x,
                  bool trans_y,
                  bool flag = false) {
#if CUDA_VERSION >= 11060
    auto* tuner = phi::autotune::MakeMatmulTuner<T>(
        MatMulFunctionImplWithBlas<phi::GPUContext, T>);
    tuner->AddCallBack(MatMulFunctionImplWithCublasLt<phi::GPUContext, T>);
    phi::funcs::MatmulPlanner matmul_planner(x_dims,
                                             y_dims,
                                             trans_x,
                                             trans_y,
                                             phi::CppTypeToDataType<T>::Type(),
                                             funcs::MatmulFusedType::kMatmul,
                                             /* bias_data */ nullptr,
                                             /* reserve_data */ nullptr,
                                             /* use_addto */ flag,
                                             /* no_exchange */ true);
    tuner->Run(ctx,
               matmul_planner.GetKey(),
               ctx,
               x,
               y,
               x_dims,
               y_dims,
               out,
               trans_x,
               trans_y,
               flag,
               &matmul_planner);
#else
    MatMulFunctionImplWithBlas<phi::GPUContext, T>(
        ctx, x, y, x_dims, y_dims, out, trans_x, trans_y, flag);
#endif
  }
};

#endif  // PADDLE_WITH_CUDA

template <typename Context, typename T>
void MatMulFunction(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const std::vector<std::int64_t>& x_dims,
                    const std::vector<std::int64_t>& y_dims,
                    DenseTensor* out,
                    bool trans_x,
                    bool trans_y,
                    bool flag = false) {
  MatMulDispatcher<Context, T>()(
      ctx, x, y, x_dims, y_dims, out, trans_x, trans_y, flag);
}

template <typename Context>
bool MatMulInt8Function(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const std::vector<std::int64_t>& x_dims,
                        const std::vector<std::int64_t>& y_dims,
                        DenseTensor* out,
                        bool trans_x,
                        bool trans_y) {
  return false;
}

#ifdef PADDLE_WITH_CUDA
template <>
bool inline MatMulInt8Function(const phi::GPUContext& ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const std::vector<std::int64_t>& x_dims,
                               const std::vector<std::int64_t>& y_dims,
                               DenseTensor* out,
                               bool trans_x,
                               bool trans_y) {
  if (x.dtype() != DataType::INT8 || y.dtype() != DataType::INT8) {
    return false;
  }
#if CUDA_VERSION >= 11060
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();
  const int8_t* x_data = x.data<int8_t>();
  const int8_t* y_data = y.data<int8_t>();
  using blaslt = phi::funcs::MatmulWithCublasLt<int8_t, int32_t>;

  phi::funcs::MatmulPlanner matmul_planner(
      x_dims,
      y_dims,
      trans_x,
      trans_y,
      phi::CppTypeToDataType<int8_t>::Type(),
      funcs::MatmulFusedType::kMatmul,
      /* bias_data */ nullptr,
      /* reserve_data */ nullptr,
      /* use_addto */ false,
      /* no_exchange */ true);

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = x.numel();
    const int N = y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        common::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers, "
            "when X/Y's dims =1. But received X has [%d] elements, s"
            "received Y has [%d] elements",
            M,
            N));
    if (!(M % 4 == 0)) {
      return false;
    }

    out->Resize(common::make_ddim({}));
    ctx.template Alloc<int32_t>(out);
    blaslt::Run(ctx,
                y_data,
                x_data,
                ctx.template Alloc<int32_t>(out),
                1,
                1,
                M,
                false,
                true,
                &matmul_planner);
    return true;
  }
  if (x_ndim == 1) {
    const int N = x.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 1,
                                          N,
                                          y_ndim - 1,
                                          y_dims[y_ndim - 1]));
      if (!(N % 4 == 0)) {
        return false;
      }
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 2,
                                          N,
                                          y_ndim - 2,
                                          y_dims[y_ndim - 2]));
      const int M = y.numel() / N;
      if (!(M == 1 || M % 4 == 0)) {
        return false;
      }
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    out->ResizeAndAllocate(common::make_ddim(out_dims));
    ctx.template Alloc<int32_t>(out);
    if (trans_y) {
      const int M = y.numel() / N;
      blaslt::Run(ctx,
                  y_data,
                  x_data,
                  ctx.template Alloc<int32_t>(out),
                  M,
                  1,
                  N,
                  false,
                  false,
                  &matmul_planner);
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = y.numel() / (M * N);
      if (batch_size == 1) {
        blaslt::Run(ctx,
                    y_data,
                    x_data,
                    ctx.template Alloc<int32_t>(out),
                    M,
                    1,
                    N,
                    true,
                    false,
                    &matmul_planner);
      } else {
        blaslt::RunWithBatch(ctx,
                             y_data,
                             x_data,
                             ctx.template Alloc<int32_t>(out),
                             M,
                             1,
                             N,
                             true,
                             false,
                             batch_size,
                             M * N,
                             0,
                             M,
                             &matmul_planner);
      }
    }
    return true;
  }

  if (y_ndim == 1) {
    const int N = y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d, "
                                          "but received X'dims[%d] is %d",
                                          x_ndim - 2,
                                          N,
                                          x_ndim - 2,
                                          x_dims[x_ndim - 2]));
      const int M = x.numel() / N;
      if (!((M == 1 || M % 4 == 0))) {
        return false;
      }
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d, "
                                          "but received X'dims[%d] is %d",
                                          x_ndim - 1,
                                          N,
                                          x_ndim - 1,
                                          x_dims[x_ndim - 1]));
      if (N % 4 != 0) {
        return false;
      }
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    out->ResizeAndAllocate(common::make_ddim(out_dims));
    ctx.template Alloc<int32_t>(out);

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = x.numel() / (M * N);
      if (batch_size == 1) {
        blaslt::Run(ctx,
                    x_data,
                    y_data,
                    ctx.template Alloc<int32_t>(out),
                    M,
                    1,
                    N,
                    true,
                    false,
                    &matmul_planner);
      } else {
        blaslt::RunWithBatch(ctx,
                             x_data,
                             y_data,
                             ctx.template Alloc<int32_t>(out),
                             M,
                             1,
                             N,
                             true,
                             false,
                             batch_size,
                             M * N,
                             0,
                             M,
                             &matmul_planner);
      }
    } else {
      const int M = x.numel() / N;
      blaslt::Run(ctx,
                  x_data,
                  y_data,
                  ctx.template Alloc<int32_t>(out),
                  M,
                  1,
                  N,
                  false,
                  false,
                  &matmul_planner);
    }
    return true;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 1,
                                        K,
                                        y_ndim - 1,
                                        y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 2,
                                        K,
                                        y_ndim - 2,
                                        y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);
  GetBroadcastFromDims(x_ndim - 2,
                       x_dims.data(),
                       y_ndim - 2,
                       y_dims.data(),
                       x_broadcast_dims.data(),
                       y_broadcast_dims.data(),
                       out_broadcast_dims.data());
  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  out->ResizeAndAllocate(common::make_ddim(out_broadcast_dims));
  ctx.template Alloc<int32_t>(out);

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims =
      !std::equal(x_broadcast_dims.cbegin(),
                  x_broadcast_dims.cbegin() + batch_dim,
                  y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size =
      std::accumulate(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size =
      std::accumulate(y_broadcast_dims.cbegin(),
                      y_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size =
      std::accumulate(out_broadcast_dims.cbegin(),
                      out_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) return true;

  if (x_batch_size == 1 && M == 1 && trans_y) {
    if (!(K % 4 == 0)) {
      return false;
    }
  } else if (!trans_x && !trans_y) {
    if (!(N % 4 == 0 || N == 1) || !(K % 4 == 0) || (M == 1 && N == 1)) {
      return false;
    }
  } else if (!trans_x && trans_y) {
    if (!(K % 4 == 0)) {
      return false;
    }
  } else if (trans_x && !trans_y) {
    if (!(M % 4 == 0 || M == 1) || !(N % 4 == 0 || N == 1)) {
      return false;
    }
  } else {
    if (!(M % 4 == 0 || M == 1) || !(K % 4 == 0)) {
      return false;
    }
  }
  if (x_batch_size == 1 && y_batch_size == 1) {
    blaslt::Run(ctx,
                x_data,
                y_data,
                ctx.template Alloc<int32_t>(out),
                M,
                N,
                K,
                trans_x,
                trans_y,
                &matmul_planner);
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      blaslt::Run(ctx,
                  y_data,
                  x_data,
                  ctx.template Alloc<int32_t>(out),
                  y_batch_size * N,
                  1,
                  K,
                  false,
                  false,
                  &matmul_planner);
    } else {
      blaslt::RunWithBatch(ctx,
                           x_data,
                           y_data,
                           ctx.template Alloc<int32_t>(out),
                           M,
                           N,
                           K,
                           trans_x,
                           trans_y,
                           out_batch_size,
                           0,
                           K * N,
                           M * N,
                           &matmul_planner);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      blaslt::Run(ctx,
                  x_data,
                  y_data,
                  ctx.template Alloc<int32_t>(out),
                  x_batch_size * M,
                  N,
                  K,
                  false,
                  trans_y,
                  &matmul_planner);
    } else {
      blaslt::RunWithBatch(ctx,
                           x_data,
                           y_data,
                           ctx.template Alloc<int32_t>(out),
                           M,
                           N,
                           K,
                           true,
                           trans_y,
                           out_batch_size,
                           M * K,
                           0,
                           M * N,
                           &matmul_planner);
    }
  } else if (!is_broadcast_dims) {
    blaslt::RunWithBatch(ctx,
                         x_data,
                         y_data,
                         ctx.template Alloc<int32_t>(out),
                         M,
                         N,
                         K,
                         trans_x,
                         trans_y,
                         out_batch_size,
                         M * K,
                         K * N,
                         M * N,
                         &matmul_planner);
  } else {
    // in the case, can't use stridedgemm
    std::vector<const int8_t*> x_ptr(out_batch_size);
    std::vector<const int8_t*> y_ptr(out_batch_size);
    std::vector<int32_t*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim, 0);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      // using the index to get offset
      const std::int64_t x_index =
          GetIndexMessage(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexMessage(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = ctx.template Alloc<int32_t>(out) + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    blaslt::RunWithBatch(ctx,
                         x_ptr.data(),
                         y_ptr.data(),
                         out_ptr.data(),
                         M,
                         N,
                         K,
                         trans_x,
                         trans_y,
                         out_batch_size,
                         &matmul_planner);
  }
  return true;
#else
  return false;
#endif
}
#endif

#ifdef PADDLE_WITH_HIP
template <>
bool inline MatMulInt8Function(const phi::GPUContext& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const std::vector<std::int64_t>& x_dims,
                               const std::vector<std::int64_t>& y_dims,
                               DenseTensor* out,
                               bool trans_x,
                               bool trans_y) {
  if (x.dtype() != DataType::INT8 || y.dtype() != DataType::INT8) {
    return false;
  }
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // Get data ptr
  const int8_t* x_data = x.data<int8_t>();
  const int8_t* y_data = y.data<int8_t>();

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = x.numel();
    const int N = y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        common::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers, "
            "when X/Y's dims =1. But received X has [%d] elements, "
            "received Y has [%d] elements.",
            M,
            N));
    VLOG(3) << "MatMul's case 1";
    out->Resize(common::make_ddim({}));
    dev_ctx.template Alloc<int32_t>(out);
    phi::funcs::Int8GEMM(dev_ctx,
                         CblasNoTrans,
                         CblasTrans,
                         1,
                         1,
                         M,
                         static_cast<int32_t>(1),
                         y_data,
                         x_data,
                         static_cast<int32_t>(0),
                         dev_ctx.template Alloc<int32_t>(out));
    return true;
  }

  if (x_ndim == 1) {
    const int N = x.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 1,
                                          N,
                                          y_ndim - 1,
                                          y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(Y) has error dim. "
                                          "Y'dims[%d] must be equal to %d, "
                                          "but received Y'dims[%d] is %d.",
                                          y_ndim - 2,
                                          N,
                                          y_ndim - 2,
                                          y_dims[y_ndim - 2]));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    out->ResizeAndAllocate(common::make_ddim(out_dims));
    dev_ctx.template Alloc<int32_t>(out);
    if (trans_y) {
      const int M = y.numel() / N;
      VLOG(3) << "MatMul's case 2";
      phi::funcs::Int8GEMV(dev_ctx,
                           false,
                           M,
                           N,
                           static_cast<int32_t>(1),
                           y_data,
                           x_data,
                           static_cast<int32_t>(0),
                           dev_ctx.template Alloc<int32_t>(out));
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = y.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 3";
        phi::funcs::Int8GEMV(dev_ctx,
                             true,
                             N,
                             M,
                             static_cast<int32_t>(1),
                             y_data,
                             x_data,
                             static_cast<int32_t>(0),
                             dev_ctx.template Alloc<int32_t>(out));
      } else {
        VLOG(3) << "MatMul's case 4";
        phi::funcs::Int8BatchedGEMM(dev_ctx,
                                    CblasTrans,
                                    CblasNoTrans,
                                    M,
                                    1,
                                    N,
                                    static_cast<int32_t>(1),
                                    y_data,
                                    x_data,
                                    static_cast<int32_t>(0),
                                    dev_ctx.template Alloc<int32_t>(out),
                                    batch_size,
                                    M * N,
                                    0);
      }
    }
    return true;
  }

  if (y_ndim == 1) {
    const int N = y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d"
                                          "But received X'dims[%d] is %d",
                                          x_ndim - 2,
                                          N,
                                          x_ndim - 2,
                                          x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          common::errors::InvalidArgument("Input(X) has error dim."
                                          "X'dims[%d] must be equal to %d"
                                          "But received X'dims[%d] is %d",
                                          x_ndim - 1,
                                          N,
                                          x_ndim - 1,
                                          x_dims[x_ndim - 1]));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    out->ResizeAndAllocate(common::make_ddim(out_dims));
    dev_ctx.template Alloc<int32_t>(out);

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = x.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 5";
        phi::funcs::Int8GEMV(dev_ctx,
                             true,
                             N,
                             M,
                             static_cast<int32_t>(1),
                             x_data,
                             y_data,
                             static_cast<int32_t>(0),
                             dev_ctx.template Alloc<int32_t>(out));
      } else {
        VLOG(3) << "MatMul's case 6";
        phi::funcs::Int8BatchedGEMM(dev_ctx,
                                    CblasTrans,
                                    CblasNoTrans,
                                    M,
                                    1,
                                    N,
                                    static_cast<int32_t>(1),
                                    x_data,
                                    y_data,
                                    static_cast<int32_t>(0),
                                    dev_ctx.template Alloc<int32_t>(out),
                                    batch_size,
                                    M * N,
                                    0);
      }
    } else {
      const int M = x.numel() / N;
      VLOG(3) << "MatMul's case 7";
      phi::funcs::Int8GEMV(dev_ctx,
                           false,
                           M,
                           N,
                           static_cast<int32_t>(1),
                           x_data,
                           y_data,
                           static_cast<int32_t>(0),
                           dev_ctx.template Alloc<int32_t>(out));
    }
    return true;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 1,
                                        K,
                                        y_ndim - 1,
                                        y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        common::errors::InvalidArgument("Input(Y) has error dim. "
                                        "Y'dims[%d] must be equal to %d, "
                                        "but received Y'dims[%d] is %d.",
                                        y_ndim - 2,
                                        K,
                                        y_ndim - 2,
                                        y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  GetBroadcastFromDims(x_ndim - 2,
                       x_dims.data(),
                       y_ndim - 2,
                       y_dims.data(),
                       x_broadcast_dims.data(),
                       y_broadcast_dims.data(),
                       out_broadcast_dims.data());
  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  out->ResizeAndAllocate(common::make_ddim(out_broadcast_dims));
  dev_ctx.template Alloc<int32_t>(out);

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims =
      !std::equal(x_broadcast_dims.cbegin(),
                  x_broadcast_dims.cbegin() + batch_dim,
                  y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size =
      std::accumulate(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size =
      std::accumulate(y_broadcast_dims.cbegin(),
                      y_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size =
      std::accumulate(out_broadcast_dims.cbegin(),
                      out_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) return true;
  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(3) << "MatMul's case 8";
    phi::funcs::Int8GEMM(dev_ctx,
                         trans_x ? CblasTrans : CblasNoTrans,
                         trans_y ? CblasTrans : CblasNoTrans,
                         M,
                         N,
                         K,
                         static_cast<int32_t>(1),
                         x_data,
                         y_data,
                         static_cast<int32_t>(0),
                         dev_ctx.template Alloc<int32_t>(out));
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul's case 9";
      phi::funcs::Int8GEMV(dev_ctx,
                           false,
                           y_batch_size * N,
                           K,
                           static_cast<int32_t>(1),
                           y_data,
                           x_data,
                           static_cast<int32_t>(0),
                           dev_ctx.template Alloc<int32_t>(out));
    } else {
      VLOG(3) << "MatMul's case 10";
      phi::funcs::Int8BatchedGEMM(dev_ctx,
                                  trans_x ? CblasTrans : CblasNoTrans,
                                  trans_y ? CblasTrans : CblasNoTrans,
                                  M,
                                  N,
                                  K,
                                  static_cast<int32_t>(1),
                                  x_data,
                                  y_data,
                                  static_cast<int32_t>(0),
                                  dev_ctx.template Alloc<int32_t>(out),
                                  out_batch_size,
                                  0,
                                  K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul's case 11";
      phi::funcs::Int8GEMM(dev_ctx,
                           CblasNoTrans,
                           trans_y ? CblasTrans : CblasNoTrans,
                           x_batch_size * M,
                           N,
                           K,
                           static_cast<int32_t>(1),
                           x_data,
                           y_data,
                           static_cast<int32_t>(0),
                           dev_ctx.template Alloc<int32_t>(out));
    } else {
      VLOG(3) << "MatMul's case 12";
      phi::funcs::Int8BatchedGEMM(dev_ctx,
                                  CblasTrans,
                                  trans_y ? CblasTrans : CblasNoTrans,
                                  M,
                                  N,
                                  K,
                                  static_cast<int32_t>(1),
                                  x_data,
                                  y_data,
                                  static_cast<int32_t>(0),
                                  dev_ctx.template Alloc<int32_t>(out),
                                  out_batch_size,
                                  M * K,
                                  0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul's case 13";
    phi::funcs::Int8BatchedGEMM(dev_ctx,
                                trans_x ? CblasTrans : CblasNoTrans,
                                trans_y ? CblasTrans : CblasNoTrans,
                                M,
                                N,
                                K,
                                static_cast<int32_t>(1),
                                x_data,
                                y_data,
                                static_cast<int32_t>(0),
                                dev_ctx.template Alloc<int32_t>(out),
                                out_batch_size,
                                M * K,
                                K * N);
  } else {
    // in the case, can't use stridedgemm
    std::vector<const int8_t*> x_ptr(out_batch_size);
    std::vector<const int8_t*> y_ptr(out_batch_size);
    std::vector<int32_t*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim, 0);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      // using the index to get offset
      const std::int64_t x_index =
          GetIndexMessage(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexMessage(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = dev_ctx.template Alloc<int32_t>(out) + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul's case 14";
    phi::funcs::Int8BatchedGEMM(dev_ctx,
                                trans_x ? CblasTrans : CblasNoTrans,
                                trans_y ? CblasTrans : CblasNoTrans,
                                M,
                                N,
                                K,
                                static_cast<int32_t>(1),
                                x_ptr.data(),
                                y_ptr.data(),
                                static_cast<int32_t>(0),
                                out_ptr.data(),
                                out_batch_size);
  }
  return true;
}
#endif

template <typename Context, typename T>
typename std::enable_if<std::is_integral<T>::value>::type
MatmulJudgeDtypeKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const std::vector<std::int64_t>& x_dims,
                       const std::vector<std::int64_t>& y_dims,
                       DenseTensor* out,
                       bool transpose_x,
                       bool transpose_y) {
#if defined(PADDLE_WITH_CUDA)
  if constexpr (std::is_same<Context, phi::GPUContext>::value &&
                std::is_same<T, int8_t>::value) {
    if (x.dtype() == phi::DataType::INT8 && x_dims[0] <= 4 &&
        y_dims.size() == 2 && y_dims[0] % 16 == 0 && y_dims[1] % 16 == 0 &&
        FLAGS_cuda_core_int8_gemm && ctx.GetComputeCapability() >= 70 &&
        transpose_y) {
      phi::CudaGemm<T, Context>(ctx, x, y, out);
      return;
    }
  }
#endif
  bool try_matmul_int8 = MatMulInt8Function<Context>(
      ctx, x, y, x_dims, y_dims, out, transpose_x, transpose_y);
  if (try_matmul_int8) {
    return;
  }
  auto x_tmp = phi::Cast<T, Context>(ctx, x, phi::DataType::FLOAT32);
  auto y_tmp = phi::Cast<T, Context>(ctx, y, phi::DataType::FLOAT32);
  DenseTensor out_tmp;
  MatMulFunction<Context, float>(
      ctx, x_tmp, y_tmp, x_dims, y_dims, &out_tmp, transpose_x, transpose_y);
  if (x.dtype() == phi::DataType::INT8) {
    phi::CastKernel<float>(ctx, out_tmp, phi::DataType::INT32, out);
    return;
  }
  phi::CastKernel<float>(ctx, out_tmp, x.dtype(), out);
}

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 12010
template <typename Context>
typename std::enable_if<std::is_same<Context, phi::GPUContext>::value>::type
DispatchMatmulFP8Kernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const std::vector<std::int64_t>& x_dims,
                        const std::vector<std::int64_t>& y_dims,
                        DenseTensor* out,
                        bool transpose_x,
                        bool transpose_y) {
  if (x.dtype() != DataType::FLOAT8_E4M3FN ||
      y.dtype() != DataType::FLOAT8_E4M3FN) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "float8 matmul needs input x and y be float8_e4m3fn"));
  }

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::InvalidArgument(
          "Mat x for matmul fp8 just support 2-dim tensor, but got %d",
          x_dims.size()));
  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      2,
      common::errors::InvalidArgument(
          "Mat y for matmul fp8 just support 2-dim tensor, but got %d",
          y_dims.size()));
  PADDLE_ENFORCE_EQ(x_dims[1],
                    y_dims[0],
                    common::errors::InvalidArgument(
                        "x_dims[1] needs to equal to y_dims[0], but "
                        "got x_dims[1] = %d, y_dims[0] = %d",
                        x_dims[1],
                        y_dims[0]));

  PADDLE_ENFORCE_EQ(
      x_dims[1] % 16,
      0,
      common::errors::InvalidArgument(
          "fp8 matmul need x_dims[1] % 16 = 0, got x_dims[1] = %d", x_dims[1]));
  PADDLE_ENFORCE_EQ(
      y_dims[0] % 16,
      0,
      common::errors::InvalidArgument(
          "fp8 matmul need y_dims[0] % 16 = 0, got y_dims[0] = %d", y_dims[0]));

  phi::DenseTensor workspace;
  workspace.Resize({30 * 1024 * 1024});
  ctx.template Alloc<int8_t>(&workspace);
  ctx.template Alloc<phi::dtype::float16>(out);

  CublasLtMatmulFP8<phi::dtype::float16>(ctx, x, y, &workspace, out);
}

template <typename Context>
typename std::enable_if<std::is_same<Context, phi::CPUContext>::value>::type
DispatchMatmulFP8Kernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const std::vector<std::int64_t>& x_dims,
                        const std::vector<std::int64_t>& y_dims,
                        DenseTensor* out,
                        bool transpose_x,
                        bool transpose_y) {}

template <typename Context, typename T>
typename std::enable_if<std::is_same<T, phi::dtype::float8_e4m3fn>::value>::type
DispatchMatmulKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const std::vector<std::int64_t>& x_dims,
                     const std::vector<std::int64_t>& y_dims,
                     DenseTensor* out,
                     bool transpose_x,
                     bool transpose_y) {
  DispatchMatmulFP8Kernel<Context>(
      ctx, x, y, x_dims, y_dims, out, transpose_x, transpose_y);
}
#endif
#endif

template <typename Context, typename T>
typename std::enable_if<
    !std::is_same<T, phi::dtype::float8_e4m3fn>::value>::type
DispatchMatmulKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const std::vector<std::int64_t>& x_dims,
                     const std::vector<std::int64_t>& y_dims,
                     DenseTensor* out,
                     bool transpose_x,
                     bool transpose_y) {
  MatMulFunction<Context, T>(
      ctx, x, y, x_dims, y_dims, out, transpose_x, transpose_y);
}

template <typename Context, typename T>
typename std::enable_if<!std::is_integral<T>::value>::type
MatmulJudgeDtypeKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const std::vector<std::int64_t>& x_dims,
                       const std::vector<std::int64_t>& y_dims,
                       DenseTensor* out,
                       bool transpose_x,
                       bool transpose_y) {
  DispatchMatmulKernel<Context, T>(
      ctx, x, y, x_dims, y_dims, out, transpose_x, transpose_y);
}

template <typename T, typename Context>
void MatmulKernel(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (transpose_x) {
      std::swap(x_dims[x_dims.size() - 1], x_dims[x_dims.size() - 2]);
    }
    if (transpose_y) {
      std::swap(y_dims[y_dims.size() - 1], y_dims[y_dims.size() - 2]);
    }
    std::vector<std::int64_t> out_dims(x_dims.size() - 1 + y_dims.size() - 1);
    for (int64_t i = 0; i < x_dims.size() - 1; ++i) {
      out_dims[i] = x_dims[i];
    }
    for (int64_t i = 1; i < y_dims.size(); ++i) {
      out_dims[x_dims.size() - 1 + i - 1] = y_dims[i];
    }
    out->Resize(phi::make_ddim(out_dims));
    ctx.template Alloc<T>(out);
    return;
  }
  PADDLE_ENFORCE_GE(
      common::product(x.dims()),
      0,
      common::errors::InvalidArgument(
          "The dims of Input(X) should be greater than or equal to 0."));
  PADDLE_ENFORCE_GE(
      common::product(y.dims()),
      0,
      common::errors::InvalidArgument(
          "The dims of Input(Y) should be greater than or equal to 0."));
  const std::vector<std::int64_t> x_dims = common::vectorize(x.dims());
  const std::vector<std::int64_t> y_dims = common::vectorize(y.dims());
  MatmulJudgeDtypeKernel<Context, T>(
      ctx, x, y, x_dims, y_dims, out, transpose_x, transpose_y);
}

template <typename T, typename Context>
void MatmulWithFlattenKernelImpl(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor* out) {
  const DenseTensor x_matrix =
      x.dims().size() > 2 ? phi::ReshapeToMatrix(x, x_num_col_dims) : x;
  const DenseTensor y_matrix =
      y.dims().size() > 2 ? phi::ReshapeToMatrix(y, y_num_col_dims) : y;

  dev_ctx.template Alloc<T>(out);
  auto z_dim = out->dims();
  if (z_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  blas.MatMul(x_matrix, y_matrix, out);
  if (z_dim.size() != 2) {
    out->Resize(z_dim);
  }
}

#ifdef PADDLE_WITH_CUDA

template <typename Context>
void MatmulWithFlattenKernelInt8Impl(const Context& dev_ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& y,
                                     int x_num_col_dims,
                                     int y_num_col_dims,
                                     DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      DataType::INT8,
      common::errors::InvalidArgument(
          "The type of input(x) used in int8 mul must be (%s) "
          "does not match the "
          "type of data (%s) currently contained in the container.",
          phi::CppTypeToDataType<int8_t>::Type(),
          x.dtype()));
  PADDLE_ENFORCE_EQ(
      y.dtype(),
      DataType::INT8,
      common::errors::InvalidArgument(
          "The type of input(y) used in int8 mul must be (%s) "
          "does not match the "
          "type of data (%s) currently contained in the container.",
          phi::CppTypeToDataType<int8_t>::Type(),
          y.dtype()));

  const DenseTensor x_matrix =
      x.dims().size() > 2 ? phi::ReshapeToMatrix(x, x_num_col_dims) : x;
  const DenseTensor y_matrix =
      y.dims().size() > 2 ? phi::ReshapeToMatrix(y, y_num_col_dims) : y;

  PADDLE_ENFORCE_EQ(
      x_matrix.dims()[1],
      y_matrix.dims()[0],
      common::errors::InvalidArgument(
          "X's numbers of columns must be equal to Y's numbers of rows."
          "But received X has [%d] columns,"
          "received Y has [%d] rows",
          x_matrix.dims()[1],
          y_matrix.dims()[0]));

  PADDLE_ENFORCE_EQ((y_matrix.dims()[1] % 4 == 0 || y_matrix.dims()[1] == 1),
                    true,
                    common::errors::InvalidArgument(
                        "The dimension size N used in int8 mul must be 1 "
                        "or a multiple of 4 does not match the size (%d)"
                        "currently contained in the container.",
                        y_matrix.dims()[1]));
  PADDLE_ENFORCE_EQ((x_matrix.dims()[1] % 4 == 0),
                    true,
                    common::errors::InvalidArgument(
                        "The dimension size K used in int8 mul must be a "
                        "multiple of 4 does not match the size (%d) currently"
                        "contained in the container.",
                        x_matrix.dims()[1]));

  dev_ctx.template Alloc<int32_t>(out);
  auto z_dim = out->dims();
  if (z_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }

#if CUDA_VERSION >= 11060
  using blaslt = phi::funcs::MatmulWithCublasLt<int8_t, int32_t>;

  const int8_t* x_data = x_matrix.data<int8_t>();
  const int8_t* y_data = y_matrix.data<int8_t>();

  std::vector<std::int64_t> x_dims = {x_matrix.dims()[0], x_matrix.dims()[1]};
  std::vector<std::int64_t> y_dims = {y_matrix.dims()[0], y_matrix.dims()[1]};
  phi::funcs::MatmulPlanner matmul_planner(
      x_dims,
      y_dims,
      false,
      false,
      phi::CppTypeToDataType<int8_t>::Type(),
      funcs::MatmulFusedType::kMatmul,
      /* bias_data */ nullptr,
      /* reserve_data */ nullptr,
      /* use_addto */ false,
      /* no_exchange */ true);

  blaslt::Run(dev_ctx,
              x_data,
              y_data,
              dev_ctx.template Alloc<int32_t>(out),
              x_matrix.dims()[0],
              y_matrix.dims()[1],
              x_matrix.dims()[1],
              false,
              false,
              &matmul_planner);

  if (z_dim.size() != 2) {
    out->Resize(z_dim);
  }
#endif
}
#endif

#ifdef PADDLE_WITH_CUDA
template <typename Context>
typename std::enable_if<std::is_same<Context, phi::GPUContext>::value,
                        void>::type
DispatchMatmulWithFlattenInt8Kernel(const phi::GPUContext& dev_ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& y,
                                    int x_num_col_dims,
                                    int y_num_col_dims,
                                    DenseTensor* out) {
  MatmulWithFlattenKernelInt8Impl<Context>(
      dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
}
#endif

template <typename Context>
typename std::enable_if<std::is_same<Context, phi::CPUContext>::value,
                        void>::type
DispatchMatmulWithFlattenInt8Kernel(const phi::CPUContext& dev_ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& y,
                                    int x_num_col_dims,
                                    int y_num_col_dims,
                                    DenseTensor* out) {
  PADDLE_THROW(common::errors::Unimplemented(
      "MatmulWithFlatten with CPU is NOT implemented "
      "yet."));
}

template <typename T, typename Context>
typename std::enable_if<std::is_same<T, int8_t>::value, void>::type
DispatchMatmulFlattenKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            int x_num_col_dims,
                            int y_num_col_dims,
                            DenseTensor* out) {
  DispatchMatmulWithFlattenInt8Kernel<Context>(
      dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
}

template <typename T, typename Context>
typename std::enable_if<!std::is_same<T, int8_t>::value, void>::type
DispatchMatmulFlattenKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            int x_num_col_dims,
                            int y_num_col_dims,
                            DenseTensor* out) {
  MatmulWithFlattenKernelImpl<T, Context>(
      dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
}

template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             DenseTensor* out) {
  DispatchMatmulFlattenKernel<T, Context>(
      dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
}

template <typename T, typename Context>
void LegacyMatmulKernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        bool transpose_x,
                        bool transpose_y,
                        float alpha,
                        DenseTensor* out) {
  MatmulKernel<T, Context>(ctx, x, y, transpose_x, transpose_y, out);
  if (std::fabs(alpha - 1.f) > 1e-6f) {
    ScaleKernel<T, Context>(ctx, *out, Scalar(alpha), Scalar(0), false, out);
  }
}
}  // namespace phi
