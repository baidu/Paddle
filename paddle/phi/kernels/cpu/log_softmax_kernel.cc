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

#include <immintrin.h>
#include <omp.h>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/log_softmax_kernel.h"

namespace phi {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrixTemplate = EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

#ifdef __AVX512F__
static inline __m512 vexp(const __m512& _x) {
  __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 p16f_half = _mm512_set1_ps(0.5f);
  __m512 p16f_127 = _mm512_set1_ps(127.f);
  __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

  __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

  // Clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

  // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
  // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
  // truncation errors. Note that we don't use the "pmadd" function here to
  // ensure that a precision-preserving FMA instruction is used.
  __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

  __m512 r2 = _mm512_mul_ps(r, r);

  // TODO(bukejiyu): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);

  // Build emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // Return 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
}

template <typename Context, typename T>
struct LogSoftmaxAvxFunctor {
  void operator()(const Context& context,
                  const DenseTensor* x,
                  DenseTensor* out,
                  const int axis) {
    const T* x_data = x->data<T>();
    T* out_data = out->data<T>();
    auto matrix_dim = common::flatten_to_2d(x->dims(), axis);
    int32_t rows = static_cast<int32_t>(matrix_dim[0]);
    int32_t cols = static_cast<int32_t>(matrix_dim[1]);
    int32_t size = cols;
    auto iStride = cols;
    auto oStride = cols;
    float max = std::numeric_limits<float>::lowest();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int r = 0; r < rows; ++r) {
      __m512 vmax = _mm512_set1_ps(max);
      const T* px = x_data + r * iStride;
      T* py = out_data + r * oStride;
      for (int off = 0; off < size; off += 16) {
        int remain = size - off;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

        __m512 vx = _mm512_maskz_loadu_ps(mask, px + off);
        vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx);
      }
      max = _mm512_reduce_max_ps(vmax);
      vmax = _mm512_set1_ps(max);

      // Compute vexp(vx - vmax) and sum it
      __m512 vsum = _mm512_set1_ps(0);
      for (int off = 0; off < size; off += 16) {
        int remain = size - off;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

        __m512 vx = _mm512_maskz_loadu_ps(mask, px + off);
        vx = _mm512_mask_sub_ps(vx, mask, vx, vmax);
        vx = vexp(vx);

        vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
      }

      float sum = _mm512_reduce_add_ps(vsum);
      float logsum = std::log(sum);
      __m512 vsub = _mm512_set1_ps(max + logsum);

      // Compute vx - max - logsum and store
      for (int off = 0; off < size; off += 16) {
        int remain = size - off;
        __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

        __m512 vx = _mm512_maskz_loadu_ps(mask, px + off);
        vx = _mm512_mask_sub_ps(vx, mask, vx, vsub);
        _mm512_mask_storeu_ps(py + off, mask, vx);
      }
    }
  }
};
#endif

template <typename Context, typename T>
struct LogSoftmaxFunctor {
  void operator()(const Context& context,
                  const DenseTensor* X,
                  DenseTensor* Y,
                  const int axis) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    int axis_dim = static_cast<int>(X->dims()[axis]);
    const int n = funcs::SizeToAxis(axis, X->dims());
    const int d = funcs::SizeFromAxis(axis, X->dims());
    phi::DDim dim_2d{n, d};

    auto logits = EigenMatrixTemplate<T>::From(*X, dim_2d);
    auto log_softmax = EigenMatrixTemplate<T>::From(*Y, dim_2d);

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_axis(kAxisDim);
    Eigen::DSizes<int, 2> batch_classes(batch_size, num_classes);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_one_remain(batch_size, 1, num_remain);
    Eigen::DSizes<int, 3> one_axis_one(1, axis_dim, 1);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);

    // For numerical stability, logits should be shifted by maximum number along
    // axis, calculate shifted_logits into log_softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      log_softmax.device(*context.eigen_device()) =
          (logits - logits.maximum(along_axis)
                        .eval()
                        .reshape(batch_by_one)
                        .broadcast(one_by_class))
              .unaryExpr(ValueClip<T>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      log_softmax.device(*context.eigen_device()) =
          (logits.reshape(batch_axis_remain) - logits.reshape(batch_axis_remain)
                                                   .maximum(along_axis)
                                                   .eval()
                                                   .reshape(batch_one_remain)
                                                   .broadcast(one_axis_one)
                                                   .reshape(batch_classes))
              .unaryExpr(ValueClip<T>());
    }

    log_softmax.device(*context.eigen_device()) =
        log_softmax - log_softmax.exp()
                          .eval()
                          .reshape(batch_axis_remain)
                          .sum(along_axis)
                          .log()
                          .broadcast(one_axis);
  }
};

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const int rank = x.dims().size();
  const int canonical_axis = funcs::CanonicalAxis(axis, rank);

  dev_ctx.template Alloc<T>(out);
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, out, static_cast<T>(0.0));
    return;
  }
  if (x.numel() != 0) {
    if (std::is_same<T, float>::value && canonical_axis == rank - 1) {
#ifdef __AVX512F__
      LogSoftmaxAvxFunctor<Context, T>()(dev_ctx, &x, out, canonical_axis);
#else
      LogSoftmaxFunctor<Context, T>()(dev_ctx, &x, out, canonical_axis);
#endif
    } else {
      LogSoftmaxFunctor<Context, T>()(dev_ctx, &x, out, canonical_axis);
    }
  }
}

}  // namespace phi

// TODO(YuanRisheng): The layout of onednn kernel should be OneDNN, we should
// support specifying the exact layout when the kernel is registered
PD_REGISTER_KERNEL(
    log_softmax, CPU, ALL_LAYOUT, phi::LogSoftmaxKernel, float, double) {}
