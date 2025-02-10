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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/phi/kernels/cholesky_kernel.h"

#include <thrust/device_vector.h>

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct MatrixBandPartFunctor {
  /*! Set output as input value outside a central band and 0 inside that band.
   * That is: output[i, j, ..., m, n] = in_band(m, n) * input[i, j, ..., m, n]
   * where: in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper
   * < 0 || (n-m) <= num_upper)
   */
  MatrixBandPartFunctor(const int m,
                        const int n,
                        const int num_lower_diags,
                        const int num_upper_diags,
                        const T* input,
                        T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = static_cast<T>(0);
    } else {
      output_[index] = input_[index];
    }
  }

  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const T* input_;
  T* output_;
};

#define FUNC_WITH_TYPES(m) m(float, S) m(double, D)

#define POTRF_INSTANCE(T, C)                                             \
  void Potrf(const GPUContext& dev_ctx,                                  \
             cublasFillMode_t uplo,                                      \
             int n,                                                      \
             T* A,                                                       \
             int lda,                                                    \
             int* info) {                                                \
    auto handle = dev_ctx.cusolver_dn_handle();                          \
    int workspace_size = 0;                                              \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##potrf_bufferSize( \
        handle, uplo, n, A, lda, &workspace_size));                      \
    auto workspace = phi::memory_utils::Alloc(                           \
        dev_ctx.GetPlace(),                                              \
        workspace_size * sizeof(T),                                      \
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))); \
    T* workspace_ptr = reinterpret_cast<T*>(workspace->ptr());           \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##potrf(            \
        handle, uplo, n, A, lda, workspace_ptr, workspace_size, info));  \
  }

FUNC_WITH_TYPES(POTRF_INSTANCE);

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
#define POTRF_BATCH_INSTANCE(T, C)                                   \
  void PotrfBatched(const GPUContext& dev_ctx,                       \
                    cublasFillMode_t uplo,                           \
                    int n,                                           \
                    T* Aarray[],                                     \
                    int lda,                                         \
                    int* info_array,                                 \
                    int batch_size) {                                \
    auto handle = dev_ctx.cusolver_dn_handle();                      \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##potrfBatched( \
        handle, uplo, n, Aarray, lda, info_array, batch_size));      \
  }

FUNC_WITH_TYPES(POTRF_BATCH_INSTANCE);
#endif

template <typename T, typename Context>
void CholeskyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    bool upper,
                    DenseTensor* out) {
  auto& dims = x.dims();
  if (x.numel() == 0) {
    out->Resize(dims);
    dev_ctx.template Alloc<T>(out);
    return;
  }
  int batch_count = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }
  int m = dims[dims.size() - 1];
  int tensor_size = batch_count * m * m;

  const auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  // matrices are assumed to be stored in column-major order in cusolver
  cublasFillMode_t uplo =
      upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  // portf is inplace, thus copy the triangular part of the input matrices to
  // the output and set the other triangular part to 0 firstly
  phi::funcs::ForRange<GPUContext> for_range(dev_ctx, tensor_size);
  if (upper) {
    MatrixBandPartFunctor<T> matrix_band_part_functor(m,
                                                      m,
                                                      /* num_lower_diags */ 0,
                                                      /* num_upper_diags */ m,
                                                      x_data,
                                                      out_data);
    for_range(matrix_band_part_functor);
  } else {
    MatrixBandPartFunctor<T> matrix_band_part_functor(m,
                                                      m,
                                                      /* num_lower_diags */ m,
                                                      /* num_upper_diags */ 0,
                                                      x_data,
                                                      out_data);
    for_range(matrix_band_part_functor);
  }

  auto info = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int) * batch_count,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  auto* info_ptr = reinterpret_cast<int*>(info->ptr());

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
  if (batch_count > 1) {
    std::vector<T*> output_ptrs;
    for (int i = 0; i < batch_count; i++) {
      output_ptrs.emplace_back(out_data + i * m * m);
    }
    thrust::device_vector<T*> dev_output_ptrs(output_ptrs.begin(),
                                              output_ptrs.end());
    PotrfBatched(dev_ctx,
                 uplo,
                 m,
                 thrust::raw_pointer_cast(dev_output_ptrs.data()),
                 m,
                 info_ptr,
                 batch_count);
    // TODO(guosheng): There seems to a bug in cusolver potrfBatched and need
    // to clear the upper triangle of the output. Remove this workaround once
    // the bug is fixed.
    if (!upper) {
      MatrixBandPartFunctor<T> matrix_band_part_functor(m,
                                                        m,
                                                        /* num_lower_diags */ m,
                                                        /* num_upper_diags */ 0,
                                                        out_data,
                                                        out_data);
      for_range(matrix_band_part_functor);
    }
  } else {
#endif
    for (int i = 0; i < batch_count; i++) {
      Potrf(dev_ctx, uplo, m, out_data + i * m * m, m, info_ptr + i);
    }

#if CUDA_VERSION >= 9020 && !defined(_WIN32)
  }
#endif
  // check the info
  std::vector<int> error_info;  // only for checking positive matrix
  error_info.resize(batch_count);

  memory_utils::Copy(CPUPlace(),
                     error_info.data(),
                     dev_ctx.GetPlace(),
                     info_ptr,
                     sizeof(int) * batch_count,
                     dev_ctx.stream());

  for (int i = 0; i < batch_count; ++i) {
    PADDLE_ENFORCE_EQ(error_info[i],
                      0,
                      errors::PreconditionNotMet(
                          "For batch [%d]: U(%d, %d) is zero, singular U.",
                          i,
                          error_info[i],
                          error_info[i]));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cholesky,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::CholeskyKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
