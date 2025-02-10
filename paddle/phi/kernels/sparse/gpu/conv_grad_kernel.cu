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

#include "paddle/phi/kernels/sparse/conv_grad_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/sparse/gpu/conv.cu.h"
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
#include "paddle/phi/kernels/sparse/gpu/gather_gemm_scatter.h"
#endif

namespace phi {
namespace sparse {
extern size_t workspace_size;

// rulebook[3, rulebook_len]:
//[
//  [kernel_index],
//  [in_i],
//  [out_i],
//]
// x_grad = out_grad * transpose(kernel)
// kernel_grad = transpose(x) * out_grad
template <typename T, typename IntT>
void Conv3dCooGradGPUKernel(const GPUContext& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& kernel,
                            const SparseCooTensor& out,
                            const DenseTensor& rulebook,
                            const DenseTensor& counter,
                            const SparseCooTensor& out_grad,
                            const std::vector<int>& paddings,
                            const std::vector<int>& dilations,
                            const std::vector<int>& strides,
                            const int groups,
                            const bool subm,
                            const std::string& key,
                            SparseCooTensor* x_grad,
                            DenseTensor* kernel_grad) {
  const bool is_params_freezing = kernel_grad == nullptr;
  const auto& kernel_dims = kernel.dims();
  const bool is2D = kernel_dims.size() == 4 ? true : false;
  const int kernel_size =
      is2D ? kernel_dims[0] * kernel_dims[1]
           : kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int in_channels = is2D ? kernel_dims[2] : kernel_dims[3];
  const int out_channels = is2D ? kernel_dims[3] : kernel_dims[4];

  int rulebook_len = 0;
  const IntT* rulebook_ptr = phi::funcs::sparse::GetRulebookPtr<IntT>(
      out, rulebook, key, &rulebook_len);
  const int* counter_ptr = phi::funcs::sparse::GetCounterPtr(out, counter, key);

  phi::DenseTensor in_features =
      phi::Empty<T>(dev_ctx, {rulebook_len, in_channels});
  phi::DenseTensor d_x_features =
      phi::Empty<T>(dev_ctx, {rulebook_len, in_channels});
  phi::DenseTensor out_grad_features =
      phi::Empty<T>(dev_ctx, {rulebook_len, out_channels});

  T* in_features_ptr = in_features.data<T>();
  T* d_x_features_ptr = d_x_features.data<T>();
  T* out_grad_features_ptr = out_grad_features.data<T>();
  T* d_kernel_ptr = nullptr;
  if (!is_params_freezing) {
    *kernel_grad = phi::EmptyLike<T>(dev_ctx, kernel);
    d_kernel_ptr = kernel_grad->data<T>();
    phi::backends::gpu::GpuMemsetAsync(
        d_kernel_ptr, 0, sizeof(T) * kernel_grad->numel(), dev_ctx.stream());
  }

  int half_kernel_size = kernel_size / 2;
  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  DenseTensor x_grad_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
  DenseTensor x_grad_values = phi::EmptyLike<T>(dev_ctx, x.values());
  T* x_grad_values_ptr = x_grad_values.data<T>();
  phi::backends::gpu::GpuMemsetAsync(x_grad_values_ptr,
                                     0,
                                     sizeof(T) * x_grad_values.numel(),
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemsetAsync(
      d_x_features_ptr, 0, sizeof(T) * d_x_features.numel(), dev_ctx.stream());
  phi::Copy<GPUContext>(
      dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &x_grad_indices);
  x_grad->SetMember(x_grad_indices, x_grad_values, x.dims(), true);

  std::vector<int> offsets(kernel_size + 1);

  int offset = 0, max_count = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter_ptr[i];
    if (i < half_kernel_size) {
      max_count = std::max(max_count, counter_ptr[i]);
    }
  }
  offsets[kernel_size] = offset;

  if (subm) {
    phi::funcs::sparse::SubmPreProcess<T, GPUContext>(dev_ctx,
                                                      x,
                                                      kernel,
                                                      out_grad.values(),
                                                      in_channels,
                                                      out_channels,
                                                      half_kernel_size,
                                                      kernel_grad,
                                                      &x_grad_values);
    if (max_count == 0) {
      return;
    }
  }

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
  DenseTensor unique_value = phi::Empty<int>(
      dev_ctx, {static_cast<int>(x_grad->nnz() * kernel_size * 2)});
  DenseTensor out_index =
      phi::Empty<int>(dev_ctx, {static_cast<int>(x.nnz() * 2)});
  int* out_index_ptr = out_index.data<int>();
  int* unique_value_ptr = unique_value.data<int>();
  phi::backends::gpu::GpuMemsetAsync(
      out_index_ptr, 0, sizeof(int) * x.nnz() * 2, dev_ctx.stream());

#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
  bool cutlass = true;
  if (dev_ctx.GetComputeCapability() < 80) cutlass = false;

  if (in_channels % 4 != 0 || out_channels % 4 != 0) cutlass = false;

  if (std::is_same<T, phi::dtype::float16>::value ||
      std::is_same<T, double>::value)
    cutlass = false;

  if (!std::is_same<IntT, int32_t>::value) cutlass = false;

  if (!cutlass) {
#endif

    GroupIndicesV2<<<config.block_per_grid,
                     config.thread_per_block,
                     0,
                     dev_ctx.stream()>>>(rulebook_len,
                                         x.nnz(),
                                         kernel_size,
                                         offsets[kernel_size / 2],
                                         rulebook_ptr,
                                         out_index_ptr,
                                         unique_value_ptr);

    GatherV2<T, IntT>(dev_ctx,
                      x.values().data<T>(),
                      out_index_ptr,
                      unique_value_ptr,
                      x.nnz(),
                      kernel_size,
                      in_channels,
                      2,
                      in_features_ptr);

    Gather<T, IntT>(dev_ctx,
                    out_grad.values().data<T>(),
                    rulebook_ptr + rulebook_len,
                    rulebook_len,
                    out_channels,
                    out_grad_features_ptr);

#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
  }
#endif
  const T* kernel_ptr = kernel.data<T>();
  T* tmp_d_x_ptr = nullptr;
  T* tmp_d_kernel_ptr = nullptr;
  for (int i = 0; i < kernel_size; i++) {
    if (counter_ptr[i] <= 0 || (subm && i == half_kernel_size)) {
      continue;
    }

    const int M = counter_ptr[i];
    const int K = in_channels;
    const int N = out_channels;
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    T* tmp_out_grad_ptr = out_grad_features_ptr + offsets[i] * out_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * in_channels * out_channels;
    tmp_d_x_ptr = d_x_features_ptr + offsets[i] * in_channels;
    if (!is_params_freezing) {
      tmp_d_kernel_ptr = d_kernel_ptr + i * in_channels * out_channels;
    }

#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
    if (cutlass) {
      const IntT* gather_x_indices = rulebook_ptr + offsets[i];
      const IntT* scatter_x_indices = rulebook_ptr + offsets[i];
      const IntT* gather_out_indices = rulebook_ptr + rulebook_len + offsets[i];
      const size_t key = autotune::GenKey(M / features_num_range, N, K);
      if (!is_params_freezing) {
        // call gemm: d_kernel = transpose(x) * out_grad
        // (in_channels, n) * (n, out_channels)
        static cutlass::device_memory::allocation<uint8_t> workspace(
            workspace_size);
        GatherGemmScatterDriver<80, true, false>(
            dev_ctx,
            key,
            x.values().data<T>(),
            out_grad.values().data<T>(),
            tmp_d_kernel_ptr,
            tmp_d_kernel_ptr,
            in_channels,
            out_channels,
            counter_ptr[i],
            gather_x_indices,
            gather_out_indices,
            static_cast<const IntT*>(nullptr),
            static_cast<const T>(1.0),
            static_cast<const T>(0.0),
            &workspace);
      }
      // call gemm: d_x = out_grad * transpose(kernel)
      // (n, out_channels) * (out_channels, in_channels)
      GatherGemmScatterDriver<80, false, true>(
          dev_ctx,
          key,
          out_grad.values().data<T>(),
          tmp_kernel_ptr,
          x_grad_values_ptr,
          x_grad_values_ptr,
          counter_ptr[i],
          in_channels,
          out_channels,
          gather_out_indices,
          static_cast<const IntT*>(nullptr),
          scatter_x_indices,
          static_cast<const T>(1.0),
          static_cast<const T>(1.0),
          nullptr);
    } else {
#endif
      if (!is_params_freezing) {
        // call gemm: d_kernel = transpose(x) * out_grad
        // (in_channels, n) * (n, out_channels)
        blas.GEMM(CblasTrans,
                  CblasNoTrans,
                  K,
                  N,
                  M,
                  static_cast<T>(1),
                  tmp_in_ptr,
                  tmp_out_grad_ptr,
                  static_cast<T>(0),
                  tmp_d_kernel_ptr);
      }

      // call gemm: d_x = out_grad * transpose(kernel)
      // (n, out_channels) * (out_channels, in_channels)
      blas.GEMM(CblasNoTrans,
                CblasTrans,
                M,
                K,
                N,
                static_cast<T>(1),
                tmp_out_grad_ptr,
                tmp_kernel_ptr,
                static_cast<T>(0),
                tmp_d_x_ptr);
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
    }
#endif
  }

  // 4. scatter
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
  if (!cutlass) {
#endif
    phi::funcs::sparse::ScatterV2<T>(dev_ctx,
                                     d_x_features_ptr,
                                     out_index.data<int>(),
                                     unique_value.data<int>(),
                                     x_grad->nnz(),
                                     kernel_size,
                                     in_channels,
                                     2,
                                     x_grad_values_ptr);
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
  }
#endif
}

template <typename T, typename Context>
void Conv3dCooGradKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& kernel,
                         const SparseCooTensor& out,
                         const DenseTensor& rulebook,
                         const DenseTensor& counter,
                         const SparseCooTensor& out_grad,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const int groups,
                         const bool subm,
                         const std::string& key,
                         SparseCooTensor* x_grad,
                         DenseTensor* kernel_grad) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "Conv3dCooGradGPUKernel", ([&] {
        Conv3dCooGradGPUKernel<T, data_t>(dev_ctx,
                                          x,
                                          kernel,
                                          out,
                                          rulebook,
                                          counter,
                                          out_grad,
                                          paddings,
                                          dilations,
                                          strides,
                                          groups,
                                          subm,
                                          key,
                                          x_grad,
                                          kernel_grad);
      }));
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dCooGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
