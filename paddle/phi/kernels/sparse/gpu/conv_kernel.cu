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

#include "paddle/phi/kernels/sparse/conv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
#include "paddle/phi/kernels/sparse/gpu/conv.cu.h"
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
#include "paddle/phi/kernels/sparse/gpu/gather_gemm_scatter.h"
#endif

#include "glog/logging.h"

namespace phi {
namespace sparse {

#define GATHER_GEMM_SCATTER(arch, input_type, x_nnz, kernel)             \
  ({                                                                     \
    const input_type* kernel_ptr = kernel.data<input_type>();            \
    const input_type* x_nnz_ptr = x_nnz.data<input_type>();              \
    for (int i = 0; i < kernel_size; i++) {                              \
      if (h_counter_ptr[i] <= 0) {                                       \
        continue;                                                        \
      }                                                                  \
      const int M = h_counter_ptr[i];                                    \
      const int K = in_channels;                                         \
      const int N = out_channels;                                        \
      const input_type* tmp_kernel_ptr = kernel_ptr + i * K * N;         \
      const IntT* gather_indices = rulebook_ptr + h_offsets_ptr[i];      \
      const IntT* scatter_indices =                                      \
          rulebook_ptr + rulebook_len + h_offsets_ptr[i];                \
      const size_t key = autotune::GenKey(M / features_num_range, N, K); \
      GatherGemmScatterDriver<arch, false, false>(                       \
          dev_ctx,                                                       \
          key,                                                           \
          x_nnz_ptr,                                                     \
          tmp_kernel_ptr,                                                \
          out_values_ptr,                                                \
          out_values_ptr,                                                \
          M,                                                             \
          N,                                                             \
          K,                                                             \
          gather_indices,                                                \
          static_cast<const IntT*>(nullptr),                             \
          scatter_indices,                                               \
          static_cast<T>(1.0),                                           \
          static_cast<T>(1.0),                                           \
          nullptr);                                                      \
    }                                                                    \
  })

template <typename T, typename IntT>
void Conv3dCooGPUKernel(const GPUContext& dev_ctx,
                        const SparseCooTensor& x,
                        const DenseTensor& kernel,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        const int groups,
                        const bool subm,
                        const std::string& key,
                        SparseCooTensor* out,
                        DenseTensor* rulebook,
                        DenseTensor* counter) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  const bool is2D = x_dims.size() == 4 ? true : false;
  int kernel_size = is2D ? kernel_dims[0] * kernel_dims[1]
                         : kernel_dims[0] * kernel_dims[1] * kernel_dims[2];

  int rank = is2D ? 4 : 5;
  std::vector<int> out_dims_vec(rank, 1);
  DDim out_dims = common::make_ddim(out_dims_vec);

  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    // the out shape of subm_conv is same as input shape
    // reset the padding=kernel_size/2 and strides=1
    phi::funcs::sparse::ResetSubmKernelSizeAndStrides(
        kernel.dims(), &subm_paddings, &subm_strides);
  }

  phi::funcs::sparse::GetOutShape(
      x_dims, kernel_sizes, subm_paddings, dilations, subm_strides, &out_dims);
  const int in_channels = is2D ? kernel_dims[2] : kernel_dims[3];
  const int out_channels = is2D ? kernel_dims[3] : kernel_dims[4];
  DenseTensor h_counter, h_offsets;
  h_counter.Resize({kernel_size});
  h_offsets.Resize({kernel_size + 1});
  int* h_counter_ptr = dev_ctx.template HostAlloc<int>(&h_counter);
  int* h_offsets_ptr = dev_ctx.template HostAlloc<int>(&h_offsets);

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensor counter_per_kernel = phi::Empty<int>(dev_ctx, {kernel_size});
  DenseTensor offsets_per_kernel = phi::Empty<int>(dev_ctx, {kernel_size});
  DenseTensor out_index = phi::Empty<int>(dev_ctx, {1});
  DenseTensor unique_value = phi::Empty<int>(dev_ctx, {1});

  if (is2D) {
    VLOG(6) << "call SubmConv2D or Conv2D " << subm << " and the key is "
            << key;
  } else {
    VLOG(6) << "call SubmConv3D or Conv3D " << subm << " and the key is "
            << key;
  }

  int rulebook_len = 0;
  const IntT* rulebook_ptr = nullptr;
  bool need_product_rulebook = true;
  if (subm && !key.empty()) {
    rulebook_ptr = phi::funcs::sparse::PrepareSubm<T, IntT, GPUContext>(
        dev_ctx,
        x,
        key,
        out_dims,
        out,
        h_counter.data<int>(),
        h_offsets.data<int>(),
        &rulebook_len,
        &need_product_rulebook);
  }

  if (need_product_rulebook) {
    DenseTensor tmp_rulebook;
    rulebook_len = ProductRuleBook<T, GPUContext, IntT>(dev_ctx,
                                                        x,
                                                        kernel_sizes,
                                                        subm_paddings,
                                                        dilations,
                                                        subm_strides,
                                                        out_dims,
                                                        subm,
                                                        &tmp_rulebook,
                                                        &counter_per_kernel,
                                                        &offsets_per_kernel,
                                                        &out_index,
                                                        &unique_value,
                                                        out,
                                                        h_counter_ptr,
                                                        h_offsets_ptr);
    rulebook_ptr = tmp_rulebook.data<IntT>();

    phi::funcs::sparse::SaveToTable(
        dev_ctx, x, key, tmp_rulebook, h_counter, out, rulebook, counter);
  }

#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
  bool mixed_precision = dev_ctx.GetComputeCapability() >= 75 &&
                         dev_ctx.GetComputeCapability() < 80 &&
                         std::is_same<T, float>::value;
  bool cutlass = true;
  // NOTE(HaipengMing): 256(in channel)x256(out channel) cutlass kernel could
  // cause CUDA Error(700).
  if (kernel_dims[kernel_dims.size() - 1] == 256 &&
      kernel_dims[kernel_dims.size() - 2] == 256)
    cutlass = false;
  if (dev_ctx.GetComputeCapability() < 75) cutlass = false;
  if (in_channels % 8 != 0 || out_channels % 8 != 0) {
    if (std::is_same<T, phi::dtype::float16>::value) cutlass = false;
    if (mixed_precision) cutlass = false;
  }
  if (in_channels % 4 != 0 || out_channels % 4 != 0) {
    if (std::is_same<T, float>::value) cutlass = false;
  }
  if (std::is_same<T, double>::value) cutlass = false;
  if (!std::is_same<IntT, int32_t>::value) cutlass = false;

  if (cutlass) {
    auto* out_values = out->mutable_non_zero_elements();
    T* out_values_ptr = out_values->data<T>();
    phi::funcs::SetConstant<GPUContext, T> set_zero;
    set_zero(dev_ctx, out_values, static_cast<T>(0.0f));

    if (mixed_precision) {
      DenseTensor kernel_fp16 =
          phi::Cast<T, GPUContext>(dev_ctx, kernel, DataType::FLOAT16);
      DenseTensor x_nnz_fp16 = phi::Cast<T, GPUContext>(
          dev_ctx, x.non_zero_elements(), DataType::FLOAT16);
      GATHER_GEMM_SCATTER(75, phi::dtype::float16, x_nnz_fp16, kernel_fp16);
    } else {
      if (dev_ctx.GetComputeCapability() < 80)
        GATHER_GEMM_SCATTER(75, T, x.non_zero_elements(), kernel);
      else
        GATHER_GEMM_SCATTER(80, T, x.non_zero_elements(), kernel);
    }
  } else {
#endif
    if (subm) {
      auto config =
          phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rulebook_len, 1);
      unique_value.ResizeAndAllocate(
          {static_cast<int>(out->nnz() * kernel_size)});
      out_index.ResizeAndAllocate({static_cast<int>(rulebook_len)});
      int* out_index_ptr = out_index.data<int>();
      int* unique_value_ptr = unique_value.data<int>();
      phi::backends::gpu::GpuMemsetAsync(
          out_index_ptr, 0, sizeof(int) * rulebook_len, dev_ctx.stream());
      GroupIndices<<<config.block_per_grid,
                     config.thread_per_block,
                     0,
                     dev_ctx.stream()>>>(rulebook_len,
                                         kernel_size,
                                         rulebook_ptr + rulebook_len,
                                         out_index_ptr,
                                         unique_value_ptr);
    }
    // 2. gather
    phi::DenseTensor in_features =
        phi::Empty<T>(dev_ctx, {rulebook_len, in_channels});
    phi::DenseTensor out_features =
        phi::Empty<T>(dev_ctx, {rulebook_len, out_channels});
    T* in_features_ptr = in_features.data<T>();
    T* out_features_ptr = out_features.data<T>();
    phi::funcs::SetConstant<GPUContext, T> set_zero;
    set_zero(dev_ctx, &out_features, static_cast<T>(0.0f));

    Gather<T, IntT>(dev_ctx,
                    x.values().data<T>(),
                    rulebook_ptr,
                    rulebook_len,
                    in_channels,
                    in_features_ptr);

    // 3. call gemm for every werght
    auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
    auto* out_values = out->mutable_values();
    T* out_values_ptr = out_values->data<T>();
    set_zero(dev_ctx, out_values, static_cast<T>(0.0f));

    const T* kernel_ptr = kernel.data<T>();
    for (int i = 0; i < kernel_size; i++) {
      if (h_counter_ptr[i] <= 0) {
        continue;
      }

      // call gemm: (n, in_channels) * (in_channels, out_channels)
      const int M = h_counter_ptr[i];
      const int K = in_channels;
      const int N = out_channels;
      T* tmp_in_ptr = in_features_ptr + h_offsets_ptr[i] * in_channels;
      const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
      T* tmp_out_ptr = out_features_ptr + h_offsets_ptr[i] * out_channels;

      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                static_cast<T>(1),
                tmp_in_ptr,
                tmp_kernel_ptr,
                static_cast<T>(0),
                tmp_out_ptr);
    }

    // 4. scatter
    phi::funcs::sparse::ScatterV2<T>(dev_ctx,
                                     out_features_ptr,
                                     out_index.data<int>(),
                                     unique_value.data<int>(),
                                     out->nnz(),
                                     kernel_size,
                                     out_channels,
                                     1,
                                     out_values_ptr);
#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS
  }
#endif
}

/**
 * x: the input SparseCooTensor, shape is (N, D, H, W, C)
 * kernel: the weight data, shape is (D, H, W, C, OC)
 * out: the output SparseCooTensor, shape is (N, D, H, W, OC)
 * rulebook: return rulebook if key is not vailed else return nullptr
 * counter: return counter if key is not vailed else return nullptr
 **/
template <typename T, typename Context>
void Conv3dCooKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     SparseCooTensor* out,
                     DenseTensor* rulebook,
                     DenseTensor* counter) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "Conv3dCooGPUKernel", ([&] {
                                 Conv3dCooGPUKernel<T, data_t>(dev_ctx,
                                                               x,
                                                               kernel,
                                                               paddings,
                                                               dilations,
                                                               strides,
                                                               groups,
                                                               subm,
                                                               key,
                                                               out,
                                                               rulebook,
                                                               counter);
                               }));
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dCooKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
  kernel->OutputAt(1).SetDataType(paddle::DataType::INT32);
  kernel->OutputAt(2).SetDataType(paddle::DataType::INT32);
}
