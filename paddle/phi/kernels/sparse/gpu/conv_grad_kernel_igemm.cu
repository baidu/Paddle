// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/sparse/gpu/conv_grad_kernel_impl.cuh"
#include "paddle/phi/kernels/sparse/gpu/conv_kernel_impl.cuh"
#include "paddle/phi/kernels/sparse/gpu/sparse_map_bwd.cuh"

#include "glog/logging.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void Conv3dGradImplicitGemmGPUKernel(const GPUContext& dev_ctx,
                                     const SparseCooTensor& x,
                                     const DenseTensor& kernel,
                                     const SparseCooTensor& out,
                                     const SparseCooTensor& out_grad,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& dilations,
                                     const std::vector<int>& strides,
                                     const int groups,
                                     const bool subm,
                                     const std::string& key,
                                     SparseCooTensor* x_grad,
                                     DenseTensor* kernel_grad) {
  // Currently, only support x.layout is NDHWC, subm = true, stride = 1, groups
  // = 1, dilations = 1
  PADDLE_ENFORCE_EQ(
      subm,
      true,
      phi::errors::InvalidArgument("The subm must be true, but received %s.",
                                   subm ? "true" : "false"));
  PADDLE_ENFORCE_EQ(groups,
                    1,
                    phi::errors::InvalidArgument(
                        "The group must be 1, but received %d.", groups));

  const bool is_params_freezing = kernel_grad == nullptr;
  const bool is_input_freezing = x_grad == nullptr;
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  const bool is2D = x_dims.size() == 4 ? true : false;

  if (is2D) {
    PADDLE_ENFORCE_EQ(
        (kernel_dims.size() == 4),
        true,
        phi::errors::InvalidArgument(
            "For 2D case, the size of kernel_dims must be 4, but received %d.",
            kernel_dims.size()));
    PADDLE_ENFORCE_EQ(
        (strides.size() == 2 && strides[0] == 1 && strides[1] == 1),
        true,
        phi::errors::InvalidArgument(
            "The strides must be 1, but received %d, %d.",
            strides[0],
            strides[1]));
    PADDLE_ENFORCE_EQ(
        (dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1),
        true,
        phi::errors::InvalidArgument(
            "The dilations must be 1, but received %d, %d.",
            dilations[0],
            dilations[1]));

  } else {
    PADDLE_ENFORCE_EQ(
        (kernel_dims.size() == 5),
        true,
        phi::errors::InvalidArgument(
            "For 3D case, the size of kernel_dims must be 5, but received %d.",
            kernel_dims.size()));
    PADDLE_ENFORCE_EQ((strides.size() == 3 && strides[0] == 1 &&
                       strides[1] == 1 && strides[2] == 1),
                      true,
                      phi::errors::InvalidArgument(
                          "The strides must be 1, but received %d, %d, %d.",
                          strides[0],
                          strides[1],
                          strides[2]));
    PADDLE_ENFORCE_EQ((dilations.size() == 3 && dilations[0] == 1 &&
                       dilations[1] == 1 && dilations[2] == 1),
                      true,
                      phi::errors::InvalidArgument(
                          "The dilations must be 1, but received %d, %d, %d.",
                          dilations[0],
                          dilations[1],
                          dilations[2]));
  }

  int kernel_volume = is2D ? kernel_dims[0] * kernel_dims[1]
                           : kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  int in_channels = is2D ? kernel_dims[2] : kernel_dims[3];
  int out_channels = is2D ? kernel_dims[3] : kernel_dims[4];

  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  // Set the x_grad tensor
  if (!is_input_freezing) {
    if (subm) {
      DenseTensor x_grad_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
      int tmpidx = is2D ? 2 : 3;
      DenseTensor x_grad_values =
          phi::Empty<T>(dev_ctx, {x.nnz(), kernel_sizes[tmpidx]});
      phi::Copy(
          dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &x_grad_indices);
      x_grad->SetMember(x_grad_indices, x_grad_values, x.dims(), false);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "The subm must be true, but received %s.", subm ? "true" : "false"));
    }
  }

  auto* out_kmap_cache_ptr = out.GetKmapCache(key);

  const int divisor = 128;
  DenseTensor* out_in_map_bwd = new phi::DenseTensor();
  out_in_map_bwd->Resize(
      {(x.nnz() + divisor - 1) / divisor * divisor, kernel_volume});
  dev_ctx.template Alloc<int32_t>(out_in_map_bwd);
  phi::funcs::SetConstant<phi::GPUContext, int32_t> set_neg_one;
  set_neg_one(dev_ctx, out_in_map_bwd, static_cast<int32_t>(-1));
  convert_transposed_out_in_map<IntT>(
      dev_ctx, *(out_kmap_cache_ptr->out_in_map), out_in_map_bwd);

  DenseTensor kernel_transpose = phi::EmptyLike<T, GPUContext>(dev_ctx, kernel);
  std::vector<int> perm1;
  if (is2D) {
    perm1 = {1, 0, 2, 3};
  } else {
    perm1 = {2, 1, 0, 3, 4};
  }
  phi::funcs::TransposeGPUKernelDriver<T>(
      dev_ctx, kernel, perm1, &kernel_transpose);

  // kernel_volume in_channels out_channels --> kernel_volume out_channels
  DenseTensor kernel_transpose2;
  std::vector<int> perm2;
  if (is2D) {
    kernel_transpose2 = phi::Empty<T>(
        dev_ctx,
        {kernel_dims[0], kernel_dims[1], kernel_dims[3], kernel_dims[2]});
    perm2 = {0, 1, 3, 2};
  } else {
    kernel_transpose2 = phi::Empty<T>(dev_ctx,
                                      {kernel_dims[0],
                                       kernel_dims[1],
                                       kernel_dims[2],
                                       kernel_dims[4],
                                       kernel_dims[3]});
    perm2 = {0, 1, 2, 4, 3};
  }
  phi::funcs::TransposeGPUKernelDriver<T>(
      dev_ctx, kernel_transpose, perm2, &kernel_transpose2);

#ifdef PADDLE_WITH_CUDA
  if (!is_input_freezing) {
    conv_forward_implicit_gemm_cuda<IntT>(dev_ctx,
                                          out_grad.values(),
                                          kernel_transpose2,
                                          *(out_in_map_bwd),
                                          x.nnz(),
                                          in_channels,
                                          *(x_grad->mutable_values()));
  }

  if (!is_params_freezing) {
    DenseTensor kernel_grad_out =
        phi::Empty<T>(dev_ctx, {32, out_channels * kernel_volume, in_channels});
    conv_backward_wgrad_implicit_gemm_cuda<IntT>(dev_ctx,
                                                 out_grad.values(),
                                                 x.values(),
                                                 *(out_in_map_bwd),
                                                 32,
                                                 kernel_grad_out);

    DenseTensor kernel_grad_sum;
    kernel_grad_sum = phi::Sum<T>(
        dev_ctx, kernel_grad_out, {0}, kernel_grad_out.dtype(), false);
    kernel_grad_sum.Resize({kernel_volume, out_channels, in_channels});
    DenseTensor kernel_grad_transpose =
        phi::Empty<T>(dev_ctx, {kernel_volume, in_channels, out_channels});
    std::vector<int> perm3 = {0, 2, 1};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx, kernel_grad_sum, perm3, &kernel_grad_transpose);
    kernel_grad_transpose.Resize(kernel_transpose.dims());

    *kernel_grad = phi::EmptyLike<T, GPUContext>(dev_ctx, kernel);
    std::vector<int> perm4;
    if (is2D) {
      perm4 = {1, 0, 2, 3};
    } else {
      perm4 = {2, 1, 0, 3, 4};
    }
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx, kernel_grad_transpose, perm4, kernel_grad);
  }
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "conv_implicit_gemm is only supported on CUDA."));
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
void Conv3dGradImplicitGemmKernel(const Context& dev_ctx,
                                  const SparseCooTensor& x,
                                  const DenseTensor& kernel,
                                  const SparseCooTensor& out,
                                  const SparseCooTensor& out_grad,
                                  const std::vector<int>& paddings,
                                  const std::vector<int>& dilations,
                                  const std::vector<int>& strides,
                                  const int groups,
                                  const bool subm,
                                  const std::string& key,
                                  SparseCooTensor* x_grad,
                                  DenseTensor* kernel_grad) {
#ifdef PADDLE_WITH_CUDA
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "Conv3dGradImplicitGemmGPUKernel", ([&] {
        Conv3dGradImplicitGemmGPUKernel<T, int64_t>(dev_ctx,
                                                    x,
                                                    kernel,
                                                    out,
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
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "Conv3dGradImplicitGemmKernel is only supported on CUDA."));
#endif
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_grad_implicit_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dGradImplicitGemmKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}
