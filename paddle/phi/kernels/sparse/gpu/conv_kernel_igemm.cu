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
#include "paddle/phi/kernels/funcs/sparse/convolution.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/sparse/gpu/conv_kernel_impl.cuh"
#include "paddle/phi/kernels/sparse/gpu/sparse_conv_hashmap.cuh"

#include "glog/logging.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void Conv3dImplicitGemmGPUKernel(const GPUContext& dev_ctx,
                                 const SparseCooTensor& x,
                                 const DenseTensor& kernel,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& dilations,
                                 const std::vector<int>& strides,
                                 const int groups,
                                 const bool subm,
                                 const std::string& key,
                                 SparseCooTensor* out) {
  // Currently, only support x.layout is NDHWC, subm = true, stride = 1, groups
  // = 1, dilations = 1
  PADDLE_ENFORCE_EQ(
      subm,
      true,
      common::errors::InvalidArgument("The subm must be true, but received %s.",
                                      subm ? "true" : "false"));
  PADDLE_ENFORCE_EQ(groups,
                    1,
                    common::errors::InvalidArgument(
                        "The group must be 1, but received %d.", groups));

  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  const bool is2D = x_dims.size() == 4 ? true : false;

  if (is2D) {
    PADDLE_ENFORCE_EQ(
        (kernel_dims.size() == 4),
        true,
        common::errors::InvalidArgument(
            "For 2D case, the size of kernel_dims must be 4, but received %d.",
            kernel_dims.size()));
    PADDLE_ENFORCE_EQ(
        (strides.size() == 2 && strides[0] == 1 && strides[1] == 1),
        true,
        common::errors::InvalidArgument(
            "The strides must be 1, but received %d, %d.",
            strides[0],
            strides[1]));
    PADDLE_ENFORCE_EQ(
        (dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1),
        true,
        common::errors::InvalidArgument(
            "The dilations must be 1, but received %d, %d.",
            dilations[0],
            dilations[1]));

  } else {
    PADDLE_ENFORCE_EQ(
        (kernel_dims.size() == 5),
        true,
        common::errors::InvalidArgument(
            "For 3D case, the size of kernel_dims must be 5, but received %d.",
            kernel_dims.size()));
    PADDLE_ENFORCE_EQ((strides.size() == 3 && strides[0] == 1 &&
                       strides[1] == 1 && strides[2] == 1),
                      true,
                      common::errors::InvalidArgument(
                          "The strides must be 1, but received %d, %d, %d.",
                          strides[0],
                          strides[1],
                          strides[2]));
    PADDLE_ENFORCE_EQ((dilations.size() == 3 && dilations[0] == 1 &&
                       dilations[1] == 1 && dilations[2] == 1),
                      true,
                      common::errors::InvalidArgument(
                          "The dilations must be 1, but received %d, %d, %d.",
                          dilations[0],
                          dilations[1],
                          dilations[2]));
  }

  int kernel_volume = is2D ? kernel_dims[0] * kernel_dims[1]
                           : kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  int in_channels = is2D ? kernel_dims[2] : kernel_dims[3];
  int out_channels = is2D ? kernel_dims[3] : kernel_dims[4];

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

  // Set the output tensor
  if (subm) {
    DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, x.indices());
    int tmpidx = is2D ? 3 : 4;
    DenseTensor out_values =
        phi::Empty<T>(dev_ctx, {x.nnz(), kernel_sizes[tmpidx]});
    phi::Copy(dev_ctx, x.indices(), dev_ctx.GetPlace(), false, &out_indices);
    out->SetMember(out_indices, out_values, out_dims, false);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "The subm must be true, but received %s.", subm ? "true" : "false"));
  }

  build_sparse_conv_kmap<IntT>(
      dev_ctx, x, key, kernel_sizes, strides, kernel_volume, is2D, out);

  auto* out_kmap_cache_ptr = out->GetKmapCache(key);

  DenseTensor kernel_transpose = phi::EmptyLike<T, GPUContext>(dev_ctx, kernel);
  std::vector<int> perm;
  if (is2D) {
    perm = {1, 0, 2, 3};
  } else {
    perm = {2, 1, 0, 3, 4};
  }
  phi::funcs::TransposeGPUKernelDriver<T>(
      dev_ctx, kernel, perm, &kernel_transpose);

#ifdef PADDLE_WITH_CUDA
  conv_forward_implicit_gemm_cuda<IntT>(dev_ctx,
                                        x.values(),
                                        kernel_transpose,
                                        *(out_kmap_cache_ptr->out_in_map),
                                        out->nnz(),
                                        out_channels,
                                        *(out->mutable_values()));
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "conv_forward_implicit_gemm_cuda is only supported on CUDA."));
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
void Conv3dImplicitGemmKernel(const Context& dev_ctx,
                              const SparseCooTensor& x,
                              const DenseTensor& kernel,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              const std::vector<int>& strides,
                              const int groups,
                              const bool subm,
                              const std::string& key,
                              SparseCooTensor* out) {
#ifdef PADDLE_WITH_CUDA
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "Conv3dImplicitGemmGPUKernel's indices", ([&] {
        // Conv3dImplicitGemmGPUKernel<T, data_t>(dev_ctx,
        Conv3dImplicitGemmGPUKernel<T, int64_t>(dev_ctx,
                                                x,
                                                kernel,
                                                paddings,
                                                dilations,
                                                strides,
                                                groups,
                                                subm,
                                                key,
                                                out);
      }));
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "Conv3dImplicitGemmKernel is only supported on CUDA."));
#endif
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_implicit_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dImplicitGemmKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}
