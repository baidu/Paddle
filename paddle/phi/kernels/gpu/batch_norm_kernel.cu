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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/norm_utils.cu.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

COMMON_DECLARE_bool(cudnn_batchnorm_spatial_persistent);
#ifdef PADDLE_WITH_HIP
COMMON_DECLARE_bool(batch_norm_use_miopen);
#endif

namespace phi {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, phi::DataLayout layout>
static __global__ void BNForwardInference(const T *x,
                                          const BatchNormParamType<T> *mean,
                                          const BatchNormParamType<T> *variance,
                                          const BatchNormParamType<T> *scale,
                                          const BatchNormParamType<T> *bias,
                                          const int C,
                                          const int N,
                                          const int HxW,
                                          const double epsilon,
                                          T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num = N * C * HxW;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> x_sub_mean =
        static_cast<BatchNormParamType<T>>(x[i]) - mean[c];
    BatchNormParamType<T> inv_var = 1 / sqrt(variance[c] + epsilon);
    y[i] = static_cast<T>(scale[c] * x_sub_mean * inv_var + bias[c]);
  }
}

template <typename T>
static __global__ void InverseVariance(const BatchNormParamType<T> *variance,
                                       const double epsilon,
                                       const int C,
                                       BatchNormParamType<T> *inv_variance) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < C) {
    inv_variance[tid] = 1 / sqrt(variance[tid] + epsilon);
  }
}

template <typename T, phi::DataLayout layout>
static __global__ void BN1DForwardInference(
    const T *x,
    const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *inv_variance,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num = N * C * HxW;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> x_sub_mean =
        static_cast<BatchNormParamType<T>>(x[i]) - mean[c];
    y[i] = static_cast<T>(scale[c] * x_sub_mean * inv_variance[c] + bias[c]);
  }
}

template <typename T, int BlockDim, phi::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNForwardTraining(
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    double exponentialAverageFactor,
    T *y,
    BatchNormParamType<T> *mean,
    BatchNormParamType<T> *variance,
    BatchNormParamType<T> *save_mean,
    BatchNormParamType<T> *save_inv_variance) {
  int outer_size = C;
  int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storage;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> variance_val;
  __shared__ BatchNormParamType<T> inv_var_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_i = static_cast<BatchNormParamType<T>>(x[index]);
      x_sum += x_i;
      x_square_sum += x_i * x_i;
    }
    x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
    x_square_sum =
        BlockReduce(variance_storage).Reduce(x_square_sum, cub::Sum());
    if (threadIdx.x == 0) {
      mean_val = x_sum / inner_size;
      variance_val = x_square_sum / inner_size - mean_val * mean_val;
      inv_var_val = 1 / sqrt(variance_val + epsilon);

      if (save_mean && save_inv_variance) {
        save_mean[i] = mean_val;
        save_inv_variance[i] = inv_var_val;
      }
      mean[i] = (1 - exponentialAverageFactor) * mean_val +
                exponentialAverageFactor * mean[i];
      variance[i] = (1 - exponentialAverageFactor) * variance_val +
                    exponentialAverageFactor * variance[i];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[index]) - mean_val;
      y[index] = scale[i] * x_sub_mean * inv_var_val + bias[i];
    }
  }
}

template <typename T>
__device__ __forceinline__ void merge_block_horizontal(
    BatchNormParamType<T> x_sum,
    BatchNormParamType<T> x_square_sum,
    BatchNormParamType<T> *smem_sum,
    BatchNormParamType<T> *smem_square_sum,
    BatchNormParamType<T> *x_sum_out,
    BatchNormParamType<T> *x_square_sum_out) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
#pragma unroll
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset * 2) {
      smem_sum[tid] = x_sum;
      smem_square_sum[tid] = x_square_sum;
    }
    __syncthreads();
    if (threadIdx.x < offset) {
      int pair_tid = tid + offset;
      x_sum += smem_sum[pair_tid];
      x_square_sum += smem_square_sum[pair_tid];
    }
  }
  if (threadIdx.x == 0) {
    *x_sum_out = x_sum;
    *x_square_sum_out = x_square_sum;
  }
}

template <typename T, int BlockDim>
static __global__ void BNForwardTraining2DChannelLastCompStat(
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    double exponentialAverageFactor,
    T *y,
    BatchNormParamType<T> *global_mean,
    BatchNormParamType<T> *global_variance,
    BatchNormParamType<T> *save_mean,
    BatchNormParamType<T> *save_inv_variance,
    BatchNormParamType<T> *compute_mean,
    BatchNormParamType<T> *compute_inv_var,
    BatchNormParamType<T> *block_data_ptr,
    int *flag_ptr) {
  int outer_size = C;
  int inner_size = N * HxW;

  __shared__ BatchNormParamType<T> smem_sum[BlockDim];
  __shared__ BatchNormParamType<T> smem_square_sum[BlockDim];

  int outer_loop_stride = gridDim.x * blockDim.x;
  int inner_loop_stride = gridDim.y * blockDim.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outer_size;
       i += outer_loop_stride) {
    BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < inner_size;
         j += inner_loop_stride) {
      const int index = j * outer_size + i;
      BatchNormParamType<T> x_i = static_cast<BatchNormParamType<T>>(x[index]);
      x_sum += x_i;
      x_square_sum += x_i * x_i;
    }

    // vertical block sum
    funcs::BlockReduceByVertical<T, BatchNormParamType<T>>(x_sum,
                                                           x_square_sum,
                                                           &smem_sum[0],
                                                           &smem_square_sum[0],
                                                           &x_sum,
                                                           &x_square_sum);

    if (gridDim.y > 1) {
      __shared__ bool is_last_block_done;
      funcs::ReduceSumPost<T, BatchNormParamType<T>>(C,
                                                     i,
                                                     &x_sum,
                                                     &x_square_sum,
                                                     &is_last_block_done,
                                                     smem_sum,
                                                     smem_square_sum,
                                                     block_data_ptr,
                                                     flag_ptr);

      if (is_last_block_done) {
        // final compute
        if (threadIdx.y == 0) {
          BatchNormParamType<T> compute_mean_val = x_sum / inner_size;
          BatchNormParamType<T> variance_val =
              x_square_sum / inner_size - compute_mean_val * compute_mean_val;
          BatchNormParamType<T> compute_inv_var_val =
              1 / sqrt(variance_val + epsilon);

          if (save_mean && save_inv_variance) {
            save_mean[i] = compute_mean_val;
            save_inv_variance[i] = compute_inv_var_val;
          }
          global_mean[i] = (1 - exponentialAverageFactor) * compute_mean_val +
                           exponentialAverageFactor * global_mean[i];
          global_variance[i] = (1 - exponentialAverageFactor) * variance_val +
                               exponentialAverageFactor * global_variance[i];

          compute_mean[i] = compute_mean_val;
          compute_inv_var[i] = compute_inv_var_val;
        }
      }
    } else {
      if (blockIdx.y == 0 && threadIdx.y == 0) {
        BatchNormParamType<T> compute_mean_val = x_sum / inner_size;
        BatchNormParamType<T> variance_val =
            x_square_sum / inner_size - compute_mean_val * compute_mean_val;
        BatchNormParamType<T> compute_inv_var_val =
            1 / sqrt(variance_val + epsilon);

        if (save_mean && save_inv_variance) {
          save_mean[i] = compute_mean_val;
          save_inv_variance[i] = compute_inv_var_val;
        }
        global_mean[i] = (1 - exponentialAverageFactor) * compute_mean_val +
                         exponentialAverageFactor * global_mean[i];
        global_variance[i] = (1 - exponentialAverageFactor) * variance_val +
                             exponentialAverageFactor * global_variance[i];

        compute_mean[i] = compute_mean_val;
        compute_inv_var[i] = compute_inv_var_val;
      }
    }
  }
}

template <typename T>
static __global__ void BNForwardTraining2DChannelLastWriteRes(
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    T *y,
    BatchNormParamType<T> *compute_mean,
    BatchNormParamType<T> *compute_inv_var) {
  int outer_size = C;
  int inner_size = N * HxW;

  int outer_loop_stride = gridDim.x * blockDim.x;
  int inner_loop_stride = gridDim.y * blockDim.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outer_size;
       i += outer_loop_stride) {
    BatchNormParamType<T> mean_val = compute_mean[i];
    BatchNormParamType<T> inv_var_val = compute_inv_var[i];
    BatchNormParamType<T> scale_val = scale[i];
    BatchNormParamType<T> bias_val = bias[i];

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < inner_size;
         j += inner_loop_stride) {
      const int index = j * outer_size + i;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[index]) - mean_val;
      y[index] = scale_val * x_sub_mean * inv_var_val + bias_val;
    }
  }
}

template <typename T, int BlockDim>
static __global__ void BNForwardTraining2DCompStat(
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    double exponentialAverageFactor,
    T *y,
    BatchNormParamType<T> *global_mean,
    BatchNormParamType<T> *global_variance,
    BatchNormParamType<T> *save_mean,
    BatchNormParamType<T> *save_inv_variance,
    BatchNormParamType<T> *compute_mean,
    BatchNormParamType<T> *compute_inv_var,
    BatchNormParamType<T> *block_data_ptr,
    int *flag_ptr) {
  int outer_size = C;
  int inner_size = N * HxW;

  __shared__ BatchNormParamType<T> smem_sum[BlockDim];
  __shared__ BatchNormParamType<T> smem_square_sum[BlockDim];

  int outer_loop_stride = gridDim.y * blockDim.y;
  int inner_loop_stride = gridDim.x * blockDim.x;

  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < outer_size;
       i += outer_loop_stride) {
    BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inner_size;
         j += inner_loop_stride) {
      const int index = (j / HxW * C + i) * HxW + j % HxW;
      BatchNormParamType<T> x_i = static_cast<BatchNormParamType<T>>(x[index]);
      x_sum += x_i;
      x_square_sum += x_i * x_i;
    }

    // horizontal block sum
    merge_block_horizontal<T>(x_sum,
                              x_square_sum,
                              &smem_sum[0],
                              &smem_square_sum[0],
                              &x_sum,
                              &x_square_sum);

    if (gridDim.x > 1) {
      volatile BatchNormParamType<T> *staging_sum = block_data_ptr;
      volatile BatchNormParamType<T> *staging_square_sum =
          &block_data_ptr[C * gridDim.x];
      // write block data to global memory
      if (threadIdx.x == 0) {
        staging_sum[i + blockIdx.x * C] = x_sum;
        staging_square_sum[i + blockIdx.x * C] = x_square_sum;
      }

      // make sure write is visible to all blocks
      __threadfence();
      __syncthreads();

      __shared__ bool is_last_block_done;
      // mark block done
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        int old = atomicAdd(&flag_ptr[blockIdx.y], 1);
        is_last_block_done = (old == (gridDim.x - 1));
      }

      __syncthreads();

      if (is_last_block_done) {
        x_sum = static_cast<BatchNormParamType<T>>(0);
        x_square_sum = static_cast<BatchNormParamType<T>>(0);
        // thread sum
        for (int x = threadIdx.x; x < gridDim.x; x += blockDim.x) {
          x_sum += staging_sum[i + x * C];
          x_square_sum += staging_square_sum[i + x * C];
        }

        // horizontal block sum
        merge_block_horizontal<T>(x_sum,
                                  x_square_sum,
                                  &smem_sum[0],
                                  &smem_square_sum[0],
                                  &x_sum,
                                  &x_square_sum);

        // final compute
        if (threadIdx.x == 0) {
          BatchNormParamType<T> compute_mean_val = x_sum / inner_size;
          BatchNormParamType<T> variance_val =
              x_square_sum / inner_size - compute_mean_val * compute_mean_val;
          BatchNormParamType<T> compute_inv_var_val =
              1 / sqrt(variance_val + epsilon);

          if (save_mean && save_inv_variance) {
            save_mean[i] = compute_mean_val;
            save_inv_variance[i] = compute_inv_var_val;
          }
          global_mean[i] = (1 - exponentialAverageFactor) * compute_mean_val +
                           exponentialAverageFactor * global_mean[i];
          global_variance[i] = (1 - exponentialAverageFactor) * variance_val +
                               exponentialAverageFactor * global_variance[i];

          compute_mean[i] = compute_mean_val;
          compute_inv_var[i] = compute_inv_var_val;
        }
      }
    } else {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        BatchNormParamType<T> compute_mean_val = x_sum / inner_size;
        BatchNormParamType<T> variance_val =
            x_square_sum / inner_size - compute_mean_val * compute_mean_val;
        BatchNormParamType<T> compute_inv_var_val =
            1 / sqrt(variance_val + epsilon);

        if (save_mean && save_inv_variance) {
          save_mean[i] = compute_mean_val;
          save_inv_variance[i] = compute_inv_var_val;
        }
        global_mean[i] = (1 - exponentialAverageFactor) * compute_mean_val +
                         exponentialAverageFactor * global_mean[i];
        global_variance[i] = (1 - exponentialAverageFactor) * variance_val +
                             exponentialAverageFactor * global_variance[i];

        compute_mean[i] = compute_mean_val;
        compute_inv_var[i] = compute_inv_var_val;
      }
    }
  }
}

template <typename T>
static __global__ void BNForwardTraining2DWriteRes(
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    T *y,
    BatchNormParamType<T> *compute_mean,
    BatchNormParamType<T> *compute_inv_var) {
  int outer_size = C;
  int inner_size = N * HxW;

  int outer_loop_stride = gridDim.y * blockDim.y;
  int inner_loop_stride = gridDim.x * blockDim.x;

  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < outer_size;
       i += outer_loop_stride) {
    BatchNormParamType<T> mean_val = compute_mean[i];
    BatchNormParamType<T> inv_var_val = compute_inv_var[i];
    BatchNormParamType<T> scale_val = scale[i];
    BatchNormParamType<T> bias_val = bias[i];

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inner_size;
         j += inner_loop_stride) {
      const int index = (j / HxW * C + i) * HxW + j % HxW;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[index]) - mean_val;
      y[index] = scale_val * x_sub_mean * inv_var_val + bias_val;
    }
  }
}

template <typename T, typename Context>
void BatchNormKernel(const Context &ctx,
                     const DenseTensor &x,
                     const DenseTensor &mean,
                     const DenseTensor &variance,
                     const paddle::optional<DenseTensor> &scale,
                     const paddle::optional<DenseTensor> &bias,
                     bool is_test,
                     float momentum,
                     float epsilon_f,
                     const std::string &data_layout_str,
                     bool use_global_stats,
                     bool trainable_statistics,
                     DenseTensor *y,
                     DenseTensor *mean_out,
                     DenseTensor *variance_out,
                     DenseTensor *saved_mean,
                     DenseTensor *saved_variance,
                     DenseTensor *reserve_space) {
  double epsilon = epsilon_f;
  const bool trainable_stats = trainable_statistics;
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  bool test_mode = is_test && (!trainable_stats);

  // Get the size for each dimension.
  // NCHW [batch_size, in_channels, in_height, in_width]
  const auto &x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      common::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  ctx.template Alloc<T>(y);
  int N, C, H, W, D;
  phi::funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

  auto *Scale = scale.get_ptr();
  auto *Bias = bias.get_ptr();

  phi::DenseTensor new_scale;
  phi::DenseTensor new_bias;

  if (Scale) {
    new_scale = scale.get();
  } else {
    new_scale = phi::Full<T, Context>(ctx, {C}, static_cast<T>(1));
  }

  if (Bias) {
    new_bias = bias.get();
  } else {
    new_bias = phi::Full<T, Context>(ctx, {C}, static_cast<T>(0));
  }

#ifdef PADDLE_WITH_HIP
  auto compute_format =
      data_layout == DataLayout::kNHWC
          ? (FLAGS_batch_norm_use_miopen == true ? DataLayout::kNCHW
                                                 : DataLayout::kNHWC)
          : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
  const bool fast_nhwc_batch_norm =
      test_mode ||
      (dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent);

  auto compute_format = fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
                            ? DataLayout::kNHWC
                            : DataLayout::kNCHW;
#endif

  DenseTensor transformed_x(x.type());
  DenseTensor transformed_y(y->type());

  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    TransToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    ResizeToChannelFirst<Context, T>(ctx, y, &transformed_y);
  } else {
    transformed_x.ShareDataWith(x);
    transformed_y.ShareDataWith(*y);
  }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
  // TODO(wangran16): wait for MIOpen to improve the performance of BN
  miopenTensorDescriptor_t data_desc_;
  miopenTensorDescriptor_t bn_param_desc_;
  miopenBatchNormMode_t mode_;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
  cudnnTensorDescriptor_t data_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;
  cudnnBatchNormMode_t mode_;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
#endif

  if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

#ifdef PADDLE_WITH_HIP
  // TODO(wangran16): wait for MIOpen to improve the performance of BN
  if (H == 1 && W == 1) {
    mode_ = miopenBNPerActivation;
  } else {
    mode_ = miopenBNSpatial;
  }
#elif CUDNN_VERSION_MIN(7, 0, 1)
  if (FLAGS_cudnn_batchnorm_spatial_persistent) {
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  } else if (H == 1 && W == 1) {
    mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode_ = CUDNN_BATCHNORM_SPATIAL;
  }
#else
  if (H == 1 && W == 1) {
    mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode_ = CUDNN_BATCHNORM_SPATIAL;
  }
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

  VLOG(3) << "Setting descriptors.";
  std::vector<int> dims;
  std::vector<int> strides;
  if (compute_format == DataLayout::kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * D * C, 1, W * D * C, D * C, C};
  }

#ifdef PADDLE_WITH_HIP
  // TODO(wangran16): wait for MIOpen to improve the performance of BN
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
      data_desc_,
      CudnnDataType<T>::type,
      x_dims.size() > 3 ? x_dims.size() : 4,
      const_cast<int *>(dims.data()),
      const_cast<int *>(strides.data())));
  // Note: PERSISTENT not implemented for inference
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDeriveBNTensorDescriptor(
      bn_param_desc_, data_desc_, mode_));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      data_desc_,
      CudnnDataType<T>::type,
      x_dims.size() > 3 ? x_dims.size() : 4,
      dims.data(),
      strides.data()));
  // Note: PERSISTENT not implemented for inference
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
      bn_param_desc_, data_desc_, test_mode ? CUDNN_BATCHNORM_SPATIAL : mode_));
#endif

  auto handle = ctx.cudnn_handle();

  // Now, depending on whether we are running test or not, we have two paths.
  // It is training mode when it's not reference AND not using pre-trained
  // model.
  bool training = !test_mode && !use_global_stats;
  if (!training) {
    // only when test we use input to do computation.
    const auto *est_mean = &mean;
    const auto *est_var = &variance;
    // Run inference mode.
    PADDLE_ENFORCE_EQ(
        est_mean->dims().size(),
        1UL,
        common::errors::InvalidArgument(
            "The size of mean's dimensions must equal to 1."
            "But received: the size of mean's dimensions mean is [%d],"
            "the dimensions of mean is [%s].",
            est_mean->dims().size(),
            est_mean->dims()));
    PADDLE_ENFORCE_EQ(
        est_var->dims().size(),
        1UL,
        common::errors::InvalidArgument(
            "The size of variance's dimensions must equal to 1."
            "But received: the size of variance's dimensions is [%d],"
            "the dimensions of variance is [%s].",
            est_var->dims().size(),
            est_var->dims()));
    PADDLE_ENFORCE_EQ(
        est_mean->dims()[0],
        C,
        common::errors::InvalidArgument(
            "The first dimension of mean must equal to the number of "
            "Channels, which is [%d]. But received: the first dimension"
            "of mean is [%d], the dimensions of mean is [%s].",
            C,
            est_mean->dims()[0],
            est_mean->dims()));
    PADDLE_ENFORCE_EQ(
        est_var->dims()[0],
        C,
        common::errors::InvalidArgument(
            "The first dimension of variance must equal to the number"
            "of Channels, which is [%d]. But received: the first dimension of"
            "variance is [%d], the dimensions of variance is [%s].",
            C,
            est_var->dims()[0],
            est_var->dims()));

#ifdef PADDLE_WITH_HIP
    const int block_size = 256;
    const int grid_size = (N * C * H * W * D + block_size - 1) / block_size;
    if (compute_format == DataLayout::kNCHW) {
      if (FLAGS_batch_norm_use_miopen == true) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::miopenBatchNormalizationForwardInference(
                handle,
                mode_,
                const_cast<void *>(
                    static_cast<const void *>(CudnnDataType<T>::kOne())),
                const_cast<void *>(
                    static_cast<const void *>(CudnnDataType<T>::kZero())),
                data_desc_,
                static_cast<const void *>(transformed_x.template data<T>()),
                data_desc_,
                static_cast<void *>(ctx.template Alloc<T>(&transformed_y)),
                bn_param_desc_,
                const_cast<void *>(static_cast<const void *>(
                    new_scale.template data<BatchNormParamType<T>>())),
                const_cast<void *>(static_cast<const void *>(
                    new_bias.template data<BatchNormParamType<T>>())),
                const_cast<void *>(static_cast<const void *>(
                    est_mean->template data<BatchNormParamType<T>>())),
                const_cast<void *>(static_cast<const void *>(
                    est_var->template data<BatchNormParamType<T>>())),
                epsilon));
      } else {
        BNForwardInference<T, DataLayout::kNCHW>
            <<<grid_size, block_size, 0, ctx.stream()>>>(
                transformed_x.template data<T>(),
                est_mean->template data<BatchNormParamType<T>>(),
                est_var->template data<BatchNormParamType<T>>(),
                new_scale.template data<BatchNormParamType<T>>(),
                new_bias.template data<BatchNormParamType<T>>(),
                C,
                N,
                H * W * D,
                epsilon,
                transformed_y.template data<T>());
      }
    } else {
      BNForwardInference<T, DataLayout::kNHWC>
          <<<grid_size, block_size, 0, ctx.stream()>>>(
              transformed_x.template data<T>(),
              est_mean->template data<BatchNormParamType<T>>(),
              est_var->template data<BatchNormParamType<T>>(),
              new_scale.template data<BatchNormParamType<T>>(),
              new_bias.template data<BatchNormParamType<T>>(),
              C,
              N,
              H * W * D,
              epsilon,
              transformed_y.template data<T>());
    }

#else
    const bool use_native_kernel =
        (x_dims.size() == 2 ||
         (x_dims.size() == 3 && N >= CUDNN_SPATIAL_THRESHOLD_EVAL));
    if (use_native_kernel) {
      const int block_size = 256;
      const int grid_size = (N * C * H * W * D + block_size - 1) / block_size;
      if (compute_format == DataLayout::kNCHW) {
        BNForwardInference<T, DataLayout::kNCHW>
            <<<grid_size, block_size, 0, ctx.stream()>>>(
                transformed_x.template data<T>(),
                est_mean->template data<BatchNormParamType<T>>(),
                est_var->template data<BatchNormParamType<T>>(),
                new_scale.template data<BatchNormParamType<T>>(),
                new_bias.template data<BatchNormParamType<T>>(),
                C,
                N,
                H * W * D,
                epsilon,
                transformed_y.template data<T>());
      } else {
        if (x_dims.size() == 2) {
          DenseTensor inv_var = phi::Empty<BatchNormParamType<T>>(ctx, {C});
          auto *inv_var_ptr = inv_var.data<BatchNormParamType<T>>();
          const int threads = 512 > C ? C : 512;
          const int blocks = (C + 511) / 512;
          InverseVariance<T><<<blocks, threads, 0, ctx.stream()>>>(
              est_var->template data<BatchNormParamType<T>>(),
              epsilon,
              C,
              inv_var_ptr);
          BN1DForwardInference<T, DataLayout::kNHWC>
              <<<grid_size, block_size, 0, ctx.stream()>>>(
                  transformed_x.template data<T>(),
                  est_mean->template data<BatchNormParamType<T>>(),
                  // est_var->template data<BatchNormParamType<T>>(),
                  inv_var_ptr,
                  new_scale.template data<BatchNormParamType<T>>(),
                  new_bias.template data<BatchNormParamType<T>>(),
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  transformed_y.template data<T>());
        } else {
          BNForwardInference<T, DataLayout::kNHWC>
              <<<grid_size, block_size, 0, ctx.stream()>>>(
                  transformed_x.template data<T>(),
                  est_mean->template data<BatchNormParamType<T>>(),
                  est_var->template data<BatchNormParamType<T>>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  new_bias.template data<BatchNormParamType<T>>(),
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  transformed_y.template data<T>());
        }
      }
    } else {
      int64_t reserve_space_size = 0;
      if (reserve_space == nullptr) {
        reserve_space = new DenseTensor();
      }
      reserve_space->Resize({reserve_space_size});
      ctx.template Alloc<T>(reserve_space);

      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnBatchNormalizationForwardInference(
              handle,
              // Note: PERSISTENT not implemented for inference
              CUDNN_BATCHNORM_SPATIAL,
              CudnnDataType<T>::kOne(),
              CudnnDataType<T>::kZero(),
              data_desc_,
              transformed_x.template data<T>(),
              data_desc_,
              ctx.template Alloc<T>(&transformed_y),
              bn_param_desc_,
              new_scale.template data<BatchNormParamType<T>>(),
              new_bias.template data<BatchNormParamType<T>>(),
              est_mean->template data<BatchNormParamType<T>>(),
              est_var->template data<BatchNormParamType<T>>(),
              epsilon));
    }
#endif
  } else {
    // if MomentumTensor is set, use MomentumTensor value, momentum
    // is only used in this training branch

    // need to solve here
    // if (ctx.HasInput("MomentumTensor")) {
    //   const auto *mom_tensor = MomentumTensor;
    //   DenseTensor mom_cpu;
    //   paddle::framework::TensorCopySync(*mom_tensor, phi::CPUPlace(),
    //                                     &mom_cpu);
    //   momentum = mom_cpu.data<float>()[0];
    // }

    // Run training mode.
    // obtain running mean and running inv var, and there is no need
    // to initialize them.
    ctx.template Alloc<BatchNormParamType<T>>(mean_out);
    ctx.template Alloc<BatchNormParamType<T>>(variance_out);

    ctx.template Alloc<BatchNormParamType<T>>(saved_mean);
    ctx.template Alloc<BatchNormParamType<T>>(saved_variance);

    if ((N * H * W * D) == 1) {
      int64_t reserve_space_size = 0;
      if (reserve_space == nullptr) {
        reserve_space = new DenseTensor();
      }
      reserve_space->Resize({reserve_space_size});
      ctx.template Alloc<T>(reserve_space);
      // Only 1 element in normalization dimension,
      // skip the batch norm calculation, let y = x.
      phi::Copy(ctx, x, ctx.GetPlace(), false, y);
    } else {
      double this_factor = 1. - momentum;
#ifdef PADDLE_WITH_HIP
      this_factor = momentum;
      const int num = transformed_x.numel();
      const int block = 256;
      const int max_threads = ctx.GetMaxPhysicalThreadCount();
      const int max_blocks = std::max(max_threads / block, 1);
      const int grid = std::min(C, max_blocks);
      if (compute_format == DataLayout::kNCHW) {
        if (FLAGS_batch_norm_use_miopen == true) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::miopenBatchNormalizationForwardTraining(
                  handle,
                  mode_,
                  const_cast<void *>(
                      static_cast<const void *>(CudnnDataType<T>::kOne())),
                  const_cast<void *>(
                      static_cast<const void *>(CudnnDataType<T>::kZero())),
                  data_desc_,
                  static_cast<const void *>(transformed_x.template data<T>()),
                  data_desc_,
                  static_cast<void *>(ctx.template Alloc<T>(&transformed_y)),
                  bn_param_desc_,
                  const_cast<void *>(static_cast<const void *>(
                      new_scale.template data<BatchNormParamType<T>>())),
                  const_cast<void *>(static_cast<const void *>(
                      new_bias.template data<BatchNormParamType<T>>())),
                  this_factor,
                  static_cast<void *>(
                      ctx.template Alloc<BatchNormParamType<T>>(mean_out)),
                  static_cast<void *>(
                      ctx.template Alloc<BatchNormParamType<T>>(variance_out)),
                  epsilon,
                  static_cast<void *>(
                      ctx.template Alloc<BatchNormParamType<T>>(saved_mean)),
                  static_cast<void *>(ctx.template Alloc<BatchNormParamType<T>>(
                      saved_variance))));
        } else {
          BNForwardTraining<T, block, DataLayout::kNCHW>
              <<<grid, block, 0, ctx.stream()>>>(
                  transformed_x.template data<T>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  new_bias.template data<BatchNormParamType<T>>(),
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  this_factor,
                  transformed_y.template data<T>(),
                  mean_out->template data<BatchNormParamType<T>>(),
                  variance_out->template data<BatchNormParamType<T>>(),
                  saved_mean->template data<BatchNormParamType<T>>(),
                  saved_variance->template data<BatchNormParamType<T>>());
        }
      } else {
        BNForwardTraining<T, block, DataLayout::kNHWC>
            <<<grid, block, 0, ctx.stream()>>>(
                transformed_x.template data<T>(),
                new_scale.template data<BatchNormParamType<T>>(),
                new_bias.template data<BatchNormParamType<T>>(),
                C,
                N,
                H * W * D,
                epsilon,
                this_factor,
                transformed_y.template data<T>(),
                mean_out->template data<BatchNormParamType<T>>(),
                variance_out->template data<BatchNormParamType<T>>(),
                saved_mean->template data<BatchNormParamType<T>>(),
                saved_variance->template data<BatchNormParamType<T>>());
      }

#else
      // const size_t CUDNN_PER_ACTIVATION_THRESHOLD = 131070;
      const bool use_native_kernel =
          ((x_dims.size() == 2 && N >= CUDNN_PER_ACTIVATION_THRESHOLD) ||
           (x_dims.size() == 3 && N >= CUDNN_SPATIAL_THRESHOLD_TRAIN));
      if (use_native_kernel) {
        double this_factor = momentum;
        dim3 block;
        dim3 grid;
        const int block_size = 512;
        const int MAX_GRID_SIZE = 128;
        const int WARP_SIZE = 32;

        // init intermediate storage
        DenseTensor block_data_tensor;
        DenseTensor flag_tensor;
        DenseTensor compute_mean_tensor =
            phi::Empty<BatchNormParamType<T>, Context>(ctx, {C});
        DenseTensor compute_inv_var_tensor =
            phi::Empty<BatchNormParamType<T>, Context>(ctx, {C});

        BatchNormParamType<T> *block_data_ptr = nullptr;
        int *flag_ptr = nullptr;

        if (x_dims.size() != 2 && compute_format == DataLayout::kNCHW) {
          // init block&grid config
          int block_x =
              std::min(phi::funcs::details::GetLastPow2(H * W * D), block_size);
          int block_y = std::min(phi::funcs::details::GetLastPow2(C),
                                 block_size / block_x);

          if (block_x * block_y != block_size) {
            block_x =
                std::min(phi::funcs::details::GetLastPow2(N * H * W * D / 16),
                         block_size / block_y);
          }

          int grid_x =
              std::min((N * H * W * D + block_x * 16 - 1) / (block_x * 16),
                       MAX_GRID_SIZE);
          int grid_y = (C + block_y - 1) / block_y;

          block.x = block_x;
          block.y = block_y;
          grid.x = grid_x;
          grid.y = grid_y;

          if (grid.x > 1) {
            block_data_tensor = phi::Empty<BatchNormParamType<T>, Context>(
                ctx, {2 * C * grid.x});
            flag_tensor = phi::Empty<int, Context>(ctx, {grid.y});

            block_data_ptr = block_data_tensor.data<BatchNormParamType<T>>();
            flag_ptr = flag_tensor.data<int>();
            funcs::SetConstant<Context, int> set_zero;
            set_zero(ctx, &flag_tensor, static_cast<int>(0));
          }
          BNForwardTraining2DCompStat<T, block_size>
              <<<grid, block, 0, ctx.stream()>>>(
                  transformed_x.template data<T>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  new_bias.template data<BatchNormParamType<T>>(),
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  this_factor,
                  transformed_y.template data<T>(),
                  mean_out->template data<BatchNormParamType<T>>(),
                  variance_out->template data<BatchNormParamType<T>>(),
                  saved_mean->template data<BatchNormParamType<T>>(),
                  saved_variance->template data<BatchNormParamType<T>>(),
                  compute_mean_tensor.data<BatchNormParamType<T>>(),
                  compute_inv_var_tensor.data<BatchNormParamType<T>>(),
                  block_data_ptr,
                  flag_ptr);

          BNForwardTraining2DWriteRes<T><<<grid, block, 0, ctx.stream()>>>(
              transformed_x.template data<T>(),
              new_scale.template data<BatchNormParamType<T>>(),
              new_bias.template data<BatchNormParamType<T>>(),
              C,
              N,
              H * W * D,
              transformed_y.template data<T>(),
              compute_mean_tensor.data<BatchNormParamType<T>>(),
              compute_inv_var_tensor.data<BatchNormParamType<T>>());
        } else {
          // init block&grid config
          int block_x =
              std::min(phi::funcs::details::GetLastPow2(C), WARP_SIZE);
          int block_y =
              std::min(phi::funcs::details::GetLastPow2(N * H * W * D / 16),
                       block_size / block_x);
          if (block_x * block_y != block_size) {
            block_x = std::min(phi::funcs::details::GetLastPow2(C),
                               block_size / block_y);
          }
          int grid_x = (C + block_x - 1) / block_x;
          int grid_y =
              std::min((N * H * W * D + block_y * 16 - 1) / (block_y * 16),
                       MAX_GRID_SIZE);

          block.x = block_x;
          block.y = block_y;
          grid.x = grid_x;
          grid.y = grid_y;

          if (grid.y > 1) {
            block_data_tensor = phi::Empty<BatchNormParamType<T>, Context>(
                ctx, {2 * C * grid.y});
            flag_tensor = phi::Empty<int, Context>(ctx, {grid.x});

            block_data_ptr = block_data_tensor.data<BatchNormParamType<T>>();
            flag_ptr = flag_tensor.data<int>();
            funcs::SetConstant<Context, int> set_zero;
            set_zero(ctx, &flag_tensor, static_cast<int>(0));
          }
          BNForwardTraining2DChannelLastCompStat<T, block_size>
              <<<grid, block, 0, ctx.stream()>>>(
                  transformed_x.template data<T>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  new_bias.template data<BatchNormParamType<T>>(),
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  this_factor,
                  transformed_y.template data<T>(),
                  mean_out->template data<BatchNormParamType<T>>(),
                  variance_out->template data<BatchNormParamType<T>>(),
                  saved_mean->template data<BatchNormParamType<T>>(),
                  saved_variance->template data<BatchNormParamType<T>>(),
                  compute_mean_tensor.data<BatchNormParamType<T>>(),
                  compute_inv_var_tensor.data<BatchNormParamType<T>>(),
                  block_data_ptr,
                  flag_ptr);

          BNForwardTraining2DChannelLastWriteRes<T>
              <<<grid, block, 0, ctx.stream()>>>(
                  transformed_x.template data<T>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  new_bias.template data<BatchNormParamType<T>>(),
                  C,
                  N,
                  H * W * D,
                  transformed_y.template data<T>(),
                  compute_mean_tensor.data<BatchNormParamType<T>>(),
                  compute_inv_var_tensor.data<BatchNormParamType<T>>());
        }
      } else {
#if CUDNN_VERSION_MIN(7, 4, 1)
        size_t workspace_size = 0;
        size_t reserve_space_size = 0;
        void *reserve_space_ptr = nullptr;
        void *workspace_ptr = nullptr;
        DenseTensor workspace_tensor;
        // Create reserve space and workspace for batch norm.
        // Create tensor for each batchnorm op, it will be used in the
        // backward. Thus this tensor shouldn't be temp.
        // auto *reserve_space = ctx.Output<phi::DenseTensor>("ReserveSpace");
        if (reserve_space == nullptr) {
          reserve_space = new DenseTensor();
        }
        PADDLE_ENFORCE_NOT_NULL(
            reserve_space,
            common::errors::NotFound(
                "The argument ReserveSpace of batch_norm op is not found."));
        // --------------- cudnn batchnorm workspace ---------------
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::
                cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                    /*handle=*/handle,
                    /*mode=*/mode_,
                    /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*xDesc=*/data_desc_,
                    /*zDesc=*/nullptr,
                    /*yDesc=*/data_desc_,
                    /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                    /*activationDesc=*/nullptr,
                    /*sizeInBytes=*/&workspace_size));

        // -------------- cudnn batchnorm reserve space --------------
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                /*handle=*/handle,
                /*mode=*/mode_,
                /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                /*activationDesc=*/nullptr,
                /*xDesc=*/data_desc_,
                /*sizeInBytes=*/&reserve_space_size));

        reserve_space->Resize({static_cast<int64_t>(reserve_space_size)});
        reserve_space_ptr =
            static_cast<void *>(ctx.template Alloc<uint8_t>(reserve_space));
        workspace_tensor.Resize({static_cast<int64_t>(workspace_size)});
        workspace_ptr =
            static_cast<void *>(ctx.template Alloc<uint8_t>(&workspace_tensor));
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnBatchNormalizationForwardTrainingEx(
                handle,
                mode_,
                CUDNN_BATCHNORM_OPS_BN,
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                data_desc_,
                transformed_x.template data<T>(),
                nullptr,
                nullptr,
                data_desc_,
                transformed_y.template data<T>(),
                bn_param_desc_,
                new_scale.template data<BatchNormParamType<T>>(),
                new_bias.template data<BatchNormParamType<T>>(),
                this_factor,
                ctx.template Alloc<BatchNormParamType<T>>(mean_out),
                ctx.template Alloc<BatchNormParamType<T>>(variance_out),
                epsilon,
                ctx.template Alloc<BatchNormParamType<T>>(saved_mean),
                ctx.template Alloc<BatchNormParamType<T>>(saved_variance),
                nullptr,
                workspace_ptr,
                workspace_size,
                reserve_space_ptr,
                reserve_space_size));
#else
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnBatchNormalizationForwardTraining(
                handle,
                mode_,
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                data_desc_,
                transformed_x.template data<T>(),
                data_desc_,
                ctx.template Alloc<T>(&transformed_y),
                bn_param_desc_,
                new_scale.template data<BatchNormParamType<T>>(),
                new_bias.template data<BatchNormParamType<T>>(),
                this_factor,
                ctx.template Alloc<BatchNormParamType<T>>(mean_out),
                ctx.template Alloc<BatchNormParamType<T>>(variance_out),
                epsilon,
                ctx.template Alloc<BatchNormParamType<T>>(saved_mean),
                ctx.template Alloc<BatchNormParamType<T>>(saved_variance)));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
      }
#endif
    }
  }

  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
    TransToChannelLast<Context, T>(ctx, &transformed_y, y);
  }
#ifdef PADDLE_WITH_HIP
  // TODO(wangran16): wait for MIOpen to improve the performance of BN
  // clean when exit.
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
  // clean when exit.
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(data_desc_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#endif
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
#if CUDNN_VERSION_MIN(7, 4, 1)
  kernel->OutputAt(5).SetDataType(phi::DataType::UINT8);
#endif
}
#else
PD_REGISTER_KERNEL(batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
#if CUDNN_VERSION_MIN(7, 4, 1)
  kernel->OutputAt(5).SetDataType(phi::DataType::UINT8);
#endif
}
#endif

#endif
