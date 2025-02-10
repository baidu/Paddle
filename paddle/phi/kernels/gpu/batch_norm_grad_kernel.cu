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

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
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

template <typename T, int BlockDim, phi::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void KeBNBackwardScaleBias(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance,
    const double epsilon,
    const int N,
    const int C,
    const int HxW,
    BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    BatchNormParamType<T> inv_var_i = 1.0 / sqrt(variance[i] + epsilon);
    BatchNormParamType<T> mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += static_cast<BatchNormParamType<T>>(dy[index]) *
                (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
      db_sum += static_cast<BatchNormParamType<T>>(dy[index]);
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum * inv_var_i;
      dbias[i] = db_sum;
    }
    __syncthreads();
  }
}

template <typename T, phi::DataLayout layout>
static __global__ void KeBNBackwardData(const T *dy,
                                        const BatchNormParamType<T> *scale,
                                        const BatchNormParamType<T> *variance,
                                        const double epsilon,
                                        const int C,
                                        const int HxW,
                                        const int num,
                                        T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> inv_var = 1.0 / sqrt(variance[c] + epsilon);
    dx[i] = static_cast<T>(static_cast<BatchNormParamType<T>>(dy[i]) *
                           scale[c] * inv_var);
  }
}

template <typename T>
static __global__ void KeBNRestoreData(const phi::DataLayout layout,
                                       T *x,
                                       const BatchNormParamType<T> *scale,
                                       const BatchNormParamType<T> *bias,
                                       const BatchNormParamType<T> *mean,
                                       const BatchNormParamType<T> *variance,
                                       double epsilon,
                                       int C,
                                       int M,
                                       const int num,
                                       const T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::kNCHW ? (i / M) % C : i % C;
    auto y_i = static_cast<BatchNormParamType<T>>(y[i]);
    auto x_i = (y_i - bias[c]) / scale[c] / variance[c] + mean[c];
    x[i] = static_cast<T>(x_i);
  }
}

template <typename T>
class InplaceHelper {
 public:
  void operator()(const phi::DataLayout layout,
                  T *x,
                  const BatchNormParamType<T> *scale,
                  const BatchNormParamType<T> *bias,
                  const BatchNormParamType<T> *mean,
                  const BatchNormParamType<T> *variance,
                  double epsilon,
                  int C,
                  int M,
                  const int num,
                  const T *y,
                  int grid2,
                  const int block,
                  const gpuStream_t &stream) {
    PADDLE_ENFORCE_EQ(x,
                      y,
                      common::errors::InvalidArgument(
                          "X and Y should be inplaced in inplace mode"));
    KeBNRestoreData<<<grid2, block, 0, stream>>>(
        layout, x, scale, bias, mean, variance, epsilon, C, M, num, y);
  }
};

template <typename T, int BlockDim, phi::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackward(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *saved_mean,
    const BatchNormParamType<T> *saved_inv_variance,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    T *dx,
    BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storage;
  __shared__ BatchNormParamType<T> inv_var_val;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> dscale_val;
  __shared__ BatchNormParamType<T> dbias_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    if (saved_mean && saved_inv_variance) {
      if (threadIdx.x == 0) {
        inv_var_val = saved_inv_variance[i];
        mean_val = saved_mean[i];
      }
    } else {
      BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
      BatchNormParamType<T> x_square_sum =
          static_cast<BatchNormParamType<T>>(0);

      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index = layout == phi::DataLayout::kNCHW
                              ? (j / HxW * C + i) * HxW + j % HxW
                              : j * outer_size + i;
        BatchNormParamType<T> x_i =
            static_cast<BatchNormParamType<T>>(x[index]);
        x_sum += x_i;
        x_square_sum += x_i * x_i;
      }

      x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
      x_square_sum =
          BlockReduce(variance_storage).Reduce(x_square_sum, cub::Sum());
      if (threadIdx.x == 0) {
        mean_val = x_sum / inner_size;
        inv_var_val =
            1 / sqrt(x_square_sum / inner_size - mean_val * mean_val + epsilon);
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      ds_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_val);
      db_sum += dy_i;
    }

    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale_val = ds_sum * inv_var_val;
      dbias_val = db_sum;
      dscale[i] = dscale_val;
      dbias[i] = dbias_val;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] = scale[i] * inv_var_val *
                  (static_cast<BatchNormParamType<T>>(dy[index]) -
                   dbias_val / static_cast<BatchNormParamType<T>>(inner_size) -
                   (static_cast<BatchNormParamType<T>>(x[index]) - mean_val) *
                       inv_var_val * dscale_val / inner_size);
    }
  }
}

template <typename T, int BlockDim>
static __global__ void BNBackward2DChannelLastStage1(
    const T *x,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    BatchNormParamType<T> *block_data_ptr,
    BatchNormParamType<T> *compute_mean,
    BatchNormParamType<T> *compute_inv_var,
    int *flag_ptr) {
  int outer_size = C;
  int inner_size = N * HxW;

  __shared__ BatchNormParamType<T> smem_sum[BlockDim];
  __shared__ BatchNormParamType<T> smem_square_sum[BlockDim];
  __shared__ BatchNormParamType<T> inv_var_val;
  __shared__ BatchNormParamType<T> mean_val;

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

          compute_mean[i] = compute_mean_val;
          compute_inv_var[i] = compute_inv_var_val;
        }
      }
    }
  }
}

template <typename T, int BlockDim>
static __global__ void BNBackward2DChannelLastStage2(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *means,
    const BatchNormParamType<T> *variances,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    const bool is_test,
    BatchNormParamType<T> *block_data_ptr,
    BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias,
    int *flag_ptr) {
  int outer_size = C;
  int inner_size = N * HxW;

  __shared__ BatchNormParamType<T> smem_ds_sum[BlockDim];
  __shared__ BatchNormParamType<T> smem_db_sum[BlockDim];
  __shared__ BatchNormParamType<T> inv_var_val;
  __shared__ BatchNormParamType<T> mean_val;

  int outer_loop_stride = gridDim.x * blockDim.x;
  int inner_loop_stride = gridDim.y * blockDim.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outer_size;
       i += outer_loop_stride) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> mean_val = means[i];
    BatchNormParamType<T> inv_var_val =
        is_test ? 1.0 / sqrt(variances[i] + epsilon) : variances[i];

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < inner_size;
         j += inner_loop_stride) {
      const int index = j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      ds_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_val);
      db_sum += dy_i;
    }

    // vertical block sum
    funcs::BlockReduceByVertical<T, BatchNormParamType<T>>(
        ds_sum, db_sum, &smem_ds_sum[0], &smem_db_sum[0], &ds_sum, &db_sum);

    if (gridDim.y > 1) {
      __shared__ bool is_last_block_done;
      funcs::ReduceSumPost<T, BatchNormParamType<T>>(C,
                                                     i,
                                                     &ds_sum,
                                                     &db_sum,
                                                     &is_last_block_done,
                                                     smem_ds_sum,
                                                     smem_db_sum,
                                                     block_data_ptr,
                                                     flag_ptr);
      if (is_last_block_done) {
        // final compute
        if (threadIdx.y == 0) {
          dscale[i] = ds_sum * inv_var_val;
          dbias[i] = db_sum;
        }
      }
    }
  }
}

template <typename T, int BlockDim>
static __global__ void BNBackward2DChannelLastStage3(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *dscales,
    const BatchNormParamType<T> *dbias,
    const BatchNormParamType<T> *means,
    const BatchNormParamType<T> *variances,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    T *dx) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  int outer_loop_stride = gridDim.x * blockDim.x;
  int inner_loop_stride = gridDim.y * blockDim.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < outer_size;
       i += outer_loop_stride) {
    BatchNormParamType<T> mean_val = means[i];
    BatchNormParamType<T> inv_var_val = variances[i];
    BatchNormParamType<T> dscale_val = dscales[i];
    BatchNormParamType<T> dbias_val = dbias[i];

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < inner_size;
         j += inner_loop_stride) {
      const int index = j * outer_size + i;
      dx[index] = scale[i] * inv_var_val *
                  (static_cast<BatchNormParamType<T>>(dy[index]) -
                   dbias_val / static_cast<BatchNormParamType<T>>(inner_size) -
                   (static_cast<BatchNormParamType<T>>(x[index]) - mean_val) *
                       inv_var_val * dscale_val / inner_size);
    }
  }
}

template <typename T, int BlockDim, phi::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackwardData(
    const T *dy,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *mean,
    const T *x,
    const BatchNormParamType<T> *variance,
    const int C,
    const int N,
    const int HxW,
    T *dx) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_x_sub_mean_storage;
  __shared__ BatchNormParamType<T> dy_sum_val;
  __shared__ BatchNormParamType<T> dy_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> inv_var_i = variance[i];
    BatchNormParamType<T> mean_i = mean[i];
    BatchNormParamType<T> dy_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> dy_x_sub_mean_sum =
        static_cast<BatchNormParamType<T>>(0);
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      dy_sum += dy_i;
      dy_x_sub_mean_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
    }

    dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
    dy_x_sub_mean_sum = BlockReduce(dy_x_sub_mean_storage)
                            .Reduce(dy_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      dy_sum_val = dy_sum;
      dy_x_sub_mean_sum_val = dy_x_sub_mean_sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] =
          (static_cast<BatchNormParamType<T>>(dy[index]) -
           dy_sum_val / static_cast<BatchNormParamType<T>>(inner_size) -
           (static_cast<BatchNormParamType<T>>(x[index]) - mean_i) *
               dy_x_sub_mean_sum_val * inv_var_i * inv_var_i / inner_size) *
          scale[i] * inv_var_i;
    }
  }
}

template <typename T, typename Context>
void BatchNormGradFunctor(const Context &ctx,
                          const DenseTensor &x,
                          const paddle::optional<DenseTensor> &scale,
                          const paddle::optional<DenseTensor> &bias,
                          const paddle::optional<DenseTensor> &mean,
                          const paddle::optional<DenseTensor> &variance,
                          const DenseTensor &saved_mean,
                          const DenseTensor &saved_variance,
                          const paddle::optional<DenseTensor> &reserve_space,
                          const DenseTensor &y_grad,
                          float momentum,
                          float epsilon_f,
                          const std::string &data_layout_str,
                          bool is_test,
                          bool use_global_stats,
                          bool trainable_statistics,
                          bool is_inplace,
                          DenseTensor *x_grad,
                          DenseTensor *scale_grad,
                          DenseTensor *bias_grad) {
  double epsilon = static_cast<double>(epsilon_f);

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  const auto *d_y = &y_grad;

  auto *d_x = x_grad;
  auto *d_scale = scale_grad;
  auto *d_bias = bias_grad;

  use_global_stats = is_test || use_global_stats;

  const auto &x_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      common::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5."
          "But received: the size of input's dimensions is [%d],"
          "the dimensions of input is [%s]",
          x_dims.size(),
          x_dims));

  PADDLE_ENFORCE_EQ((d_scale == nullptr && d_bias == nullptr) ||
                        (d_scale != nullptr && d_bias != nullptr),
                    true,
                    common::errors::InvalidArgument(
                        "Weight and bias's stop_gradient of BatchNorm must be "
                        "True or False at the same time."));

  int N, C, H, W, D;
  phi::funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  // init output
  if (d_x) {
    ctx.template Alloc<T>(d_x);
  }

  if (d_scale && d_bias) {
    ctx.template Alloc<BatchNormParamType<T>>(d_scale);
    ctx.template Alloc<BatchNormParamType<T>>(d_bias);
  }

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

  PADDLE_ENFORCE_EQ(
      new_scale.dims().size(),
      1UL,
      common::errors::InvalidArgument(
          "The size of scale's dimensions must equal to 1. But received: "
          "the size of scale's dimensions is [%d], the dimensions of scale "
          "is [%s].",
          new_scale.dims().size(),
          new_scale.dims()));
  PADDLE_ENFORCE_EQ(
      new_scale.dims()[0],
      C,
      common::errors::InvalidArgument(
          "The first dimension of scale must equal to Channels[%d]. But "
          "received: the first dimension of scale is [%d]",
          C,
          new_scale.dims()[0]));

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;
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
  const bool fast_nhwc_batch_norm = dtype == CUDNN_DATA_HALF &&
                                    FLAGS_cudnn_batchnorm_spatial_persistent &&
                                    (reserve_space.get_ptr() != nullptr);
  auto compute_format = fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
                            ? DataLayout::kNHWC
                            : DataLayout::kNCHW;
#endif

  DenseTensor transformed_x(x.type());
  DenseTensor transformed_d_y(d_y->type());
  DenseTensor transformed_d_x;
  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    TransToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    ResizeToChannelFirst<Context, T>(ctx, d_y, &transformed_d_y);
    TransToChannelFirst<Context, T>(ctx, d_y, &transformed_d_y);
    if (d_x) {
      ResizeToChannelFirst<Context, T>(ctx, d_x, &transformed_d_x);
    }
  } else {
    transformed_x.ShareDataWith(x);
    transformed_d_y.ShareDataWith(*d_y);
    if (d_x) {
      transformed_d_x.ShareDataWith(*d_x);
    }
  }

  std::vector<int> dims;
  std::vector<int> strides;
  if (compute_format == DataLayout::kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * C * D, 1, W * D * C, D * C, C};
  }

  const int num = transformed_x.numel();
#ifdef HIPCC
  const int block = 256;
#else
  const int block = 512;
#endif
  int max_threads = ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid1 = (num + block - 1) / block;
  int grid2 = std::min(C, max_blocks);
  auto stream = ctx.stream();
  InplaceHelper<T> inplace_functor;

  if (!use_global_stats) {
    if ((N * H * W * D) == 1) {
      if (d_x) {
        phi::Copy(ctx, *d_y, ctx.GetPlace(), false, d_x);
      }
      phi::funcs::SetConstant<Context, BatchNormParamType<T>> functor;
      functor(ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
      functor(ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
      return;
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

#ifdef PADDLE_WITH_HIP
    // TODO(wangran16): wait for MIOpen to improve the performance of BN
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        data_desc_,
        CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4,
        const_cast<int *>(dims.data()),
        const_cast<int *>(strides.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        data_desc_,
        CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4,
        dims.data(),
        strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));
#endif

    const auto *saved_mean_data =
        saved_mean.template data<BatchNormParamType<T>>();
    const auto *saved_var_data =
        saved_variance.template data<BatchNormParamType<T>>();

    if (is_inplace) {
      inplace_functor(compute_format,
                      transformed_x.data<T>(),
                      new_scale.template data<BatchNormParamType<T>>(),
                      new_bias.template data<BatchNormParamType<T>>(),
                      saved_mean_data,
                      saved_var_data,
                      epsilon,
                      C,
                      H * W * D,
                      num,
                      transformed_x.data<T>(),
                      grid2,
                      block,
                      stream);
    }

    // This branch calls CUDNN APIs
    if (d_x && d_scale && d_bias) {
#ifdef PADDLE_WITH_HIP
      if (compute_format == DataLayout::kNCHW) {
        if (FLAGS_batch_norm_use_miopen == true) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::miopenBatchNormalizationBackward(
                  ctx.cudnn_handle(),
                  mode_,
                  CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(),
                  CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(),
                  data_desc_,
                  transformed_x.template data<T>(),
                  data_desc_,
                  transformed_d_y.template data<T>(),
                  data_desc_,
                  ctx.template Alloc<T>(&transformed_d_x),
                  bn_param_desc_,
                  new_scale.template data<BatchNormParamType<T>>(),
                  ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                  ctx.template Alloc<BatchNormParamType<T>>(d_bias),
                  epsilon,
                  saved_mean_data,
                  saved_var_data));
        } else {
          BNBackward<T, block, DataLayout::kNCHW>
              <<<grid2, block, 0, ctx.stream()>>>(
                  transformed_d_y.template data<T>(),
                  transformed_x.template data<T>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  saved_mean_data,
                  saved_var_data,
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  transformed_d_x.template data<T>(),
                  ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                  ctx.template Alloc<BatchNormParamType<T>>(d_bias));
        }
      } else {
        BNBackward<T, block, DataLayout::kNHWC>
            <<<grid2, block, 0, ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                new_scale.template data<BatchNormParamType<T>>(),
                saved_mean_data,
                saved_var_data,
                C,
                N,
                H * W * D,
                epsilon,
                transformed_d_x.template data<T>(),
                ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                ctx.template Alloc<BatchNormParamType<T>>(d_bias));
      }

#else
    }
    // CUDNN only support small batch size
    bool use_native_nhwc =
        d_x ? (x_dims.size() == 4 && compute_format == DataLayout::kNHWC &&
               H * W >= CUDNN_SPATIAL_THRESHOLD_EVAL)
            : false;
    const bool use_native_kernel =
        ((x_dims.size() == 2 && N >= CUDNN_PER_ACTIVATION_THRESHOLD) ||
         (x_dims.size() == 3 && N >= CUDNN_SPATIAL_THRESHOLD_TRAIN));
    if (use_native_nhwc || (d_x && d_scale && d_bias)) {
      if (use_native_kernel || use_native_nhwc) {
        if (x_dims.size() == 2 || use_native_nhwc) {
          dim3 block;
          dim3 grid;
          const int block_size = 512;

          // init intermediate storage
          DenseTensor block_data_tensor;
          DenseTensor flag_tensor;
          DenseTensor compute_mean_tensor =
              phi::Empty<BatchNormParamType<T>, Context>(ctx, {C});
          DenseTensor compute_inv_var_tensor =
              phi::Empty<BatchNormParamType<T>, Context>(ctx, {C});

          BatchNormParamType<T> *block_data_ptr = nullptr;
          int *flag_ptr = nullptr;

          funcs::SetLaunchConfigInfoForChannelLast<T, BatchNormParamType<T>>(
              ctx,
              &block_data_tensor,
              &flag_tensor,
              &block_data_ptr,
              &flag_ptr,
              N,
              H,
              W,
              D,
              C,
              block_size,
              &block,
              &grid);

          // 1. reduce_sum(x) => mean, inv_var
          auto *mean_ptr =
              saved_mean_data == nullptr
                  ? compute_mean_tensor.data<BatchNormParamType<T>>()
                  : saved_mean_data;
          auto *variance_ptr =
              saved_var_data == nullptr
                  ? compute_inv_var_tensor.data<BatchNormParamType<T>>()
                  : saved_var_data;

          if (saved_mean_data == nullptr) {
            BNBackward2DChannelLastStage1<T, block_size>
                <<<grid, block, 0, ctx.stream()>>>(
                    transformed_x.template data<T>(),
                    C,
                    N,
                    H * W * D,
                    epsilon,
                    block_data_ptr,
                    compute_mean_tensor.data<BatchNormParamType<T>>(),
                    compute_inv_var_tensor.data<BatchNormParamType<T>>(),
                    flag_ptr);
          }
          // 2. reduce_sum(x, dy, mean) => dscale, dbias
          BatchNormParamType<T> *dscale = nullptr;
          BatchNormParamType<T> *dbias = nullptr;
          bool with_scale = false;
          if (d_scale && d_bias) {
            dscale = ctx.template Alloc<BatchNormParamType<T>>(d_scale);
            dbias = ctx.template Alloc<BatchNormParamType<T>>(d_bias);
          } else {
            DenseTensor dscale_mem =
                phi::Empty<BatchNormParamType<T>, Context>(ctx, {C});
            DenseTensor dbias_mem =
                phi::Empty<BatchNormParamType<T>, Context>(ctx, {C});
            dscale = dscale_mem.data<BatchNormParamType<T>>();
            dbias = dbias_mem.data<BatchNormParamType<T>>();
          }

          BNBackward2DChannelLastStage2<T, block_size>
              <<<grid, block, 0, ctx.stream()>>>(
                  transformed_d_y.template data<T>(),
                  transformed_x.template data<T>(),
                  mean_ptr,
                  variance_ptr,
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  false,
                  block_data_ptr,
                  dscale,
                  dbias,
                  flag_ptr);

          // 3. elementwise_mul(scale, mean, inv_var, dy, dscale, dbias) => dx
          BNBackward2DChannelLastStage3<T, block_size>
              <<<grid, block, 0, ctx.stream()>>>(
                  transformed_d_y.template data<T>(),
                  transformed_x.template data<T>(),
                  new_scale.template data<BatchNormParamType<T>>(),
                  dscale,
                  dbias,
                  mean_ptr,
                  variance_ptr,
                  C,
                  N,
                  H * W * D,
                  epsilon,
                  transformed_d_x.template data<T>());

        } else {
          if (compute_format == DataLayout::kNCHW) {
            BNBackward<T, block, DataLayout::kNCHW>
                <<<grid2, block, 0, ctx.stream()>>>(
                    transformed_d_y.template data<T>(),
                    transformed_x.template data<T>(),
                    new_scale.template data<BatchNormParamType<T>>(),
                    saved_mean_data,
                    saved_var_data,
                    C,
                    N,
                    H * W * D,
                    epsilon,
                    transformed_d_x.template data<T>(),
                    ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                    ctx.template Alloc<BatchNormParamType<T>>(d_bias));
          } else {
            BNBackward<T, block, DataLayout::kNHWC>
                <<<grid2, block, 0, ctx.stream()>>>(
                    transformed_d_y.template data<T>(),
                    transformed_x.template data<T>(),
                    new_scale.template data<BatchNormParamType<T>>(),
                    saved_mean_data,
                    saved_var_data,
                    C,
                    N,
                    H * W * D,
                    epsilon,
                    transformed_d_x.template data<T>(),
                    ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                    ctx.template Alloc<BatchNormParamType<T>>(d_bias));
          }
        }
      } else {
#if CUDNN_VERSION_MIN(7, 4, 1)
        size_t workspace_size = 0;
        void *workspace_ptr = nullptr;
        DenseTensor workspace_tensor;
        auto reserve_space_size = reserve_space->memory_size();
        // --------------- cudnn batchnorm workspace ---------------
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                /*handle=*/ctx.cudnn_handle(),
                /*mode=*/mode_,
                /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                /*xDesc=*/data_desc_,
                /*yDesc=*/data_desc_,
                /*dyDesc=*/data_desc_,
                /*dzDesc=*/nullptr,
                /*dxDesc=*/data_desc_,
                /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                /*activationDesc=*/nullptr,
                /*sizeInBytes=*/&workspace_size));

        workspace_tensor.Resize({static_cast<int64_t>(workspace_size)});
        workspace_ptr =
            static_cast<void *>(ctx.template Alloc<uint8_t>(&workspace_tensor));
        uint8_t *reserve_space_ptr = nullptr;
        if (reserve_space_size != 0) {
          reserve_space_ptr =
              const_cast<uint8_t *>(reserve_space->template data<uint8_t>());
        }
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnBatchNormalizationBackwardEx(
                /*handle=*/ctx.cudnn_handle(),
                /*mode=*/mode_,
                /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
                /*betaDataDiff=*/CudnnDataType<T>::kZero(),
                /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
                /*betaParamDiff=*/CudnnDataType<T>::kZero(),
                /*xDesc=*/data_desc_,
                /*xData=*/transformed_x.template data<T>(),
                /*yDesc=*/nullptr,
                /*yData=*/nullptr,
                /*dyDesc=*/data_desc_,
                /*dyData=*/transformed_d_y.template data<T>(),
                /*dzDesc=*/nullptr,
                /*dzData=*/nullptr,
                /*dxDesc=*/data_desc_,
                /*dxData=*/ctx.template Alloc<T>(&transformed_d_x),
                /*dBnScaleBiasDesc=*/bn_param_desc_,
                /*bnScaleData=*/
                new_scale.template data<BatchNormParamType<T>>(),
                /*bnBiasData=*/nullptr,
                /*dBnScaleData=*/
                ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                /*dBnBiasData=*/
                ctx.template Alloc<BatchNormParamType<T>>(d_bias),
                /*epsilon=*/epsilon,
                /*savedMean=*/saved_mean_data,
                /*savedInvVariance=*/saved_var_data,
                /*activationDesc=*/nullptr,
                /*workspace=*/workspace_ptr,
                /*workSpaceSizeInBytes=*/workspace_size,
                /*reserveSpace=*/
                // const_cast<uint8_t *>(reserve_space->template
                // data<uint8_t>()),
                reserve_space_ptr,
                /*reserveSpaceSizeInBytes=*/reserve_space_size));
#else
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnBatchNormalizationBackward(
                ctx.cudnn_handle(),
                mode_,
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                data_desc_,
                transformed_x.template data<T>(),
                data_desc_,
                transformed_d_y.template data<T>(),
                data_desc_,
                ctx.template Alloc<T>(&transformed_d_x),
                bn_param_desc_,
                new_scale.template data<BatchNormParamType<T>>(),
                ctx.template Alloc<BatchNormParamType<T>>(d_scale),
                ctx.template Alloc<BatchNormParamType<T>>(d_bias),
                epsilon,
                saved_mean_data,
                saved_var_data));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
      }
#endif

      if (data_layout == DataLayout::kNHWC &&
          compute_format == DataLayout::kNCHW) {
        VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
        TransToChannelLast<Context, T>(ctx, &transformed_d_x, d_x);
      }
    } else {
      // This branch call CUDA kernels
      if (compute_format == DataLayout::kNCHW) {
        if (data_layout == DataLayout::kNHWC) {
          if (d_x) {
            BNBackwardData<T, block, phi::DataLayout::kNHWC>
                <<<grid2, block, 0, ctx.stream()>>>(
                    d_y->data<T>(),
                    new_scale.data<BatchNormParamType<T>>(),
                    saved_mean_data,
                    x.data<T>(),
                    saved_var_data,
                    C,
                    N,
                    H * W * D,
                    d_x->data<T>());
          }
          if (d_scale && d_bias) {
            KeBNBackwardScaleBias<T, block, phi::DataLayout::kNHWC>
                <<<grid2, block, 0, stream>>>(
                    d_y->data<T>(),
                    x.data<T>(),
                    saved_mean_data,
                    saved_var_data,
                    epsilon,
                    N,
                    C,
                    H * W * D,
                    d_scale->data<BatchNormParamType<T>>(),
                    d_bias->data<BatchNormParamType<T>>());
          }
        } else {
          if (d_x) {
            BNBackwardData<T, block, phi::DataLayout::kNCHW>
                <<<grid2, block, 0, ctx.stream()>>>(
                    d_y->data<T>(),
                    new_scale.data<BatchNormParamType<T>>(),
                    saved_mean_data,
                    x.data<T>(),
                    saved_var_data,
                    C,
                    N,
                    H * W * D,
                    d_x->data<T>());
          }
          if (d_scale && d_bias) {
            KeBNBackwardScaleBias<T, block, phi::DataLayout::kNCHW>
                <<<grid2, block, 0, stream>>>(
                    d_y->data<T>(),
                    x.data<T>(),
                    saved_mean_data,
                    saved_var_data,
                    epsilon,
                    N,
                    C,
                    H * W * D,
                    d_scale->data<BatchNormParamType<T>>(),
                    d_bias->data<BatchNormParamType<T>>());
          }
        }
      } else {
        if (d_x) {
          BNBackwardData<T, block, phi::DataLayout::kNHWC>
              <<<grid2, block, 0, ctx.stream()>>>(
                  d_y->data<T>(),
                  new_scale.data<BatchNormParamType<T>>(),
                  saved_mean_data,
                  x.data<T>(),
                  saved_var_data,
                  C,
                  N,
                  H * W * D,
                  d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<T, block, phi::DataLayout::kNHWC>
              <<<grid2, block, 0, stream>>>(
                  d_y->data<T>(),
                  x.data<T>(),
                  saved_mean_data,
                  saved_var_data,
                  epsilon,
                  N,
                  C,
                  H * W * D,
                  d_scale->data<BatchNormParamType<T>>(),
                  d_bias->data<BatchNormParamType<T>>());
        }
      }
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

  } else {
    const auto *running_mean = mean.get_ptr();
    const auto *running_var = variance.get_ptr();

    const auto *running_mean_data =
        running_mean->template data<BatchNormParamType<T>>();
    const auto *running_var_data =
        running_var->template data<BatchNormParamType<T>>();

    if (is_inplace) {
      auto px = x;
      inplace_functor(data_layout,
                      ctx.template Alloc<T>(&px),
                      new_scale.template data<BatchNormParamType<T>>(),
                      new_bias.template data<BatchNormParamType<T>>(),
                      running_mean_data,
                      running_var_data,
                      epsilon,
                      C,
                      H * W * D,
                      num,
                      x.data<T>(),
                      grid2,
                      block,
                      stream);
    }

    if (compute_format == DataLayout::kNCHW) {
      if (data_layout == DataLayout::kNHWC) {
        if (d_x) {
          KeBNBackwardData<T, phi::DataLayout::kNHWC>
              <<<grid1, block, 0, stream>>>(
                  d_y->data<T>(),
                  new_scale.data<BatchNormParamType<T>>(),
                  running_var_data,
                  epsilon,
                  C,
                  H * W,
                  num,
                  d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<T, block, phi::DataLayout::kNHWC>
              <<<grid2, block, 0, stream>>>(
                  d_y->data<T>(),
                  x.data<T>(),
                  running_mean_data,
                  running_var_data,
                  epsilon,
                  N,
                  C,
                  H * W * D,
                  d_scale->data<BatchNormParamType<T>>(),
                  d_bias->data<BatchNormParamType<T>>());
        }
      } else {
        if (d_x) {
          KeBNBackwardData<T, phi::DataLayout::kNCHW>
              <<<grid1, block, 0, stream>>>(
                  d_y->data<T>(),
                  new_scale.data<BatchNormParamType<T>>(),
                  running_var_data,
                  epsilon,
                  C,
                  H * W,
                  num,
                  d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<T, block, phi::DataLayout::kNCHW>
              <<<grid2, block, 0, stream>>>(
                  d_y->data<T>(),
                  x.data<T>(),
                  running_mean_data,
                  running_var_data,
                  epsilon,
                  N,
                  C,
                  H * W * D,
                  d_scale->data<BatchNormParamType<T>>(),
                  d_bias->data<BatchNormParamType<T>>());
        }
      }
    } else {
      if (d_x) {
        KeBNBackwardData<T, phi::DataLayout::kNHWC>
            <<<grid1, block, 0, stream>>>(
                d_y->data<T>(),
                new_scale.data<BatchNormParamType<T>>(),
                running_var_data,
                epsilon,
                C,
                H * W,
                num,
                d_x->data<T>());
      }
      if (d_scale && d_bias) {
        dim3 block;
        dim3 grid;
        const int block_size = 512;

        // init intermediate storage
        DenseTensor block_data_tensor;
        DenseTensor flag_tensor;
        BatchNormParamType<T> *block_data_ptr = nullptr;
        int *flag_ptr = nullptr;

        funcs::SetLaunchConfigInfoForChannelLast<T, BatchNormParamType<T>>(
            ctx,
            &block_data_tensor,
            &flag_tensor,
            &block_data_ptr,
            &flag_ptr,
            N,
            H,
            W,
            D,
            C,
            block_size,
            &block,
            &grid);
        BNBackward2DChannelLastStage2<T, block_size>
            <<<grid, block, 0, ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                running_mean_data,
                running_var_data,
                C,
                N,
                H * W * D,
                epsilon,
                true,
                block_data_ptr,
                d_scale->data<BatchNormParamType<T>>(),
                d_bias->data<BatchNormParamType<T>>(),
                flag_ptr);
      }
    }
  }
}

template <typename T, typename Context>
void BatchNormGradKernel(const Context &dev_ctx,
                         const DenseTensor &x,
                         const paddle::optional<DenseTensor> &scale,
                         const paddle::optional<DenseTensor> &bias,
                         const paddle::optional<DenseTensor> &mean,
                         const paddle::optional<DenseTensor> &variance,
                         const DenseTensor &saved_mean,
                         const DenseTensor &saved_variance,
                         const paddle::optional<DenseTensor> &reserve_space,
                         const DenseTensor &y_grad,
                         float momentum,
                         float epsilon,
                         const std::string &data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         DenseTensor *x_grad,
                         DenseTensor *scale_grad,
                         DenseTensor *bias_grad) {
  BatchNormGradFunctor<T, Context>(dev_ctx,
                                   x,
                                   scale,
                                   bias,
                                   mean,
                                   variance,
                                   saved_mean,
                                   saved_variance,
                                   reserve_space,
                                   y_grad,
                                   momentum,
                                   epsilon,
                                   data_layout,
                                   is_test,
                                   use_global_stats,
                                   trainable_statistics,
                                   false,
                                   x_grad,
                                   scale_grad,
                                   bias_grad);
}

template <typename T, typename Context>
void BatchNormDoubleGradKernel(
    const Context &ctx,
    const DenseTensor &x,
    const paddle::optional<DenseTensor> &scale,
    const paddle::optional<DenseTensor> &mean,
    const paddle::optional<DenseTensor> &variance,
    const DenseTensor &saved_mean,
    const DenseTensor &saved_variance,
    const DenseTensor &y_grad,
    const paddle::optional<DenseTensor> &x_grad_grad,
    const paddle::optional<DenseTensor> &scale_grad_grad,
    const paddle::optional<DenseTensor> &bias_grad_grad,
    float momentum,
    float epsilon,
    const std::string &data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    DenseTensor *x_grad,
    DenseTensor *scale_grad,
    DenseTensor *y_grad_grad) {
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    common::errors::InvalidArgument(
                        "`is_test = True` CANNOT be used in train program. If "
                        "you want to use global status in pre_train model, "
                        "please set `use_global_stats = True`"));

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  const DenseTensor *running_mean = nullptr;
  const DenseTensor *running_variance = nullptr;
  if (use_global_stats) {
    running_mean = mean.get_ptr();
    running_variance = variance.get_ptr();
  }
  const auto &x_dims = x.dims();
  int N, C, H, W, D;
  phi::funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);
  auto *Scale = scale.get_ptr();
  phi::DenseTensor new_scale;
  if (Scale) {
    new_scale = scale.get();
  } else {
    new_scale = phi::Full<T, Context>(ctx, {C}, static_cast<T>(1));
  }
  phi::funcs::NormDoubleGradFunctor<Context, T>(ctx,
                                                data_layout,
                                                &x,
                                                &new_scale,
                                                &y_grad,
                                                &saved_mean,
                                                &saved_variance,
                                                running_mean,
                                                running_variance,
                                                epsilon,
                                                use_global_stats,
                                                x_grad_grad.get_ptr(),
                                                scale_grad_grad.get_ptr(),
                                                bias_grad_grad.get_ptr(),
                                                x_grad,
                                                scale_grad,
                                                y_grad_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_DECLARE_BN_GRAD_FUNCTOR(float, GPU);
PD_DECLARE_BN_GRAD_FUNCTOR(phi::dtype::float16, GPU);

PD_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradKernel,
                   float,
                   phi::dtype::float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)

PD_DECLARE_BN_GRAD_FUNCTOR(float, GPU);
PD_DECLARE_BN_GRAD_FUNCTOR(double, GPU);
PD_DECLARE_BN_GRAD_FUNCTOR(phi::dtype::bfloat16, GPU);
PD_DECLARE_BN_GRAD_FUNCTOR(phi::dtype::float16, GPU);

PD_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
#else
PD_DECLARE_BN_GRAD_FUNCTOR(float, GPU);
PD_DECLARE_BN_GRAD_FUNCTOR(double, GPU);
PD_DECLARE_BN_GRAD_FUNCTOR(phi::dtype::float16, GPU);

PD_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
#endif
#endif

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormDoubleGradKernel,
                   float,
                   double) {}
#else
PD_REGISTER_KERNEL(batch_norm_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormDoubleGradKernel,
                   float,
                   double) {}
#endif
