/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/common/layout.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

namespace phi {
namespace funcs {

// math: dx = scale * ((x - mean) * inv_var / NxHxW * (np.mean(ddx,
// axis=(n,h,w)) *
//          np.sum(dy, axis=(n,h,w)) -
//          np.sum(dy * ddx, axis=(n,h,w)) + 3 * np.mean(dy * (x -
//          mean),
//          axis=(n,h,w)) * inv_var.pow(2) *
//          np.sum(ddx * (x - mean), axis=(n,h,w))) + inv_var.pow(3) /
//          NxHxW *
//          np.sum(ddx * (x - mean)) *
//          (np.mean(dy, axis=(n,h,w)) - dy) + inv_var.pow(3) / NxHxW *
//          np.sum(dy,
//          axis=(n,h,w)) * (x - mean) *
//          (np.mean(ddx, axis=(n,h,w)) - ddx)) + ddr * (dy * inv_var -
//          inv_var
//          *
//          np.mean(dy, axis=(n,h,w)) -
//          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
//          axis=(n,h,w)))

template <typename T, int BlockDim, phi::DataLayout layout>
__global__ LAUNCH_BOUNDS(BlockDim) void DoubleGradComputeDX(
    const T *x,
    const T *mean,
    const T *variance,
    const T *ddx,
    const T *dy,
    const T *scale,
    const T *ddscale,
    const int N,
    const int C,
    const int sample_size,
    const double epsilon,
    T *dx) {
  const int outer_size = C;
  const int inner_size = N * sample_size;

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage ddx_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_ddx_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_x_sub_mean_storage;
  __shared__ typename BlockReduce::TempStorage ddx_mul_x_sub_mean_storage;
  __shared__ T dy_sum_val;
  __shared__ T ddx_sum_val;
  __shared__ T dy_mul_ddx_sum_val;
  __shared__ T dy_mul_x_sub_mean_sum_val;
  __shared__ T ddx_mul_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T mean_val = mean[i];
    T var_val = variance[i];
    T dy_sum = 0;
    T ddx_sum = 0;
    T dy_mul_ddx_sum = 0;
    T dy_mul_x_sub_mean_sum = 0;
    T ddx_mul_x_sub_mean_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == phi::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T ddx_i = ddx[index];
      T dy_i = dy[index];
      T tmp = x[index] - mean_val;

      dy_sum += dy_i;
      ddx_sum += ddx_i;
      dy_mul_ddx_sum += (ddx_i * dy_i);

      dy_mul_x_sub_mean_sum += (dy_i * tmp);
      ddx_mul_x_sub_mean_sum += (ddx_i * tmp);
    }

    dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
    ddx_sum = BlockReduce(ddx_storage).Reduce(ddx_sum, cub::Sum());
    dy_mul_ddx_sum =
        BlockReduce(dy_mul_ddx_storage).Reduce(dy_mul_ddx_sum, cub::Sum());
    dy_mul_x_sub_mean_sum = BlockReduce(dy_mul_x_sub_mean_storage)
                                .Reduce(dy_mul_x_sub_mean_sum, cub::Sum());
    ddx_mul_x_sub_mean_sum = BlockReduce(ddx_mul_x_sub_mean_storage)
                                 .Reduce(ddx_mul_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      dy_sum_val = dy_sum;
      ddx_sum_val = ddx_sum;
      dy_mul_ddx_sum_val = dy_mul_ddx_sum;
      dy_mul_x_sub_mean_sum_val = dy_mul_x_sub_mean_sum;
      ddx_mul_x_sub_mean_sum_val = ddx_mul_x_sub_mean_sum;
    }
    __syncthreads();

    if (ddx != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == phi::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        dx[index] +=
            ((x[index] - mean_val) * var_val * var_val * var_val / inner_size *
                 (ddx_sum_val * dy_sum_val / inner_size - dy_mul_ddx_sum_val +
                  3. * dy_mul_x_sub_mean_sum_val * var_val *
                      ddx_mul_x_sub_mean_sum_val * var_val / inner_size) +
             ddx_mul_x_sub_mean_sum_val * var_val / inner_size * var_val *
                 var_val * (dy_sum_val / inner_size - dy[index]) +
             dy_mul_x_sub_mean_sum_val * var_val / inner_size * var_val *
                 var_val * (ddx_sum_val / inner_size - ddx[index])) *
            scale[i];
      }
    }
    __syncthreads();
    if (ddscale != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == phi::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        dx[index] += (dy[index] * var_val - dy_sum_val / inner_size * var_val -
                      (x[index] - mean_val) * var_val * var_val *
                          dy_mul_x_sub_mean_sum_val * var_val / inner_size) *
                     ddscale[i];
      }
    }
  }
}

// math: ddy = (x - mean) * inv_var * ddscale + ddbias +
//           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
//           np.mean(ddx * (x - mean), axis=(n,h,w)))
template <typename T, int BlockDim, phi::DataLayout layout>
__global__ LAUNCH_BOUNDS(BlockDim) void DoubleGradComputeDDY(
    const T *x,
    const T *mean,
    const T *variance,
    const T *ddscale,
    const T *ddbias,
    const T *ddx,
    const T *scale,
    const int N,
    const int C,
    const int sample_size,
    const double epsilon,
    T *ddy) {
  const int outer_size = C;
  const int inner_size = N * sample_size;

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ddx_storage;
  __shared__ typename BlockReduce::TempStorage ddx_mul_x_sub_mean_storage;
  __shared__ T ddx_sum_val;
  __shared__ T ddx_mul_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T mean_val = mean[i];
    T var_val = variance[i];
    T ddx_sum = 0;
    T ddx_mul_x_sub_mean_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == phi::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T ddx_i = ddx[index];
      ddx_sum += ddx_i;
      ddx_mul_x_sub_mean_sum += (ddx_i * (x[index] - mean_val));
    }
    ddx_sum = BlockReduce(ddx_storage).Reduce(ddx_sum, cub::Sum());
    ddx_mul_x_sub_mean_sum = BlockReduce(ddx_mul_x_sub_mean_storage)
                                 .Reduce(ddx_mul_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      ddx_sum_val = ddx_sum;
      ddx_mul_x_sub_mean_sum_val = ddx_mul_x_sub_mean_sum;
    }
    __syncthreads();

    if (ddx != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == phi::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        ddy[index] += scale[i] * var_val *
                      (ddx[index] - ddx_sum_val / inner_size -
                       (x[index] - mean_val) * var_val *
                           ddx_mul_x_sub_mean_sum_val * var_val / inner_size);
      }
    }
    __syncthreads();
    if (ddscale != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == phi::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        ddy[index] += (x[index] - mean_val) * var_val * ddscale[i];
      }
    }
    __syncthreads();
    if (ddbias != nullptr) {
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == phi::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        ddy[index] += ddbias[i];
      }
    }
  }
}

// math: dscale = inv_var * (dy - np.mean(dy, axis=(n,h,w) - (x-mean) *
//            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(n,h,w)))) *
//            ddx
template <typename T, int BlockDim, phi::DataLayout layout>
__global__ LAUNCH_BOUNDS(BlockDim) void DoubleGradComputeDScale(
    const T *x,
    const T *mean,
    const T *variance,
    const T *ddx,
    const T *dy,
    const int N,
    const int C,
    const int sample_size,
    const double epsilon,
    T *dscale) {
  const int outer_size = C;
  const int inner_size = N * sample_size;

  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_mul_x_sub_mean_storage;
  __shared__ typename BlockReduce::TempStorage dscale_tmp_storage;
  __shared__ T dy_sum_val;
  __shared__ T dy_mul_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T dy_sum = 0;
    T dy_mul_x_sub_mean_sum = 0;
    T mean_val = mean[i];
    T var_val = variance[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == phi::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T dy_i = dy[index];
      dy_sum += dy_i;
      dy_mul_x_sub_mean_sum += (dy_i * (x[index] - mean_val));
    }
    dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
    dy_mul_x_sub_mean_sum = BlockReduce(dy_mul_x_sub_mean_storage)
                                .Reduce(dy_mul_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      dy_sum_val = dy_sum;
      dy_mul_x_sub_mean_sum_val = dy_mul_x_sub_mean_sum;
    }
    __syncthreads();

    if (ddx != nullptr) {
      T dscale_tmp = 0;
      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index =
            layout == phi::DataLayout::kNCHW
                ? (j / sample_size * C + i) * sample_size + j % sample_size
                : j * outer_size + i;
        dscale_tmp += ddx[index] * var_val *
                      (dy[index] - dy_sum_val / inner_size -
                       dy_mul_x_sub_mean_sum_val * (x[index] - mean_val) *
                           var_val * var_val / inner_size);
      }
      dscale_tmp =
          BlockReduce(dscale_tmp_storage).Reduce(dscale_tmp, cub::Sum());

      if (threadIdx.x == 0) {
        dscale[i] += dscale_tmp;
      }
      __syncthreads();
    }
  }
}

// math: dscale = np.sum(ddx * dy, axis=(n,h,w)) * inv_var
template <typename T, int BlockDim, phi::DataLayout layout>
__global__ LAUNCH_BOUNDS(BlockDim) void DoubleGradComputeDScaleWithGlobal(
    const T *ddx,
    const T *variance,
    const T *dy,
    const double epsilon,
    const int N,
    const int C,
    const int sample_size,
    T *dscale) {
  int outer_size = C;
  int inner_size = N * sample_size;
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ddx_mul_dy_storage;
  __shared__ T ddx_mul_dy_sum_val;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T inv_var_i = 1.0 / sqrt(variance[i] + epsilon);
    T ddx_mul_dy_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index =
          layout == phi::DataLayout::kNCHW
              ? (j / sample_size * C + i) * sample_size + j % sample_size
              : j * outer_size + i;
      T ddx_i = ddx[index];
      T dy_i = dy[index];
      ddx_mul_dy_sum += (ddx_i * dy_i);
    }
    ddx_mul_dy_sum =
        BlockReduce(ddx_mul_dy_storage).Reduce(ddx_mul_dy_sum, cub::Sum());
    if (threadIdx.x == 0) {
      ddx_mul_dy_sum_val = ddx_mul_dy_sum;
    }
    __syncthreads();

    if (ddx != nullptr) {
      dscale[i] = inv_var_i * ddx_mul_dy_sum_val;
    }
  }
}

// math: dx = ddscale * dy * inv_var
template <typename T, phi::DataLayout layout>
__global__ void DoubleGradComputeDXWithGlobal(const T *dy,
                                              const T *ddscale,
                                              const T *variance,
                                              const double epsilon,
                                              const int C,
                                              const int sample_size,
                                              const int num,
                                              T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (ddscale != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == phi::DataLayout::kNCHW ? i / sample_size % C : i % C;
      T inv_var = 1.0 / sqrt(variance[c] + epsilon);
      dx[i] = dy[i] * ddscale[c] * inv_var;
    }
  }
}

// math: ddy = scale * ddx * inv_var + ddbias +
//             ddscale * (x - mean) * inv_var
template <typename T, phi::DataLayout layout>
__global__ void DoubleGradComputeDDYWithGlobal(const T *ddx,
                                               const T *scale,
                                               const T *mean,
                                               const T *variance,
                                               const T *x,
                                               const T *ddbias,
                                               const T *ddscale,
                                               const double epsilon,
                                               const int C,
                                               const int sample_size,
                                               const int num,
                                               T *ddy) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (ddx != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == phi::DataLayout::kNCHW ? i / sample_size % C : i % C;
      T inv_var = 1.0 / sqrt(variance[c] + epsilon);
      ddy[i] += ddx[i] * scale[c] * inv_var;
    }
  }
  __syncthreads();
  if (ddscale != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == phi::DataLayout::kNCHW ? i / sample_size % C : i % C;
      T inv_var = 1.0 / sqrt(variance[c] + epsilon);
      ddy[i] += (x[i] - mean[c]) * inv_var * ddscale[c];
    }
  }
  __syncthreads();
  if (ddbias != nullptr) {
    for (int i = gid; i < num; i += stride) {
      const int c =
          layout == phi::DataLayout::kNCHW ? i / sample_size % C : i % C;
      ddy[i] += ddbias[c];
    }
  }
}

template <typename DeviceContext, typename T>
void NormDoubleGradFunctor(const DeviceContext &ctx,
                           const DataLayout data_layout,
                           const phi::DenseTensor *X,
                           const phi::DenseTensor *Scale,
                           const phi::DenseTensor *dY,
                           const phi::DenseTensor *Saved_mean,
                           const phi::DenseTensor *Saved_variance,
                           const phi::DenseTensor *Mean,
                           const phi::DenseTensor *Variance,
                           const double epsilon,
                           const bool use_global_stats,
                           const phi::DenseTensor *ddX,
                           const phi::DenseTensor *ddScale,
                           const phi::DenseTensor *ddBias,
                           phi::DenseTensor *dX,
                           phi::DenseTensor *dScale,
                           phi::DenseTensor *ddY) {
  const T *x_data = X->data<T>();
  const T *dy_data = dY->data<T>();
  const T *ddx_data = (ddX == nullptr ? nullptr : ddX->data<T>());

  const T *ddscale_data = (ddScale == nullptr ? nullptr : ddScale->data<T>());
  const T *ddbias_data = (ddBias == nullptr ? nullptr : ddBias->data<T>());

  phi::funcs::SetConstant<DeviceContext, T> set_constant;

  auto &x_dims = X->dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int N = x_dims[0];
  const int num = X->numel();
  const int sample_size = num / N / C;
  phi::DenseTensor scale_tmp;
  if (!Scale) {
    scale_tmp.Resize({C});
    ctx.template Alloc<T>(&scale_tmp);
    set_constant(ctx, &scale_tmp, static_cast<T>(1));
  }
  const T *scale_data = Scale ? Scale->data<T>() : scale_tmp.data<T>();
  const int block = 512;
  int max_threads = ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid = std::min(C, max_blocks);
  int grid1 = (num + block - 1) / block;

  const T *mean_data, *variance_data;
  if (use_global_stats) {
    const auto *running_mean = Mean;
    const auto *running_var = Variance;
    const auto *running_mean_data = running_mean->template data<T>();
    const auto *running_var_data = running_var->template data<T>();
    mean_data = running_mean_data;
    variance_data = running_var_data;
  } else {
    const T *smean_data = Saved_mean->data<T>();
    const T *svariance_data = Saved_variance->data<T>();

    mean_data = smean_data;
    variance_data = svariance_data;
  }

  if (dX) {
    T *dx_data = ctx.template Alloc<T>(dX);
    set_constant(ctx, dX, static_cast<T>(0));
    if (use_global_stats) {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDXWithGlobal<T, DataLayout::kNHWC>
            <<<grid1, block, 0, ctx.stream()>>>(dy_data,
                                                ddscale_data,
                                                variance_data,
                                                epsilon,
                                                C,
                                                sample_size,
                                                num,
                                                dx_data);
      } else {
        DoubleGradComputeDXWithGlobal<T, DataLayout::kNCHW>
            <<<grid1, block, 0, ctx.stream()>>>(dy_data,
                                                ddscale_data,
                                                variance_data,
                                                epsilon,
                                                C,
                                                sample_size,
                                                num,
                                                dx_data);
      }
    } else {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDX<T, block, DataLayout::kNHWC>
            <<<grid, block, 0, ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddx_data,
                                               dy_data,
                                               scale_data,
                                               ddscale_data,
                                               N,
                                               C,
                                               sample_size,
                                               epsilon,
                                               dx_data);
      } else {
        DoubleGradComputeDX<T, block, DataLayout::kNCHW>
            <<<grid, block, 0, ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddx_data,
                                               dy_data,
                                               scale_data,
                                               ddscale_data,
                                               N,
                                               C,
                                               sample_size,
                                               epsilon,
                                               dx_data);
      }
    }
  }
  if (dScale) {
    T *dscale_data = ctx.template Alloc<T>(dScale);
    set_constant(ctx, dScale, static_cast<T>(0));
    if (use_global_stats) {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDScaleWithGlobal<T, block, DataLayout::kNHWC>
            <<<grid, block, 0, ctx.stream()>>>(ddx_data,
                                               variance_data,
                                               dy_data,
                                               epsilon,
                                               N,
                                               C,
                                               sample_size,
                                               dscale_data);
      } else {
        DoubleGradComputeDScaleWithGlobal<T, block, DataLayout::kNCHW>
            <<<grid, block, 0, ctx.stream()>>>(ddx_data,
                                               variance_data,
                                               dy_data,
                                               epsilon,
                                               N,
                                               C,
                                               sample_size,
                                               dscale_data);
      }
    } else {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDScale<T, block, DataLayout::kNHWC>
            <<<grid, block, 0, ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddx_data,
                                               dy_data,
                                               N,
                                               C,
                                               sample_size,
                                               epsilon,
                                               dscale_data);
      } else {
        DoubleGradComputeDScale<T, block, DataLayout::kNCHW>
            <<<grid, block, 0, ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddx_data,
                                               dy_data,
                                               N,
                                               C,
                                               sample_size,
                                               epsilon,
                                               dscale_data);
      }
    }
  }
  if (ddY) {
    T *ddy_data = ctx.template Alloc<T>(ddY);
    set_constant(ctx, ddY, static_cast<T>(0));
    if (use_global_stats) {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDDYWithGlobal<T, DataLayout::kNHWC>
            <<<grid1, block, 0, ctx.stream()>>>(ddx_data,
                                                scale_data,
                                                mean_data,
                                                variance_data,
                                                x_data,
                                                ddbias_data,
                                                ddscale_data,
                                                epsilon,
                                                C,
                                                sample_size,
                                                num,
                                                ddy_data);
      } else {
        DoubleGradComputeDDYWithGlobal<T, DataLayout::kNCHW>
            <<<grid1, block, 0, ctx.stream()>>>(ddx_data,
                                                scale_data,
                                                mean_data,
                                                variance_data,
                                                x_data,
                                                ddbias_data,
                                                ddscale_data,
                                                epsilon,
                                                C,
                                                sample_size,
                                                num,
                                                ddy_data);
      }
    } else {
      if (data_layout == DataLayout::kNHWC) {
        DoubleGradComputeDDY<T, block, DataLayout::kNHWC>
            <<<grid, block, 0, ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddscale_data,
                                               ddbias_data,
                                               ddx_data,
                                               scale_data,
                                               N,
                                               C,
                                               sample_size,
                                               epsilon,
                                               ddy_data);
      } else {
        DoubleGradComputeDDY<T, block, DataLayout::kNCHW>
            <<<grid, block, 0, ctx.stream()>>>(x_data,
                                               mean_data,
                                               variance_data,
                                               ddscale_data,
                                               ddbias_data,
                                               ddx_data,
                                               scale_data,
                                               N,
                                               C,
                                               sample_size,
                                               epsilon,
                                               ddy_data);
      }
    }
  }
}

template <typename T, typename BnT>
__device__ __forceinline__ void BlockReduceByVertical(BnT x_sum,
                                                      BnT x_square_sum,
                                                      BnT *smem_sum,
                                                      BnT *smem_square_sum,
                                                      BnT *x_sum_out,
                                                      BnT *x_square_sum_out) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      smem_sum[tid] = x_sum;
      smem_square_sum[tid] = x_square_sum;
    }
    __syncthreads();
    if (threadIdx.y < offset) {
      int pair_tid = tid + offset * blockDim.x;
      x_sum += smem_sum[pair_tid];
      x_square_sum += smem_square_sum[pair_tid];
    }
  }
  if (threadIdx.y == 0) {
    *x_sum_out = x_sum;
    *x_square_sum_out = x_square_sum;
  }
}

template <typename T, typename BnT>
__device__ __forceinline__ void ReduceSumPost(const int C,  // channels
                                              const int c,  // channel index
                                              BnT *sum1,
                                              BnT *sum2,
                                              bool *is_last_block_done,
                                              BnT *cache1,
                                              BnT *cache2,
                                              BnT *block_data_ptr,
                                              int *flag_ptr) {
  volatile BnT *staging_sum = block_data_ptr;
  volatile BnT *staging_sum2 = &block_data_ptr[C * gridDim.y];
  // write block data to global memory
  if (threadIdx.y == 0) {
    staging_sum[c + blockIdx.y * C] = *sum1;
    staging_sum2[c + blockIdx.y * C] = *sum2;
  }

  // make sure write is visible to all blocks
  __threadfence();
  __syncthreads();

  // mark block done
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int old = atomicAdd(&flag_ptr[blockIdx.x], 1);
    *is_last_block_done = (old == (gridDim.y - 1));
  }

  __syncthreads();

  if (*is_last_block_done) {
    *sum1 = static_cast<BnT>(0);
    *sum2 = static_cast<BnT>(0);
    // thread sum
    for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
      *sum1 += staging_sum[c + y * C];
      *sum2 += staging_sum2[c + y * C];
    }

    // vertical block sum
    funcs::BlockReduceByVertical<T, BnT>(
        *sum1, *sum2, &cache1[0], &cache2[0], sum1, sum2);
  }
}

template <typename T, typename BnT, typename Context>
void SetLaunchConfigInfoForChannelLast(const Context &ctx,
                                       DenseTensor *block_data_tensor,
                                       DenseTensor *flag_tensor,
                                       BnT **block_data_ptr,
                                       int **flag_ptr,
                                       const int N,
                                       const int H,
                                       const int W,
                                       const int D,
                                       const int C,
                                       const int block_size,
                                       dim3 *block,
                                       dim3 *grid) {
  const int MAX_GRID_SIZE = 128;
  const int WARP_SIZE = 32;

  int block_x = std::min(phi::funcs::details::GetLastPow2(C), WARP_SIZE);
  int block_y = std::min(phi::funcs::details::GetLastPow2(N * H * W * D / 16),
                         block_size / block_x);
  if (block_x * block_y != block_size) {
    block_x =
        std::min(phi::funcs::details::GetLastPow2(C), block_size / block_y);
  }
  int grid_x = (C + block_x - 1) / block_x;
  int grid_y = std::min((N * H * W * D + block_y * 16 - 1) / (block_y * 16),
                        MAX_GRID_SIZE);

  block->x = block_x;
  block->y = block_y;
  grid->x = grid_x;
  grid->y = grid_y;

  if (grid->y > 1) {
    *block_data_tensor = phi::Empty<BnT, Context>(ctx, {2 * C * grid->y});
    *flag_tensor = phi::Empty<int, Context>(ctx, {grid->x});

    *block_data_ptr = block_data_tensor->data<BnT>();
    *flag_ptr = flag_tensor->data<int>();
    funcs::SetConstant<Context, int> set_zero;
    set_zero(ctx, flag_tensor, static_cast<int>(0));
  }
}

}  // namespace funcs
}  // namespace phi
