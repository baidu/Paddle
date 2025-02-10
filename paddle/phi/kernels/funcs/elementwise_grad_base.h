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

#pragma once

#include "glog/logging.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/elementwise_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

#endif

constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;

#define BLOCK_X 32
#define BLOCK_Y 32

#define GetDivMod(dividend, divisor, div, mod) \
  do {                                         \
    const auto dividend_copy = dividend;       \
    *div = dividend_copy / divisor;            \
    *mod = dividend_copy % divisor;            \
  } while (0)

namespace phi {

namespace funcs {
using DDim = phi::DDim;

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonGradBroadcastCPU(const DenseTensor &x,
                            const DenseTensor &y,
                            const DenseTensor &out,
                            const DenseTensor &dout,
                            DenseTensor *dx,
                            DenseTensor *dy,
                            int *x_dims_array,
                            int *y_dims_array,
                            int *out_dims_array,
                            int max_dim,
                            const CPUContext &ctx,
                            DX_OP dx_op,
                            DY_OP dy_op) {
  std::vector<int> index_array(max_dim, 0);
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const Tout *out_data = out.data<Tout>();
  const Tout *dout_data = dout.data<Tout>();
  T *dx_data = dx == nullptr ? nullptr : ctx.Alloc<T>(dx);
  T *dy_data = dy == nullptr ? nullptr : ctx.Alloc<T>(dy);
  if (dx_data != nullptr) {
    memset(dx_data, 0, dx->numel() * sizeof(T));
  }
  if (dy_data != nullptr) {
    memset(dy_data, 0, dy->numel() * sizeof(T));
  }
  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (dx_data != nullptr) {
      dx_data[x_index] += dx_op(x_data[x_index],
                                y_data[y_index],
                                out_data[out_index],
                                dout_data[out_index]);
    }
    if (dy_data != nullptr) {
      dy_data[y_index] += dy_op(x_data[x_index],
                                y_data[y_index],
                                out_data[out_index],
                                dout_data[out_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast1CPU(const T *x,
                                      const T *y,
                                      const Tout *out,
                                      const Tout *dout,
                                      int h,
                                      int w,
                                      bool is_xsize_larger,
                                      DX_OP dx_op,
                                      DY_OP dy_op,
                                      T *dx,
                                      T *dy) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  if (is_xsize_larger) {
    for (int j = 0; j < w; ++j) {
      MPType sum_y = static_cast<MPType>(0);
      for (int i = 0; i < h; ++i) {
        int x_offset = i * w + j;
        if (dx != nullptr) {
          dx[x_offset] =
              dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
        }
        if (dy != nullptr) {
          sum_y += static_cast<MPType>(
              dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]));
        }
      }
      if (dy != nullptr) {
        dy[j] = static_cast<T>(sum_y);
      }
    }
  } else {
    for (int j = 0; j < w; ++j) {
      MPType sum_x = static_cast<MPType>(0);
      for (int i = 0; i < h; ++i) {
        int y_offset = i * w + j;
        if (dy != nullptr) {
          dy[y_offset] =
              dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx != nullptr) {
          sum_x += static_cast<MPType>(
              dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]));
        }
      }
      if (dx != nullptr) {
        dx[j] = static_cast<T>(sum_x);
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast2CPU(const T *x,
                                      const T *y,
                                      const Tout *out,
                                      const Tout *dout,
                                      int pre,
                                      int n,
                                      int post,
                                      bool is_xsize_larger,
                                      DX_OP dx_op,
                                      DY_OP dy_op,
                                      T *dx,
                                      T *dy) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  if (is_xsize_larger) {
    for (int j = 0; j < n; ++j) {
      MPType sum_y = static_cast<MPType>(0);
      for (int i = 0; i < pre; ++i) {
        for (int k = 0; k < post; ++k) {
          int x_offset = i * n * post + j * post + k;
          if (dx != nullptr) {
            dx[x_offset] =
                dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
          }
          if (dy != nullptr) {
            sum_y += static_cast<MPType>(
                dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]));
          }
        }
      }
      if (dy != nullptr) {
        dy[j] = static_cast<T>(sum_y);
      }
    }
  } else {
    for (int j = 0; j < n; ++j) {
      MPType sum_x = static_cast<MPType>(0);
      for (int i = 0; i < pre; ++i) {
        for (int k = 0; k < post; ++k) {
          int y_offset = i * n * post + j * post + k;
          if (dy != nullptr) {
            dy[y_offset] =
                dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
          }
          if (dx != nullptr) {
            sum_x += static_cast<MPType>(
                dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]));
          }
        }
      }
      if (dx != nullptr) {
        dx[j] = static_cast<T>(sum_x);
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonElementwiseBroadcastBackward(const CPUContext &ctx,
                                        const DDim &x_dims,
                                        const DDim &y_dims,
                                        const DenseTensor &x,
                                        const DenseTensor &y,
                                        const DenseTensor &out,
                                        const DenseTensor &dout,
                                        int axis,
                                        DenseTensor *dx,
                                        DenseTensor *dy,
                                        DX_OP dx_op,
                                        DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->Resize(x_dims);
    ctx.template Alloc<T>(dx);
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << common::make_ddim(x_dims_array)
          << " ydim:" << common::make_ddim(y_dims_array);

  CommonGradBroadcastCPU<T, DX_OP, DY_OP, Tout>(x,
                                                y,
                                                out,
                                                dout,
                                                dx,
                                                dy,
                                                x_dims_array.data(),
                                                y_dims_array.data(),
                                                out_dims_array.data(),
                                                max_dim,
                                                ctx,
                                                dx_op,
                                                dy_op);
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void ElemwiseGradComputeWithBroadcast(const CPUContext &ctx,
                                      const DDim &x_dims,
                                      const DDim &y_dims,
                                      const DenseTensor &x,
                                      const DenseTensor &y,
                                      const DenseTensor &out,
                                      const DenseTensor &dout,
                                      int axis,
                                      DenseTensor *dx,
                                      DenseTensor *dy,
                                      DX_OP dx_op,
                                      DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimmed = TrimTrailingSingularDims(y_dims);
    axis_trim = (y_dims_trimmed.size() == 0) ? x_dims.size() : axis;
    GetMidDims(x_dims,
               y_dims_trimmed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  } else {
    auto x_dims_trimmed = TrimTrailingSingularDims(x_dims);
    axis_trim = (x_dims_trimmed.size() == 0) ? y_dims.size() : axis;
    GetMidDims(y_dims,
               x_dims_trimmed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<T, DX_OP, DY_OP, Tout>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    ElemwiseGradBroadcast1CPU(x.data<T>(),
                              y.data<T>(),
                              out.data<Tout>(),
                              dout.data<Tout>(),
                              pre,
                              n,
                              is_xsize_larger,
                              dx_op,
                              dy_op,
                              dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                              dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  } else {
    ElemwiseGradBroadcast2CPU(x.data<T>(),
                              y.data<T>(),
                              out.data<Tout>(),
                              dout.data<Tout>(),
                              pre,
                              n,
                              post,
                              is_xsize_larger,
                              dx_op,
                              dy_op,
                              dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                              dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
struct ElemwiseGradNoBroadcast {
  const T *x_;
  const T *y_;
  const Tout *out_;
  const Tout *dout_;

  HOSTDEVICE void operator()(size_t i) {
    if (dx_ != nullptr) {
      dx_[i] = dx_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
    if (dy_ != nullptr) {
      dy_[i] = dy_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
  }

  DX_OP dx_op_;
  DY_OP dy_op_;
  T *dx_;
  T *dy_;
};

template <typename DeviceContext,
          typename T,
          typename DX_OP,
          typename DY_OP,
          typename Tout = T>
void ElemwiseGradComputeNoBroadcast(const DeviceContext &dev_ctx,
                                    const DDim &x_dim,
                                    const DDim &y_dim UNUSED,
                                    const DenseTensor &x,
                                    const DenseTensor &y,
                                    const DenseTensor &out,
                                    const DenseTensor &dout,
                                    int axis UNUSED,
                                    DenseTensor *dx,
                                    DenseTensor *dy,
                                    DX_OP dx_op,
                                    DY_OP dy_op) {
  size_t N = static_cast<size_t>(common::product(x_dim));
  phi::funcs::ForRange<DeviceContext> for_range(dev_ctx, N);
  for_range(ElemwiseGradNoBroadcast<T, DX_OP, DY_OP, Tout>{
      x.data<T>(),
      y.data<T>(),
      out.data<Tout>(),
      dout.data<Tout>(),
      dx_op,
      dy_op,
      dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
      dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy)});
}

#if defined(__NVCC__) || defined(__HIPCC__)
// Suppose only has contiguous dims
static inline bool CheckContiguousDims(const std::vector<int> &broadcast_pos) {
  for (int i = 1; i < broadcast_pos.size(); ++i) {
    if (broadcast_pos[i] != broadcast_pos[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

inline void ComputeBroadcastTranspositionArray(const int *x_one_indices,
                                               int *x_trans_indices,
                                               const int max_dim,
                                               const int x_one_size) {
  int diff = max_dim - x_one_size;
  std::copy_n(x_one_indices, x_one_size, x_trans_indices + diff);
  int p = 0;
  int q = diff;
  for (int i = 0; i < max_dim; ++i) {
    if (q < max_dim && i == x_trans_indices[q]) {
      ++q;
    } else {
      x_trans_indices[p++] = i;
    }
  }
}

// Check input can be split into 2 parts
static inline bool SplitDims(const std::vector<int> &y_broadcast_pos,
                             int max_dim) {
  bool can_split_dim2 = true;
  // must at start or end.
  if (y_broadcast_pos[0] != 0 &&
      y_broadcast_pos[y_broadcast_pos.size() - 1] != max_dim - 1) {
    can_split_dim2 = false;
  } else {
    for (int i = 1; i < y_broadcast_pos.size(); ++i) {
      // dim must be continue
      if (y_broadcast_pos[i] != y_broadcast_pos[i - 1] + 1) {
        can_split_dim2 = false;
        break;
      }
    }
  }
  return can_split_dim2;
}

inline void ComputeBroadcastKernelSize(int *x_dims_array,
                                       int *out_dims_array,
                                       int *x_blocks,
                                       int *x_threads,
                                       int max_dim) {
  *x_blocks = 1;
  *x_threads = 1;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] == out_dims_array[i]) {
      *x_blocks *= x_dims_array[i];
    } else {
      *x_threads *= out_dims_array[i];
    }
  }
}

template <typename T, typename OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastOneCUDAKernel(const T *x,
                                                            const T *y,
                                                            const Tout *out,
                                                            const Tout *dout,
                                                            int pre,
                                                            int n,
                                                            int post,
                                                            int y_pre,
                                                            int y_n,
                                                            int y_post,
                                                            bool is_xsize,
                                                            OP op,
                                                            T *dd) {
  int tid = THREAD_ID_X;
  int bid = BLOCK_ID_X;

  T val(0);
  if (is_xsize) {
    // do reduce for x
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      // Get y pre rows id with x post and y_pre.
      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int y_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  } else {
    // do reduce for y
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int x_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  }
}

template <typename T, typename DY_OP, typename DX_OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastAllCUDAKernel(
    const T *x,
    const T *y,
    const Tout *out,
    const Tout *dout,
    int pre,
    int n,
    int post,
    bool is_xsize_larger,
    DX_OP dx_op,
    DY_OP dy_op,
    T *dx,
    T *dy) {
  int tid = THREAD_ID_X;
  int bid = BLOCK_ID_X;

  T val(0);
  if (is_xsize_larger) {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + i * post + b_j;
      int y_offset = b_i * post + b_j;
      if (dx) {
        dx[x_offset] =
            dx_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
    }
    if (dy) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (tid == 0) {
        dy[bid] = val;
      }
    }
  } else {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + i * post + b_j;
      int x_offset = b_i * post + b_j;
      if (dy) {
        dy[y_offset] =
            dy_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
    }
    if (dx) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (tid == 0) {
        dx[bid] = val;
      }
    }
  }
}

template <typename T, typename DY_OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastCUDAKernelHeight(const T *x,
                                                               const T *y,
                                                               const Tout *out,
                                                               const Tout *dout,
                                                               int h,
                                                               int w,
                                                               DY_OP dy_op,
                                                               T *dy,
                                                               int x_h,
                                                               int x_w,
                                                               bool is_y) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = GRID_NUM_X * BLOCK_NUM_X;
  size_t idx = THREAD_ID_X + BLOCK_NUM_X * BLOCK_ID_X;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_y) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int x_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[out_offset], dout[out_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val +=
              phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dy[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  } else {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int y_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[m], y[y_offset], out[out_offset], dout[out_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val +=
              phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dy[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  }
}

template <typename T, typename DY_OP, typename Tout = T>
static __global__ void CommonGradBroadcast1CUDAKernelHeight(const T *x,
                                                            const T *y,
                                                            const Tout *out,
                                                            const Tout *dout,
                                                            int h,
                                                            int w,
                                                            DY_OP dy_op,
                                                            T *dy,
                                                            int x_h,
                                                            int x_w,
                                                            bool is_y) {
  int j = BLOCK_ID_X;
  int i = THREAD_ID_X;
  int tid = THREAD_ID_X;
  T val(0);

  if (is_y) {
    do {
      int out_offset = i * w + j;
      int x_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  } else {
    do {
      int out_offset = i * w + j;
      int y_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[j], y[y_offset], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void ElemwiseGradBroadcast1CUDAKernel(const T *x,
                                                        const T *y,
                                                        const Tout *out,
                                                        const Tout *dout,
                                                        int h,
                                                        int w,
                                                        bool is_xsize_larger,
                                                        DX_OP dx_op,
                                                        DY_OP dy_op,
                                                        T *dx,
                                                        T *dy) {
  int j = BLOCK_ID_X;
  int i = THREAD_ID_X;
  int tid = THREAD_ID_X;
  T val(0);
  if (is_xsize_larger) {
    do {
      int x_offset = i * w + j;
      if (dx) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    do {
      int y_offset = i * w + j;
      if (dy) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dx) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dx[j] = val;
      }
    }
  }
}

// suppose use 2D block is fast because more parallel
// and memory coalesced
template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void FastElemwiseGradBroadcast1CUDAKernel(
    const T *x,
    const T *y,
    const Tout *out,
    const Tout *dout,
    int h,
    int w,
    bool is_xsize_larger,
    DX_OP dx_op,
    DY_OP dy_op,
    T *dx,
    T *dy) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = GRID_NUM_X * BLOCK_NUM_X;
  size_t idx = THREAD_ID_X + BLOCK_NUM_X * BLOCK_ID_X;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_xsize_larger) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int x_offset = n * w + m;
        if (dx && m < w && n < h) {
          dx[x_offset] =
              dx_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
        }
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val +=
              phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dy[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int y_offset = n * w + m;
        if (dy && m < w && n < h) {
          dy[y_offset] =
              dy_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx) {
          if (m < w && n < h) {
            T val = dx_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dx) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val +=
              phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dx[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(const T *x,
                                                        const T *y,
                                                        const Tout *out,
                                                        const Tout *dout,
                                                        int pre,
                                                        int n,
                                                        int post,
                                                        bool is_xsize_larger,
                                                        DX_OP dx_op,
                                                        DY_OP dy_op,
                                                        T *dx,
                                                        T *dy) {
  int tid = THREAD_ID_X;
  int j = BLOCK_ID_X;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast1CUDA(gpuStream_t stream,
                                       const T *x,
                                       const T *y,
                                       const Tout *out,
                                       const Tout *dout,
                                       int h,
                                       int w,
                                       bool is_xsize_larger,
                                       DX_OP dx_op,
                                       DY_OP dy_op,
                                       T *dx,
                                       T *dy) {
  // For small case use 1D block
  constexpr int half_walf = 16;
  if (w < half_walf || h < half_walf) {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
    int grid_size = w;
    ElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else {
    // suppose performance improves with h increased.
    dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
    dim3 grid_size = dim3((w + BLOCK_X - 1) / BLOCK_X);
    auto gplace = phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId());
    auto *ctx = static_cast<GPUContext *>(
        phi::DeviceContextPool::Instance().Get(gplace));
    phi::backends::gpu::LimitGridDim(*ctx, &grid_size);
    FastElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast2CUDA(gpuStream_t stream,
                                       const T *x,
                                       const T *y,
                                       const Tout *out,
                                       const Tout *dout,
                                       int pre,
                                       int n,
                                       int post,
                                       bool is_xsize_larger,
                                       DX_OP dx_op,
                                       DY_OP dy_op,
                                       T *dx,
                                       T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  dim3 grid_size = dim3(n);
  auto gplace = phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId());
  auto *ctx =
      static_cast<GPUContext *>(phi::DeviceContextPool::Instance().Get(gplace));
  phi::backends::gpu::LimitGridDim(*ctx, &grid_size);
  ElemwiseGradBroadcast2CUDAKernel<<<grid_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
}

template <typename T, typename DX_OP, typename Tout = T>
__global__ void CommonGradBroadcastCUDAKernel(const int *x_strides_array,
                                              const int *y_strides_array,
                                              const int *out_dims_array,
                                              const int *y_strides_order,
                                              const int *y_dims_order,
                                              const T *x,
                                              const T *y,
                                              const Tout *out,
                                              const Tout *dout,
                                              T *dx,
                                              int out_size,
                                              int max_dim,
                                              int thread_num,
                                              DX_OP dx_op) {
  T val(0);
  int i = BLOCK_ID_X;
  int tid = THREAD_ID_X;
  for (int j = tid; j < thread_num; j += BLOCK_NUM_X) {
    const int X_index = i * thread_num + j;
    int out_index = X_index;
    int C_index = 0;
    int B_index = i * thread_num + j;
    int remainder = 0;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(B_index, y_dims_order[d], &B_index, &remainder);
      C_index += remainder * y_strides_order[d];
    }
    int x_index = 0;
    int y_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(C_index_val, out_dims_array[d], &C_index_val, &remainder);
      x_index += remainder * x_strides_array[d];
      y_index += remainder * y_strides_array[d];
    }
    out_index = C_index;
    val += dx_op(x[x_index], y[y_index], out[out_index], dout[out_index]);
  }
  val = phi::backends::gpu::reduceSum(val, tid, thread_num);
  if (THREAD_ID_X == 0) {
    dx[i] = val;
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonGradBroadcastCUDA(const DenseTensor &x,
                             const DenseTensor &y,
                             const DenseTensor &out,
                             const DenseTensor &dout,
                             DenseTensor *dx,
                             DenseTensor *dy,
                             int *x_dims_array,
                             int *y_dims_array,
                             int *out_dims_array,
                             int max_dim,
                             const GPUContext &ctx,
                             DX_OP dx_op,
                             DY_OP dy_op) {
  const auto gplace = ctx.GetPlace();
  auto cplace = phi::CPUPlace();
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const Tout *out_data = out.data<Tout>();
  const Tout *dout_data = dout.data<Tout>();
  T *dx_data = dx == nullptr ? nullptr : ctx.Alloc<T>(dx);
  T *dy_data = dy == nullptr ? nullptr : ctx.Alloc<T>(dy);

  std::vector<int> x_one_indices;
  std::vector<int> y_one_indices;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] != y_dims_array[i]) {
      if (x_dims_array[i] == 1) {
        x_one_indices.push_back(i);
      }
      if (y_dims_array[i] == 1) {
        y_one_indices.push_back(i);
      }
    }
  }

  std::vector<int> x_trans_indices(max_dim);
  std::vector<int> y_trans_indices(max_dim);
  ComputeBroadcastTranspositionArray(x_one_indices.data(),
                                     x_trans_indices.data(),
                                     max_dim,
                                     x_one_indices.size());
  ComputeBroadcastTranspositionArray(y_one_indices.data(),
                                     y_trans_indices.data(),
                                     max_dim,
                                     y_one_indices.size());

  // compute array stride for cuda kernel;
  // e.g. x.dims=[2,3,4], x_stride=[12,4,1]
  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  std::vector<int> out_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  int z_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    out_strides_array[i] = z_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
    z_stride *= out_dims_array[i];
  }

  std::vector<int> x_strides_order(max_dim);
  std::vector<int> y_strides_order(max_dim);
  std::vector<int> x_dims_order(max_dim);
  std::vector<int> y_dims_order(max_dim);
  for (int i = 0; i < max_dim; ++i) {
    x_strides_order[i] = out_strides_array[x_trans_indices[i]];
    y_strides_order[i] = out_strides_array[y_trans_indices[i]];
    x_dims_order[i] = out_dims_array[x_trans_indices[i]];
    y_dims_order[i] = out_dims_array[y_trans_indices[i]];
  }
  std::vector<int> x_broadcast_pos;
  std::vector<int> y_broadcast_pos;

  int bytes = max_dim * sizeof(int);

  for (int i = 0; i < max_dim; ++i) {
    if (x_dims_array[i] != out_dims_array[i] && x_dims_array[i] == 1) {
      x_broadcast_pos.emplace_back(i);
    }
    if (y_dims_array[i] != out_dims_array[i] && y_dims_array[i] == 1) {
      y_broadcast_pos.emplace_back(i);
    }
  }

  auto stream = ctx.stream();
  bool can_split_x = false;
  bool can_split_y = false;

  auto FastCommonCUDAF = [&](const std::vector<int> &broadcast_pos, bool is_y) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastCommonCUDAF elementwise w:" << w << " h:" << h
            << " is_y:" << is_y;

    int split_h;
    int split_w;
    int kh = h;
    int kw = w;

    if (is_y) {
      split_h = std::accumulate(x_dims_array,
                                x_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(x_dims_array + broadcast_pos.size(),
                                x_dims_array + max_dim,
                                1,
                                std::multiplies<int>());

    } else {
      split_h = std::accumulate(y_dims_array,
                                y_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(y_dims_array + broadcast_pos.size(),
                                y_dims_array + max_dim,
                                1,
                                std::multiplies<int>());
    }

    if (h > split_h) kh = split_h;
    if (w > split_w) kw = split_w;

    if (is_y) {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                               block_size,
                                               0,
                                               stream>>>(x_data,
                                                         y_data,
                                                         out_data,
                                                         dout_data,
                                                         h,
                                                         w,
                                                         dy_op,
                                                         dy_data,
                                                         kh,
                                                         kw,
                                                         is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        dim3 grid_size = dim3((w + BLOCK_X - 1) / BLOCK_X);
        phi::backends::gpu::LimitGridDim(ctx, &grid_size);
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            h,
                                                            w,
                                                            dy_op,
                                                            dy_data,
                                                            kh,
                                                            kw,
                                                            is_y);
      }
    } else {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                               block_size,
                                               0,
                                               stream>>>(x_data,
                                                         y_data,
                                                         out_data,
                                                         dout_data,
                                                         h,
                                                         w,
                                                         dx_op,
                                                         dx_data,
                                                         kh,
                                                         kw,
                                                         is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        dim3 grid_size = dim3((w + BLOCK_X - 1) / BLOCK_X);
        phi::backends::gpu::LimitGridDim(ctx, &grid_size);
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            h,
                                                            w,
                                                            dx_op,
                                                            dx_data,
                                                            kh,
                                                            kw,
                                                            is_y);
      }
    }
  };

  auto FastBroadCastHeightCUDAF = [&](const std::vector<int> &broadcast_pos,
                                      bool x_large) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastBroadCastHeightCUDAF w:" << w << " h:" << h;

    if (w < 16 || h < 16) {
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
      int grid_size = w;
      ElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
          x_data,
          y_data,
          out_data,
          dout_data,
          h,
          w,
          x_large,
          dx_op,
          dy_op,
          dx_data,
          dy_data);
    } else {
      dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
      int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
      FastElemwiseGradBroadcast1CUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       h,
                                                       w,
                                                       x_large,
                                                       dx_op,
                                                       dy_op,
                                                       dx_data,
                                                       dy_data);
    }
  };

  auto FastBroadCastAllCUDAF = [&](const std::vector<int> &broadcast_pos,
                                   int max_dim,
                                   bool is_x_large) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = 1;
    int post = 1;

    if (broadcast_pos.size() == 1) {
      mid = out_dims_array[axis];
      post = std::accumulate(out_dims_array + axis + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    } else {
      mid = std::accumulate(out_dims_array + axis,
                            out_dims_array + broadcast_pos.back() + 1,
                            1,
                            std::multiplies<int>());
      post = std::accumulate(out_dims_array + broadcast_pos.back() + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    }

    VLOG(3) << "FastBroadCastAllCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;

    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
    dim3 grid_size = dim3(pre * post);
    phi::backends::gpu::LimitGridDim(ctx, &grid_size);

    FastCommonGradBroadcastAllCUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x_data,
        y_data,
        out_data,
        dout_data,
        pre,
        mid,
        post,
        is_x_large,
        dx_op,
        dy_op,
        dx_data,
        dy_data);
  };

  auto FastBroadCastOneCUDAF =
      [&](const std::vector<int> &broadcast_pos, int max_dim, bool is_x) {
        int axis = broadcast_pos[0];
        int pre = std::accumulate(
            out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
        int mid = out_dims_array[axis];
        int post = std::accumulate(out_dims_array + axis + 1,
                                   out_dims_array + max_dim,
                                   1,
                                   std::multiplies<int>());

        int k_pre;
        int k_mid;
        int k_post;

        if (is_x) {
          k_pre = std::accumulate(
              y_dims_array, y_dims_array + axis, 1, std::multiplies<int>());
          k_mid = y_dims_array[axis];
          k_post = std::accumulate(y_dims_array + axis + 1,
                                   y_dims_array + max_dim,
                                   1,
                                   std::multiplies<int>());
          int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
          dim3 grid_size = dim3(pre * post);
          phi::backends::gpu::LimitGridDim(ctx, &grid_size);
          // we need to calc y offset with blockid, so do x_pre/y_pre to get
          // left size.
          if (k_pre != pre) k_pre = pre / k_pre;

          FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                                 block_size,
                                                 0,
                                                 stream>>>(x_data,
                                                           y_data,
                                                           out_data,
                                                           dout_data,
                                                           pre,
                                                           mid,
                                                           post,
                                                           k_pre,
                                                           k_mid,
                                                           k_post,
                                                           true,
                                                           dx_op,
                                                           dx_data);
        } else {
          k_pre = std::accumulate(
              x_dims_array, x_dims_array + axis, 1, std::multiplies<int>());
          k_mid = x_dims_array[axis];
          k_post = std::accumulate(x_dims_array + axis + 1,
                                   x_dims_array + max_dim,
                                   1,
                                   std::multiplies<int>());
          int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
          dim3 grid_size = dim3(pre * post);
          phi::backends::gpu::LimitGridDim(ctx, &grid_size);
          if (k_pre != pre) k_pre = pre / k_pre;

          FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                                 block_size,
                                                 0,
                                                 stream>>>(x_data,
                                                           y_data,
                                                           out_data,
                                                           dout_data,
                                                           pre,
                                                           mid,
                                                           post,
                                                           k_pre,
                                                           k_mid,
                                                           k_post,
                                                           false,
                                                           dy_op,
                                                           dy_data);
        }
        VLOG(3) << "FastBroadCastOneCUDAF pre:" << pre << " mid:" << mid
                << " post:" << post;
      };

  // do fast elementwise if: 1. only one input need to do broadcast, we can
  // fallback
  // to old fast path.
  // 2. if both x and y need broadcast, then do it one by one.
  bool fast_broadcast = false;
  if (x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    if (can_split_y) {
      // only y need to do broadcast on h
      if (y_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(y_broadcast_pos, true);
        fast_broadcast = true;
      }
    } else if (y_broadcast_pos.size() == 1 ||
               CheckContiguousDims(y_broadcast_pos)) {  // for only one dim and
                                                        // contiguous broadcast.
      // If cannot split,  which means input has 3 parts
      FastBroadCastAllCUDAF(y_broadcast_pos, max_dim, true);
      fast_broadcast = true;
    }
  } else if (y_broadcast_pos.empty() && !x_broadcast_pos.empty()) {
    // only x need broadcast
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(x_broadcast_pos, false);
        fast_broadcast = true;
      }
    } else if (x_broadcast_pos.size() == 1 ||
               CheckContiguousDims(x_broadcast_pos)) {
      FastBroadCastAllCUDAF(x_broadcast_pos, max_dim, false);
      fast_broadcast = true;
    }
  } else if (!x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    // do x and y broadcast each.
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    bool fast_broadcast_x = false;
    bool fast_broadcast_y = false;
    if (can_split_y) {
      // begin at start.
      if (y_broadcast_pos[0] == 0) {
        FastCommonCUDAF(y_broadcast_pos, true);
        fast_broadcast_y = true;
      }
    } else if (y_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(y_broadcast_pos, max_dim, false);
      can_split_y = true;
      fast_broadcast_y = true;
    }
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastCommonCUDAF(x_broadcast_pos, false);
        fast_broadcast_x = true;
      }
    } else if (x_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(x_broadcast_pos, max_dim, true);
      can_split_x = true;
      fast_broadcast_x = true;
    }
    VLOG(3) << "CommonBroadcast can_split_y:" << can_split_y
            << " can_split_x:" << can_split_x;
    // if both x and y into fast path then return

    //* It's possible that some bugs are a result of early returns, comment out
    // the code for checking.
    // if (fast_broadcast_x && fast_broadcast_y) {
    //   fast_broadcast = true;
    // }
    // if (can_split_y && can_split_x && fast_broadcast) return;
  }

  // Should remove memory copy, use reg instead.
  if (fast_broadcast) {
    return;
  }
  int x_blocks = 0;
  int x_threads = 0;
  ComputeBroadcastKernelSize(
      x_dims_array, out_dims_array, &x_blocks, &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  ComputeBroadcastKernelSize(
      y_dims_array, out_dims_array, &y_blocks, &y_threads, max_dim);

  // One part buffer for x_strides_array, rest for y_strides_array and
  // out_dims_array.
  size_t tmp_total_bytes = bytes * 3;
  auto tmp_buffer = phi::memory_utils::Alloc(
      ctx.GetPlace(),
      tmp_total_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  int *x_strides_array_gpu = reinterpret_cast<int *>(tmp_buffer->ptr());
  int *y_strides_array_gpu =
      reinterpret_cast<int *>(x_strides_array_gpu + max_dim);
  int *out_dims_array_gpu =
      reinterpret_cast<int *>(y_strides_array_gpu + max_dim);

  memory_utils::Copy(gplace,
                     x_strides_array_gpu,
                     cplace,
                     x_strides_array.data(),
                     bytes,
                     ctx.stream());
  memory_utils::Copy(gplace,
                     y_strides_array_gpu,
                     cplace,
                     y_strides_array.data(),
                     bytes,
                     ctx.stream());
  memory_utils::Copy(
      gplace, out_dims_array_gpu, cplace, out_dims_array, bytes, ctx.stream());

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, x_threads);
  int y_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, y_threads);
  if (dx) {
    size_t dx_total_bytes = bytes * 2;
    auto dx_tmp_buffer = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        dx_total_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    int *x_strides_order_gpu = reinterpret_cast<int *>(dx_tmp_buffer->ptr());
    int *x_dims_order_gpu =
        reinterpret_cast<int *>(x_strides_order_gpu + max_dim);

    memory_utils::Copy(gplace,
                       x_strides_order_gpu,
                       cplace,
                       x_strides_order.data(),
                       bytes,
                       ctx.stream());
    memory_utils::Copy(gplace,
                       x_dims_order_gpu,
                       cplace,
                       x_dims_order.data(),
                       bytes,
                       ctx.stream());
    CommonGradBroadcastCUDAKernel<T, DX_OP, Tout>
        <<<x_blocks, x_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                      y_strides_array_gpu,
                                                      out_dims_array_gpu,
                                                      x_strides_order_gpu,
                                                      x_dims_order_gpu,
                                                      x_data,
                                                      y_data,
                                                      out_data,
                                                      dout_data,
                                                      dx_data,
                                                      out_size,
                                                      max_dim,
                                                      x_threads,
                                                      dx_op);
  }
  if (dy) {
    // One part buffer for y_strides_order_gpu, the other for y_dims_order_gpu
    size_t dy_total_bytes = bytes * 2;
    auto dy_tmp_buffer = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        dy_total_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    int *y_strides_order_gpu = reinterpret_cast<int *>(dy_tmp_buffer->ptr());
    int *y_dims_order_gpu =
        reinterpret_cast<int *>(y_strides_order_gpu + max_dim);

    memory_utils::Copy(gplace,
                       y_strides_order_gpu,
                       cplace,
                       y_strides_order.data(),
                       bytes,
                       ctx.stream());
    memory_utils::Copy(gplace,
                       y_dims_order_gpu,
                       cplace,
                       y_dims_order.data(),
                       bytes,
                       ctx.stream());
    CommonGradBroadcastCUDAKernel<T, DY_OP, Tout>
        <<<y_blocks, y_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                      y_strides_array_gpu,
                                                      out_dims_array_gpu,
                                                      y_strides_order_gpu,
                                                      y_dims_order_gpu,
                                                      x_data,
                                                      y_data,
                                                      out_data,
                                                      dout_data,
                                                      dy_data,
                                                      out_size,
                                                      max_dim,
                                                      y_threads,
                                                      dy_op);
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonElementwiseBroadcastBackward(const GPUContext &ctx,
                                        const DDim &x_dims,
                                        const DDim &y_dims,
                                        const DenseTensor &x,
                                        const DenseTensor &y,
                                        const DenseTensor &out,
                                        const DenseTensor &dout,
                                        int axis,
                                        DenseTensor *dx,
                                        DenseTensor *dy,
                                        DX_OP dx_op,
                                        DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->Resize(x_dims);
    ctx.template Alloc<T>(dx);
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << common::make_ddim(x_dims_array)
          << " ydim:" << common::make_ddim(y_dims_array);

  CommonGradBroadcastCUDA<T, DX_OP, DY_OP, Tout>(x,
                                                 y,
                                                 out,
                                                 dout,
                                                 dx,
                                                 dy,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 ctx,
                                                 dx_op,
                                                 dy_op);
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void ElemwiseGradComputeWithBroadcast(const GPUContext &ctx,
                                      const DDim &x_dims,
                                      const DDim &y_dims,
                                      const DenseTensor &x,
                                      const DenseTensor &y,
                                      const DenseTensor &out,
                                      const DenseTensor &dout,
                                      int axis,
                                      DenseTensor *dx,
                                      DenseTensor *dy,
                                      DX_OP dx_op,
                                      DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimmed = TrimTrailingSingularDims(y_dims);
    axis_trim = (y_dims_trimmed.size() == 0) ? x_dims.size() : axis;
    GetMidDims(x_dims,
               y_dims_trimmed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  } else {
    auto x_dims_trimmed = TrimTrailingSingularDims(x_dims);
    axis_trim = (x_dims_trimmed.size() == 0) ? y_dims.size() : axis;
    GetMidDims(y_dims,
               x_dims_trimmed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<T, DX_OP, DY_OP, Tout>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    ElemwiseGradBroadcast1CUDA(ctx.stream(),
                               x.data<T>(),
                               y.data<T>(),
                               out.data<Tout>(),
                               dout.data<Tout>(),
                               pre,
                               n,
                               is_xsize_larger,
                               dx_op,
                               dy_op,
                               dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                               dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  } else {
    ElemwiseGradBroadcast2CUDA(ctx.stream(),
                               x.data<T>(),
                               y.data<T>(),
                               out.data<Tout>(),
                               dout.data<Tout>(),
                               pre,
                               n,
                               post,
                               is_xsize_larger,
                               dx_op,
                               dy_op,
                               dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                               dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  }
}

#endif

template <typename DeviceContext,
          typename T,
          typename DX_OP,
          typename DY_OP,
          typename Tout = T>
void ElemwiseGradCompute(const DeviceContext &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor &y,
                         const DenseTensor &out,
                         const DenseTensor &dout,
                         int axis,
                         DenseTensor *dx,
                         DenseTensor *dy,
                         DX_OP dx_op,
                         DY_OP dy_op) {
  const DDim &x_dim = x.dims();
  const DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP, Tout>(
        dev_ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<T, DX_OP, DY_OP, Tout>(
        dev_ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  }
}

}  // namespace funcs
}  // namespace phi
