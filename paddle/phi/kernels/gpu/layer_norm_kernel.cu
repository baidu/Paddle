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

#include "paddle/phi/kernels/layer_norm_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"

COMMON_DECLARE_bool(use_fast_math);

namespace phi {

#ifdef PADDLE_WITH_CUDA
template <typename U>
__device__ inline void WelfordOnline(U val, U *mean, U *square, U *count) {
  *count += 1;
  U delta1 = val - *mean;
  *mean += delta1 / (*count);
  U delta2 = val - *mean;
  *square += delta1 * delta2;
}

template <typename U>
__device__ inline void WelfordOnline(
    U b_mean, U b_square, U b_cnt, U *mean, U *square, U *count) {
  if (b_cnt == 0) {
    return;
  }

  U new_cnt = *count + b_cnt;
  U nb_n = b_cnt / new_cnt;
  U delta = b_mean - *mean;
  *mean += delta * nb_n;
  *square += b_square + delta * delta * (*count) * nb_n;
  *count = new_cnt;
}

template <typename U>
__device__ inline void WelfordWarpAllReduce(U *mean, U *square, U *count) {
  constexpr int kWarpSize = 32;
#pragma unroll
  for (int mask = 1; mask < kWarpSize; mask *= 2) {
    U b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    U b_square = __shfl_down_sync(0xffffffff, *square, mask);
    U b_cnt = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordOnline<U>(b_mean, b_square, b_cnt, mean, square, count);
  }

  *mean = __shfl_sync(0xffffffff, *mean, 0, kWarpSize);
  *square = __shfl_sync(0xffffffff, *square, 0, kWarpSize);
  *count = __shfl_sync(0xffffffff, *count, 0, kWarpSize);
}

template <int VecSize>
struct ThreadAssigner {
  __device__ __forceinline__ int operator()(const int cols,
                                            const int cols_per_thread,
                                            int32_t *last_tid_idx) {
    return cols_per_thread;
  }
};

template <>
struct ThreadAssigner<1> {
  __device__ inline int operator()(const int cols,
                                   const int cols_per_thread,
                                   int *last_tid_idx) {
    int cols_this_thread = cols_per_thread;
    int last_tid = (cols / cols_per_thread);
    *last_tid_idx = last_tid;
    if (threadIdx.x == last_tid) {
      cols_this_thread = cols - cols_per_thread * last_tid;
    } else if (threadIdx.x > last_tid) {
      cols_this_thread = 0;
    }
    return cols_this_thread;
  }
};

template <typename T, typename U, int VecSize>
struct LayerNormDataReader {
  __device__ inline void operator()(const T *__restrict__ row_src,
                                    U *buffer,
                                    const int last_tid_idx,
                                    const int read_times,
                                    const int cols_this_thread) {
    using VecT = phi::AlignedVector<T, VecSize>;
    const VecT *__restrict__ v_src =
        reinterpret_cast<const VecT *__restrict__>(row_src);

    for (int i = 0; i < read_times; ++i) {
      VecT temp_src = v_src[threadIdx.x + i * blockDim.x];
#pragma unroll
      for (int j = 0; j < VecSize; ++j) {
        buffer[i * VecSize + j] = static_cast<U>(temp_src[j]);
      }
    }
  }
};

template <typename T, typename U>
struct LayerNormDataReader<T, U, 1> {
  __device__ inline void operator()(const T *__restrict__ row_src,
                                    U *buffer,
                                    const int last_tid_idx,
                                    const int read_times,
                                    const int cols_this_thread) {
    // read_time is just cols_per_thread while VecSize is 1.
    if (threadIdx.x < last_tid_idx) {
      for (int i = 0; i < cols_this_thread; ++i) {
        buffer[i] = static_cast<U>(row_src[threadIdx.x + last_tid_idx * i]);
      }
    } else {
      for (int i = 0; i < cols_this_thread; ++i) {
        buffer[i] = static_cast<U>(row_src[i + read_times * last_tid_idx]);
      }
    }
  }
};

template <typename T, typename U, bool IsSameType, int VecSize>
struct LayerNormDataWriter {
  __device__ inline void operator()(
      T *__restrict__ row_dst,
      const U *__restrict__ buffer,
      const funcs::LayerNormScaleBiasT<T, U, IsSameType> *__restrict__ scale,
      const funcs::LayerNormScaleBiasT<T, U, IsSameType> *__restrict__ bias,
      const U row_mean,
      const U row_inv_var,
      const int write_times,
      const int cols_this_thread,
      const int last_tid_idx,
      const bool valid_scale,
      const bool valid_bias) {
    using VecT = phi::AlignedVector<T, VecSize>;
    using ScaleT = funcs::LayerNormScaleBiasT<T, U, IsSameType>;
    using VecScaleT = phi::AlignedVector<ScaleT, VecSize>;
    VecT *v_dst = reinterpret_cast<VecT *>(row_dst);

    // cols_this_thread is just cols_per_thread
    if ((!valid_scale) && (!valid_bias)) {
      for (int i = 0; i < write_times; ++i) {
        VecT temp_dst;
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
          temp_dst[j] = static_cast<T>((buffer[i * VecSize + j] - row_mean) *
                                       row_inv_var);
        }
        v_dst[threadIdx.x + blockDim.x * i] = temp_dst;
      }
    } else {
      const VecScaleT *__restrict__ v_scale =
          reinterpret_cast<const VecScaleT *__restrict__>(scale);
      const VecScaleT *__restrict__ v_bias =
          reinterpret_cast<const VecScaleT *__restrict__>(bias);
      if (valid_scale && valid_bias) {
        for (int i = 0; i < write_times; ++i) {
          int idx = threadIdx.x + blockDim.x * i;
          VecT temp_dst;
          VecScaleT temp_v_scale = v_scale[idx];
          VecScaleT temp_v_bias = v_bias[idx];
#pragma unroll
          for (int j = 0; j < VecSize; ++j) {
            temp_dst[j] = static_cast<T>(
                static_cast<U>(temp_v_scale[j]) *
                    (buffer[i * VecSize + j] - row_mean) * row_inv_var +
                static_cast<U>(temp_v_bias[j]));
          }
          v_dst[idx] = temp_dst;
        }
      } else {
        if (valid_scale) {
          for (int i = 0; i < write_times; ++i) {
            int idx = threadIdx.x + blockDim.x * i;
            VecT temp_dst;
            VecScaleT temp_v_scale = v_scale[idx];
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
              temp_dst[j] = static_cast<T>(
                  static_cast<U>(temp_v_scale[j]) *
                  (buffer[i * VecSize + j] - row_mean) * row_inv_var);
            }
            v_dst[idx] = temp_dst;
          }
        } else {
          for (int i = 0; i < write_times; ++i) {
            int idx = threadIdx.x + blockDim.x * i;
            VecT temp_dst;
            VecScaleT temp_v_bias = v_bias[idx];
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
              temp_dst[j] = static_cast<T>(
                  (buffer[i * VecSize + j] - row_mean) * row_inv_var +
                  static_cast<U>(temp_v_bias[j]));
            }
            v_dst[idx] = temp_dst;
          }
        }
      }
    }
  }
};

template <typename T, typename U, bool IsSameType>
struct LayerNormDataWriter<T, U, IsSameType, 1> {
  __device__ __forceinline__ void operator()(
      T *__restrict__ row_dst,
      U *__restrict__ buffer,
      const funcs::LayerNormScaleBiasT<T, U, IsSameType> *__restrict__ scale,
      const funcs::LayerNormScaleBiasT<T, U, IsSameType> *__restrict__ bias,
      const U row_mean,
      const U row_inv_var,
      const int write_times,
      const int cols_this_thread,
      const int last_tid_idx,
      const bool valid_scale,
      const bool valid_bias) {
    // write_times is just col_per_thread.
    if ((!valid_scale) && (!valid_bias)) {
      if (threadIdx.x < last_tid_idx) {
        for (int i = 0; i < cols_this_thread; ++i) {
          row_dst[threadIdx.x + last_tid_idx * i] =
              (buffer[i] - row_mean) * row_inv_var;
        }
      } else {
        for (int i = 0; i < cols_this_thread; ++i) {
          row_dst[last_tid_idx * write_times + i] =
              (buffer[i] - row_mean) * row_inv_var;
        }
      }
    } else if (valid_scale && valid_bias) {
      if (threadIdx.x < last_tid_idx) {
        for (int i = 0; i < cols_this_thread; ++i) {
          int idx = threadIdx.x + last_tid_idx * i;
          row_dst[idx] =
              static_cast<T>(static_cast<U>(scale[idx]) *
                                 (buffer[i] - row_mean) * row_inv_var +
                             static_cast<U>(bias[idx]));
        }
      } else {
        for (int i = 0; i < cols_this_thread; ++i) {
          int idx = last_tid_idx * write_times + i;
          row_dst[idx] =
              static_cast<T>(static_cast<U>(scale[idx]) *
                                 (buffer[i] - row_mean) * row_inv_var +
                             static_cast<U>(bias[idx]));
        }
      }
    } else {
      if (valid_scale) {
        if (threadIdx.x < last_tid_idx) {
          for (int i = 0; i < cols_this_thread; ++i) {
            int idx = threadIdx.x + last_tid_idx * i;
            row_dst[idx] = static_cast<T>(static_cast<U>(scale[idx]) *
                                          (buffer[i] - row_mean) * row_inv_var);
          }
        } else {
          for (int i = 0; i < cols_this_thread; ++i) {
            int idx = last_tid_idx * write_times + i;
            row_dst[idx] = static_cast<T>(static_cast<U>(scale[idx]) *
                                          (buffer[i] - row_mean) * row_inv_var);
          }
        }
      } else {
        if (threadIdx.x < last_tid_idx) {
          for (int i = 0; i < cols_this_thread; ++i) {
            int idx = threadIdx.x + last_tid_idx * i;
            row_dst[idx] = static_cast<T>((buffer[i] - row_mean) * row_inv_var +
                                          static_cast<U>(bias[idx]));
          }
        } else {
          for (int i = 0; i < cols_this_thread; ++i) {
            int idx = last_tid_idx * write_times + i;
            row_dst[idx] = static_cast<T>((buffer[i] - row_mean) * row_inv_var +
                                          static_cast<U>(bias[idx]));
          }
        }
      }
    }
  }
};

template <typename IndexT, typename T, typename U, bool IsSameType, int VecSize>
__global__ void LayerNormFwdWithWelford(
    const T *__restrict__ src_data,
    T *dst_data,
    const funcs::LayerNormScaleBiasT<T, U, IsSameType> *__restrict__ scale,
    const funcs::LayerNormScaleBiasT<T, U, IsSameType> *__restrict__ bias,
    U *mean,
    U *var,
    const U epsilon,
    const IndexT rows,
    const int32_t cols,
    const int32_t cols_per_thread,
    const bool valid_scale,
    const bool valid_bias) {
  constexpr int kWarpSize = 32;
  int last_tid_idx = 0;  // For condition once vecSize is 1.
  IndexT row_offset = blockIdx.x * blockDim.y + threadIdx.y;
  int cols_this_thread =
      ThreadAssigner<VecSize>()(cols, cols_per_thread, &last_tid_idx);
  int read_times = cols_per_thread / VecSize;

  if (row_offset < rows) {
    U buffer[kWarpSize];
    U tid_cnt = static_cast<U>(0);
    U tid_mean = static_cast<U>(0);
    U tid_square = static_cast<U>(0);

    const T *__restrict__ row_src = src_data + row_offset * cols;
    T *row_dst = dst_data + row_offset * cols;
    LayerNormDataReader<T, U, VecSize>()(
        row_src, buffer, last_tid_idx, read_times, cols_this_thread);

    for (int i = 0; i < cols_this_thread; i++) {
      WelfordOnline<U>(buffer[i], &tid_mean, &tid_square, &tid_cnt);
    }

    U warp_cnt = tid_cnt;
    U warp_mean = tid_mean;
    U warp_square = tid_square;
    WelfordWarpAllReduce<U>(&warp_mean, &warp_square, &warp_cnt);

    U row_variance = max(warp_square / warp_cnt, 0.f);
    U row_inv_var = funcs::rsqrt_(row_variance + epsilon);

    // TODO(limingshu): make code below vectorization.
    if (threadIdx.x == 0) {
      // warp_mean is just row_mean here.
      mean[row_offset] = warp_mean;
      var[row_offset] = row_variance;
    }
    LayerNormDataWriter<T, U, IsSameType, VecSize>()(row_dst,
                                                     buffer,
                                                     scale,
                                                     bias,
                                                     warp_mean,
                                                     row_inv_var,
                                                     read_times,
                                                     cols_this_thread,
                                                     last_tid_idx,
                                                     valid_scale,
                                                     valid_bias);
  }
}

template <typename Context, typename T, typename U>
void LaunchLayerNormKernel(const Context &dev_ctx,
                           const T *x_data,
                           T *y_data,
                           const void *void_scale_data,
                           const void *void_bias_data,
                           U *mean_data,
                           U *var_data,
                           float epsilon,
                           const int64_t rows,
                           const int cols,
                           const bool valid_scale,
                           const bool valid_bias,
                           const bool is_same_type) {
  constexpr int WarpSize = 32;
  constexpr int RowPerBlock = 4;
  int64_t block_size = (rows + (RowPerBlock - 1)) / RowPerBlock;
  dim3 threads(WarpSize, RowPerBlock, 1);

  int vec_size = 1;
  int cols_per_thread = (cols + (WarpSize - 1)) / WarpSize;
  if (cols_per_thread > 1 && (cols % WarpSize == 0)) {
    int data_vec_size = 0;
    uint64_t addr = (reinterpret_cast<uint64_t>(x_data) |
                     reinterpret_cast<uint64_t>(y_data));
    if (valid_bias || valid_scale) {
      if (is_same_type) {
        addr = valid_scale
                   ? (addr | reinterpret_cast<uint64_t>(void_scale_data))
                   : addr;
        addr = valid_bias ? (addr | reinterpret_cast<uint64_t>(void_bias_data))
                          : addr;
        data_vec_size = phi::GetVectorizedSize<T>(reinterpret_cast<T *>(addr));
      } else {
        uint64_t bias_addr = reinterpret_cast<uint64_t>(void_bias_data);
        uint64_t attr_addr = valid_scale
                                 ? reinterpret_cast<uint64_t>(void_scale_data)
                                 : bias_addr;
        attr_addr = valid_bias
                        ? (valid_scale ? (attr_addr | bias_addr) : attr_addr)
                        : attr_addr;
        data_vec_size = std::min(
            phi::GetVectorizedSize<T>(reinterpret_cast<T *>(addr)),
            phi::GetVectorizedSize<U>(reinterpret_cast<U *>(attr_addr)));
      }
    }
    for (int size = data_vec_size; size > 0; size /= 2) {
      if (cols_per_thread % size == 0) {
        vec_size = size;
        break;
      }
    }
  }

#define IMPL_LAYER_NORM_WELFORD_CASE(index_t, scale_t, is_same_, vec_size_) \
  case (vec_size_): {                                                       \
    LayerNormFwdWithWelford<index_t, T, U, is_same_, vec_size_>             \
        <<<block_size, threads, 0, dev_ctx.stream()>>>(                     \
            x_data,                                                         \
            y_data,                                                         \
            static_cast<const scale_t *>(void_scale_data),                  \
            static_cast<const scale_t *>(void_bias_data),                   \
            mean_data,                                                      \
            var_data,                                                       \
            static_cast<const U>(epsilon),                                  \
            rows,                                                           \
            cols,                                                           \
            cols_per_thread,                                                \
            valid_scale,                                                    \
            valid_bias);                                                    \
  } break

#define IMPL_LAYER_NORM_WELFORD(index_t, scale_t, is_same_)    \
  IMPL_LAYER_NORM_WELFORD_CASE(index_t, scale_t, is_same_, 4); \
  IMPL_LAYER_NORM_WELFORD_CASE(index_t, scale_t, is_same_, 2); \
  IMPL_LAYER_NORM_WELFORD_CASE(index_t, scale_t, is_same_, 1);

  if (rows < std::numeric_limits<int32_t>::max()) {
    if (is_same_type) {
      switch (vec_size) { IMPL_LAYER_NORM_WELFORD(int32_t, T, true); }
    } else {
      switch (vec_size) { IMPL_LAYER_NORM_WELFORD(int32_t, U, false); }
    }
  } else {
    if (is_same_type) {
      switch (vec_size) { IMPL_LAYER_NORM_WELFORD(int64_t, T, true); }
    } else {
      switch (vec_size) { IMPL_LAYER_NORM_WELFORD(int64_t, U, false); }
    }
  }
#undef IMPL_LAYER_NORM_WELFORD_CASE
#undef IMPL_LAYER_NORM_WELFORD
}
#endif  // PADDLE_WITH_CUDA

template <typename T, typename U>
void LayerNormDirectCUDAFunctor<T, U>::operator()(
    gpuStream_t stream,
    const T *input,
    std::vector<int64_t> input_shape,
    const U *bias,
    const U *scale,
    T *output,
    U *mean,
    U *variance,
    int begin_norm_axis,
    float eps) {
  const auto x_dims = common::make_ddim(input_shape);
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);
  switch (phi::funcs::GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        phi::funcs::LayerNormForward<T, U, kBlockDim>
        <<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

template class LayerNormDirectCUDAFunctor<float, float>;
template class LayerNormDirectCUDAFunctor<double, double>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class LayerNormDirectCUDAFunctor<half, float>;
#endif

template <typename T, typename Context>
void LayerNormKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const paddle::optional<DenseTensor> &scale_opt,
                     const paddle::optional<DenseTensor> &bias_opt,
                     float epsilon,
                     int begin_norm_axis,
                     DenseTensor *y,
                     DenseTensor *mean,
                     DenseTensor *var) {
  using U = phi::funcs::LayerNormParamType<T>;
  auto *scale = scale_opt.get_ptr();
  auto *bias = bias_opt.get_ptr();

  const auto x_dims = x.dims();
  auto *x_data = x.data<T>();
  auto *y_data = dev_ctx.template Alloc<T>(y);
  auto *mean_data = dev_ctx.template Alloc<U>(mean);
  auto *var_data = dev_ctx.template Alloc<U>(var);

  bool valid_scale = (scale != nullptr);
  bool valid_bias = (bias != nullptr);
  auto *void_scale_data = valid_scale ? scale->data() : nullptr;
  auto *void_bias_data = valid_bias ? bias->data() : nullptr;

  auto x_dtype = x.dtype();
  phi::DataType scale_bias_dtype;
  if (valid_scale) {
    scale_bias_dtype = scale->dtype();
    if (valid_bias) {
      PADDLE_ENFORCE_EQ(scale->dtype(),
                        bias->dtype(),
                        common::errors::InvalidArgument(
                            "This Scale and Bias of layer_norm op "
                            "should have the same data type."));
    }
  } else {
    scale_bias_dtype = valid_bias ? bias->dtype() : x_dtype;
  }

  bool is_scale_bias_same_dtype_with_x = x_dtype == scale_bias_dtype;
  if (!is_scale_bias_same_dtype_with_x) {
    PADDLE_ENFORCE_EQ(scale_bias_dtype,
                      phi::CppTypeToDataType<U>::Type(),
                      common::errors::InvalidArgument(
                          "Unsupported data type of Scale and Bias"));
  }

  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);
  auto stream = dev_ctx.stream();

#define PADDLE_LAUNCH_LAYERNORM_FWD(ScaleBiasT, IsScaleBiasSameDTypeWithX) \
  do {                                                                     \
    switch (phi::funcs::GetDesiredBlockDim(feature_size)) {                \
      FIXED_BLOCK_DIM_CASE(                                                \
          phi::funcs::                                                     \
              LayerNormForward<T, U, kBlockDim, IsScaleBiasSameDTypeWithX> \
          <<<batch_size, kBlockDim, 0, stream>>>(                          \
              x_data,                                                      \
              static_cast<const ScaleBiasT *>(void_scale_data),            \
              static_cast<const ScaleBiasT *>(void_bias_data),             \
              y_data,                                                      \
              mean_data,                                                   \
              var_data,                                                    \
              epsilon,                                                     \
              feature_size));                                              \
      default:                                                             \
        PADDLE_THROW(common::errors::InvalidArgument(                      \
            "Product from begin_norm_axis to end must be larger than 1")); \
        break;                                                             \
    }                                                                      \
  } while (0)

#define PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, feature_size)          \
  case (feature_size): {                                                     \
    constexpr int WARPS_N = feature_size < 1024 ? 1 : (feature_size / 1024); \
    constexpr int WARPS_M = 4 / WARPS_N;                                     \
    const int THREADS_PER_WARP = 32;                                         \
    const int BYTES_PER_LDG = 16;                                            \
    const int VecSize = BYTES_PER_LDG / sizeof(T);                           \
    const int THREADS_PER_CTA = WARPS_N * THREADS_PER_WARP * WARPS_M;        \
    const int ROWS_PER_CTA = WARPS_M;                                        \
    const int grid = static_cast<int>(                                       \
        std::ceil(batch_size / static_cast<float>(ROWS_PER_CTA)));           \
    phi::funcs::fast_ln_fwd_kernel<T,                                        \
                                   U,                                        \
                                   ScaleT,                                   \
                                   VecSize,                                  \
                                   WARPS_M,                                  \
                                   WARPS_N,                                  \
                                   BYTES_PER_LDG,                            \
                                   feature_size>                             \
        <<<grid, THREADS_PER_CTA, 0, stream>>>(                              \
            batch_size,                                                      \
            feature_size,                                                    \
            epsilon,                                                         \
            x_data,                                                          \
            static_cast<const ScaleT *>(void_scale_data),                    \
            static_cast<const ScaleT *>(void_bias_data),                     \
            mean_data,                                                       \
            var_data,                                                        \
            y_data);                                                         \
  } break

#define PADDLE_LAUNCH_FAST_LAYERNORM_FWD(ScaleT)       \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 768);  \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1024); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1280); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1536); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1792); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 2048); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 4096)

#ifdef PADDLE_WITH_CUDA
  bool can_call_fast_kernel = false;
  if ((feature_size >= 768 && feature_size <= 2048 && feature_size % 256 == 0 ||
       feature_size == 4096) &&
      scale != nullptr && bias != nullptr) {
    can_call_fast_kernel = true;
  }

  if (can_call_fast_kernel) {
    if (is_scale_bias_same_dtype_with_x) {
      switch (feature_size) {
        PADDLE_LAUNCH_FAST_LAYERNORM_FWD(T);
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "Only when feature_size is from 256 to 4096 and is diviaible by "
              "256 is supported "
              "now"));
          break;
      }
    } else {
      switch (feature_size) {
        PADDLE_LAUNCH_FAST_LAYERNORM_FWD(U);
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "Only when feature_size is from 256 to 4096 and is diviaible by "
              "is supported "
              "now"));
          break;
      }
    }
  } else {
    // WarpShuffle intrinsics is involved in LaunchLayerNormKernel.
    if (FLAGS_use_fast_math && feature_size <= 1024 &&
        (!std::is_same<T, int8_t>::value)) {
      LaunchLayerNormKernel<Context, T, U>(dev_ctx,
                                           x_data,
                                           y_data,
                                           void_scale_data,
                                           void_bias_data,
                                           mean_data,
                                           var_data,
                                           epsilon,
                                           batch_size,
                                           feature_size,
                                           valid_scale,
                                           valid_bias,
                                           is_scale_bias_same_dtype_with_x);
    } else {
#endif
      if (is_scale_bias_same_dtype_with_x) {
        PADDLE_LAUNCH_LAYERNORM_FWD(T, true);
      } else {
        PADDLE_LAUNCH_LAYERNORM_FWD(U, false);
      }
#ifdef PADDLE_WITH_CUDA
    }
  }
#endif

#undef PADDLE_LAUNCH_LAYERNORM_FWD
#undef PADDLE_LAUNCH_FAST_LAYERNORM_FWD
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
#else
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
#endif
