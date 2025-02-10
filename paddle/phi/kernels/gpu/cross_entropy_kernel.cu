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

#include "paddle/phi/kernels/cross_entropy_kernel.h"

#include "glog/logging.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

#define ALIGN_BYTES 16

enum class SoftmaxMode { kSoftmax, kLogSoftmax, kCrossEntropy };

// Wrapper of log function. Use log(float32) for float16
template <typename T>
static __device__ __forceinline__ T Log(T x) {
  using AccT = typename dtype::MPTypeTrait<T>::Type;
  AccT logx = std::log(static_cast<AccT>(x));
  return phi::funcs::TolerableValue<T>()(static_cast<T>(logx));
}

// Wrapper of exp function. Use exp(float32) for float16
template <typename T>
static __device__ __forceinline__ T Exp(T x) {
  using AccT = typename dtype::MPTypeTrait<T>::Type;
  AccT expx = std::exp(static_cast<AccT>(x));
  return phi::funcs::TolerableValue<T>()(static_cast<T>(expx));
}

template <typename Tx, typename Ty = Tx>
struct ExpAddFunctor {
  HOSTDEVICE inline ExpAddFunctor(Tx max) : max(max) {}

  HOSTDEVICE inline Ty operator()(const Tx& sum, const Tx& x) const {
    return static_cast<Ty>(sum + std::exp(x - max));
  }

 private:
  Tx max;
};

/*
  Cross entropy soft label with dynamic size on axis (log2_elements is
  variable).
  - if the input is softmax, compute loss with softmax
  - if the input is log_softmax, compute loss with log_softmax and update
  softmax
*/
template <typename T, typename VecT, bool InLogMode = false>
__global__ void CrossEntropySoftLabel(T* loss,
                                      T* softmaxwrt,
                                      const T* softmax,
                                      const T* labels,
                                      const int n,
                                      const int dim,
                                      const int d,
                                      int log2_elements) {
  const int kDimCeil = 1 << log2_elements;
  const int kVSize = sizeof(VecT) / sizeof(T);

  const int kThreadPerBlock = 512;
  const int kBatchPerBlock = 1;
  const int kWarpSize = 32;  // (dim < 32) ? dim : 32;
  const int kBatchSize = 1;
  const int kThreadPerBatch = kThreadPerBlock / kBatchPerBlock;
  const int kWarpPerBatch = kThreadPerBatch / kWarpSize;

  const int kIterations = (dim + kThreadPerBatch - 1) / kThreadPerBatch;
  const int kIterationsV = (kIterations >= kVSize) ? (kIterations / kVSize) : 1;

  const int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;

  T sum[kBatchSize]{static_cast<T>(0.0)};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    int ids = first_batch + i;
    if (ids >= n * d) break;
    int idx_n = ids / d;
    int idx_d = ids % d;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      int idx_dim = it * kThreadPerBatch + threadIdx.x;
      int idx = idx_n * dim * d + idx_dim * d + idx_d;

      if (idx_n < n && idx_dim < dim) {
        VecT softmaxdata;
        if (InLogMode) {
          softmaxdata = reinterpret_cast<VecT*>(&softmaxwrt[idx])[0];
        } else {
          softmaxdata = reinterpret_cast<const VecT*>(&softmax[idx])[0];
        }
        VecT labelsdata = reinterpret_cast<const VecT*>(&labels[idx])[0];
        T* softmaxptr = reinterpret_cast<T*>(&softmaxdata);
        T* labelsptr = reinterpret_cast<T*>(&labelsdata);
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          if (InLogMode) {
            sum[i] -= softmaxptr[s] * labelsptr[s];
            softmaxptr[s] = Exp(softmaxptr[s]);
          } else {
            sum[i] -= Log(softmaxptr[s]) * labelsptr[s];
          }
        }
        if (InLogMode) {
          reinterpret_cast<VecT*>(&softmaxwrt[idx])[0] = softmaxdata;
        }
      }
    }
  }
  phi::WarpReduceSum<T, kBatchSize, kWarpSize>(sum);
  __syncthreads();

  __shared__ T sumshare[kWarpPerBatch][kBatchPerBlock][kBatchSize];
  if (threadIdx.x % kWarpSize == 0) {
#pragma unroll
    for (int i = 0; i < kBatchSize; i++) {
      sumshare[threadIdx.x / kWarpSize][threadIdx.y][i] = sum[i];
    }
  }
  __syncthreads();

  // write
  if (threadIdx.x == 0) {
    for (int i = 0; i < kBatchSize; i++) {
      int ids = first_batch + i;
      if (ids < n * d) {
        loss[ids] = sumshare[0][threadIdx.y][i];
        for (int s = 1; s < kWarpPerBatch; s++) {
          loss[ids] += sumshare[s][threadIdx.y][i];
        }
      }
    }
  }
}

/*
  Hard label cross entropy.
*/
template <typename T, typename LabelT>
__global__ void CrossEntropyHardLabel(T* loss,
                                      const T* softmax,
                                      const LabelT* labels,
                                      const int n,
                                      const int dim,
                                      const int d,
                                      const int ignore_idx) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = ids / d;
  int64_t idx_d = ids % d;

  // thread ids compute loss[ids] using softmax[idx]
  if (ids < n * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    PADDLE_ENFORCE(lbl >= 0 && lbl < dim || lbl == ignore_idx,
                   "The value of label expected >= 0 and < %d, or == %d, "
                   "but got %ld. Please check label value.",
                   dim,
                   ignore_idx,
                   lbl);
    if (lbl == ignore_idx) {
      loss[ids] = static_cast<T>(0.0);
    } else {
      int64_t idx = idx_n * dim * d + lbl * d + idx_d;
      loss[ids] = -Log(softmax[idx]);
    }
  }
}

/*
  Hard label cross entropy with exp.
  Input: log softmax
  Output: loss and exp(input)
*/
template <typename T, typename LabelT>
__global__ void CrossEntropyExpHardLabel(T* loss,
                                         T* softmax,
                                         const LabelT* labels,
                                         const int n,
                                         const int dim,
                                         const int d,
                                         const int ignore_idx) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    PADDLE_ENFORCE(lbl >= 0 && lbl < dim || lbl == ignore_idx,
                   "The value of label expected >= 0 and < %d, or == %d, "
                   "but got %ld. Please check label value.",
                   dim,
                   ignore_idx,
                   lbl);
    if (lbl == ignore_idx) {
      loss[ids] = static_cast<T>(0.0);
    } else {
      if (lbl == idx_dim) {
        loss[ids] = -softmax[idx];
      }
    }
    softmax[idx] = Exp(softmax[idx]);
  }
}

template <typename T, typename AccT, int VecSize, class ReduceFunctor>
__device__ __forceinline__ AccT ThreadReduce(const T* input,
                                             int size,
                                             const int offset,
                                             AccT init,
                                             ReduceFunctor reducer) {
  using VecT = kps::details::VectorType<T, VecSize>;
  int tid = threadIdx.x;
  AccT val = init;

  if (offset > 0) {
    input -= offset;
    size += offset;
    if (tid >= offset) {
      val = reducer(val, input[tid]);
    }
    size -= blockDim.x;
    input += blockDim.x;
  }
  int remain = size % (VecSize * blockDim.x);

  T ins[VecSize];
  VecT* ins_vec = reinterpret_cast<VecT*>(&ins);

  // vector part
  for (; VecSize * tid < (size - remain); tid += blockDim.x) {
    *ins_vec = reinterpret_cast<const VecT*>(input)[tid];

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      val = reducer(val, ins[i]);
    }
  }

  // scalar part
  tid = size - remain + threadIdx.x;
  for (; tid < size; tid += blockDim.x) {
    val = reducer(val, input[tid]);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ void ComputeLoss(T* loss,
                                            const T loss_value,
                                            const int label_id,
                                            const int64_t label_value,
                                            const int tid,
                                            const int vec_size,
                                            const int offset,
                                            const int ignore_index) {
  int loss_id = vec_size * tid + offset;
  if (label_value == ignore_index) {
    loss[label_id] = static_cast<T>(0.0f);
  } else {
    if (label_value == loss_id) {
      loss[label_id] = loss_value;
    }
  }
}

template <typename T, typename AccT, typename LabelT, int VecSize>
__device__ __forceinline__ void VectorizedSoftmaxForwardImpl(
    T* loss,
    T* softmax,
    const T* logits,
    const LabelT* label,
    int size,
    const int offset,
    const phi::LogSoftmaxForwardFunctor<AccT>& func,
    const int ignore_index) {
  using VecT = kps::details::VectorType<T, VecSize>;
  int tid = threadIdx.x;
  int label_id = blockIdx.x;
  auto label_value = static_cast<int64_t>(label[label_id]);
  PADDLE_ENFORCE(
      label_value >= 0 && label_value < size || label_value == ignore_index,
      "The value of label expected >= 0 and < %d, or == %d, "
      "but got %ld. Please check label value.",
      size,
      ignore_index,
      label_value);
  int loss_id_offset = 0;

  if (offset > 0) {
    logits -= offset;
    softmax -= offset;
    size += offset;
    loss_id_offset -= offset;
    if (tid >= offset) {
      AccT log_softmax = func(static_cast<AccT>(logits[tid]));
      softmax[tid] = static_cast<T>(std::exp(log_softmax));
      // loss
      ComputeLoss<T>(loss,
                     static_cast<T>(-log_softmax),
                     label_id,
                     label_value,
                     tid,
                     1,
                     loss_id_offset,
                     ignore_index);
    }
    size -= blockDim.x;
    logits += blockDim.x;
    softmax += blockDim.x;
    loss_id_offset += blockDim.x;
  }
  int remain = size % (VecSize * blockDim.x);

  T ins[VecSize];
  T outs[VecSize];
  VecT* ins_vec = reinterpret_cast<VecT*>(&ins);
  VecT* outs_vec = reinterpret_cast<VecT*>(&outs);

  // vector part
  for (; VecSize * tid < (size - remain); tid += blockDim.x) {
    // read
    *ins_vec = reinterpret_cast<const VecT*>(logits)[tid];

#pragma unroll
    // compute
    for (int i = 0; i < VecSize; ++i) {
      AccT log_softmax = func(static_cast<AccT>(ins[i]));
      outs[i] = static_cast<T>(std::exp(log_softmax));

      // loss
      ComputeLoss<T>(loss,
                     static_cast<T>(-log_softmax),
                     label_id,
                     label_value,
                     tid,
                     VecSize,
                     loss_id_offset + i,
                     ignore_index);
    }

    // write
    reinterpret_cast<VecT*>(softmax)[tid] = *outs_vec;
  }

  // scalar part
  tid = size - remain + threadIdx.x;
  for (; tid < size; tid += blockDim.x) {
    AccT log_softmax = func(static_cast<AccT>(logits[tid]));
    softmax[tid] = static_cast<T>(std::exp(log_softmax));

    // loss
    ComputeLoss<T>(loss,
                   static_cast<T>(-log_softmax),
                   label_id,
                   label_value,
                   tid,
                   1,
                   loss_id_offset,
                   ignore_index);
  }
}

template <typename T, typename AccT, typename LabelT, int VecSize>
__device__ __forceinline__ void ScalarSoftmaxForwardImpl(
    T* loss,
    T* softmax,
    const T* logits,
    const LabelT* label,
    const int size,
    const phi::LogSoftmaxForwardFunctor<AccT>& func,
    const int ignore_index) {
  int tid = threadIdx.x;
  int remain = size % (VecSize * blockDim.x);
  int label_id = blockIdx.x;
  auto label_value = static_cast<int64_t>(label[label_id]);
  PADDLE_ENFORCE(
      label_value >= 0 && label_value < size || label_value == ignore_index,
      "The value of label expected >= 0 and < %d, or == %d, "
      "but got %ld. Please check label value.",
      size,
      ignore_index,
      label_value);

  // main part
  for (; tid < (size - remain); tid += VecSize * blockDim.x) {
    T ins[VecSize];

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      ins[i] = logits[tid + i * blockDim.x];
    }
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      AccT log_softmax = func(static_cast<AccT>(ins[i]));
      softmax[tid + i * blockDim.x] = static_cast<T>(std::exp(log_softmax));
      // loss
      ComputeLoss<T>(loss,
                     static_cast<T>(-log_softmax),
                     label_id,
                     label_value,
                     tid,
                     VecSize,
                     i,
                     ignore_index);
    }
  }

  // tail part
  for (; tid < size; tid += blockDim.x) {
    AccT log_softmax = func(static_cast<AccT>(logits[tid]));
    softmax[tid] = static_cast<T>(std::exp(log_softmax));
    // loss
    ComputeLoss<T>(loss,
                   static_cast<T>(-log_softmax),
                   label_id,
                   label_value,
                   tid,
                   1,
                   0,
                   ignore_index);
  }
}

template <typename T, typename AccT, typename LabelT, int VecSize>
__global__ void VectorizedSoftmaxForward(T* loss,
                                         T* softmax,
                                         const T* logits,
                                         const LabelT* label,
                                         const int high_dim,
                                         const int mid_dim,
                                         const int ignore_index) {
  using VecT = kps::details::VectorType<T, VecSize>;

  // each block deal with one batch
  logits += static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(mid_dim);
  softmax += static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(mid_dim);

  const int input_offset = ((uint64_t)logits) % ALIGN_BYTES / sizeof(T);
  const int output_offset = ((uint64_t)softmax) % ALIGN_BYTES / sizeof(T);

  // 1. reduce max
  AccT max = ThreadReduce<T, AccT, VecSize, kps::MaxFunctor<AccT>>(
      logits,
      mid_dim,
      input_offset,
      -std::numeric_limits<AccT>::infinity(),
      kps::MaxFunctor<AccT>());
  max = kps::details::BlockXReduce<AccT, kps::MaxFunctor<AccT>>(
      max, kps::MaxFunctor<AccT>());

  // 2. reduce sum
  AccT sum = ThreadReduce<T, AccT, VecSize, ExpAddFunctor<AccT>>(
      logits,
      mid_dim,
      input_offset,
      static_cast<AccT>(0),
      ExpAddFunctor<AccT>(max));
  sum = kps::details::BlockXReduce<AccT, kps::AddFunctor<AccT>>(
      sum, kps::AddFunctor<AccT>());

  // 3. softmax
  phi::LogSoftmaxForwardFunctor<AccT> func(max, sum);
  if (input_offset == output_offset) {
    VectorizedSoftmaxForwardImpl<T, AccT, LabelT, VecSize>(loss,
                                                           softmax,
                                                           logits,
                                                           label,
                                                           mid_dim,
                                                           input_offset,
                                                           func,
                                                           ignore_index);
  } else {
    ScalarSoftmaxForwardImpl<T, AccT, LabelT, VecSize>(
        loss, softmax, logits, label, mid_dim, func, ignore_index);
  }
}

/*
Core function of softmax with cross entropy forward soft label.
The computation includes
  - Compute maximum of batch: maxvalue_{i} = max_j src_{i,j}
  - Compute sum of exp batch: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  - Compute: sum of - sum_{j}{ label_{i,j} * (src_{i,j} - maxvalue_{i} -
log(sum[i]))}
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use
shuffle api to compute max (sum) in one warp.
*/
template <typename T, typename VecT, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForwardSoftLabel(T* loss,
                                            T* softmax,
                                            const T* src,
                                            const T* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count) {
  const bool LogMode = true;

  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  // read data from global memory
  VecT srcdata[kBatchSize][kIterationsV];
  VecT labeldata[kBatchSize][kIterationsV];

  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    const VecT* label_v =
        reinterpret_cast<const VecT*>(&label[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;
#pragma unroll
    // read data
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (src_idx < idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
        labeldata[i][it] = label_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<AccT>::max();
          reinterpret_cast<T*>(&labeldata[i][it])[s] = 0.0;
        }
      }
    }
  }

  // compute max value
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    max_value[i] = -std::numeric_limits<AccT>::infinity();
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = srcptr_v[0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcptr_v[s]) ? valmax : srcptr_v[s];
      }
      max_value[i] = (max_value[i] > static_cast<AccT>(valmax))
                         ? max_value[i]
                         : static_cast<AccT>(valmax);
    }
  }
  phi::WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum
  AccT sum[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          sum[i] += std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
        } else {
          srcptr_v[s] = std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
          sum[i] += static_cast<AccT>(srcptr_v[s]);
        }
      }
    }
  }
  phi::WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // log_softmax and loss
  AccT sumloss[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;

    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    if (LogMode) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T* labelvp = reinterpret_cast<T*>(&labeldata[i][it]);
      VecT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (LogMode) {
          AccT logsoftmax = static_cast<AccT>(srcvp[s]) - max_value[i] - sum[i];
          sumloss[i] -= logsoftmax * static_cast<AccT>(labelvp[s]);
          tmpvp[s] = std::exp(logsoftmax);
        } else {
          tmpvp[s] = static_cast<AccT>(srcvp[s]) / sum[i];
        }
      }

      int idx = threadIdx.x + it * kWarpSize;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      }
    }
  }

  // loss
  phi::WarpReduceSum<AccT, kBatchSize, kWarpSize>(sumloss);
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    if (i >= local_batches) break;
    loss[first_batch + i] = sumloss[i];
  }
}

#define SOFTMAX_WARP_FORWARD_SOFT_CASE(Log2Elements, VecT, AccT)           \
  case Log2Elements:                                                       \
    WarpSoftmaxForwardSoftLabel<T, VecT, AccT, Log2Elements>               \
        <<<blocks, threads, 0, stream>>>(                                  \
            loss, softmax, src, label, batch_size, stride, element_count); \
    break;

/*
  Wrapper of softmax with cross entropy forward soft label.
*/
template <typename T>
void SwitchWarpSoftmaxForwardSoftLabel(const int blocks,
                                       const dim3 threads,
                                       gpuStream_t stream,
                                       T* loss,
                                       T* softmax,
                                       const T* src,
                                       const T* label,
                                       const int batch_size,
                                       const int stride,
                                       const int element_count,
                                       const int log2_elements) {
  using AccT = typename dtype::MPTypeTrait<T>::Type;
  switch (log2_elements) {
    SOFTMAX_WARP_FORWARD_SOFT_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(9, T, AccT);
    default:
      break;
  }
}

template <typename T>
static void SoftmaxWithCrossEntropySoftLabel(const GPUContext& dev_ctx,
                                             const int rank,
                                             const int axis,
                                             const T* logits_data,
                                             const T* labels_data,
                                             T* softmax_data,
                                             T* loss_data,
                                             int N,
                                             int dim,
                                             int D) {
  constexpr int kMaxBlockDim = 512;
  int64_t block_dim = dim >= kMaxBlockDim
                          ? kMaxBlockDim
                          : (1 << static_cast<int>(std::log2(dim)));

  int64_t grid_dim = N * D;
  constexpr int max_dim = 320;

  const int kDimLog2 = static_cast<int>(Log2Ceil(dim));
  const int kDimCeil = 1 << kDimLog2;
  auto stream = dev_ctx.stream();

  if (D == 1 && dim <= max_dim) {
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    SwitchWarpSoftmaxForwardSoftLabel<T>(blocks,
                                         threads,
                                         stream,
                                         loss_data,
                                         softmax_data,
                                         logits_data,
                                         labels_data,
                                         N,
                                         dim,
                                         dim,
                                         kDimLog2);

  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    GPUDNNDataLayout layout = GPUDNNDataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSoftmaxForward_V2(
        handle,
        phi::backends::gpu::CudnnDataType<T>::kOne(),
        descp,
        logits_data,
        phi::backends::gpu::CudnnDataType<T>::kZero(),
        descp,
        softmax_data,
        MIOPEN_SOFTMAX_LOG,
        mode));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_LOG,
        mode,
        phi::backends::gpu::CudnnDataType<T>::kOne(),
        descp,
        logits_data,
        phi::backends::gpu::CudnnDataType<T>::kZero(),
        descp,
        softmax_data));
#endif

    const int kDimLog2 = static_cast<int>(Log2Ceil(dim));
    const int kDimCeil = 1 << kDimLog2;
    int kThreadPerBlock = 512;

    int kBatchPerBlock = 1;
    int blocks = (N * D + kBatchPerBlock - 1) / kBatchPerBlock;
    dim3 threads(kThreadPerBlock / kBatchPerBlock, kBatchPerBlock, 1);

    CrossEntropySoftLabel<T, T, true><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, NULL, labels_data, N, dim, D, kDimLog2);
  }
}

/*
  Core function of softmax with cross entropy forward
    - softmax, SoftmaxMode=kSoftmax
    - log softmax, SoftmaxMode=kLogSoftmax
    - softmax with cross entropy hard label, SoftmaxMode=kCrossEntropy
  The computation includes
    - Compute max value: maxvalue_{i} = max_j src_{i,j}
    - Compute sum of exp: s_{i} = sum_{j}{e^{src_{i,j} - maxvalue_{i}}}
    - Compute: softmax_{i,j} = e^{src_{i,j} - maxvalue_{i}} / s_{i}
    - Compute: logsoftmax_{i,j} = src_{i,j} - maxvalue_{i} - log(s_{i})
    - Compute: loss_{i} = -logsoftmax[i,label[i]] (Hard label)
  This computation results from following formula:
    softmax_{i,j} = e^{src_{i,j}} / sum_{j}{e^{src_{i,j}}}
                  = e^{src_{i,j} - maxvalue_{i}}
                    / sum_{j}{e^{src_{i,j} - maxvalue_{i}}}
                  = e^{src_{i,j} - maxvalue_{i}} / s_{i}
    logsoftmax_{i,j} = log(softmax_{i,j})
                     = src_{i,j} - maxvalue_{i} - log(s_{i})
  One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
  For reduction max (sum), firstly compute max (sum) to one warp, then use
  shuffle api to compute max (sum) in one warp.
*/
template <typename T,
          typename LabelT,
          typename VecT,
          typename AccT,
          int Log2Elements,
          SoftmaxMode mode>
__global__ void WarpSoftmaxForward(T* loss,
                                   T* softmax,
                                   const T* src,
                                   const LabelT* label,
                                   const int batch_size,
                                   const int stride,
                                   const int element_count,
                                   const int ignore_index) {
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;

  // max index to read
  int idx_max_v[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    int idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // read data from global memory
  AccT srcdata[kBatchSize][kIterationsV][kVSize];

#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
// read data to srcdata: - KVSize==1, - KVSize>1
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {
        if (src_idx < idx_max_v[i]) {
          srcdata[i][it][0] =
              static_cast<AccT>(src[(first_batch + i) * stride + src_idx]);
        } else {
          srcdata[i][it][0] = -std::numeric_limits<AccT>::infinity();
        }
      } else {
        const VecT* src_v =
            reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
        if (src_idx < idx_max_v[i]) {
          VecT srctmp = src_v[src_idx];
          const T* srcinptr = reinterpret_cast<const T*>(&srctmp);
#pragma unroll
          for (int s = 0; s < kVSize; s++) {
            srcdata[i][it][s] = static_cast<AccT>(srcinptr[s]);
          }
        } else {
#pragma unroll
          for (int s = 0; s < kVSize; s++) {
            srcdata[i][it][s] = -std::numeric_limits<AccT>::infinity();
          }
        }
      }
    }
  }

  // compute max value: maxvalue_{i} = max_j src_{i,j}
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    AccT valmax = srcdata[i][0][0];
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      valmax = (valmax > srcdata[i][0][s]) ? valmax : srcdata[i][0][s];
    }
    max_value[i] = valmax;

// it = 1, 2, ...
#pragma unroll
    for (int it = 1; it < kIterationsV; ++it) {
      AccT valmax = srcdata[i][it][0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcdata[i][it][s]) ? valmax : srcdata[i][it][s];
      }
      max_value[i] = (max_value[i] > valmax) ? max_value[i] : valmax;
    }
  }
  phi::WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  AccT sum[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    // it = 0
    if (mode == SoftmaxMode::kLogSoftmax ||
        mode == SoftmaxMode::kCrossEntropy) {
      sum[i] = std::exp(srcdata[i][0][0] - max_value[i]);
    } else {
      srcdata[i][0][0] = std::exp(srcdata[i][0][0] - max_value[i]);
      sum[i] = srcdata[i][0][0];
    }
#pragma unroll
    for (int s = 1; s < kVSize; ++s) {
      if (mode == SoftmaxMode::kLogSoftmax ||
          mode == SoftmaxMode::kCrossEntropy) {
        sum[i] += std::exp(srcdata[i][0][s] - max_value[i]);
      } else {
        srcdata[i][0][s] = std::exp(srcdata[i][0][s] - max_value[i]);
        sum[i] += srcdata[i][0][s];
      }
    }

// it = 1, 2, ...
#pragma unroll
    for (int it = 1; it < kIterationsV; ++it) {
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (mode == SoftmaxMode::kLogSoftmax ||
            mode == SoftmaxMode::kCrossEntropy) {
          sum[i] += std::exp(srcdata[i][it][s] - max_value[i]);
        } else {
          srcdata[i][it][s] = std::exp(srcdata[i][it][s] - max_value[i]);
          sum[i] += srcdata[i][it][s];
        }
      }
    }
  }
  phi::WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write data
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (mode == SoftmaxMode::kLogSoftmax ||
        mode == SoftmaxMode::kCrossEntropy) {
      sum[i] = std::log(sum[i]);
    }

#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int idx = threadIdx.x + it * kWarpSize;
      if (kVSize == 1) {  // kVSize==1
        if (idx < idx_max_v[i]) {
          if (mode == SoftmaxMode::kLogSoftmax) {  // log softmax
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] - max_value[i] - sum[i];
            // softmax with cross entropy hard label
          } else if (mode == SoftmaxMode::kCrossEntropy) {
            AccT logsoftmax = srcdata[i][it][0] - max_value[i] - sum[i];
            // softmax
            softmax[(first_batch + i) * stride + idx] = std::exp(logsoftmax);
            // label
            int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize;
            auto lbl = static_cast<int64_t>(label[first_batch + i]);
            if (lbl == ignore_index) {
              loss[first_batch + i] = static_cast<T>(0.0);
            } else {
              if (lbl >= 0 && lbl < element_count) {
                if (lbl == loss_idx) {
                  loss[first_batch + i] = -logsoftmax;
                }
              } else {
                PADDLE_ENFORCE(
                    false,
                    "The value of label expected >= 0 and < %d, or == %d, "
                    "but got %ld. Please check label value.",
                    element_count,
                    ignore_index,
                    lbl);
              }
            }
          } else {  // softmax
            softmax[(first_batch + i) * stride + idx] =
                srcdata[i][it][0] / sum[i];
          }
        } else {
          break;
        }
      } else {  // KVSize>1
        VecT* softmax_v =
            reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);
        VecT tmpdata;
        T* tmpptr = reinterpret_cast<T*>(&tmpdata);
#pragma unroll
        for (int s = 0; s < kVSize; ++s) {
          if (mode == SoftmaxMode::kLogSoftmax) {  // log softmax
            tmpptr[s] = srcdata[i][it][s] - max_value[i] - sum[i];
            // softmax with cross entropy hard label
          } else if (mode == SoftmaxMode::kCrossEntropy) {
            AccT logsoftmax = srcdata[i][it][s] - max_value[i] - sum[i];
            // softmax
            tmpptr[s] = std::exp(logsoftmax);
            // label
            int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize + s;
            auto lbl = static_cast<int64_t>(label[first_batch + i]);
            if (lbl == ignore_index) {
              loss[first_batch + i] = static_cast<T>(0.0);
            } else {
              if (lbl >= 0 && lbl < element_count) {
                if (lbl == loss_idx) {
                  loss[first_batch + i] = -logsoftmax;
                }
              } else {
                PADDLE_ENFORCE(
                    false,
                    "The value of label expected >= 0 and < %d, or == %d, "
                    "but got %ld. Please check label value.",
                    element_count,
                    ignore_index,
                    lbl);
              }
            }
          } else {  // softmax
            tmpptr[s] = srcdata[i][it][s] / sum[i];
          }
        }
        if (idx < idx_max_v[i]) {
          softmax_v[idx] = tmpdata;
        } else {
          break;
        }
      }
    }
  }
}

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, LabelT, VecT, AccT) \
  case Log2Elements:                                                \
    WarpSoftmaxForward<T, LabelT, VecT, AccT, Log2Elements, mode>   \
        <<<blocks, threads, 0, stream>>>(loss,                      \
                                         softmax,                   \
                                         src,                       \
                                         label,                     \
                                         batch_size,                \
                                         stride,                    \
                                         element_count,             \
                                         ignore_index);             \
    break;

/*
  Wrapper of softmax with cross entropy forward hard label.
*/
template <typename T, typename LabelT, SoftmaxMode mode>
void SwitchWarpSoftmaxForward(T* loss,
                              T* softmax,
                              const T* src,
                              const LabelT* label,
                              const int batch_size,
                              const int stride,
                              const int element_count,
                              const int ignore_index,
                              gpuStream_t stream) {
  using AccT = typename dtype::MPTypeTrait<T>::Type;

  // use 128 threads per block to maximimize gpu utilization
  const int log2_elements = static_cast<int>(Log2Ceil(element_count));
  const int kDimCeil = 1 << log2_elements;
  int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / kWarpSize);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_size + batches_per_block - 1) / batches_per_block;
  dim3 threads(kWarpSize, warps_per_block, 1);

  switch (log2_elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, LabelT, T, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, LabelT, T, AccT);
    default:
      break;
  }
}

template <typename T, typename LabelT>
void LaunchVectorizedSoftmaxForward(T* loss,
                                    T* softmax,
                                    const T* logits,
                                    const LabelT* label,
                                    const int high_dim,
                                    const int mid_dim,
                                    const int ignore_index,
                                    gpuStream_t stream) {
  using AccT = typename dtype::MPTypeTrait<T>::Type;
  constexpr int vec_size = sizeof(float4) / sizeof(T);
  const int max_num_threads = 1024;
  int max_block_size = std::min(mid_dim / vec_size, max_num_threads);
  if (vec_size > 1) {
    max_block_size /= 2;
  }

  int block_size = 1;
  while (block_size < max_block_size) {
    block_size *= 2;
  }
  block_size = std::max(block_size, kps::details::kWarpSize);
  dim3 grids(high_dim);
  dim3 blocks(block_size);
  VectorizedSoftmaxForward<T, AccT, LabelT, vec_size>
      <<<grids, blocks, 0, stream>>>(
          loss, softmax, logits, label, high_dim, mid_dim, ignore_index);
}

/*
  Wrapper of softmax with cross entropy hard label.
  - SwitchWarpSoftmaxForward for small size when axis == -1
  - LaunchVectorizedSoftmaxForward for large size when axis == -1
  - cudnn function for axis != -1
*/
template <typename T, typename LabelT>
static void SoftmaxWithCrossEntropyHardLabel(const GPUContext& dev_ctx,
                                             int rank,
                                             int axis,
                                             const T* logits_data,
                                             const LabelT* labels_data,
                                             T* loss_data,
                                             T* softmax_data,
                                             int N,
                                             int dim,
                                             int D,
                                             const int ignore_index) {
  VLOG(7) << "rank=" << rank << ", axis = " << axis << ", N = " << N
          << ", dim = " << dim << ", D = " << D;
  auto stream = dev_ctx.stream();
  constexpr int max_dim = 320;
  if (D == 1) {
    if (dim <= max_dim) {  // small size
      const SoftmaxMode mode = SoftmaxMode::kCrossEntropy;
      SwitchWarpSoftmaxForward<T, LabelT, mode>(loss_data,
                                                softmax_data,
                                                logits_data,
                                                labels_data,
                                                N,
                                                dim,
                                                dim,
                                                ignore_index,
                                                stream);
    } else {  // large size
      LaunchVectorizedSoftmaxForward<T, LabelT>(loss_data,
                                                softmax_data,
                                                logits_data,
                                                labels_data,
                                                N,
                                                dim,
                                                ignore_index,
                                                stream);
    }
  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    GPUDNNDataLayout layout = GPUDNNDataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t descp = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSoftmaxForward_V2(
        handle,
        phi::backends::gpu::CudnnDataType<T>::kOne(),
        descp,
        logits_data,
        phi::backends::gpu::CudnnDataType<T>::kZero(),
        descp,
        softmax_data,
        MIOPEN_SOFTMAX_LOG,
        mode));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_LOG,
        mode,
        phi::backends::gpu::CudnnDataType<T>::kOne(),
        descp,
        logits_data,
        phi::backends::gpu::CudnnDataType<T>::kZero(),
        descp,
        softmax_data));
#endif
    int threads = 128;
    int blocks = (N * dim * D + threads - 1) / threads;
    // compute cross entropy, input is log softmax
    CrossEntropyExpHardLabel<T, LabelT><<<blocks, threads, 0, stream>>>(
        loss_data, softmax_data, labels_data, N, dim, D, ignore_index);
  }
}

template <typename T, typename LabelT>
void CrossEntropyWithSoftmaxCUDAKernel(const GPUContext& dev_ctx,
                                       const DenseTensor& logits,
                                       const DenseTensor& label,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       DenseTensor* softmax,
                                       DenseTensor* loss) {
  VLOG(7) << "logits.shape={" << logits.dims() << "}, label.shape={"
          << label.dims() << "}, soft_label=" << soft_label
          << ", use_softmax=" << use_softmax
          << ", numeric_stable_mode=" << numeric_stable_mode
          << ", ignore_index=" << ignore_index << ", axis=" << axis;

  // do not with softmax op, and input is softmax
  if (!use_softmax) {
    DenseTensor* softmax_out = softmax;
    const DenseTensor* softmax = &logits;
    const DenseTensor& labels = label;

    const int rank = softmax->dims().size();
    const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
    const int axis_dim = softmax->dims()[axis_v];

    const int n = phi::funcs::SizeToAxis(axis_v, softmax->dims());
    const int d = phi::funcs::SizeFromAxis(axis_v, softmax->dims());

    auto* softmax_out_data = dev_ctx.template Alloc<T>(softmax_out);
    auto* loss_data = dev_ctx.template Alloc<T>(loss);

    phi::funcs::SetConstant<GPUContext, T> set_constant;
    set_constant(dev_ctx, loss, static_cast<T>(0));
    if (axis_dim == 1) {
      set_constant(dev_ctx, softmax_out, static_cast<T>(1));
      return;
    }

    DenseTensor softmax_2d(*softmax);
    softmax_2d.Resize({n, d});
    DenseTensor labels_2d(labels);
    labels_2d.Resize({n, labels.numel() / n});
    DenseTensor loss_2d(*loss);
    loss_2d.Resize({n, 1});
    DenseTensor softmax_out_2d(*softmax_out);
    softmax_out_2d.Resize({n, d});

    // phi::funcs::CrossEntropyFunctor support axis is the last
    if (axis_v == -1) {
      phi::funcs::CrossEntropyFunctor<GPUContext, T>()(dev_ctx,
                                                       &loss_2d,
                                                       &softmax_2d,
                                                       &labels_2d,
                                                       soft_label,
                                                       ignore_index,
                                                       axis_dim);
      return;
    }

    // if axis is not the last, we need a new implement
    if (soft_label) {
      auto* logits_data = softmax->data<T>();
      auto* labels_data = labels.data<T>();

      const int kDimLog2 = static_cast<int>(Log2Ceil(axis_dim));
      const int kDimCeil = 1 << kDimLog2;
      int kThreadPerBlock = 512;
      int kBatchPerBlock = 1;
      int blocks = (n * d + kBatchPerBlock - 1) / kBatchPerBlock;
      dim3 threads(kThreadPerBlock / kBatchPerBlock, kBatchPerBlock, 1);

      CrossEntropySoftLabel<T, T, false>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_data,
                                                     NULL,
                                                     logits_data,
                                                     labels_data,
                                                     n,
                                                     axis_dim,
                                                     d / axis_dim,
                                                     kDimLog2);
    } else {  // HardLabel
      auto* logits_data = softmax->data<T>();
      auto* labels_data = labels.data<LabelT>();
      int threads = 128;
      int blocks = (n * d / axis_dim + threads - 1) / threads;
      CrossEntropyHardLabel<T, LabelT>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(loss_data,
                                                     logits_data,
                                                     labels_data,
                                                     n,
                                                     axis_dim,
                                                     d / axis_dim,
                                                     ignore_index);
    }

    // cause of input is softmax
    // copy to output softmax, directly
    phi::Copy<GPUContext>(
        dev_ctx, *softmax, dev_ctx.GetPlace(), false, softmax_out);

    return;
  }

  const int rank = logits.dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = logits.dims()[axis_v];

  const int64_t n = phi::funcs::SizeToAxis(axis_v, logits.dims());
  const int64_t d = phi::funcs::SizeFromAxis(axis_v, logits.dims());

  auto* softmax_data = dev_ctx.template Alloc<T>(softmax);
  auto* loss_data = dev_ctx.template Alloc<T>(loss);

  if (axis_dim == 1) {
    phi::funcs::SetConstant<GPUContext, T> set_constant;
    set_constant(dev_ctx, softmax, static_cast<T>(1));
    set_constant(dev_ctx, loss, static_cast<T>(0));
    return;
  }

  if (soft_label) {
    auto* logits_data = logits.data<T>();
    auto* labels_data = label.data<T>();
    SoftmaxWithCrossEntropySoftLabel<T>(dev_ctx,
                                        rank,
                                        axis_v,
                                        logits_data,
                                        labels_data,
                                        softmax_data,
                                        loss_data,
                                        n,
                                        axis_dim,
                                        d / axis_dim);
  } else {
    if (!numeric_stable_mode) {
      // CUDNN kernel only suppoer 2-D tensor and perform softmax on last dim
      DenseTensor logits_2d(logits);
      logits_2d.Resize({n, d});
      DenseTensor softmax_2d(*softmax);
      softmax_2d.Resize({n, d});
      DenseTensor labels_2d(label);
      labels_2d.Resize({n, label.numel() / n});
      DenseTensor loss_2d(*loss);
      loss_2d.Resize({n, 1});
      phi::funcs::SoftmaxCUDNNFunctor<T, GPUContext>()(
          dev_ctx, &logits_2d, &softmax_2d);
      phi::funcs::CrossEntropyFunctor<GPUContext, T>()(dev_ctx,
                                                       &loss_2d,
                                                       &softmax_2d,
                                                       &labels_2d,
                                                       false,
                                                       ignore_index,
                                                       axis_dim);
    } else {
      auto* logits_data = logits.data<T>();
      auto* labels_data = label.data<LabelT>();
      SoftmaxWithCrossEntropyHardLabel<T, LabelT>(dev_ctx,
                                                  rank,
                                                  axis_v,
                                                  logits_data,
                                                  labels_data,
                                                  loss_data,
                                                  softmax_data,
                                                  n,
                                                  axis_dim,
                                                  d / axis_dim,
                                                  ignore_index);
    }
  }
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const DenseTensor& logits,
                                   const DenseTensor& label,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   DenseTensor* softmax,
                                   DenseTensor* loss) {
  auto dtype = label.dtype();
  if (soft_label) {
    PADDLE_ENFORCE_EQ(
        dtype,
        phi::CppTypeToDataType<T>::Type(),
        common::errors::InvalidArgument("The Input(Label) should be with the "
                                        "same data type as Input(Logits)."));
    CrossEntropyWithSoftmaxCUDAKernel<T, T>(dev_ctx,
                                            logits,
                                            label,
                                            soft_label,
                                            use_softmax,
                                            numeric_stable_mode,
                                            ignore_index,
                                            axis,
                                            softmax,
                                            loss);
  } else {
    PD_VISIT_INTEGRAL_TYPES(dtype, "CrossEntropyWithSoftmaxCUDAKernel", ([&] {
                              CrossEntropyWithSoftmaxCUDAKernel<T, data_t>(
                                  dev_ctx,
                                  logits,
                                  label,
                                  soft_label,
                                  use_softmax,
                                  numeric_stable_mode,
                                  ignore_index,
                                  axis,
                                  softmax,
                                  loss);
                            }));
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(cross_entropy_with_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxKernel,
                   float,
                   phi::dtype::float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(cross_entropy_with_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(cross_entropy_with_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
#endif
