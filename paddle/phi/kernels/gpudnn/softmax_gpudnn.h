/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"

#define MATRIX_SOFTMAX_ALIGN_BYTES 16
#define MATRIX_SOFTMAX_THRESHOLD 100000

namespace phi {

using ScopedTensorDescriptor = phi::backends::gpu::ScopedTensorDescriptor;
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

// Vectorization trait 4 * sizeof(T)
template <typename T>
class VecT4 {};
template <>
class VecT4<double> {
 public:
  using Type = long4;
};
template <>
class VecT4<float> {
 public:
  using Type = int4;
};
template <>
class VecT4<phi::dtype::float16> {
 public:
  using Type = int2;
};
template <>
class VecT4<phi::dtype::bfloat16> {
 public:
  using Type = int2;
};

// Vectorization trait 2 * sizeof(T)
template <typename T>
class VecT2 {};
template <>
class VecT2<double> {
 public:
  using Type = int4;
};
template <>
class VecT2<float> {
 public:
  using Type = int2;
};
template <>
class VecT2<phi::dtype::float16> {
 public:
  using Type = int;
};
template <>
class VecT2<phi::dtype::bfloat16> {
 public:
  using Type = int;
};

static inline int Log2Ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

inline int CalcBlockSize(int vec_size, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size =
      std::min(dim_size / vec_size, static_cast<uint64_t>(1024));

  if (vec_size > 1) {
    max_block_size /= 2;
  }

  while (block_size < (max_block_size)) block_size *= 2;
  block_size = std::max(block_size, static_cast<uint64_t>(32));
  return block_size;
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val =
          phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T max_val =
          phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T>
__inline__ __device__ void BlockReduceMax(T* val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  WarpReduceMax<T, 1, 32>(val);

  if (lane == 0) shared[wid] = *val;

  __syncthreads();

  int block_span = (blockDim.x + warpSize - 1) >> 5;
  *val = (lane < block_span) ? shared[lane] : -1e10f;
  WarpReduceMax<T, 1, 32>(val);
}

template <typename T>
__inline__ __device__ void BlockReduceSum(T* val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  WarpReduceSum<T, 1, 32>(val);

  __syncthreads();
  if (lane == 0) shared[wid] = *val;

  __syncthreads();

  int block_span = (blockDim.x + warpSize - 1) >> 5;
  *val = (lane < block_span) ? shared[lane] : static_cast<T>(0.0f);
  WarpReduceSum<T, 1, 32>(val);
}

template <typename Tx, typename Ty = Tx>
struct ReduceMaxFunctor {
  inline Ty initial() { return -std::numeric_limits<Ty>::infinity(); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return max(a, b);
  }
};

template <typename T, typename AccT>
struct MaxFunctor {
  __device__ __forceinline__ AccT operator()(const AccT& max_v,
                                             const T& v) const {
    return max(max_v, static_cast<AccT>(v));
  }
};

template <typename Tx, typename Ty = Tx>
struct ExpFunctor {
  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::exp(x));
  }
};

template <typename Tx, typename Ty = Tx>
struct ExpMulFunctor {
  HOSTDEVICE inline ExpMulFunctor() { y = static_cast<Tx>(1.0f); }

  HOSTDEVICE explicit inline ExpMulFunctor(Tx y) : y((Tx)(y)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::exp(x) * y);
  }

 private:
  Tx y;
};

template <typename Tx, typename Ty = Tx>
struct UnarySubFunctor {
  HOSTDEVICE inline UnarySubFunctor() { y = static_cast<Tx>(0.0f); }

  HOSTDEVICE explicit inline UnarySubFunctor(Tx y) : y((Tx)(y)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x - y);
  }

 private:
  Tx y;
};

template <typename Tx, typename Ty = Tx>
struct UnaryLogFunctor {
  HOSTDEVICE inline UnaryLogFunctor() {}

  HOSTDEVICE explicit inline UnaryLogFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::log(x));
  }
};

template <typename Tx, typename Ty>
struct DataTransFunctor {
  HOSTDEVICE inline DataTransFunctor() {}

  HOSTDEVICE explicit inline DataTransFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return x == -std::numeric_limits<Tx>::infinity()
               ? -std::numeric_limits<Ty>::infinity()
               : static_cast<Ty>(x);
  }
};

template <typename Tx, typename Ty = Tx>
struct UnaryDivFunctor {
  HOSTDEVICE inline UnaryDivFunctor() { n_inv = static_cast<Tx>(1.0f); }

  HOSTDEVICE explicit inline UnaryDivFunctor(Tx n) : n_inv((Tx)(1.0 / n)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x * n_inv);
  }

 private:
  Tx n_inv;
};

template <typename Tx, typename Ty = Tx>
struct SoftmaxForwardFunctor {
  HOSTDEVICE inline SoftmaxForwardFunctor(Tx max, Tx sum)
      : max(max), sum(sum) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(std::exp(x - max) / sum);
  }

 private:
  Tx max;
  Tx sum;
};

template <typename Tx, typename Ty = Tx>
struct SoftmaxBackwardFunctor {
  HOSTDEVICE inline SoftmaxBackwardFunctor(Tx sum) : sum(sum) {}

  HOSTDEVICE inline Ty operator()(const Tx& grad_out, const Tx& out) const {
    return static_cast<Ty>(out * (grad_out - sum));
  }

 private:
  Tx sum;
};

template <typename Tx, typename Ty = Tx>
struct LogSoftmaxForwardFunctor {
  HOSTDEVICE inline LogSoftmaxForwardFunctor(Tx max, Tx sum)
      : max(max), log_sum(std::log(sum)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x - max - log_sum);
  }

 private:
  Tx max;
  Tx log_sum;
};

template <typename Tx, typename Ty = Tx>
struct LogSoftmaxBackwardFunctor {
  HOSTDEVICE inline LogSoftmaxBackwardFunctor(Tx sum) : sum(sum) {}

  HOSTDEVICE inline Ty operator()(const Tx& grad_out, const Tx& out) const {
    return static_cast<Ty>(grad_out - std::exp(out) * sum);
  }

 private:
  Tx sum;
};

template <typename T, typename AccT>
struct SumExpFunctor {
  HOSTDEVICE inline SumExpFunctor(AccT v) : max_v(v) {}

  HOSTDEVICE inline AccT operator()(AccT sum, T v) const {
    return sum + std::exp(static_cast<AccT>(v) - max_v);
  }

 private:
  AccT max_v;
};

template <template <typename, typename> class Reduction,
          typename T,
          typename AccT,
          int VecSize>
__device__ __forceinline__ AccT
ThreadVecReduce(T* data,
                int dim_size,
                const int shift,
                const Reduction<T, AccT>& functor,
                AccT default_value) {
  using VecT = phi::AlignedVector<T, VecSize>;
  AccT thread_val = default_value;

  // for memory align, handle the unaligned data in first block.
  int offset = threadIdx.x;
  if (shift > 0) {
    data -= shift;
    dim_size += shift;
    if (offset >= shift) {
      thread_val = functor(thread_val, data[offset]);
    }
    dim_size -= blockDim.x;
    data += blockDim.x;
  }

  const int last = dim_size % (VecSize * blockDim.x);

  T v[VecSize];
  VecT* value = reinterpret_cast<VecT*>(&v);

  for (; offset * VecSize < dim_size - last; offset += blockDim.x) {
    *value = reinterpret_cast<VecT*>(data)[offset];
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      thread_val = functor(thread_val, v[i]);
    }
  }

  offset = dim_size - last + threadIdx.x;
  for (; offset < dim_size; offset += blockDim.x) {
    thread_val = functor(thread_val, data[offset]);
  }
  return thread_val;
}

template <template <typename, typename> class Reduction,
          typename T,
          typename AccT,
          int VecSize>
__device__ __forceinline__ void ThreadVecWriteVec(T* out,
                                                  T* input,
                                                  int dim_size,
                                                  const int shift,
                                                  Reduction<AccT, T> functor) {
  using VecT = phi::AlignedVector<T, VecSize>;

  // for memory align, handle the unaligned data in first block.
  int offset = threadIdx.x;
  if (shift > 0) {
    input -= shift;
    out -= shift;
    dim_size += shift;
    if (offset >= shift) {
      out[offset] = functor(static_cast<AccT>(input[offset]));
    }
    dim_size -= blockDim.x;
    input += blockDim.x;
    out += blockDim.x;
  }

  const int last = dim_size % (VecSize * blockDim.x);

  T in_v[VecSize];
  VecT* in_value = reinterpret_cast<VecT*>(&in_v);

  T out_v[VecSize];
  VecT* out_value = reinterpret_cast<VecT*>(&out_v);

  for (; offset * VecSize < dim_size - last; offset += blockDim.x) {
    *in_value = reinterpret_cast<VecT*>(input)[offset];
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      out_v[i] = functor(static_cast<AccT>(in_v[i]));
    }
    reinterpret_cast<VecT*>(out)[offset] = *out_value;
  }

  offset = dim_size - last + threadIdx.x;
  // the tail
  for (; offset < dim_size; offset += blockDim.x) {
    out[offset] = functor(static_cast<AccT>(input[offset]));
  }
}

template <template <typename, typename> class Reduction,
          typename T,
          typename AccT,
          int VecSize>
__device__ __forceinline__ void ThreadVecWrite(T* out,
                                               T* input,
                                               int dim_size,
                                               Reduction<AccT, T> functor) {
  const int last = dim_size % (VecSize * blockDim.x);

  for (int offset = threadIdx.x; offset < dim_size - last;
       offset += blockDim.x * VecSize) {
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      out[offset + i * blockDim.x] =
          functor(static_cast<AccT>(input[offset + i * blockDim.x]));
    }
  }

  // the tail
  for (int offset = dim_size - last + threadIdx.x; offset < dim_size;
       offset += blockDim.x) {
    out[offset] = functor(static_cast<AccT>(input[offset]));
  }
}

template <typename T, typename AccT, typename IndexType, bool LogMode = false>
__global__ void KeMatrixSoftmaxForward(T* softmax, const T* src, int dim_size) {
  constexpr int kVecSize =
      MaxWithOne<MATRIX_SOFTMAX_ALIGN_BYTES / sizeof(T)>::kValue;
  using VecT = phi::AlignedVector<T, kVecSize>;

  int bid = blockIdx.x;
  T* batch_input = const_cast<T*>(src) + bid * dim_size;
  T* batch_output = softmax + bid * dim_size;

  const int input_align_shift =
      ((uint64_t)batch_input) % MATRIX_SOFTMAX_ALIGN_BYTES / sizeof(T);
  const int output_align_shift =
      ((uint64_t)batch_output) % MATRIX_SOFTMAX_ALIGN_BYTES / sizeof(T);

  // get max value
  AccT thread_max = ThreadVecReduce<MaxFunctor, T, AccT, kVecSize>(
      batch_input,
      dim_size,
      input_align_shift,
      MaxFunctor<T, AccT>(),
      -std::numeric_limits<AccT>::infinity());
  BlockReduceMax<AccT>(&thread_max);

  // get exp value and sum all
  AccT thread_exp = ThreadVecReduce<SumExpFunctor, T, AccT, kVecSize>(
      batch_input,
      dim_size,
      input_align_shift,
      SumExpFunctor<T, AccT>(thread_max),
      static_cast<AccT>(0.));
  BlockReduceSum<AccT>(&thread_exp);

  // write data to softmax_output according to the LogMode
  if (LogMode) {
    LogSoftmaxForwardFunctor<AccT, T> reduction(thread_max, thread_exp);
    if (input_align_shift == output_align_shift) {
      ThreadVecWriteVec<LogSoftmaxForwardFunctor, T, AccT, kVecSize>(
          batch_output, batch_input, dim_size, input_align_shift, reduction);
    } else {
      ThreadVecWrite<LogSoftmaxForwardFunctor, T, AccT, kVecSize>(
          batch_output, batch_input, dim_size, reduction);
    }
  } else {
    SoftmaxForwardFunctor<AccT, T> reduction(thread_max, thread_exp);
    if (input_align_shift == output_align_shift) {
      ThreadVecWriteVec<SoftmaxForwardFunctor, T, AccT, kVecSize>(
          batch_output, batch_input, dim_size, input_align_shift, reduction);
    } else {
      ThreadVecWrite<SoftmaxForwardFunctor, T, AccT, kVecSize>(
          batch_output, batch_input, dim_size, reduction);
    }
  }
}

/*
Core function of computing softmax forward for axis=-1.
The computation includes
  - Compute maximum of batch: maxvalue_{i} = max_j src_{i,j}
  - Compute sum of exp batch: s_{i} = sum_{j}{ exp(src_{i,j} - maxvalue_{i} }
  - Compute: (a_{i,j} - maxvalue_{i}) / s_{i}
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T,
          typename VecT,
          typename AccT,
          typename IndexType,
          int Log2Elements,
          bool LogMode = false>
__global__ void WarpSoftmaxForward(T* softmax,
                                   const T* src,
                                   const IndexType batch_size,
                                   const IndexType stride,
                                   const IndexType element_count) {
  constexpr IndexType kDimCeil = 1 << Log2Elements;
  constexpr IndexType kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr IndexType kVSize = sizeof(VecT) / sizeof(T);
  constexpr IndexType kLoops = kDimCeil / kWarpSize;
  constexpr IndexType kLoopsV = (kLoops >= kVSize) ? (kLoops / kVSize) : 1;
  constexpr IndexType kBatchSize = (kDimCeil <= 32) ? 2 : 1;
  IndexType first_batch =
      (static_cast<IndexType>(blockDim.y) * blockIdx.x + threadIdx.y) *
      kBatchSize;
  constexpr IndexType kStep = kBatchSize * kLoopsV * kVSize;
  constexpr IndexType kVItem = kLoopsV * kVSize;
  constexpr AccT kLowInf = -std::numeric_limits<AccT>::infinity();
  using kMode = kps::details::ReduceMode;

  // max index to read
  IndexType idx_max_v[kBatchSize];
#pragma unroll
  for (IndexType i = 0; i < kBatchSize; i++) {
    IndexType idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // data src
  // src_data: the raw data form global memory
  // sub_data: store the data obtained by (src_data - max), used by log_softmax
  // exp_data: store the data obtained by (exp(sub_data)), used by softmax
  T src_data[kBatchSize][kLoopsV][kVSize];
  AccT sub_data[kBatchSize][kLoopsV][kVSize];
  AccT exp_data[kBatchSize][kLoopsV][kVSize];
  kps::Init<AccT, kStep>(&sub_data[0][0][0], kLowInf);
  kps::Init<T, kStep>(&src_data[0][0][0], -std::numeric_limits<T>::infinity());

  // data dst
  T out_tmp[kBatchSize][kLoopsV][kVSize];

  // max value
  AccT max[kBatchSize];
  kps::Init<AccT, kBatchSize>(&max[0], kLowInf);

  // sum value
  AccT sum[kBatchSize] = {0};

// read data from global memory
#pragma unroll
  for (IndexType i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    VecT* reg_v = reinterpret_cast<VecT*>(&src_data[i][0][0]);
    kps::ReadData<VecT, VecT, kLoopsV, 1, true>(
        &reg_v[0], &src_v[0], idx_max_v[i], 0, kWarpSize, 1);
    kps::ElementwiseUnary<T, AccT, kVItem, 1, DataTransFunctor<T, AccT>>(
        &sub_data[i][0][0], &src_data[i][0][0], DataTransFunctor<T, AccT>());
  }

  // compute max
  kps::Reduce<AccT,
              kVItem,
              kBatchSize,
              ReduceMaxFunctor<AccT>,
              kMode::kLocalMode>(
      &max[0], &sub_data[0][0][0], ReduceMaxFunctor<AccT>(), true);
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max);

// compute sum
#pragma unroll
  for (IndexType i = 0; i < kBatchSize; ++i) {
    kps::ElementwiseUnary<AccT, AccT, kVItem, 1, UnarySubFunctor<AccT>>(
        &sub_data[i][0][0], &sub_data[i][0][0], UnarySubFunctor<AccT>(max[i]));
    kps::ElementwiseUnary<AccT, AccT, kVItem, 1, ExpFunctor<AccT>>(
        &exp_data[i][0][0], &sub_data[i][0][0], ExpFunctor<AccT>());
  }
  kps::Reduce<AccT,
              kVItem,
              kBatchSize,
              kps::AddFunctor<AccT>,
              kMode::kLocalMode>(
      &sum[0], &exp_data[0][0][0], kps::AddFunctor<AccT>(), true);
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

// write data to global memory
#pragma unroll
  for (IndexType i = 0; i < kBatchSize; ++i) {
    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);
    VecT* reg_v = reinterpret_cast<VecT*>(&out_tmp[i][0][0]);
    if (LogMode) {
      kps::ElementwiseUnary<AccT, T, kVItem, 1, UnarySubFunctor<AccT>>(
          &out_tmp[i][0][0],
          &sub_data[i][0][0],
          UnarySubFunctor<AccT>(std::log(sum[i])));
    } else {
      kps::ElementwiseUnary<AccT, T, kVItem, 1, UnaryDivFunctor<AccT>>(
          &out_tmp[i][0][0], &exp_data[i][0][0], UnaryDivFunctor<AccT>(sum[i]));
    }
    kps::WriteData<VecT, VecT, kLoopsV, 1, true>(
        &softmax_v[0], &reg_v[0], idx_max_v[i], 0, kWarpSize, 1);
  }
}

/*
Core function of computing softmax backward for axis=-1.
The computation includes
  - Compute sum of exp batch: s_{i} = sum_{j} {src_{i,j} * grad_{i,j}
  - Compute src_{i,j} * ( grad_{i,j}) - s_{i} )
One warp (32 threads) is used to compute 1 or 2 batch (kBatchSize).
For reduction max (sum), firstly compute max (sum) to one warp, then use shuffle
api to compute max (sum) in one warp.
*/
template <typename T,
          typename VecT,
          typename AccT,
          int Log2Elements,
          bool LogMode = false>
__global__ void WarpSoftmaxBackward(T* dst,
                                    const T* grad,
                                    const T* src,
                                    int batch_size,
                                    int stride,
                                    int element_count) {
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kLoops = kDimCeil / kWarpSize;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;
  constexpr int kLoopsV = (kLoops >= kVSize) ? (kLoops / kVSize) : 1;
  int element_count_v = element_count / kVSize;
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = min(batch_size - first_batch, kBatchSize);

  // max index to read
  int idx_max_v[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; i++) {
    int idx_max = ((i + first_batch) < batch_size) ? element_count : 0;
    idx_max_v[i] = idx_max / kVSize;
  }

  // read data from global memory
  VecT src_reg[kBatchSize][kLoopsV];
  VecT grad_reg[kBatchSize][kLoopsV];
  VecT k_value;
  for (int s = 0; s < kVSize; s++) {
    reinterpret_cast<T*>(&k_value)[s] = 0.0;
  }
  kps::Init<VecT, kBatchSize * kLoopsV>(&src_reg[0][0], k_value);
  kps::Init<VecT, kBatchSize * kLoopsV>(&grad_reg[0][0], k_value);
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    int flag = i < local_batches ? 1 : 0;
    int ptr = (first_batch + i) * stride;
    const VecT* src_v = reinterpret_cast<const VecT*>(&src[ptr]);
    const VecT* grad_v = reinterpret_cast<const VecT*>(&grad[ptr]);
    kps::ReadData<VecT, VecT, kLoopsV, 1, true>(
        &src_reg[i][0], &src_v[0], idx_max_v[i], 0, kWarpSize, flag);
    kps::ReadData<VecT, VecT, kLoopsV, 1, true>(
        &grad_reg[i][0], &grad_v[0], idx_max_v[i], 0, kWarpSize, flag);
  }

  // change T to AccT
  AccT src_tmp[kBatchSize][kLoopsV][kVSize];
  AccT grad_tmp[kBatchSize][kLoopsV][kVSize];
  const T* src_ptr = reinterpret_cast<const T*>(&src_reg[0][0]);
  const T* grad_ptr = reinterpret_cast<const T*>(&grad_reg[0][0]);
  constexpr int kStep = kBatchSize * kLoopsV * kVSize;
  constexpr int kVItem = kLoopsV * kVSize;
  kps::ElementwiseUnary<T, AccT, kStep, 1, DataTransFunctor<T, AccT>>(
      &src_tmp[0][0][0], &src_ptr[0], DataTransFunctor<T, AccT>());
  kps::ElementwiseUnary<T, AccT, kStep, 1, DataTransFunctor<T, AccT>>(
      &grad_tmp[0][0][0], &grad_ptr[0], DataTransFunctor<T, AccT>());

  // compute sum
  AccT sum[kBatchSize]{0.0};
  AccT sum_tmp[kBatchSize][kLoopsV][kVSize];
  AccT* gradptr = reinterpret_cast<AccT*>(&grad_tmp[0][0][0]);
  AccT* srcptr = reinterpret_cast<AccT*>(&src_tmp[0][0][0]);
  if (LogMode) {
    kps::Reduce<AccT,
                kVItem,
                kBatchSize,
                kps::AddFunctor<AccT>,
                kps::details::ReduceMode::kLocalMode>(
        &sum[0], &grad_tmp[0][0][0], kps::AddFunctor<AccT>(), true);
  } else {
    kps::ElementwiseBinary<AccT, AccT, kStep, 1, kps::MulFunctor<AccT>>(
        &sum_tmp[0][0][0], &gradptr[0], &srcptr[0], kps::MulFunctor<AccT>());
    kps::Reduce<AccT,
                kVItem,
                kBatchSize,
                kps::AddFunctor<AccT>,
                kps::details::ReduceMode::kLocalMode>(
        &sum[0], &sum_tmp[0][0][0], kps::AddFunctor<AccT>(), true);
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // write result to global memory
  AccT out[kBatchSize][kLoopsV][kVSize];
  T out_tmp[kBatchSize][kLoopsV][kVSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;
    AccT* gradptr = reinterpret_cast<AccT*>(&grad_tmp[i][0][0]);
    AccT* srcptr = reinterpret_cast<AccT*>(&src_tmp[i][0][0]);
    if (LogMode) {
      kps::ElementwiseUnary<AccT, AccT, kVItem, 1, ExpMulFunctor<AccT>>(
          &out[i][0][0], &srcptr[0], ExpMulFunctor<AccT>(sum[i]));
      kps::ElementwiseBinary<AccT, T, kVItem, 1, kps::SubFunctor<AccT>>(
          &out_tmp[i][0][0],
          &gradptr[0],
          &out[i][0][0],
          kps::SubFunctor<AccT>());
    } else {
      kps::ElementwiseUnary<AccT, AccT, kVItem, 1, UnarySubFunctor<AccT>>(
          &out[i][0][0], &gradptr[0], UnarySubFunctor<AccT>(sum[i]));
      kps::ElementwiseBinary<AccT, T, kVItem, 1, kps::MulFunctor<AccT>>(
          &out_tmp[i][0][0],
          &srcptr[0],
          &out[i][0][0],
          kps::MulFunctor<AccT>());
    }
    VecT* dst_v = reinterpret_cast<VecT*>(&dst[(first_batch + i) * stride]);
    VecT* reg_v = reinterpret_cast<VecT*>(&out_tmp[i][0][0]);
    kps::WriteData<VecT, VecT, kLoopsV, 1, true>(
        &dst_v[0], &reg_v[0], idx_max_v[i], 0, kWarpSize, 1);
  }
}

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, AccT)                   \
  case Log2Elements:                                                    \
    WarpSoftmaxForward<T, VecT, AccT, IndexType, Log2Elements, LogMode> \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
            dst, src, batch_size, stride, element_count);               \
    break;

/*
  Wrapper of softmax formward with template instantiation on size of input.
*/
template <typename T, typename VecT, typename IndexType, bool LogMode>
void SwitchWarpSoftmaxForward(const IndexType blocks,
                              const dim3 threads,
                              const GPUContext& dev_ctx,
                              T* dst,
                              const T* src,
                              const IndexType batch_size,
                              const IndexType stride,
                              const IndexType element_count,
                              IndexType log2_element_count) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  switch (log2_element_count) {
    SOFTMAX_WARP_FORWARD_CASE(0, AccT);
    SOFTMAX_WARP_FORWARD_CASE(1, AccT);
    SOFTMAX_WARP_FORWARD_CASE(2, AccT);
    SOFTMAX_WARP_FORWARD_CASE(3, AccT);
    SOFTMAX_WARP_FORWARD_CASE(4, AccT);
    SOFTMAX_WARP_FORWARD_CASE(5, AccT);
    SOFTMAX_WARP_FORWARD_CASE(6, AccT);
    SOFTMAX_WARP_FORWARD_CASE(7, AccT);
    SOFTMAX_WARP_FORWARD_CASE(8, AccT);
    SOFTMAX_WARP_FORWARD_CASE(9, AccT);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported softmax dim: element_count=%d, log2_element_count=%d!",
          element_count,
          log2_element_count));
      break;
  }
}

#define SOFTMAX_WARP_BACKWARD_CASE(Log2Elements, AccT)          \
  case Log2Elements:                                            \
    WarpSoftmaxBackward<T, VecT, AccT, Log2Elements, LogMode>   \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(             \
            dst, grad, src, batch_size, stride, element_count); \
    break;

/*
Wrapper of softmax backward with template instantiation on size of input.
*/
template <typename T, typename VecT, bool LogMode>
void SwitchWarpSoftmaxBackward(const int blocks,
                               const dim3 threads,
                               const GPUContext& dev_ctx,
                               T* dst,
                               const T* grad,
                               const T* src,
                               const int batch_size,
                               const int stride,
                               const int element_count,
                               int log2_element_count) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  switch (log2_element_count) {
    SOFTMAX_WARP_BACKWARD_CASE(0, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(1, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(2, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(3, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(4, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(5, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(6, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(7, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(8, AccT);
    SOFTMAX_WARP_BACKWARD_CASE(9, AccT);
    default:
      // PADDLE_THROW(common::errors::Unimplemented(
      //     "Unsupported softmax dim: element_count=%d,
      //     log2_element_count=%d!", element_count, log2_element_count));
      break;
  }
}

#undef SOFTMAX_WARP_FORWARD_CASE
#undef SOFTMAX_WARP_BACKWARD_CASE

/**
 * <NormalSoftmaxKernel>
 * Better performance when axis != -1
 */

static void GetGridDim(
    int high_dim, int mid_dim, int low_dim, const dim3& block, dim3* grid) {
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  int max_mp = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  int max_threads_per_mp =
      phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);
  int max_threads = max_threads_per_mp * max_mp;
  int num_threads = block.x * block.y;
  int max_num_blocks = max_threads / num_threads;

  int grid_x = (low_dim + block.x - 1) / block.x;
  grid_x = std::min(grid_x, max_num_blocks);
  int grid_y = (max_num_blocks + grid_x - 1) / grid_x;
  grid_y = std::min(grid_y, high_dim);
  grid->x = grid_x;
  grid->y = grid_y;
}

static void GetBlockDim(int mid_dim, int low_dim, dim3* block) {
  constexpr int max_num_threads = 1024;
  int block_x = 1 << Log2Ceil(low_dim);
  int block_y = 1 << Log2Ceil(mid_dim);
  block->x = std::min(block_x, 32);
  block->y = std::min(block_y, static_cast<int>(max_num_threads / block->x));
  block->x = std::min(block_x, static_cast<int>(max_num_threads / block->y));
}

static void GetLaunchConfig(
    int high_dim, int mid_dim, int low_dim, dim3* grid, dim3* block) {
  GetBlockDim(mid_dim, low_dim, block);
  GetGridDim(high_dim, mid_dim, low_dim, *block, grid);
}

template <typename T,
          typename AccT,
          template <typename, typename>
          class Functor>
__global__ void NormalSoftmaxForward(
    T* output, const T* input, int high_dim, int mid_dim, int low_dim) {
  using kMode = kps::details::ReduceMode;
  const int high_stride = mid_dim * low_dim;
  const int mid_stride = low_dim;
  for (int high_id = blockIdx.y; high_id < high_dim; high_id += gridDim.y) {
    for (int low_id = blockIdx.x * blockDim.x + threadIdx.x; low_id < low_dim;
         low_id += blockDim.x * gridDim.x) {
      const int input_offset = high_id * high_stride + low_id;

      // 1. reduce max
      AccT max_value = -std::numeric_limits<AccT>::infinity();
      AccT value = -std::numeric_limits<AccT>::infinity();
      for (int mid_id = threadIdx.y; mid_id < mid_dim; mid_id += blockDim.y) {
        value = static_cast<AccT>(input[input_offset + mid_id * mid_stride]);
        max_value = kps::MaxFunctor<AccT>()(max_value, value);
      }

      if (blockDim.y > 1) {
        kps::Reduce<AccT, 1, 1, kps::MaxFunctor<AccT>, kMode::kGlobalMode>(
            &max_value, &max_value, kps::MaxFunctor<AccT>(), false);
      }

      // 2. reduce sum
      AccT sum = 0;
      for (int mid_id = threadIdx.y; mid_id < mid_dim; mid_id += blockDim.y) {
        value = static_cast<AccT>(input[input_offset + mid_id * mid_stride]);
        sum += std::exp(value - max_value);
      }
      if (blockDim.y > 1) {
        kps::Reduce<AccT, 1, 1, kps::AddFunctor<AccT>, kMode::kGlobalMode>(
            &sum, &sum, kps::AddFunctor<AccT>(), false);
      }

      // 3. (log)softmax
      Functor<AccT, T> functor(max_value, sum);
      for (int mid_id = threadIdx.y; mid_id < mid_dim; mid_id += blockDim.y) {
        int data_offset = input_offset + mid_id * mid_stride;
        output[data_offset] = functor(static_cast<AccT>(input[data_offset]));
      }
    }
  }
}

template <typename T,
          typename AccT,
          template <typename, typename>
          class Functor,
          bool LogMode>
__global__ void NormalSoftmaxBackward(T* input_grad,
                                      const T* output_grad,
                                      const T* output,
                                      int high_dim,
                                      int mid_dim,
                                      int low_dim) {
  using kMode = kps::details::ReduceMode;
  const int high_stride = mid_dim * low_dim;
  const int mid_stride = low_dim;
  for (int high_id = blockIdx.y; high_id < high_dim; high_id += gridDim.y) {
    for (int low_id = blockIdx.x * blockDim.x + threadIdx.x; low_id < low_dim;
         low_id += blockDim.x * gridDim.x) {
      const int grad_offset = high_id * high_stride + low_id;

      // 1. reduce sum
      AccT sum = 0;
      if (LogMode) {
        for (int mid_id = threadIdx.y; mid_id < mid_dim; mid_id += blockDim.y) {
          int data_offset = grad_offset + mid_id * mid_stride;
          sum += static_cast<AccT>(output_grad[data_offset]);
        }
      } else {
        for (int mid_id = threadIdx.y; mid_id < mid_dim; mid_id += blockDim.y) {
          int data_offset = grad_offset + mid_id * mid_stride;
          sum += static_cast<AccT>(output_grad[data_offset]) *
                 static_cast<AccT>(output[data_offset]);
        }
      }
      if (blockDim.y > 1) {
        kps::Reduce<AccT, 1, 1, kps::AddFunctor<AccT>, kMode::kGlobalMode>(
            &sum, &sum, kps::AddFunctor<AccT>(), false);
      }

      // 2. (log)softmax backward
      Functor<AccT, T> functor(sum);
      for (int mid_id = threadIdx.y; mid_id < mid_dim; mid_id += blockDim.y) {
        int data_offset = grad_offset + mid_id * mid_stride;
        input_grad[data_offset] =
            functor(static_cast<AccT>(output_grad[data_offset]),
                    static_cast<AccT>(output[data_offset]));
      }
    }
  }
}

template <typename T, bool LogMode = false>
void LaunchNormalSoftmaxForward(const GPUContext& dev_ctx,
                                T* output_data,
                                const T* input_data,
                                int high_dim,
                                int mid_dim,
                                int low_dim) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  dim3 grid, block;
  GetLaunchConfig(high_dim, mid_dim, low_dim, &grid, &block);
  if (LogMode) {
    NormalSoftmaxForward<T, AccT, LogSoftmaxForwardFunctor>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            output_data, input_data, high_dim, mid_dim, low_dim);
  } else {
    NormalSoftmaxForward<T, AccT, SoftmaxForwardFunctor>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            output_data, input_data, high_dim, mid_dim, low_dim);
  }
}

template <typename T, bool LogMode = false>
void LaunchNormalSoftmaxBackward(const GPUContext& dev_ctx,
                                 T* input_grad_data,
                                 const T* output_grad_data,
                                 const T* output_data,
                                 int high_dim,
                                 int mid_dim,
                                 int low_dim) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  dim3 grid, block;
  GetLaunchConfig(high_dim, mid_dim, low_dim, &grid, &block);
  if (LogMode) {
    NormalSoftmaxBackward<T, AccT, LogSoftmaxBackwardFunctor, LogMode>
        <<<grid, block, 0, dev_ctx.stream()>>>(input_grad_data,
                                               output_grad_data,
                                               output_data,
                                               high_dim,
                                               mid_dim,
                                               low_dim);
  } else {
    NormalSoftmaxBackward<T, AccT, SoftmaxBackwardFunctor, LogMode>
        <<<grid, block, 0, dev_ctx.stream()>>>(input_grad_data,
                                               output_grad_data,
                                               output_data,
                                               high_dim,
                                               mid_dim,
                                               low_dim);
  }
}

template <typename T = int>
static std::vector<T> GetSoftmaxTensorDims(const phi::DDim& dims,
                                           const int axis) {
  auto dim = static_cast<T>(dims[axis]);
  auto N = phi::funcs::SizeToAxis<T>(axis, dims);
  auto D = phi::funcs::SizeOutAxis<T>(axis, dims);
  return {N, dim, D, 1};
}

template <typename T>
void SoftmaxForwardCudnnKernel(const GPUContext& dev_ctx,
                               const T* x_data,
                               const int axis,
                               const int rank,
                               const bool log_mode,
                               const std::vector<int>& tensor_dims,
                               T* out_data) {
  auto handle = dev_ctx.cudnn_handle();
  GPUDNNDataLayout layout = GPUDNNDataLayout::kNCHW;

  ScopedTensorDescriptor scoped_desc;
#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t desc =
      scoped_desc.descriptor<T>(layout, tensor_dims);
  auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                               : MIOPEN_SOFTMAX_MODE_CHANNEL;
  auto algo = log_mode ? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSoftmaxForward_V2(
      handle,
      phi::backends::gpu::CudnnDataType<T>::kOne(),
      desc,
      x_data,
      phi::backends::gpu::CudnnDataType<T>::kZero(),
      desc,
      out_data,
      algo,
      mode));
#else
  cudnnTensorDescriptor_t desc = scoped_desc.descriptor<T>(layout, tensor_dims);
  auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                               : CUDNN_SOFTMAX_MODE_CHANNEL;
  auto algo = log_mode ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSoftmaxForward(
      handle,
      algo,
      mode,
      phi::backends::gpu::CudnnDataType<T>::kOne(),
      desc,
      x_data,
      phi::backends::gpu::CudnnDataType<T>::kZero(),
      desc,
      out_data));
#endif
}

template <typename T>
void LaunchSoftmaxForwardCudnnKernel(const GPUContext& dev_ctx,
                                     const DenseTensor& x,
                                     const int axis,
                                     const bool log_mode,
                                     DenseTensor* out) {
  auto* out_data = out->data<T>();
  auto* x_data = x.data<T>();
  const int rank = x.dims().size();

  std::vector<int> tensor_dims = GetSoftmaxTensorDims(x.dims(), axis);
  int64_t remaining = tensor_dims[0];
  int dim = tensor_dims[1];
  int64_t batch_size = std::numeric_limits<int32_t>::max() / dim;
  int offset = batch_size * dim;
  while (remaining > 0) {
    tensor_dims[0] = std::min<int64_t>(remaining, batch_size);
    SoftmaxForwardCudnnKernel<T>(
        dev_ctx, x_data, axis, rank, log_mode, tensor_dims, out_data);
    x_data += offset;
    out_data += offset;
    remaining -= batch_size;
  }
}

template <typename T>
void SoftmaxBackwardCudnnKernel(const GPUContext& dev_ctx,
                                const T* out_data,
                                const T* dout_data,
                                const int axis,
                                const int rank,
                                const bool log_mode,
                                const std::vector<int>& tensor_dims,
                                T* dx_data) {
  auto handle = dev_ctx.cudnn_handle();
  GPUDNNDataLayout layout = GPUDNNDataLayout::kNCHW;

  ScopedTensorDescriptor scoped_desc;
#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t desc =
      scoped_desc.descriptor<T>(layout, tensor_dims);
  auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                               : MIOPEN_SOFTMAX_MODE_CHANNEL;
  auto algo = log_mode ? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSoftmaxBackward_V2(
      handle,
      phi::backends::gpu::CudnnDataType<T>::kOne(),
      desc,
      out_data,
      desc,
      dout_data,
      phi::backends::gpu::CudnnDataType<T>::kZero(),
      desc,
      dx_data,
      algo,
      mode));
#else
  cudnnTensorDescriptor_t desc = scoped_desc.descriptor<T>(layout, tensor_dims);
  auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                               : CUDNN_SOFTMAX_MODE_CHANNEL;
  auto algo = log_mode ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSoftmaxBackward(
      handle,
      algo,
      mode,
      phi::backends::gpu::CudnnDataType<T>::kOne(),
      desc,
      out_data,
      desc,
      dout_data,
      phi::backends::gpu::CudnnDataType<T>::kZero(),
      desc,
      dx_data));
#endif
}

template <typename T>
void LaunchSoftmaxBackwardCudnnKernel(const GPUContext& dev_ctx,
                                      const DenseTensor& out,
                                      const DenseTensor& dout,
                                      const int axis,
                                      const bool log_mode,
                                      DenseTensor* dx) {
  auto* dx_data = dx->data<T>();
  auto* out_data = out.data<T>();
  auto* dout_data = dout.data<T>();
  int rank = out.dims().size();

  std::vector<int> tensor_dims = GetSoftmaxTensorDims(out.dims(), axis);
  int64_t remaining = tensor_dims[0];
  int dim = tensor_dims[1];
  int64_t batch_size = std::numeric_limits<int32_t>::max() / dim;
  int offset = batch_size * dim;
  while (remaining > 0) {
    tensor_dims[0] = std::min<int64_t>(remaining, batch_size);
    SoftmaxBackwardCudnnKernel<T>(dev_ctx,
                                  out_data,
                                  dout_data,
                                  axis,
                                  rank,
                                  log_mode,
                                  tensor_dims,
                                  dx_data);
    out_data += offset;
    dout_data += offset;
    dx_data += offset;
    remaining -= batch_size;
  }
}

template <typename T, typename IndexType, bool LogMode>
void LaunchKeMatrixSoftmaxForwardKernel(
    const GPUContext& dev_ctx, T* out, const T* input, int N, int dim_size) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  constexpr int kVecSize =
      MaxWithOne<MATRIX_SOFTMAX_ALIGN_BYTES / sizeof(T)>::kValue;
  int block_dim = CalcBlockSize(kVecSize, dim_size);
  KeMatrixSoftmaxForward<T, AccT, IndexType, LogMode>
      <<<N, block_dim, 0, dev_ctx.stream()>>>(out, input, dim_size);
}

#if CUDNN_VERSION < 8100
template <>
inline void LaunchSoftmaxForwardCudnnKernel<phi::dtype::bfloat16>(
    const GPUContext& dev_ctx,
    const DenseTensor& x,
    const int axis,
    const bool log_mode,
    DenseTensor* out) {
  PADDLE_THROW(errors::Unavailable(
      "This kernel is not supported when the dtype is bf16 and CUDNN_VERSION < "
      "8100."));
}
template <>
inline void LaunchSoftmaxBackwardCudnnKernel<phi::dtype::bfloat16>(
    const GPUContext& dev_ctx,
    const DenseTensor& out,
    const DenseTensor& dout,
    const int axis,
    const bool log_mode,
    DenseTensor* dx) {
  PADDLE_THROW(errors::Unavailable(
      "This kernel is not supported when the dtype is bf16 and CUDNN_VERSION < "
      "8100."));
}
#endif

template <typename T>
bool UseCudnnSoftmax(const GPUContext& ctx,
                     int64_t softmax_dim,
                     bool last_dim) {
  bool cudnn_available = ctx.cudnn_handle();
  if (!ctx.cudnn_handle()) {
    if (std::is_same<T, phi::dtype::bfloat16>::value) {
#if CUDNN_VERSION < 8100
      cudnn_available = false;
#endif
    }
  }
  constexpr int max_dim = 512;
  if (!cudnn_available || !last_dim ||
      (softmax_dim <= max_dim && sizeof(T) <= 4)) {
    return false;
  } else {
    return true;
  }
}

template <typename T, typename IndexType, bool LogMode = false>
void SoftmaxForwardCUDAKernelDriverImpl(const GPUContext& dev_ctx,
                                        const DenseTensor& x,
                                        const int input_axis,
                                        DenseTensor* out) {
  auto* out_data = out->data<T>();

  int rank = x.dims().size();
  int axis = phi::funcs::CanonicalAxis(input_axis, rank);
  std::vector<IndexType> tensor_dims =
      GetSoftmaxTensorDims<IndexType>(x.dims(), axis);
  IndexType N = tensor_dims[0];
  IndexType dim = tensor_dims[1];
  int D = tensor_dims[2];

  if (D == 1) {
    if (!UseCudnnSoftmax<T>(dev_ctx, dim, true)) {
      int dim_log2 = static_cast<int>(Log2Ceil(dim));
      IndexType dim_ceil = 1 << dim_log2;
      int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
      int batches_per_warp = (dim_ceil <= 32) ? 2 : 1;

      // use 128 threads per block to maximimize gpu utilization
      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / warp_size);
      int batches_per_block = warps_per_block * batches_per_warp;
      IndexType blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(warp_size, warps_per_block, 1);

      // vectorization read/write
      using T4 = typename VecT4<T>::Type;
      using T2 = typename VecT2<T>::Type;

      if (dim % 4 == 0) {
        SwitchWarpSoftmaxForward<T, T4, IndexType, LogMode>(blocks,
                                                            threads,
                                                            dev_ctx,
                                                            out_data,
                                                            x.data<T>(),
                                                            N,
                                                            dim,
                                                            dim,
                                                            dim_log2);
      } else if (dim % 2 == 0) {
        SwitchWarpSoftmaxForward<T, T2, IndexType, LogMode>(blocks,
                                                            threads,
                                                            dev_ctx,
                                                            out_data,
                                                            x.data<T>(),
                                                            N,
                                                            dim,
                                                            dim,
                                                            dim_log2);
      } else {
        SwitchWarpSoftmaxForward<T, T, IndexType, LogMode>(blocks,
                                                           threads,
                                                           dev_ctx,
                                                           out_data,
                                                           x.data<T>(),
                                                           N,
                                                           dim,
                                                           dim,
                                                           dim_log2);
      }
    } else {
      if (dim >= MATRIX_SOFTMAX_THRESHOLD) {
        LaunchKeMatrixSoftmaxForwardKernel<T, IndexType, LogMode>(
            dev_ctx, out_data, x.data<T>(), N, dim);
      } else {
        LaunchSoftmaxForwardCudnnKernel<T>(dev_ctx, x, axis, LogMode, out);
      }
    }
  } else {
    LaunchNormalSoftmaxForward<T, LogMode>(
        dev_ctx, out_data, x.data<T>(), N, dim, D);
  }
}

template <typename T, bool LogMode = false>
void SoftmaxForwardCUDAKernelDriver(const GPUContext& dev_ctx,
                                    const DenseTensor& x,
                                    const int input_axis,
                                    DenseTensor* out) {
  if (x.numel() >= std::numeric_limits<int32_t>::max()) {
    SoftmaxForwardCUDAKernelDriverImpl<T, int64_t, LogMode>(
        dev_ctx, x, input_axis, out);
  } else {
    SoftmaxForwardCUDAKernelDriverImpl<T, int32_t, LogMode>(
        dev_ctx, x, input_axis, out);
  }
}

template <typename T, bool LogMode = false>
void SoftmaxBackwardCUDAKernelDriver(const GPUContext& dev_ctx,
                                     const DenseTensor& out,
                                     const DenseTensor& dout,
                                     const int input_axis,
                                     DenseTensor* dx) {
  auto* dx_data = dx->data<T>();

  int rank = out.dims().size();
  int axis = phi::funcs::CanonicalAxis(input_axis, rank);
  std::vector<int> tensor_dims = GetSoftmaxTensorDims(out.dims(), axis);
  int N = tensor_dims[0];
  int dim = tensor_dims[1];
  int D = tensor_dims[2];

  if (D == 1) {
    if (!UseCudnnSoftmax<T>(dev_ctx, dim, true)) {
      int dim_log2 = Log2Ceil(dim);
      int dim_ceil = 1 << dim_log2;
      int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
      int batches_per_warp = (dim_ceil <= 128) ? 2 : 1;

      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / warp_size);
      int batches_per_block = warps_per_block * batches_per_warp;
      int blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(warp_size, warps_per_block, 1);

      // vectorization read/write
      using T4 = typename VecT4<T>::Type;
      using T2 = typename VecT2<T>::Type;
      if (dim % 4 == 0) {
        SwitchWarpSoftmaxBackward<T, T4, LogMode>(blocks,
                                                  threads,
                                                  dev_ctx,
                                                  dx_data,
                                                  dout.data<T>(),
                                                  out.data<T>(),
                                                  N,
                                                  dim,
                                                  dim,
                                                  dim_log2);
      } else if (dim % 2 == 0) {
        SwitchWarpSoftmaxBackward<T, T2, LogMode>(blocks,
                                                  threads,
                                                  dev_ctx,
                                                  dx_data,
                                                  dout.data<T>(),
                                                  out.data<T>(),
                                                  N,
                                                  dim,
                                                  dim,
                                                  dim_log2);
      } else {
        SwitchWarpSoftmaxBackward<T, T, LogMode>(blocks,
                                                 threads,
                                                 dev_ctx,
                                                 dx_data,
                                                 dout.data<T>(),
                                                 out.data<T>(),
                                                 N,
                                                 dim,
                                                 dim,
                                                 dim_log2);
      }
    } else {
      LaunchSoftmaxBackwardCudnnKernel<T>(
          dev_ctx, out, dout, axis, LogMode, dx);
    }
  } else {
    LaunchNormalSoftmaxBackward<T, LogMode>(
        dev_ctx, dx_data, dout.data<T>(), out.data<T>(), N, dim, D);
  }
}

}  // namespace phi
