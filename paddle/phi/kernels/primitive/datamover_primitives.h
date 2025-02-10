// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#endif
#include "paddle/common/ddim.h"

namespace phi {
namespace kps {
namespace details {

#define INT_BITS 32

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};
/**
 * Fast division : Replace division in CUDA with multiplication to improve
 * kernel performance.
 * 1. Complete the division calculation on the CPU, and record the calculation
 * results by using the divider and shift_val.
 * 2. Set the divisor on the GPU through Div() to complete the calculation.
 */
struct FastDivMod {
  // 1st value represents the result of input number divides by recorded divisor
  // 2nd value represents the result of input number modulo by recorded divisor
  using DivModT = VectorType<uint32_t, 2>;

  FastDivMod() {}
  HOSTDEVICE FastDivMod(uint32_t d) : divisor(d) {
    static_assert(sizeof(unsigned int) == 4,
                  "Only Support 32-bit unsigned int.");

    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      auto shift_limit = 1 << shift_val;
      if (shift_limit >= divisor) break;
    }
    uint64_t long_one = 1;
    uint64_t temp_div =
        ((long_one << INT_BITS) * ((long_one << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
    uint32_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint32_t n) const {
    uint32_t q = Div(n);
    DivModT result = {q, n - q * divisor};
    return result;
  }

  int32_t divisor;
  int32_t shift_val;
  uint32_t multiplier;
};

/**
 * Configuration of broadcast. Calculate the input data index according to the
 * index of the output data. if input or output shape is [dim0, dim1] then dims
 * must be [dim1, dim0].
 */
struct BroadcastConfig {
  FastDivMod divmoders[phi::DDim::kMaxRank];
  uint32_t strides[phi::DDim::kMaxRank];
  int rank{0};

  // BroadcastConfig should be defined on host used on device.
  BroadcastConfig() {}

  BroadcastConfig(const std::vector<int64_t>& out_dims,
                  const std::vector<int64_t>& in_dims,
                  int dim_size) {
    for (int i = 0; i < dim_size; ++i) {
      divmoders[i] = FastDivMod(out_dims[i]);
    }

    for (int i = 0; i < dim_size; ++i) {
      strides[i] = in_dims[i] == 1 ? 0 : 1;
      strides[i] = (i != 0 && strides[i] != 0)
                       ? std::accumulate(in_dims.begin(),
                                         in_dims.begin() + i,
                                         1,
                                         std::multiplies<int64_t>())
                       : strides[i];
    }
    rank = dim_size;
  }
};

template <typename T>
__device__ __forceinline__ void WriteData(T* dst,
                                          T* __restrict__ src,
                                          int num) {
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
  }
}

template <typename T>
__device__ __forceinline__ void ReadData(T* dst,
                                         const T* __restrict__ src,
                                         int num) {
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
  }
}
#undef INT_BITS
}  // namespace details

/**
 * @brief Read 2D data from global memory to register according to Tx type, and
 * store it as Ty type into register.
 *
 * @template paraments
 * Tx: The type of data stored in the global memory.
 * Ty: The type of data that needs to be stored in registers.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size_nx: The maximum offset of the current block is size_nx elements in the
 * lowest dimension. The parameters are only calculated when isboundary = true.
 * size_ny: The maximum offset of the current block is size_ny elements in the
 * first dimension. The parameters are only calculated when isboundary = true.
 * stride_nx: Each read one element stride stride_nx elements in the last dim.
 * stride_ny: Each read one element stride stride_ny elements in the first dim.
 */
template <typename Tx, typename Ty, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(Ty* dst,
                                         const Tx* __restrict__ src,
                                         int size_nx,
                                         int size_ny,
                                         int stride_nx,
                                         int stride_ny) {
  int thread_offset = threadIdx.x;
  int left_size_nx = size_nx - thread_offset;

  // Each branch is added for better performance
  if (NX == 1 && NY == 1) {  // for NX == 1 and NY == 1
    if (IsBoundary) {
      if (left_size_nx > 0) {
        dst[0] = static_cast<Ty>(src[thread_offset]);
      }
    } else {
      dst[0] = static_cast<Ty>(src[thread_offset]);
    }
  } else if (NX == 1) {  // for NX == 1 and NY != 1
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy * stride_ny >= size_ny) {
          break;
        }
      }
      dst[idy] = static_cast<Ty>(src[thread_offset + idy * stride_ny]);
    }
  } else if (NY == 1) {  // for NY == 1 and NX != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
      dst[idx] = static_cast<Ty>(src[thread_offset + idx * stride_nx]);
    }
  } else {  // for NX != 1 and NY != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        if (IsBoundary) {
          if (idy * stride_ny >= size_ny) {
            break;
          }
        }
        dst[idy * NX + idx] = static_cast<Ty>(
            src[thread_offset + idx * stride_nx + idy * stride_ny]);
      }
    }
  }
}

/**
 * @brief Initialize register with init_data.
 *
 * @template paraments
 * T: Data type of register.
 * NX: Number of data to initialize.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * init_data: Initial value.
 */
template <typename T, int NX>
__device__ __forceinline__ void Init(T* dst, T init_data) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    dst[i] = init_data;
  }
}

template <typename T, int NX>
__device__ __forceinline__ void Init(T* dst, T init_data, int read_lens) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    dst[i] = init_data;
  }
}

/**
 * The difference from the above function is that
 * it supports different data types of inputs.
 */
template <typename T, typename ArgsT, int Index, int NX>
__device__ __forceinline__ void Init(ArgsT* dst, T init_data, int read_lens) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    std::get<Index>(dst[i]) = init_data;
  }
}

/**
 * The difference from the above function is that
 * it supports different data types of inputs.
 */
template <typename T, typename ArgsT, int Index, int NX>
__device__ __forceinline__ void Init(ArgsT* dst, T init_data) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    std::get<Index>(dst[i]) = init_data;
  }
}

/**
 * @brief Read 1D data from global memory to register. When IsBoundary = true
 * and (NX % 4 == 0 or Nx % 2 == 0), vectorized load data will be used to
 * improve memory access efficiency.
 *
 * @template paraments
 * T: The type of data.
 * NX: Each thread load NX data from global memory continuously.
 * NY: Each thread need to load NY rows, only NY = 1 was supported.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Whether to make an out-of-bounds judgment on access to memory.
 * When the number of data processed by this block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size: The current block needs to load size data continuously.
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T* dst,
                                         const T* __restrict__ src,
                                         int num) {
  if (IsBoundary) {  // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        dst[idx] = src[thread_offset + idx];
      }
    }
  } else {  // blockDim,x * NX < num
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;
    int thread_offset = threadIdx.x * kVectorsPerThread;

    using VecType = details::VectorType<T, kVectorSize>;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[thread_offset + i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
      }
    }
  }
}

/**
 * @brief Read 1D data from global memory to register.
 * @template paraments
 * T: The type of data.
 * NX: Each thread load NX data from global memory continuously.
 * NY: Each thread need to load NY rows, only NY = 1 was supported.
 * IsBoundary: Whether to make an out-of-bounds judgment on access to memory.
 * When the number of data processed by this block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size: The current block needs to load size data continuously.
 */

template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T* dst,
                                         const T* __restrict__ src,
                                         int num,
                                         int read_lens) {
  if (IsBoundary) {  // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        dst[idx] = src[thread_offset + idx];
      }
    }
  } else {  // blockDim,x * NX < num
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;
    int thread_offset = threadIdx.x * kVectorsPerThread;

    using VecType = details::VectorType<T, kVectorSize>;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[thread_offset + i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
      }
    }
  }
}
/**
 * @brief Read 1D data from global memory to register. The difference
 * from the above function is that it supports different data types of inputs.
 *
 * @template paraments
 * T: The type of data.
 * NX: Each thread load NX data from global memory continuously.
 * NY: Each thread need to load NY rows, only NY = 1 was supported.
 * ArgsT: The Type of dst, ArgsT can be std::tuple<T> or std::tuple<Args>
 * Index: The index of data stored in dst.
 * IsBoundary: Whether to make an out-of-bounds judgment on access to memory.
 * When the number of data processed by this block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size: The current block needs to load size data continuously.
 */
template <typename T,
          int NX,
          int NY,
          typename ArgsT,
          int Index,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadData(ArgsT* dst,
                                         const T* __restrict__ src,
                                         int num,
                                         int read_lens = 0) {
  if (IsBoundary) {  // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        std::get<Index>(dst[idx]) = src[thread_offset + idx];
      }
    }
  } else {  // blockDim,x * NX < num
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;
    int thread_offset = threadIdx.x * kVectorsPerThread;

    using VecType = details::VectorType<T, kVectorSize>;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[thread_offset + i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        std::get<Index>(dst[idx]) = *(reinterpret_cast<T*>(vec_temp) + idx);
      }
    }
  }
}

/**
 * @brief Read 2D data from global memory to registers with broadcast form.
 *
 * @template paraments
 * T: The type of data stored in the global memory.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The original input data pointer of this kernel.
 * block_offset: The data offset of this block, blockDim.x * blockIdx.x * NX.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * total_num_output: Total number of original output.
 * stride_nx: Each read one element stride stride_nx elements in the last dim.
 * stride_ny: Each read one element stride stride_ny elements in the first dim.
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadDataBc(
    T* dst,
    const T* __restrict__ src,
    uint32_t block_offset,
    const details::BroadcastConfig& config,
    int total_num_output,
    int stride_nx,
    int stride_ny) {
  uint32_t thread_offset = block_offset + threadIdx.x;
  uint32_t index_src = 0;

#pragma unroll
  for (int ny = 0; ny < NY; ++ny) {
#pragma unroll
    for (uint32_t nx = 0; nx < NX; ++nx) {
      uint32_t index_output = thread_offset + ny * stride_ny + nx * stride_nx;
      index_src = 0;
      if (IsBoundary) {
        if (index_output >= total_num_output) {
          break;
        }
      }
#pragma unroll
      for (int i = 0; i < phi::DDim::kMaxRank; ++i) {
        if (i >= config.rank) break;
        auto fast_divmoder = config.divmoders[i].Divmod(index_output);
        index_output = fast_divmoder.val[0];
        index_src += fast_divmoder.val[1] * config.strides[i];
      }
      dst[nx + ny * NX] = src[index_src];
    }
  }
}

/**
 * @brief Read 2D data from global memory to register with reduce form.
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The input data pointer of this block.
 * block_offset: The data offset of this block, blockDim.x * blockIdx.x * NX.
 * index_cal: Calculation configuration of Reduce. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * size_nx: The current block needs to load size_nx columns of data, this
 * parameter will participate in the calculation when isboundary = true.
 * size_ny: The current block needs to load size_ny rows of data, this parameter
 * will participate in the calculation when isboundary = true.
 * will be used when IsBoundary = true.
 * stride_nx: Each read one element stride stride_nx columns.
 * stride_ny: Each read one element stride stride_ny raws.
 * reduce_last_dim: Used to indicate whether the dimension of reduce contains
 * the lowest dimension.
 */
template <typename Tx,
          typename Ty,
          int NX,
          int NY,
          int Rank,
          typename IndexCal,
          typename Functor,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadDataReduce(Ty* dst,
                                               const Tx* __restrict__ src,
                                               int block_offset,
                                               const IndexCal& index_cal,
                                               int size_nx,
                                               int size_ny,
                                               int stride_nx,
                                               int stride_ny,
                                               Functor func,
                                               bool reduce_last_dim) {
  int thread_offset = 0;
  int left_idx = 0;
  if (reduce_last_dim) {
    thread_offset = threadIdx.x;
    left_idx = threadIdx.y;
  } else {
    thread_offset = threadIdx.y;
    left_idx = threadIdx.x;
  }

  if (NX == 1) {
#pragma unroll
    for (int ny = 0; ny < NY; ++ny) {
      if (IsBoundary) {
        if (thread_offset >= size_ny) {
          break;
        }
      }
      uint32_t index_src = index_cal(thread_offset + block_offset);
      dst[ny] = static_cast<Ty>(func(src[index_src]));
      thread_offset += stride_ny;
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < NX; ++nx) {
#pragma unroll
      for (int ny = 0; ny < NY; ++ny) {
        if (IsBoundary) {
          if ((thread_offset >= size_ny) ||
              (left_idx + nx * stride_nx >= size_nx)) {
            break;
          }
        }
        uint32_t index_src = index_cal(thread_offset + block_offset);
        dst[nx + ny * NX] = static_cast<Ty>(func(src[index_src]));
        thread_offset += stride_ny;
      }
    }
  }
}

/**
 * @brief Write 2D data from registers to global memory. When IsBoundary = true
 * and (NX % 4 == 0 or Nx % 2 == 0), the data will be vectorized to improve the
 * data loading efficiency
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data continuously written by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The data pointer of the current block.
 * src: The register pointer, the size is NX * NY.
 * size: The current block needs to load size elements continuously.
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(T* dst,
                                          T* __restrict__ src,
                                          int64_t num) {
  if (IsBoundary) {
    int64_t thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if ((thread_offset + idx) < num) {
        dst[thread_offset + idx] = src[idx];
      }
    }
  } else {
    // Vector type
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;

    int64_t thread_offset = threadIdx.x * kVectorsPerThread;
    using VecType = details::VectorType<T, kVectorSize>;
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    VecType vec_temp[kVectorsPerThread];
#pragma unroll
    for (int idx = 0; idx < kVectorsPerThread; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType*>(src) + idx);
      vec_dst[thread_offset + idx] = vec_temp[idx];
    }
  }
}

template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(T* dst,
                                          T* __restrict__ src,
                                          int num,
                                          int read_lens) {
  if (IsBoundary) {
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if ((thread_offset + idx) < num) {
        dst[thread_offset + idx] = src[idx];
      }
    }
  } else {
    // Vector type
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;

    int thread_offset = threadIdx.x * kVectorsPerThread;
    using VecType = details::VectorType<T, kVectorSize>;
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    VecType vec_temp[kVectorsPerThread];
#pragma unroll
    for (int idx = 0; idx < kVectorsPerThread; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType*>(src) + idx);
      vec_dst[thread_offset + idx] = vec_temp[idx];
    }
  }
}

/**
 * @brief Write 2D data from register to global memory according to Tx type, and
 * store it as Ty type.
 *
 * @template paraments
 * Tx: The type of data that needs to be stored in registers.
 * Ty: The type of data that stored in the global memory.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The data pointer of the current block.
 * src: The register pointer of the thread, the size is NX * NY.
 * size_nx: The maximum offset of the current block is size_nx elements in the
 * lowest dimension. The parameters are only calculated when isboundary = true.
 * size_ny: The maximum offset of the current block is size_ny elements in the
 * first dimension. The parameters are only calculated when isboundary = true.
 * stride_nx: Each read one element stride stride_nx elements in the last dim.
 * stride_ny: Each read one element stride stride_ny elements in the first dim.
 */
template <typename Tx, typename Ty, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(Ty* dst,
                                          const Tx* __restrict__ src,
                                          int64_t size_nx,
                                          int size_ny,
                                          int stride_nx,
                                          int stride_ny) {
  int thread_offset = threadIdx.x;
  int64_t left_size_nx = size_nx - thread_offset;

  // Each branch is added for better performance
  if (NX == 1 && NY == 1) {  // for NX == 1 and NY == 1
    if (IsBoundary) {
      if (left_size_nx > 0) {
        dst[thread_offset] = static_cast<Ty>(src[0]);
      }
    } else {
      dst[thread_offset] = static_cast<Ty>(src[0]);
    }
  } else if (NX == 1) {  // for NX == 1 and NY != 1
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy * stride_ny >= size_ny) {
          break;
        }
      }
      dst[thread_offset + idy * stride_ny] = static_cast<Ty>(src[idy]);
    }
  } else if (NY == 1) {  // for NY == 1 and NX != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
      dst[thread_offset + idx * stride_nx] = static_cast<Ty>(src[idx]);
    }
  } else {  // for NX != 1 and NY != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        if (IsBoundary) {
          if (idy * stride_ny >= size_ny) {
            break;
          }
        }
        dst[thread_offset + idx * stride_nx + idy * stride_ny] =
            static_cast<Ty>(src[idy * NX + idx]);
      }
    }
  }
}

/**
 * @brief Initialize register with init_data.
 *
 * @template paraments
 * T: Data type of register.
 * NX: Number of data to initialize.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * init_data: The register pointer of init data, the size is NX.
 */
template <typename T, int NX, bool IsBoundary = false>
__device__ __forceinline__ void Init(T* dst, T* init_data, int num) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    if (IsBoundary) {
      if (i >= num) {
        break;
      }
    }
    dst[i] = init_data[i];
  }
}

/**
 * @brief Read 1D data from global memory to register with broadcast form.
 *
 * @template paraments
 * T: The type of data stored in the global memory.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The original input data pointer of kernel.
 * block_offset: The data offset of this block, blockDim.x * blockIdx.x * NX;
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * total_num_output: Total number of original output.
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadDataBc(
    T* dst,
    const T* __restrict__ src,
    uint32_t block_offset,
    const details::BroadcastConfig& config,
    int total_num_output,
    int read_lens = NX) {
  uint32_t thread_offset = block_offset + threadIdx.x * NX;
  uint32_t index_src = 0;

#pragma unroll
  for (uint32_t nx = 0; nx < NX; ++nx) {
    uint32_t index_output = thread_offset + nx;
    index_src = 0;
    if (IsBoundary) {
      if (index_output >= total_num_output) {
        break;
      }
    }
#pragma unroll
    for (int i = 0; i < phi::DDim::kMaxRank; ++i) {
      if (i >= config.rank) break;
      auto fast_divmoder = config.divmoders[i].Divmod(index_output);
      index_output = fast_divmoder.val[0];
      index_src += fast_divmoder.val[1] * config.strides[i];
    }
    dst[nx] = src[index_src];
  }
}

/**
 * @brief Read 1D data from global memory to register with broadcast form.
 * The difference from the above function is that it supports different data
 * types of inputs.
 *
 * @template paraments
 * T: The type of data stored in the global memory.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * ArgsT: The Type of dst, ArgsT can be std::tuple<T> or std::tuple<Args>
 * Index: The index of data stored in dst.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The original input data pointer of kernel.
 * block_offset: The data offset of this block, blockDim.x * blockIdx.x * NX;
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * total_num_output: Total number of original output.
 */

template <typename T,
          int NX,
          int NY,
          typename ArgsT,
          int Index,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadDataBc(
    ArgsT* dst,
    const T* __restrict__ src,
    uint32_t block_offset,
    const details::BroadcastConfig& config,
    int total_num_output,
    int read_lens = NX) {
  uint32_t thread_offset = block_offset + threadIdx.x * NX;
  uint32_t index_src = 0;

#pragma unroll
  for (uint32_t nx = 0; nx < NX; ++nx) {
    uint32_t index_output = thread_offset + nx;
    index_src = 0;
    if (IsBoundary) {
      if (index_output >= total_num_output) {
        break;
      }
    }
#pragma unroll
    for (int i = 0; i < phi::DDim::kMaxRank; ++i) {
      if (i >= config.rank) break;
      auto fast_divmoder = config.divmoders[i].Divmod(index_output);
      index_output = fast_divmoder.val[0];
      index_src += fast_divmoder.val[1] * config.strides[i];
    }
    std::get<Index>(dst[nx]) = src[index_src];
  }
}

/**
 * @brief Initialize register with data index.
 *
 * @template paraments
 * T: Data type of register.
 * NX: Number of data to initialize.
 * NY: Number of data to initialize, NY only can be 1.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * init_data: The register pointer of init data, the size is NX.
 */
template <typename T, int NX, int NY>
__device__ __forceinline__ void InitWithDataIndex(T* dst, int block_offset) {
  int thread_offset = block_offset + threadIdx.x * NX;
#pragma unroll
  for (int nx = 0; nx < NX; ++nx) {
    dst[nx] = static_cast<T>(thread_offset + nx);
  }
}

}  // namespace kps
}  // namespace phi
