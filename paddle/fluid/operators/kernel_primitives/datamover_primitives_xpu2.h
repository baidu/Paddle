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
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

/**
 * Configuration of broadcast. Calculate the input data index according to the
 * index of the output data. if input or output shape is [dim0, dim1] then dims
 * must be [dim1, dim0].
 */
template <int kDims>
struct BroadcastConfig {
  uint32_t stride_in[framework::DDim::kMaxRank];
  uint32_t stride_out[framework::DDim::kMaxRank];
  uint32_t shape_in[framework::DDim::kMaxRank];

  HOSTDEVICE BroadcastConfig() {}

  HOSTDEVICE BroadcastConfig(const std::vector<int64_t>& out_dims,
                             const std::vector<int64_t>& in_dims,
                             int dim_size) {
    std::vector<uint32_t> strides_in;
    std::vector<uint32_t> strides_out;
    std::vector<uint32_t> shapes_in;

    strides_out.resize(dim_size, 1);
    strides_in.resize(dim_size, 1);
    shapes_in.resize(dim_size, 1);

    for (int i = 0; i < dim_size; ++i) {
      shape_in[i] = in_dims[dim_size - i - 1];
    }

    for (int i = 1; i < dim_size - 1; ++i) {
      strides_out[dim_size - i - 1] = std::accumulate(
          out_dims.begin(), out_dims.begin() + i, 1, std::multiplies<int64_t>())
          strides_in[dim_size - i - 1] =
              std::accumulate(in_dims.begin(), in_dims.begin() + i, 1,
                              std::multiplies<int64_t>())
    }

    memcpy(stride_in, strides_in.data(), kDims * sizeof(uint32_t));
    memcpy(stride_out, strides_out.data(), kDims * sizeof(uint32_t));
    memcpy(shape_in, shapes_in.data(), kDims * sizeof(uint32_t));
  }
};

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
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
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
template <typename Tx, typename Ty, int NX, int NY, int BlockSize,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadData(Ty* dst, const Tx _global_ptr_* src,
                                         int size_nx, int size_ny,
                                         int stride_nx, int stride_ny) {
  int thread_offset = core_id();
  int left_size_nx = size_nx - thread_offset;
  __local__ T in_temp[1];
  // Each branch is added for better performance
  if (NX == 1 && NY == 1) {  // for NX == 1 and NY == 1
    if (IsBoundary) {
      if (left_size_nx > 0) {
        GM2LM(src + thread_offset, in_temp, sizeof(Tx));
        dst[0] = static_cast<Ty>(in_temp[0]);
      }
    } else {
      GM2LM(src + thread_offset, in_temp, sizeof(Tx));
      dst[0] = static_cast<Ty>(in_temp[0]);
    }
  } else if (NX == 1) {  // for NX == 1 and NY != 1
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy * stride_ny >= size_ny) {
          break;
        }
      }
      GM2LM(src + thread_offset + idy * stride_ny, in_temp, sizeof(Tx));
      dst[idy] = static_cast<Ty>(in_temp[0]);
    }
  } else if (NY == 1) {  // for NY == 1 and NX != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
      GM2LM(src + thread_offset + idx * stride_nx, in_temp, sizeof(Tx));
      dst[idx] = static_cast<Ty>(in_temp[0]);
    }
  } else {  // for NX != 1 and NY != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        if (IsBoundary) {
          if (idy * stride_ny >= size_ny || idx * stride_nx >= left_size_nx) {
            break;
          }
        }
        int fix = thread_offset + idx * stride_nx + idy * stride_ny;
        GM2LM(src + fix, in_temp, sizeof(Tx));
        dst[idy * NX + idx] = static_cast<Ty>(in_temp[0]);
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

/**
 * @brief Read 1D data from global memory to register. When IsBoundary = true
 * and (NX % 4 == 0 or Nx % 2 == 0), vectorized load data will be used to
 * improve memory access efficiency.
 *
 * @template paraments
 * T: The type of data.
 * NX: Each thread load NX data from global memory continuously.
 * NY: Each thread need to load NY rows, only NY = 1 was supported.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * IsBoundary: Whether to make an out-of-bounds judgment on access to memory.
 * When the number of data processed by this block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size: The current block needs to load size data continuously.
 */
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T* dst, const T _global_ptr_* src,
                                         int num) {
  int thread_offset = core_id() * NX;
  __local__ T in_temp[1];
  if (IsBoundary) {  // core_num() * NX > num
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        GM2LM(src + thread_offset + idx, in_temp, sizeof(T));
        dst[idx] = in_temp[0];
      }
    }
  } else {  // core_num() * NX < num
    GM2LM(src + thread_offset, dst, NX * sizeof(T));
  }
}

/**
 * @brief Read 2D data from global memory to registers with broadcast form.
 *
 * @template paraments
 * T: The type of data stored in the global memory.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: Raw input data pointer of kernel.
 * block_offset: Data offset of this block, core_num() *  cluster_id() * NX;
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * total_num_output: Total number of original output.
 * stride_nx: Each read one element stride stride_nx elements in the last dim.
 * stride_ny: Each read one element stride stride_ny elements in the first dim.
 */
template <typename T, int NX, int NY, int BlockSize, int Rank,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadDataBc(
    T* dst, const T _global_ptr_* src, uint32_t block_offset,
    details::BroadcastConfig<Rank> config, int total_num_output, int stride_nx,
    int stride_ny) {
  uint32_t thread_offset = block_offset + core_id();
  uint32_t index_src = 0;
  __local__ T in_temp[1];

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
      for (int i = 0; i < Rank; ++i) {
        uint32_t tmp = index_output / config.stride_out[i];
        index_output = index_output - tmp * config.stride_out[i];
        index_src += (tmp % config.shape_in[i]) * config.stride_in[i];
      }
      GM2LM(src + index_src, in_temp, sizeof(T));
      dst[nx + ny * NX] = in_temp[0];
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
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The input data pointer of this block.
 * block_offset: The data offset of this block, blockDim.x * cluster_id() * NX.
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
template <typename T, int NX, int NY, int BlockSize, int Rank,
          typename IndexCal, bool IsBoundary = false>
__device__ __forceinline__ void ReadDataReduce(
    T* dst, const T _global_ptr_* src, int block_offset,
    const IndexCal& index_cal, int size_nx, int size_ny, int stride_nx,
    int stride_ny, bool reduce_last_dim) {
  __local__ T in_temp[1];
  int thread_offset = 0;
  int left_size_nx = size_nx;
  int left_size_ny = size_ny;
  if (reduce_last_dim) {
    thread_offset = block_offset + core_id();
    left_size_nx -= thread_offset;
  } else {
    thread_offset = block_offset + core_id();
    left_size_ny -= thread_offset;
  }

  if (NX == 1) {
#pragma unroll
    for (int ny = 0; ny < NY; ++ny) {
      if (IsBoundary) {
        if (ny * stride_ny >= left_size_ny) {
          break;
        }
      }
      uint32_t index_src = index_cal(thread_offset);
      GM2LM(src + index_src, in_temp, sizeof(T));
      dst[ny] = in_temp[0];
      thread_offset += stride_ny;
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < NX; ++nx) {
#pragma unroll
      for (int ny = 0; ny < NY; ++ny) {
        if (IsBoundary) {
          if ((ny * stride_ny >= left_size_ny) ||
              (nx * stride_nx >= left_size_nx)) {
            break;
          }
        }
        uint32_t index_src = index_cal(thread_offset);
        GM2LM(src + index_src, in_temp, sizeof(T));
        dst[nx + ny * NX] = in_temp[0];
        thread_offset += stride_ny;
      }
      thread_offset += stride_nx;
    }
  }
}
/**
 * @brief Write 1D data from registers to global memory. When IsBoundary = true
 * and (NX % 4 == 0 or Nx % 2 == 0), the data will be vectorized to improve the
 * data loading efficiency
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data continuously writed by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The data pointer of the current block.
 * src: The register pointer, the size is NX * NY.
 * size: The current block needs to load size elements continuously.
 */

template <typename T, int NX, int NY, int BlockSize, bool IsBoundary>
__device__ void WriteData(T _global_ptr_* dst, const T* src, int num) {
  int thread_offset = core_id() * NX;
  __local__ T in_temp[1];
  if (IsBoundary) {  // core_num() * NX > num
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        in_temp[0] = src[idx];
        LM2GM(in_temp, dst + idx + thread_offset, sizeof(T));
      }
    }
  } else {  // core_num() * NX < num
    LM2GM(src, dst + thread_offset, NX * sizeof(T));
  }
}

/**
 * @brief Write 2D data from register to global memory according to Tx type, and
 * store it as Ty type.
 *
 * @template paraments
 * Tx: The type of data that needs to be stored in registers.
 * Ty: The type of data stored in the global memory.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: Data pointer of the current block.
 * src: The register pointer of the thread, the size is NX * NY.
 * size_nx: The current block needs to load size_nx columns of data, this
 * parameter will be used when IsBoundary = true.
 * size_ny: The current block needs to load size_ny rows of data. This parameter
 * will be used when IsBoundary = true.
 * stride_nx: Each read one element stride stride_nx elements in the last dim.
 * stride_ny: Each read one element stride stride_ny elements in the first dim.
 */
template <typename Tx, typename Ty, int NX, int NY, int BlockSize,
          bool IsBoundary = false>
__device__ __forceinline__ void WriteData(Ty _global_ptr_* dst, const Tx* src,
                                          int size_nx, int size_ny,
                                          int stride_nx, int stride_ny) {
  int thread_offset = core_id();
  int left_size_nx = size_nx - thread_offset;
  __local__ Ty in_temp[1];

  // Each branch is added for better performance
  if (NX == 1 && NY == 1) {  // for NX == 1 and NY == 1
    if (IsBoundary) {
      if (left_size_nx > 0) {
        in_temp[0] = static_cast<Ty>(src[0]);
        LM2GM(in_temp, dst + thread_offset, sizeof(T));
      }
    } else {
      in_temp[0] = static_cast<Ty>(src[0]);
      LM2GM(in_temp, dst + thread_offset, sizeof(T));
    }
  } else if (NX == 1) {  // for NX == 1 and NY != 1
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy * stride_ny >= size_ny) {
          break;
        }
      }

      in_temp[0] = static_cast<Ty>(src[idy]);
      LM2GM(in_temp, dst + thread_offset + idy * stride_ny, sizeof(T));
    }
  } else if (NY == 1) {  // for NY == 1 and NX != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }

      in_temp[0] = static_cast<Ty>(src[idx]);
      LM2GM(in_temp, dst + thread_offset + idx * stride_nx, sizeof(T));
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
        in_temp[0] = static_cast<Ty>(src[idx + idy * NX]);
        LM2GM(in_temp, dst + thread_offset + idx * stride_nx + idy * stride_ny,
              sizeof(T));
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
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x core_num(), boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The original input data pointer of kernel.
 * block_offset: The data offset of this block, core_num() * blockIdx.x * NX;
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * total_num_output: Total number of original output.
 */
template <typename T, int NX, int NY, int BlockSize, int Rank,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadDataBc(
    T* dst, const T _global_ptr_* src, uint32_t block_offset,
    details::BroadcastConfig<Rank> config, int total_num_output) {
  uint32_t thread_offset = block_offset + core_id() * NX;
  uint32_t index_src = 0;
  __local__ T in_temp[1];

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
    for (int i = 0; i < Rank; ++i) {
      uint32_t tmp = index_output / config.stride_out[i];
      index_output = index_output - tmp * config.stride_out[i];
      index_src += (tmp % config.shape_in[i]) * config.stride_in[i];
    }
    GM2LM(src + index_src, in_temp, sizeof(T));
    dst[nx + ny * NX] = in_temp[0];
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
