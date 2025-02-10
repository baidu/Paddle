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

namespace phi {
namespace kps {
namespace details {

static inline int RoundUpDiv(int n, int k) { return (n + k - 1) / k; }

static inline int GetXpuReadLens(int numel, int block_num, int grid_num) {
  const int buf_size = 256;
  int nthreads = block_num * grid_num;
  if (numel / nthreads == 1) {
    return numel / nthreads * 4;
  }
  int read_lens = std::min(buf_size, RoundUpDiv(numel, 32 * nthreads) * 32);
  return read_lens;
}

enum class OptType {    // Optimize type of calc after input shape compressed
  CanNotOptimize = -1,  // can not optimize, broadcast first
  N_1,                  // just like {1} op {100} or {100} op {1}
  MN_N,                 // just like {100} op {3, 100} or {3, 100} op {100}
  MN_M,                 // just like {3} op {3, 100} or {3, 100} op {3}
  MNK_1N1,              // just like {3} op {2, 3, 100} or {2, 3, 100} op {3}
  MNK_M1K,  // just like {2, 1, 100} op {2, 3, 100} or {2, 3, 100} op {2, 1,
            // 100}
};

// Rules to determine whether dimensions can be merged
// rule 0 - xshape[idx] == yshape[idx]
// rule 1 - xshape[idx] == 1 && yshape[idx] != 1
// rule 2 - xshape[idx] != 1 && yshape[idx] == 1
static int judge_case(int a, int b) {
  if (a == b) {
    return 0;
  } else if (a == 1 && b != 1) {
    return 1;
  } else if (a != 1 && b == 1) {
    return 2;
  }
  return -1;
}

static bool case_is_same(int case_front, int case_back) {
  if (case_front == case_back) {
    return true;
  } else {
    return false;
  }
}

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

/**
 * Configuration of broadcast. Calculate the input data index according to the
 * index of the output data. if input or output shape is [dim0, dim1] then dims
 * must be [dim1, dim0].
 */
#pragma pack(4)
struct BroadcastConfig {
  int strides_in[phi::DDim::kMaxRank];
  int strides_out[phi::DDim::kMaxRank];
  int in_dim[phi::DDim::kMaxRank];
  int dim_after_cmp[phi::DDim::kMaxRank];
  int y_dim_after_cmp[phi::DDim::kMaxRank];
  int dim_size_after_cmp = 0;
  int cmp_res = 0;
  OptType cmp_type = OptType::CanNotOptimize;
  int m = 1;
  int n = 1;
  int k = 1;
  int buf_len = 0;
  int kDims;
  HOSTDEVICE BroadcastConfig() {}

  HOSTDEVICE BroadcastConfig(const std::vector<int64_t>& out_dims,
                             const std::vector<int64_t>& in_dims,
                             const std::vector<int64_t>& y_in_dims,
                             int dim_size) {
    std::vector<int> strides_in_tmp;
    std::vector<int> strides_out_tmp;
    std::vector<int> dim_tmp;
    strides_in_tmp.resize(dim_size, 1);
    strides_out_tmp.resize(dim_size, 1);
    dim_tmp.resize(dim_size, 1);
    for (int i = 1; i < dim_size; i++) {
      strides_in_tmp[i] = strides_in_tmp[i - 1] * in_dims[i - 1];
      strides_out_tmp[i] = strides_out_tmp[i - 1] * out_dims[i - 1];
    }

    int numel_out = 1;
    for (int i = 0; i < dim_size; i++) {
      dim_tmp[i] = in_dims[i];
      numel_out = out_dims[i] * numel_out;
    }
    kDims = dim_size;
    memcpy(strides_in, strides_in_tmp.data(), kDims * sizeof(int));
    memcpy(strides_out, strides_out_tmp.data(), kDims * sizeof(int));
    memcpy(in_dim, dim_tmp.data(), kDims * sizeof(int));

    cmp_res = get_mnk_for_broadcast_ops(in_dims, y_in_dims);
    get_opt_type();
    buf_len = get_buf_len(numel_out);
    int numel_x = 1;
    int numel_y = 1;
    for (int i = 0; i < dim_size; i++) {
      numel_x = in_dims[i] * numel_x;
      numel_y = y_in_dims[i] * numel_y;
    }
    if (numel_out == numel_x && numel_out == numel_y) {
      buf_len = GetXpuReadLens(numel_out, 8, 64);
    }
  }

  int get_buf_len(int numel) {
    if (cmp_type == OptType::CanNotOptimize) {
      return 256;
    }
    if (cmp_type == OptType::N_1) {
      return kps::details::GetXpuReadLens(numel, 8, 64);
    }
    int max_buf_len = 512;
    int buf_len = m / 16 * 16;
    if (buf_len == 0) {
      buf_len = m;
    }
    return std::min(max_buf_len, buf_len);
  }

  __device__ inline int operator()(int index_output) const {
    int index_src = 0;

    switch (cmp_type) {
      int div, mod, tmp_index;
      case OptType::MNK_M1K:
        div = index_output / (m * n);
        mod = index_output % (m * n) % m;
        index_src = div * m + mod;
        break;
      case OptType::MNK_1N1:
        // index_src = index_output / m % n;
        index_src = index_output % (m * n) / m;
        break;
      case OptType::N_1:
        index_src = 0;
        break;
      case OptType::MN_N:
        index_src = index_output / m;
        break;
      case OptType::MN_M:
        index_src = index_output % m;
        break;
      case OptType::CanNotOptimize:
        for (int i = kDims - 1; i >= 0; --i) {
          tmp_index = (index_output / strides_out[i]);
          index_output = index_output - tmp_index * strides_out[i];
          index_src += (tmp_index % in_dim[i]) * strides_in[i];
        }
        break;
    }
    return index_src;
  }

  void get_opt_type() {
    if (dim_size_after_cmp == 1) {
      if (dim_after_cmp[0] == 1 && y_dim_after_cmp[0] != 1) {  // {1} op {n}
        n = y_dim_after_cmp[0];
        cmp_type = OptType::N_1;
      } else if (dim_after_cmp[0] != 1 &&
                 y_dim_after_cmp[0] == 1) {  // {n} op {1}
        n = dim_after_cmp[0];
        cmp_type = OptType::N_1;
      } else {
        cmp_type = OptType::CanNotOptimize;  // xshape == yshape
      }
    }
    if (dim_size_after_cmp == 2) {
      if (dim_after_cmp[0] == 1 && dim_after_cmp[1] != 1 &&
          y_dim_after_cmp[0] != 1 &&
          y_dim_after_cmp[1] != 1) {  // {n} op {m, n}
        m = y_dim_after_cmp[0];
        n = y_dim_after_cmp[1];
        cmp_type = OptType::MN_N;
      } else if (dim_after_cmp[0] != 1 && dim_after_cmp[1] == 1 &&
                 y_dim_after_cmp[0] != 1 &&
                 y_dim_after_cmp[1] != 1) {  // {m} op {m, n}
        m = y_dim_after_cmp[0];
        n = y_dim_after_cmp[1];
        cmp_type = OptType::MN_M;
      } else if (dim_after_cmp[0] != 1 && dim_after_cmp[1] != 1 &&
                 y_dim_after_cmp[0] == 1 &&
                 y_dim_after_cmp[1] != 1) {  // {m, n} op {n}
        m = dim_after_cmp[0];
        n = dim_after_cmp[1];
        cmp_type = OptType::MN_N;
      } else if (dim_after_cmp[0] != 1 && dim_after_cmp[1] != 1 &&
                 y_dim_after_cmp[0] != 1 &&
                 y_dim_after_cmp[1] == 1) {  // {m, n} op {m}
        m = dim_after_cmp[0];
        n = dim_after_cmp[1];
        cmp_type = OptType::MN_M;
      } else {
        cmp_type = OptType::CanNotOptimize;
      }
    }
    if (dim_size_after_cmp == 3) {
      if (dim_after_cmp[0] == 1 && dim_after_cmp[1] != 1 &&
          dim_after_cmp[2] == 1 && y_dim_after_cmp[0] != 1 &&
          y_dim_after_cmp[1] != 1 &&
          y_dim_after_cmp[2] != 1) {  // {1, n, 1} op {m, n, k}
        m = y_dim_after_cmp[0];
        n = y_dim_after_cmp[1];
        k = y_dim_after_cmp[2];
        cmp_type = OptType::MNK_1N1;
      } else if (dim_after_cmp[0] != 1 && dim_after_cmp[1] != 1 &&
                 dim_after_cmp[2] != 1 && y_dim_after_cmp[0] == 1 &&
                 y_dim_after_cmp[1] != 1 &&
                 y_dim_after_cmp[2] == 1) {  // {m, n, k} op {1, n, 1}
        m = dim_after_cmp[0];
        n = dim_after_cmp[1];
        k = dim_after_cmp[2];
        cmp_type = OptType::MNK_1N1;
      } else if (dim_after_cmp[0] != 1 && dim_after_cmp[1] == 1 &&
                 dim_after_cmp[2] != 1 && y_dim_after_cmp[0] != 1 &&
                 y_dim_after_cmp[1] != 1 &&
                 y_dim_after_cmp[2] != 1) {  // {m, 1, k} op {m, n, k}
        m = y_dim_after_cmp[0];
        n = y_dim_after_cmp[1];
        k = y_dim_after_cmp[2];
        cmp_type = OptType::MNK_M1K;
      } else if (dim_after_cmp[0] != 1 && dim_after_cmp[1] != 1 &&
                 dim_after_cmp[2] != 1 && y_dim_after_cmp[0] != 1 &&
                 y_dim_after_cmp[1] == 1 &&
                 y_dim_after_cmp[2] != 1) {  // {m, n, k} op {m, 1, k}
        m = dim_after_cmp[0];
        n = dim_after_cmp[1];
        k = dim_after_cmp[2];
        cmp_type = OptType::MNK_M1K;
      } else {
        cmp_type = OptType::CanNotOptimize;
      }
    }
  }

  int get_mnk_for_broadcast_ops(const std::vector<int64_t>& xshape,
                                const std::vector<int64_t>& yshape) {
    int idx = 0;
    int cmp_x = 0;
    int cmp_y = 0;
    bool is_same = false;

    std::vector<int64_t> xshape_after_remove_ones = xshape;
    std::vector<int64_t> yshape_after_remove_ones = yshape;
    // first step: remove excess ones
    std::vector<int64_t>::iterator x_iter = xshape_after_remove_ones.begin();
    std::vector<int64_t>::iterator y_iter = yshape_after_remove_ones.begin();
    for (; x_iter != xshape_after_remove_ones.end();) {
      if (*x_iter == 1 && *y_iter == 1) {
        x_iter = xshape_after_remove_ones.erase(x_iter);
        y_iter = yshape_after_remove_ones.erase(y_iter);
      } else {
        x_iter++;
        y_iter++;
      }
    }
    // second step: compress dims
    int after_cmp_idx = 0;
    for (int i = 0; i < 3; i++) {
      cmp_x = xshape_after_remove_ones[idx];
      cmp_y = yshape_after_remove_ones[idx];
      while ((idx + 1) < xshape_after_remove_ones.size()) {
        is_same = case_is_same(judge_case(xshape_after_remove_ones[idx],
                                          yshape_after_remove_ones[idx]),
                               judge_case(xshape_after_remove_ones[idx + 1],
                                          yshape_after_remove_ones[idx + 1]));
        if (is_same) {
          cmp_x = cmp_x * xshape_after_remove_ones[idx + 1];
          cmp_y = cmp_y * yshape_after_remove_ones[idx + 1];
          idx++;
        } else {
          break;
        }
      }
      idx = idx + 1;
      dim_after_cmp[after_cmp_idx] = cmp_x;
      y_dim_after_cmp[after_cmp_idx] = cmp_y;
      after_cmp_idx++;
      if (idx == xshape_after_remove_ones.size()) {
        dim_size_after_cmp = after_cmp_idx;
        return 0;
      }
    }
    return -1;  // can not compress dims
  }
};
#pragma pack()

template <typename T>
__device__ __forceinline__ void WriteData(T _global_ptr_* dst,
                                          T* src,
                                          int num) {
  if (num > 0) {
    mfence_local();
    LM2GM(src, dst, num * sizeof(T));
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
template <typename Tx, typename Ty, int NX, int NY, bool IsBoundary = false>
__device__ __inline__ void ReadData(Ty* dst,
                                    const Tx _global_ptr_* src,
                                    int size_nx,
                                    int size_ny,
                                    int stride_nx,
                                    int stride_ny) {
  int thread_offset = core_id();
  int left_size_nx = size_nx - thread_offset;
  __local__ Tx in_temp[1];
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
      mfence_local();
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
      mfence_local();
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
        mfence_local();
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
__device__ __inline__ void Init(T* dst, T init_data) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    dst[i] = init_data;
  }
}

template <typename T, int NX>
__device__ __inline__ void Init(T* dst, T init_data, int read_lens) {
#pragma unroll
  for (int i = 0; i < read_lens; i++) {
    dst[i] = init_data;
  }
}

/**
 * The difference from the above function is that
 * it supports different data types of inputs.
 */
template <typename T, typename ArgsT, int Index, int NX>
__device__ __forceinline__ void Init(ArgsT* dst, T init_data, int read_lens) {
  mfence();
#pragma unroll
  for (int i = 0; i < read_lens; i++) {
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
template <typename T, int NX, int NY, bool IsBoundary>
__device__ __inline__ void ReadData(T* dst,
                                    const T _global_ptr_* src,
                                    int num) {
  mfence_local();
  int thread_offset = core_id() * NX;
  if (IsBoundary) {  // core_num() * NX > num
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        GM2LM(src + thread_offset + idx, dst + idx, sizeof(T));
      }
    }
  } else {  // core_num() * NX < num
    GM2LM(src + thread_offset, dst, NX * sizeof(T));
  }
}

template <typename T, int NX, int NY, bool IsBoundary>
__device__ __inline__ void ReadData(T* dst,
                                    const T _global_ptr_* src,
                                    int num,
                                    int read_lens) {
  int thread_offset = core_id() * read_lens;
  mfence_local();
  if (IsBoundary) {  // core_num() * read_lens > num
#pragma unroll
    for (int idx = 0; idx < read_lens; ++idx) {
      if (idx + thread_offset < num) {
        GM2LM(src + thread_offset + idx, dst + idx, sizeof(T));
      }
    }
  } else {  // core_num() * read_lens < num
    GM2LM(src + thread_offset, dst, read_lens * sizeof(T));
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
 * ArgsT: The Type if dst, ArgsT can be std::tuple<T> or std::tuple<Args>
 * Index: The index of data stored in dst.
 * core_id() is used as the index.
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
          bool IsBoundary>
__device__ __forceinline__ void ReadData(ArgsT* dst,
                                         const T _global_ptr_* src,
                                         int num,
                                         int read_lens) {
  int thread_offset = core_id() * read_lens;
  __local__ T in_temp[1];
  __local__ T in_vec[NX];
  if (IsBoundary) {  // core_num() * read_lens > num
#pragma unroll
    for (int idx = 0; idx < read_lens; ++idx) {
      if (idx + thread_offset < num) {
        GM2LM(src + thread_offset + idx, in_temp, sizeof(T));
        std::get<Index>(dst[idx]) = in_temp[0];
        mfence();
      }
    }
  } else {  // core_num() * read_lens < num
    GM2LM(src + thread_offset, in_vec, read_lens * sizeof(T));
#pragma unroll
    for (int idx = 0; idx < read_lens; ++idx) {
      std::get<Index>(dst[idx]) = in_vec[idx];
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
 * core_id() is used as the index.
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
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __inline__ void ReadDataBc(T* dst,
                                      const T _global_ptr_* src,
                                      uint32_t block_offset,
                                      const details::BroadcastConfig& config,
                                      int total_num_output,
                                      int stride_nx,
                                      int stride_ny) {
  uint32_t thread_offset = block_offset + core_id();
  uint32_t index_src = 0;
  mfence_local();
#pragma unroll
  for (int ny = 0; ny < NY; ++ny) {
#pragma unroll
    for (uint32_t nx = 0; nx < NX; ++nx) {
      uint32_t index_output = thread_offset + ny * stride_ny + nx * stride_nx;
      index_src = 0;
      if (IsBoundary) {
        if (index_output >= (uint32_t)total_num_output) {
          break;
        }
      }
      index_src = config(index_output);
      GM2LM(src + index_src, dst + nx + ny * NX, sizeof(T));
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
template <typename Tx,
          typename Ty,
          int NX,
          int NY,
          int Rank,
          typename IndexCal,
          typename Functor,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadDataReduce(
    Ty* dst,
    const Tx _global_ptr_* __restrict__ src,
    int block_offset,
    const IndexCal& index_cal,
    int size_nx,
    int size_ny,
    int stride_nx,
    int stride_ny,
    Functor func,
    bool reduce_last_dim) {
  __local__ Tx in_temp[1];
  int thread_offset = 0;
  int left_idx = 0;
  if (reduce_last_dim) {
    thread_offset = core_id();
    left_idx = 0;
  } else {
    thread_offset = 0;
    left_idx = 0;
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
      mfence_local();
      GM2LM(src + index_src, in_temp, sizeof(Tx));
      dst[ny] = static_cast<Ty>(func(in_temp[0]));

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
        mfence_local();
        GM2LM(src + index_src, in_temp, sizeof(Tx));
        dst[nx + ny * NX] = static_cast<Ty>(func(in_temp[0]));
        thread_offset += stride_ny;
      }
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
 * NX: The number of data continuously written by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
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

template <typename T, int NX, int NY, bool IsBoundary>
__device__ void WriteData(T _global_ptr_* dst,
                          const T* src,
                          int num,
                          int read_lens) {
  int thread_offset = core_id() * read_lens;
  mfence_local();

  if (IsBoundary) {  // core_num() * read_lens > num
#pragma unroll
    for (int idx = 0; idx < read_lens; ++idx) {
      if (idx + thread_offset < num) {
        LM2GM(src + idx, dst + idx + thread_offset, sizeof(T));
      }
    }
  } else {  // core_num() * read_lens < num
    LM2GM(src, dst + thread_offset, read_lens * sizeof(T));
  }
}

template <typename T, int NX, int NY, bool IsBoundary>
__device__ void WriteData(T _global_ptr_* dst, const T* src, int num) {
  int thread_offset = core_id() * NX;
  mfence_local();

  if (IsBoundary) {  // core_num() * NX > num
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        LM2GM(src + idx, dst + idx + thread_offset, sizeof(T));
      }
    }
  } else {  // core_num() * NX < num
    mfence_local();
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
template <typename Tx, typename Ty, int NX, int NY, bool IsBoundary = false>
__device__ __inline__ void WriteData(Ty _global_ptr_* dst,
                                     const Tx* src,
                                     int size_nx,
                                     int size_ny,
                                     int stride_nx,
                                     int stride_ny) {
  int thread_offset = core_id();
  int left_size_nx = size_nx - thread_offset;
  __local__ Ty in_temp[1];

  // Each branch is added for better performance
  if (NX == 1 && NY == 1) {
    if (IsBoundary) {
      if (left_size_nx > 0) {
        in_temp[0] = static_cast<Ty>(src[0]);
        mfence_local();
        LM2GM(in_temp, dst + thread_offset, sizeof(Ty));
      }
    } else {
      in_temp[0] = static_cast<Ty>(src[0]);
      mfence_local();
      LM2GM(in_temp, dst + thread_offset, sizeof(Ty));
    }
  } else if (NX == 1) {
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy * stride_ny >= size_ny) {
          break;
        }
      }

      in_temp[0] = static_cast<Ty>(src[idy]);
      mfence_local();
      LM2GM(in_temp, dst + thread_offset + idy * stride_ny, sizeof(Ty));
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
      mfence_local();
      LM2GM(in_temp, dst + thread_offset + idx * stride_nx, sizeof(Ty));
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
        mfence_local();
        LM2GM(in_temp,
              dst + thread_offset + idx * stride_nx + idy * stride_ny,
              sizeof(Ty));
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
__device__ __inline__ void Init(T* dst, T* init_data, int num) {
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
 * @brief Read data from global memory to local memory with broadcast
 * {m, 1, k}-> {m, n, k} form.
 *
 * @template paraments
 * T: Data type of register.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * src: The original input data pointer of kernel.
 * thread_offset: The data offset of this thread.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * read_lens: The number of data continuously loaded by each thread.
 */
template <typename T>
__device__ __inline__ void ReadDataBcM1kMnk(
    T* dst,
    const T _global_ptr_* src,
    int thread_offset,
    const details::BroadcastConfig& config,
    int read_lens) {
  int index_output = thread_offset;
  int index_base = config(index_output);
  int m = config.m;
  int n = config.n;

  int m_pos = index_base % m;
  if ((m - m_pos) < read_lens) {
    int last_col = m - m_pos;
    GM2LM(src + index_base, dst, last_col * sizeof(T));
    int n_pos = index_output % (m * n) / m;
    int next_part_index = 0;
    if (n_pos != config.n - 1) {
      next_part_index = index_base / m * m;
    } else {
      next_part_index = (index_base / m + 1) * m;
    }
    GM2LM(src + next_part_index,
          dst + last_col,
          (read_lens - last_col) * sizeof(T));
  } else {
    GM2LM(src + index_base, dst, read_lens * sizeof(T));
  }
}

/**
 * @brief Read data from global memory to local memory with broadcast
 * {m, 1}-> {m, n} form.
 *
 * @template paraments
 * T: Data type of register.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * src: The original input data pointer of kernel.
 * thread_offset: The data offset of this thread.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * read_lens: The number of data continuously loaded by each thread.
 */
template <typename T>
__device__ __inline__ void ReadDataBcM1Mn(
    T* dst,
    const T _global_ptr_* src,
    int thread_offset,
    const details::BroadcastConfig& config,
    int read_lens) {
  int index_output = thread_offset;
  int index_base = config(index_output);
  int m = config.m;
  int n = config.n;

  int m_pos = index_base % m;
  if ((m - m_pos) < read_lens) {
    int last_col = m - m_pos;
    GM2LM(src + index_base, dst, last_col * sizeof(T));
    GM2LM(src, dst + last_col, (read_lens - last_col) * sizeof(T));
  } else {
    GM2LM(src + index_base, dst, read_lens * sizeof(T));
  }
}

/**
 * @brief Read data from global memory to local memory with broadcast
 * {1, n}-> {m, n} form.
 *
 * @template paraments
 * T: Data type of register.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * src: The original input data pointer of kernel.
 * thread_offset: The data offset of this thread.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * read_lens: The number of data continuously loaded by each thread.
 */
template <typename T>
__device__ __inline__ void ReadDataBc1NMn(
    T* dst,
    const T _global_ptr_* src,
    int thread_offset,
    const details::BroadcastConfig& config,
    int read_lens) {
  int index_output = thread_offset;
  int index_base = config(index_output);
  int m = config.m;
  int n = config.n;
  T in_temp;

  int m_pos = index_output % m;
  if ((m - m_pos) < read_lens) {
    int last_col = m - m_pos;
    GM2LM(src + index_base, &in_temp, sizeof(T));
    for (int i = 0; i < last_col; i++) {
      dst[i] = in_temp;
    }
    mfence_local();
    GM2LM(src + index_base + 1, &in_temp, sizeof(T));
    for (int i = 0; i < read_lens - last_col; i++) {
      dst[last_col + i] = in_temp;
    }
  } else {
    GM2LM(src + index_base, &in_temp, sizeof(T));
    for (int i = 0; i < read_lens; i++) {
      dst[i] = in_temp;
    }
  }
}

/**
 * @brief Read data from global memory to local memory with broadcast
 * {1, n, 1}-> {m, n, k} form.
 *
 * @template paraments
 * T: Data type of register.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * src: The original input data pointer of kernel.
 * thread_offset: The data offset of this thread.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * read_lens: The number of data continuously loaded by each thread.
 */
template <typename T>
__device__ __inline__ void ReadDataBc1N1Mnk(
    T* dst,
    const T _global_ptr_* src,
    int thread_offset,
    const details::BroadcastConfig& config,
    int read_lens) {
  int index_output = thread_offset;
  int index_base = config(index_output);
  int m = config.m;
  int n = config.n;
  T in_temp;

  int m_pos = index_output % m;
  if ((m - m_pos) < read_lens) {
    int last_col = m - m_pos;
    GM2LM(src + index_base, &in_temp, sizeof(T));
    for (int i = 0; i < last_col; i++) {
      dst[i] = in_temp;
    }
    int n_pos = index_output % (m * n) / m;
    int next_part_index = 0;
    if (n_pos != n - 1) {
      next_part_index = n_pos + 1;
    } else {
      next_part_index = 0;
    }
    mfence_local();
    GM2LM(src + next_part_index, &in_temp, sizeof(T));
    for (int i = 0; i < read_lens - last_col; i++) {
      dst[last_col + i] = in_temp;
    }
  } else {
    GM2LM(src + index_base, &in_temp, sizeof(T));
    for (int i = 0; i < read_lens; i++) {
      dst[i] = in_temp;
    }
  }
}

/**
 * @brief Read data from global memory to local memory with broadcast
 * {1}-> {n} form.
 *
 * @template paraments
 * T: Data type of register.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * src: The original input data pointer of kernel.
 * thread_offset: The data offset of this thread.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * read_lens: The number of data continuously loaded by each thread.
 */
template <typename T>
__device__ __inline__ void ReadDataBc1N(T* dst,
                                        const T _global_ptr_* src,
                                        int thread_offset,
                                        const details::BroadcastConfig& config,
                                        int read_lens) {
  int index_output = thread_offset;
  int index_base = config(index_output);
  T in_temp;

  GM2LM(src + index_base, &in_temp, sizeof(T));
  for (int i = 0; i < read_lens; i++) {
    dst[i] = in_temp;
  }
}

/**
 * @brief Read data from global memory to local memory with broadcast
 * form which can not compress.
 *
 * @template paraments
 * T: Data type of register.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * src: The original input data pointer of kernel.
 * thread_offset: The data offset of this thread.
 * config: Calculation configuration of broadcast. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * total_num_output: Total number of original output.
 * read_lens: The number of data continuously loaded by each thread.
 */
template <typename T, bool IsBoundary = false>
__device__ __inline__ void ReadDataBcCanNotCmp(
    T* dst,
    const T _global_ptr_* src,
    int thread_offset,
    const details::BroadcastConfig& config,
    int total_num_output,
    int read_lens) {
  int index_output = thread_offset;
  int index_base = config(index_output);
  T in_temp;
  int cache_size = 256;
  __local__ T src_temp[cache_size];
  GM2LM(src + index_base, src_temp, cache_size * sizeof(T));

  for (int nx = 0; nx < read_lens; ++nx) {
    index_output = thread_offset + nx;
    if (IsBoundary) {
      if (index_output >= total_num_output) {
        break;
      }
    }
    int index_src = config(index_output);
    if (index_src >= index_base && index_src < index_base + cache_size) {
      in_temp = src_temp[index_src - index_base];
    } else {
      mfence_local();
      GM2LM(src + index_src, &in_temp, sizeof(T));
    }
    dst[nx] = in_temp;
  }
}

/**
 * @brief Read 1D data from global memory to register with broadcast form.
 *
 * @template paraments
 * T: The type of data stored in the global memory.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * core_id() is used as the index.
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
 * read_lens: The number of data continuously loaded by each thread.
 * total_num_output: Total number of original output.
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __inline__ void ReadDataBc(T* dst,
                                      const T _global_ptr_* src,
                                      uint32_t block_offset,
                                      const details::BroadcastConfig& config,
                                      int total_num_output,
                                      int read_lens) {
  int thread_offset = block_offset + core_id() * read_lens;

  if (config.cmp_type == details::OptType::MNK_M1K) {
    ReadDataBcM1kMnk<T>(dst, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::N_1) {
    ReadDataBc1N<T>(dst, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::MN_M) {
    ReadDataBcM1Mn<T>(dst, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::MN_N) {
    ReadDataBc1NMn<T>(dst, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::MNK_1N1) {
    ReadDataBc1N1Mnk<T>(dst, src, thread_offset, config, read_lens);
  } else {
    ReadDataBcCanNotCmp<T, IsBoundary>(
        dst, src, thread_offset, config, total_num_output, read_lens);
  }
}

/**
 * @brief Read 1D data from global memory to register with broadcast form.
 * The difference from the above function is that it supports different data
 * types of inputs.
 * @template paraments
 * T: The type of data stored in the global memory.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * core_id() is used as the index.
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
 * read_lens: The number of data continuously loaded by each thread.
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
    const T _global_ptr_* src,
    int block_offset,
    const details::BroadcastConfig& config,
    int total_num_output,
    int read_lens = NX) {
  int thread_offset = block_offset + core_id() * read_lens;
  __local__ T in_temp[NX];

  if (config.cmp_type == details::OptType::MNK_M1K) {
    ReadDataBcM1kMnk<T>(in_temp, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::N_1) {
    ReadDataBc1N<T>(in_temp, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::MN_M) {
    ReadDataBcM1Mn<T>(in_temp, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::MN_N) {
    ReadDataBc1NMn<T>(in_temp, src, thread_offset, config, read_lens);
  } else if (config.cmp_type == details::OptType::MNK_1N1) {
    ReadDataBc1N1Mnk<T>(in_temp, src, thread_offset, config, read_lens);
  } else {
    ReadDataBcCanNotCmp<T, IsBoundary>(
        in_temp, src, thread_offset, config, total_num_output, read_lens);
  }
#pragma unroll
  for (int idx = 0; idx < read_lens; ++idx) {
    std::get<Index>(dst[idx]) = in_temp[idx];
  }
}

/**
 * @brief Initialize register with data index.
 *
 * @template paraments
 * T: Data type of register.
 * NX: Number of data to initialize.
 * NY: Number of data to initialize, NY only can be 1.
 * core_id() is used as the index.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * init_data: The register pointer of init data, the size is NX.
 */
template <typename T, int NX, int NY>
__device__ __forceinline__ void InitWithDataIndex(T* dst, int block_offset) {
  int thread_offset = block_offset + core_id() * NX;
#pragma unroll
  for (int nx = 0; nx < NX; ++nx) {
    dst[nx] = static_cast<T>(thread_offset + nx);
  }
}

}  // namespace kps
}  // namespace phi
