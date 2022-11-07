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

#include "paddle/fluid/framework/gpu_utils.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using Dim3 = framework::Dim3;
using Index3 = framework::Index3;

struct EqualTo {
  constexpr bool operator()(int a, int b) const { return a == b; }
};

struct GreaterThan {
  constexpr bool operator()(int a, int b) const { return a > b; }
};

// Value can be decided in compile time.
template <typename FUN, int INT_32 = 32>
constexpr bool CheckProperTileSize(int tile_long,
                                   int tile_short,
                                   int size_T,
                                   FUN op) {
  return (size_T == 16 && ((tile_long == INT_32 && op(tile_short, 4)) ||
                           (tile_long == 2 * INT_32 && op(tile_short, 4)) ||
                           (tile_long == 4 * INT_32 && op(tile_short, 4)) ||
                           (tile_long == 8 * INT_32 && op(tile_short, 2)))) ||
         (size_T == 8 && ((tile_long == INT_32 && op(tile_short, 15)) ||
                          (tile_long == 2 * INT_32 && op(tile_short, 15)) ||
                          (tile_long == 4 * INT_32 && op(tile_short, 8)) ||
                          (tile_long == 8 * INT_32 && op(tile_short, 4)) ||
                          (tile_long == 16 * INT_32 && op(tile_short, 2)))) ||
         ((size_T == 4 || size_T == 2 || size_T == 1) &&
          ((tile_long == INT_32 && op(tile_short, 15)) ||
           (tile_long == 2 * INT_32 && op(tile_short, 15)) ||
           (tile_long == 4 * INT_32 && op(tile_short, 8)) ||
           (tile_long == 8 * INT_32 && op(tile_short, 4)) ||
           (tile_long == 16 * INT_32 && op(tile_short, 2)) ||
           (tile_long == 16 * INT_32 && op(tile_short, 2))));
}

constexpr bool CheckLongTileSize(int tile_long, int tile_short, int size_T) {
  return CheckProperTileSize(tile_long, tile_short, size_T, EqualTo());
}

constexpr bool CheckOutsideTileSize(int tile_long, int tile_short, int size_T) {
  return CheckProperTileSize(tile_long, tile_short, size_T, GreaterThan());
}

constexpr bool CheckNonLongTileSize(int tile_long, int tile_short, int size_T) {
  return !CheckOutsideTileSize(tile_long, tile_short, size_T) &&
         (CheckOutsideTileSize(tile_long * 2, tile_short, size_T) ||
          CheckOutsideTileSize(tile_long, tile_short + 1, size_T)) &&
         !CheckLongTileSize(tile_long, tile_short, size_T);
}

// Use SM to do data transfer, load a tile into SM then store out.
// All tile read and write are colascing, so can speedup memory copy
template <typename T,
          int NumThreads,
          int TileX,
          int TileY,
          typename IndexType = int>
__global__ void TilingSwapDim1And2(const T* __restrict__ input,
                                   Dim3 input_dims,
                                   T* __restrict__ output) {
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  constexpr int BlockReadRows = NumThreads / TileY;
  constexpr int BlockWriteRows = NumThreads / TileX;

  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ __align__(
      alignof(T)) char share_mem_ptr[TileX * (TileY + 1) * sizeof(T)];
  typedef T(*ShareMemory)[TileY + 1];

  ShareMemory tile_sm = reinterpret_cast<ShareMemory>(share_mem_ptr);

  int x = threadIdx.x;

  Dim3 output_dims = {
      input_dims[0],
      input_dims[2],
      input_dims[1],
  };

  // Align dim to Tiles
  Dim3 tile_aligned_input_dim = {
      input_dims[0],
      (input_dims[1] + TileX - 1) / TileX,
      (input_dims[2] + TileY - 1) / TileY,
  };

  // Converts block idx to tile index, each block process a tile
  Index3 input_block_tile_index = framework::ConvertTensorIndex<IndexType>(
      blockIdx.x, tile_aligned_input_dim);

  // Compute real index align to tile:0, 32, 64...
  Index3 block_tile_index_in_input = {
      input_block_tile_index[0],
      input_block_tile_index[1] * TileX,
      input_block_tile_index[2] * TileY,
  };

  // Compute block flat index against input dims.
  IndexType input_origin_block_flat_index =
      framework::FlatTensorIndex<IndexType>(block_tile_index_in_input,
                                            input_dims);

  bool full_tile = true;
  IndexType tile_width = TileY;

  // Last row is not full.
  if (input_block_tile_index[2] == tile_aligned_input_dim[2] - 1) {
    tile_width = input_dims[2] - (tile_aligned_input_dim[2] - 1) * TileY;
    full_tile &= false;
  }

  IndexType tile_height = TileX;

  if (input_block_tile_index[1] == tile_aligned_input_dim[1] - 1) {
    tile_height = input_dims[1] - (tile_aligned_input_dim[1] - 1) * TileX;
    full_tile &= false;
  }

  constexpr IndexType in_effective_thread_num = NumThreads / TileY * TileY;

  if (x < in_effective_thread_num) {
    // Read a tile from input using block.
    int x_i = x / TileY;
    int x_j = x % TileY;
    IndexType input_ind =
        input_origin_block_flat_index + x_i * input_dims[2] + x_j;
    IndexType input_inc = BlockReadRows * input_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileX); ind_i += BlockReadRows) {
        tile_sm[ind_i][x_j] = input[input_ind];
        input_ind += input_inc;
      }
    } else {
      if (x_j < tile_width) {
#pragma unroll
        for (IndexType ind_i = x_i; ind_i < (tile_height);
             ind_i += BlockReadRows) {
          tile_sm[ind_i][x_j] = input[input_ind];
          input_ind += input_inc;
        }
      }
    }
  }

  __syncthreads();

  // Store sm value back to out
  Index3 block_tile_index_in_output = {
      input_block_tile_index[0],
      input_block_tile_index[2] * TileY,
      input_block_tile_index[1] * TileX,
  };

  IndexType output_origin_block_flat_index =
      framework::FlatTensorIndex<IndexType>(block_tile_index_in_output,
                                            output_dims);

  constexpr IndexType out_effective_thread_num = NumThreads / TileX * TileX;

  if (x < out_effective_thread_num) {
    int x_i = x / TileX;
    int x_j = x % TileX;
    IndexType output_ind =
        output_origin_block_flat_index + x_i * output_dims[2] + x_j;
    IndexType output_inc = BlockWriteRows * output_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileY); ind_i += BlockWriteRows) {
        output[output_ind] = tile_sm[x_j][ind_i];
        output_ind += output_inc;
      }
    } else {
      if (x_j < tile_height) {
#pragma unroll
        for (IndexType ind_i = x_i; ind_i < (tile_width);
             ind_i += BlockWriteRows) {
          output[output_ind] = tile_sm[x_j][ind_i];
          output_ind += output_inc;
        }
      }
    }
  }
}

// This function will find combination of long_side X short_side in backups
template <int TSIZE>
bool SelectProperTileSize(std::vector<std::pair<int, int>>* tiles) {
  PADDLE_ENFORCE_LE(
      TSIZE,
      16,
      platform::errors::InvalidArgument(
          "The tile size should smaller than 16, but received is:%d.", TSIZE));

  PADDLE_ENFORCE_EQ(
      (TSIZE & (TSIZE - 1)),
      0,
      platform::errors::InvalidArgument(
          "Data types should be powers of 2, but reived size is:%d.", TSIZE));

  constexpr int kMaxLongSideLen = 1024;
  constexpr int kMaxShortSideLen = 15;

  for (int long_side = 32; long_side <= kMaxLongSideLen; long_side *= 2) {
    for (int short_side = 2; short_side <= kMaxShortSideLen; short_side += 1) {
      if (CheckLongTileSize(long_side, short_side, TSIZE)) {
        tiles->push_back(std::make_pair(long_side, short_side));

        if (short_side == 2) return true;

        break;
      }
    }
  }
  return false;
}

// Use system built in type
template <int ByteSize>
struct SystemElemType;
template <>
struct SystemElemType<1> {
  using type = uint8_t;
};
template <>
struct SystemElemType<2> {
  using type = uint16_t;
};
template <>
struct SystemElemType<4> {
  using type = uint32_t;
};
template <>
struct SystemElemType<8> {
  using type = uint64_t;
};
template <>
struct SystemElemType<16> {
  using type = float4;
};

template <typename T, int tile_long, int tile_short, typename IndexType = int>
void LaunchNarrowDims2TransposeKernel(const phi::GPUContext& d,
                                      int tile_size_i,
                                      int tile_size_j,
                                      IndexType total_tiles_count,
                                      const T* input,
                                      const Dim3& input_dims,
                                      T* output) {
  constexpr int NumThreads = tile_long;
  if (tile_size_i <= tile_long && tile_size_j <= tile_short) {
    TilingSwapDim1And2<T, NumThreads, tile_long, tile_short, IndexType>
        <<<total_tiles_count, NumThreads, 0, d.stream()>>>(
            input, input_dims, output);
  } else {
    TilingSwapDim1And2<T, NumThreads, tile_short, tile_long, IndexType>
        <<<total_tiles_count, NumThreads, 0, d.stream()>>>(
            input, input_dims, output);
  }
}

template <typename T,
          int tile_long,
          int tile_short,
          typename IndexType = int,
          typename dummy = void>
struct NarrowDims2TransposeDispatch {
  static void DoTranspose(const phi::GPUContext& d,
                          int tile_size_i,
                          int tile_size_j,
                          IndexType total_tiles_count,
                          const T* input,
                          const Dim3& input_dims,
                          T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)),
        0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2."
            " But received value is:%d.",
            tile_long));

    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short, IndexType>(
          d,
          tile_size_i,
          tile_size_j,
          total_tiles_count,
          input,
          input_dims,
          output);
      return;
    }

    const bool long_side_request_not_satisfied =
        std::max(tile_size_i, tile_size_j) > tile_long;

    if (long_side_request_not_satisfied) {
      NarrowDims2TransposeDispatch<T, tile_long * 2, tile_short, IndexType>::
          DoTranspose(d,
                      tile_size_i,
                      tile_size_j,
                      total_tiles_count,
                      input,
                      input_dims,
                      output);
    } else {
      NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1, IndexType>::
          DoTranspose(d,
                      tile_size_i,
                      tile_size_j,
                      total_tiles_count,
                      input,
                      input_dims,
                      output);
    }
  }
};

// If Not long tile size, goto this function when compile.
template <typename T, int tile_long, int tile_short, typename IndexType>
struct NarrowDims2TransposeDispatch<
    T,
    tile_long,
    tile_short,
    IndexType,
    typename std::enable_if<CheckNonLongTileSize(
                                tile_long, tile_short, sizeof(T)),
                            void>::type> {
  static void DoTranspose(const phi::GPUContext& d,
                          int tile_size_i,
                          int tile_size_j,
                          IndexType total_tiles_count,
                          const T* input,
                          const Dim3& input_dims,
                          T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)),
        0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2."
            " But received value is:%d.",
            tile_long));

    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short, IndexType>(
          d,
          tile_size_i,
          tile_size_j,
          total_tiles_count,
          input,
          input_dims,
          output);
      return;
    }

    NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1, IndexType>::
        DoTranspose(d,
                    tile_size_i,
                    tile_size_j,
                    total_tiles_count,
                    input,
                    input_dims,
                    output);
  }
};

// If long tile size, goto this function when compile.
template <typename T, int tile_long, int tile_short, typename IndexType>
struct NarrowDims2TransposeDispatch<
    T,
    tile_long,
    tile_short,
    IndexType,
    typename std::enable_if<CheckLongTileSize(tile_long, tile_short, sizeof(T)),
                            void>::type> {
  static void DoTranspose(const phi::GPUContext& d,
                          int tile_size_i,
                          int tile_size_j,
                          IndexType total_tiles_count,
                          const T* input,
                          const Dim3& input_dims,
                          T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)),
        0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2,"
            " but received is:%d.",
            tile_long));

    LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short, IndexType>(
        d,
        tile_size_i,
        tile_size_j,
        total_tiles_count,
        input,
        input_dims,
        output);
  }
};

template <typename T, bool conjugate = false, typename IndexType = int>
void SwapDim1And2InNarrow(const phi::GPUContext& d,
                          const T* input,
                          const Dim3& input_dims,
                          T* output,
                          const int kMinTileSize) {
  // First get available tile sizes for the data type requested as backups
  std::vector<std::pair<int, int>> tile_sele;
  auto ret = SelectProperTileSize<sizeof(T)>(&tile_sele);
  PADDLE_ENFORCE_EQ(
      ret,
      true,
      platform::errors::InvalidArgument(
          "SelectProperTileSize should return true, but return value is:%d.",
          ret));

  int tile_long_edge = 0;
  int tile_short_edge = 0;
  float lowest_cost = std::numeric_limits<float>::max();
  int input_long_edge = std::max(input_dims[1], input_dims[2]);

  // Find the tile size that best suit in  inputs.
  for (auto tile_size_pair : tile_sele) {
    int proposed_tile_long_edge = tile_size_pair.first;
    // data may not aligned to tile, so some threads wasted, we need
    // to find least wasted threads, which means we need to find tile
    // can split input properly, in another words: num_wasted_threads=0.
    int num_wasted_threads =
        input_long_edge - framework::CeilOrFloor<int, false>(
                              input_long_edge, proposed_tile_long_edge) *
                              proposed_tile_long_edge;

    int num_full_tiles = framework::CeilOrFloor<int, false>(
        input_long_edge, proposed_tile_long_edge);

    float cost = num_wasted_threads;

    if (cost <= lowest_cost) {
      tile_long_edge = proposed_tile_long_edge;
      tile_short_edge = tile_size_pair.second;
      lowest_cost = cost;
    }
    // break as we already find best tile size.
    if (cost == 0) break;
  }

  // The tile size we select should be match with input dim, long side to long
  // short side to short.
  // First set long side  as i if dim1 > Tile min size, then set dim2 as j.
  bool is_first_dim_big = input_dims[1] >= kMinTileSize;
  int select_tile_size_i = is_first_dim_big ? tile_long_edge : input_dims[1];
  int select_tile_size_j = is_first_dim_big ? input_dims[2] : tile_long_edge;

  // Check if i is long edge, if not set i as short.
  select_tile_size_i = is_first_dim_big
                           ? tile_long_edge
                           : std::min(select_tile_size_i, tile_short_edge);

  // Check if j is long edge, if not set j as short.
  select_tile_size_j = is_first_dim_big
                           ? std::min(select_tile_size_j, tile_short_edge)
                           : tile_long_edge;

  // Here finally get proper long X short tile size.
  Dim3 input_dims_aligned = {
      input_dims[0],
      framework::CeilOrFloor<int, true>(input_dims[1], select_tile_size_i),
      framework::CeilOrFloor<int, true>(input_dims[2], select_tile_size_j),
  };

  IndexType total_tiles_count =
      input_dims_aligned[0] * input_dims_aligned[1] * input_dims_aligned[2];

  // Suppose T can be replaced by system builtin types
  using ElemType = typename SystemElemType<sizeof(T)>::type;

  NarrowDims2TransposeDispatch<ElemType, 32, 2, IndexType>::DoTranspose(
      d,
      select_tile_size_i,
      select_tile_size_j,
      total_tiles_count,
      reinterpret_cast<const ElemType*>(input),
      input_dims,
      reinterpret_cast<ElemType*>(output));
}

// This is for case that cannot do coalescing read and write.
// Or input is too small to split into tiles.
template <typename T, int pos0, int pos1, int pos2, typename IndexType = int>
__global__ void TransposeSimpleKernel(IndexType nthreads,
                                      const T* __restrict__ input,
                                      Dim3 input_dims,
                                      T* __restrict__ output) {
  Dim3 output_dims;
  output_dims[pos0] = input_dims[0];
  output_dims[pos1] = input_dims[1];
  output_dims[pos2] = input_dims[2];

  CUDA_KERNEL_LOOP_TYPE(output_index, nthreads, IndexType) {
    Index3 output_tensor_index =
        framework::ConvertTensorIndex<IndexType>(output_index, output_dims);

    Index3 input_tensor_index;
    input_tensor_index[0] = output_tensor_index[pos0];
    input_tensor_index[1] = output_tensor_index[pos1];
    input_tensor_index[2] = output_tensor_index[pos2];

    IndexType input_index =
        framework::FlatTensorIndex<IndexType>(input_tensor_index, input_dims);

    output[output_index] = input[input_index];
  }
}

// Here suppose convert all tensor to dim3, so just change dim1 and 2.
template <typename T, typename IndexType = int>
void SendSwapDim1And2InTranspose(const phi::GPUContext& d,
                                 const T* input,
                                 const Dim3& input_dims,
                                 T* output) {
  // Suppose tile size > 16
  static const int kMinTileSize = 16;
  static const int kMinNarrowTileSize = 96;

  bool large_tile =
      input_dims[1] >= kMinTileSize && input_dims[2] >= kMinTileSize;
  bool narrow_tile = input_dims[1] >= kMinNarrowTileSize ||
                     input_dims[2] >= kMinNarrowTileSize;
  if (large_tile) {
    // If input is large square, such as 32X32, use SM to do copy.
    // suppose 32 X 32 gives best performance, and 8 warp in block.
    constexpr int kNumThreads = 256;
    Dim3 input_dims_aligned = {
        input_dims[0],
        framework::CeilOrFloor<int, true>(input_dims[1], kTileSize),
        framework::CeilOrFloor<int, true>(input_dims[2], kTileSize),
    };

    IndexType total_tiles_count =
        input_dims_aligned[0] * input_dims_aligned[1] * input_dims_aligned[2];

    TilingSwapDim1And2<T, kNumThreads, kTileSize, kTileSize, IndexType>
        <<<total_tiles_count, kNumThreads, 0, d.stream()>>>(
            input, input_dims, output);

  } else if (narrow_tile) {
    // If input shape is like Rect, such as 2X100, use Narrow tile size.
    // It makes things complicated, because need to find a tile can coverr
    // input and also reach best coalescing.
    SwapDim1And2InNarrow<T, false, IndexType>(
        d, input, input_dims, output, kMinTileSize);
  } else {
    // If input shape is small, such as 8X8, just do simple copy
    IndexType total_elements = input_dims[0];
    total_elements *= input_dims[1];
    total_elements *= input_dims[2];
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(d, total_elements);
    TransposeSimpleKernel<T, 0, 2, 1, IndexType>
        <<<config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
            total_elements, input, input_dims, output);
  }
}

template <typename T, typename IndexType = int>
struct SwapDim1And2InTranspose {
  typedef phi::GPUContext Device;
  void operator()(const Device& d,
                  const T* in,
                  const std::vector<int>& combined_dims,
                  T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};
    SendSwapDim1And2InTranspose<T, IndexType>(d, in, input_dims, out);
  }
};

template <typename T, typename IndexType = int>
struct SwapDim0And2InTranspose {
  typedef phi::GPUContext Device;
  void operator()(const Device& d,
                  const T* in,
                  const std::vector<int>& combined_dims,
                  T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};

    IndexType total_size = combined_dims[0];
    total_size *= combined_dims[1];
    total_size *= combined_dims[2];
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(d, total_size);

    TransposeSimpleKernel<T, 2, 1, 0, IndexType>
        <<<config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
            total_size, in, input_dims, out);
  }
};

template <typename T>
inline void PermuteWithEigen(const phi::GPUContext& ctx,
                             phi::DenseTensor* in,
                             phi::DenseTensor* out,
                             const DimsSimplifier<T>& simplifier) {
  const bool not_same_dims =
      simplifier.GetRank() != static_cast<int>(in->dims().size());
  if (not_same_dims) {
    phi::DDim src_dims = in->dims();
    phi::DDim dst_dims = out->dims();
    in->ResizeAndAllocate(phi::make_ddim(simplifier.GetSrcDims()));
    out->ResizeAndAllocate(phi::make_ddim(simplifier.GetDstDims()));

    TransCompute<phi::GPUContext, T>(
        simplifier.GetRank(), ctx, *in, out, simplifier.GetPerm());
    in->ResizeAndAllocate(src_dims);
    out->ResizeAndAllocate(dst_dims);
  } else {
    TransCompute<phi::GPUContext, T>(
        simplifier.GetRank(), ctx, *in, out, simplifier.GetPerm());
  }
}

template <typename T, typename IndexType = int>
inline void TransposeSimple(const phi::GPUContext& ctx,
                            phi::DenseTensor* in,
                            phi::DenseTensor* out,
                            const DimsSimplifier<T>& simplifier) {
  const std::vector<int> new_perm = simplifier.GetPerm();
  std::vector<int> new_dims(simplifier.GetSrcDims().begin(),
                            simplifier.GetSrcDims().end());
  auto in_data = in->data<T>();
  auto out_data = out->data<T>();

  if (simplifier.GetRank() == 2 && new_perm[0] == 1 && new_perm[1] == 0) {
    // Add the first dimension size as 1.
    new_dims.insert(new_dims.begin(), 1);
    SwapDim1And2InTranspose<T, IndexType>()(ctx, in_data, new_dims, out_data);
  } else if (new_perm == std::vector<int>({0, 2, 1})) {
    SwapDim1And2InTranspose<T, IndexType>()(ctx, in_data, new_dims, out_data);
  } else if (new_perm == std::vector<int>({2, 1, 0})) {
    // May optimized later, find a way to do coalescing memory copy.
    // But it depends on data size. If span is not large, coalescing may work.
    SwapDim0And2InTranspose<T, IndexType>()(ctx, in_data, new_dims, out_data);
  } else {
    PermuteWithEigen<T>(ctx, in, out, simplifier);
  }
}

template <typename T>
inline void TransposeWithSimple(const phi::GPUContext& ctx,
                                phi::DenseTensor* in,
                                phi::DenseTensor* out,
                                const DimsSimplifier<T>& simplifier) {
  if (simplifier.GetCount() < std::numeric_limits<int>::max()) {
    TransposeSimple<T>(ctx, in, out, simplifier);
  } else {
    TransposeSimple<T, int64_t>(ctx, in, out, simplifier);
  }
}

template <int N, typename T>
class IdxHelper {
 public:
  IdxHelper() {}
  explicit IdxHelper(const T* dims) {
    for (int i = N - 1; i >= 0; --i) {
      stride_[i] = i < (N - 1) ? dims[i + 1] * stride_[i + 1] : 1;
    }
  }

  __device__ __forceinline__ T GetStride(int idx) const { return stride_[idx]; }
  __device__ __forceinline__ void GetIndexFromOffset(T offset, T* index) const {
    T remaining = offset;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      remaining -= idx * stride_[i];
      index[i] = idx;
    }
    index[N - 1] = remaining;
  }

 private:
  T stride_[N];
};

template <int N>
class IdxHelper<N, uint32_t> {
 public:
  IdxHelper() {}
  explicit IdxHelper(const uint32_t* dims) {
    for (int i = N - 1; i >= 0; --i) {
      uint32_t value = i < (N - 1) ? dims[i + 1] * stride_[i + 1] : 1;
      divmoder_[i] = paddle::platform::FastDivMod(value);
      stride_[i] = value;
    }
  }

  __device__ __forceinline__ uint32_t GetStride(int idx) const {
    return stride_[idx];
  }
  __device__ __forceinline__ void GetIndexFromOffset(uint32_t offset,
                                                     uint32_t* index) const {
    uint32_t remaining = offset;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      uint32_t idx = divmoder_[i].Div(remaining);
      index[i] = idx;
      remaining -= idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

 private:
  uint32_t stride_[N];
  paddle::platform::FastDivMod divmoder_[N];
};

// Transform index between memory offset and shape coodinate.
template <typename T, int N>
class IdxAndOffsetHelper {
 public:
  IdxAndOffsetHelper() {}
  ~IdxAndOffsetHelper() = default;

  explicit IdxAndOffsetHelper(const T* dims) {
    index_helper = IdxHelper<N, T>(dims);
  }

  __device__ __forceinline__ T IndexToOffset(const T* index) const {
    T offset = 0;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      offset += index[i] * index_helper.GetStride(i);
    }
    offset += index[N - 1];
    return offset;
  }

  __device__ __forceinline__ void OffsetToIndex(T offset, T* index) const {
    index_helper.GetIndexFromOffset(offset, index);
  }

 private:
  IdxHelper<N, T> index_helper;
};

template <int Rank, typename IndexT>
struct PermuteParams {
 public:
  IdxAndOffsetHelper<IndexT, Rank> src_index_helper;
  IdxAndOffsetHelper<IndexT, Rank> dst_index_helper;
  int perm[Rank]{};

  explicit PermuteParams(const std::vector<int64_t>& dims,
                         const std::vector<int>& perm_) {
    IndexT dst_dims[Rank];
    IndexT src_dims[Rank];
    for (auto i = 0; i < Rank; ++i) {
      src_dims[i] = dims[i];
      dst_dims[i] = dims[perm_[i]];
      perm[i] = perm_[i];
    }
    dst_index_helper = IdxAndOffsetHelper<IndexT, Rank>(dst_dims);
    src_index_helper = IdxAndOffsetHelper<IndexT, Rank>(src_dims);
  }
};

// A special kernel for target case, both vectorized read and write supported.
template <typename T, typename IndexT, int VecSize, int Rank>
__global__ void VectorizedPermuteKernel(PermuteParams<Rank, IndexT> params,
                                        const IndexT count,
                                        const T* __restrict__ src_data,
                                        T* dst_data) {
  using VecT = phi::AlignedVector<T, VecSize>;
  IndexT src_index[Rank];
  IndexT dst_index[Rank];

  const VecT* __restrict__ vec_src =
      reinterpret_cast<const VecT* __restrict__>(src_data);
  VecT* vec_dst = reinterpret_cast<VecT*>(dst_data);

  IndexT tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (IndexT i = tid; i < count; i += blockDim.x * gridDim.x) {
    params.dst_index_helper.OffsetToIndex(i, dst_index);

#pragma unroll
    for (int j = 0; j < Rank; ++j) {
      src_index[params.perm[j]] = dst_index[j];
    }
    IndexT src_offset = params.src_index_helper.IndexToOffset(src_index);
    vec_dst[i] = vec_src[src_offset];
  }
}

// A general kernel for normal case, only support vectorized write.
template <typename T, typename IndexT, int VecSize, int Rank>
__global__ void GeneralPermuteKernel(PermuteParams<Rank, IndexT> params,
                                     const IndexT main_cnt,
                                     const IndexT tail_cnt,
                                     const IndexT offset,
                                     const T* __restrict__ src,
                                     T* dst) {
  using VecT = phi::AlignedVector<T, VecSize>;
  VecT* vec_dst = reinterpret_cast<VecT*>(dst);

  IndexT src_index[VecSize][Rank];
  IndexT dst_index[VecSize][Rank];

  // Vectorized load data.
  IndexT tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (IndexT idx = tid; idx < main_cnt; idx += blockDim.x * gridDim.x) {
    VecT vec_data;
    IndexT vec_idx = idx * VecSize;

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      params.dst_index_helper.OffsetToIndex(vec_idx + i, dst_index[i]);

#pragma unroll
      for (int j = 0; j < Rank; ++j) {
        src_index[i][params.perm[j]] = dst_index[i][j];
      }
      IndexT src_offset = params.src_index_helper.IndexToOffset(src_index[i]);
      vec_data[i] = src[src_offset];
    }
    vec_dst[idx] = vec_data;
  }

  // Singularized load data.
  if (tid < tail_cnt) {
    IndexT idx = tid + offset;
    params.dst_index_helper.OffsetToIndex(idx, dst_index[0]);

#pragma unroll
    for (int j = 0; j < Rank; ++j) {
      src_index[0][params.perm[j]] = dst_index[0][j];
    }
    IndexT src_offset = params.src_index_helper.IndexToOffset(src_index[0]);
    dst[idx] = src[src_offset];
  }
}

template <typename T, typename IndexT, int ReadSize, int WriteSize>
struct TransposeDataWriter {
  __device__ __forceinline__ void operator()(T* dst_data,
                                             const T* s_data,
                                             const IndexT rows,
                                             const IndexT cols,
                                             const IndexT chs_stride,
                                             const IndexT round_tile_cols,
                                             const IndexT col_stride = 1) {
    using OutVecT = phi::AlignedVector<T, WriteSize>;
    OutVecT* vec_dst = reinterpret_cast<OutVecT*>(dst_data);

    constexpr int kColTile = kTileSize * ReadSize;
    constexpr int kColStride = kShareCol * ReadSize;

    const IndexT vec_rows = rows / WriteSize;
    const IndexT col_in_mat = blockIdx.y * kTileSize + threadIdx.x;

    if (col_in_mat < /*dst_cols=*/vec_rows) {
      const int cols_range = (blockIdx.x < round_tile_cols)
                                 ? kTileSize
                                 : (cols - round_tile_cols * kTileSize);
      const int share_tile = threadIdx.x * (WriteSize * kColStride);
      const IndexT write_offset = blockIdx.z * chs_stride + col_in_mat;
#pragma unroll
      for (int tile_y = threadIdx.y; tile_y < cols_range;
           tile_y += kBlockRows) {
        OutVecT tmp_data[ReadSize];
#pragma unroll
        for (int i = 0; i < ReadSize; ++i) {
          const int tile_tail = tile_y * ReadSize + i;
          const int major_share_idx = share_tile + tile_tail;
          const IndexT row_in_mat =
              (blockIdx.x * kColTile + tile_tail) * col_stride;
#pragma unroll
          for (int j = 0; j < WriteSize; ++j) {
            tmp_data[i].val[j] = s_data[j * kColStride + major_share_idx];
          }
          vec_dst[write_offset + row_in_mat * vec_rows] = tmp_data[i];
        }
      }
    }
  }
};

template <typename T, typename IndexT, int ReadSize>
struct TransposeDataWriter<T, IndexT, ReadSize, 1> {
  __device__ __forceinline__ void operator()(T* dst_data,
                                             const T* s_data,
                                             const IndexT rows,
                                             const IndexT cols,
                                             const IndexT chs_stride,
                                             const IndexT round_tile_cols,
                                             const IndexT col_stride = 1) {
    const IndexT col_in_mat = blockIdx.y * kTileSize + threadIdx.x;
    if (col_in_mat < /*dst_cols=*/rows) {
      const int cols_range = (blockIdx.x < round_tile_cols)
                                 ? kTileSize
                                 : (cols - round_tile_cols * kTileSize);
      const IndexT row_tile = blockIdx.x * kTileSize * ReadSize;
      const IndexT write_offset = blockIdx.z * chs_stride + col_in_mat;
      const int shared_tile = threadIdx.x * kShareCol * ReadSize;
#pragma unroll
      for (int tile_y = threadIdx.y; tile_y < cols_range;
           tile_y += kBlockRows) {
        const int shared_major = shared_tile + tile_y * ReadSize;
        const IndexT row_major = (row_tile + tile_y * ReadSize) * col_stride;
#pragma unroll
        for (int i = 0; i < ReadSize; ++i) {
          const IndexT row_in_mat = row_major + i * col_stride;
          dst_data[write_offset + row_in_mat * rows] = s_data[shared_major + i];
        }
      }
    }
  }
};

template <typename T, typename IndexT, int VecSize, IndexT kRowTile>
struct TransposeDataReader {
  __device__ __forceinline__ void operator()(const T* __restrict__ src,
                                             T* s_shared,
                                             const IndexT cols,
                                             const IndexT rows,
                                             const IndexT chs_stride,
                                             const IndexT cols_thresh,
                                             const IndexT round_tile_rows) {
    using VecT = phi::AlignedVector<T, VecSize>;
    const VecT* __restrict__ v_src =
        reinterpret_cast<const VecT* __restrict__>(src);
    VecT* v_shared = reinterpret_cast<VecT*>(s_shared);

    const IndexT col_in_mat = blockIdx.x * kTileSize + threadIdx.x;
    if (col_in_mat < cols_thresh) {
      const int row_range = (blockIdx.y < round_tile_rows)
                                ? kRowTile
                                : (rows - kRowTile * round_tile_rows);
      const IndexT src_idx_major = blockIdx.z * chs_stride + col_in_mat;
#pragma unroll
      for (int tile_y = threadIdx.y; tile_y < row_range; tile_y += kBlockRows) {
        const IndexT row_in_mat = blockIdx.y * kRowTile + tile_y;
        v_shared[tile_y * kShareCol + threadIdx.x] =
            v_src[row_in_mat * cols + src_idx_major];
      }
    }
    __syncthreads();
  }
};

// Aim at transposing the last 2 dimensions. Reference from
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template <typename T,
          typename IndexT,
          bool IsVecWrite,
          int ReadSize,
          int WriteSize = (IsVecWrite && (sizeof(T) < sizeof(float)))
                              ? sizeof(float) / sizeof(T)
                              : 1>
__global__ void SwapTransposeKernel(const T* __restrict__ src_data,
                                    T* dst_data,
                                    const IndexT round_tile_rows,
                                    const IndexT round_tile_cols,
                                    const IndexT cols,
                                    const IndexT rows,
                                    const IndexT chs /*=channel*/) {
  constexpr int kRowTile = kTileSize * WriteSize;
  __shared__ T s_data[kRowTile * kShareCol * ReadSize];

  const IndexT chs_stride = chs * cols;
  TransposeDataReader<T, IndexT, ReadSize, kRowTile>()(
      src_data, s_data, chs_stride, rows, cols, cols, round_tile_rows);
  TransposeDataWriter<T, IndexT, ReadSize, WriteSize>()(
      dst_data, s_data, rows, cols, rows / WriteSize, round_tile_cols, chs);
}

template <typename T,
          typename IndexT,
          bool IsVecWrite,
          int ReadSize,
          int WriteSize = (IsVecWrite && (sizeof(T) < sizeof(float)))
                              ? sizeof(float) / sizeof(T)
                              : 1>
__global__ void BatchTransposeKernel(const T* __restrict__ src_data,
                                     T* dst_data,
                                     const IndexT round_tile_rows,
                                     const IndexT round_tile_cols,
                                     const IndexT cols,
                                     const IndexT rows) {
  constexpr int kRowTile = kTileSize * WriteSize;
  __shared__ T s_data[kRowTile * kShareCol * ReadSize];

  const IndexT chs_stride = rows * cols;
  TransposeDataReader<T, IndexT, ReadSize, kRowTile>()(
      src_data, s_data, cols, rows, chs_stride, cols, round_tile_rows);
  TransposeDataWriter<T, IndexT, ReadSize, WriteSize>()(
      dst_data,
      s_data,
      rows,
      cols,
      chs_stride * ReadSize / WriteSize,
      round_tile_cols);
}

template <typename T, typename IndexT, int VecSize>
struct PermuteLauncher {
 public:
  PermuteLauncher(const phi::GPUContext& ctx,
                  const int& rank,
                  const IndexT& count,
                  const PermuteType& perm_type,
                  const std::vector<int64_t>& dims,
                  const std::vector<int32_t>& perm,
                  const T* src,
                  T* dst)
      : dims_(dims) {
    main_cnt_ = count / VecSize;
#define CALL_PERMUTE_DISPATCH_RANK(rank_)              \
  case rank_: {                                        \
    Run<rank_>(ctx, perm, perm_type, count, src, dst); \
    break;                                             \
  }

    switch (rank) {
      CALL_PERMUTE_DISPATCH_RANK(3);
      CALL_PERMUTE_DISPATCH_RANK(4);
      CALL_PERMUTE_DISPATCH_RANK(5);
      CALL_PERMUTE_DISPATCH_RANK(6);
      CALL_PERMUTE_DISPATCH_RANK(7);
      CALL_PERMUTE_DISPATCH_RANK(8);
      CALL_PERMUTE_DISPATCH_RANK(9);
    }
#undef CALL_PERMUTE_DISPATCH_RANK
  }
  ~PermuteLauncher() {}

 private:
  IndexT main_cnt_{0};
  std::vector<int64_t> dims_;

  template <int Rank>
  void Run(const phi::GPUContext& ctx,
           const std::vector<int32_t>& perm,
           const PermuteType& perm_type,
           const IndexT& count,
           const T* src,
           T* dst) {
    auto cfg = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, main_cnt_);
    if (perm_type == PermuteType::kVecPermute) {
      dims_[Rank - 1] /= VecSize;
      const auto params = PermuteParams<Rank, IndexT>(dims_, perm);

      VectorizedPermuteKernel<T, IndexT, VecSize, Rank>
          <<<cfg.block_per_grid, cfg.thread_per_block, 0, ctx.stream()>>>(
              params, main_cnt_, src, dst);
    } else {
      IndexT tail_cnt = count - main_cnt_ * VecSize;
      IndexT main_offset = count - tail_cnt;
      const auto params = PermuteParams<Rank, IndexT>(dims_, perm);

      GeneralPermuteKernel<T, IndexT, VecSize, Rank>
          <<<cfg.block_per_grid, cfg.thread_per_block, 0, ctx.stream()>>>(
              params, main_cnt_, tail_cnt, main_offset, src, dst);
    }
  }
};

template <typename T, typename IndexT, int VecSize>
struct TransposeLauncher {
 public:
  TransposeLauncher(const phi::GPUContext& ctx,
                    const int& rank,
                    const PermuteType& perm_type,
                    const std::vector<int64_t>& dims,
                    const IndexT& num_rows_tile,
                    const T* src,
                    T* dst) {
    constexpr int ReadSize = sizeof(T) > sizeof(float) ? 1 : VecSize;
    const IndexT cols = dims[rank - 1] / VecSize;
    const IndexT n_cols_tile = GETTILESIZE(cols, kTileSize);

    if (perm_type == PermuteType::kGeneralTranspose) {
      IndexT chs = (rank == 2) ? 1 : dims[0];
      IndexT rows = dims[rank - 2];
      IndexT n_rows_tile =
          FindRowTiles(chs, rows, num_rows_tile, n_cols_tile, ctx.GetSMCount());
      dim3 blocks(n_cols_tile, n_rows_tile, chs);
      dim3 threads(kTileSize, kBlockRows, 1);

      if (is_vec_write) {
        BatchTransposeKernel<T, IndexT, true, ReadSize>
            <<<blocks, threads, 0, ctx.stream()>>>(
                src, dst, n_rows_tile - 1, n_cols_tile - 1, cols, rows);
      } else {
        BatchTransposeKernel<T, IndexT, false, ReadSize>
            <<<blocks, threads, 0, ctx.stream()>>>(
                src, dst, n_rows_tile - 1, n_cols_tile - 1, cols, rows);
      }
    } else {
      IndexT rows = dims[0];
      IndexT chs = dims[rank - 2];
      IndexT n_rows_tile =
          FindRowTiles(chs, rows, num_rows_tile, n_cols_tile, ctx.GetSMCount());
      dim3 blocks(n_cols_tile, n_rows_tile, chs);
      dim3 threads(kTileSize, kBlockRows, 1);

      if (is_vec_write) {
        SwapTransposeKernel<T, IndexT, true, ReadSize>
            <<<blocks, threads, 0, ctx.stream()>>>(
                src, dst, n_rows_tile - 1, n_cols_tile - 1, cols, rows, chs);
      } else {
        SwapTransposeKernel<T, IndexT, false, ReadSize>
            <<<blocks, threads, 0, ctx.stream()>>>(
                src, dst, n_rows_tile - 1, n_cols_tile - 1, cols, rows, chs);
      }
    }
  }
  ~TransposeLauncher() {}

 private:
  bool is_vec_write{false};
  inline IndexT FindRowTiles(const IndexT& chs,
                             const IndexT& rows,
                             const IndexT& num_rows_tile,
                             const IndexT& num_cols_tile,
                             const int& sm_count) {
    constexpr int kVecRow = sizeof(float) / sizeof(T);
    is_vec_write =
        (sizeof(T) < sizeof(float)) ? ((rows % kVecRow) ? false : true) : false;

    int vec_write = 1;
    if (is_vec_write) {
      is_vec_write = (chs * num_cols_tile * num_rows_tile) > sm_count;
      vec_write = is_vec_write ? kVecRow : 1;
    }
    IndexT n_rows_tile = is_vec_write
                             ? GETTILESIZE(rows, (kTileSize * vec_write))
                             : num_rows_tile;
    return n_rows_tile;
  }
};

template <typename T, typename IndexT>
struct PermuteDispatch {
 public:
  PermuteDispatch(const phi::GPUContext& ctx,
                  PermTypeClassifier<T>* cls_ptr,
                  const std::vector<int64_t>& dims,
                  const std::vector<int32_t>& perm,
                  const IndexT count,
                  const T* src,
                  T* dst)
      : dims_(dims), cls_(cls_ptr) {
    rank_ = dims_.size();
    type_ = cls_->GetPermType();
    KernelTypeDispatch(ctx, count, perm, src, dst);
  }
  ~PermuteDispatch() {}

 private:
  int rank_{0};
  std::vector<int64_t> dims_;
  PermTypeClassifier<T>* cls_;
  PermuteType type_{kGeneralPermute};

  void KernelTypeDispatch(const phi::GPUContext& ctx,
                          const IndexT& count,
                          const std::vector<int32_t>& perm,
                          const T* src,
                          T* dst) {
#define TRANSPOSE_DISPATCH_VEC_SIZE(size)                         \
  case size: {                                                    \
    TransposeLauncher<T, IndexT, size>(                           \
        ctx, rank_, type_, dims_, cls_->GetRowsTile(), src, dst); \
    break;                                                        \
  }

#define PERMUTE_DISPATCH_VEC_SIZE(size)                   \
  case size: {                                            \
    PermuteLauncher<T, IndexT, size>(                     \
        ctx, rank_, count, type_, dims_, perm, src, dst); \
    break;                                                \
  }

    switch (type_) {
      case kSwapTranspose:
      case kGeneralTranspose:
        switch (cls_->GetVecSize()) {
          TRANSPOSE_DISPATCH_VEC_SIZE(1);
          TRANSPOSE_DISPATCH_VEC_SIZE(2);
          TRANSPOSE_DISPATCH_VEC_SIZE(4);
        }
        break;
      default:
        switch (cls_->GetVecSize()) {
          PERMUTE_DISPATCH_VEC_SIZE(1);
          PERMUTE_DISPATCH_VEC_SIZE(2);
          PERMUTE_DISPATCH_VEC_SIZE(4);
        }
        break;
    }
#define TRANSPOSE_DISPATCH_VEC_SIZE
#define PERMUTE_DISPATCH_VEC_SIZE
  }
};

template <typename T>
inline void PermuteAndTranspose(const phi::GPUContext& ctx,
                                phi::DenseTensor* in,
                                phi::DenseTensor* out,
                                const DimsSimplifier<T>& simplifier) {
  T* dst_data = out->data<T>();
  const T* src_data = in->data<T>();
  const auto count = simplifier.GetCount();
  auto classifier = PermTypeClassifier<T>(ctx.GetSMCount(),
                                          simplifier.GetRank(),
                                          simplifier.GetPerm(),
                                          simplifier.GetSrcDims(),
                                          src_data,
                                          dst_data);
  if (classifier.GetPermType() == PermuteType::kCopy) {
    // If perm is [0,1,2,3], then just operate a DtoD copy.
    platform::GpuMemcpyAsync(dst_data,
                             src_data,
                             count * sizeof(T),
                             phi::gpuMemcpyDeviceToDevice,
                             ctx.stream());
  } else {
    if (count < std::numeric_limits<uint32_t>::max()) {
      PermuteDispatch<T, uint32_t>(ctx,
                                   &classifier,
                                   simplifier.GetSrcDims(),
                                   simplifier.GetPerm(),
                                   static_cast<uint32_t>(count),
                                   src_data,
                                   dst_data);
    } else {
      PermuteDispatch<T, int64_t>(ctx,
                                  &classifier,
                                  simplifier.GetSrcDims(),
                                  simplifier.GetPerm(),
                                  static_cast<int64_t>(count),
                                  src_data,
                                  dst_data);
    }
  }
}

template <typename T>
void TransposeGPUKernelDriver(const phi::GPUContext& ctx,
                              const phi::DenseTensor& in,
                              const std::vector<int32_t>& perm,
                              phi::DenseTensor* out) {
  auto simplifier = DimsSimplifier<T>(
      perm.size(), in.numel(), perm, phi::vectorize<int64_t>(in.dims()));
  auto* tuner = phi::autotune::MakeTransposeTuner<T>(TransposeWithSimple<T>);
  tuner->AddCallBack(PermuteWithEigen<T>);
  tuner->AddCallBack(PermuteAndTranspose<T>);

  size_t key = phi::autotune::TransposeKey(
      simplifier.GetSrcDims(),
      simplifier.GetPerm(),
      paddle::experimental::CppTypeToDataType<T>::Type());

  tuner->Run(ctx,
             phi::autotune::AlgorithmType::kTranspose,
             key,
             ctx,
             const_cast<phi::DenseTensor*>(&in),
             out,
             simplifier);
}

}  // namespace operators
}  // namespace paddle
