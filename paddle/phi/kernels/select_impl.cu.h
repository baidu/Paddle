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

#pragma once

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <algorithm>
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
namespace phi {
using Mode = kps::details::ReduceMode;

/*
* Count how many of the data being processed by the current block are true
* 1. Load data from global memory and cast from bool to int64_t
* 2. Get result of this thread according to thread reduce
* 3. Get result of this block according to block reduce
* 4. first block store 0 and current result
*/
template <typename T>
struct NonZeroFunctor {
  HOSTDEVICE NonZeroFunctor() {}
  HOSTDEVICE inline T operator()(const T in) {
    if (in) {
      return static_cast<T>(1);
    } else {
      return static_cast<T>(0);
    }
  }
};

template <typename InT, typename OutT, int VecSize, int IsBoundary>
__device__ void GetBlockCountImpl(const InT *in,
                                  OutT *out,
                                  int num,
                                  int repeat) {
  InT in_data[VecSize];
  OutT temp[VecSize];
  OutT result = static_cast<OutT>(0.0f);
  using Add = kps::AddFunctor<OutT>;
  using Cast = NonZeroFunctor<InT>;
  int store_fix = BLOCK_ID_X + repeat * GRID_NUM_X;

  kps::Init<InT, VecSize>(&in_data[0], static_cast<InT>(0.0f));
  kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(&in_data[0], in, num);
  kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Cast>(
      &temp[0], &in_data[0], Cast());
  kps::Reduce<OutT, VecSize, 1, 1, Add, Mode::kLocalMode>(
      &result, &temp[0], Add(), true);
  kps::Reduce<OutT, 1, 1, 1, Add, Mode::kGlobalMode>(
      &result, &result, Add(), true);
  if (store_fix == 0) {
    // first block's fix_size = 0;
    OutT tmp = static_cast<OutT>(0.0f);
    kps::WriteData<OutT, 1, 1, 1, true>(out + store_fix, &tmp, 1);
  }

  // store num of this block
  kps::WriteData<OutT, 1, 1, 1, true>(out + store_fix + 1, &result, 1);
}

template <typename InT, typename OutT, int VecSize>
__global__ void GetBlockCountKernel(const InT *in,
                                    OutT *out,
                                    int64_t numel,
                                    int64_t main_offset) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  int repeat = 0;
  for (; data_offset < main_offset; data_offset += stride) {
    GetBlockCountImpl<InT, OutT, VecSize, false>(
        in + data_offset, out, BLOCK_NUM_X * VecSize, repeat);
    repeat++;  // to get the real blockIdx
  }

  int num = numel - main_offset;
  if (num > 0) {
    GetBlockCountImpl<InT, OutT, VecSize, true>(
        in + data_offset, out, num, repeat);
  }
}

/*
* Get block num prefix us one block, VecSize must be 2
* 1. Each thread load 2 data : threadIdx.x and threadIdx.x + blockDimx.x
* 2. Cumsum limitation is blockDim.x must be less than 512
*/

template <typename InT,
          typename OutT,
          typename Functor,
          int VecSize,
          bool IsBoundary>
__device__ void CumsumImpl(
    const InT *in, OutT *out, OutT *pre_cumsum, int num, Functor func) {
  __shared__ OutT max_thread_data;
  OutT temp[VecSize];
  InT arg[VecSize];
  OutT result[VecSize];
  // init data_pr
  kps::Init<InT, VecSize>(&arg[0], static_cast<InT>(0.0f));
  // set pre_cumsum
  kps::Init<OutT, VecSize>(&temp[0], *pre_cumsum);
  // load data to arg
  kps::ReadData<InT, InT, VecSize, 1, 1, IsBoundary>(
      &arg[0], in, num, 1, BLOCK_NUM_X, 1);
  // block cumsum
  kps::Cumsum<InT, OutT, 1, Functor>(&result[0], &arg[0], func);
  // result = cumsum_result + pre_cumsum
  kps::ElementwiseBinary<OutT, OutT, VecSize, 1, 1, Functor>(
      &result[0], &result[0], &temp[0], func);
  // get the last prefix sum
  if ((THREAD_ID_X == BLOCK_NUM_X - 1) && !IsBoundary) {
    max_thread_data = result[VecSize - 1];
  }
  __syncthreads();
  // update pre_cumsum
  *pre_cumsum = max_thread_data;
  kps::WriteData<OutT, OutT, VecSize, 1, 1, IsBoundary>(
      out, &result[0], num, 1, BLOCK_NUM_X, 1);
}

template <typename InT, typename OutT, typename Functor, int VecSize>
__global__ void CumsumOneBlock(
    const InT *in, OutT *out, int numel, int main_offset, Functor func) {
  int stride = BLOCK_NUM_X * VecSize;
  int offset = 0;
  OutT pre_cumsum = static_cast<OutT>(0);
  for (; offset < main_offset; offset += stride) {
    CumsumImpl<InT, OutT, Functor, VecSize, false>(
        in + offset, out + offset, &pre_cumsum, BLOCK_NUM_X * VecSize, func);
  }

  int num = numel - offset;
  if (num > 0) {
    CumsumImpl<InT, OutT, Functor, VecSize, true>(
        in + offset, out + offset, &pre_cumsum, num, func);
  }
}

template <typename OutT,
          typename MT,
          typename InT,
          typename IdT,
          typename Functor,
          int VecSize,
          int IsBoundary,
          int IsMaskData>
struct SelectCaller {
  __device__ void inline operator()(OutT *store_data,
                                    const MT *mask_data,
                                    const InT *in,
                                    Functor func,
                                    int num,
                                    int data_offset);
};

template <typename OutT,
          typename MT,
          typename InT,
          typename IdT,
          typename Functor,
          int VecSize,
          int IsBoundary>
struct SelectCaller<OutT,
                    MT,
                    InT,
                    IdT,
                    Functor,
                    VecSize,
                    IsBoundary,
                    0> {  // where index
  __device__ void inline operator()(OutT *store_data,
                                    const MT *mask_data,
                                    const InT *in,
                                    Functor func,
                                    int num,
                                    int data_offset) {
    IdT index_reg[VecSize];
    // Set data index of global
    kps::InitWithDataIndex<IdT, VecSize, 1, 1>(&index_reg[0], data_offset);
    // Get store data according to mask_idt
    kps::OperatorTernary<MT, IdT, OutT, Functor>(
        store_data, mask_data, &index_reg[0], func, VecSize);
  }
};

template <typename OutT,
          typename MT,
          typename InT,
          typename IdT,
          typename Functor,
          int VecSize,
          int IsBoundary>
struct SelectCaller<OutT,
                    MT,
                    InT,
                    IdT,
                    Functor,
                    VecSize,
                    IsBoundary,
                    1> {  // masked_select
  __device__ void inline operator()(OutT *store_data,
                                    const MT *mask_data,
                                    const InT *in,
                                    Functor func,
                                    int num,
                                    int data_offset) {
    InT in_data[VecSize];
    kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(&in_data[0], in, num);
    // Get store data according to mask_idt
    kps::OperatorTernary<MT, InT, OutT, Functor>(
        store_data, mask_data, &in_data[0], func, VecSize);
  }
};

/**
* Get mask's index if mask == true
*/
template <typename InT,
          typename MT,
          typename OutT,
          typename Functor,
          int VecSize,
          int MaskData,
          int IsBoundary>  // SelectType = 1 Mask_select else where_index
__device__ void
SelectKernelImpl(OutT *out,
                 const MT *mask,
                 const InT *in,
                 Functor func,
                 int num,
                 int data_offset,
                 int store_rank) {
  const int kCVecSize = 2;
  // each thread cumsum 2 data
  using IdT = int64_t;
  // Set index data type
  using Add = kps::AddFunctor<IdT>;  // for cumsum
  using Cast = NonZeroFunctor<InT>;  // for mask

  IdT init_idx = static_cast<IdT>(0.0f);
  MT init_mask = static_cast<MT>(0.0f);

  IdT num_thread[kCVecSize];
  IdT cumsum_thread[kCVecSize];

  OutT store_data[VecSize * phi::DDim::kMaxRank];
  MT mask_data[VecSize];
  IdT mask_idt[VecSize];
  // init data_pr
  kps::Init<IdT, kCVecSize>(&cumsum_thread[0], init_idx);
  kps::Init<IdT, kCVecSize>(&num_thread[0], init_idx);
  kps::Init<MT, VecSize>(&mask_data[0], init_mask);
  // Load mask
  kps::ReadData<MT, VecSize, 1, 1, IsBoundary>(&mask_data[0], mask, num);
  // Cast from MT to int
  kps::ElementwiseUnary<MT, IdT, VecSize, 1, 1, Cast>(
      &mask_idt[0], &mask_data[0], Cast());
  // Get the num of thread only num_thread[1] has data
  kps::Reduce<IdT, VecSize, 1, 1, Add, Mode::kLocalMode>(
      &num_thread[0], &mask_idt[0], Add(), true);
  // Get cumsum_thread cumsum from 0 to num_thread cumsum_thread[0] is the
  // thread_fix
  kps::Cumsum<IdT, IdT, 1, Add>(&cumsum_thread[0], &num_thread[0], Add());
  // Get store data(index) according to mask_idt
  SelectCaller<OutT, MT, InT, IdT, Functor, VecSize, IsBoundary, MaskData>
      compute;
  compute(&store_data[0], &mask_data[0], in, func, num, data_offset);
  // get thread_fix
  int thread_fix =
      (static_cast<int>(cumsum_thread[0] - num_thread[0]) * store_rank);
  // get how many data need to store
  int store_num = static_cast<int>(num_thread[0]) * store_rank;
  // thread store num data, each thread may has different num
  kps::details::WriteData<OutT>(out + thread_fix, &store_data[0], store_num);
}

template <typename MT,
          typename InT,
          typename CT,
          typename OutT,
          typename Functor,
          int VecSize,
          int MaskData>
__global__ void SelectKernel(OutT *out,
                             const MT *mask,
                             const InT *in,
                             CT *cumsum,
                             Functor func,
                             const int64_t numel,
                             int64_t main_offset,
                             int store_rank) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  int repeat = 0;
  int size = VecSize * BLOCK_ID_X;
  for (; data_offset < main_offset; data_offset += stride) {
    // Cumsum index
    int idx_cumsum = repeat * GRID_NUM_X + BLOCK_ID_X;
    // niuliling todo: us ReadData API
    int block_store_offset = cumsum[idx_cumsum];
    SelectKernelImpl<InT, MT, OutT, Functor, VecSize, MaskData, false>(
        out + block_store_offset * store_rank,
        mask + data_offset,
        in + data_offset,
        func,
        size,
        data_offset,
        store_rank);
    repeat++;
  }

  int num = numel - data_offset;
  if (num > 0) {
    // Cumsum index
    int idx_cumsum = repeat * GRID_NUM_X + BLOCK_ID_X;
    // niuliling todo: us ReadData API
    int block_store_offset = static_cast<int>(cumsum[idx_cumsum]);
    SelectKernelImpl<InT, MT, OutT, Functor, VecSize, MaskData, true>(
        out + block_store_offset * store_rank,
        mask + data_offset,
        in + data_offset,
        func,
        num,
        data_offset,
        store_rank);
  }
}

inline int64_t Ceil(int64_t in, int64_t div) { return in / div * div; }

// SelectData = 1 then masked_select; SelectData = 0 then where_index
template <typename MT,
          typename InT,
          typename OutT,
          int SelectData,
          typename Functor>
void SelectKernel(const KPDevice &dev_ctx,
                  const DenseTensor &condition,
                  const DenseTensor &in_data,
                  DenseTensor *out,
                  Functor func) {
  const MT *cond_data = condition.data<MT>();
  const int64_t numel = condition.numel();
  auto dims = condition.dims();
  int rank = SelectData ? 1 : dims.size();
  const InT *in_data_ptr = SelectData ? in_data.data<InT>() : nullptr;
  // calculate the inclusive prefix sum of "true_num_array"
  // to get the index of "out" tensor,
  // and the total number of cond_data[i]==true.
  // Example:
  // condition: F T T F F F T T
  // before:    0 1 1 0 0 0 1 1
  // after:     0 1 2 2 2 2 3 4
  // out:       1 2 6 7
  // alloc for cpu
  using CT = int64_t;  // set Count_data Type
  const int t_size = sizeof(CT);
  const paddle::platform::CUDAPlace &cuda_place = dev_ctx.GetPlace();
  paddle::platform::CPUPlace cpu_place = paddle::platform::CPUPlace();

  auto cpu_buf_holder =
      paddle::memory::Alloc(cpu_place, (rank + 1 + 190) * t_size);
  CT *cpu_buf = reinterpret_cast<CT *>(cpu_buf_holder->ptr());

  // 1.1 get stored data num of per block
  const int kVecSize = 4;
#ifdef PADDLE_WITH_XPU_KP
  int block = 64;
  auto stream = dev_ctx.x_context()->xpu_stream;
  const int num_per_block = kVecSize * block;
  const int need_grids = (numel + num_per_block - 1) / num_per_block;
  const int grid = std::min(need_grids, 8);
#else
  const int block = 256;
  const int num_per_block = kVecSize * block;
  const int need_grids = (numel + num_per_block - 1) / num_per_block;
  const int grid = std::min(need_grids, 256);
  auto stream = dev_ctx.stream();
#endif
  const int64_t main_offset = Ceil(numel, num_per_block);
  // 1.2 alloc tmp data for CoutBlock
  const int size_count_block = need_grids + 1;
  std::vector<int> dims_vec = {size_count_block};
  ScalarArray dims_array(dims_vec);
  // DenseTensor count_mem = phi::Empty<CT, KPDevice>(dev_ctx, dims_array);
  // CT *count_data = count_mem.data<CT>();
  auto count_mem = paddle::memory::Alloc(cuda_place, size_count_block * t_size);
  CT *count_data = reinterpret_cast<CT *>(count_mem->ptr());
  // 1.3 launch CountKernl
  GetBlockCountKernel<MT, CT, kVecSize><<<grid, block, 0, stream>>>(
      cond_data, count_data, numel, main_offset);
  // 2.1 alloc cumsum data for CoutBlock prefix
  auto cumsum_mem =
      paddle::memory::Alloc(cuda_place, size_count_block * t_size);
  CT *cumsum_data = reinterpret_cast<CT *>(cumsum_mem->ptr());
  // 2.2 get prefix of count_data for real out_index
  const int kCumVesize = 2;
  const int block_c = 256;
  const int main_offset_c = Ceil(size_count_block, (kCumVesize * block_c));
  using Add = kps::AddFunctor<CT>;
  CumsumOneBlock<CT, CT, Add, kCumVesize><<<1, block_c, 0, stream>>>(
      count_data, cumsum_data, size_count_block, main_offset_c, Add());
  // 3.1 set temp ptr for in;
  // 3.1 alloc for out
  // 3.1.1 get true_num for gpu place the last cumsum is the true_num
  paddle::memory::Copy(cpu_place,
                       cpu_buf,
                       cuda_place,
                       cumsum_data + need_grids,
                       t_size,
                       dev_ctx.stream());

  dev_ctx.Wait();
  // 3.1.2 allock for out with total_true_num
  std::vector<int64_t> out_dim = {static_cast<int64_t>(cpu_buf[0])};
  if (SelectData == 0) {  // where_index
    out_dim.push_back(rank);
  }
  out->Resize(phi::make_ddim(out_dim));
  auto out_data = out->mutable_data<OutT>(cuda_place);
  // 3.2 get true data's index according to cond_data and cumsum_data
  if (cpu_buf[0] <= 0) return;
  SelectKernel<MT,
               InT,
               CT,
               OutT,
               Functor,
               kVecSize,
               SelectData><<<grid, block, 0, stream>>>(out_data,
                                                       cond_data,
                                                       in_data_ptr,
                                                       cumsum_data,
                                                       func,
                                                       numel,
                                                       main_offset,
                                                       rank);
}

}  // namespace phi
