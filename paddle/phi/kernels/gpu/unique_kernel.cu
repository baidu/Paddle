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

#include "paddle/phi/kernels/unique_kernel.h"

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <iostream>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include "cub/cub.cuh"
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/unique_functor.h"
#include "paddle/phi/kernels/index_select_kernel.h"

namespace phi {

// Binary function 'less than'
template <typename InT>
struct LessThan {
  int col;
  const InT* in_trans_data;

  LessThan(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }
};

// Binary function 'equal_to'
template <typename InT>
struct BinaryEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
};

// Binary function 'not_equal_to'
template <typename InT>
struct BinaryNotEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryNotEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return true;
      }
    }
    return false;
  }
};

// The core logic of computing Unique for a flattened DenseTensor
template <typename Context, typename InT, typename IndexT>
static typename std::enable_if<
    !std::is_same<InT, phi::dtype::float16>::value &&
    !std::is_same<InT, phi::dtype::bfloat16>::value>::type
UniqueFlattenedCUDATensor(const Context& context,
                          const DenseTensor& in,
                          DenseTensor* out,
                          DenseTensor* indices,
                          DenseTensor* index,
                          DenseTensor* counts,
                          bool return_index,
                          bool return_inverse,
                          bool return_counts,
                          int64_t num_input) {
  // 0. Preparation
  auto equal = thrust::equal_to<InT>();
  auto not_equal = thrust::not_equal_to<InT>();
  DenseTensor in_hat;
  phi::Copy(context, in, context.GetPlace(), false, &in_hat);
  auto* in_data_hat = context.template Alloc<InT>(&in_hat);
  DenseTensor tmp;
  if (!indices) {
    indices = &tmp;
  }

  indices->Resize(common::make_ddim({num_input}));
  auto* indices_data = context.template Alloc<IndexT>(indices);

#ifdef PADDLE_WITH_CUDA
  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(context.GetPlace(),
                                                             context.stream());
  const auto& exec_policy = thrust::cuda::par(allocator).on(context.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(context.stream());
#endif

  thrust::sequence(exec_policy, indices_data, indices_data + num_input);
  thrust::sort_by_key(
      exec_policy, in_data_hat, in_data_hat + num_input, indices_data);

  // 1. Calculate op result: 'out'
  DenseTensor range;
  range.Resize(common::make_ddim({num_input + 1}));
  auto* range_data_ptr = context.template Alloc<IndexT>(&range);
  thrust::sequence(exec_policy, range_data_ptr, range_data_ptr + num_input + 1);
  phi::Copy(context, in_hat, context.GetPlace(), false, out);
  int num_out;
  auto out_data = context.template Alloc<InT>(out);
  num_out =
      thrust::unique_by_key(
          exec_policy, out_data, out_data + num_input, range_data_ptr, equal)
          .first -
      out_data;
  out->Resize(common::make_ddim({num_out}));

  // 3. Calculate inverse index: 'inverse'
  if (return_inverse) {
    index->Resize(common::make_ddim({num_input}));
    auto* inverse_data = context.template Alloc<IndexT>(index);
    DenseTensor inv_loc;
    inv_loc.Resize(common::make_ddim({num_input}));
    auto inv_loc_data_ptr = context.template Alloc<IndexT>(&inv_loc);
    thrust::adjacent_difference(exec_policy,
                                in_data_hat,
                                in_data_hat + num_input,
                                inv_loc_data_ptr,
                                not_equal);
#ifdef PADDLE_WITH_HIP
    hipMemset(inv_loc_data_ptr, 0, sizeof(IndexT));
#else
    thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
    inv_loc_data_dev[0] = 0;  // without device_ptr, segmentation fault
#endif

#ifdef PADDLE_WITH_HIP
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(NULL,
                                  temp_storage_bytes,
                                  inv_loc_data_ptr,
                                  inv_loc_data_ptr,
                                  num_input,
                                  context.stream());
    auto d_temp_storage =
        phi::memory_utils::Alloc(context.GetPlace(), temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                  temp_storage_bytes,
                                  inv_loc_data_ptr,
                                  inv_loc_data_ptr,
                                  num_input,
                                  context.stream());
#else
    thrust::inclusive_scan(exec_policy,
                           inv_loc_data_ptr,
                           inv_loc_data_ptr + num_input,
                           inv_loc_data_ptr);
#endif
    thrust::scatter(exec_policy,
                    inv_loc_data_ptr,
                    inv_loc_data_ptr + num_input,
                    indices_data,
                    inverse_data);
  }

  // 2. Calculate sorted index: 'indices'
  if (return_index) {
    DenseTensor tmp_indices;
    tmp_indices.Resize(common::make_ddim({num_input}));
    auto* tmp_indices_data_ptr = context.template Alloc<IndexT>(&tmp_indices);
    thrust::copy(exec_policy,
                 in_data_hat,
                 in_data_hat + num_input,
                 tmp_indices_data_ptr);
    thrust::unique_by_key(exec_policy,
                          tmp_indices_data_ptr,
                          tmp_indices_data_ptr + num_input,
                          indices_data,
                          equal);
    indices->Resize(common::make_ddim({num_out}));
  }

  // 4. Calculate 'counts'
  if (return_counts) {
    counts->Resize(common::make_ddim({num_out}));
    auto count_data = context.template Alloc<IndexT>(counts);
    // init 'count_data' as 0
    thrust::fill(exec_policy, count_data, count_data + num_out, 0);
    thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
    range_data_ptr_dev[num_out] = num_input;
    thrust::adjacent_difference(exec_policy,
                                range_data_ptr + 1,
                                range_data_ptr + num_out + 1,
                                count_data);
  }
}

// The core logic of computing Unique for a flattened DenseTensor
template <typename Context, typename InT, typename IndexT>
static typename std::enable_if<
    std::is_same<InT, phi::dtype::float16>::value ||
    std::is_same<InT, phi::dtype::bfloat16>::value>::type
UniqueFlattenedCUDATensor(const Context& context,
                          const DenseTensor& in,
                          DenseTensor* out,
                          DenseTensor* indices,
                          DenseTensor* index,
                          DenseTensor* counts,
                          bool return_index,
                          bool return_inverse,
                          bool return_counts,
                          int64_t num_input) {
  // 1. Sort indices
  DenseTensor in_resize;
  in_resize.ShareDataWith(in);
  in_resize.Resize(common::make_ddim({num_input}));
  const InT* in_data = in_resize.data<InT>();
  auto equal = BinaryEqual<InT>(1, in_data);
  auto not_equal = BinaryNotEqual<InT>(1, in_data);

  DenseTensor tmp;
  if (!indices) {
    indices = &tmp;
  }

  indices->Resize(common::make_ddim({num_input}));
  auto* indices_data = context.template Alloc<IndexT>(indices);

#ifdef PADDLE_WITH_CUDA
  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(context.GetPlace(),
                                                             context.stream());
  const auto& exec_policy = thrust::cuda::par(allocator).on(context.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(context.stream());
#endif
  thrust::sequence(exec_policy, indices_data, indices_data + num_input);
  thrust::sort(exec_policy,
               indices_data,
               indices_data + num_input,
               LessThan<InT>(1, in_data));

  // 2. Calculate inverse indices: 'index'
  if (return_inverse) {
    index->Resize(common::make_ddim({num_input}));
    auto* inverse_data = context.template Alloc<IndexT>(index);
    DenseTensor inv_loc;
    inv_loc.Resize(common::make_ddim({num_input}));
    auto inv_loc_data_ptr = context.template Alloc<IndexT>(&inv_loc);
    thrust::adjacent_difference(exec_policy,
                                indices_data,
                                indices_data + num_input,
                                inv_loc_data_ptr,
                                not_equal);
    thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
    inv_loc_data_dev[0] = 0;  // without device_ptr, segmentation fault
    thrust::inclusive_scan(exec_policy,
                           inv_loc_data_ptr,
                           inv_loc_data_ptr + num_input,
                           inv_loc_data_ptr);
    thrust::scatter(exec_policy,
                    inv_loc_data_ptr,
                    inv_loc_data_ptr + num_input,
                    indices_data,
                    inverse_data);
  }

  // 3. Calculate op result and sorted index: 'out' & 'indices'
  DenseTensor range;
  range.Resize(common::make_ddim({num_input + 1}));
  auto* range_data_ptr = context.template Alloc<IndexT>(&range);
  thrust::sequence(exec_policy, range_data_ptr, range_data_ptr + num_input + 1);
  int num_out;
  num_out = thrust::unique_by_key(exec_policy,
                                  indices_data,
                                  indices_data + num_input,
                                  range_data_ptr,
                                  equal)
                .first -
            indices_data;
  indices->Resize(common::make_ddim({num_out}));
  out->Resize(common::make_ddim({num_out}));
  context.template Alloc<InT>(out);
  phi::IndexSelectKernel<InT, Context>(context, in_resize, *indices, 0, out);

  // 4. Calculate 'counts'
  if (return_counts) {
    counts->Resize(common::make_ddim({num_out}));
    auto count_data = context.template Alloc<IndexT>(counts);
    // init 'count_data' as 0
    thrust::fill(exec_policy, count_data, count_data + num_out, 0);
    thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
    range_data_ptr_dev[num_out] = num_input;
    thrust::adjacent_difference(exec_policy,
                                range_data_ptr + 1,
                                range_data_ptr + num_out + 1,
                                count_data);
  }
}

// The logic of compute unique with axis required, it's a little different
// from above function
template <typename Context,
          typename InT,
          typename IndexT,
          typename equal_T,
          typename not_equal_T>
static void ComputeUniqueDims(const Context& context,
                              DenseTensor* sorted_indices,
                              IndexT* sorted_indices_data,
                              DenseTensor* out,
                              DenseTensor* inverse,
                              DenseTensor* counts,
                              bool return_index,
                              bool return_inverse,
                              bool return_counts,
                              equal_T equal,
                              not_equal_T not_equal,
                              int64_t row) {
#ifdef PADDLE_WITH_CUDA
  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(context.GetPlace(),
                                                             context.stream());
  const auto& exec_policy = thrust::cuda::par(allocator).on(context.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(context.stream());
#endif
  // 1. inverse indices: 'inverse'
  inverse->Resize(common::make_ddim({row}));
  auto* inverse_data = context.template Alloc<IndexT>(inverse);
  DenseTensor inv_loc;
  inv_loc.Resize(common::make_ddim({row}));
  auto inv_loc_data_ptr = context.template Alloc<IndexT>(&inv_loc);
  thrust::adjacent_difference(exec_policy,
                              sorted_indices_data,
                              sorted_indices_data + row,
                              inv_loc_data_ptr,
                              not_equal);
  thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
  inv_loc_data_dev[0] = 0;
  thrust::inclusive_scan(
      exec_policy, inv_loc_data_ptr, inv_loc_data_ptr + row, inv_loc_data_ptr);
  thrust::scatter(exec_policy,
                  inv_loc_data_ptr,
                  inv_loc_data_ptr + row,
                  sorted_indices_data,
                  inverse_data);

  // 2. sorted indices
  DenseTensor range;
  range.Resize(common::make_ddim({row + 1}));
  auto range_data_ptr = context.template Alloc<IndexT>(&range);
  thrust::sequence(exec_policy, range_data_ptr, range_data_ptr + row + 1);
  int num_out;
  num_out = thrust::unique_by_key(exec_policy,
                                  sorted_indices_data,
                                  sorted_indices_data + row,
                                  range_data_ptr,
                                  equal)
                .first -
            sorted_indices_data;
  thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
  range_data_ptr_dev[num_out] = row;
  sorted_indices->Resize(common::make_ddim({num_out}));

  // 3. counts: 'counts'
  if (return_counts) {
    counts->Resize(common::make_ddim({num_out}));
    auto* count_data = context.template Alloc<IndexT>(counts);
    thrust::fill(exec_policy, count_data, count_data + num_out, 0);
    thrust::adjacent_difference(exec_policy,
                                range_data_ptr + 1,
                                range_data_ptr + num_out + 1,
                                count_data);
  }
}

// Calculate unique when 'axis' is set
template <typename Context, typename InT, typename IndexT>
static void UniqueDimsCUDATensor(const Context& context,
                                 const DenseTensor& in,
                                 DenseTensor* out,
                                 DenseTensor* indices,
                                 DenseTensor* index,
                                 DenseTensor* counts,
                                 bool return_index,
                                 bool return_inverse,
                                 bool return_counts,
                                 int axis) {
  // 1. Transpose & reshape
  // Transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  DenseTensor in_trans;
  std::vector<int64_t> in_trans_dims_vec(common::vectorize(in.dims()));
  auto in_trans_dims = common::make_ddim(in_trans_dims_vec);
  std::vector<int> permute(in.dims().size());
  bool is_transpose = axis != 0;
  if (is_transpose) {
    std::iota(permute.begin(), permute.end(), 0);
    permute[axis] = 0;
    permute[0] = axis;
    in_trans_dims_vec[axis] = in.dims()[0];
    in_trans_dims_vec[0] = in.dims()[axis];
    in_trans_dims = common::make_ddim(in_trans_dims_vec);
    in_trans.Resize(in_trans_dims);
    context.template Alloc<InT>(&in_trans);
    phi::funcs::TransCompute<Context, InT>(
        in.dims().size(),  // num of dims
        context,           // device
        in,                // original DenseTensor
        &in_trans,         // DenseTensor after reshape
        permute);          // index of axis
  } else {
    in_trans.ShareDataWith(in);
  }
  // Reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  auto in_trans_flat_dims = common::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // now 'in_trans' is 2D
  int64_t col = in_trans.dims()[1];
  int64_t row = in_trans.dims()[0];
  const InT* in_trans_data = in_trans.data<InT>();

  DenseTensor tmp;
  if (!indices) {
    indices = &tmp;
  }

  indices->Resize(common::make_ddim({row}));
  auto* sorted_indices_data = context.template Alloc<IndexT>(indices);

  // 2. Calculate 'indices', 'inverse', 'counts'
  // Init index and sort
#ifdef PADDLE_WITH_CUDA
  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(context.GetPlace(),
                                                             context.stream());
  const auto& exec_policy = thrust::cuda::par(allocator).on(context.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(context.stream());
#endif
  thrust::sequence(exec_policy, sorted_indices_data, sorted_indices_data + row);
  thrust::sort(exec_policy,
               sorted_indices_data,
               sorted_indices_data + row,
               LessThan<InT>(col, in_trans_data));
  ComputeUniqueDims<Context, InT, IndexT>(
      context,
      indices,
      sorted_indices_data,
      out,
      index,
      counts,
      return_index,
      return_inverse,
      return_counts,
      BinaryEqual<InT>(col, in_trans_data),
      BinaryNotEqual<InT>(col, in_trans_data),
      row);

  // 3. Select indices and reshape back to get 'out'
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = indices->numel();
  if (is_transpose) {
    DenseTensor out_trans;
    out_trans.Resize(common::make_ddim(out_trans_dims_vec));
    context.template Alloc<InT>(&out_trans);

    phi::IndexSelectKernel<InT, Context>(
        context, in_trans, *indices, 0, &out_trans);

    std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
    out->Resize(common::make_ddim(out_trans_dims_vec));
    context.template Alloc<InT>(out);
    phi::funcs::TransCompute<Context, InT>(
        out_trans.dims().size(), context, out_trans, out, permute);
  } else {
    out->Resize(common::make_ddim(out_trans_dims_vec));
    context.template Alloc<InT>(out);

    phi::IndexSelectKernel<InT, Context>(context, in_trans, *indices, 0, out);
  }
}

// functor for processing a flattened DenseTensor
template <typename Context, typename InT>
struct UniqueFlattenedCUDAFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  DenseTensor* indices_;
  DenseTensor* index_;
  DenseTensor* counts_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueFlattenedCUDAFunctor(const Context& context,
                             const DenseTensor& in,
                             DenseTensor* out,
                             DenseTensor* indices,
                             DenseTensor* index,
                             DenseTensor* counts,
                             bool return_index,
                             bool return_inverse,
                             bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        indices_(indices),
        index_(index),
        counts_(counts),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueFlattenedCUDATensor<Context, InT, IndexT>(ctx_,
                                                    in_,
                                                    out_,
                                                    indices_,
                                                    index_,
                                                    counts_,
                                                    return_index_,
                                                    return_inverse_,
                                                    return_counts_,
                                                    in_.numel());
  }
};

// functor for processing a multi-dimensional DenseTensor
template <typename Context, typename InT>
struct UniqueDimsCUDAFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  DenseTensor* indices_;
  DenseTensor* index_;
  DenseTensor* counts_;
  const int axis_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueDimsCUDAFunctor(const Context& context,
                        const DenseTensor& in,
                        DenseTensor* out,
                        DenseTensor* indices,
                        DenseTensor* index,
                        DenseTensor* counts,
                        const int axis,
                        bool return_index,
                        bool return_inverse,
                        bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        indices_(indices),
        index_(index),
        counts_(counts),
        axis_(axis),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueDimsCUDATensor<Context, InT, IndexT>(ctx_,
                                               in_,
                                               out_,
                                               indices_,
                                               index_,
                                               counts_,
                                               return_index_,
                                               return_inverse_,
                                               return_counts_,
                                               axis_);
  }
};

template <typename T, typename Context>
void UniqueRawKernel(const Context& context,
                     const DenseTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     bool is_sorted,
                     DenseTensor* out,
                     DenseTensor* indices,
                     DenseTensor* index,
                     DenseTensor* counts) {
  if (dtype == phi::DataType::INT32) {
    PADDLE_ENFORCE_LE(
        x.numel() + 1,
        INT_MAX,
        common::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }
  // if 'axis' is not required, flatten the DenseTensor.
  if (axis.empty()) {
    phi::VisitDataTypeTiny(
        dtype,
        UniqueFlattenedCUDAFunctor<Context, T>(context,
                                               x,
                                               out,
                                               indices,
                                               index,
                                               counts,
                                               return_index,
                                               return_inverse,
                                               return_counts));
  } else {
    // 'axis' is required.
    int axis_value = axis[0];
    axis_value = (axis_value == -1) ? (x.dims().size() - 1) : axis_value;
    phi::VisitDataTypeTiny(dtype,
                           UniqueDimsCUDAFunctor<Context, T>(context,
                                                             x,
                                                             out,
                                                             indices,
                                                             index,
                                                             counts,
                                                             axis_value,
                                                             return_index,
                                                             return_inverse,
                                                             return_counts));
  }
}

template <typename T, typename Context>
void UniqueKernel(const Context& context,
                  const DenseTensor& x,
                  bool return_index,
                  bool return_inverse,
                  bool return_counts,
                  const std::vector<int>& axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices,
                  DenseTensor* index,
                  DenseTensor* counts) {
  bool is_sorted = true;
  UniqueRawKernel<T, Context>(context,
                              x,
                              return_index,
                              return_inverse,
                              return_counts,
                              axis,
                              dtype,
                              is_sorted,
                              out,
                              indices,
                              index,
                              counts);
}

}  // namespace phi

PD_REGISTER_KERNEL(unique,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniqueKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(unique_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniqueRawKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}
