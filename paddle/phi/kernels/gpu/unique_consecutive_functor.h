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
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <iostream>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/unique_functor.h"

namespace phi {

// The core logic of computing Unique Consecutive for a flattened Tensor
template <typename Context,
          typename InT,
          typename IndexT,
          typename equal_T,
          typename not_equal_T>
static void UniqueConsecutiveFlattenedCUDATensor(const Context& context,
                                                 const DenseTensor& in,
                                                 DenseTensor* out,
                                                 bool return_inverse,
                                                 bool return_counts,
                                                 equal_T equal,
                                                 not_equal_T not_equal,
                                                 int64_t num_input,
                                                 DenseTensor* inverse,
                                                 DenseTensor* counts) {
#ifdef PADDLE_WITH_CUDA
  const auto& exec_policy = thrust::cuda::par.on(context.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(context.stream());
#endif
  // 0. Preparation
  DenseTensor in_hat;
  phi::Copy(context, in, context.GetPlace(), false, &in_hat);
  auto in_data_hat = context.template Alloc<InT>(&in_hat);

  DenseTensor sorted_indices;
  sorted_indices.Resize(common::make_ddim({num_input}));
  auto sorted_indices_data = context.template Alloc<IndexT>(&sorted_indices);
  thrust::sequence(
      exec_policy, sorted_indices_data, sorted_indices_data + num_input);
  // 1. Calculate op result: 'out'
  DenseTensor range;
  range.Resize(common::make_ddim({num_input + 1}));
  auto range_data_ptr = context.template Alloc<IndexT>(&range);
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

  // 2. Calculate inverse index: 'inverse'
  if (return_inverse) {
    inverse->Resize(common::make_ddim({num_input}));
    auto inverse_data = context.template Alloc<IndexT>(inverse);
    DenseTensor inv_loc;
    inv_loc.Resize(common::make_ddim({num_input}));
    auto inv_loc_data_ptr = context.template Alloc<IndexT>(&inv_loc);
    thrust::adjacent_difference(exec_policy,
                                in_data_hat,
                                in_data_hat + num_input,
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
                    sorted_indices_data,
                    inverse_data);
  }
  // 3. Calculate 'counts'
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

// functor for processing a flattened Tensor
template <typename Context, typename InT>
struct UniqueConsecutiveFlattenedCUDAFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  const bool return_inverse_;
  const bool return_counts_;
  DenseTensor* inverse_;
  DenseTensor* count_;

  UniqueConsecutiveFlattenedCUDAFunctor(const Context& context,
                                        const DenseTensor& in,
                                        DenseTensor* out,
                                        bool return_inverse,
                                        bool return_counts,
                                        DenseTensor* inverse,
                                        DenseTensor* count)
      : ctx_(context),
        in_(in),
        out_(out),
        return_inverse_(return_inverse),
        return_counts_(return_counts),
        inverse_(inverse),
        count_(count) {}

  template <typename IndexT>
  void apply() const {
    UniqueConsecutiveFlattenedCUDATensor<Context, InT, IndexT>(
        ctx_,
        in_,
        out_,
        return_inverse_,
        return_counts_,
        thrust::equal_to<InT>(),
        thrust::not_equal_to<InT>(),
        in_.numel(),
        inverse_,
        count_);
  }
};

// The logic of compute unique with axis required, it's a little different
// from above function
template <typename Context,
          typename InT,
          typename IndexT,
          typename equal_T,
          typename not_equal_T>
static void ComputeUniqueConsecutiveDims(const Context& context,
                                         DenseTensor* sorted_indices,
                                         IndexT* sorted_indices_data,
                                         DenseTensor* out,
                                         bool return_inverse,
                                         bool return_counts,
                                         equal_T equal,
                                         not_equal_T not_equal,
                                         int64_t row,
                                         DenseTensor* inverse,
                                         DenseTensor* counts) {
#ifdef PADDLE_WITH_CUDA
  const auto& exec_policy = thrust::cuda::par.on(context.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(context.stream());
#endif
  // 1. inverse indices: 'inverse'
  DenseTensor tmp;
  if (!inverse) {
    inverse = &tmp;
  }

  inverse->Resize(common::make_ddim({row}));
  auto inverse_data = context.template Alloc<IndexT>(inverse);
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
    auto count_data = context.template Alloc<IndexT>(counts);
    thrust::fill(exec_policy, count_data, count_data + row, 0);
    thrust::adjacent_difference(
        exec_policy, range_data_ptr + 1, range_data_ptr + row + 1, count_data);
  }
}

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

// index_select() function for Tensor
template <typename Context, typename InT, typename IndexT>
void IndexSelect(const Context& context,
                 const DenseTensor& input,
                 const DenseTensor& index,
                 DenseTensor* output,
                 int dim) {
  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  auto index_size = index.dims()[0];

  std::vector<InT> input_vec;
  std::vector<IndexT> index_vec;
  phi::TensorToVector(input, context, &input_vec);
  phi::TensorToVector(index, context, &index_vec);
  std::vector<InT> out_vec(output->numel());

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_vec[i],
        -input_dim[dim],
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -input_dim[dim],
            input_dim[dim],
            index_vec[i]));
    PADDLE_ENFORCE_LT(
        index_vec[i],
        input_dim[dim],
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -input_dim[dim],
            input_dim[dim],
            index_vec[i]));
  }

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = index_vec[j];
      if (index_value < 0) {
        index_value += input_dim[dim];
      }
      for (auto k = 0; k < slice_size; k++) {
        out_vec[output_start_offset + j * slice_size + k] =
            input_vec[input_start_offset + index_value * slice_size + k];
      }
    }
  }
  context.template Alloc<InT>(output);
  phi::TensorFromVector(out_vec, context, output);
  output->Resize(output_dim);
}

// Calculate unique consecutive when 'axis' is set
template <typename Context, typename InT, typename IndexT>
static void UniqueConsecutiveDimsCUDATensor(const Context& context,
                                            const DenseTensor& in,
                                            DenseTensor* out,
                                            bool return_inverse,
                                            bool return_counts,
                                            int axis,
                                            DenseTensor* inverse,
                                            DenseTensor* counts) {
  // 1. Transpose & reshape
  // Transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(common::vectorize(in.dims()));
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  DenseTensor in_trans;
  DDim in_trans_dims = common::make_ddim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  context.template Alloc<InT>(&in_trans);
  phi::funcs::TransCompute<Context, InT>(in.dims().size(),  // num of dims
                                         context,           // device
                                         in,                // original Tensor
                                         &in_trans,  // Tensor after reshape
                                         permute);   // index of axis

  // Reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  DDim in_trans_flat_dims = common::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // now 'in_trans' is 2D
  int64_t col = in_trans.dims()[1];
  int64_t row = in_trans.dims()[0];
  const InT* in_trans_data = in_trans.data<InT>();

  DenseTensor sorted_indices;
  sorted_indices.Resize(common::make_ddim({row}));
  auto sorted_indices_data = context.template Alloc<IndexT>(&sorted_indices);

  // 2. Calculate 'inverse', 'counts'
  // Init index
  thrust::sequence(exec_policy, sorted_indices_data, sorted_indices_data + row);
  ComputeUniqueConsecutiveDims<Context, InT, IndexT>(
      context,
      &sorted_indices,
      sorted_indices_data,
      out,
      return_inverse,
      return_counts,
      BinaryEqual<InT>(col, in_trans_data),
      BinaryNotEqual<InT>(col, in_trans_data),
      row,
      inverse,
      counts);

  // 3. Select indices and reshape back to get 'out'
  DenseTensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = sorted_indices.numel();
  out_trans.Resize(common::make_ddim(out_trans_dims_vec));
  context.template Alloc<InT>(&out_trans);

  IndexSelect<Context, InT, IndexT>(
      context, in_trans, sorted_indices, &out_trans, 0);

  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(common::make_ddim(out_trans_dims_vec));
  context.template Alloc<InT>(out);
  std::vector<DenseTensor> out_trans_unbind = phi::funcs::Unbind(out_trans);
  phi::funcs::ConcatFunctor<Context, InT> concat_functor;
  concat_functor(context, out_trans_unbind, 0, &out_trans);
  phi::funcs::TransCompute<Context, InT>(
      out_trans.dims().size(), context, out_trans, out, permute);
}

// functor for processing a multi-dimensional Tensor
template <typename Context, typename InT>
struct UniqueConsecutiveDimsCUDAFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  const int axis_;
  const bool return_inverse_;
  const bool return_counts_;
  DenseTensor* inverse_;
  DenseTensor* count_;

  UniqueConsecutiveDimsCUDAFunctor(const Context& context,
                                   const DenseTensor& in,
                                   DenseTensor* out,
                                   const int axis,
                                   bool return_inverse,
                                   bool return_counts,
                                   DenseTensor* inverse,
                                   DenseTensor* count)
      : ctx_(context),
        in_(in),
        out_(out),
        axis_(axis),
        return_inverse_(return_inverse),
        return_counts_(return_counts),
        inverse_(inverse),
        count_(count) {}

  template <typename IndexT>
  void apply() const {
    UniqueConsecutiveDimsCUDATensor<Context, InT, IndexT>(ctx_,
                                                          in_,
                                                          out_,
                                                          return_inverse_,
                                                          return_counts_,
                                                          axis_,
                                                          inverse_,
                                                          count_);
  }
};

}  // namespace phi
