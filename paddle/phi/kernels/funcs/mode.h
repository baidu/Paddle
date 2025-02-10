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

#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

static int ComputeBlockSize(int col) {
  if (col > 512)
    return 1024;
  else if (col > 256 && col <= 512)
    return 512;
  else if (col > 128 && col <= 256)
    return 256;
  else if (col > 64 && col <= 128)
    return 128;
  else
    return 64;
}

static inline void GetDims(
    const phi::DDim& dim, int axis, int* pre, int* n, int* post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

template <typename T, typename Type>
static void GetMode(Type input_height,
                    Type input_width,
                    int input_dim,
                    const DenseTensor* input,
                    T* t_out,
                    Type* t_indices) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    std::sort(col_vec.begin(),
              col_vec.end(),
              [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                return (!std::isnan(static_cast<double>(l.first)) &&
                        std::isnan(static_cast<double>(r.first))) ||
                       (l.first < r.first);
              });
    T mode = 0;
    int64_t indice = 0;
    int64_t cur_freq = 0;
    int64_t max_freq = 0;
    for (int64_t i = 0; i < input_width; ++i) {
      ++cur_freq;
      if (i == input_width - 1 || (col_vec[i + 1].first != col_vec[i].first)) {
        if (cur_freq > max_freq) {
          max_freq = cur_freq;
          mode = col_vec[i].first;
          indice = col_vec[i].second;
        }
        cur_freq = 0;
      }
    }
    t_out[i] = mode;
    t_indices[i] = indice;
  }
}

template <typename T, typename Type>
static void ModeAssign(const Type& input_height,
                       const Type& input_width,
                       const int& input_dim,
                       const DenseTensor* input,
                       const DenseTensor* indices,
                       T* output_data) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      auto e_indices = EigenVector<Type>::Flatten(*indices);
      output_data[i * input_width + e_indices(0)] = e_input(0);
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices = EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      output_data[i * input_width + e_indices(i, 0)] = e_input(i, 0);
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
static void GetModebySort(const phi::GPUContext& dev_ctx,
                          const DenseTensor* input_tensor,
                          const int64_t num_cols,
                          const int64_t num_rows,
                          T* out_tensor,
                          int64_t* indices_tensor) {
  DenseTensor input_tmp;
  input_tmp.Resize(common::make_ddim({num_rows, num_cols}));
  T* input_tmp_data = dev_ctx.Alloc<T>(&input_tmp);
  phi::Copy(dev_ctx, *input_tensor, dev_ctx.GetPlace(), false, &input_tmp);

  thrust::device_ptr<T> out_tensor_ptr(out_tensor);
  thrust::device_ptr<int64_t> indices_tensor_ptr(indices_tensor);

#ifdef PADDLE_WITH_CUDA
  const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif

  for (int64_t i = 0; i < num_rows; ++i) {
    T* begin = input_tmp_data + num_cols * i;
    T* end = input_tmp_data + num_cols * (i + 1);
    thrust::device_vector<int64_t> indices_data(num_cols);
    thrust::sequence(
        exec_policy, indices_data.begin(), indices_data.begin() + num_cols);
    thrust::sort_by_key(exec_policy, begin, end, indices_data.begin());
    int unique = 1 + thrust::inner_product(exec_policy,
                                           begin,
                                           end - 1,
                                           begin + 1,
                                           0,
                                           thrust::plus<int>(),
                                           thrust::not_equal_to<T>());
    thrust::device_vector<T> keys_data(unique);
    thrust::device_vector<int64_t> cnts_data(unique);
    thrust::reduce_by_key(exec_policy,
                          begin,
                          end,
                          thrust::constant_iterator<int>(1),
                          keys_data.begin(),
                          cnts_data.begin());
    auto it = thrust::max_element(
        exec_policy, cnts_data.begin(), cnts_data.begin() + unique);
    T mode = keys_data[it - cnts_data.begin()];
    int64_t counts = cnts_data[it - cnts_data.begin()];
    auto pos = thrust::find(exec_policy, begin, end, mode);
    int64_t index = indices_data[pos - begin + counts - 1];
    out_tensor_ptr[i] = static_cast<T>(mode);
    indices_tensor_ptr[i] = static_cast<int64_t>(index);
  }
}
#endif

}  // namespace funcs
}  // namespace phi
