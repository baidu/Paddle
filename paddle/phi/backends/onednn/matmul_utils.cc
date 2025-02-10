// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/onednn/matmul_utils.h"

namespace phi::funcs {

DDim RowMatrixDimsFromVector(const DDim& x_dim) {
  return x_dim.size() > 1 ? x_dim : common::make_ddim({1, x_dim[0]});
}

DDim ColumnMatrixDimsFromVector(const DDim& y_dim) {
  return y_dim.size() > 1 ? y_dim : common::make_ddim({y_dim[0], 1});
}

std::vector<int64_t> TransposeAxis(const std::vector<int64_t>& x,
                                   const std::vector<int>& axis) {
  size_t in_rank = x.size();
  size_t axis_size = axis.size();

  auto axis_set = std::set<int>(axis.begin(), axis.end());
  PADDLE_ENFORCE_EQ(
      axis_set.size(),
      axis_size,
      errors::InvalidArgument("In an axis array, elements must be unique."));

  PADDLE_ENFORCE_EQ(
      in_rank,
      axis_size,
      errors::InvalidArgument("The input dimension's size "
                              "should be equal to the axis's size. "
                              "But received dimension is %d, "
                              "axis's size is %d",
                              in_rank,
                              axis_size));

  PADDLE_ENFORCE_LT(*std::max_element(axis.begin(), axis.end()),
                    axis_size,
                    errors::InvalidArgument(
                        "Axis values must be ranging from 0 to (dims - 1)."));

  std::vector<int64_t> new_x(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    new_x[i] = x[axis[i]];
  }
  return new_x;
}

std::vector<int64_t> GetInputStrides(const std::string input_name,
                                     const DDim& input_dims,
                                     const bool transpose_input,
                                     std::vector<int> shape,
                                     std::vector<int> axis) {
  auto new_dims = input_dims;
  if (!shape.empty() && !axis.empty()) {
    new_dims = input_dims.reshape(shape).transpose(axis);
  }

  auto& MatrixDimsFromVector =
      input_name == "X" ? RowMatrixDimsFromVector : ColumnMatrixDimsFromVector;
  MatDescriptor mat_dim = CreateMatrixDescriptor(
      MatrixDimsFromVector(new_dims), 0, transpose_input);

  std::vector<int64_t> strides;
  if (!shape.empty()) {
    auto shape2 = input_dims.reshape(shape);
    strides.push_back(1);
    for (auto i = shape2.size() - 1; i > 0; --i) {
      strides.insert(strides.begin(),
                     strides.front() * static_cast<int64_t>(shape2[i]));
    }
    strides = TransposeAxis(strides, axis);
    if (shape.size() == 2)
      strides.insert(strides.begin(),
                     static_cast<int64_t>(shape[0] * shape[1]));
    mat_dim.stride_ = strides[0];
    if (mat_dim.trans_) std::swap(*strides.rbegin(), *(++strides.rbegin()));
  }
  return strides;
}

}  // namespace phi::funcs
