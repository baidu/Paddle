//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi::funcs {
MatDescriptor CreateMatrixDescriptor(const DDim &tensor_dim,
                                     int num_flatten_cols,
                                     bool trans) {
  PADDLE_ENFORCE_GT(
      tensor_dim.size(),
      1,
      common::errors::InvalidArgument("The tensor dim size should be greater "
                                      "than 1, but received dim size is %d",
                                      tensor_dim.size()));
  MatDescriptor retv;
  if (num_flatten_cols > 1) {
    auto flatten_dim = common::flatten_to_2d(tensor_dim, num_flatten_cols);
    retv.height_ = flatten_dim[0];
    retv.width_ = flatten_dim[1];
  } else {
    if (tensor_dim.size() == 2) {
      retv.height_ = tensor_dim[0];
      retv.width_ = tensor_dim[1];
    } else {
      auto dim_vec = common::vectorize(tensor_dim);
      retv.batch_size_ = 1;
      for (size_t i = 0; i < dim_vec.size() - 2; ++i) {
        retv.batch_size_ *= dim_vec[i];
      }
      retv.height_ = dim_vec[dim_vec.size() - 2];
      retv.width_ = dim_vec[dim_vec.size() - 1];
      retv.stride_ = retv.height_ * retv.width_;
    }
  }
  if (trans) {
    std::swap(retv.width_, retv.height_);
  }
  retv.trans_ = trans;
  return retv;
}
}  // namespace phi::funcs
