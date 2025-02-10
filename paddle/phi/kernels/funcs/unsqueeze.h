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

#include "paddle/common/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

// TODO(paddle-dev): Remove this file when we can call related Kernel directly

namespace phi {
namespace funcs {
inline DDim GetOutputSqueezeShape(const std::vector<int> squeeze_dims,
                                  const DDim& in_dims,
                                  bool is_runtime) {
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims.size(), false);

  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (int i = 0; i < in_dims.size(); ++i) {
      if (in_dims[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      if (in_dims.size() == 0) {
        PADDLE_ENFORCE_GE(
            squeeze_dims[i],
            -1,
            common::errors::InvalidArgument(
                "For 0D Tensor, Each axis in Attr(axes) should be in the range "
                "of [-1, 0]"
                "But current axis is:%d, input tensor's shape = [%s]."));
        PADDLE_ENFORCE_LE(
            squeeze_dims[i],
            0,
            common::errors::InvalidArgument(
                "For 0D Tensor, Each axis in Attr(axes) should be in the range "
                "of [-1, 0]"
                "But current axis is:%d, input tensor's shape = [%s]."));
        continue;
      }

      int current = squeeze_dims[i] < 0 ? squeeze_dims[i] + in_dims.size()
                                        : squeeze_dims[i];

      PADDLE_ENFORCE_GE(
          current,
          0,
          common::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(),
              in_dims.size() - 1,
              current,
              in_dims));
      PADDLE_ENFORCE_LT(
          current,
          in_dims.size(),
          common::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(),
              in_dims.size() - 1,
              current,
              in_dims));

      if (!should_squeeze[current]) {
        if (is_runtime) {
          // At run time, dim of 1 is allowed to squeeze
          if (in_dims[current] == 1) {
            should_squeeze[current] = true;
          }
        } else {
          // At compile time, dim of -1 or 1 is allowed to squeeze
          if (in_dims[current] == 1 || in_dims[current] == -1) {
            should_squeeze[current] = true;
          }
        }
      }
    }
  }
  // Make output dimensions
  std::vector<int64_t> output_shape;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape.push_back(in_dims[i]);
    }
  }
  return common::make_ddim(output_shape);
}

inline DDim GetUnsqueezeShape(const std::vector<int64_t> unsqz_dims,
                              const DDim& in_dims) {
#define UNSQUEEZE_MAX_RANK_SUPPORTED 8
  int output_rank = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_rank = in_dims.size();
  std::vector<int64_t> output_shape(output_rank, 0);
  // Validity Check: rank range.
  PADDLE_ENFORCE_LE(
      output_rank,
      UNSQUEEZE_MAX_RANK_SUPPORTED,
      common::errors::InvalidArgument("The output "
                                      "tensor's rank should be less than %d.",
                                      UNSQUEEZE_MAX_RANK_SUPPORTED));

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_rank + 1 : axis;
    // Validity Check: the axis bound
    PADDLE_ENFORCE_GE(
        cur,
        0,
        common::errors::InvalidArgument("The insert dimension value should "
                                        "not be less than 0"));
    PADDLE_ENFORCE_LE(cur,
                      cur_output_rank,
                      common::errors::InvalidArgument(
                          "The insert dimension value should not be larger "
                          "than the dimension size of input tensor"));
    // Move old axis, and insert new axis
    for (int i = cur_output_rank; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }
    output_shape[cur] = 1;
    // Add the output size.
    cur_output_rank++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_rank; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }
#undef UNSQUEEZE_MAX_RANK_SUPPORTED
  return common::make_ddim(output_shape);
}

inline const DenseTensor Unsqueeze(const DenseTensor& x, int axis = 0) {
  // don't copy data, only change the dims
  DenseTensor out(x);
  std::vector<int> out_shape = common::vectorize<int>(x.dims());
  if (axis >= 0) {
    auto index = (out_shape.begin() + axis);
    out_shape.insert(index, 1);
  } else if (axis < 0) {
    auto index = (out_shape.end() + axis + 1);
    out_shape.insert(index, 1);
  }
  out.Resize(common::make_ddim(out_shape));
  return out;
}

}  // namespace funcs
}  // namespace phi
