/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <stdint.h>

#include "paddle/common/ddim.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename IntT>
inline const IntT HOSTDEVICE CoordinateToIndex(const IntT* indices,
                                               const IntT* sparse_offsets,
                                               const int64_t non_zero_num,
                                               const int64_t sparse_dim,
                                               const int i) {
  IntT index = 0;
  for (IntT j = 0; j < sparse_dim; j++) {
    index += indices[j * non_zero_num + i] * sparse_offsets[j];
  }
  return index;
}

template <typename IntT>
inline void HOSTDEVICE FlattenIndices(const IntT* indices,
                                      const IntT* sparse_offsets,
                                      const int64_t non_zero_num,
                                      const int64_t sparse_dim,
                                      const int64_t start,
                                      const int64_t stride,
                                      IntT* out) {
  for (int64_t i = start; i < non_zero_num; i += stride) {
    out[i] =
        CoordinateToIndex(indices, sparse_offsets, non_zero_num, sparse_dim, i);
  }
}

// 1. indices.dims().size() == 2
template <typename IntT>
inline void CalcOffsetsPerDim(const DDim& dims,
                              const int64_t sparse_dim,
                              IntT* offsets) {
  IntT offset = 1;
  for (IntT i = sparse_dim - 1; i >= 0; i--) {
    offsets[i] = offset;
    offset *= dims[i];
  }
}

template <typename IntT>
inline void HOSTDEVICE IndexToCoordinate(const IntT index,
                                         const Dim<DDim::kMaxRank>& dims,
                                         const int64_t non_zero_num,
                                         const int64_t sparse_dim,
                                         const int indices_offset,
                                         IntT* indices) {
  IntT tmp_index = index;
  for (int j = sparse_dim - 1; j >= 0; j--) {
    indices[j * non_zero_num + indices_offset] = tmp_index % dims[j];
    tmp_index /= dims[j];
  }
}

template <typename IntT>
inline void HOSTDEVICE IndexToCoordinate(const IntT* index,
                                         const Dim<DDim::kMaxRank>& dims,
                                         const int64_t non_zero_num,
                                         const int64_t sparse_dim,
                                         const int64_t start,
                                         const int64_t stride,
                                         IntT* indices) {
  for (int64_t i = start; i < non_zero_num; i += stride) {
    IntT tmp_index = index[i];
    IndexToCoordinate(tmp_index, dims, non_zero_num, sparse_dim, i, indices);
  }
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
