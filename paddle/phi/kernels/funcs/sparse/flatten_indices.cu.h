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

#include "paddle/phi/kernels/funcs/sparse/flatten_indices.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename IntT>
__global__ void FlattenIndicesKernel(const IntT* indices,
                                     const IntT* sparse_offsets,
                                     const int64_t non_zero_num,
                                     const int64_t sparse_dim,
                                     IntT* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  phi::funcs::sparse::FlattenIndices<IntT>(indices,
                                           sparse_offsets,
                                           non_zero_num,
                                           sparse_dim,
                                           tid,
                                           gridDim.x * blockDim.x,
                                           out);
}

template <typename IntT>
__global__ void IndexToCoordinateKernel(const IntT* index,
                                        const Dim<DDim::kMaxRank> dims,
                                        const int64_t non_zero_num,
                                        const int64_t sparse_dim,
                                        IntT* indices) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  IndexToCoordinate(index,
                    dims,
                    non_zero_num,
                    sparse_dim,
                    tid,
                    gridDim.x * blockDim.x,
                    indices);
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
