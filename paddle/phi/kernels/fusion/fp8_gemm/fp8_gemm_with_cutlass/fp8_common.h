// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include "cuda.h"  // NOLINT

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef struct {
  const void *A;
  const void *B;
  void *D;
  float scale = 1.0;
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldd;
  int batch_count = 1;
  const phi::GPUPlace &place;
  cudaStream_t stream;
  int sm_version = 80;
  float leaky_alpha = 1.0;
  const void *bias;
  std::vector<int64_t> &bias_dims;
  std::string &gemm_config;
} GemmEpilogueAllParams;

typedef bool (*func)(GemmEpilogueAllParams);

typedef struct {
  const void *A;
  const void *B0;
  const void *B1;
  void *D;
  float scale0 = 1.0;
  float scale1 = 1.0;
  float scale_out = 1.0;
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldd;
  int batch_count = 1;
  const phi::GPUPlace &place;
  cudaStream_t stream;
  int sm_version = 80;
  const void *bias0;
  const void *bias1;
  std::vector<int64_t> &bias_dims0;
  std::vector<int64_t> &bias_dims1;
  std::string &gemm_config;
} DualGemmEpilogueAllParams;

typedef bool (*func1)(DualGemmEpilogueAllParams);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
