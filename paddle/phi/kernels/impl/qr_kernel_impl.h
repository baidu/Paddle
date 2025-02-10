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

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/optional.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

namespace phi {

template <typename DeviceContext, typename T>
void BatchedGeqrf(const DeviceContext& dev_ctx,
                  int batch_size,
                  int m,
                  int n,
                  T* a,
                  int lda,
                  T* tau,
                  int a_stride,
                  int tau_stride);

template <typename DeviceContext, typename T>
void BatchedOrgqr(const DeviceContext& dev_ctx,
                  int batch_size,
                  int m,
                  int n,
                  int k,
                  T* a,
                  int lda,
                  T* tau,
                  int a_stride,
                  int tau_stride);

}  // namespace phi
