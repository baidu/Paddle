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

#include "../fp8_common.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

extern "C" bool fp8_fp8_dual_gemm_scale_bias_act(
    DualGemmEpilogueAllParams params);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
