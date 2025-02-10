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
#include <map>

#include "fp8_fp8_gemm_scale_bias_act.h"  // NOLINT

#include "gemm_scale.h"            // NOLINT
#include "gemm_scale_bias.h"       // NOLINT
#include "gemm_scale_bias_gelu.h"  // NOLINT
#include "gemm_scale_bias_relu.h"  // NOLINT
#include "gemm_scale_gelu.h"       // NOLINT

namespace phi {
namespace fusion {
namespace cutlass_internal {

std::map<std::string, int> config_map{
    {"e4m3_bf16_identity", 0},       {"e4m3_bf16_bias_identity", 1},
    {"e4m3_bf16_bias_relu", 2},      {"e4m3_bf16_bias_gelu", 3},
    {"e4m3_bf16_gelu", 4},           {"e4m3_fp16_identity", 5},
    {"e4m3_fp16_bias_identity", 6},  {"e4m3_fp16_bias_relu", 7},
    {"e4m3_fp16_bias_gelu", 8},      {"e4m3_fp16_gelu", 9},
    {"e5m2_bf16_identity", 10},      {"e5m2_bf16_bias_identity", 11},
    {"e5m2_bf16_bias_relu", 12},     {"e5m2_bf16_bias_gelu", 13},
    {"e5m2_bf16_gelu", 14},          {"e5m2_fp16_identity", 15},
    {"e5m2_fp16_bias_identity", 16}, {"e5m2_fp16_bias_relu", 17},
    {"e5m2_fp16_bias_gelu", 18},     {"e5m2_fp16_gelu", 19},
};

bool fp8_fp8_gemm_scale_bias_act(GemmEpilogueAllParams params) {
  switch (config_map[params.gemm_config]) {
    case 0:
      dispatch_gemm_scale<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(
          params);
      break;
    case 1:
      dispatch_gemm_scale_bias<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(
          params);
      break;
    case 2:
      dispatch_gemm_scale_bias_relu<phi::dtype::float8_e4m3fn,
                                    phi::dtype::bfloat16>(params);
      break;
    case 3:
      dispatch_gemm_scale_bias_gelu<phi::dtype::float8_e4m3fn,
                                    phi::dtype::bfloat16>(params);
      break;
    case 4:
      dispatch_gemm_scale_gelu<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(
          params);
      break;
    case 5:
      dispatch_gemm_scale<phi::dtype::float8_e4m3fn, phi::dtype::float16>(
          params);
      break;
    case 6:
      dispatch_gemm_scale_bias<phi::dtype::float8_e4m3fn, phi::dtype::float16>(
          params);
      break;
    case 7:
      dispatch_gemm_scale_bias_relu<phi::dtype::float8_e4m3fn,
                                    phi::dtype::float16>(params);
      break;
    case 8:
      dispatch_gemm_scale_bias_gelu<phi::dtype::float8_e4m3fn,
                                    phi::dtype::float16>(params);
      break;
    case 9:
      dispatch_gemm_scale_gelu<phi::dtype::float8_e4m3fn, phi::dtype::float16>(
          params);
      break;
    case 10:
      dispatch_gemm_scale<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(
          params);
      break;
    case 11:
      dispatch_gemm_scale_bias<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(
          params);
      break;
    case 12:
      dispatch_gemm_scale_bias_relu<phi::dtype::float8_e5m2,
                                    phi::dtype::bfloat16>(params);
      break;
    case 13:
      dispatch_gemm_scale_bias_gelu<phi::dtype::float8_e5m2,
                                    phi::dtype::bfloat16>(params);
      break;
    case 14:
      dispatch_gemm_scale_gelu<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(
          params);
      break;
    case 15:
      dispatch_gemm_scale<phi::dtype::float8_e5m2, phi::dtype::float16>(params);
      break;
    case 16:
      dispatch_gemm_scale_bias<phi::dtype::float8_e5m2, phi::dtype::float16>(
          params);
      break;
    case 17:
      dispatch_gemm_scale_bias_relu<phi::dtype::float8_e5m2,
                                    phi::dtype::float16>(params);
      break;
    case 18:
      dispatch_gemm_scale_bias_gelu<phi::dtype::float8_e5m2,
                                    phi::dtype::float16>(params);
      break;
    case 19:
      dispatch_gemm_scale_gelu<phi::dtype::float8_e5m2, phi::dtype::float16>(
          params);
      break;
    default:
      throw std::runtime_error("fp8_fp8_bf16_gemm_fused Config is invalid.");
      break;
  }
  return false;
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
