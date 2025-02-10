// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
template <typename T>
inline XPUFCCalcType GetConvCalcType() {
  std::unordered_map<const char*, XPUFCCalcType> calc_type_map = {
      {"XPU_PADDLE_CONV_FLOAT16", XPUFCCalcType::FC_FLOAT16},
      {"XPU_PADDLE_CONV_TF32", XPUFCCalcType::FC_TF32},
      {"XPU_PADDLE_CONV_FLOAT", XPUFCCalcType::FC_FLOAT},
      {"XPU_PADDLE_CONV_INT16", XPUFCCalcType::FC_INT16},
      {"XPU_PADDLE_CONV_INT32", XPUFCCalcType::FC_INT32},
      {"XPU_PADDLE_CONV_INT32_WITH_LL", XPUFCCalcType::FC_INT32_WITH_LL},
  };

  for (auto [env_str, calc_type] : calc_type_map) {
    if (std::getenv(env_str)) {
      return calc_type;
    }
  }

  return FCCalcType<T>();
}
using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
using XPUTypeBF16 = typename XPUTypeTrait<phi::dtype::bfloat16>::Type;
template <typename QuantType>
struct XPUDefaultQuantType {
  using Type = tfloat32;
};

template <>
struct XPUDefaultQuantType<XPUTypeFP16> {
  using Type = XPUTypeFP16;
};

template <typename XPUType, typename QuantType>
struct XPUConvQuantTypeTrait {
  using Type = QuantType;
};

#define DECLARE_XPU_UNSUPPORTED_QUANT_TYPE(TYPE, QUANT_TYPE) \
  template <>                                                \
  struct XPUConvQuantTypeTrait<TYPE, QUANT_TYPE> {           \
    using Type = typename XPUDefaultQuantType<TYPE>::Type;   \
  };

#define XPU_CONV_FOR_EACH_UNSUPPORTED_QUANT_TYPE(_) \
  _(float, XPUTypeFP16)                             \
  _(XPUTypeFP16, int)                               \
  _(XPUTypeFP16, int_with_ll_t)                     \
  _(XPUTypeFP16, tfloat32)                          \
  _(XPUTypeBF16, int16_t)                           \
  _(XPUTypeBF16, int)                               \
  _(XPUTypeBF16, int_with_ll_t)                     \
  _(XPUTypeBF16, XPUTypeFP16)                       \
  _(XPUTypeBF16, float)

XPU_CONV_FOR_EACH_UNSUPPORTED_QUANT_TYPE(DECLARE_XPU_UNSUPPORTED_QUANT_TYPE)

#ifndef PADDLE_WITH_XPU_XRE5
DECLARE_XPU_UNSUPPORTED_QUANT_TYPE(XPUTypeFP16, float)
DECLARE_XPU_UNSUPPORTED_QUANT_TYPE(XPUTypeFP16, XPUTypeFP16)
#endif

#define PD_PRIVATE_XPU_CONV_CASE(TYPE, calc_type, QUANT_TYPE, ...)        \
  case calc_type: {                                                       \
    using TGEMM = typename XPUConvQuantTypeTrait<TYPE, QUANT_TYPE>::Type; \
    __VA_ARGS__();                                                        \
    break;                                                                \
  }

#define PD_VISIT_XPU_CONV_TYPES(TYPE, calc_type, func_name, ...)             \
  do {                                                                       \
    switch (calc_type) {                                                     \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_FLOAT, float, __VA_ARGS__)                 \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_TF32, tfloat32, __VA_ARGS__)               \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_FLOAT16, XPUTypeFP16, __VA_ARGS__)         \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_INT16, int16_t, __VA_ARGS__)               \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_INT32, int, __VA_ARGS__)                   \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_INT32_WITH_LL, int_with_ll_t, __VA_ARGS__) \
      default:                                                               \
        PADDLE_THROW(common::errors::InvalidArgument(                        \
            "Function " #func_name " got invalid fc calc type %d",           \
            static_cast<int>(calc_type)));                                   \
    }                                                                        \
  } while (0)
}  // namespace phi
