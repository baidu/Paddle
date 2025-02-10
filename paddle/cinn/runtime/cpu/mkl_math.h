// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * This file defines some math extern functions those can be called in CINN IR.
 */
#pragma once

#include "paddle/cinn/runtime/cinn_runtime.h"

extern "C" {

//! 1 buffer as input, and 1 buffer as output
// @{
#define CINN_DECL_MKL_VECTOR_MATH_FP(fn__)                             \
  void cinn_mkl_##fn__##_v_fp32(cinn_buffer_t* x, cinn_buffer_t* out); \
  void cinn_mkl_##fn__##_v_fp64(cinn_buffer_t* x, cinn_buffer_t* out);

CINN_DECL_MKL_VECTOR_MATH_FP(exp);
CINN_DECL_MKL_VECTOR_MATH_FP(erf);
CINN_DECL_MKL_VECTOR_MATH_FP(sqrt);
CINN_DECL_MKL_VECTOR_MATH_FP(log);
CINN_DECL_MKL_VECTOR_MATH_FP(log2);
CINN_DECL_MKL_VECTOR_MATH_FP(log10);
CINN_DECL_MKL_VECTOR_MATH_FP(floor);
CINN_DECL_MKL_VECTOR_MATH_FP(ceil);
CINN_DECL_MKL_VECTOR_MATH_FP(round);
CINN_DECL_MKL_VECTOR_MATH_FP(trunc);
CINN_DECL_MKL_VECTOR_MATH_FP(cos);
CINN_DECL_MKL_VECTOR_MATH_FP(sin);
CINN_DECL_MKL_VECTOR_MATH_FP(cosh);
CINN_DECL_MKL_VECTOR_MATH_FP(tan);
CINN_DECL_MKL_VECTOR_MATH_FP(tanh);
CINN_DECL_MKL_VECTOR_MATH_FP(sinh);
CINN_DECL_MKL_VECTOR_MATH_FP(acos);
CINN_DECL_MKL_VECTOR_MATH_FP(acosh);
CINN_DECL_MKL_VECTOR_MATH_FP(asin);
CINN_DECL_MKL_VECTOR_MATH_FP(asinh);
CINN_DECL_MKL_VECTOR_MATH_FP(atan);
CINN_DECL_MKL_VECTOR_MATH_FP(atanh);
// @}
}
