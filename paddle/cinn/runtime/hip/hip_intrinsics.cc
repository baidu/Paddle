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
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
using cinn::backends::GlobalSymbolRegistry;
#include "paddle/cinn/runtime/hip/hip_backend_api.h"
using cinn::runtime::hip::HIPBackendAPI;
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/runtime/hip/hip_util.h"
using cinn::runtime::hip::cinn_call_hip_kernel;

CINN_REGISTER_HELPER(cinn_hip_host_api) {
  GlobalSymbolRegistry::Global().RegisterFn(
      "backend_api.hip", reinterpret_cast<void *>(HIPBackendAPI::Global()));

  using cinn::runtime::hip::cinn_get_value_in_hip_kernel_args;
  REGISTER_EXTERN_FUNC_HELPER(cinn_get_value_in_hip_kernel_args,
                              cinn::common::DefaultHostTarget())
      .SetRetType<int64_t>()
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // index
      .End();

  using cinn::runtime::hip::cinn_get_item_in_hip_kernel_args;
  REGISTER_EXTERN_FUNC_HELPER(cinn_get_item_in_hip_kernel_args,
                              cinn::common::DefaultHostTarget())
      .SetRetType<void *>()
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // index
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_call_hip_kernel,
                              cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // kernel_fn
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // grid_x
      .AddInputType<int>()     // grid_y
      .AddInputType<int>()     // grid_z
      .AddInputType<int>()     // block_x
      .AddInputType<int>()     // block_y
      .AddInputType<int>()     // block_z
      .AddInputType<int>()     // shared_memory_bytes
      .AddInputType<void *>()  // stream
      .End();
  using cinn::runtime::hip::infer_shape_set_value;

  REGISTER_EXTERN_FUNC_HELPER(infer_shape_set_value,
                              cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int64_t>()
      .AddInputType<int64_t **>()
      .End();
  return true;
}

CINN_REGISTER_HELPER(hip_intrinsics) {
  auto target = cinn::common::DefaultHygonDcuHipTarget();

// bool for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(            \
      cinn_hip_##func__##_bool, target, bool, bool)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_BOOL

// bool for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(            \
      cinn_hip_##func__##_bool, target, bool, bool, bool)

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL(bitwise_xor);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_BOOL

// uint8 for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_UINT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(             \
      cinn_hip_##func__##_uint8, target, uint8_t, uint8_t)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_UINT8(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_UINT8

// uint8 for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(             \
      cinn_hip_##func__##_uint8, target, uint8_t, uint8_t, uint8_t);

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(bitwise_xor);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8(logical_right_shift);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_UINT8

// int8 for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(            \
      cinn_hip_##func__##_int8, target, int8_t, int8_t)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT8(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT8

// int8 for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(            \
      cinn_hip_##func__##_int8, target, int8_t, int8_t, int8_t);

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(bitwise_xor);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8(logical_right_shift);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT8

// int16 for 1 input 1 output
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(             \
      cinn_hip_##func__##_int16, target, int16_t, int16_t)

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT16(bitwise_not);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_INT16

// int16 for 2 input 1 output
#define REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(             \
      cinn_hip_##func__##_int16, target, int16_t, int16_t, int16_t);

  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(bitwise_and);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(bitwise_or);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(bitwise_xor);
  REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16(logical_right_shift);

#undef REGISTER_EXTERN_FUNC_2_IN_1_OUT_INT16

// float
#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(             \
      cinn_hip_##func__##_fp32, target, float, float);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(abs);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(exp);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(erf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(rsqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log2);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(log10);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(floor);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(ceil);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(round);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(trunc);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cos);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cosh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sin);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sinh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(acos);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(acosh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(asin);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(asinh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(atan);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(atanh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(tanh);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(cbrt);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT(sigmoid);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FLOAT

#define REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                  \
      cinn_hip_##func__##_fp32, target, float, bool);

  REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(isnan);
  REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(isfinite);
  REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL(isinf);

#undef REGISTER_EXTERN_FUNC_1_IN_FLOAT_1_OUT_BOOL

#define REGISTER_EXTERN_FUNC_2_IN_1_FLOAT(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
      cinn_hip_##func__##_fp32, target, float, float, float);

  REGISTER_EXTERN_FUNC_2_IN_1_FLOAT(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_FLOAT(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FLOAT

  // double

#define REGISTER_EXTERN_FUNC_1_IN_1_FP64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(        \
      cinn_hip_##func__##_fp64, target, double, double);

  REGISTER_EXTERN_FUNC_1_IN_1_FP64(abs);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(exp);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(erf);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(rsqrt);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(log);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(log2);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(log10);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(floor);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(ceil);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(round);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(trunc);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(cos);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(cosh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(tan);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sin);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sinh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(acos);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(acosh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(asin);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(asinh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(atan);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(atanh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(tanh);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(cbrt);
  REGISTER_EXTERN_FUNC_1_IN_1_FP64(sigmoid);

#undef REGISTER_EXTERN_FUNC_1_IN_1_FP64

#define REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                 \
      cinn_hip_##func__##_fp64, target, double, bool);

  REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(isnan);
  REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(isfinite);
  REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL(isinf);

#undef REGISTER_EXTERN_FUNC_1_IN_FP64_1_OUT_BOOL

#define REGISTER_EXTERN_FUNC_2_IN_1_FP64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(        \
      cinn_hip_##func__##_fp64, target, double, double, double);

  REGISTER_EXTERN_FUNC_2_IN_1_FP64(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_FP64(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP64

  // int32

#define REGISTER_EXTERN_FUNC_1_IN_1_INT32(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(         \
      cinn_hip_##func__##_int32, target, int, int);

  REGISTER_EXTERN_FUNC_1_IN_1_INT32(bitwise_not)
  REGISTER_EXTERN_FUNC_1_IN_1_INT32(clz)
  REGISTER_EXTERN_FUNC_1_IN_1_INT32(popc)
  REGISTER_EXTERN_FUNC_1_IN_1_INT32(trunc)

#undef REGISTER_EXTERN_FUNC_1_IN_1_INT32

#define REGISTER_EXTERN_FUNC_1_IN_1_INT64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(         \
      cinn_hip_##func__##_int64, target, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_1_IN_1_INT64(bitwise_not)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(clz)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(popc)
  REGISTER_EXTERN_FUNC_1_IN_1_INT64(trunc)

#undef REGISTER_EXTERN_FUNC_1_IN_1_INT64

#define REGISTER_EXTERN_FUNC_2_IN_1_INT32(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
      cinn_hip_##func__##_int32, target, int, int, int);

  REGISTER_EXTERN_FUNC_2_IN_1_INT32(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(left_shift)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(right_shift)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(bitwise_and)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(bitwise_or)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(bitwise_xor)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(logical_right_shift)
  REGISTER_EXTERN_FUNC_2_IN_1_INT32(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT32

#define REGISTER_EXTERN_FUNC_2_IN_1_INT64(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(         \
      cinn_hip_##func__##_int64, target, int64_t, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_2_IN_1_INT64(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(bitwise_and)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(bitwise_or)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(bitwise_xor)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(mod)
  REGISTER_EXTERN_FUNC_2_IN_1_INT64(logical_right_shift)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT64

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_find_int, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_find_float, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<float>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_find_int_nd, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_find_float_nd, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<float>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_find_int_from, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_find_float_from, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<float>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_next_smallest_int32, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

#define _REGISTER_CINN_NVGPU_LT_NUM(TYPE_SUFFIX, TYPE)                      \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_lt_num_##TYPE_SUFFIX, target) \
      .SetRetType<int>()                                                    \
      .AddInputType<cinn_buffer_t *>()                                      \
      .AddInputType<int>()                                                  \
      .AddInputType<TYPE>()                                                 \
      .AddInputType<int>()                                                  \
      .AddInputType<int>()                                                  \
      .End();

  _REGISTER_CINN_NVGPU_LT_NUM(fp32, float);
  _REGISTER_CINN_NVGPU_LT_NUM(fp64, double);
  _REGISTER_CINN_NVGPU_LT_NUM(uint8, uint8_t);
  _REGISTER_CINN_NVGPU_LT_NUM(int16, int16_t);

  _REGISTER_CINN_NVGPU_LT_NUM(int32, int);
  _REGISTER_CINN_NVGPU_LT_NUM(int64, int64_t);

#undef _REGISTER_CINN_NVGPU_LT_NUM

#define _REGISTER_CINN_NVGPU_GT_NUM(TYPE_SUFFIX, TYPE)                      \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_gt_num_##TYPE_SUFFIX, target) \
      .SetRetType<int>()                                                    \
      .AddInputType<cinn_buffer_t *>()                                      \
      .AddInputType<int>()                                                  \
      .AddInputType<TYPE>()                                                 \
      .AddInputType<int>()                                                  \
      .AddInputType<int>()                                                  \
      .End();

  _REGISTER_CINN_NVGPU_GT_NUM(fp32, float);
  _REGISTER_CINN_NVGPU_GT_NUM(fp64, double);
  _REGISTER_CINN_NVGPU_GT_NUM(uint8, uint8_t);
  _REGISTER_CINN_NVGPU_GT_NUM(int16, int16_t);
  _REGISTER_CINN_NVGPU_GT_NUM(int32, int);
  _REGISTER_CINN_NVGPU_GT_NUM(int64, int64_t);

#undef _REGISTER_CINN_NVGPU_GT_NUM

#define _REGISTER_CINN_NVGPU_INDEX_ADD(TYPE_SUFFIX, TYPE)                      \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_index_add_##TYPE_SUFFIX, target) \
      .SetRetType<TYPE>()                                                      \
      .AddInputType<TYPE>()                                                    \
      .AddInputType<int>()                                                     \
      .AddInputType<cinn_buffer_t *>()                                         \
      .AddInputType<int>()                                                     \
      .AddInputType<int>()                                                     \
      .AddInputType<cinn_buffer_t *>()                                         \
      .AddInputType<int>()                                                     \
      .End();

  _REGISTER_CINN_NVGPU_INDEX_ADD(bool, bool);
  _REGISTER_CINN_NVGPU_INDEX_ADD(int8, int8_t);
  _REGISTER_CINN_NVGPU_INDEX_ADD(int32, int32_t);
  _REGISTER_CINN_NVGPU_INDEX_ADD(int64, int64_t);
  _REGISTER_CINN_NVGPU_INDEX_ADD(fp32, float);
  _REGISTER_CINN_NVGPU_INDEX_ADD(fp64, double);

#undef _REGISTER_CINN_NVGPU_INDEX_ADD

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_resize_bilinear, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_hip_resize_bicubic, target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  return true;
}
