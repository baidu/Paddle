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

#pragma once
#include <string>
#include <vector>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/intrinsic_types.h"

/**
 * \file This file implements some runtime concepts used in analysis and
 * codegen.
 */
namespace cinn {

namespace ir {
class Expr;
}  // namespace ir

namespace runtime {

namespace intrinsic {

//! cinn_buffer_t::new_(buffer)
static const char* buffer_create = "cinn_buffer_t::new_";
//! cinn_buffer_t::delete_(buffer)
static const char* buffer_destroy = "cinn_buffer_t::delete_";

static const char* buffer_load = "cinn_buffer_load";

static const char* buffer_malloc = "cinn_buffer_malloc";
static const char* buffer_free = "cinn_buffer_free";
static const char* buffer_create_default = "cinn_buffer_new_default";

static const char* buffer_get_data_handle = "cinn_buffer_get_data_handle";
static const char* buffer_get_data_const_handle =
    "cinn_buffer_get_data_const_handle";

//! Buffer load an element of some primitive type
// @{
static const char* buffer_load_bfloat16 = "buffer_load_bfloat16";
static const char* buffer_load_float16 = "buffer_load_float16";
static const char* buffer_load_float32 = "buffer_load_float32";
static const char* buffer_load_float64 = "buffer_load_float64";
// @}

static const char* pod_value_ty = "cinn_pod_value_t";

static const char* float_to_cinn_pod_value_repr = "float_to_cinn_pod_value";
static const char* double_to_cinn_pod_value_repr = "double_to_cinn_pod_value";
static const char* bfloat16_to_cinn_pod_value_repr =
    "bfloat16_to_cinn_pod_value";
static const char* float16_to_cinn_pod_value_repr = "float16_to_cinn_pod_value";

static const char* bool_to_cinn_pod_value_repr = "bool_to_cinn_pod_value";

static const char* int8_to_cinn_pod_value_repr = "int8_to_cinn_pod_value";
static const char* int16_to_cinn_pod_value_repr = "int16_to_cinn_pod_value";
static const char* int32_to_cinn_pod_value_repr = "int32_to_cinn_pod_value";
static const char* int64_to_cinn_pod_value_repr = "int64_to_cinn_pod_value";

static const char* uint8_to_cinn_pod_value_repr = "uint8_to_cinn_pod_value";
static const char* uint16_to_cinn_pod_value_repr = "uint16_to_cinn_pod_value";
static const char* uint32_to_cinn_pod_value_repr = "uint32_to_cinn_pod_value";
static const char* uint64_to_cinn_pod_value_repr = "uint64_to_cinn_pod_value";

static const char* buffer_p_to_cinn_pod_value_repr =
    "buffer_p_to_cinn_pod_value";

static const char* pod_value_to_buffer_p = "cinn_pod_value_to_buffer_p";
static const char* pod_value_to_bool = "cinn_pod_value_to_bool";

static const char* pod_value_to_int8 = "cinn_pod_value_to_int8";
static const char* pod_value_to_int16 = "cinn_pod_value_to_int16";
static const char* pod_value_to_int32 = "cinn_pod_value_to_int32";
static const char* pod_value_to_int64 = "cinn_pod_value_to_int64";

static const char* pod_value_to_uint8 = "cinn_pod_value_to_uint8";
static const char* pod_value_to_uint16 = "cinn_pod_value_to_uint16";
static const char* pod_value_to_uint32 = "cinn_pod_value_to_uint32";
static const char* pod_value_to_uint64 = "cinn_pod_value_to_uint64";

static const char* pod_value_to_float = "cinn_pod_value_to_float";
static const char* pod_value_to_double = "cinn_pod_value_to_double";
static const char* pod_value_to_bfloat16 = "cinn_pod_value_to_bfloat16";
static const char* pod_value_to_float16 = "cinn_pod_value_to_float16";

static const char* pod_value_to_void_p = "cinn_pod_value_to_void_p";

static const char* print_debug_args_repr = "cinn_print_debug_args";

static const char* call_cuda_kernel = "cinn_call_cuda_kernel";

static const char* call_hip_kernel = "cinn_call_hip_kernel";

static const char* call_sycl_kernel = "cinn_call_sycl_kernel";

static const char* call_cuda_memset = "cinn_call_cuda_memset";

static const char* get_value_in_cuda_kernel_args =
    "cinn_get_value_in_cuda_kernel_args";

static const char* get_value_in_hip_kernel_args =
    "cinn_get_value_in_hip_kernel_args";

static const char* get_item_in_cuda_kernel_args =
    "cinn_get_item_in_cuda_kernel_args";

static const char* get_item_in_hip_kernel_args =
    "cinn_get_item_in_hip_kernel_args";

static const char* infer_shape_set_value = "infer_shape_set_value";

static const char* pod_values_to_array_repr = "pod_values_to_array";

static const char* get_address_repr = "get_address";

static const char* args_construct_repr = "cinn_args_construct";

static const char* builtin_intrin_repr = "cinn_builtin_intrin";

//! Name of the helper intrinsic used to display debug string.
static const char* debug_log_repr = "cinn_print_debug_string";

static const char* cuda_sync_threads = "__syncthreads";

static const char* cuda_builtin_assume = "__builtin_assume";

static const char* parallel_launch = "cinn_backend_parallel_launch";

}  // namespace intrinsic

/**
 * Call an intrinsic function.
 * @param type Return type of the function.
 * @param fn_name Name of the function.
 * @param args The arguments for the function.
 * @return The Call node.
 */
Expr IntrinsicCall(Type type,
                   const std::string& fn_name,
                   const std::vector<Expr>& args,
                   const std::vector<Expr>& write_args = {});

//! Convert the Type in compile time to runtime type.
cinn_type_t ToRuntimeType(Type type);

}  // namespace runtime
}  // namespace cinn
