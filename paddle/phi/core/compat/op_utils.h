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

#include <string>
#include <unordered_set>

#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/flat_hash_map.h"

namespace phi {

const std::unordered_set<std::string> standard_kernel_suffixs({
    "sr",  // SelectedRows kernel
    "raw"  // fallback kernel of origfinal fluid op
});

/**
 * Some fluid ops are no longer used under the corresponding official API
 * system of 2.0. These names need to correspond to the official API names
 * after 2.0, and can no longer be occupied by the previously abandoned ops.
 * They are marked here uniformly.
 */
const std::unordered_set<std::string> deprecated_op_names({"diag",
                                                           "flatten",
                                                           "flatten_grad",
                                                           "isinf",
                                                           "isnan",
                                                           "isfinite",
                                                           "matmul",
                                                           "matmul_grad",
                                                           "matmul_grad_grad",
                                                           "mean",
                                                           "max",
                                                           "min",
                                                           "any",
                                                           "all",
                                                           "reshape",
                                                           "reshape_grad",
                                                           "expand",
                                                           "expand_as",
                                                           "expand_grad",
                                                           "expand_as_grad",
                                                           "sum",
                                                           "sum_grad",
                                                           "top_k",
                                                           "top_k_grad"});

class DefaultKernelSignatureMap {
 public:
  static DefaultKernelSignatureMap& Instance();

  bool Has(const std::string& op_type) const { return map_.count(op_type) > 0; }

  const KernelSignature& Get(const std::string& op_type) const {
    auto it = map_.find(op_type);
    PADDLE_ENFORCE_NE(
        it,
        map_.end(),
        phi::errors::NotFound(
            "Operator `%s`'s kernel signature is not registered.", op_type));
    return it->second;
  }

  void Insert(std::string op_type, KernelSignature signature) {
    PADDLE_ENFORCE_NE(
        Has(op_type),
        true,
        phi::errors::AlreadyExists(
            "Operator (%s)'s Kernel Siginature has been registered.", op_type));
    map_.insert({std::move(op_type), std::move(signature)});
  }

 private:
  DefaultKernelSignatureMap() = default;

  paddle::flat_hash_map<std::string, KernelSignature> map_;

  DISABLE_COPY_AND_ASSIGN(DefaultKernelSignatureMap);
};

class OpUtilsMap {
 public:
  static OpUtilsMap& Instance();

  bool Contains(const std::string& op_type) const {
    return base_kernel_name_map_.count(op_type) ||
           arg_mapping_fn_map_.count(op_type);
  }

  void InsertBaseKernelName(std::string op_type, std::string base_kernel_name) {
    PADDLE_ENFORCE_EQ(
        base_kernel_name_map_.count(op_type),
        0UL,
        phi::errors::AlreadyExists(
            "Operator (%s)'s api name has been registered.", op_type));
    base_kernel_name_map_.insert(
        {std::move(op_type), std::move(base_kernel_name)});
  }

  void InsertArgumentMappingFn(std::string op_type, ArgumentMappingFn fn) {
    PADDLE_ENFORCE_EQ(
        arg_mapping_fn_map_.count(op_type),
        0UL,
        phi::errors::AlreadyExists(
            "Operator (%s)'s argu,emt mapping function has been registered.",
            op_type));
    arg_mapping_fn_map_.insert({std::move(op_type), std::move(fn)});
  }

  std::string GetBaseKernelName(const std::string& op_type) const {
    if (deprecated_op_names.find(op_type) != deprecated_op_names.end()) {
      return "deprecated";
    }
    auto it = base_kernel_name_map_.find(op_type);
    if (it == base_kernel_name_map_.end()) {
      return op_type;
    } else {
      return it->second;
    }
  }

  ArgumentMappingFn GetArgumentMappingFn(const std::string& op_type) const {
    auto it = arg_mapping_fn_map_.find(op_type);
    if (it == arg_mapping_fn_map_.end()) {
      auto func =
          [op_type](const ArgumentMappingContext& ctx) -> KernelSignature {
        return DefaultKernelSignatureMap::Instance().Get(op_type);
      };
      return func;
    } else {
      return it->second;
    }
  }

  const paddle::flat_hash_map<std::string, std::string>& base_kernel_name_map()
      const {
    return base_kernel_name_map_;
  }

 private:
  OpUtilsMap() = default;

  paddle::flat_hash_map<std::string, std::string> base_kernel_name_map_;
  paddle::flat_hash_map<std::string, ArgumentMappingFn> arg_mapping_fn_map_;

  DISABLE_COPY_AND_ASSIGN(OpUtilsMap);
};

struct BaseKernelNameRegistrar {
  BaseKernelNameRegistrar(const char* op_type, const char* base_kernel_name) {
    OpUtilsMap::Instance().InsertBaseKernelName(op_type, base_kernel_name);
  }
};

struct ArgumentMappingFnRegistrar {
  ArgumentMappingFnRegistrar(const char* op_type,
                             ArgumentMappingFn arg_mapping_fn) {
    OpUtilsMap::Instance().InsertArgumentMappingFn(op_type,
                                                   std::move(arg_mapping_fn));
  }
};

#define PD_REGISTER_BASE_KERNEL_NAME(op_type, base_kernel_name)                \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      PD_REGISTER_base_kernel_name_ns_check_##op_type,                         \
      "PD_REGISTER_BASE_KERNEL_NAME must be called in global namespace.");     \
  static const ::phi::BaseKernelNameRegistrar                                  \
      __registrar_base_kernel_name_for_##op_type(#op_type, #base_kernel_name); \
  int TouchBaseKernelNameSymbol_##op_type() { return 0; }

#define PD_DECLARE_BASE_KERNEL_NAME(op_type)                              \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      PD_DECLARE_ai_name_ns_check_##op_type,                              \
      "PD_DECLARE_BASE_KERNEL_NAME must be called in global namespace."); \
  extern int TouchBaseKernelNameSymbol_##op_type();                       \
  UNUSED static int __declare_base_kernel_name_symbol_for_##op_type =     \
      TouchBaseKernelNameSymbol_##op_type()

#define PD_REGISTER_ARG_MAPPING_FN(op_type, arg_mapping_fn)              \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                     \
      PD_REGISTER_arg_map_fn_ns_check_##op_type,                         \
      "PD_REGISTER_ARG_MAPPING_FN must be called in global namespace."); \
  static const ::phi::ArgumentMappingFnRegistrar                         \
      __registrar_arg_map_fn_for_##op_type(#op_type, arg_mapping_fn);    \
  int TouchArgumentMappingFnSymbol_##op_type() { return 0; }

#define PD_DECLARE_ARG_MAPPING_FN(op_type)                              \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                    \
      PD_DECLARE_arg_map_fn_ns_check_##op_type,                         \
      "PD_DECLARE_ARG_MAPPING_FN must be called in global namespace."); \
  extern int TouchArgumentMappingFnSymbol_##op_type();                  \
  UNUSED static int __declare_arg_map_fn_symbol_for_##op_type =         \
      TouchArgumentMappingFnSymbol_##op_type()

}  // namespace phi
