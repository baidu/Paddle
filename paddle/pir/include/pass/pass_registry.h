// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <unordered_map>

#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/pir/include/pass/pass.h"

namespace pir {

using PassCreator = std::function<std::unique_ptr<Pass>()>;

class IR_API PassRegistry {
 public:
  static PassRegistry &Instance();

  bool Has(const std::string &pass_type) const {
    return pass_map_.find(pass_type) != pass_map_.end();
  }

  void Insert(const std::string &pass_type, const PassCreator &pass_creator) {
    PADDLE_ENFORCE_NE(Has(pass_type),
                      true,
                      common::errors::InvalidArgument(
                          "Pass %s has been registered.", pass_type));
    if (Has(pass_type)) {
      return;
    }
    pass_map_.insert({pass_type, pass_creator});
  }

  std::unique_ptr<Pass> Get(const std::string &pass_type) const {
    PADDLE_ENFORCE_EQ(Has(pass_type),
                      true,
                      common::errors::InvalidArgument(
                          "Pass %s has not been registered.", pass_type));
    return pass_map_.at(pass_type)();
  }

 private:
  PassRegistry() = default;
  std::unordered_map<std::string, PassCreator> pass_map_;

  DISABLE_COPY_AND_ASSIGN(PassRegistry);
};

template <typename PassType>
class IR_API PassRegistrar {
 public:
  // In our design, various kinds of passes,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_PASS macros to
  // call this method. So, as long as the callee code calls USE_PASS, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
  explicit PassRegistrar(const char *pass_type) {
    PassRegistry::Instance().Insert(
        pass_type, []() { return std::make_unique<PassType>(); });
  }
};

#define STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(uniq_name, msg)                   \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// Register a new pass that can be applied on the IR.
#define REGISTER_IR_PASS(pass_type, pass_class)               \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                        \
      __reg_pir_pass__##pass_type,                            \
      "REGISTER_IR_PASS must be called in global namespace"); \
  static ::pir::PassRegistrar<pass_class>                     \
      __pir_pass_registrar_##pass_type##__(#pass_type);       \
  IR_API int TouchPirPassRegistrar_##pass_type() {            \
    __pir_pass_registrar_##pass_type##__.Touch();             \
    return 0;                                                 \
  }                                                           \
  static ::pir::PassRegistrar<pass_class>                     \
      &__pir_ass_tmp_registrar_##pass_type##__ UNUSED =       \
          __pir_pass_registrar_##pass_type##__

#define USE_PIR_PASS(pass_type)                          \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                   \
      __use_pir_pass_itself_##pass_type,                 \
      "USE_PASS must be called in global namespace");    \
  extern int TouchPirPassRegistrar_##pass_type();        \
  static int use_pir_pass_itself_##pass_type##_ UNUSED = \
      TouchPirPassRegistrar_##pass_type()

}  // namespace pir
