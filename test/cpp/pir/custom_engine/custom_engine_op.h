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

#include <glog/logging.h>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "test/cpp/pir/tools/macros_utils.h"

#if defined(_WIN32)
#ifndef EXPORT_API
#define EXPORT_API __declspec(dllexport)
#endif  // EXPORT_API
#else
#define EXPORT_API
#endif  // _WIN32

#define IR_DECLARE_EXPLICIT_PLUGIN_TYPE_ID(TYPE_CLASS) \
  namespace pir {                                      \
  namespace detail {                                   \
  template <>                                          \
  class EXPORT_API TypeIdResolver<TYPE_CLASS> {        \
   public:                                             \
    static TypeId Resolve() { return id_; }            \
    static UniqueingId id_;                            \
  };                                                   \
  }                                                    \
  }  // namespace pir

namespace paddle {
namespace dialect {

class FakeEngineOp
    : public pir::Op<FakeEngineOp, paddle::dialect::OpYamlInfoInterface> {
 public:
  using Op::Op;
  static const char *name() { return "custom_engine.fake_engine"; }
  static const char *attributes_name[2];
  static constexpr uint32_t attributes_num = 2;
  static OpInfoTuple GetOpInfo();

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x,
                    std::vector<std::string> input_names,
                    std::vector<std::string> output_names,
                    std::vector<std::vector<int64_t>> outputs_shape,
                    std::vector<phi::DataType> outputs_dtype);

  void VerifySig();
  pir::Block *block();
  pir::Block *block() const;
  pir::Value x() { return operand_source(0); }
  pir::Value out() { return result(0); }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_PLUGIN_TYPE_ID(paddle::dialect::FakeEngineOp)
