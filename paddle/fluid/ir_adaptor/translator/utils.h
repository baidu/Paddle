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

#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/pd_op_sig.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {
namespace translator {

pir::Operation* InsertSliceOperationForTarget(
    pir::IrContext* ctx,
    TranslationContext* param_map,
    pir::Block* block,
    const VariableDefiningInfo& defining_info,
    const std::string& arg_name);

std::ostream& operator<<(std::ostream& os,
                         const std::vector<std::string>& vec_str);

TEST_API std::vector<std::string> CheckUnregisteredOperation(
    pir::IrContext* ctx, const framework::ProgramDesc& legacy_program);

inline DataType VarTypeToDataType(
    ::paddle::framework::proto::VarType_Type var_type) {
  switch (var_type) {
    case paddle::framework::proto::VarType_Type::VarType_Type_BOOL:
      return DataType::BOOL;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT16:
      return DataType::INT16;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT32:
      return DataType::INT32;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT64:
      return DataType::INT64;
    case paddle::framework::proto::VarType_Type::VarType_Type_FP16:
      return DataType::FLOAT16;
    case paddle::framework::proto::VarType_Type::VarType_Type_FP32:
      return DataType::FLOAT32;
    case paddle::framework::proto::VarType_Type::VarType_Type_FP64:
      return DataType::FLOAT64;
    case paddle::framework::proto::VarType_Type::VarType_Type_SIZE_T:
      return DataType::UINT64;
    case paddle::framework::proto::VarType_Type::VarType_Type_UINT8:
      return DataType::UINT8;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT8:
      return DataType::INT8;
    case paddle::framework::proto::VarType_Type::VarType_Type_BF16:
      return DataType::BFLOAT16;
    case paddle::framework::proto::VarType_Type::VarType_Type_COMPLEX64:
      return DataType::COMPLEX64;
    case paddle::framework::proto::VarType_Type::VarType_Type_COMPLEX128:
      return DataType::COMPLEX128;
    case paddle::framework::proto::VarType_Type::VarType_Type_PSTRING:
      return DataType::PSTRING;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported proto::VarType_Type `%s` when casting it into DataType.",
          var_type));
  }
}

}  // namespace translator
}  // namespace paddle
