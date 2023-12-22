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

#include "paddle/fluid/ir_adaptor/translator/utils.h"

#include <unordered_map>

#include "paddle/common/enforce.h"
#include "paddle/fluid/ir_adaptor/translator/op_translator.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/utils.h"

namespace paddle {
namespace dialect {
bool HaveOpToMultiKernelsMap(std::string op_name) {
  return op_to_multi_kernels_map.find(op_name) != op_to_multi_kernels_map.end();
}

const std::vector<PdOpSig>& LegacyOpToPdOpsMapping(std::string op_name) {
  return op_to_multi_kernels_map[op_name];
}
}  // namespace dialect
}  // namespace paddle

namespace paddle {
namespace translator {

pir::Operation* InsertSliceOperationForTarget(
    pir::IrContext* ctx,
    TranslationContext* param_map,
    pir::Block* block,
    const VariableDefiningInfo& defining_info,
    const std::string& arg_name) {
  std::string slice_op_name(pir::SliceOp::name());
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(slice_op_name);
  std::unordered_map<std::string, pir::Attribute> op_attribute_map = {
      {"index", pir::Int32Attribute::get(ctx, defining_info.idx_in_vector)},
  };
  pir::VectorType src_vec_type =
      defining_info.value.type().dyn_cast<pir::VectorType>();
  pir::Operation* operation =
      pir::Operation::Create({defining_info.value},
                             op_attribute_map,
                             {src_vec_type[defining_info.idx_in_vector]},
                             op_info);
  block->push_back(operation);
  pir::OpResult target_op_result = operation->result(0);
  param_map->PushValue(arg_name, VariableDefiningInfo(target_op_result));
  return operation;
}

std::ostream& operator<<(std::ostream& os,
                         const std::vector<std::string>& vec_str) {
  pir::PrintInterleave(
      vec_str.begin(),
      vec_str.end(),
      [&os](std::string s) { os << s; },
      [&os]() { os << ", "; });
  return os;
}

std::vector<std::string> CheckUnregisteredOperationInBlock(
    pir::IrContext* ctx, const framework::BlockDesc& block) {
  auto& op_translator = OpTranslator::instance();
  std::vector<std::string> unregistered_ops;
  for (auto op : block.AllOps()) {
    if (op_translator.HasSpecialHandler(op->Type())) {
      continue;
    }
    OpTranscriber general_handler;
    try {
      general_handler.LoopkUpOpInfo(ctx, *op);
    } catch (pir::IrNotMetException& e) {
      unregistered_ops.push_back(op->Type());
    }
  }
  return unregistered_ops;
}

std::vector<std::string> CheckUnregisteredOperation(
    pir::IrContext* ctx, const framework::ProgramDesc& legacy_program) {
  ctx->GetOrRegisterDialect<dialect::OperatorDialect>();

  std::vector<std::string> unregistered_ops;
  for (size_t block_idx = 0; block_idx < legacy_program.Size(); block_idx++) {
    const framework::BlockDesc& block = legacy_program.Block(block_idx);
    auto ops = CheckUnregisteredOperationInBlock(ctx, block);
    unregistered_ops.insert(unregistered_ops.end(), ops.begin(), ops.end());
  }

  return unregistered_ops;
}

phi::DataType PirTypeToPhiDType(pir::Type type) {
  if (type.isa<pir::UInt8Type>()) {
    return phi::DataType::UINT8;
  } else if (type.isa<pir::Int8Type>()) {
    return phi::DataType::INT8;
  } else if (type.isa<pir::Int16Type>()) {
    return phi::DataType::INT16;
  } else if (type.isa<pir::Int32Type>()) {
    return phi::DataType::INT32;
  } else if (type.isa<pir::Int64Type>()) {
    return phi::DataType::INT64;
  } else if (type.isa<pir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (type.isa<pir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (type.isa<pir::BoolType>()) {
    return phi::DataType::BOOL;
  } else if (type.isa<pir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (type.isa<pir::BFloat16Type>()) {
    return phi::DataType::BFLOAT16;
  } else if (type.isa<pir::Complex64Type>()) {
    return phi::DataType::COMPLEX64;
  } else if (type.isa<pir::Complex128Type>()) {
    return phi::DataType::COMPLEX128;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported pirType `%s` when casting it into phi::DataType.", type));
  }
}

}  // namespace translator
}  // namespace paddle
