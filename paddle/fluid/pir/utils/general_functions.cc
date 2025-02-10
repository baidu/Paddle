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

#include "paddle/fluid/pir/utils/general_functions.h"

#include <unordered_set>

#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/src/ir_operation_factory.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace {

void GetUsedExternalValueImpl(
    std::unordered_set<pir::Value>& defined_values,  // NOLINT
    std::vector<pir::Value>& used_values,            // NOLINT
    const pir::Operation& op) {
  for (size_t index = 0; index < op.num_operands(); ++index) {
    pir::Value value = op.operand_source(index);
    if (defined_values.find(value) == defined_values.end()) {
      used_values.push_back(value);
      defined_values.insert(value);
    }
  }
  for (auto& region : op) {
    for (auto& block : region) {
      for (auto value : block.args()) {
        defined_values.insert(value);
      }
      for (const auto& [_, value] : block.kwargs()) {
        defined_values.insert(value);
      }
    }
    for (auto& block : region) {
      for (auto& inner_op : block) {
        GetUsedExternalValueImpl(defined_values, used_values, inner_op);
      }
    }
  }
  for (size_t index = 0; index < op.num_results(); ++index) {
    defined_values.insert(op.result(index));
  }
}

}  // namespace

namespace pir {

void TensorCopySync(const phi::DenseTensor& src,
                    phi::DenseTensor* dst,
                    const phi::Place& dst_place) {
  paddle::framework::TensorCopySync(src, dst_place, dst);
}

void DenseTensorCastToFp32(phi::DenseTensor* in,
                           phi::DenseTensor* out,
                           int world_size) {
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));

  phi::DenseTensor fp32_tensor;
  phi::DenseTensor* out_ptr = out == nullptr ? &fp32_tensor : out;
  out_ptr->Resize(in->dims());
  out_ptr->set_type(phi::DataType::FLOAT32);
  out_ptr->set_layout(in->layout());

  switch (in->dtype()) {
    case phi::DataType::FLOAT16:
      phi::CastKernel<phi::dtype::float16, phi::CPUContext>(
          *cpu_ctx, *in, phi::DataType::FLOAT32, out_ptr);
      break;
    case phi::DataType::FLOAT32:
      if (out == nullptr) {
        if (world_size > 1) {
          phi::ScaleKernel<float, phi::CPUContext>(
              *cpu_ctx, *in, 1.0f / world_size, 0.f, false, in);
        }
        return;
      } else {
        phi::AssignKernel(*cpu_ctx, *in, out_ptr);
      }
      break;
    default:
      PADDLE_THROW(common::errors::InvalidType(
          "Only support fp16 and fp32, but received dtype is %s.",
          phi::DataTypeToString(in->dtype())));
      break;
  }
  if (world_size > 1) {
    phi::ScaleKernel<float, phi::CPUContext>(
        *cpu_ctx, *out_ptr, 1.0f / world_size, 0.f, false, out_ptr);
  }
  if (out == nullptr) {
    phi::AssignKernel(*cpu_ctx, *in, out_ptr);
  }
}

pir::Type TranslateToIrDataType(phi::DataType dtype) {
  // Get Meta
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Type data_type = paddle::dialect::TransToIrDataType(dtype, ctx);
  return data_type;
}

pir::Operation* CreateOperationByName(const std::string& op_name,
                                      const std::vector<pir::Value>& inputs,
                                      const pir::AttributeMap& attrs,
                                      const pir::PatternRewriter& rewriter) {
  return paddle::drr::OperationFactory::Instance().CreateOperation(
      op_name, inputs, attrs, const_cast<pir::PatternRewriter&>(rewriter));
}

pir::Attribute CreateDataTypeAttr(pir::IrContext* ctx, phi::DataType dtype) {
  return paddle::dialect::DataTypeAttribute::get(ctx, dtype);
}

template <typename T>
T* VarGetMutable(Variable* var) {
  return var->GetMutable<T>();
}

template <typename T>
bool VarIsType(Variable* var) {
  return var->IsType<T>();
}

template phi::DenseTensor* VarGetMutable<phi::DenseTensor>(Variable*);
template bool VarIsType<phi::DenseTensor>(Variable*);

Variable* ScopeFindVar(Scope* scope_, const std::string& name) {
  return scope_->FindVar(name);
}

Variable* ScopeGetVar(Scope* scope_, const std::string& name) {
  return scope_->GetVar(name);
}

Variable* ScopeVar(Scope* scope_, const std::string& name) {
  return scope_->Var(name);
}

std::vector<std::string> ScopeGetVarNames(Scope* scope_) {
  return scope_->LocalVarNames();
}

Scope* GetScopeImpl(pir::Pass* pass) {
  // get scope from pass
  return &pass->Get<Scope>(pir::Pass::kParamScopeAttr);
}

std::string GetParameterNameFromValue(const pir::Value& value) {
  pir::Operation* owner = value.defining_op();
  std::string name;
  if (owner->isa<ParameterOp>()) {
    pir::ParameterOp op = owner->dyn_cast<pir::ParameterOp>();
    name = op.param_name();
  } else if (owner->isa<ConstantTensorOp>()) {
    pir::ConstantTensorOp op = owner->dyn_cast<pir::ConstantTensorOp>();
    name = op.tensor_name();
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("Value must be a weight from a Parameter "
                                      "or a ConstantTensorOp op."));
  }
  return name;
}

std::vector<int64_t> GetShapeFromValue(const pir::Value& value) {
  if (value.type().isa<paddle::dialect::DenseTensorType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
  } else if (value.type().isa<paddle::dialect::SelectedRowsType>()) {
    return phi::vectorize(
        value.type().dyn_cast<paddle::dialect::SelectedRowsType>().dims());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get shape for dense_tensor or selected_rows."));
  }
}

pir::Type GetDataTypeFromValue(const pir::Value& value) {
  // TODO(dev): Support other types like DenseTensor.
  PADDLE_ENFORCE_EQ(value.type().isa<paddle::dialect::DenseTensorType>(),
                    true,
                    common::errors::InvalidArgument(
                        "Value's type must be a DenseTensorType."));
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype();
}

Operation* GetDefiningOpForInput(const Operation* op, uint32_t index) {
  PADDLE_ENFORCE_EQ(
      index < op->num_operands() && op->operand_source(index),
      true,
      common::errors::InvalidArgument("Input operand's index must be valid."));
  return op->operand_source(index).defining_op();
}

std::vector<std::pair<Operation*, int32_t>> GetUseOpsForOutput(
    const Operation* op, uint32_t index) {
  PADDLE_ENFORCE_EQ(index < op->num_results(),
                    true,
                    common::errors::InvalidArgument(
                        "Output op result's index must be valid."));
  auto result = op->result(index);
  std::vector<std::pair<Operation*, int32_t>> use_ops;
  for (auto it = result.use_begin(); it != result.use_end(); ++it) {
    use_ops.emplace_back(it->owner(), it->index());
  }
  return use_ops;
}

std::vector<pir::Value> GetUsedExternalValue(const pir::Operation& op) {
  std::unordered_set<pir::Value> defined_values{nullptr};
  std::vector<pir::Value> used_values;
  GetUsedExternalValueImpl(defined_values, used_values, op);
  return used_values;
}

std::vector<pir::Value> GetUsedExternalValue(const pir::Block& block) {
  auto& args = block.args();
  std::unordered_set<pir::Value> defined_values(args.begin(), args.end());
  std::vector<pir::Value> used_values;
  for (auto& op : block) {
    GetUsedExternalValueImpl(defined_values, used_values, op);
  }
  return used_values;
}

bool ValueIsPersistable(const pir::Value& value) {
  if (!value.defining_op()) {
    return false;
  }
  if (value.defining_op()->num_operands() > 0) {
    for (const auto& source_value : value.defining_op()->operands_source()) {
      if (!ValueIsPersistable(source_value)) {
        return false;
      }
    }
  } else {
    if (!value.defining_op()->isa<pir::ParameterOp>() &&
        !value.defining_op()->isa<paddle::dialect::FullOp>() &&
        !value.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {
      return false;
    }
  }
  return true;
}

phi::DataType GetTensorDtype(pir::Type type) {
  if (!type) {
    PADDLE_THROW(
        common::errors::InvalidArgument("The type of value is nullptr."));
  }
  if (auto dense_tensor_type = type.dyn_cast<pir::DenseTensorType>()) {
    return paddle::dialect::TransToPhiDataType(dense_tensor_type.dtype());
  } else if (auto sparse_coo_tensor_type =
                 type.dyn_cast<paddle::dialect::SparseCooTensorType>()) {
    return paddle::dialect::TransToPhiDataType(sparse_coo_tensor_type.dtype());
  } else if (auto sparse_csr_tensor_type =
                 type.dyn_cast<paddle::dialect::SparseCsrTensorType>()) {
    return paddle::dialect::TransToPhiDataType(sparse_csr_tensor_type.dtype());
  } else if (auto select_rows =
                 type.dyn_cast<paddle::dialect::SelectedRowsType>()) {
    return paddle::dialect::TransToPhiDataType(select_rows.dtype());
  } else if (auto dense_array =
                 type.dyn_cast<paddle::dialect::DenseTensorArrayType>()) {
    return paddle::dialect::TransToPhiDataType(dense_array.dtype());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get phi::DataType from DenseTensorType and "
        "SelectedRowsType, DenseTensorArrayType,SparseCooTensorType or "
        "SparseCsrTensorType."));
  }
}

phi::DataType GetValueDtype(const pir::Value& val) {
  return GetTensorDtype(val.type());
}

}  // namespace pir
