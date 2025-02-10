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

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

#include <vector>
#include "glog/logging.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_meta_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace cinn {
namespace dialect {

using DenseTensorType = paddle::dialect::DenseTensorType;

const char* GroupOp::attributes_name[GroupOp::attributes_num] = {"group_info"};
const char* FusionOp::attributes_name[FusionOp::attributes_num] = {
    "group_info", "fusion_tracker"};
const char* ConcatOp::attributes_name[ConcatOp::attributes_num] = {"axis"};
const char* SplitOp::attributes_name[SplitOp::attributes_num] = {
    "num_or_sections", "axis"};

void GroupOp::Build(pir::Builder& builder,
                    pir::OperationArgument& argument,
                    const std::vector<pir::Type>& output_types) {
  argument.AddRegion(nullptr);
  argument.output_types = output_types;
}

void GroupOp::Build(pir::Builder& builder,             // NOLINT
                    pir::OperationArgument& argument,  // NOLINT
                    const std::vector<pir::Type>& output_types,
                    const cinn::dialect::GroupInfo& group_info) {
  argument.AddRegion(nullptr);
  argument.output_types = output_types;

  argument.AddAttribute("group_info",
                        cinn::dialect::GroupInfoAttribute::get(
                            pir::IrContext::Instance(), group_info));
}

void GroupOp::Build(pir::Builder& builder,             // NOLINT
                    pir::OperationArgument& argument,  // NOLINT
                    std::unique_ptr<pir::Block>&& block) {
  VLOG(4) << "Start build GroupOp";
  if (block && !block->empty()) {
    PADDLE_ENFORCE_EQ(block->back().isa<pir::YieldOp>(), true);
    auto& op = block->back();
    for (size_t i = 0; i < op.num_operands(); ++i) {
      argument.AddOutput(op.operand(i).type());
    }
  }
  argument.AddRegion().push_back(block.release());
}

pir::Block* GroupOp::block() {
  pir::Region& region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return &region.front();
}

pir::Block* GroupOp::block() const {
  pir::Region& region = (*this)->region(0);
  PADDLE_ENFORCE_EQ(region.empty(),
                    false,
                    ::common::errors::Unavailable(
                        "Required GroupOp's region must not be emptpy."));
  return &region.front();
}

std::vector<pir::Operation*> GroupOp::GetOperators() const {
  std::vector<pir::Operation*> rt_ops;
  for (auto& op : *block()) {
    rt_ops.push_back(&op);
  }
  return rt_ops;
}

void GroupOp::VerifySig() {}

void GroupOp::Print(pir::IrPrinter& printer) {
  auto& os = printer.os;
  auto op = operation();
  printer.PrintOpResult(*op);
  os << " = ";
  printer.PrintOpName(*op);
  printer.PrintOpId(*op);
  printer.PrintOpOperands(*op);
  os << " -> ";
  printer.PrintOpReturnType(*op);
  os << " {\n";
  printer.AddIndentation();
  for (auto& sub_op : GetOperators()) {
    printer.PrintOperation(*sub_op);
    os << "\n";
  }
  printer.DecreaseIndentation();
  os << printer.indentation() << "}";
}

bool GroupOp::InferSymbolicShape(
    ::pir::InferSymbolicShapeContext* infer_context) {
  ::pir::InferSymExprForBlock(*block(), infer_context);

  for (uint32_t rst_idx = 0; rst_idx < num_results(); rst_idx++) {
    auto inner_yield_value = block()->back().operand_source(rst_idx);
    const auto& shape =
        infer_context->GetShapeOrDataForValue(inner_yield_value);
    infer_context->SetShapeOrDataForValue(result(rst_idx), shape);
  }

  if (VLOG_IS_ON(4)) {
    ::std::cerr << ">>>>>>>>>>>>>>>>>>>> cinn_op.group(op_id: op_"
                << block()->back().id() << ") END." << ::std::endl;
  }
  return true;
}

void FusionOp::Build(pir::Builder& builder,
                     pir::OperationArgument& argument,
                     const std::vector<pir::Type>& output_types) {
  argument.AddRegion(nullptr);
  argument.output_types = output_types;
}

void FusionOp::Build(pir::Builder& builder,             // NOLINT
                     pir::OperationArgument& argument,  // NOLINT
                     const std::vector<pir::Type>& output_types,
                     const cinn::dialect::GroupInfo& group_info,
                     const cinn::fusion::FusionTrackerPtr& tracker) {
  argument.AddRegion(nullptr);
  argument.output_types = output_types;

  argument.AddAttribute("group_info",
                        cinn::dialect::GroupInfoAttribute::get(
                            pir::IrContext::Instance(), group_info));
  argument.AddAttribute("fusion_tracker",
                        cinn::dialect::FusionTrackerPtrAttribute::get(
                            pir::IrContext::Instance(), tracker));
}

pir::Block* FusionOp::block() {
  pir::Region& region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return &region.front();
}

pir::Block* FusionOp::block() const {
  pir::Region& region = (*this)->region(0);
  PADDLE_ENFORCE_EQ(region.empty(),
                    false,
                    ::common::errors::Unavailable(
                        "Required FusionOp's region must not be emptpy."));
  return &region.front();
}

std::vector<pir::Operation*> FusionOp::GetOperators() const {
  std::vector<pir::Operation*> rt_ops;
  for (auto& op : *block()) {
    rt_ops.push_back(&op);
  }
  return rt_ops;
}

void FusionOp::VerifySig() {}

void FusionOp::Print(pir::IrPrinter& printer) {
  auto& os = printer.os;
  auto op = operation();
  printer.PrintOpResult(*op);
  os << " = ";
  printer.PrintOpName(*op);
  printer.PrintOpId(*op);
  printer.PrintOpOperands(*op);
  os << " -> ";
  printer.PrintOpReturnType(*op);
  os << " {\n";
  printer.AddIndentation();
  for (auto& sub_op : GetOperators()) {
    printer.PrintOperation(*sub_op);
    os << "\n";
  }
  printer.DecreaseIndentation();
  os << printer.indentation() << "}";
}

bool FusionOp::InferSymbolicShape(
    ::pir::InferSymbolicShapeContext* infer_context) {
  ::pir::InferSymExprForBlock(*block(), infer_context);

  for (uint32_t rst_idx = 0; rst_idx < num_results(); rst_idx++) {
    auto inner_yield_value = block()->back().operand_source(rst_idx);
    const auto& shape =
        infer_context->GetShapeOrDataForValue(inner_yield_value);
    infer_context->SetShapeOrDataForValue(result(rst_idx), shape);
  }
  return true;
}

void YieldStoreOp::Build(pir::Builder& builder,
                         pir::OperationArgument& argument,
                         pir::Value x,
                         pir::Type output_type) {
  argument.inputs = {x};
  argument.output_types = {output_type};
}

void YieldStoreOp::VerifySig() {}

bool YieldStoreOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  infer_context->SetShapeOrDataForValue(
      result(0), infer_context->GetShapeOrDataForValue(operand_source(0)));
  return true;
}

bool ConcatOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  VLOG(4) << "Infer symbolic shape for cinn_op.concat";
  return ConcatOpInferSymbolicShape(this->operation(), infer_context);
}

void ConcatOp::Build(pir::Builder& builder,             // NOLINT
                     pir::OperationArgument& argument,  // NOLINT
                     const std::vector<pir::Value>& inputs,
                     int axis) {
  VLOG(4) << "Start build ConcatOp";

  argument.inputs = inputs;
  std::vector<pir::Type> inputs_type(inputs.size());

  PADDLE_ENFORCE_GT(inputs.size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "input size [%d] is less than 0", inputs.size()));

  const pir::Type out_type = [&]() {
    auto first_ele = inputs[0].type().dyn_cast<DenseTensorType>();
    phi::DDim out_dims = first_ele.dims();
    if (axis < 0) axis += out_dims.size();

    for (size_t idx = 1; idx < inputs.size(); ++idx) {
      inputs_type[idx] = inputs[idx].type();
      auto dim_i = inputs[idx].type().dyn_cast<DenseTensorType>().dims();

      if (out_dims[axis] > 0 && dim_i[axis] > 0) {
        out_dims[axis] += dim_i[axis];
      } else {
        out_dims[axis] = -1;
        break;
      }
    }
    return DenseTensorType::get(pir::IrContext::Instance(),
                                first_ele.dtype(),
                                out_dims,
                                first_ele.data_layout(),
                                first_ele.lod(),
                                first_ele.offset());
  }();
  argument.output_types.emplace_back(out_type);
  PassStopGradientsDefaultly(argument);
  argument.AddAttribute(
      "axis", pir::Int32Attribute::get(pir::IrContext::Instance(), axis));
}

void SplitOp::Build(pir::Builder& builder,             // NOLINT
                    pir::OperationArgument& argument,  // NOLINT
                    pir::Value input,
                    const std::vector<int>& sections,
                    int axis) {
  VLOG(4) << "Start build SplitOp";

  argument.inputs.push_back(input);

  std::vector<pir::Type> output_type(sections.size());

  auto input_ele = input.type().dyn_cast<DenseTensorType>();

  if (axis < 0) {
    axis += input_ele.dims().size();
  }
  std::vector<pir::Attribute> section_attrs;
  for (size_t idx = 0; idx < sections.size(); ++idx) {
    auto out_dims = input_ele.dims();
    out_dims[axis] = sections[idx];
    auto out_type = DenseTensorType::get(pir::IrContext::Instance(),
                                         input_ele.dtype(),
                                         out_dims,
                                         input_ele.data_layout(),
                                         input_ele.lod(),
                                         input_ele.offset());

    argument.output_types.emplace_back(out_type);

    pir::Attribute attr_axis =
        pir::Int32Attribute::get(pir::IrContext::Instance(), sections[idx]);

    section_attrs.push_back(attr_axis);
  }

  PassStopGradientsDefaultly(argument);

  argument.AddAttribute(
      "num_or_sections",
      pir::ArrayAttribute::get(pir::IrContext::Instance(), section_attrs));

  argument.AddAttribute(
      "axis", pir::Int32Attribute::get(pir::IrContext::Instance(), axis));
}

bool SplitOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  VLOG(4) << "Infer symbolic shape for cinn_op.split";
  return SplitOpInferSymbolicShape(this->operation(), infer_context);
}

const char* GenerateShapeOp::attributes_name[attributes_num] = {
    "output_dim_exprs", "symbol_bindings"};

void GenerateShapeOp::Build(pir::Builder& builder,
                            pir::OperationArgument& argument,
                            const std::vector<pir::Value>& inputs,
                            const std::vector<pir::Attribute>& output_dim_exprs,
                            const SymbolBindings& symbol_bindings,
                            const pir::Type& output_type) {
  if (inputs.empty()) {
    VLOG(3) << "GenerateShapeOp inputs is empty";
    for (const auto& attr : output_dim_exprs) {
      PADDLE_ENFORCE(attr.isa<pir::Int64Attribute>(),
                     ::common::errors::PreconditionNotMet(
                         "Required attr must be Int64Attribute."));
    }
  }
  argument.AddInputs(inputs);
  argument.AddAttribute("output_dim_exprs",
                        builder.array_attr(output_dim_exprs));
  argument.AddAttribute(
      "symbol_bindings",
      ConvertSymbolBindingsToAttribute(builder, symbol_bindings));
  argument.AddOutput(output_type);
  ::pir::PassStopGradientsDefaultly(argument);
}

namespace {

const char* GetSymbolBindingTypeImpl(
    const GenerateShapeOp::DataSymbolBinding& binding) {
  return "DataSymbolBinding";
}

const char* GetSymbolBindingTypeImpl(
    const GenerateShapeOp::ShapeSymbolBinding& binding) {
  return "ShapeSymbolBinding";
}

const char* GetSymbolBindingType(
    const GenerateShapeOp::SymbolBinding& binding) {
  return std::visit(
      [](const auto& impl) { return GetSymbolBindingTypeImpl(impl); }, binding);
}

const GenerateShapeOp::SymbolBindingBase* GetSymbolBindingBaseImpl(
    const GenerateShapeOp::DataSymbolBinding& binding) {
  return &binding;
}

const GenerateShapeOp::SymbolBindingBase* GetSymbolBindingBaseImpl(
    const GenerateShapeOp::ShapeSymbolBinding& binding) {
  return &binding;
}

const GenerateShapeOp::SymbolBindingBase* GetSymbolBindingBase(
    const GenerateShapeOp::SymbolBinding& binding) {
  return std::visit(
      [](const auto& impl) { return GetSymbolBindingBaseImpl(impl); }, binding);
}

typedef GenerateShapeOp::SymbolBinding (*SymbolBindingConstructorT)(
    const std::string& symbol_name,
    int64_t input_tensor_idx,
    int64_t input_tensor_dim_idx);

GenerateShapeOp::SymbolBinding MakeDataSymbolBinding(
    const std::string& symbol_name,
    int64_t input_tensor_idx,
    int64_t input_tensor_dim_idx) {
  return GenerateShapeOp::DataSymbolBinding{
      symbol_name, input_tensor_idx, input_tensor_dim_idx};
}

GenerateShapeOp::SymbolBinding MakeShapeSymbolBinding(
    const std::string& symbol_name,
    int64_t input_tensor_idx,
    int64_t input_tensor_dim_idx) {
  return GenerateShapeOp::ShapeSymbolBinding{
      symbol_name, input_tensor_idx, input_tensor_dim_idx};
}

std::optional<SymbolBindingConstructorT> GetMakerSymbolBinding(
    const std::string& type) {
  static std::map<std::string, SymbolBindingConstructorT> map{
      {GetSymbolBindingTypeImpl(GenerateShapeOp::DataSymbolBinding{}),
       &MakeDataSymbolBinding},
      {GetSymbolBindingTypeImpl(GenerateShapeOp::ShapeSymbolBinding{}),
       &MakeShapeSymbolBinding},
  };
  const auto& iter = map.find(type);
  if (iter == map.end()) return std::nullopt;
  return iter->second;
}

std::optional<GenerateShapeOp::SymbolBinding> MakeSymbolBinding(
    const std::string& type,
    const std::string& symbol_name,
    int64_t input_tensor_idx,
    int64_t input_tensor_dim_idx) {
  auto opt_creator = GetMakerSymbolBinding(type);
  if (!opt_creator.has_value()) return std::nullopt;
  return opt_creator.value()(
      symbol_name, input_tensor_idx, input_tensor_dim_idx);
}

}  // namespace

pir::Attribute GenerateShapeOp::ConvertSymbolBindingsToAttribute(
    pir::Builder& builder,
    const GenerateShapeOp::SymbolBindings& symbol_bindings) {
  const auto& ConvertSymbolBindingToAttr = [&](const SymbolBinding& binding) {
    const auto* type = GetSymbolBindingType(binding);
    const auto& [symbol_name, input_tensor_idx, input_tensor_dim_idx] =
        *GetSymbolBindingBase(binding);
    return builder.array_attr({
        builder.str_attr(type),
        builder.str_attr(symbol_name),
        builder.int64_attr(input_tensor_idx),
        builder.int64_attr(input_tensor_dim_idx),
    });
  };
  std::vector<pir::Attribute> bindings_attr{};
  for (const auto& symbol_binding : symbol_bindings) {
    bindings_attr.push_back(ConvertSymbolBindingToAttr(symbol_binding));
  }
  return builder.array_attr(bindings_attr);
}

std::optional<GenerateShapeOp::SymbolBindings>
GenerateShapeOp::ConvertAttributeToSymbolBindings(
    const pir::Attribute& symbol_bindings) {
  if (!symbol_bindings.isa<pir::ArrayAttribute>()) return std::nullopt;
  const auto& symbol_bindings_array_attr =
      symbol_bindings.dyn_cast<pir::ArrayAttribute>();
  GenerateShapeOp::SymbolBindings ret{GenerateShapeOp::SymbolBindings{}};
  for (int i = 0; i < symbol_bindings_array_attr.size(); ++i) {
    const auto& symbol_binding = symbol_bindings_array_attr.at(i);
    if (!symbol_binding.isa<pir::ArrayAttribute>()) return std::nullopt;
    const auto& symbol_binding_array_attr =
        symbol_binding.dyn_cast<pir::ArrayAttribute>();
    if (symbol_binding_array_attr.size() != 4) return std::nullopt;
    if (!symbol_binding_array_attr.at(0).isa<pir::StrAttribute>())
      return std::nullopt;
    if (!symbol_binding_array_attr.at(1).isa<pir::StrAttribute>())
      return std::nullopt;
    if (!symbol_binding_array_attr.at(2).isa<pir::Int64Attribute>())
      return std::nullopt;
    if (!symbol_binding_array_attr.at(3).isa<pir::Int64Attribute>())
      return std::nullopt;
    const auto& opt_symbol_binding = MakeSymbolBinding(
        symbol_binding_array_attr.at(0)
            .dyn_cast<pir::StrAttribute>()
            .AsString(),
        symbol_binding_array_attr.at(1)
            .dyn_cast<pir::StrAttribute>()
            .AsString(),
        symbol_binding_array_attr.at(2).dyn_cast<pir::Int64Attribute>().data(),
        symbol_binding_array_attr.at(3).dyn_cast<pir::Int64Attribute>().data());
    if (!opt_symbol_binding.has_value()) return std::nullopt;
    ret.emplace_back(opt_symbol_binding.value());
  }
  return std::move(ret);
}
bool GenerateShapeOp::InferSymbolicShape(
    pir::InferSymbolicShapeContext* infer_context) {
  const auto attr_dim_exprs = [&] {
    pir::Attribute dim_expr_attr = this->attributes().at("output_dim_exprs");
    auto dim_exprs = ConvertAttributeToDimExprs(dim_expr_attr);

    PADDLE_ENFORCE_EQ(
        dim_exprs.has_value(),
        true,
        ::common::errors::PreconditionNotMet(
            "Required success to execute convert attribute to dim exprs."));

    return dim_exprs.value();
  }();
  const auto symbol_bindings = [&] {
    pir::Attribute symbol_bindings_attr =
        this->attributes().at("symbol_bindings");
    auto symbol_bindings =
        ConvertAttributeToSymbolBindings(symbol_bindings_attr);
    PADDLE_ENFORCE(symbol_bindings.has_value(),
                   ::common::errors::PreconditionNotMet(
                       "Required symbol_bindings.has_value()==true."));
    return symbol_bindings.value();
  }();
  auto DimExprs4InputDim =
      [&](int input_idx) -> const symbol::ShapeOrDataDimExprs& {
    return infer_context->GetShapeOrDataForValue(
        this->operand_source(input_idx));
  };
  auto DimExprs4SymbolName =
      MakeGetterDimExpr4SymbolName(symbol_bindings, DimExprs4InputDim);
  const auto substituted_dim_exprs = [&] {
    std::vector<symbol::DimExpr> dim_exprs{};
    dim_exprs.reserve(attr_dim_exprs.size());
    for (const auto& attr_dim_expr : attr_dim_exprs) {
      const auto& substituted =
          SubstituteDimExpr(attr_dim_expr, DimExprs4SymbolName);
      const auto& simplified = symbol::SimplifyDimExpr(substituted);
      dim_exprs.push_back(simplified);
    }
    return dim_exprs;
  }();

  const auto& out_dims = this->out().type().dyn_cast<DenseTensorType>().dims();
  const auto shape = [&] {
    std::vector<symbol::DimExpr> result;
    for (int i = 0; i < out_dims.size(); ++i) {
      result.emplace_back(out_dims.at(i));
    }
    return result;
  }();
  symbol::ShapeOrDataDimExprs shape_or_data_dim_exprs{
      symbol::TensorShapeOrDataDimExprs(shape, substituted_dim_exprs)};

  infer_context->SetShapeOrDataForValue(this->out(), shape_or_data_dim_exprs);

  return true;
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::GroupOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::FusionOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::ConcatOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::SplitOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::GenerateShapeOp);
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::YieldStoreOp);
