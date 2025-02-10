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

#include "paddle/fluid/framework/new_executor/instruction/custom_kernel_instruction.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

namespace paddle::framework {

void CustomKernelInstruction::BuildCustomContext(
    const paddle::dialect::OpYamlInfoParser& op_yaml_info) {
  auto& op_inplace_map = OpMetaInfoHelper::GetInplaceMap(*custom_op_meta_);
  // check inplace
  for (auto const& pair : op_inplace_map) {
    pir::Value output_value =
        op_->result(op_yaml_info.OutputName2Id().at(pair.second));
    if (paddle::framework::detail::IsDuplicableVar(pair.first) &&
        !IsInvalid(output_value)) {
      // make sure ctx has valid inplace optional outputs
      PADDLE_ENFORCE(
          paddle::framework::detail::IsOptionalVar(pair.second),
          common::errors::InvalidArgument(
              "Custom operator couldn't find custom output name for %s. If "
              "you are using inplace optional inputs & outputs, please "
              "check "
              "your InplaceMap and `Outputs` again and make sure %s is "
              "wrapped by `paddle::Optional`",
              pair.second,
              pair.second));
    }
  }

  Scope* inner_scope = value_exec_info_.GetScope();
  VLOG(6) << "Build custom op infermeta param inner_scope[" << inner_scope
          << "]";

  auto attr_map = op_->attributes();

  // EmplaceBackInputs
  auto& vec_input_tensor_params = op_yaml_info.TensorParams(true);
  auto& name2id = op_yaml_info.InputName2Id();
  auto inplace_id_map = op_yaml_info.GetInplaceIdMap();
  int input_index = 0;
  int vec_input_index = 0;
  for (auto& t : vec_input_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(t),
        true,
        common::errors::NotFound("param [%s] MUST in name2id map", t));

    pir::Value ptr = op_->operand_source(op_yaml_info.InputName2Id().at(t));
    if (!IsInvalid(ptr)) {
      if (op_yaml_info.GetInputType(op_yaml_info.InputName2Id().at(t)) ==
          "pir::VectorType<paddle::dialect::DenseTensorType>") {
        vec_input_name2id_map_[t] = vec_input_index;
        vec_input_index++;
        vec_input_ptrs_.emplace_back();
        // NOTE(YuanRisheng): In dygraph mode, we can not distinguish Tensor and
        // vector<Tensor> when user inputs None, so dygraph mode appends one
        // un-initialized Tensor to CustomOpKernelContext. To be compatible with
        // dygraph mode, `custom_vec_in` also emplace_back one un-initialized
        // tensor here.
        std::vector<paddle::Tensor> custom_vec_in;
        custom_vec_in.emplace_back(paddle::Tensor());
        custom_kernel_ctx_.EmplaceBackInputs(std::move(custom_vec_in));
      } else {
        input_name2id_map_[t] = input_index;
        input_index++;
        input_ptrs_.emplace_back(nullptr);
        custom_kernel_ctx_.EmplaceBackInput(paddle::Tensor());
      }
      VLOG(8) << "ctx->EmplaceBackInput : an optional input " << t;
      continue;
    }
    auto in_var_name = value_exec_info_.GetVarName(ptr);

    VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            common::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    if (var->IsType<phi::DenseTensor>()) {
      auto dense_tensor_in = var->GetMutable<phi::DenseTensor>();

      std::shared_ptr<phi::DenseTensor> tensor_in(
          dense_tensor_in, [](phi::DenseTensor* ptr) {
            VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
          });
      input_name2id_map_[t] = input_index;
      input_index++;
      input_ptrs_.push_back(dense_tensor_in);
      paddle::Tensor custom_in;
      custom_in.set_impl(tensor_in);
      custom_kernel_ctx_.EmplaceBackInput(std::move(custom_in));
    } else if (var->IsType<VariableRefArray>()) {
      std::vector<phi::DenseTensor*> vec_input_ptrs;
      std::vector<paddle::Tensor> vec_custom_in;
      auto& variable_array = var->Get<VariableRefArray>();
      for (size_t i = 0; i < variable_array.size(); ++i) {
        if (variable_array[i]->IsType<phi::DenseTensor>()) {
          phi::DenseTensor* dense_tensor_in = const_cast<phi::DenseTensor*>(
              &(variable_array[i]->Get<phi::DenseTensor>()));
          std::shared_ptr<phi::DenseTensor> tensor_in(
              dense_tensor_in, [](phi::DenseTensor* ptr) {
                VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
              });
          vec_input_ptrs.push_back(dense_tensor_in);
          paddle::Tensor custom_in;
          custom_in.set_impl(tensor_in);
          vec_custom_in.push_back(std::move(custom_in));
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "Only support Vector<DenseTensor> and vector<SelectedRows> now, "
              "not support vector<%d>.",
              variable_array[i]->Type()));
        }
      }
      vec_input_name2id_map_[t] = vec_input_index;
      vec_input_index++;
      vec_input_ptrs_.push_back(vec_input_ptrs);
      custom_kernel_ctx_.EmplaceBackInputs(vec_custom_in);
    } else {
      PADDLE_THROW(common::errors::Unimplemented("Not support var type [%d] ",
                                                 var->Type()));
    }
  }
  // EmplaceBackAttributes
  auto& vec_attr_params = op_yaml_info.AttrParams(true);
  for (auto& t : vec_attr_params) {
    PADDLE_ENFORCE_NE(attr_map.find(t),
                      attr_map.end(),
                      common::errors::NotFound(
                          "Not found %s in attr_map, it maybe need mapping "
                          "it in OpTranslator.",
                          t));
    auto& attr_type_name = op_yaml_info.AttrTypeName(t);
    if (attr_type_name == "pir::Int32Attribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::Int32Attribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::Int32Attribute>().data());
    } else if (attr_type_name == "pir::Int64Attribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::Int64Attribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::Int64Attribute>().data());
    } else if (attr_type_name == "pir::FloatAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::FloatAttribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::FloatAttribute>().data());
    } else if (attr_type_name == "pir::DoubleAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::DoubleAttribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::DoubleAttribute>().data());
    } else if (attr_type_name == "pir::BoolAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::BoolAttribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::BoolAttribute>().data());
    } else if (attr_type_name == "pir::StrAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::StrAttribute>().AsString());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::StrAttribute>().AsString());
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int32Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<int32_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int32Attribute>(),
            true,
            common::errors::Unimplemented(
                "the 0th elementwise MUST be pir::Int32Attribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int32Attribute>().data());
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::FloatAttribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<float> vec_res;
      if (array_list.size() > 0) {
        if (array_list[0].isa<pir::FloatAttribute>()) {
          for (size_t i = 0; i < array_list.size(); ++i) {
            vec_res.push_back(
                array_list[i].dyn_cast<pir::FloatAttribute>().data());
          }

        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "attr type not support [%s] ", attr_type_name));
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int64Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int64Attribute>(),
            true,
            common::errors::PreconditionNotMet(
                "Element in array list MUST be pir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int64Attribute>().data());
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::StrAttribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<std::string> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::StrAttribute>(),
            true,
            common::errors::PreconditionNotMet(
                "Element in array list MUST be pir::StrAttribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::StrAttribute>().AsString());
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);

    } else {
      PADDLE_THROW(common::errors::Unimplemented("attr type not support [%s] ",
                                                 attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }

  // EmplaceBackOutputs
  VLOG(8) << "ctx->EmplaceBackOutput: ";
  for (size_t i = 0; i < op_->num_results(); ++i) {
    pir::Value out_ptr = op_->result(i);
    auto out_name = op_yaml_info.OutputNames()[i];
    if (!IsInvalid(out_ptr)) {
      PADDLE_ENFORCE(
          paddle::framework::detail::IsOptionalVar(out_name) &&
              !inplace_id_map.empty(),
          common::errors::InvalidArgument(
              "Custom operator couldn't find custom output for name %s. If "
              "you "
              "are using inplace optional inputs & outputs, please check "
              "your "
              "InplaceMap and `Outputs` again and make sure %s is wrapped by "
              "`paddle::Optional`",
              out_name,
              out_name));
      VLOG(3) << "Custom Operator: BuildContext - inplace optional outputs : "
              << out_name << " is None.";
      custom_kernel_ctx_.EmplaceBackOutput(paddle::Tensor());

      VLOG(8) << "ctx->EmplaceBackOutput : an optional output";
      continue;
    }

    if (out_ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto dense_tensor_out =
          inner_scope->FindVar(value_exec_info_.GetVarName(out_ptr))
              ->GetMutable<phi::DenseTensor>();
      cache_out_ptrs_.push_back(dense_tensor_out);
      std::shared_ptr<phi::DenseTensor> tensor_out(
          dense_tensor_out, [](phi::DenseTensor* ptr) {
            VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
          });
      paddle::Tensor custom_out;
      // here only can copy the output tensor into context
      custom_out.set_impl(tensor_out);

      custom_kernel_ctx_.EmplaceBackOutput(std::move(custom_out));
      VLOG(8) << "ctx->EmplaceBackOutput DenseTensor: "
              << value_exec_info_.GetVarName(out_ptr);
    } else if (out_ptr.type().isa<pir::VectorType>()) {
      std::vector<paddle::Tensor> vec_custom_out;
      auto& variable_array =
          inner_scope->FindVar(value_exec_info_.GetVarName(out_ptr))
              ->Get<VariableRefArray>();
      std::vector<paddle::Tensor> custom_vec_out;
      PADDLE_ENFORCE(
          !inplace_id_map.empty() || (i == 0UL && op_->num_results() == 1UL),
          common::errors::PreconditionNotMet(
              "If custom operator's outputs contains `paddle::Vec()` type "
              "without setting InplaceMap, it only can hold one output."));
      for (size_t j = 0; j < variable_array.size(); ++j) {
        if (variable_array[j]->IsType<phi::DenseTensor>()) {
          auto dense_tensor_out = const_cast<phi::DenseTensor*>(
              &(variable_array[j]->Get<phi::DenseTensor>()));
          cache_out_ptrs_.emplace_back(dense_tensor_out);
          std::shared_ptr<phi::DenseTensor> tensor_out(
              dense_tensor_out, [](phi::DenseTensor* ptr) {
                VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
              });
          paddle::Tensor custom_out;
          custom_out.set_impl(tensor_out);
          custom_vec_out.push_back(std::move(custom_out));
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "Only support Vector<DenseTensor> now, "
              "not support vector<%d>.",
              variable_array[j]->Type()));
        }
      }
      VLOG(8) << "ctx->EmplaceBackOutput VariableRefArray: "
              << value_exec_info_.GetVarName(out_ptr);
      custom_kernel_ctx_.EmplaceBackOutputs(custom_vec_out);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "only support DenseTensor and vector "));
    }
  }

  auto& op_inputs = OpMetaInfoHelper::GetInputs(*custom_op_meta_);
  auto& op_outputs = OpMetaInfoHelper::GetOutputs(*custom_op_meta_);

  // handle inplace map
  custom_kernel_ctx_.UpdatePlainOutputs(op_inputs, op_outputs, op_inplace_map);
  VLOG(6) << "Done build custom context";
}

CustomKernelInstruction::CustomKernelInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo& value_exec_info)
    : InstructionBase(id, place),
      input_name2id_map_(),
      vec_input_name2id_map_(),
      input_shapes_(),
      vec_input_shapes_(),
      custom_attrs_(),
      input_dtypes_(),
      vec_input_dtypes_(),
      input_ptrs_(),
      vec_input_ptrs_(),
      cache_out_ptrs_(),
      value_exec_info_(value_exec_info) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  custom_op_name_ = op_name;
  VLOG(6) << "construct custom kernel instruction for: " << custom_op_name_;

  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      common::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", custom_op_name_));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_(custom_op_name_),
      paddle::dialect::IsLegacyOp(custom_op_name_));
  VLOG(6) << "finish process yaml_info_parser";

  const auto& op_meta =
      paddle::framework::detail::GetOpInfoByPirName(custom_op_name_);
  custom_op_meta_ = &op_meta;
  infershape_func_ = OpMetaInfoHelper::GetInferShapeFn(op_meta);
  inferdtype_func_ = OpMetaInfoHelper::GetInferDtypeFn(op_meta);
  kernel_func_ = OpMetaInfoHelper::GetKernelFn(op_meta);
  BuildCustomContext(yaml_info_parser);
  VLOG(6) << "finish process custom context";
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  auto& op_inplace_map = OpMetaInfoHelper::GetInplaceMap(*custom_op_meta_);
  for (auto const& pair : op_inplace_map) {
    pir::Value input_value =
        op->operand_source(yaml_info_parser.InputName2Id().at(pair.first));
    pir::Value output_value =
        op->result(yaml_info_parser.OutputName2Id().at(pair.second));
    if (IsInvalid(output_value) && IsInvalid(input_value)) {
      this->AddInplace(value_exec_info_.GetVarByValue(input_value),
                       value_exec_info_.GetVarByValue(output_value));
    }
  }

  InitInputsOutputsIds(op, value_exec_info_);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<pir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";
}

void CustomKernelInstruction::UpdateOutputMeta(
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<DataType>& output_dtypes) {
  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      cache_out_ptrs_.size(),
      common::errors::InvalidArgument(
          "The number of output shapes after running custom operator's "
          "InferShapeFunc is wrong, "
          "expected contains %d Tensors' shape, but actually contains %d "
          "Tensors' shape",
          cache_out_ptrs_.size(),
          output_shapes.size()));

  PADDLE_ENFORCE_EQ(
      output_dtypes.size(),
      cache_out_ptrs_.size(),
      common::errors::InvalidArgument(
          "The number of output dtypes after running custom operator's "
          "InferDtypeFunc is wrong, "
          "expected contains %d Tensors' dtype, but actually contains %d "
          "Tensors' dtype",
          cache_out_ptrs_.size(),
          output_dtypes.size()));

  for (size_t i = 0; i < cache_out_ptrs_.size(); ++i) {
    auto out_in_scope = cache_out_ptrs_.at(i);
    // update dims and dtype
    auto out_meta = phi::DenseTensorUtils::GetMutableMeta(out_in_scope);
    out_meta->dims = phi::make_ddim(output_shapes[i]);
    out_meta->dtype = output_dtypes[i];
    // need not calc_strides right now, stride is not used in static graph
    // out_meta->strides = out_meta->calc_strides(out_meta->dims);
  }
}

void CustomKernelInstruction::BuildShapeDtype() {
  input_shapes_.clear();
  input_dtypes_.clear();
  vec_input_shapes_.clear();
  vec_input_dtypes_.clear();
  for (auto in_tensor : input_ptrs_) {
    if (in_tensor) {
      input_shapes_.push_back(phi::vectorize(in_tensor->dims()));
      input_dtypes_.push_back(in_tensor->dtype());
    } else {
      input_shapes_.emplace_back();
      input_dtypes_.emplace_back();
    }
  }
  for (auto in_tensors : vec_input_ptrs_) {
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<phi::DataType> input_dtypes;
    if (in_tensors.size() > 0) {
      for (auto in_tensor : in_tensors) {
        input_shapes.push_back(phi::vectorize(in_tensor->dims()));
        input_dtypes.push_back(in_tensor->dtype());
      }
    }
    vec_input_shapes_.push_back(input_shapes);
    vec_input_dtypes_.push_back(input_dtypes);
  }
}

void CustomKernelInstruction::Run() {
  VLOG(3) << "Custom Operator: InferShape - calc output ddim.";
  BuildShapeDtype();
  std::vector<std::vector<int64_t>> output_shapes =
      RunInferShape(infershape_func_,
                    *custom_op_meta_,
                    input_shapes_,
                    input_name2id_map_,
                    vec_input_shapes_,
                    vec_input_name2id_map_,
                    custom_attrs_);
  std::vector<phi::DataType> output_dtypes =
      RunInferDtype(inferdtype_func_,
                    *custom_op_meta_,
                    input_dtypes_,
                    input_name2id_map_,
                    vec_input_dtypes_,
                    vec_input_name2id_map_,
                    custom_attrs_);
  UpdateOutputMeta(output_shapes, output_dtypes);
  for (auto& pair : this->InplaceInfo()) {
    ShareVarBuffer(pair.first, pair.second);
  }
  VLOG(6) << "Run custom op " << custom_op_name_ << " kernel.";
  kernel_func_(&custom_kernel_ctx_);
}
}  // namespace paddle::framework
