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

#include "test/cpp/pir/custom_engine/custom_engine_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

namespace paddle {
namespace dialect {

const char *FakeEngineOp::attributes_name[2] = {"input_names", "output_names"};

OpInfoTuple FakeEngineOp::GetOpInfo() {
  std::vector<paddle::dialect::OpInputInfo> inputs = {
      OpInputInfo("x",
                  "pir::VectorType<paddle::dialect::DenseTensorType>",
                  false,
                  false,
                  false,
                  false)};

  std::vector<paddle::dialect::OpAttributeInfo> attributes = {
      paddle::dialect::OpAttributeInfo(
          "input_names", "pir::ArrayAttribute", ""),
      paddle::dialect::OpAttributeInfo(
          "output_names", "pir::ArrayAttribute", "")};

  std::vector<paddle::dialect::OpOutputInfo> outputs = {
      OpOutputInfo("out",
                   "pir::VectorType<paddle::dialect::DenseTensorType>",
                   false,
                   false)};

  paddle::dialect::OpRunTimeInfo run_time_info =
      OpRunTimeInfo("", {}, "", {}, {}, {}, {}, {});

  return std::make_tuple(
      inputs, attributes, outputs, run_time_info, "fake_engine");
}

#define ADD_VEC_ATTRIBUTE(type, name)                                   \
  std::vector<pir::Attribute> name##_tmp;                               \
  name##_tmp.reserve(name.size());                                      \
  for (const auto &v : name) {                                          \
    name##_tmp.push_back(type::get(pir::IrContext::Instance(), v));     \
  }                                                                     \
  pir::Attribute attr_##name =                                          \
      pir::ArrayAttribute::get(pir::IrContext::Instance(), name##_tmp); \
  argument.AddAttribute(#name, attr_##name)

#define VERIFY_ATTRIBUTE(type, name)                              \
  PADDLE_ENFORCE_GT(                                              \
      attributes.count(#name),                                    \
      0,                                                          \
      common::errors::InvalidArgument(#name " does not exist.")); \
  PADDLE_ENFORCE_EQ(attributes.at(#name).isa<type>(),             \
                    true,                                         \
                    common::errors::InvalidArgument(              \
                        "Type of attribute: " #name " is not " #type))

void FakeEngineOp::Build(pir::Builder &builder,             // NOLINT
                         pir::OperationArgument &argument,  // NOLINT
                         pir::Value x,
                         std::vector<std::string> input_names,
                         std::vector<std::string> output_names,
                         std::vector<std::vector<int64_t>> outputs_shape,
                         std::vector<phi::DataType> outputs_dtype) {
  VLOG(4) << "Start build FakeEngineOp";

  VLOG(4) << "Builder construction inputs";
  std::vector<pir::Value> argument_inputs = {x};
  argument.AddInputs(argument_inputs);

  VLOG(4) << "Builder construction attributes";

  ADD_VEC_ATTRIBUTE(pir::StrAttribute, input_names);
  ADD_VEC_ATTRIBUTE(pir::StrAttribute, output_names);

  VLOG(4) << "Builder construction outputs";

  std::vector<pir::Type> argument_outputs;
  std::vector<pir::Type> out_types;
  for (size_t i = 0; i < static_cast<size_t>(outputs_shape.size()); i++) {
    if (outputs_dtype[i] == phi::DataType::UNDEFINED) {
      out_types.push_back(pir::Type());
    } else {
      out_types.push_back(pir::DenseTensorType::get(
          pir::IrContext::Instance(),
          TransToIrDataType(outputs_dtype[i]),
          phi::DDim(outputs_shape[i].data(), outputs_shape[i].size()),
          phi::DataLayout::kNCHW,
          phi::LoD(),
          0));
    }
  }
  pir::Type out_vector_type =
      pir::VectorType::get(pir::IrContext::Instance(), out_types);
  argument_outputs.push_back(out_vector_type);

  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  argument.AddRegion(nullptr);
  ::pir::PassStopGradientsDefaultly(argument);
}

void FakeEngineOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: "
             "FakeEngineOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(input_size,
                      1,
                      common::errors::InvalidArgument(
                          "The size of inputs must be equal to 1."));
    PADDLE_ENFORCE_EQ((*this)->operand_source(0).type().isa<pir::VectorType>(),
                      true,
                      common::errors::InvalidArgument(
                          "Type validation failed for the 0th input, got %s.",
                          (*this)->operand_source(0).type()));
    if (auto vec_type =
            (*this)->operand_source(0).type().dyn_cast<pir::VectorType>()) {
      for (size_t i = 0; i < vec_type.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            vec_type[i].isa<pir::DenseTensorType>(),
            true,
            common::errors::InvalidArgument(
                "Type validation failed for the 0th input, got %s.",
                (*this)->operand_source(0).type()));
      }
    }
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto &attributes = this->attributes();
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, input_names);
    VERIFY_ATTRIBUTE(pir::ArrayAttribute, output_names);
  }

  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(output_size,
                      1,
                      common::errors::InvalidArgument(
                          "The size of outputs must be equal to 1."));
    auto output_type = (*this)->result(0).type();

    PADDLE_ENFORCE_EQ(output_type.isa<pir::VectorType>(),
                      true,
                      common::errors::InvalidArgument(
                          "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: FakeEngineOp.";
}

pir::Block *FakeEngineOp::block() {
  pir::Region &region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return &region.front();
}

pir::Block *FakeEngineOp::block() const {
  pir::Region &region = (*this)->region(0);
  PADDLE_ENFORCE_EQ(region.empty(),
                    false,
                    ::common::errors::Unavailable(
                        "Required CustomEngineOp's region must not be empty."));
  return &region.front();
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::FakeEngineOp)
