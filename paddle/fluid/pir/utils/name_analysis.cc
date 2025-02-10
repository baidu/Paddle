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

#include "paddle/fluid/pir/utils/name_analysis.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
namespace pir {
namespace utils {
namespace name_analysis {

pir::Value GetOutputValueByName(const pir::Program &program,
                                const std::string &name) {
  auto &block = *program.block();
  pir::StrAttribute name_attr =
      pir::StrAttribute::get(pir::IrContext::Instance(), name);
  pir::Value value;
  for (auto &op : block) {
    if (op.isa<pir::ShadowOutputOp>()) {
      if (op.attribute("output_name") == name_attr) {
        if (value) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "More than one shadow output named with %s found.", name));
        }
        value = op.operand_source(0);
      }
    } else if (op.isa<paddle::dialect::FeedOp>() ||
               op.isa<paddle::dialect::FetchOp>()) {
      if (op.attribute("name") == name_attr) {
        if (value) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "More than one feed/fetch named with %s found.", name));
        }
        value = op.result(0);
      }
    }
  }
  return value;
}

pir::Value GetParameterValueByName(const pir::Program &program,
                                   const std::string &name) {
  auto &block = *program.block();
  pir::StrAttribute name_attr =
      pir::StrAttribute::get(pir::IrContext::Instance(), name);
  pir::Value value;
  for (auto &op : block) {
    if (op.isa<pir::ParameterOp>()) {
      if (op.attribute("parameter_name") == name_attr) {
        if (value) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "More than one parameter named with %s found.", name));
        }
        value = op.result(0);
      }
    }
  }
  return value;
}

std::unordered_map<std::string, pir::Value> GetAllParameterValues(
    const pir::Program &program) {
  auto &block = *program.block();
  std::unordered_map<std::string, pir::Value> values;
  for (auto &op : block) {
    if (op.isa<pir::ParameterOp>()) {
      values[op.attribute("parameter_name")
                 .dyn_cast<pir::StrAttribute>()
                 .AsString()] = op.result(0);
    }
  }
  return values;
}

void SetValueName(pir::Value value, const std::string name) {
  pir::Operation *define_op = value.defining_op();
  if (define_op->isa<pir::ParameterOp>()) {
    define_op->set_attribute(
        "parameter_name",
        pir::StrAttribute::get(pir::IrContext::Instance(), name));
  } else if (define_op->isa<paddle::dialect::DataOp>()) {
    define_op->set_attribute(
        "name", pir::StrAttribute::get(pir::IrContext::Instance(), name));
  } else if (auto block_arg = value.dyn_cast<pir::BlockArgument>()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Can Not set name for BlockArgument! "));
  } else if (value.first_use()) {
    auto nextOp = value.first_use().owner();
    if (nextOp->isa<::pir::ShadowOutputOp>()) {
      nextOp->set_attribute(
          "output_name",
          pir::StrAttribute::get(pir::IrContext::Instance(), name));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Currently, we can only set name of Value which is "
          "shadowoutput "));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only set name of Value that "
        "is persistable"));
  }
}

std::map<std::string, std::string> RenameValue(Value value,
                                               const std::string &new_name,
                                               Block *block) {
  std::map<std::string, std::string> rename_mapping;
  VLOG(5) << "Starting to rename value to " << new_name;
  // Handle kwarg
  for (auto [name, kwarg] : block->kwargs()) {
    if (kwarg == value) {
      if (name == new_name) {
        break;
      }
      Value new_value;
      if (block->kwargs().count(new_name)) {
        new_value = block->kwargs().at(new_name);
      } else {
        new_value = block->AddKwarg(new_name, value.type());
      }
      value.ReplaceAllUsesWith(new_value);
      block->EraseKwarg(name);
      value = new_value;
      VLOG(5) << "Value is kwarg, rename it from " << name << " to "
              << new_name;
      rename_mapping.insert({name, new_name});
      break;
    }
  }

  // Handle inputs
  auto defining_op = value.defining_op();
  if (defining_op) {
    // Handle DataOp
    if (defining_op->isa<paddle::dialect::DataOp>()) {
      auto name = defining_op->attribute<StrAttribute>("name").AsString();
      if (name != new_name) {
        defining_op->set_attribute(
            "name", StrAttribute::get(pir::IrContext::Instance(), new_name));
        VLOG(5) << "Value is defined by DataOp, rename it from " << name
                << " to " << new_name;
        rename_mapping.insert({name, new_name});
      }
    }

    // Handle ParameterOp
    if (defining_op->isa<pir::ParameterOp>()) {
      auto name =
          defining_op->attribute<StrAttribute>("parameter_name").AsString();
      if (name != new_name) {
        defining_op->set_attribute(
            "parameter_name",
            StrAttribute::get(pir::IrContext::Instance(), new_name));
        VLOG(5) << "Value is defined by ParameterOp, rename it from " << name
                << " to " << new_name;
        rename_mapping.insert({name, new_name});
      }
    }

    // Handle ConstantTensorOp
    if (defining_op->isa<::pir::ConstantTensorOp>()) {
      auto name =
          defining_op->attribute<StrAttribute>("tensor_name").AsString();
      if (name != new_name) {
        defining_op->set_attribute(
            "tensor_name",
            StrAttribute::get(pir::IrContext::Instance(), new_name));
        VLOG(5) << "Value is defined by ConstantTensorOp, rename it from "
                << name << " to " << new_name;
        rename_mapping.insert({name, new_name});
      }
    }
  }

  // Handle outputs
  for (auto iter = value.use_begin(); iter != value.use_end(); ++iter) {
    auto user_op = iter->owner();
    if (user_op->isa<::pir::ShadowOutputOp>()) {
      // Handle ShadowOutputOp
      auto name = user_op->attribute<StrAttribute>("output_name").AsString();
      if (name == new_name) {
        continue;
      }
      user_op->set_attribute(
          "output_name",
          StrAttribute::get(pir::IrContext::Instance(), new_name));
      VLOG(5) << "Value is used by ShadowOutputOp, rename it from " << name
              << " to " << new_name;
      rename_mapping.insert({name, new_name});
    } else if (user_op->isa<::pir::SetParameterOp>()) {
      // Handle SetParameterOp
      auto name = user_op->attribute<StrAttribute>("parameter_name").AsString();
      if (name == new_name) {
        continue;
      }
      user_op->set_attribute(
          "parameter_name",
          StrAttribute::get(pir::IrContext::Instance(), new_name));
      VLOG(5) << "Value is used by SetParameterOp, rename it from " << name
              << " to " << new_name;
      rename_mapping.insert({name, new_name});
    }
  }
  return rename_mapping;
}

pir::Value GetCorrespondingInplaceValue(pir::Operation *op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto op_name = op->name();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));
  for (size_t i = 0; i < op->num_results(); ++i) {
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string &inplace_name = yaml_parser.InplaceName(value_name);
      pir::Value inplace_value =
          op->operand_source(yaml_parser.InputName2Id().at(inplace_name));
      return inplace_value;
    }
  }
  return nullptr;
}

std::optional<std::string> GetValueInputName(pir::Value value) {
  std::optional<std::string> name;
  if (auto block_arg = value.dyn_cast<pir::BlockArgument>()) {
    if (block_arg.is_kwarg()) {
      name = block_arg.keyword();
    } else {
      name = "arg_" + std::to_string(block_arg.index());
    }
  } else if (auto param_op = value.defining_op<::pir::ParameterOp>()) {
    name = param_op.param_name();
  } else if (auto data_op = value.defining_op<paddle::dialect::DataOp>()) {
    name = data_op.attribute<pir::StrAttribute>("name").AsString();
  } else if (auto constant_op = value.defining_op<::pir::ConstantTensorOp>()) {
    name = constant_op.tensor_name();
  } else if (value.defining_op()->HasTrait<paddle::dialect::InplaceTrait>()) {
    auto defining_op = value.defining_op();
    auto value = GetCorrespondingInplaceValue(defining_op);
    pir::Operation *owner = value.defining_op();
    if (owner->isa<pir::ParameterOp>()) {
      pir::ParameterOp parameter_op = owner->dyn_cast<pir::ParameterOp>();
      name = parameter_op.param_name();
    }
  }
  return name;
}

std::vector<std::string> GetValueOutputNames(pir::Value value) {
  std::vector<std::string> names;
  for (auto iter = value.use_begin(); iter != value.use_end(); ++iter) {
    if (iter->owner()->isa<::pir::ShadowOutputOp>()) {
      names.push_back(iter->owner()
                          ->attribute<pir::StrAttribute>("output_name")
                          .AsString());
    } else if (iter->owner()->isa<::pir::SetParameterOp>()) {
      names.push_back(iter->owner()
                          ->attribute<pir::StrAttribute>("parameter_name")
                          .AsString());
    }
  }
  return names;
}

std::vector<std::string> GetValueAllNames(pir::Value value) {
  std::vector<std::string> names;
  std::optional<std::string> input_name = GetValueInputName(value);
  if (input_name.has_value()) {
    names.push_back(input_name.value());
  }

  std::vector<std::string> output_name = GetValueOutputNames(value);
  for (auto &name : output_name) {
    names.push_back(name);
  }

  return names;
}

std::optional<std::string> TryGetValueFirstName(pir::Value value) {
  std::optional<std::string> name;

  auto names = GetValueAllNames(value);
  if (!names.empty()) {
    return names[0];
  }

  return name;
}

std::string GetValueFirstName(pir::Value value) {
  auto name = TryGetValueFirstName(value);
  PADDLE_ENFORCE(name.has_value(),
                 common::errors::InvalidArgument(
                     "Currently, we can only get name of Value from "
                     "DataOp/ParameterOp/BlockArgument/ConstantTensorOp/"
                     "SetParameterOp and ShadowOutputOp."));

  return name.value();
}

}  // namespace name_analysis
}  // namespace utils
}  // namespace pir
