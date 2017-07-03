/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <vector>

#include "paddle/framework/attribute_reader.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/variable.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace framework {

/**
 * following is just for demo
 */
class Context {};
class CpuContext : public Context {};
class GpuContext : public Context {};

class OpDesc {};

/// OperatorBase provide base element of an Operator without any template.
class OperatorBase {
 public:
  explicit OperatorBase(const OpDesc& desc);
  virtual ~OperatorBase() {}

  /// initialize Attributes of this OP from proto message desc.attrs()
  /// you should derive this function to init the attr you need in OP.
  virtual Error InitializeAttributes(const AttributeMap& attrs) = 0;
  virtual Error Run(Scope* scope, Context* context) const = 0;

 protected:
  std::string type_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

/// Operator is the class your should derive when implement a new Operator.
template <typename DeviceContext>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OpDesc& desc) : OperatorBase(desc) {}

 private:
  /// This function will get all input and output Vars from scope and ten call
  /// Run(std::vector<Variable> inputs, std::vector<Variable> outputs, T*
  /// context)
  Error Run(Scope* scope, Context* context) const final {
    DeviceContext* dev_context = dynamic_cast<DeviceContext*>(context);
    if (dev_context == nullptr) {
      return Error("dynamic_cast devContext failed!");
    }

    std::vector<Variable*> input_vars;
    std::vector<Variable*> output_vars;

    input_vars.reserve(inputs_.size());
    for (auto& input : inputs_) {
      input_vars.push_back(scope->CreateVariable(input));
    }
    output_vars.reserve(outputs_.size());
    for (auto& input : outputs_) {
      output_vars.push_back(scope->CreateVariable(input));
    }

    return Run(input_vars, output_vars, dev_context);
  }

  // when implement an Op, your should implement this function.
  virtual Error Run(std::vector<Variable*>& inputs,
                    std::vector<Variable*>& outputs,
                    DeviceContext* context) const = 0;
};

class Net {
 public:
  Error Run(Scope* scope, Context* context) {
    for (auto& op : operators_) {
      Error err = op->Run(scope, context);
      if (!err.isOK()) {
        return err;
      }
    }
    return Error();
  }

 private:
  std::vector<std::unique_ptr<OperatorBase>> operators_;
};

}  // namespace framework
}  // namespace paddle
