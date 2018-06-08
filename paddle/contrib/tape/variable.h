// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/fluid/framework/operator.h"  // framework::kGradVarSuffix
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace tape {

class Variable;
using VariableHandle = std::shared_ptr<Variable>;

/*
 * Currently it depends on framework::Scope and framework::Variable
 * Later on will only depend on framework::Variable
 */
class Variable {
 public:
  Variable(const std::string pre_fix)
      : desc_(pre_fix + std::to_string(count())) {}

  Variable(const std::string pre_fix, bool is_grad)
      : desc_(pre_fix + (is_grad ? framework::kGradVarSuffix
                                 : std::to_string(count()))) {}

  ~Variable() { LOG(INFO) << "Deleting " << Name(); }

  void InitializeVariable() {
    LOG(INFO) << "Initialzing " << desc_.Name() << " as " << desc_.GetType();
    framework::proto::VarType::Type var_type = desc_.GetType();
    if (var_type == framework::proto::VarType::LOD_TENSOR) {
      var_.GetMutable<framework::LoDTensor>();
    } else if (var_type == framework::proto::VarType::SELECTED_ROWS) {
      var_.GetMutable<framework::SelectedRows>();
    } else {
      PADDLE_THROW("Variable type %d is not in [LOD_TENSOR, SELECTED_ROWS]",
                   var_type);
    }
  }

  VariableHandle Grad() {
    if (grad_ == nullptr) {
      grad_.reset(new Variable(desc_.Name(), true));
    }

    return grad_;
  }

  //  VariableHandle Momentum ();

  //  void init(const std::string& initializer,
  //            const framework::AttributeMap& attrs);

  // void value() {};

  const framework::VarDesc& Desc() const { return desc_; }
  framework::VarDesc* MutableDesc() { return &desc_; }

  // TODO(tonyyang-svail): No need to expose name
  std::string Name() const { return desc_.Name(); }

  framework::Variable* Var() { return &var_; }

 private:
  int count() {
    static int counter = 0;
    return counter++;
  }

  framework::VarDesc desc_;
  framework::Variable var_;

  VariableHandle grad_;
};
}
}
