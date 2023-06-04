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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"

namespace paddle {
namespace translator {

struct VariableDefiningInfo {
  VariableDefiningInfo(ir::OpResult value,
                       bool generated_by_vector = false,
                       int idx_in_vector = -1)
      : value(value),
        generated_by_vector(generated_by_vector),
        idx_in_vector(idx_in_vector) {}
  VariableDefiningInfo() {}

  ir::OpResult value;

  bool generated_by_vector =
      false;  // true if target variable is generated by Vector<Tensor>
  int idx_in_vector =
      -1;  // positive if target variable is generated by Vector<Tensor>
};

using TranslationContext =
    std::unordered_map<std::string, VariableDefiningInfo>;

class ProgramTranslator {
  using ProgramDesc = ::paddle::framework::ProgramDesc;
  using BlockDesc = ::paddle::framework::BlockDesc;

 public:
  explicit ProgramTranslator(const ProgramDesc* legacy_program,
                             ir::Program* program);

  void Translate();

 private:
  const ProgramDesc* legacy_program;
  ir::Program* program;
  TranslationContext param_map;
  ir::IrContext* ctx;

  /// In the legacy program desc, there are two special named varibales:
  /// 1. "feed", the input variable of feed op
  /// 2. "fetch", the output variable of fetch op
  /// However, new feed has no input and new fetch has no output
  /// So we don't handle these two vairables when
  /// `ExtractParameterFromSingleBlock`
  static const std::unordered_set<std::string> no_cast_var_names;

  void ExtractParameterFromSingleBlock(const BlockDesc& block);
  void InsertOperationToSingleBlock(const BlockDesc& block);
};

}  // namespace translator
}  // namespace paddle
