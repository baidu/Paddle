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

#include <functional>

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"

namespace ir {
class Operation;
}  // namespace ir

namespace paddle {
namespace framework {
class Scope;
class Value;
class PirInterpreter;
class ValueExecutionInfo;

/// The execute semantics of while op ['output' = while_op('cond', 'input')]
/// is:
///   'output' = 'input';
///   while('cond') {
///      'cond', 'output' = body_block('output');
///  }
class WhileInstruction : public InstructionBase {
 private:
  using CheckGCEarlyHook = std::function<void(InstructionBase*)>;

 public:
  WhileInstruction(size_t id,
                   const phi::Place& place,
                   ::pir::Operation* op,
                   ValueExecutionInfo* parent_exe_info,
                   interpreter::ExecutionConfig execution_config);

  void Run() override;

  const std::string& Name() const override { return name_; }

  ::pir::Operation* Operation() const override { return op_; }

  PirInterpreter* BodyInterpreter() const { return body_inter_.get(); }

  void SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs);

  void SetInputHooks(const std::vector<PirHookFunc>& hookfuncs);

  // CheckGCEarly is designed to recycle unwanted inputs in advance, which can
  // effectively reduce peak memory in certain scenarios.
  void CheckGCEarly(const CheckGCEarlyHook& check_gc_early);

 private:
  // 'output' = 'input'
  void ShareInputsToOutputs();

  // Pass argument to body_block for execution.
  void ShareOutputsToBlockArgs();

  // Get condition value from body_block after each execution.
  void ShareConditionData();

  std::string name_{"while_instruction"};

  Variable* cond_var_;
  std::string inner_cond_;

  std::vector<Variable*> inputs_;
  std::vector<Variable*> outputs_;

  std::unique_ptr<PirInterpreter> body_inter_;
  std::set<std::string> external_input_names_;

  ::pir::Block* body_block_;

  ::pir::Operation* op_;

  CheckGCEarlyHook check_gc_early_;
};

}  // namespace framework
}  // namespace paddle
