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

#pragma once
#include "paddle/fluid/custom_engine/custom_engine_ext.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
// #include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
// #include "paddle/phi/core/platform/device_context.h"
namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;
class ValueExecutionInfo;

class CustomEngineInstruction : public InstructionBase {
 public:
  typedef void (*DeletePtr)(void*);
  CustomEngineInstruction(size_t id,
                          const phi::Place& place,
                          ::pir::Operation* op,
                          const ValueExecutionInfo* value_exec_info);

  ::pir::Operation* Operation() const override { return op_; }
  const ValueExecutionInfo* GetValueExecutionInfo() const {
    return value_exec_info_;
  }

  void Run() override;

  void SetName(const std::string& name) { op_name_ = name; }
  const std::string& Name() const override { return op_name_; }

  void SetCustomEngine(void* custom_engine) { custom_engine_ = custom_engine; }

  void* CustomEngine() { return custom_engine_; }

  void SetCustomEngineDeleter(DeletePtr custom_engine_deleter) {
    custom_engine_deleter_ = custom_engine_deleter;
  }

  ~CustomEngineInstruction() override {
    if (custom_engine_deleter_) {
      custom_engine_deleter_(custom_engine_);
    }
  }

 private:
  phi::Place place_;
  std::string op_name_ = "custom_engine.base_op";
  ::pir::Operation* op_{nullptr};  // not owned

  C_CustomEngineInterface* interface_;
  const ValueExecutionInfo* value_exec_info_;  // not owned
  void* custom_engine_;                        // not owned
  DeletePtr custom_engine_deleter_{nullptr};   // not owned
};
}  // namespace framework
}  // namespace paddle
