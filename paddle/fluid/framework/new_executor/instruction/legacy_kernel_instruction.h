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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;
class ValueExecutionInfo;

class LegacyKernelInstruction : public InstructionBase {
 public:
  LegacyKernelInstruction(size_t id,
                          const phi::Place& place,
                          ::pir::Operation* op,
                          const ValueExecutionInfo* value_exec_info);

  ~LegacyKernelInstruction();
  phi::Kernel* PhiKernel() const { return phi_kernel_; }

  const phi::InferMetaContext& InferMetaContext() const {
    return infer_meta_context_;
  }

  paddle::dialect::InferMetaInterface::Concept* InferMetaInterface() const {
    return infer_meta_interface_;
  }

  void Run() override;

  const std::string& Name() const override { return legacy_op_name_; }

  ::pir::Operation* Operation() const override { return op_; }

 private:
  std::string legacy_op_name_;

  paddle::dialect::InferMetaInterface::Concept* infer_meta_interface_{
      nullptr};  // not owned

  phi::InferMetaContext infer_meta_context_;

  paddle::framework::ExecutionContext* kernel_context_{nullptr};
  std::shared_ptr<framework::RuntimeContext> runtime_context_;
  std::shared_ptr<paddle::framework::OperatorBase> operator_base_;

  phi::Kernel* phi_kernel_{nullptr};  // not owned

  ::pir::Operation* op_{nullptr};  // not owned

  const ValueExecutionInfo* value_exec_info_;  // not owned
};

}  // namespace framework
}  // namespace paddle
