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

class OneDNNPhiKernelInstruction : public InstructionBase {
 public:
  OneDNNPhiKernelInstruction(size_t id,
                             const phi::Place& place,
                             ::pir::Operation* op,
                             const ValueExecutionInfo* value_exec_info);

  ~OneDNNPhiKernelInstruction();

  phi::Kernel* PhiKernel() const { return phi_kernel_; }

  const phi::KernelContext& KernelContext() const { return kernel_context_; }

  const phi::InferMetaContext& InferMetaContext() const {
    return infer_meta_context_;
  }

  paddle::dialect::InferMetaInterface::Concept* InferMetaInterface() const {
    return infer_meta_interface_;
  }

  ::pir::Operation* Operation() const override { return op_; }

  void Run() override;

  const std::string& Name() const override { return phi_op_name_; }

 protected:
  paddle::dialect::InferMetaInterface::Concept* infer_meta_interface_{
      nullptr};  // not owned

  phi::InferMetaContext infer_meta_context_;

  phi::KernelContext kernel_context_;

  phi::Kernel* phi_kernel_{nullptr};  // not owned

  std::string phi_op_name_;

  ::pir::Operation* op_{nullptr};  // not owned

  const ValueExecutionInfo* value_exec_info_;  // not owned

  std::set<int> data_format_tensors_{};
  std::set<int> skip_format_tensors_{};
  phi::DataLayout input_layout_{phi::DataLayout::kAnyLayout};
  std::map<std::string, phi::Attribute> extra_attr_{};
  std::map<std::string, phi::Attribute> ctx_attr_{};
  std::map<std::string, std::vector<std::string>> inputs_{};
  std::map<std::string, std::vector<std::string>> outputs_{};
  std::string kernel_name_;
  phi::KernelKey kernel_key_;
};
}  // namespace framework
}  // namespace paddle
