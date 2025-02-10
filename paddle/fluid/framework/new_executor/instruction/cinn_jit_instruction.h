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
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/phi/core/ddim.h"

namespace pir {
class Operation;
}

namespace paddle {
namespace framework {
class Scope;

class CinnJitInstruction : public InstructionBase {
 public:
  CinnJitInstruction(size_t id,
                     const phi::Place& place,
                     ::pir::Operation* op,
                     const ValueExecutionInfo* value_exec_info);

  // TODO(Aurelius84): Only implement core interface and need implement GC and
  // Event logic.
  void Run() override;

  const std::string& Name() const override;

  ::pir::Operation* Operation() const override { return op_; }

 private:
  class FnPtrImpl;

  std::shared_ptr<FnPtrImpl> fn_ptr_impl_{nullptr};

  phi::Place place_;

  phi::DeviceContext* dev_ctx_;

  int32_t input_tensor_size;
  int32_t output_tensor_size;

  bool need_update_shape{false};
  std::vector<phi::DenseTensor*> tensor_args_;
  std::vector<phi::DDim> ir_dims_;

  // Tensors that hold the temporary spaces used by the kernel. These tensors
  // are managed by CinnJitInstruction, and not exposed to phi executor.
  std::vector<phi::DenseTensor> temp_space_tensors_;

  ::pir::Operation* op_{nullptr};  // not owned
};

}  // namespace framework
}  // namespace paddle
