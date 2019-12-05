// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

static constexpr char kStepBlock[] = "sub_block";
static constexpr char kCondition[] = "Condition";
static constexpr char kStepScopes[] = "StepScopes";
static constexpr char kX[] = "X";
static constexpr char kXGRAD[] = "X@GRAD";
static constexpr char kOutputs[] = "Out";
static constexpr char kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

void PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
    const framework::ProgramDesc &program, int block_id,
    const std::vector<std::unique_ptr<framework::OperatorBase>> &all_ops);

void PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
    const framework::ProgramDesc &program,
    const std::vector<framework::OperatorBase *> &while_ops,
    const std::vector<framework::OperatorBase *> &while_grad_ops);

inline bool GetCondData(const framework::LoDTensor &cond) {
  if (platform::is_cpu_place(cond.place())) {
    return cond.data<bool>()[0];
  }
  // when platform::is_gpu_place(cond.place()) is true
  std::unique_ptr<framework::LoDTensor> cpu_cond{new framework::LoDTensor()};
#ifdef PADDLE_WITH_CUDA
  framework::TensorCopySync(cond, platform::CPUPlace(), cpu_cond.get());
#else
  PADDLE_THROW(
      "This version of PaddlePaddle doen NOT support GPU but got GPU tensor "
      "Cond in WhileOp. Please compile WITH_GPU option");
#endif
  return cpu_cond->data<bool>()[0];
}

}  // namespace operators
}  // namespace paddle
