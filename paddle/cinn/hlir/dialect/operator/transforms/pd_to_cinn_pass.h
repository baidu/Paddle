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

#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"

namespace cinn {
namespace dialect {
namespace ir {

class PdOpToCinnOpPass : public pir::Pass {
 public:
  PdOpToCinnOpPass();

  bool Initialize(pir::IrContext *context) override;

  void Run(pir::Operation *op) override;

  bool CanApplyOn(pir::Operation *op) const override;

 private:
  pir::FrozenRewritePatternSet patterns_;
};

void PdOp2CinnOpConverter(::pir::Program *program);

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
