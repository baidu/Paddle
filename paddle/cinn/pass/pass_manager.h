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

#include <string>

#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {

namespace detail {
class FuncPassAdaptor;
class FuncToBlockPassAdaptor;
class FuncToStmtPassAdaptor;
class FuncToExprPassAdaptor;
}  // namespace detail

template <typename PassT, typename PassAdaptorT>
class PassManager {
 public:
  explicit PassManager(bool need_converge = false)
      : need_converge_(need_converge) {}
  virtual void Run(ir::LoweredFunc func) {
    adaptor_.RunPipeline(func, passes_, need_converge_);
  }
  void AddPass(PassT* pass) { passes_.emplace_back(pass); }

 private:
  std::vector<PassT*> passes_;
  PassAdaptorT adaptor_;
  bool need_converge_;
};

using FuncPassManager = PassManager<FuncPass, detail::FuncPassAdaptor>;
using BlockPassManager = PassManager<BlockPass, detail::FuncToBlockPassAdaptor>;
using StmtPassManager = PassManager<StmtPass, detail::FuncToStmtPassAdaptor>;
using ExprPassManager = PassManager<ExprPass, detail::FuncToExprPassAdaptor>;

}  // namespace optim
}  // namespace cinn
