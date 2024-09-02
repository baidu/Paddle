// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/jit/engine/pir_interpreter_engine.h"
#include "paddle/fluid/jit/engine/interpreter_engine.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
// #include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace jit {

PirInterpreterEngine::PirInterpreterEngine(
    const std::shared_ptr<PirFunctionInfo> &info,
    const std::shared_ptr<VariableMap> &params_dict,
    const phi::Place &place,
    const std::shared_ptr<pir::Program> &prog)
    : info_(info), params_dict_(params_dict), place_(place), prog_(prog) {
  info_->RemoveFeedFetch();
  LOG(INFO) << "1";
  PADDLE_ENFORCE_GT(
      static_cast<int64_t>(info_->Program()->block()->size()),
      0,
      common::errors::PreconditionNotMet(
          "There is no operator in ProgramDesc."));
  LOG(INFO) << "2";
  utils::ShareParamsIntoScope(info_->ParamNames(), params_dict_, &scope_);
  LOG(INFO) << "3";
//   VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
  LOG(INFO) << "3.1";
  CreateInterpreterCore();
  LOG(INFO) << "4";
}

// need modify
void PirInterpreterEngine::CreateInterpreterCore() {

// #ifdef PADDLE_WITH_DNNL
//   auto onednn_pass =
//       framework::ir::PassRegistry::Instance().Get("onednn_placement_pass");
//   onednn_pass->Set("mkldnn_enabled_op_types",
//                    new std::unordered_set<std::string>({}));
//   onednn_pass->Apply(&graph);
// #endif

  LOG(INFO) << "31";

  framework::interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_jit = true;
  LOG(INFO) << "32";

  auto in_names = info_->InputArgNames();
  auto out_names = info_->OutputArgNames();
  execution_config.skip_gc_vars.insert(in_names.begin(), in_names.end());
  execution_config.skip_gc_vars.insert(out_names.begin(), out_names.end());
  LOG(INFO) << "33";

  inner_interpreter_ = std::make_shared<PirInterpreter>(
      place_, out_names, prog_->block(), &scope_, execution_config);
  LOG(INFO) << "34";
}

std::vector<Tensor> PirInterpreterEngine::operator()(
    const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> PirInterpreterEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);

  // the latter can be moved to python side.
  auto &feed_names = info_->InputArgNames();
  LOG(INFO) << "51";
  paddle::framework::FetchList outs = inner_interpreter_->Run(feed_names);
  LOG(INFO) << "52";

  std::vector<DenseTensor> outputs;
  utils::FetchOuts(info_->OutputArgNames(), scope_, &outputs);
  scope_.DropKids();

  return outputs;
}

const std::shared_ptr<PirFunctionInfo> &PirInterpreterEngine::Info() const {
  return info_;
}

std::unique_ptr<BaseEngine> PirInterpreterEngine::Clone(void *stream) {
  auto *x = new PirInterpreterEngine(info_, params_dict_, place_, prog_);
  return std::unique_ptr<BaseEngine>(x);
}

}  // namespace jit
}  // namespace paddle
