// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"
#include "paddle/fluid/inference/analysis/ir_pass_manager.h"

namespace paddle {
namespace inference {
namespace analysis {

void IrAnalysisPass::RunImpl(Argument* argument) {
  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  ARGUMENT_CHECK_FIELD(argument, main_program);
  ARGUMENT_CHECK_FIELD(argument, scope);

  auto* the_graph = argument->Release<Graph>(argument->k_main_graph());
  auto graph = std::unique_ptr<Graph>(the_graph);

  // Apply passes.
  IRPassManager the_ir_manager(argument);
  graph = the_ir_manager.Apply(std::move(graph));
  PADDLE_ENFORCE_GT(graph->Nodes().size(), 0);
  argument->SetIrAnalyzedProgram(
      the_ir_manager.AcquireProgram(&graph, *argument->main_program()));
  argument->SetMainGraph(std::move(graph));
}

std::string IrAnalysisPass::repr() const { return "ir-analysis-pass"; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
