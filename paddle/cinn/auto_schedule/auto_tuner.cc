// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/auto_tuner.h"

#include <glog/logging.h>
#include <pybind11/embed.h>

#include <algorithm>
#include <memory>
#include <utility>

#include "paddle/cinn/auto_schedule/database/jsonfile_database.h"
#include "paddle/cinn/auto_schedule/measure/schedule_measurer.h"
#include "paddle/cinn/auto_schedule/measure/simple_builder.h"
#include "paddle/cinn/auto_schedule/measure/simple_runner.h"
#include "paddle/cinn/auto_schedule/task/task_creator.h"
#include "paddle/cinn/auto_schedule/task/task_registry.h"
#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/auto_schedule/task_scheduler/task_scheduler.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace auto_schedule {

AutoTuner::AutoTuner(const cinn::common::Target& target,
                     hlir::framework::Graph* graph)
    : target_(target), graph_(graph) {}

void AutoTuner::Initialize(const Config& config,
                           hlir::framework::GraphCompiler* graph_compiler) {
  // create builder, runner, and schedule measurer
  builder_ = std::make_unique<SimpleBuilder>(graph_compiler);
  runner_ = std::make_unique<SimpleRunner>(config.runner_repeat_times);
  schedule_measurer_ =
      std::make_unique<ScheduleMeasurer>(builder_.get(), runner_.get());

  // initialize database
  database_ = std::move(Database::Make(config.database_config));

  // create tasks
  TaskCreator task_creator;
  tasks_ = task_creator.CreateTuneTaskOpLevel(graph_);

  const auto& dtype_dict =
      graph_->GetAttrs<absl::flat_hash_map<std::string, cinn::common::Type>>(
          "inferdtype");
  const auto& shape_dict = graph_->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");

  op_lowerer_ = std::make_unique<hlir::framework::OpLowerer<GroupPtr>>(
      new hlir::framework::OpLowererImpl(dtype_dict, shape_dict, target_));
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  for (auto i = 0; i < tasks_.size(); ++i) {
    auto&& task = tasks_[i];
    task.Initialize(shape_dict, dtype_dict, op_lowerer_.get());
    // Register the initial ModuleExpr corresponding to the task
    task_registry->Register(task.serialized_key,
                            ir::ModuleExpr(task.GetLoweredFuncBodyExprs()));
    VLOG(3) << "Add a task, id:" << i << ", serialized_key:\n"
            << task.serialized_key;
  }

  // create task optimizers
  utils::LinearRandomEngine::StateType initial_seed =
      utils::LinearRandomEngine::GetDeviceRandomValue();
  task_optimizers_.resize(tasks_.size());
  std::transform(tasks_.begin(),
                 tasks_.end(),
                 task_optimizers_.begin(),
                 [&](TuneTask& task) {
                   return std::make_unique<TaskOptimizer>(
                       &task,
                       schedule_measurer_.get(),
                       database_.get(),
                       utils::ForkRandomState(&initial_seed));
                 });

  // create task scheduler
  task_scheduler_ = TaskScheduler::Make(
      tasks_, config.task_schedule_config, config.task_schedule_strategy);
}

void PrintResult(std::shared_ptr<hlir::framework::Graph::Group> group) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  auto nodes = group->CollectNodes();
  VLOG(3) << "Node size:" << nodes.size();
  VLOG(3) << "Group {";
  for (auto* node : nodes) {
    VLOG(3) << "  " << hlir::framework::DebugString(node);
  }
  VLOG(3) << "}";
}

void PrintResult(const FunctionGroup& functions) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  VLOG(3) << "Function size:" << functions.size();
  for (auto i = 0; i < functions.size(); ++i) {
    const ir::LoweredFunc& func = functions.at(i);
    VLOG(3) << "LoweredFunc-" << i << " detail:\n" << func;
  }
}

void PrintResult(const TuningResult& result) {
  if (!VLOG_IS_ON(3)) {
    return;
  }
  VLOG(3) << "###### Debug TuningResult ######\n";
  VLOG(3) << "Tuned SubGraph num:" << result.subgraphs.size();
  for (auto i = 0; i < result.subgraphs.size(); ++i) {
    VLOG(3) << "****** SubGraph-" << i << " Detail ******\n";
    PrintResult(result.subgraphs.at(i));
    VLOG(3) << "****** SubGraph End ******";
  }

  VLOG(3) << "Tuned FunctionGroup num:" << result.function_groups.size();
  for (auto i = 0; i < result.function_groups.size(); ++i) {
    VLOG(3) << "****** FunctionGroup-" << i << " Detail ******\n";
    PrintResult(result.function_groups.at(i));
    VLOG(3) << "****** FunctionGroup End ******";
  }
  VLOG(3) << "###### TuningResult End ######";
}

TuningResult AutoTuner::Tune(const TuningOptions& options) {
  PADDLE_ENFORCE_GT(options.num_tuning_rounds,
                    0,
                    ::common::errors::InvalidArgument(
                        "The num_tuning_rounds should be greater than 0."));

  TuningResult result;
  result.subgraphs.resize(tasks_.size());
  result.function_groups.resize(tasks_.size());
  // A task only tunes schedule now, so we populate its sub_graph
  // as default result of graph tuning, and that should be updated
  // once we support graph tuning.
  for (auto i = 0; i < tasks_.size(); ++i) {
    auto&& task = tasks_.at(i);
    result.subgraphs[i] = task.subgraph;
  }

  for (int r = 0; r < options.num_tuning_rounds; ++r) {
    VLOG(3) << "<<<<<< Round " << r << " >>>>>>";
    int run_id = -1;
    task_scheduler_->Reset();
    while ((run_id = task_scheduler_->NextTaskId()) != -1) {
      VLOG(3) << "Start tuning Task-" << run_id;
      auto* opt = task_optimizers_.at(run_id).get();
      auto function_group = opt->Optimize(options);
      VLOG(3) << "Task-" << run_id << " finished, print optimized functions:\n";
      PrintResult(function_group);
      // update the best schedules searched so far.
      result.function_groups.at(run_id) = std::move(function_group);
    }
  }

  PrintResult(result);
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn
