// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/feed_hook.h"
#include "paddle/fluid/framework/new_executor/feed_fetch_utils.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/framework/new_executor/program_interpreter.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/inplace_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_bool(enable_pir_in_executor);
COMMON_DECLARE_bool(enable_pir_api);
COMMON_DECLARE_bool(pir_apply_inplace_pass);
COMMON_DECLARE_string(enable_custom_engine);

namespace paddle::framework {
StandaloneExecutor::StandaloneExecutor(const phi::Place& place,
                                       const interpreter::Plan& plan,
                                       Scope* scope)
    : place_(place),
      plan_(plan),
      interpretercores_(),
      scope_(scope),
      micro_batch_scopes_(),
      fetch_var_names_(),
      fetch_list_(),
      vec_force_events_to_wait_() {
  int64_t micro_batch_num = plan_.MicroBatchNum();
  vec_force_events_to_wait_.resize(micro_batch_num);
  for (int64_t i = 0; i < micro_batch_num; ++i) {
    micro_batch_scopes_.emplace_back(&scope->NewScope());
  }
  std::stringstream ss;
  ss << "Create " << micro_batch_num << " micro_batch_scopes for scope "
     << scope_ << " : ";
  for (Scope* scope : micro_batch_scopes_) {
    ss << scope << ", ";
  }
  VLOG(6) << ss.str();

  const auto& jobs = plan_.JobList();
  for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
    const auto& job = jobs[job_idx];
    const std::string& job_type = job->Type();
    std::shared_ptr<ProgramDesc> program = nullptr;
    std::shared_ptr<::pir::Program> ir_program = nullptr;
    if (FLAGS_enable_pir_api || FLAGS_enable_pir_in_executor) {  // NOLINT
      ir_program = plan_.IrProgram(job_type);
      RunFeedHooks(*ir_program, *scope);
    } else {
      // NOTE (liuchenghao): std::make_shared will duplicate ProgramDesc object,
      // maybe std::make_unique is better?
      program = std::make_shared<ProgramDesc>(*(plan_.Program(job_type)));
    }

    int64_t micro_batch_id = job->MicroBatchId();
    PADDLE_ENFORCE(micro_batch_id >= 0 && micro_batch_id < micro_batch_num,
                   common::errors::Unavailable(
                       "The micro batch id (%lld) out of bound, "
                       "which should be in the range of [0, %lld].",
                       micro_batch_id,
                       micro_batch_num));

    if (!FLAGS_enable_pir_api && !FLAGS_enable_pir_in_executor) {
      SetColAttrForFeedFetchOps(program, micro_batch_num, micro_batch_id);
    }

    interpreter::ExecutionConfig execution_config;
    execution_config.create_local_scope = false;
    execution_config.skip_gc_vars = job->SkipGcVars();
    execution_config.force_sync_ops =
        interpreter::GetForceSyncOps(micro_batch_id, job_type);

    // TODO(phlrain) we only support cpu for now
    if (FLAGS_enable_pir_in_executor) {
      std::shared_ptr<::pir::Program> base_program = ir_program;
      auto block = base_program->block();
      for (auto it = block->begin(); it != block->end(); ++it) {
        if (it->isa<paddle::dialect::FetchOp>()) {
          size_t index =
              it->attributes().at("col").dyn_cast<pir::Int32Attribute>().data();

          if (fetch_var_names_.size() < index + 1) {
            fetch_var_names_.resize(index + 1);
          }

          fetch_var_names_[index] = it->attributes()
                                        .at("name")
                                        .dyn_cast<pir::StrAttribute>()
                                        .AsString() +
                                    "@fetch";
          job->SetFetchVarName(fetch_var_names_[index]);
        }
      }
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      if (!FLAGS_enable_custom_engine.empty()) {
        std::string custom_engine_translate_pass = FLAGS_enable_custom_engine;
        std::istringstream ss(custom_engine_translate_pass);
        std::string pass;
        std::vector<std::string> passes;

        while (std::getline(ss, pass, ',')) {
          passes.push_back(pass);
          VLOG(4) << "Add CustomEngine pass : " << pass;
        }

        ::pir::PassManager pass_pm(::pir::IrContext::Instance(), 3);
        for (std::string custom_pass : passes) {
          pass_pm.AddPass(pir::PassRegistry::Instance().Get(custom_pass));
          pass_pm.Run(base_program.get());
        }
      }
#endif
      auto kernel_program =
          paddle::dialect::PdOpLowerToKernelPass(base_program.get(), place);
      std::shared_ptr<pir::Program> shared_program = std::move(kernel_program);
      plan_.SetIrProgram("job_" + std::to_string(job_idx), shared_program);

      if (FLAGS_pir_apply_inplace_pass) {
        pir::PassManager pm(pir::IrContext::Instance(), 3);
        pm.AddPass(pir::CreateInplacePass());
        pm.Run(shared_program.get());
      }

      interpretercores_.emplace_back(
          std::make_shared<InterpreterCore>(place_,
                                            job->FetchVarNames(),
                                            shared_program->block(),
                                            micro_batch_scopes_[micro_batch_id],
                                            execution_config));
      // Note(lizhiyu): Add manual event info
      auto pir_inter = const_cast<PirInterpreter*>(
          static_cast<const PirInterpreter*>(interpretercores_.back()->Impl()));
      pir_inter->SetForceEventsToWaitInfo(
          &(vec_force_events_to_wait_[micro_batch_id]));
    } else {
      interpretercores_.emplace_back(
          std::make_shared<InterpreterCore>(place_,
                                            program->Block(0),
                                            micro_batch_scopes_[micro_batch_id],
                                            execution_config));
      interpretercores_.back()->SetCopyProgram(program);

      // Note(lizhiyu): Add manual event info
      auto prog_inter = const_cast<ProgramInterpreter*>(
          static_cast<const ProgramInterpreter*>(
              interpretercores_.back()->Impl()));
      prog_inter->SetForceEventsToWaitInfo(
          &(vec_force_events_to_wait_[micro_batch_id]));

      // NOTE(lizhiyu): Now we only check backward subprogram. After static
      // build strategy is completely, we should
      //                check all the program in the PP strategy.
      if (job_type == "backward" && jobs.size() > 1) {
        PADDLE_ENFORCE_EQ(static_cast<const ProgramInterpreter*>(
                              interpretercores_.back()->Impl())
                              ->IsStaticBuild(),
                          true,
                          common::errors::InvalidArgument(
                              "When using pipeline strategy in auto "
                              "parallelism with new executor, "
                              "the backward subprogram must be built in real "
                              "static build mode, but it can not "
                              "be statically built in this case. You can "
                              "enable 'GLOG_v=1' to obtain log information."));
      }
    }
  }
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names,
    const bool enable_job_schedule_profiler) {
  phi::RecordEvent record_event(
      "StandaloneExecutor::run", phi::TracerEventType::UserDefined, 1);

  const auto& jobs = plan_.JobList();

  std::map<std::string, size_t> type_to_first_id;
  if (!is_interpretercore_build_result_shared_) {
    type_to_first_id[jobs[0]->Type()] = 0;
    for (size_t job_idx = 1; job_idx < jobs.size(); ++job_idx) {
      interpretercores_[job_idx]->ShareWorkQueueFrom(interpretercores_[0]);
      // TODO(Ruibiao): Share other build result, e.g., kernel choosing, data
      // transfer, op dependency, thread scheduling, GC, event analyzer, and so
      // on.
      if (type_to_first_id.count(jobs[job_idx]->Type()) == 0) {
        type_to_first_id[jobs[job_idx]->Type()] = job_idx;
      }
    }
    is_interpretercore_build_result_shared_ = true;
  }

  std::vector<std::vector<phi::DenseTensor>> splited_feeds;
  if (FLAGS_enable_pir_in_executor) {
    SplitFeedTensors(feed_names, plan_.MicroBatchNum(), scope_, &splited_feeds);
  }

  fetch_list_.resize(plan_.MicroBatchNum());
  for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
    const auto& job = jobs[job_idx];
    const std::string& job_type = job->Type();
    phi::RecordEvent record_event(
        job_type + "-" + std::to_string(job->MicroBatchId()),
        phi::TracerEventType::UserDefined,
        1);

    VLOG(6) << "Run job (" << job_idx << "), type = " << job_type
            << ", micro_batch_id =" << job->MicroBatchId();

    // NOTE(sonder): Share build results don't work for new IR now.
    if (type_to_first_id.count(job_type) != 0 &&
        !FLAGS_enable_pir_in_executor) {
      interpretercores_[job_idx]->ShareBuildResultsFrom(
          interpretercores_[type_to_first_id[job_type]]);
    }

    if (FLAGS_enable_pir_in_executor) {
      interpretercores_[job_idx]->Run(feed_names,
                                      splited_feeds[job->MicroBatchId()],
                                      /*need_fetch = */ false,
                                      /*enable_job_schedule_profiler = */
                                      enable_job_schedule_profiler);

      FetchTensors(job->FetchVarNames(),
                   fetch_var_names_,
                   job->MicroBatchId(),
                   micro_batch_scopes_[job->MicroBatchId()],
                   &fetch_list_);
    } else {
      if (jobs.size() > 1 && job_type != "forward") {
        const std::vector<std::string> tmp_feed_names = {};
        interpretercores_[job_idx]->Run(tmp_feed_names,
                                        /*need_fetch = */ false,
                                        /*enable_job_schedule_profiler = */
                                        enable_job_schedule_profiler);
      } else {
        interpretercores_[job_idx]->Run(feed_names,
                                        /*need_fetch = */ false,
                                        /*enable_job_schedule_profiler = */
                                        enable_job_schedule_profiler);
      }
    }
  }

  // record each job's run time
#if defined(PADDLE_WITH_CUDA)
  if (enable_job_schedule_profiler) {
    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
      const auto& job = jobs[job_idx];
      const std::string& job_type = job->Type();
      double start_time, end_time;
      std::tie(start_time, end_time) =
          interpretercores_[job_idx]->InterpreterRunTime();

      // Note(sonder): Used to record the runtime of each job in order to
      // generate a parallel pipeline timeline. Job runtime information can be
      // extracted from the logs using the scripts "profiler_helper_static.py".
      // Do not modify, as it may affect the results of regular expression
      // matching.
      VLOG(0) << "Profiler Info: Job (" << job->MicroBatchId()
              << "), type = " << job_type
              << ", micro_batch_id = " << job->MicroBatchId()
              << ", job_start_time = " << std::to_string(start_time)
              << ", job_end_time = " << std::to_string(end_time);
    }
  }
#endif

  // return Fetch Tensors
  if (FLAGS_enable_pir_in_executor) {
    framework::FetchList fetch_res;
    MergeFetchTensors(fetch_list_, plan_.MicroBatchNum(), &fetch_res);
    return fetch_res;
  } else {
    auto* fetch_var = scope_->FindVar(interpreter::kFetchVarName);
    if (fetch_var) {
      return std::move(*fetch_var->GetMutable<framework::FetchList>());
    } else {
      return {};
    }
  }
}

std::shared_ptr<framework::ProgramDesc> StandaloneExecutor::RunProfile(
    const std::vector<std::string>& feed_names) {
  phi::RecordEvent record_event(
      "StandaloneExecutor::run_profile", phi::TracerEventType::UserDefined, 1);

  // in profiling run, there can be one and only one job ("default")
  interpretercores_[0]->Run(feed_names,
                            /*need_fetch = */ false,
                            /*enable_job_schedule_profiler = */ false,
                            /*enable_op_profiling = */ true);

  // Don't return program desc directly, instead, return a copy of it since we
  // don't know how the program desc will be further processed in Python side.
  // If we return a raw shared_ptr, the program desc will be easily altered
  // externally, result in unexpected behavior during the next profiling run.
  std::shared_ptr<framework::ProgramDesc> copy_desc =
      std::make_shared<framework::ProgramDesc>(
          *(interpretercores_[0]->GetMutableCopyProgram()));

  return copy_desc;
}

}  // namespace paddle::framework
