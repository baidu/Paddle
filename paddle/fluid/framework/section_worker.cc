/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include <cfloat>

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle::framework {

class TrainerDesc;

uint64_t SectionWorker::batch_id_(0);

void SectionWorker::Initialize(const TrainerDesc &desc) {
  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);
  program_ = std::make_unique<ProgramDesc>(
      desc.section_param().section_config().program_desc());
  for (auto &op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }

  for (auto &op : ops_) {
    // cache the op type during the init part
    // reduce unnecessary op visit during running
    int op_role = op->Attr<int>("op_role");
    if ((op_role == static_cast<int>(OpRole::kForward)) ||
        (op_role == (static_cast<int>(OpRole::kForward) |
                     static_cast<int>(OpRole::kLoss))) ||
        (op_role == static_cast<int>(OpRole::kLRSched))) {
      // forward ops and lr schedule ops, used for first micro step
      forward_and_lr_ops_.push_back(op.get());
      if ((op_role != static_cast<int>(OpRole::kLRSched))) {
        // only forward ops, used for second and later micro steps
        forward_ops_.push_back(op.get());
      }
    } else if ((op_role == static_cast<int>(OpRole::kBackward)) ||
               (op_role == (static_cast<int>(OpRole::kBackward) |
                            static_cast<int>(OpRole::kLoss)))) {
      backward_ops_.push_back(op.get());
    } else if (op_role == static_cast<int>(OpRole::kOptimize)) {
      optimizer_ops_.push_back(op.get());
    } else {
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "The op %s is None of LRSched, Forward, Backward or Optimize.",
          op->Type()));
    }
  }

  // if not 1F1B scheduler
  if (schedule_mode_ != 1) return;

  bool is_first_stage = (pipeline_stage_ == 0);
  int BACKWARD = static_cast<int>(OpRole::kBackward);
  for (auto &op : ops_) {
    int op_role = op->Attr<int>("op_role");
    auto op_type = op->Type();

    // pipeline backward send op
    if (op_role != BACKWARD) continue;
    if (op_type != "p_send" && op_type != "partial_send") continue;

    auto var_name = op->InputVars()[0];
    VLOG(3) << "Pipeline backward send var " << var_name;
    PADDLE_ENFORCE_NE(is_first_stage,
                      true,
                      common::errors::PreconditionNotMet(
                          "The first pipeline stage must do not have a "
                          "backward send var, please check var %s",
                          var_name));

    backward_send_vars_.push_back(var_name);
    skip_vars_.push_back(var_name);
  }
}

void SectionWorker::PrepareUnusedVar() {
  VLOG(5) << "begin prepare the unused vars";
  unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
}

void SectionWorker::RunForward(
    int micro_id,
    std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  std::vector<OperatorBase *> &forward_tmp =
      micro_id == 0 ? forward_and_lr_ops_ : forward_ops_;
  for (auto &op : forward_tmp) {
    VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
            << micro_id;
    op->Run(*microbatch_scopes_[micro_id], place_);
    if (gc) {
      DeleteUnusedTensors(
          *microbatch_scopes_[micro_id], op, unused_vars_, gc.get());
    }
  }
}

void SectionWorker::RunBackward(
    int micro_id,
    std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : backward_ops_) {
    VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
            << micro_id;
    op->Run(*microbatch_scopes_[micro_id], place_);
    if (gc) {
      DeleteUnusedTensors(
          *microbatch_scopes_[micro_id], op, unused_vars_, gc.get());
    }
  }
}

void SectionWorker::RunUpdate(
    std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : optimizer_ops_) {
    VLOG(3) << "Update: running op " << op->Type();
    op->Run(*microbatch_scopes_[num_microbatches_ - 1], place_);
    if (gc) {
      DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                          op,
                          unused_vars_,
                          gc.get());
    }
  }
}

void SectionWorker::RunFThenB(std::unique_ptr<GarbageCollector> &gc) {
  // F-then-B scheduler which runs Forward phase for all microbatches,
  // then runs Backward phase for all microbatches.
  // step1: run forward
  for (int i = 0; i < num_microbatches_; ++i) {
    RunForward(i, gc, unused_vars_);
  }
  // step2: run backward
  for (int i = 0; i < num_microbatches_; ++i) {
    RunBackward(i, gc, unused_vars_);
  }
  // step3: run update
  RunUpdate(gc, unused_vars_);
}

void SectionWorker::Run1F1B(std::unique_ptr<GarbageCollector> &gc) {
  // 1F1B scheduler, which runs forward phase and backward phase alternatively
  // after startup phase. For a stage, the number of microbatches for
  // startup is num_pipeline_stages_ - pipeline_stage_ - 1, where
  // num_pipeline_stages_ is the total number of pipeline stages and
  // pipeline_stage_ is the pipeline stage of the current device.
  auto startup_steps = num_pipeline_stages_ - pipeline_stage_ - 1;
  VLOG(3) << "startup_steps:" << startup_steps
          << ", num_stages: " << num_pipeline_stages_
          << ", stage:" << pipeline_stage_;
  PADDLE_ENFORCE_GT(
      num_microbatches_,
      startup_steps,
      common::errors::InvalidArgument(
          "To use pipeline with 1F1B scheduler, please make sure number of "
          "microbatches (%d) is than startup steps (%d).",
          num_microbatches_,
          startup_steps));
  int fw_step = 0;
  int bw_step = 0;

  // startup phase
  while (fw_step < startup_steps) {
    RunForward(fw_step, gc, unused_vars_);
    fw_step += 1;
    VLOG(2) << "micro steps fw_step:" << fw_step;
  }

  // 1f1b phase
  while (fw_step < num_microbatches_) {
    RunForward(fw_step, gc, unused_vars_);

    // delete backward send var at step=(bw_step - 2)
    if (gc && bw_step >= 2) {
      DeleteUnusedTensors(
          *microbatch_scopes_[bw_step - 2], backward_send_vars_, gc.get());
    }

    RunBackward(bw_step, gc, unused_vars_);

    fw_step += 1;
    bw_step += 1;
    VLOG(2) << "micro steps fw_step:" << fw_step << ", bw_step:" << bw_step;
  }

  int reserve_bw_send_step = bw_step - 2;
  // backward phase
  while (bw_step < num_microbatches_) {
    RunBackward(bw_step, gc, unused_vars_);
    bw_step += 1;
    VLOG(2) << "micro steps  bw_step:" << bw_step;
  }

  VLOG(2) << "run update";
  RunUpdate(gc, unused_vars_);

  if (gc) {
    // NOTE(wangxi): program must add sync backward send comm at update
    // delete backward send var
    for (int i = reserve_bw_send_step; i < num_microbatches_; ++i) {
      DeleteUnusedTensors(
          *microbatch_scopes_[i], backward_send_vars_, gc.get());
    }
  }
}

void SectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";
  VLOG(2) << "mini batch steps:" << batch_id_;

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  if (max_memory_size >= 0) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (phi::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc = std::make_unique<UnsafeFastGPUGarbageCollector>(place_,
                                                             max_memory_size);
      }
    }
#endif
  }  // max_memory_size >= 0

  if (schedule_mode_ == 0) {  // NOLINT
    RunFThenB(gc);
  } else {
    Run1F1B(gc);
  }

  dev_ctx_->Wait();
  ++batch_id_;
}

}  // namespace paddle::framework
#endif
