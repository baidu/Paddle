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

#include "paddle/fluid/framework/new_executor/program_interpreter.h"

#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/io/save_load_tensor.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/static_build.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/os_info.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

PHI_DECLARE_bool(enable_host_event_recorder_hook);
PD_DECLARE_bool(log_memory_stats);
COMMON_DECLARE_string(static_runtime_data_save_path);
COMMON_DECLARE_bool(save_static_runtime_data);
namespace paddle::framework {

ProgramInterpreter::ProgramInterpreter(const phi::Place& place,
                                       const BlockDesc& block,
                                       framework::Scope* scope,
                                       const ExecutionConfig& execution_config)
    : is_build_(false),
      static_build_(false),
      is_shared_results_build_(false),
      is_in_op_profiling_mode_(false),
      place_(place),
      block_(block),
      dependency_builder_(),
      stream_analyzer_(place),
      copy_program_(nullptr),
      var_list_(),
      name2id_(),
      vec_meta_info_(),
      vec_instruction_(),
      unfinished_op_number_(0),
      execution_config_(execution_config),
      force_events_to_wait_(nullptr),
      var_scope_(scope),
      local_scope_(nullptr),
      main_thread_blocker_(),
      async_work_queue_(nullptr),
      exception_holder_(),
      exception_notifier_(nullptr),
      completion_notifier_(nullptr),
      gc_(nullptr),
      last_live_ops_(),
      dependency_count_(std::make_shared<std::vector<size_t>>()),
      deps_(),
      refs_(),
      sync_op_num_(-1),
      trace_execute_order_(),
      instruction_scheduling_priority_less(),
      output_hookfuncs_(),
      input_hookfuncs_(),
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      calculate_stream_timer_(
          std::make_unique<phi::CalculateStreamTimer>(place)),
#endif
      last_calculate_instr_id_(0),
      enable_job_schedule_profiler_(false) {
  VLOG(4) << "ProgramInterpreter(): " << this << " on " << place_;

  exception_notifier_ = main_thread_blocker_.RegisterEvent(kExceptionCaught);
  completion_notifier_ = main_thread_blocker_.RegisterEvent(kTaskCompletion);

  if (!FLAGS_new_executor_use_local_scope) {
    execution_config_.create_local_scope = false;
  }
  execution_config_.AnalyzeThreadPoolConfig(place, block.OpSize());
  execution_config_.Log(/*log_level=*/8);

  if (execution_config_.create_local_scope) {
    auto local_scope = &var_scope_.GetMutableScope()->NewScope();
    local_scope_ = local_scope;
  }
  var_scope_.SetLocalScope(local_scope_);

  static_build_ = FLAGS_new_executor_static_build &&
                  !FLAGS_new_executor_use_cuda_graph &&
                  interpreter::BlockCanBeStaticBuilt(block);

  instruction_scheduling_priority_less = [this](size_t lhs, size_t rhs) {
    SchedulingPriority lhs_scheduling_priority =
        vec_instruction_[lhs].GetSchedulingPriority();
    SchedulingPriority rhs_scheduling_priority =
        vec_instruction_[rhs].GetSchedulingPriority();
    if (lhs_scheduling_priority == rhs_scheduling_priority) {
      return lhs > rhs;
    }
    return lhs_scheduling_priority > rhs_scheduling_priority;
  };

  PrepareForCUDAGraphCapture();
}

ProgramInterpreter::~ProgramInterpreter() {
  // cancel gc's thread
  gc_.reset(nullptr);
  async_work_queue_.reset();
  VLOG(4) << "~ProgramInterpreter(): " << this << " on " << place_;

#ifdef PADDLE_WITH_DNNL
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

void ProgramInterpreter::RunImpl() {
  // lazy initialization of gc, do not create gc is the program only run once
  if (!gc_) {
    gc_ = CreateInterpreterCoreGarbageCollector(place_, vec_instruction_);
  }

  interpreter::ResetAtomicGuard guard(&deps_, &refs_);

  if (is_in_op_profiling_mode_ || execution_config_.used_for_inference ||
      ((execution_config_.used_for_jit || execution_config_.used_for_cinn) &&
       (sync_op_num_ == 0))) {
    VLOG(4) << "Tracing Instruction List";
    TraceInstructionList(vec_instruction_);
  } else {
    VLOG(4) << "Non-tracing";
    // For the program that only run once, it is no need to
    // create work_queue, so the async_work_queue_ is created
    // until the second step run.
    async_work_queue_ = GetWorkQueue();
    ExecuteInstructionList(vec_instruction_);
  }

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (phi::is_custom_place(place_)) {
    phi::DeviceContextPool::Instance().Get(place_)->Wait();
  }
#endif
}

FetchList ProgramInterpreter::Run(const std::vector<std::string>& feed_names,
                                  bool need_fetch,
                                  bool enable_job_schedule_profiler,
                                  bool enable_op_profiling,
                                  bool switch_stream) {
  enable_job_schedule_profiler_ = enable_job_schedule_profiler;
  is_in_op_profiling_mode_ = enable_op_profiling;

  std::vector<paddle::framework::OpFuncNode> op_func_nodes;
  Build(feed_names, &op_func_nodes, switch_stream);

  if (!is_build_ || switch_stream) {
    SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);
    UpdateSyncOpNum();
    if (static_build_) {
      VLOG(4) << "RUN impl";
      RunImpl();
    }
    is_build_ = true;
    is_shared_results_build_ = true;
  } else {
    RunImpl();
  }

  if (HasLocalScope()) {
    ClearDenseTensorArrayInLocalScope();
  }

  // NOTE (liuchenghao): we need to reset "is_in_op_profiling_mode_" to false.
  // This is because ProgramInterpreter::Run(...) has two implementations, only
  // this implementation correctly updates its state, if user switches to
  // another implementation of Run(...) half way, its state can cause potential
  // problems.
  is_in_op_profiling_mode_ = false;

  if (need_fetch) {
    // return Fetch Tensors
    Scope* inner_scope =
        HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
    auto* fetch_var = inner_scope->FindVar(interpreter::kFetchVarName);
    if (fetch_var) {
      auto fetch_list =
          std::move(*fetch_var->GetMutable<framework::FetchList>());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (platform::IsCUDAGraphCapturing()) {
        PADDLE_ENFORCE_EQ(fetch_list.empty(),
                          true,
                          common::errors::InvalidArgument(
                              "Cannot fetch data when using CUDA Graph."));
      }
#endif
      return fetch_list;
    }
  }

  return {};
}

void ProgramInterpreter::Build(
    const std::vector<std::string>& feed_names,
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes,
    bool switch_stream) {
  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_DNNL
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  if (!is_build_ || switch_stream) {
    LOG_FIRST_N(INFO, 1) << "New Executor is Running.";
    paddle::framework::interpreter::BuildVariableScope(
        block_, execution_config_, &var_scope_);

    paddle::framework::interpreter::BuildOpFuncList(
        place_,
        block_,
        execution_config_.skip_gc_vars,
        op_func_nodes,
        &var_scope_,
        execution_config_,
        input_hookfuncs_,
        output_hookfuncs_,
        HasLocalScope(),
        static_build_);
  }
}

FetchList ProgramInterpreter::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors,
    bool need_fetch,
    bool enable_job_schedule_profiler,
    bool switch_stream) {
  enable_job_schedule_profiler_ = enable_job_schedule_profiler;

  SetDeviceId(place_);
  CheckCUDAGraphBeforeRun(feed_names);

#ifdef PADDLE_WITH_DNNL
  platform::AttachPointerHashToMKLDNNKey(this, place_);
#endif

  bool is_build = is_build_;
  Prepare(feed_names, feed_tensors, is_build, switch_stream);

  if (is_build && !switch_stream) {
    RunImpl();
  }

  if (HasLocalScope()) {
    ClearDenseTensorArrayInLocalScope();
  }

  if (need_fetch) {
    // return Fetch Tensors
    Scope* inner_scope =
        HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
    auto* fetch_var = inner_scope->FindVar(interpreter::kFetchVarName);
    if (fetch_var) {
      auto fetch_list =
          std::move(*fetch_var->GetMutable<framework::FetchList>());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (platform::IsCUDAGraphCapturing()) {
        PADDLE_ENFORCE_EQ(fetch_list.empty(),
                          true,
                          common::errors::InvalidArgument(
                              "Cannot fetch data when using CUDA Graph."));
      }
#endif
      return fetch_list;
    }
  }

  return {};
}

void ProgramInterpreter::SetCopyProgram(std::shared_ptr<ProgramDesc> prog) {
  copy_program_ = prog;
}

void ProgramInterpreter::SetSkipGcVars(
    const std::set<std::string>& skip_gc_vars) {
  PADDLE_ENFORCE_EQ(
      execution_config_.skip_gc_vars.empty(),
      true,
      common::errors::PreconditionNotMet(
          "execution_config_.skip_gc_vars can only be initialized once, now "
          "execution_config_.skip_gc_vars is "
          "not empty, do not call SetSkipGcVars method repeatedly."));
  execution_config_.skip_gc_vars = skip_gc_vars;
}

void ProgramInterpreter::SetJitInputVars(
    const std::set<std::string>& jit_input_vars) {
  PADDLE_ENFORCE_EQ(
      execution_config_.jit_input_vars.empty(),
      true,
      common::errors::PreconditionNotMet(
          "execution_config_.jit_input_vars can only be initialized once, now "
          "execution_config_.jit_input_vars is "
          "not empty, do not call SetJitInputVars method repeatedly."));
  execution_config_.jit_input_vars = jit_input_vars;
}

const std::set<std::string>& ProgramInterpreter::JitInputVars() const {
  return execution_config_.jit_input_vars;
}

const VariableScope* ProgramInterpreter::GetVariableScope() const {
  return &var_scope_;
}

void ProgramInterpreter::reset_scope(Scope* new_scope) {
  var_scope_.SetScope(new_scope);
  auto& var_list = var_scope_.MutableVarList();
  for (size_t i = 0; i < var_list.size(); i++) {
    const auto& var_name = var_scope_.GetNameById(static_cast<int>(i));
    var_list[i] = new_scope->FindVar(var_name);
  }
  // The index should be assured valid, cause the InterpreterCore may not be
  // fully built, but was still cached and used. For example, see unit test
  // `test_assert.py`, it may exit before `ProgramInterpreter::Convert`,
  // but still was cached and used by later tests.
  for (size_t i = 0; i < std::min(refs_.size(), var_list.size()); i++) {
    refs_[i]->ResetVariable(var_list[i]);
  }

  for (auto& ins : vec_instruction_) {
    BuildAndCacheInstructionCtx(&ins);
  }
}

const Scope* ProgramInterpreter::local_scope() const { return local_scope_; }
void ProgramInterpreter::ShareWorkQueueFrom(InterpreterBaseImpl* src) {
  async_work_queue_ =
      reinterpret_cast<ProgramInterpreter*>(src)->GetWorkQueue();
  VLOG(8) << "Share AsyncWorkQueue from InterpreterCore(" << src
          << ") to InterpreterCore(" << this << ")";
}

void ProgramInterpreter::ShareBuildResultsFrom(const InterpreterBaseImpl& src) {
  const ProgramInterpreter& impl = dynamic_cast<const ProgramInterpreter&>(src);
  if (is_shared_results_build_ || !impl.IsSharedResultsBuild()) {
    return;
  }
  // share op dependency
  dependency_builder_.ShareDependencyFrom(impl.GetDependencyBuilder());
  dependency_count_ = impl.GetDependencyCount();
  // share event analysis
  stream_analyzer_.ShareEventInfoFrom(impl.GetStreamAnalyzer());
  is_shared_results_build_ = true;
  VLOG(8) << "Share Build Results from InterpreterCore(" << &impl
          << ") to InterpreterCore(" << this << ")";
}

bool ProgramInterpreter::BuildInplaceCheckVarIsOnlyInput(
    const std::vector<std::vector<size_t>>& input_var2op, size_t var_index) {
  if (!var_scope_.VarDesc(static_cast<int>(var_index))) {
    return input_var2op.at(var_index).size() == 1;
  } else {
    int is_input_cnt = 0;
    for (auto inst_id : input_var2op.at(var_index)) {
      OpInOutInfo info;
      info.Build(vec_instruction_.at(inst_id).OpBase());
      if (info.IsInArgBufferNeeded(
              var_scope_.VarDesc(static_cast<int>(var_index))->Name())) {
        is_input_cnt++;
      }
    }
    return is_input_cnt == 1;
  }
}

std::shared_ptr<interpreter::AsyncWorkQueue>
ProgramInterpreter::GetWorkQueue() {
  if (async_work_queue_ == nullptr) {
    async_work_queue_ = std::make_shared<interpreter::AsyncWorkQueue>(
        execution_config_.host_num_threads,
        execution_config_.device_num_threads,
        nullptr);
  }
  return async_work_queue_;
}

const interpreter::DependencyBuilder& ProgramInterpreter::GetDependencyBuilder()
    const {
  return dependency_builder_;
}

std::shared_ptr<std::vector<size_t>> ProgramInterpreter::GetDependencyCount()
    const {
  return dependency_count_;
}

const interpreter::StreamAnalyzer& ProgramInterpreter::GetStreamAnalyzer()
    const {
  return stream_analyzer_;
}

bool ProgramInterpreter::IsSharedResultsBuild() const {
  return is_shared_results_build_;
}

void ProgramInterpreter::BuildAndCacheInstructionCtx(Instruction* instr_node) {
  Scope* inner_scope =
      HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
  VariableValueMap ins_map;
  for (auto& var_name_item : instr_node->Inputs()) {
    std::vector<Variable*> input_vars;

    input_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      input_vars.emplace_back(inner_scope->FindVar(var_scope_.GetNameById(id)));
    }
    ins_map.emplace(var_name_item.first, std::move(input_vars));
  }

  VariableValueMap outs_map;
  for (auto& var_name_item : instr_node->Outputs()) {
    std::vector<Variable*> out_vars;

    out_vars.reserve(var_name_item.second.size());
    for (auto& id : var_name_item.second) {
      out_vars.emplace_back(inner_scope->FindVar(var_scope_.GetNameById(id)));
    }
    outs_map.emplace(var_name_item.first, std::move(out_vars));
  }

  instr_node->ResetContext(ins_map, outs_map, instr_node->OpBase()->Type());
}

void ProgramInterpreter::BuildInplace() {
  // NOTE(Ruibiao): coalesce_tensor_op outputs a FusedOutput phi::DenseTensor
  // and a list of Output Tensors which are sliced from the FusedOutput. These
  // outputs should not be the outvar of the in-place var-pair since memory
  // reuse between FusedOutput and Output Tensors is assumed. For the following
  // example:
  // fused_var, var1, var2, var3 = coalesce_tensor(var1, var2, var3)
  // var1 = sum(var4, var5)
  // ...
  //
  // After running coalesce_tensor_op, var1 is assumed to share the buffer
  // slices from fused_var. However, if sum_op is in-place, then var1 would
  // re-share the buffer with var4 instead of fused_var.
  std::set<std::string> skip_inplace_outvars;
  for (Instruction& instr : vec_instruction_) {
    OperatorBase* op = instr.OpBase();
    if (op->Type() == kCoalesceTensor) {
      const std::vector<std::string>& outputs =
          op->OutputVars(/*has_intermediate=*/false);
      skip_inplace_outvars.insert(outputs.begin(), outputs.end());
    }
  }

  Scope* local_scope = HasLocalScope() ? var_scope_.GetMutableLocalScope()
                                       : var_scope_.GetMutableScope();
  std::vector<std::vector<size_t>> input_var2op(var_scope_.VarSize());
  for (Instruction& instr : vec_instruction_) {
    for (auto& item : instr.Inputs()) {
      for (int var_id : item.second) {
        if (var_id != kEmptyVarIndex) {
          input_var2op.at(var_id).push_back(instr.Id());
        }
      }
    }
  }

  for (auto& instr : vec_instruction_) {
    auto* op_base = instr.OpBase();
    if (!op_base->Info().infer_inplace_) {
      continue;
    }

    auto in_to_outs = op_base->Info().infer_inplace_(
        phi::is_gpu_place(instr.DeviceContext().GetPlace()));

    auto& inputs = instr.Inputs();
    auto& outputs = instr.Outputs();
    for (auto& pair : in_to_outs) {
      auto iter = inputs.find(pair.first);
      if (iter != inputs.end() && !iter->second.empty()) {
        auto in_var_desc = var_scope_.VarDesc(iter->second[0]);
        if (in_var_desc && in_var_desc->Persistable()) {
          continue;
        }
        if (var_scope_.GetVarSkipInplace(iter->second[0])) {
          continue;
        }
        if (BuildInplaceCheckVarIsOnlyInput(input_var2op, iter->second[0])) {
          auto iterout = outputs.find(pair.second);
          if (iterout != outputs.end() && !iterout->second.empty()) {
            const std::string& invar_name =
                var_scope_.GetNameById(iter->second[0]);
            const std::string& outvar_name =
                var_scope_.GetNameById(iterout->second[0]);
            auto invar = local_scope->FindVar(invar_name);
            auto outvar = local_scope->FindVar(outvar_name);

            if (invar && outvar && invar->IsType<phi::DenseTensor>() &&
                outvar->IsType<phi::DenseTensor>() &&
                skip_inplace_outvars.find(outvar_name) ==
                    skip_inplace_outvars.end()) {
              instr.AddInplace(invar, outvar);
              VLOG(3) << "inplace " << op_base->Type() << " " << invar_name
                      << " -> " << outvar_name;
            }
          }
        }
      }
    }
  }
}

void ProgramInterpreter::PrepareForCUDAGraphCapture() {
  if (!FLAGS_new_executor_use_cuda_graph) return;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_EQ(
      platform::IsCUDAGraphCapturing(),
      false,
      common::errors::PermissionDenied("CUDA Graph is not allowed to capture "
                                       "before prepare."));
  PADDLE_ENFORCE_EQ(phi::is_gpu_place(place_),
                    true,
                    common::errors::InvalidArgument(
                        "CUDA Graph is only supported on NVIDIA GPU device."));
  // If set true, will call `cudaStreamSynchronize(nccl_stream)`after allreduce.
  // which may cause error in cuda graph. This behavior is consistent with PE.
  PADDLE_ENFORCE_EQ(FLAGS_sync_nccl_allreduce,
                    false,
                    common::errors::InvalidArgument(
                        "FLAGS_sync_nccl_allreduce must be False to support "
                        "CUDA Graph capturing."));

  // All output vars of coalesce_tensor op should be persistable.
  // If fused output var of coalesce_tensor is gc, it will cause accuracy
  // problem. The specific reasons need to be analyzed.
  for (auto& op_desc : block_.AllOps()) {
    if (op_desc->Type() == kCoalesceTensor) {
      for (auto& out_var_name : op_desc->OutputArgumentNames()) {
        // The fused var needs to be set to persistable, not just added to
        // skip_gc_vars.
        // In the case where the feed fetch var is changed, StandaloneExecutor
        // will be newly constructed. If the fused var is not persistable,
        // these vars will be recreated and initialized, resulting in
        // precision problems.
        auto* out_var = op_desc->Block()->FindVarRecursive(out_var_name);
        if (out_var) {
          out_var->SetPersistable(true);
          VLOG(4) << "Mark Var(" << out_var_name << ") as Persistable.";
        }
      }
    }
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "CUDA Graph is only supported on NVIDIA GPU device."));
#endif
}

void ProgramInterpreter::CheckCUDAGraphBeforeRun(
    const std::vector<std::string>& feed_names) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::IsCUDAGraphCapturing()) {
    PADDLE_ENFORCE_EQ(
        feed_names.empty(),
        true,
        common::errors::InvalidArgument(
            "Feeding data is not permitted when capturing CUDA Graph."));
    PADDLE_ENFORCE_EQ(
        FLAGS_new_executor_use_cuda_graph,
        true,
        common::errors::InvalidArgument(
            "You must turn on FLAGS_new_executor_use_cuda_graph to True "
            "to enable CUDA Graph capturing."));
    PADDLE_ENFORCE_EQ(
        place_,
        platform::CUDAGraphCapturingPlace(),
        common::errors::InvalidArgument("The place to capture CUDAGraph is "
                                        "not the same as the place to run."));
  }
#endif
}

void ProgramInterpreter::BuildOperatorDependences() {
  // analysis the dependences between ops, add next_instr_list to each instr,
  // and set the dependency_count_
  size_t instr_num = vec_instruction_.size();
  dependency_count_ = GetDependencyCount();
  if (!is_shared_results_build_) {
    dependency_count_->assign(instr_num, 0);
  }

  auto downstream_map = dependency_builder_.Build(vec_instruction_);

  for (size_t instr_id = 0; instr_id < instr_num; ++instr_id) {
    Instruction& cur_instr = vec_instruction_[instr_id];
    const std::set<size_t>& next_instr_ids = downstream_map[instr_id];

    if (FLAGS_new_executor_serial_run) {
      for (size_t next_instr_id : next_instr_ids) {
        cur_instr.AddNextInstrInSameThread(next_instr_id);
      }
    } else {
      if (cur_instr.KernelType() == OpFuncType::kGpuAsync) {
        for (size_t next_instr_id : next_instr_ids) {
          if (vec_instruction_[next_instr_id].KernelType() ==
              OpFuncType::kGpuAsync) {
            cur_instr.AddNextInstrInSameThread(next_instr_id);
          } else {
            cur_instr.AddNextInstrInDifferentThread(next_instr_id);
          }
        }
      } else {
        bool has_instr_in_same_thread = false;
        for (size_t next_instr_id : next_instr_ids) {
          if (!has_instr_in_same_thread &&
              vec_instruction_[next_instr_id].KernelType() !=
                  OpFuncType::kGpuAsync) {
            cur_instr.AddNextInstrInSameThread(next_instr_id);
            has_instr_in_same_thread = true;
          } else {
            cur_instr.AddNextInstrInDifferentThread(next_instr_id);
          }
        }
      }
    }

    if (!is_shared_results_build_) {
      for (size_t next_instr_id : next_instr_ids) {
        ++(*dependency_count_)[next_instr_id];
      }
    }
  }
}

// At the end of each step, the holder of phi::DenseTensor in phi::TensorArray
// is null. Clear these Tensors and leave phi::TensorArray empty, otherwise an
// exception will occur in the next step
void ProgramInterpreter::ClearDenseTensorArrayInLocalScope() {
  auto vars = local_scope_->LocalVars();
  for (auto var : vars) {
    if (var->IsType<phi::TensorArray>()) {
      auto* dense_tensor_arr = var->GetMutable<phi::TensorArray>();
      dense_tensor_arr->clear();
    }
  }
}

std::tuple<double, double> ProgramInterpreter::InterpreterRunTime() {
  double start_time = 0, end_time = 0;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  start_time = calculate_stream_timer_->StartTime();
  end_time = calculate_stream_timer_->EndTime();
#endif
  return std::make_tuple(start_time, end_time);
}

void ProgramInterpreter::Convert(
    std::vector<paddle::framework::OpFuncNode>* op_func_nodes) {
  auto& vec_meta_info = var_scope_.MutableVecMetaInfo();
  auto nodes = *op_func_nodes;
  auto op_nums = nodes.size();
  vec_instruction_.clear();
  vec_instruction_.reserve(op_nums);
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    auto& op_func_node = nodes[op_idx];
    stream_analyzer_.SetForceEventsToWaitInfo(force_events_to_wait_);
    auto* dev_ctx_ = stream_analyzer_.ParseDeviceContext(op_func_node);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (FLAGS_new_executor_use_cuda_graph) {
      auto& op = op_func_node.operator_base_;
      auto& op_type = op->Type();
      if (op_type == interpreter::kMemcpyD2H ||
          op_type == interpreter::kMemcpyH2D) {
        PADDLE_THROW(common::errors::Fatal(
            "Cuda memory copy d2h/h2d is not allowed while using cuda graph."));
      }
      PADDLE_ENFORCE_EQ(typeid(*dev_ctx_) == typeid(phi::GPUContext),
                        true,
                        common::errors::InvalidArgument(
                            "Device context of op %s must be [%s] while using "
                            "cuda graph, but got [%s].",
                            op_type,
                            typeid(phi::GPUContext).name(),
                            typeid(*dev_ctx_).name()));
      // cuda graph needs to record all stream
      phi::backends::gpu::CUDAGraphContextManager::Instance()
          .RecordCapturingDeviceContext(dev_ctx_);
    }
#endif
    vec_instruction_.emplace_back(op_idx, std::move(op_func_node), *dev_ctx_);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    vec_instruction_.back().UpdateRecordStreamForGcInfo();
#endif
  }

  BuildOperatorDependences();

  // NOTE(Ruibiao): For cross-step stream synchronization, an event may be
  // recorded in the first step and waited in the second step. So, in the first
  // step, the WaitEvent may be called without RecordEvent. Considering that
  // before the first call to RecordEvent, an Event represents an empty set of
  // work and WaitEvent always return succeed immediately, we omit the
  // prelude-record for the first step here.
  stream_analyzer_.ConstructEvents(&vec_instruction_);

  // add event for the input var of jit program, since there are async copied
  // from gpu_pinned place to gpu place on compute stream.
  for (size_t i = 0; i < dependency_count_->size(); ++i) {
    if ((*dependency_count_)[i] == 0) {
      auto& inst = vec_instruction_[i];
      if (inst.OpBase()->Type() == interpreter::kMemcpyD2H &&
          phi::is_gpu_place(place_)) {
        for (auto& item : inst.Inputs()) {
          for (auto var_id : item.second) {
            auto name = var_scope_.GetNameById(var_id);
            if (JitInputVars().count(name)) {
              auto device_event = std::make_shared<platform::DeviceEvent>(
                  place_, platform::GenerateDeviceEventFlag());
              VLOG(4) << "Add input event for input: " << name << " of "
                      << inst.OpBase()->Type();
              inst.AddEventToWait(
                  i, device_event, stream_analyzer_.GetWaiterType(inst));
            }
          }
        }
      }
    }
  }

  // calculate last_live_ops_
  for (size_t op_idx = 0; op_idx < op_nums; ++op_idx) {
    Instruction& instr = vec_instruction_[op_idx];
    OpInOutInfo info;
    info.Build(instr.OpBase());

    std::set<size_t> gc_check_vars;

    const std::map<std::string, std::vector<int>>& ins = instr.Inputs();
    const std::map<std::string, std::vector<int>>& outs = instr.Outputs();
    std::multimap<std::string, std::vector<int>> ins_and_outs{ins.begin(),
                                                              ins.end()};
    ins_and_outs.insert(outs.begin(), outs.end());

    for (auto& item : ins_and_outs) {
      for (auto id : item.second) {
        if (id == kEmptyVarIndex) {
          continue;
        }
        auto* var_desc = var_scope_.VarDesc(id);
        // skip no_need_buffer input vars
        if (var_desc && ins.count(item.first) &&
            !info.IsInArgBufferNeeded(var_desc->Name())) {
          continue;
        }
        // skip when this var is not in block and not a data_transferred var,
        // which means this var is managed by other block
        const auto& var_name = var_scope_.GetNameById(id);
        bool not_owned = !block_.HasVar(var_name);
        const auto& transferred_vars = var_scope_.DataTransferAddedVars();
        bool not_transferred =
            std::all_of(transferred_vars.begin(),
                        transferred_vars.end(),
                        [&](const std::pair<std::string, int>& elem) {
                          return elem.first != var_name;
                        });
        if (not_owned && not_transferred) {
          VLOG(10) << "[gc_check_inputs] skip gc: " << var_name;
          continue;
        }
        gc_check_vars.insert(id);
      }
    }

    for (auto var_id : gc_check_vars) {
      Scope* inner_scope =
          HasLocalScope() ? local_scope_ : var_scope_.GetMutableScope();
      paddle::framework::Variable* var = inner_scope->FindVar(
          var_scope_.GetNameById(static_cast<int>(var_id)));
      if (var->IsType<phi::DenseTensor>() || var->IsType<phi::SelectedRows>() ||
          var->IsType<phi::TensorArray>() ||
          var->IsType<phi::SparseCooTensor>() ||
          var->IsType<phi::SparseCsrTensor>()) {
        last_live_ops_[var_id].insert(op_idx);
      } else {
        VLOG(4) << "not clear "
                << var_scope_.GetNameById(static_cast<int>(var_id)) << " after "
                << instr.OpBase()->Type() << " because its type is "
                << framework::ToTypeName(var->Type());
      }
    }
  }

  // clear the last_live_ops list for all vars in skip_gc_vars
  for (const std::string& skip_gc_var : execution_config_.skip_gc_vars) {
    int var_id = var_scope_.GetIdByName(skip_gc_var);
    if (var_id != -1) {
      last_live_ops_[var_id].clear();
      VLOG(8) << "Skip gc for var: " << skip_gc_var;
    }
  }

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  // For example,
  // b = op1(a)
  // c = op2(a, b)
  // in this case, a is the input of op1 and op2, we only need to check
  // a after op2, because op2 always uses a after op1.
  for (size_t i = 0; i < last_live_ops_.size(); ++i) {
    std::set<size_t> minimum_last_live_ops;
    for (size_t item : last_live_ops_[i]) {
      bool not_before_any = true;
      // find the op that is not executed before any
      for (size_t other_item : last_live_ops_[i]) {
        if (dependency_builder_.OpHappensBefore(item, other_item)) {
          VLOG(8) << "happens_before: " << item << "->" << other_item
                  << ", so skip " << item;
          not_before_any = false;
          break;
        }
      }
      if (not_before_any) {
        VLOG(8) << "last live op of var " << i << " "
                << var_scope_.GetNameById(static_cast<int>(i)) << " : " << item
                << " " << vec_instruction_[item].OpBase()->Type();
        minimum_last_live_ops.insert(item);
        if (!(var_scope_.VarDesc(static_cast<int>(i)) &&
              var_scope_.VarDesc(static_cast<int>(i))->Persistable())) {
          vec_instruction_[item].AddGCCheckVar(i);
        }
      }
    }
    last_live_ops_[i] = minimum_last_live_ops;
    vec_meta_info[i].var_ref_count_ =
        static_cast<int>(last_live_ops_[i].size());
  }

  for (auto& ins : vec_instruction_) {
    BuildAndCacheInstructionCtx(&ins);
  }

  bool inplaced = false;
  for (const Instruction& inst : vec_instruction_) {
    if (inst.OpBase()->Type() == "share_buffer" ||
        inst.OpBase()->Type() == "share_data") {
      VLOG(4) << "Already inplaced, skip inplace now.";
      inplaced = true;
    }
  }

  if (FLAGS_new_executor_use_inplace && !inplaced) {
    BuildInplace();
  }

  for (auto& dep : *dependency_count_) {
    deps_.emplace_back(std::make_shared<interpreter::OpDepInfo>(dep));
  }
  for (size_t i = 0; i < vec_meta_info.size(); ++i) {
    refs_.emplace_back(std::make_shared<interpreter::VarRefInfo>(
        vec_meta_info[i].var_ref_count_,
        var_scope_.VarRef(static_cast<int>(i))));
  }

  AnalyseExecuteOrderForTrace();
}

void ProgramInterpreter::BuildSkipShareLoDInfo() {
  for (size_t i = 0; i < vec_instruction_.size(); ++i) {
    bool can_skip_lod = true;
    for (auto& input : vec_instruction_[i].InnerRuntimeContext()->inputs) {
      for (auto& var : input.second) {
        if (var->IsType<phi::DenseTensor>()) {
          if (!var->Get<phi::DenseTensor>().lod().empty()) {
            can_skip_lod = false;
            break;
          }
        } else {
          can_skip_lod = false;
          break;
        }
      }
    }
    if (can_skip_lod) {
      VLOG(8) << "skip share lod for: " << vec_instruction_[i].OpBase()->Type()
              << " (" << i << ")";
    }
    vec_instruction_[i].InnerInferShapeContext()->SetSkipLoD(can_skip_lod);
  }
}

void ProgramInterpreter::RunOperator(const Instruction& instr_node) {
  auto* op = instr_node.OpBase();
  auto place = instr_node.DeviceContext().GetPlace();
  Scope* local_scope = HasLocalScope() ? var_scope_.GetMutableLocalScope()
                                       : var_scope_.GetMutableScope();
  VLOG(4) << "Start run " << place << " " << op->DebugStringEx(local_scope);

  if (execution_config_.used_for_inference) {
    for (auto& hook : input_hookfuncs_) {
      hook(op, local_scope);
    }

    if (op->Type() == "while" || op->Type() == "conditional_block") {
      op->SetInputHooks(input_hookfuncs_);
      op->SetOutputHooks(output_hookfuncs_);
      auto runtime_attrs = op->RuntimeAttrs();
      runtime_attrs.insert(std::make_pair("used_for_inference", true));
      op->SetRuntimeAttributeMap(runtime_attrs);
    }
  }

  auto op_with_kernel = dynamic_cast<const framework::OperatorWithKernel*>(op);
  {
    // If it is OperatorBase, InferShape do nothing.
    if (op_with_kernel != nullptr) {
      phi::RecordEvent infershape_event("infer_shape",
                                        phi::TracerEventType::OperatorInner,
                                        1,
                                        phi::EventRole::kInnerOp);

      // see OperatorWithKernel::RunImpl in operator.cc for why
      if (!(op_with_kernel->HasAttr(kAllKernelsMustComputeRuntimeShape) &&
            op_with_kernel->Attr<bool>(kAllKernelsMustComputeRuntimeShape))) {
        if (instr_node.can_use_infermeta_ctx_) {
          op_with_kernel->Info().infer_meta_(const_cast<phi::InferMetaContext*>(
              instr_node.InnerCompatInferMetaContext()));
        } else {
          op_with_kernel->Info().infer_shape_(
              instr_node.InnerInferShapeContext().get());
        }
      }
      if (FLAGS_enable_host_event_recorder_hook) {
        platform::RecordOpInfoSupplement(op->Type(),
                                         op->Attrs(),
                                         *(instr_node.InnerInferShapeContext()),
                                         *(instr_node.InnerRuntimeContext()),
                                         op->Id());
      }
    }
  }
  if (op_with_kernel != nullptr && FLAGS_new_executor_use_inplace) {
    // TODO(xiongkun03) Does operator base support inplace ?
    for (auto& pair : instr_node.InplaceInfo()) {
      const auto& in = GetTensorFromVar(pair.first);
      auto* out = GetMutableTensorFromVar(pair.second);
      if (in.dims() == out->dims()) {
        out->ShareBufferWith(in);
      }
    }
  }

  if (is_in_op_profiling_mode_ && interpreter::IsCommunicationOp(op)) {
    // skip communication op if enabled runtime profiling feature since their
    // run time are mainly determined by other ops and they require other
    // sub-graphs also run on the same machine concurrently, which cannot be
    // guaranteed in most of the time.
  } else {
    phi::RecordEvent compute_event("compute",
                                   phi::TracerEventType::OperatorInner,
                                   1,
                                   phi::EventRole::kInnerOp);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (is_in_op_profiling_mode_) {
      platform::GpuDeviceSync();
    }
#endif

    if (op_with_kernel == nullptr) {  // operator base
      instr_node.OpBase()->Run(*local_scope, place_);
    } else {
      phi::Kernel* kernel = instr_node.PhiKernel();
      if (kernel && kernel->IsValid()) {  // phi kernel
        if (kernel->GetKernelRegisteredType() ==
            phi::KernelRegisteredType::FUNCTION) {
          VLOG(4) << "Run function kernel: " << op->Type();
          VLOG(4) << instr_node.InnerRuntimeContext().get() << " "
                  << &instr_node.DeviceContext();

          auto dev_ctx =
              const_cast<phi::DeviceContext*>(&instr_node.DeviceContext());
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
          auto attrs = op->Attrs();
          if ((!dev_ctx->GetCommContext() || op->Type() == "p_send" ||
               op->Type() == "p_recv") &&
              attrs.find("ring_id") != attrs.end()) {
            auto ring_id_attr = attrs.at("ring_id");
            int ring_id = PADDLE_GET(int, ring_id_attr);
            auto map = distributed::ProcessGroupMapFromGid::getInstance();
            const auto& comm_context_manager =
                phi::distributed::CommContextManager::GetInstance();
            phi::distributed::CommContext* comm_context = nullptr;
            if (comm_context_manager.Has(std::to_string(ring_id))) {
              comm_context = comm_context_manager.Get(std::to_string(ring_id));
            } else if (map->has(ring_id)) {
              distributed::ProcessGroup* pg = map->get(ring_id);
              comm_context =
                  static_cast<paddle::distributed::ProcessGroupNCCL*>(pg)
                      ->GetOrCreateCommContext(place);
            }
            PADDLE_ENFORCE_NE(
                comm_context,
                nullptr,
                common::errors::Unavailable(
                    "NCCLCommContext is nullptr. For op with ring_id attr, "
                    "comm_context should be set in dev_ctx, but it cannot be "
                    "get from CommContextManager or ProcessGroup."));
            dev_ctx =
                static_cast<phi::distributed::NCCLCommContext*>(comm_context)
                    ->GetDevContext();
            dev_ctx->SetCommContext(comm_context);
          }
#endif
          phi::KernelContext phi_kernel_context;
          op_with_kernel->BuildPhiKernelContext(
              *instr_node.InnerRuntimeContext().get(),
              dev_ctx,
              &phi_kernel_context);

          (*kernel)(&phi_kernel_context);
        } else {
          VLOG(4) << "Run structure kernel: " << op->Type();
          (*kernel)(instr_node.InnerExecutionContext().get());
        }
      } else {  // fluid kernel
        instr_node.KernelFunc()(*instr_node.InnerExecutionContext().get());
      }
    }

    if (is_in_op_profiling_mode_ && op->Id() != UINT64_MAX) {
      OperatorDistAttr* op_dist_attr = block_.Op(op->Id())->MutableDistAttr();
      platform::Timer op_timer;
      op_timer.Start();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::GpuDeviceSync();
#endif
      op_timer.Pause();
      if (op_dist_attr) op_dist_attr->set_run_time_us(op_timer.ElapsedUS());
    }
  }

  VLOG(4) << "End run " << place << " "
          << op->DebugStringEx(local_scope);  // NOLINT

  if (!instr_node.InplaceBackMap().empty()) {
    phi::RecordEvent inplaceback_event(
        "InplaceVarsBack", phi::TracerEventType::UserDefined, 10);
    auto& m = instr_node.InplaceBackMap();
    // NOTE(zhiqiu): same logic as TransferInplaceVarsBack() in operator.cc
    for (auto& p : m) {
      auto* transformed_tensor =
          GetMutableDenseTensorOrSelectedRowsValueFromVar(
              var_scope_.VarRef(p.first));
      auto* original_tensor = GetMutableDenseTensorOrSelectedRowsValueFromVar(
          var_scope_.VarRef(p.second));
      original_tensor->ShareDataWith(*transformed_tensor);
      VLOG(4) << "Transfer inplace variable back form "
              << var_scope_.GetNameById(p.first) << " to "
              << var_scope_.GetNameById(p.second);
    }
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    instr_node.DeviceContext().Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    VLOG(4) << "Operator(" << op->Type()  // NOLINT
            << "): context wait and get last error";
#endif
  }

  if (execution_config_.used_for_inference) {
    for (auto& hook : output_hookfuncs_) {
      hook(op, local_scope);
    }
  }

  // for debug
  if (FLAGS_save_static_runtime_data) {
    VLOG(6) << "start to save paddle variable";
    auto root_path = FLAGS_static_runtime_data_save_path;
    for (auto& vname : op->InputVars()) {
      auto* var = local_scope->FindVar(vname);
      if (var == nullptr) continue;
      const phi::DenseTensor* tensor{nullptr};
      if (var->IsType<phi::DenseTensor>()) {
        tensor = &var->Get<phi::DenseTensor>();
      } else {
        VLOG(6) << vname << " is not DenseTensor";
        continue;
      }
      if (!tensor->IsInitialized()) continue;
      paddle::framework::SaveTensor(
          *tensor,
          root_path + "/saved_tensors/" + op->Type() + "-input-" + vname,
          false);
    }
    for (auto& vname : op->OutputVars(true)) {
      auto* var = local_scope->FindVar(vname);
      if (var == nullptr) continue;
      const phi::DenseTensor* tensor{nullptr};
      if (var->IsType<phi::DenseTensor>()) {
        tensor = &var->Get<phi::DenseTensor>();
      } else {
        VLOG(6) << vname << "  is not DenseTensor";
        continue;
      }
      if (!tensor->IsInitialized()) continue;
      paddle::framework::SaveTensor(
          *tensor,
          root_path + "/saved_tensors/" + op->Type() + "-output-" + vname,
          false);
    }
    VLOG(6) << "end save paddle variable";
  }

  // for debug nan/inf
  if (op_with_kernel != nullptr && FLAGS_check_nan_inf) {
    VLOG(4) << "Check nan/inf";
    try {
      framework::details::CheckOpHasNanOrInf(
          *op,
          *local_scope,
          place);  // TODO(xiongkun03) change it to inner scope.
    } catch (...) {
      const std::vector<std::string>* callstack = nullptr;
      auto attrs = op->Attrs();
      auto iter =
          attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
      if (iter != attrs.end()) {
        callstack = &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
        if (callstack->empty()) callstack = nullptr;
      }
      std::ostringstream sout;
      if (callstack) {
        if (FLAGS_call_stack_level > 1) {
          sout << "\n\n  Compile Traceback (most recent call last):";
        } else {
          sout << "In user code:\n";
        }
        for (auto& line : *callstack) {
          sout << "\n  " << line;
        }
      }
      std::cout << sout.str() << std::endl;
      std::rethrow_exception(std::current_exception());
    }
  }
}

void ProgramInterpreter::RunInstruction(const Instruction& instr_node) {
  VLOG(5) << __func__ << " OP id:" << instr_node.Id()
          << " name:" << instr_node.OpBase()->Type() << " type:"
          << (instr_node.KernelType() == OpFuncType::kCpuSync
                  ? "kCpuSync"
                  : (instr_node.KernelType() == OpFuncType::kGpuSync
                         ? "kGpuSync"
                         : "kGpuAsync"))
          << " runs on " << phi::GetCurrentThreadName();

  auto* op = instr_node.OpBase();
  phi::RecordEvent instruction_event(
      op->Type(), phi::TracerEventType::Operator, 1);

  SetDeviceId(instr_node.DeviceContext().GetPlace());

  try {
    instr_node.WaitEvent(place_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (enable_job_schedule_profiler_) {
      if (!calculate_stream_timer_->IsStarted() && op->Type() != "feed" &&
          !interpreter::IsCommunicationOp(instr_node)) {
        VLOG(3) << "Start calculated stream timer from op: " << op->Type();
        calculate_stream_timer_->Start();
      }
    }
#endif

    if (!instr_node.IsArtificial()) {
      RunOperator(instr_node);
      CheckGC(instr_node);
      if (FLAGS_log_memory_stats) {
        memory::LogDeviceMemoryStats(place_, instr_node.OpBase()->Type());
      }
    }

    instr_node.RecordEvent(place_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (enable_job_schedule_profiler_) {
      if (instr_node.Id() == last_calculate_instr_id_ &&
          calculate_stream_timer_->IsStarted()) {
        VLOG(3) << "Stop calculated stream timer from op: " << op->Type();
        calculate_stream_timer_->Stop();
      }
    }
#endif
  } catch (platform::EnforceNotMet& ex) {
    framework::InsertCallStackInfo(op->Type(), op->Attrs(), &ex);
    exception_holder_.Catch(std::make_exception_ptr(ex));
  } catch (platform::EOFException&) {
    exception_holder_.Catch(std::current_exception());
  } catch (std::exception& ex) {
    LOG(WARNING) << op->Type() << " raises an exception "
                 << common::demangle(typeid(ex).name()) << ", " << ex.what();
    exception_holder_.Catch(std::current_exception());
  } catch (...) {
    LOG(WARNING) << op->Type() << " raises an unknown exception";
    exception_holder_.Catch(std::current_exception());
  }
}

std::string ProgramInterpreter::GetDepsString() const {
  std::stringstream ss;
  auto downstream_map = dependency_builder_.OpDownstreamMap();
  ss << "Note: when static_dep is 1, it is ok that the dynamic_dep will not "
        "be decreased to 0."
     << std::endl;
  ss << "unfinished_op_number_:" << unfinished_op_number_ << std::endl;
  for (size_t i = 0; i < deps_.size(); ++i) {
    ss << "op:" << i << ", type: " << vec_instruction_[i].OpBase()->Type()
       << ", static_dep:" << deps_[i]->StaticDep()
       << ", dynamic_dep:" << deps_[i]->DynamicDep() << ", downstream op: ";
    for (auto id : downstream_map[i]) {
      ss << id << ", ";
    }
    ss << std::endl;
  }
  return ss.str();
}

void ProgramInterpreter::ExecuteInstructionList(
    const std::vector<Instruction>& vec_instr) {
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  if (enable_job_schedule_profiler_) {
    for (int i = vec_instr.size() - 1; i >= 0; --i) {
      auto& instr_node = vec_instr[i];
      if (!interpreter::IsCommunicationOp(instr_node)) {
        VLOG(3) << "Last calculated op type: " << instr_node.OpBase()->Type();
        last_calculate_instr_id_ = instr_node.Id();
        break;
      }
    }
  }

  for (size_t i = 0; i < dependency_count_->size(); ++i) {
    if ((*dependency_count_)[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i));
      if (FLAGS_new_executor_serial_run) {
        RunInstructionAsync(i);
      } else {
        async_work_queue_->AddTask(vec_instr.at(i).KernelType(),
                                   [this, i] { RunInstructionAsync(i); });
      }
    }
  }

  // For debug hang in main_thread_blocker_.WaitEvent(),
  // launch async task to log deps every
  // FLAGS_executor_log_deps_every_microseconds, then cancel the std::async when
  // main_thread_blocker_.WaitEvent() executed. Why not use std::async instead
  // of workqueue? To make sure that the logging thread itself will not affect
  // the workqueue
  //  used in interpretercore.

  std::future<int> logged_times;
  std::atomic_bool cancel_log = ATOMIC_VAR_INIT(false);
  if (FLAGS_executor_log_deps_every_microseconds) {
    logged_times = std::async(
        std::launch::async,
        [this](const std::atomic_bool& cancel) {
          int times = 0;
          while (!cancel) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                FLAGS_executor_log_deps_every_microseconds));
            // check again, since cancel may be changed during sleep
            if (cancel) {
              break;
            }
            VLOG(0) << "deps:\n" << GetDepsString();
            times++;
          }
          return times;
        },
        std::ref(cancel_log));
  }

  auto event_name = main_thread_blocker_.WaitEvent();
  VLOG(1) << "main_thread_blocker_(" << &main_thread_blocker_
          << ") got event_name: " << event_name;

  cancel_log = true;
  if (logged_times.valid()) {
    VLOG(1) << "Logged deps for " << logged_times.get() << " times";
  }

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(1) << "Exception caught " << exception_holder_.Type();
    // Graceful exit when the executor encountered a fatal error.
    // EOF is not a fatal error.
    if (exception_holder_.Type() != "EOF") {
      async_work_queue_->Cancel();
      async_work_queue_.reset();
    }
    VLOG(4) << "Cancel ok";
    PADDLE_ENFORCE_EQ(
        main_thread_blocker_.Clear(),
        0,
        common::errors::PreconditionNotMet(
            "main_thread_blocker_.Clear() return -1, clear failed"));
    VLOG(4) << "clear ok";
    exception_holder_.ReThrow();
  }
}

void ProgramInterpreter::RunNextInstructions(
    const Instruction& instr, SchedulingQueue* reserved_next_ops) {
  phi::RecordEvent record(
      "RunNextInstructions", phi::TracerEventType::UserDefined, 10);

  auto IsReady = [this](size_t next_id) {
    VLOG(4) << "op_id: " << next_id
            << ", remain deps: " << deps_[next_id]->DynamicDep();
    return deps_[next_id]->CheckAndDecrease();
  };

  for (size_t next_instr_id : instr.NextInstrsInDifferenceThread()) {
    if (IsReady(next_instr_id)) {
      async_work_queue_->AddTask(
          vec_instruction_[next_instr_id].KernelType(),
          [this, next_instr_id]() { RunInstructionAsync(next_instr_id); });
    }
  }

  for (size_t next_instr_id : instr.NextInstrsInSameThread()) {
    if (IsReady(next_instr_id)) {
      reserved_next_ops->push(next_instr_id);
    }
  }
}

void ProgramInterpreter::RunInstructionAsync(size_t instr_id) {
  // NOTE(Ruibiao): Due to the uncertain order in multi-threading asynchronous
  // scheduling, the priority order involved cross-thread scheduling is not
  // guaranteed. Only Ops scheduled by the same AddTask call have the guarantee
  // of priority order.
  SchedulingQueue ready_ops(instruction_scheduling_priority_less);
  ready_ops.push(instr_id);
  while (!ready_ops.empty()) {
    instr_id = ready_ops.top();
    ready_ops.pop();
    auto& instr_node = vec_instruction_.at(instr_id);

    RunInstruction(instr_node);

    if (UNLIKELY(exception_holder_.IsCaught())) {
      VLOG(4) << "Exception caught";
      if (exception_notifier_ != nullptr) {
        exception_notifier_->NotifyEvent();
      }
      return;
    }

    VLOG(4) << "unfinished_op_number_: " << unfinished_op_number_;
    if (UNLIKELY(unfinished_op_number_.fetch_sub(
                     1, std::memory_order_relaxed) == 1)) {
      if (completion_notifier_ != nullptr) {
        completion_notifier_->NotifyEvent();
      }
    }

    RunNextInstructions(instr_node, &ready_ops);
  }
}

void ProgramInterpreter::RecordStreamForGC(const Instruction& instr) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  PADDLE_THROW(common::errors::Unimplemented(
      "RecordStreamForGC is only implemented when compiled with GPU."));
#else
  phi::RecordEvent record(
      "RecordStreamForGC", phi::TracerEventType::UserDefined, 10);

  auto TensorRecordStream = [](phi::DenseTensor& tensor,
                               const gpuStream_t& stream) {
    auto allocation = tensor.Holder();
    if (allocation == nullptr) {
      return;
    }

    const phi::Place& place = allocation->place();
    if (phi::is_gpu_place(place)) {
      memory::RecordStream(allocation, stream);
    } else if (phi::is_cuda_pinned_place(place)) {
      // TODO(Ruibiao): Here should do something to make sure that the tensor
      // is not freed until the H2D copies done. However, simply launch a
      // CUDA runtime callback to the H2D stream may lead a high performance
      // overhead. As all the cases we meet in H2D are copies from CPUPlace at
      // present, we just log a WARNING here. A better design is required.
      LOG(WARNING) << "Copy data from a CUDAPinned tensor in an asynchronous "
                      "manner may lead a data inconsistent";
    } else {
      // memory copies involve CPUPlace are always synchronous, so just do
      // nothing here
    }
  };

  /* NOTE(Ruibiao)：Cross-stream tensor synchronization is required only when
   * all the following conditions are satisfied:
   * 1. The tensor will be GC after running the instruction, i.e., in
   * instr.GCCheckVars.
   * 2. The stream which initializes this tensor is different from the stream
   * which the instruction run in.
   * 3. The tensor is the instruction's input, cause we assume that
   * instruction will initialize all output tensors with its running stream.
   * 4. In the OP function of this instruction, the tensor is an input of a
   * async CUDA kernel.
   *
   * Here we only process the first condition, because:
   * 1. Since the RecordStream function will directly return when the recorded
   * stream is equal to the owning stream, recording a stream same as which
   * initialized this tensor has less time overhead. Conversely, it may take
   * more time if we try to extract those cross-stream input vars from
   * instr.GCCheckVars.
   * 2. Now the instruction has no idea of which vars involving async running
   * in OP function, and thus we can not recognize condition 4. It should be
   * supported later.
   */
  for (int var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC sync " << var_scope_.GetNameById(var_id) << " "
            << var_scope_.VarDesc(var_id);

    paddle::framework::Variable* var = var_scope_.VarRef(var_id);
    if (var == nullptr) {
      continue;
    }

    if (var->IsType<phi::DenseTensor>()) {
      TensorRecordStream(*(var->GetMutable<phi::DenseTensor>()), instr.stream_);
    } else if (
        var->IsType<
            operators::reader::
                OrderedMultiDeviceDenseTensorBlockingQueueHolder>()) {  // NOLINT
      // do nothing
    } else if (var->IsType<phi::SelectedRows>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SelectedRows>()->mutable_value()),
          instr.stream_);
    } else if (var->IsType<phi::TensorArray>()) {
      auto* tensor_arr = var->GetMutable<phi::TensorArray>();
      for (auto& tensor : *tensor_arr) {
        TensorRecordStream(tensor, instr.stream_);
      }
    } else if (var->IsType<phi::SparseCooTensor>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCooTensor>()->mutable_indices()),
          instr.stream_);
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCooTensor>()->mutable_values()),
          instr.stream_);
    } else if (var->IsType<phi::SparseCsrTensor>()) {
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCsrTensor>()->mutable_cols()),
          instr.stream_);
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCsrTensor>()->mutable_crows()),
          instr.stream_);
      TensorRecordStream(
          *(var->GetMutable<phi::SparseCsrTensor>()->mutable_values()),
          instr.stream_);
    } else if (var->IsType<std::vector<Scope*>>()) {
      // do nothing
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "The variable(%s) is not supported in eager deletion.",
          framework::ToTypeName(var->Type())));
    }
  }
#endif
}

void ProgramInterpreter::CheckGC(const Instruction& instr) {
  phi::RecordEvent record("CheckGC", phi::TracerEventType::UserDefined, 10);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (instr.need_record_stream_for_gc_) {
    RecordStreamForGC(instr);
  }
#endif
  auto& var_scope = var_scope_;

  for (auto var_id : instr.GCCheckVars()) {
    VLOG(4) << "GC:" << var_scope_.GetNameById(static_cast<int>(var_id))
            << ", id:" << var_id << ", ref:" << refs_[var_id]->DynamicRef();
    bool is_ready = refs_[var_id]->CheckAndDecrease();
    if (is_ready) {
      VLOG(6) << "Async delete variable with name : "
              << var_scope.GetNameById(static_cast<int>(var_id));
      gc_->Add(refs_[var_id]->Var(), instr);
    }
  }
}

void ProgramInterpreter::Prepare(
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors,
    bool prepare_feed,
    bool switch_stream) {
  PADDLE_ENFORCE_EQ(feed_names.size(),
                    feed_tensors.size(),
                    common::errors::PreconditionNotMet(
                        "Required feed_names.size() == feed_tensors.size(), "
                        "but received %d != %d",
                        feed_names.size(),
                        feed_tensors.size()));
  auto FeedInput = [&] {
    VLOG(4) << "Feed inputs";
    for (size_t i = 0; i < feed_names.size(); ++i) {
      auto* feed_var = local_scope_->FindVar(feed_names[i]);
      PADDLE_ENFORCE_NOT_NULL(
          feed_var,
          common::errors::NotFound("Variable %s should not be nullptr.",
                                   feed_names[i]));

      auto feed_tensor = feed_var->GetMutable<phi::DenseTensor>();
      feed_tensor->ShareDataWith(feed_tensors[i]);
      feed_tensor->set_lod(feed_tensors[i].lod());
    }
  };

  if (!is_build_ || switch_stream) {
    paddle::framework::interpreter::BuildVariableScope(
        block_, execution_config_, &var_scope_);
    FeedInput();
    std::vector<paddle::framework::OpFuncNode> op_func_nodes;
    paddle::framework::interpreter::BuildOpFuncList(
        place_,
        block_,
        execution_config_.skip_gc_vars,
        &op_func_nodes,
        &var_scope_,
        execution_config_,
        input_hookfuncs_,
        output_hookfuncs_,
        HasLocalScope(),
        static_build_);
    SetFeedVarsInplaceSkip(feed_names);
    // convert vec func_list to graph
    Convert(&op_func_nodes);
    UpdateSyncOpNum();
    if (static_build_) {
      VLOG(4) << "RUN impl";
      RunImpl();
    }
    BuildSkipShareLoDInfo();
    is_build_ = true;
    is_shared_results_build_ = true;
  }
  // NOTE: Because feed_tensor will be GC after
  // paddle::framework::BuildOpFuncList, so we should
  // call FeedInput again.
  if (prepare_feed) {
    FeedInput();
  }
}

std::shared_ptr<ProgramDesc> ProgramInterpreter::GetMutableCopyProgram() {
  return copy_program_;
}

void ProgramInterpreter::SetFeedVarsInplaceSkip(
    const std::vector<std::string>& feed_names) {
  for (auto& feed_name : feed_names) {
    var_scope_.SetVarSkipInplace(feed_name, true);
  }
}

bool ProgramInterpreter::HasLocalScope() const {
  return local_scope_ != nullptr;
}

// Note(zhangbo):
// (1) What is "Trace"?
// The OP execute scheduling rule adopted by Interpretercore by default is a
// multi-threaded scheduling mode(see ExecuteInstructionList). By maintaining a
// high-performance thread pool, the OP's execute scheduling is distributed to
// the sub threads maintained by the thread pool, but the main thread does not
// have any tasks. In Trace mode, the executor will execute directly in the main
// thread according to the pre provided OP sequence(trace_execute_order_),
// instead of being distributed to the thread pool.
// (2) When we use "Trace"?
// In dygraph to static, This scheduling causes that the execution of the
// forward and backward OPs and the execution of the dygraph optimizer cannot be
// executed in the same thread. Executing thread switch may cause cpu cache
// miss. When a model is all KQueueAsync type OPs, all OPs will be distributed
// to the DeviceThread for execution, and the multithreading scheduling will not
// have any benefits. Therefore, in the dynamic to static, when the number of
// KQueueSync Ops is 0, we choose Trace mode.
void ProgramInterpreter::TraceInstructionList(
    const std::vector<Instruction>& vec_instr) {
  unfinished_op_number_ = vec_instr.size();
  if (unfinished_op_number_ == 0) {
    VLOG(4) << "No op to run, return";
    return;
  }

  exception_holder_.Clear();

  for (size_t i = 0; i < dependency_count_->size(); ++i) {
    if ((*dependency_count_)[i] == 0) {
      // NOTE(zhiqiu): hot fix for jit input var
      RecordMemcpyD2H(vec_instr.at(i));
    }
  }

  for (auto instr_id : trace_execute_order_) {
    auto& instr_node = vec_instruction_.at(instr_id);

    RunInstruction(instr_node);

    if (UNLIKELY(exception_holder_.IsCaught())) {
      VLOG(4) << "Exception caught";
      break;
    }
  }

  if (UNLIKELY(exception_holder_.IsCaught())) {
    VLOG(1) << "Exception caught " << exception_holder_.Type();
    PADDLE_ENFORCE_EQ(
        main_thread_blocker_.Clear(),
        0,
        common::errors::PreconditionNotMet(
            "main_thread_blocker_.Clear() return -1, clear failed"));
    VLOG(4) << "clear ok";
    exception_holder_.ReThrow();
  }
}

void ProgramInterpreter::RecordMemcpyD2H(const Instruction& instr_node) {
  // NOTE(zhiqiu): hot fix for jit input var
  if (instr_node.OpBase()->Type() == interpreter::kMemcpyD2H) {
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* default_dev_ctx = pool.Get(place_);
    for (auto& event : instr_node.EventsToWait()) {
      phi::RecordEvent record(
          "RecordStreamEvent", phi::TracerEventType::UserDefined, 10);
      VLOG(3) << "Record event on default stream in jit_input_var at op: "
              << instr_node.OpBase()->Type();
      event.event_->Record(default_dev_ctx);
    }
  }
}

void ProgramInterpreter::UpdateSyncOpNum() {
  int64_t sync_op_num = 0;
  for (auto& ins : vec_instruction_) {
    if (ins.KernelType() == OpFuncType::kCpuSync ||
        ins.KernelType() == OpFuncType::kGpuSync) {
      sync_op_num = sync_op_num + 1;
    }
  }
  sync_op_num_ = sync_op_num;
  VLOG(4) << "Update sync op num, sync op num is: " << sync_op_num_;
}

// Note(zhangbo):
// When there is a KQueueSync type OP in the model, breadth traversal is better
// than depth traversal. For example: OP(O) ->(direct_run)-> OP(A)
// ->(sync_run)-> OP(B) OP(O) ->(direct_run)-> OP(C) ->(direct_run)-> OP(D) If B
// is run before C, B may always block to wait for A to finish executing, but in
// fact, C can be executed first during this time.
void ProgramInterpreter::AnalyseExecuteOrderForTrace() {
  VLOG(4) << "Analyze the execution order of Trace scheduling mode.";
  interpreter::ResetAtomicGuard guard(&deps_, &refs_);

  auto op_downstream_map = dependency_builder_.OpDownstreamMap();

  auto IsReady = [this](size_t next_id) {
    VLOG(4) << "op_id: " << next_id
            << ", remain deps: " << deps_[next_id]->DynamicDep();
    return deps_[next_id]->CheckAndDecrease();
  };

  std::vector<size_t> trace_order;
  SchedulingQueue ready_ops(instruction_scheduling_priority_less);

  for (size_t instr_id = 0; instr_id < dependency_count_->size(); ++instr_id) {
    if ((*dependency_count_)[instr_id] == 0) {
      ready_ops.push(instr_id);
    }
  }

  while (!ready_ops.empty()) {
    size_t now_id = ready_ops.top();
    ready_ops.pop();
    trace_order.push_back(now_id);

    auto next_op_set = op_downstream_map[now_id];

    for (size_t next_op_id : next_op_set) {
      if (IsReady(next_op_id)) {
        ready_ops.push(next_op_id);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      trace_order.size(),
      dependency_count_->size(),
      common::errors::PreconditionNotMet(
          "trace_order size should be equal to dependency_count_."));

  trace_execute_order_ = trace_order;

  if (VLOG_IS_ON(6)) {
    std::stringstream ss;
    ss << "trace order: ";
    for (size_t idx = 0; idx < trace_execute_order_.size(); idx++) {
      ss << vec_instruction_[trace_execute_order_[idx]]
                .OpFunc()
                ->operator_base_->Type()
         << "[" << trace_execute_order_[idx] << "]"
         << " -> ";
    }
    ss << "end\n";
    VLOG(6) << ss.str();
  }
}

Variable* ProgramInterpreter::DebugVar(const std::string& name) const {
  PADDLE_THROW(common::errors::Unimplemented(
      "DebugVar is not implemented in ProgramInterpreter."));
}
}  // namespace paddle::framework
