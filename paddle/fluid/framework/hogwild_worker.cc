/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <array>
#include <ctime>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/barrier.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/fluid/platform/densetensor_printer.h"
#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#endif
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/program_utils.h"
#include "paddle/utils/string/string_helper.h"

COMMON_DECLARE_bool(enable_exit_when_partial_worker);
COMMON_DECLARE_int32(enable_adjust_op_order);
PHI_DEFINE_EXPORTED_bool(gpugraph_force_device_batch_num_equal,
                         false,
                         "enable force_device_batch_num_equal, default false");
COMMON_DECLARE_bool(enable_dump_main_program);
PHI_DEFINE_EXPORTED_int32(gpugraph_offload_param_stat,
                          0,
                          "enable offload param stat, default 0");
PHI_DEFINE_EXPORTED_string(gpugraph_offload_param_extends,
                           ".w_0_moment,.b_0_moment",
                           "offload param extends list");
PHI_DEFINE_EXPORTED_int32(gpugraph_offload_gather_copy_maxsize,
                          16,
                          "offload gather copy max size , default 16M");
PHI_DEFINE_EXPORTED_int32(gpugraph_parallel_copyer_split_maxsize,
                          64,
                          "offload gather copy max size , default 64M");
PHI_DEFINE_EXPORTED_int32(gpugraph_parallel_stream_num,
                          8,
                          "offload parallel copy stream num");
PHI_DEFINE_EXPORTED_bool(gpugraph_enable_print_op_debug,
                         false,
                         "enable print op debug ,default false");

namespace paddle::framework {

std::atomic<bool> HogwildWorker::quit_flag_(false);
Barrier g_barrier;

#if defined(PADDLE_WITH_CUDA)
class GPUParallelCopyer {
 public:
  GPUParallelCopyer(const phi::gpuStream_t &stream,
                    const int device_id,
                    const int stream_num)
      : dev_stream_(stream),
        device_id_(device_id),
        max_stream_(stream_num),
        streams_(stream_num) {
    platform::CUDADeviceGuard guard(device_id_);
    for (size_t i = 0; i < max_stream_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&streams_[i]));
    }
  }
  ~GPUParallelCopyer() {
    platform::CUDADeviceGuard guard(device_id_);
    for (size_t i = 0; i < max_stream_; ++i) {
      PADDLE_WARN_GPU_SUCCESS(cudaStreamDestroy(streams_[i]));
    }
  }
  void Copy(const phi::DenseTensor &src_tensor,
            const phi::Place &dest_place,
            phi::DenseTensor *dest_tensor) {
    size_t mem_len = src_tensor.memory_size();
    if (!dest_tensor->IsInitialized()) {
      dest_tensor->Resize(src_tensor.dims());
      dest_tensor->set_layout(src_tensor.layout());
    }

    const char *src_ptr = (const char *)src_tensor.data();
    char *dest_ptr = reinterpret_cast<char *>(
        dest_tensor->mutable_data(dest_place, src_tensor.dtype(), mem_len));
    if (copy_count_ == 0) {
      platform::GpuStreamSync(dev_stream_);
    }
    size_t pos = 0;
    auto &src_place = src_tensor.place();
    while (pos < mem_len) {
      size_t data_len = mem_len - pos;
      if (data_len > split_max_len_) {
        data_len = split_max_len_;
      }
      auto &cur_stream = streams_[copy_count_ % max_stream_];
      const char *src = src_ptr + pos;
      char *dst = dest_ptr + pos;
      memory::Copy(dest_place, dst, src_place, src, data_len, cur_stream);
      pos = pos + split_max_len_;
      ++copy_count_;
    }
  }
  void Wait() {
    if (copy_count_ == 0) {
      return;
    }
    if (copy_count_ > max_stream_) {
      for (auto &ss : streams_) {
        platform::GpuStreamSync(ss);
      }
    } else {
      for (size_t i = 0; i < copy_count_; ++i) {
        platform::GpuStreamSync(streams_[i]);
      }
    }
    copy_count_ = 0;
  }
  void SyncDevStream() { platform::GpuStreamSync(dev_stream_); }

 private:
  phi::gpuStream_t dev_stream_ = nullptr;
  int device_id_ = -1;
  size_t max_stream_ = 0;
  std::vector<phi::gpuStream_t> streams_;
  size_t copy_count_ = 0;
  size_t split_max_len_ =
      FLAGS_gpugraph_parallel_copyer_split_maxsize * 1024 * 1024;
};
#endif
template <typename TStream>
inline void Tensor2Pinned(phi::DenseTensor *tensor, const TStream &stream) {
#if defined(PADDLE_WITH_CUDA)
  const size_t mem_len = tensor->memory_size();
  auto place = phi::GPUPinnedPlace();
  auto holder = memory::AllocShared(place, mem_len);
  memory::Copy(
      place, holder->ptr(), tensor->place(), tensor->data(), mem_len, stream);
  tensor->ResetHolderWithType(holder, tensor->dtype());
#endif
}
template <typename TCopyer>
void HogwildWorker::OffLoadVarInfo::CopyInputs(const Scope *root,
                                               const phi::Place &place,
                                               Scope *scope,
                                               TCopyer *copyer) {
  if (!cast_vars.empty()) {
    for (auto &obj : cast_vars) {
      auto src_var = root->FindLocalVar(obj.second);
      PADDLE_ENFORCE_NE(
          src_var,
          nullptr,
          common::errors::NotFound("root scope not found var name=%s",
                                   obj.second.c_str()));
      auto &src_tensor = src_var->Get<phi::DenseTensor>();
      auto dest_var = scope->FindLocalVar(obj.first);
      PADDLE_ENFORCE_NE(dest_var,
                        nullptr,
                        common::errors::NotFound("dest name=%s is nullptr",
                                                 obj.first.c_str()));
      auto *dest_tensor = dest_var->GetMutable<phi::DenseTensor>();
      auto dtype = framework::TransToProtoVarType(dest_tensor->dtype());
      framework::TransDataType(src_tensor, dtype, dest_tensor);
    }
  }
#if defined(PADDLE_WITH_CUDA)
  if (copy_vars.empty()) {
    return;
  }
  for (auto &name : copy_vars) {
    auto src_var = root->FindLocalVar(name);
    PADDLE_ENFORCE_NE(src_var,
                      nullptr,
                      common::errors::NotFound(
                          "root scope not found var name=%s", name.c_str()));
    auto &src_tensor = src_var->Get<phi::DenseTensor>();
    auto dest_var = scope->FindLocalVar(name);
    PADDLE_ENFORCE_NE(
        dest_var,
        nullptr,
        common::errors::NotFound("dest name=%s is nullptr", name.c_str()));
    auto *dest_tensor = dest_var->GetMutable<phi::DenseTensor>();
    copyer->Copy(src_tensor, place, dest_tensor);
  }
  copyer->Wait();
#endif
}
template <typename TCopyer>
void HogwildWorker::OffLoadVarInfo::BackUpInputs(Scope *root_scope,
                                                 Scope *scope,
                                                 TCopyer *copyer) {
#if defined(PADDLE_WITH_CUDA)
  if (backup_vars.empty()) {
    return;
  }
  for (auto &name : backup_vars) {
    auto var = scope->FindLocalVar(name);
    if (var == nullptr) {
      continue;
    }
    auto src_tensor = var->Get<phi::DenseTensor>();
    auto root_var = root_scope->FindLocalVar(name);
    if (root_var == nullptr) {
      root_var = root_scope->Var(name);
      auto root_tensor = root_var->GetMutable<phi::DenseTensor>();
      auto place = phi::GPUPinnedPlace();
      copyer->Copy(src_tensor, place, root_tensor);
    } else {
      auto root_tensor = root_var->GetMutable<phi::DenseTensor>();
      if (root_tensor->IsInitialized() &&
          !phi::is_gpu_place(root_tensor->place())) {
        copyer->Copy(src_tensor, root_tensor->place(), root_tensor);
      }
    }
  }
  copyer->Wait();
#endif
}

void HogwildWorker::Initialize(const TrainerDesc &desc) {
  fetch_config_ = desc.fetch_config();
  param_ = desc.hogwild_param();
  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  use_cvm_ = desc.use_cvm();
  thread_barrier_ = desc.thread_barrier();
  use_ps_gpu_ = desc.use_ps_gpu();
  use_gpu_graph_ = desc.use_gpu_graph();

  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    std::string name = param_.stat_var_names(i);
    stat_var_name_map_[name] = 1;
    skip_vars_.push_back(name);
  }
  is_offload_communication_ = (FLAGS_gpugraph_offload_param_stat & 0x01);
  is_offload_param_ = (FLAGS_gpugraph_offload_param_stat & 0x02);
  // split extends
  offload_exts_ =
      paddle::string::split_string(FLAGS_gpugraph_offload_param_extends, ",");
  if (is_offload_param_ && !offload_exts_.empty()) {
    VLOG(0) << "need offload extends="
            << paddle::string::join_strings(offload_exts_, ",");
  }
}
int HogwildWorker::IsParameter(const std::string &name, bool full_match) {
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  auto gpu_ps = PSGPUWrapper::GetInstance();
  bool last_device = ((thread_num_ - 1) == thread_id_);
  if (full_match) {
    auto it = params2rootid_.find(name);
    if (it == params2rootid_.end()) {
      return -1;
    }
    if (use_gpu_graph_ && use_ps_gpu_) {
      if (last_device && !gpu_ps->IsKeyForSelfRank(it->second)) {
        free_param_vars_.insert(name);
      }
    }
    if (it->second == nccl_rank_id_) {
      return 1;
    }
    return 0;
  } else {
    // moment, acc
    for (auto it = params2rootid_.begin(); it != params2rootid_.end(); ++it) {
      if (strncmp(name.c_str(), it->first.c_str(), it->first.length()) != 0) {
        continue;
      }
      if (use_gpu_graph_ && use_ps_gpu_) {
        if (last_device && !gpu_ps->IsKeyForSelfRank(it->second)) {
          free_param_vars_.insert(name);
        }
      }
      if (it->second == nccl_rank_id_) {
        return 1;
      }
      return 0;
    }
    return -1;
  }
#else
  return -1;
#endif
}
void HogwildWorker::BuildShardingDepends(const ProgramDesc &program) {
  nccl_rank_id_ =
      static_cast<int>(static_cast<unsigned char>(place_.GetDeviceId()));
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  if (use_gpu_graph_ && use_ps_gpu_) {
    auto gpu_ps = PSGPUWrapper::GetInstance();
    nccl_rank_id_ = gpu_ps->GetNCCLRankId(nccl_rank_id_);
    is_multi_node_ = (gpu_ps->GetRankNum() > 1);
  }
#endif

  auto &block = program.Block(0);
  auto all_desc = block.AllOps();

  bool is_has_sync_comm_stream = false;
  for (auto &op_desc : all_desc) {
    // broadcast op
    if (op_desc->Type() != "c_broadcast") {
      // has sync comm stream
      if (op_desc->Type() == "c_sync_comm_stream") {
        is_has_sync_comm_stream = true;
      }
      continue;
    }
    int root_id = op_desc->GetAttrIfExists<int>("root");
    int ring_id = op_desc->GetAttrIfExists<int>("ring_id");
    if (ring_id >= 0 && ring_id != ring_id_) {
      ring_id_ = ring_id;
    }
    std::string new_name;
    for (auto &o : op_desc->Inputs()) {
      for (auto &name : o.second) {
        // amp
        size_t pos = name.find(".cast_fp16");
        if (pos != std::string::npos) {
          new_name = name.substr(0, pos);
          cast_fp16_vars_.insert(std::make_pair(name, new_name));
          param_cast_vars_.insert(std::make_pair(new_name, name));
          if (root_id == nccl_rank_id_) {
            need_cast_vars_.insert(std::make_pair(name, new_name));
          }
        } else {
          new_name = name;
        }
        auto var = block.FindVar(new_name);
        if (!var->Persistable() || !var->IsParameter()) {
          continue;
        }
        if (params2rootid_.find(new_name) != params2rootid_.end()) {
          continue;
        }
        params2rootid_.insert(std::make_pair(new_name, root_id));
      }
    }
  }
  // adjust op order need sync comm stream op
  enable_adjust_op_order_ =
      (is_has_sync_comm_stream && FLAGS_enable_adjust_op_order);
  if (params2rootid_.empty()) {
    return;
  }
  sharding_mode_ = true;
  // check find
  for (auto &var : block.AllVars()) {
    if (!var->Persistable()) {
      continue;
    }
    int ret = IsParameter(var->Name(), var->IsParameter());
    if (ret < 0 || ret == 1) {
      if (ret == 1) {
        persist_param_vars_.insert(var->Name());
      }
      continue;
    }
    if (var->IsParameter()) {
      unpersist_vars_.insert(var->Name());
    } else {
      remove_vars_.insert(var->Name());
    }
  }
  int total_broadcast = 0;
  int remove_broadcast = 0;
  int remove_sync_stream = 0;
  int remove_cast_op = 0;
  std::multiset<std::string> param2refs;
  std::multiset<std::string> out2refs;
  for (auto &op_desc : all_desc) {
    bool find = false;
    if (op_desc->Type() == "c_sync_calc_stream") {  // remove error sync
      auto &inputs = op_desc->Input("X");
      std::vector<std::string> removenames;
      for (auto &name : inputs) {
        auto it = out2refs.find(name);
        if (it != out2refs.end()) {
          removenames.push_back(name);
          continue;
        }
        find = true;
        ++remove_sync_stream;
        break;
      }
      if (!removenames.empty()) {
        for (auto &name : removenames) {
          auto it = out2refs.find(name);
          if (it == out2refs.end()) {
            continue;
          }
          out2refs.erase(it);
        }
      }
    } else if (!param_cast_vars_.empty() && op_desc->Type() == "cast") {  // AMP
      auto &inputs = op_desc->Input("X");
      for (auto &name : inputs) {
        auto it = param_cast_vars_.find(name);
        if (it == param_cast_vars_.end()) {
          break;
        }
        find = true;
        ++remove_cast_op;
        break;
      }
    } else if (is_offload_communication_ && op_desc->Type() == "c_broadcast") {
      ++total_broadcast;
      // single node p2p copy
      if (!is_multi_node_ && cast_fp16_vars_.empty()) {
        find = true;
        ++remove_broadcast;
      } else {
        auto &inputs = op_desc->Input("X");
        for (auto &name : inputs) {
          if (cast_fp16_vars_.find(name) != cast_fp16_vars_.end()) {
            break;
          }
          if (param2refs.find(name) != param2refs.end()) {
            find = true;
            continue;
          }
          param2refs.insert(name);
        }
        if (find) {
          ++remove_broadcast;
        }
      }
    } else {
      for (auto &o : op_desc->Inputs()) {
        for (auto &name : o.second) {
          if (remove_vars_.find(name) == remove_vars_.end()) {
            continue;
          }
          find = true;
          break;
        }
        if (find) {
          break;
        }
      }
    }
    if (find) {
      remove_ops_.insert(op_desc);
    } else {
      for (auto &o : op_desc->Outputs()) {
        for (auto &name : o.second) {
          out2refs.insert(name);
        }
      }
    }
  }
  // add offload
  if (is_offload_communication_) {
    for (auto &it : params2rootid_) {
      if (it.second == nccl_rank_id_) {
        continue;
      }
      if (param_cast_vars_.find(it.first) != param_cast_vars_.end()) {
        continue;
      }
      offload_names_.insert(it.first);
    }
  }

  // reset dump param
  if (need_dump_param_ && dump_param_ != nullptr) {
    for (auto &name : *dump_param_) {
      auto var = block.FindVar(name);
      if (var == nullptr) {
        continue;
      }
      std::string new_name = name;
      size_t pos = new_name.find("@");
      if (pos != std::string::npos) {
        new_name = name.substr(0, pos);
      }
      if (persist_param_vars_.find(new_name) == persist_param_vars_.end()) {
        continue;
      }
      shard_dump_params_.push_back(name);
    }
    dump_param_ = &shard_dump_params_;
  }
  // reset dump fields
  if (need_dump_field_ && dump_fields_ != nullptr) {
    for (auto &name : *dump_fields_) {
      auto var = block.FindVar(name);
      if (var == nullptr) {
        continue;
      }
      if (remove_vars_.find(name) != remove_vars_.end()) {
        continue;
      }
      shard_dump_fields_.push_back(name);
    }
    dump_fields_ = &shard_dump_fields_;
  }
  VLOG(0) << "device id=" << int(place_.GetDeviceId())
          << ", nccl rank=" << nccl_rank_id_
          << ", total param count=" << params2rootid_.size()
          << ", remove op count=" << remove_ops_.size()
          << ", remove var count=" << remove_vars_.size()
          << ", unpersist var count=" << unpersist_vars_.size()
          << ", persist var count=" << persist_param_vars_.size()
          << ", dump param count=" << shard_dump_params_.size()
          << ", dump fields count=" << shard_dump_fields_.size()
          << ", offload var name count=" << offload_names_.size()
          << ", total_broadcast=" << total_broadcast
          << ", remove_broadcast=" << remove_broadcast
          << ", remove c_sync_calc_stream=" << remove_sync_stream
          << ", remove cast_op=" << remove_cast_op;
}
size_t HogwildWorker::AdjustOffloadOps(const ProgramDesc &program) {
  // offload
  size_t offload_cnt = 0;
  if (offload_names_.empty()) {
    return offload_cnt;
  }
  // offload adam
  std::multiset<std::string> param2refs;
  for (size_t op_id = 0; op_id < ops_.size(); ++op_id) {
    auto &op = ops_[op_id];
    if (op->Type() == "c_broadcast") {
      continue;
    }
    // offload
    int cnt = 0;
    bool is_first = false;
    for (auto &o : op->Inputs()) {
      for (auto &name : o.second) {
        if (offload_names_.find(name) == offload_names_.end()) {
          continue;
        }
        auto dest_var = thread_scope_->Var(name);  // init local var
        PADDLE_ENFORCE_NE(dest_var,
                          nullptr,
                          common::errors::InvalidArgument(
                              "init var error name=%s", name.c_str()));
        offload_vars_[op.get()].copy_vars.push_back(name);
        // nccl broadcast param
        if (is_offload_communication_) {
          if (param2refs.find(name) == param2refs.end()) {
            param2refs.insert(name);
            is_first = true;
          }
        }
        ++cnt;
      }
    }
    offload_cnt += cnt;
    if (cnt > 0) {
      int op_role = op->Attr<int>("op_role");
      auto &op_offload = offload_vars_[op.get()];
      // add gc
      auto it = unused_vars_.find(op.get());
      if (it != unused_vars_.end()) {
        for (auto &name : op_offload.copy_vars) {
          if (std::find(it->second.begin(), it->second.end(), name) !=
              it->second.end()) {
            continue;
          }
          it->second.push_back(name);
        }
      } else {
        unused_vars_.insert(std::make_pair(op.get(), op_offload.copy_vars));
      }

      if (is_first) {
        // first used single node used p2p copy, multi node used nccl broadcast
        if (is_multi_node_) {
          op_offload.backup_vars = std::move(op_offload.copy_vars);
          op_offload.copy_vars.clear();
        }
      } else {
        // offload adam need backup param to pinned memory
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          for (auto &name : op_offload.copy_vars) {
            auto it = params2rootid_.find(name);
            if (it != params2rootid_.end() && it->second != nccl_rank_id_) {
              continue;
            }
            // only copy adam status
            op_offload.backup_vars.push_back(name);
          }
        }
      }
    }
  }
  // if not need gather
  if (FLAGS_gpugraph_offload_gather_copy_maxsize <= 0) {
    return offload_cnt;
  }
  // gather copy inputs
  const int64_t max_gather_len =
      FLAGS_gpugraph_offload_gather_copy_maxsize * 1024 * 1024;
  std::vector<const OperatorBase *> recycle_ops;
  std::multimap<std::string, int> name2refs;
  auto &block = program.Block(0);
  // get param length
  auto get_length_func = [&block](const std::vector<std::string> &vars,
                                  std::vector<std::string> *out_vars) {
    int64_t total_len = 0;
    for (auto &name : vars) {
      if (out_vars != nullptr) {
        auto it = std::find(out_vars->begin(), out_vars->end(), name);
        if (it != out_vars->end()) {
          continue;
        }
        out_vars->push_back(name);
      }
      auto desc = block.FindVar(name);
      int64_t len = 1;
      for (auto &num : desc->GetShape()) {
        len = len * num;
      }
      total_len += len;
    }
    return total_len;
  };
  // check vars gc
  auto add_gc_refs_func = [this, &name2refs](const OperatorBase *op) {
    auto it = unused_vars_.find(op);
    if (it == unused_vars_.end()) {
      return;
    }
    for (auto &name : it->second) {
      if (offload_names_.find(name) == offload_names_.end()) {
        continue;
      }
      auto itx = name2refs.find(name);
      if (itx == name2refs.end()) {
        name2refs.insert(std::make_pair(name, 1));
      } else {
        ++itx->second;
      }
    }
  };
  auto remove_gc_vars_func = [this, &name2refs](const size_t &start_idx,
                                                const size_t &end_idx) {
    for (size_t op_idx = start_idx; op_idx < end_idx; ++op_idx) {
      auto &op = ops_[op_idx];
      auto it = unused_vars_.find(op.get());
      if (it == unused_vars_.end()) {
        continue;
      }
      std::vector<std::string> new_vars;
      for (auto &name : it->second) {
        auto itx = name2refs.find(name);
        if (itx == name2refs.end()) {
          new_vars.push_back(name);
          continue;
        }
        if (--itx->second == 0) {
          new_vars.push_back(name);
        }
      }
      it->second = new_vars;
    }
  };

  size_t op_idx = 0;
  if (is_multi_node_) {
    while (op_idx < ops_.size()) {
      int op_role = ops_[op_idx]->Attr<int>("op_role");
      if (op_role == static_cast<int>(OpRole::kBackward)) {
        break;
      }
      ++op_idx;
    }
  }
  size_t start_op_idx = 0;
  int64_t total_len = 0;
  std::vector<std::string> *out_vars = nullptr;
  while (op_idx < ops_.size()) {
    auto op = ops_[op_idx].get();
    auto it = offload_vars_.find(op);
    if (it == offload_vars_.end()) {
      if (out_vars != nullptr) {
        add_gc_refs_func(op);
      }
      ++op_idx;
      continue;
    }
    // add self length
    if (out_vars == nullptr) {
      start_op_idx = op_idx;
      total_len = get_length_func(it->second.copy_vars, nullptr);
      out_vars = &it->second.copy_vars;
    } else {
      total_len += get_length_func(it->second.copy_vars, out_vars);
      it->second.copy_vars.clear();
      if (it->second.copy_vars.empty() && it->second.backup_vars.empty() &&
          it->second.cast_vars.empty()) {
        recycle_ops.push_back(it->first);
      }
    }
    add_gc_refs_func(op);
    // max length reset
    if (total_len > max_gather_len) {
      out_vars = nullptr;
      // remove gc vars names
      remove_gc_vars_func(start_op_idx, op_idx + 1);
    }
    ++op_idx;
  }
  // remove last gc vars names
  if (out_vars != nullptr && start_op_idx < op_idx) {
    remove_gc_vars_func(start_op_idx, op_idx);
  }
  // erase empty offload ops
  for (auto &op : recycle_ops) {
    offload_vars_.erase(op);
  }
  VLOG(0) << "device id=" << thread_id_
          << ", gather offload ops size=" << offload_vars_.size()
          << ", recycle size=" << recycle_ops.size();

  return offload_cnt;
}
void HogwildWorker::CreateThreadOperators(const ProgramDesc &program) {
  auto &block = program.Block(0);
  op_names_.clear();
  auto all_desc = block.AllOps();
  std::set<size_t> remove_ids;
  size_t op_index = 0;
  for (auto &op_desc : all_desc) {
    // skip feed fetch op
    std::string op_name = op_desc->Type();
    if (op_name == "feed" || op_name == "fetch") {
      for (auto &o : op_desc->Inputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
      for (auto &o : op_desc->Outputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
    }
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op_desc->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (need_skip) {
      continue;
    }
    // skip remove ops, remove sync
    if (remove_ops_.find(op_desc) != remove_ops_.end() ||
        op_name == "c_sync_comm_stream") {
      if (enable_adjust_op_order_) {
        remove_ids.insert(op_index);
      } else {
        continue;
      }
    }
    op_names_.push_back(op_name);
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
    // change to device stream
    if (op_name == "c_broadcast" || op_name == "c_allreduce_sum" ||
        (op_name == "reduce" &&
         op_desc->GetAttrIfExists<int>("reduce_type") ==
             static_cast<int>(phi::ReduceType::kRedSum))) {
      ops_[op_index]->SetAttr("use_calc_stream", true);
    }
    op_index++;
  }
  if (enable_adjust_op_order_) {
    std::vector<size_t> new_order;
    size_t start_index = 0;
    for (auto &op : ops_) {
      int op_role = op->Attr<int>("op_role");
      if ((op_role == static_cast<int>(OpRole::kForward)) ||
          (op_role == (static_cast<int>(OpRole::kForward) |
                       static_cast<int>(OpRole::kLoss))) ||
          (op_role == static_cast<int>(OpRole::kLRSched))) {
        start_index++;
      } else {
        break;
      }
    }

    if (start_index < ops_.size()) {
      platform::Timer tm;
      tm.Start();
      interpreter::DependencyBuilderSimplify depend_builder;
      // depend_builder.Build(ops_, start_index, sharding_mode_);  hbm not safe
      // should run in debug model need to fix
      depend_builder.Build(ops_, start_index, false);
      new_order = depend_builder.get_new_executor_order();
      std::vector<std::unique_ptr<OperatorBase>> new_ops;
      std::vector<size_t> final_order;
      std::vector<std::string> new_op_names;
      for (auto &index : new_order) {
        if (remove_ids.count(index) == 0) {
          new_ops.push_back(std::move(ops_[index]));
          final_order.push_back(index);
          new_op_names.push_back(op_names_[index]);
        }
      }
      ops_ = std::move(new_ops);
      op_names_ = std::move(new_op_names);
      tm.Pause();
      // add log
      VLOG(0) << "device id=" << thread_id_
              << ", total op size=" << all_desc.size()
              << ", remove op size=" << remove_ids.size()
              << ", adjust op size=" << new_order.size()
              << ", final op size=" << final_order.size()
              << ", span time=" << tm.ElapsedSec() << "sec";
    }
  }
  operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
      program, 0, ops_);
  // not need gc
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (max_memory_size < 0) {
    return;
  }
  // skip dump fields
  if (need_dump_field_ && dump_fields_ != nullptr) {
    skip_vars_.insert(
        skip_vars_.end(), dump_fields_->begin(), dump_fields_->end());
  }
  // skip dump params
  if (need_dump_param_ && dump_param_ != nullptr) {
    skip_vars_.insert(
        skip_vars_.end(), dump_param_->begin(), dump_param_->end());
  }
  int fetch_var_num = fetch_config_.fetch_var_names_size();
  if (fetch_var_num > 0) {
    for (int i = 0; i < fetch_var_num; ++i) {
      std::string name = fetch_config_.fetch_var_names(i);
      skip_vars_.push_back(name);
    }
  }
  unused_vars_ =
      GetUnusedVars(block, ops_, skip_vars_, &unpersist_vars_, sharding_mode_);
  // adjust offload ops
  size_t offload_cnt = AdjustOffloadOps(program);
  // add cast ops
  size_t cast_cnt = 0;
  if (!need_cast_vars_.empty()) {
    for (size_t op_id = 0; op_id < ops_.size(); ++op_id) {
      auto &op = ops_[op_id];
      if (op->Type() != "c_broadcast") {
        continue;
      }
      for (auto &o : op->Inputs()) {
        for (auto &name : o.second) {
          auto it = need_cast_vars_.find(name);
          if (it == need_cast_vars_.end()) {
            continue;
          }
          ++cast_cnt;
          offload_vars_[op.get()].cast_vars.push_back(
              std::make_pair(it->first, it->second));
        }
      }
    }
  }
  // debug str
  if (FLAGS_enable_dump_main_program) {
    std::ostringstream str_os;
    for (auto &op : ops_) {
      str_os << op->DebugStringEx(thread_scope_);
      // add gc
      auto it = unused_vars_.find(op.get());
      if (it != unused_vars_.end()) {
        str_os << ", gc names: [";
        for (auto &name : it->second) {
          str_os << name << ",";
        }
        str_os << "]";
      }
      // add offload
      auto itx = offload_vars_.find(op.get());
      if (itx != offload_vars_.end()) {
        str_os << ", offload copies: [";
        for (auto &name : itx->second.copy_vars) {
          str_os << name << ",";
        }
        str_os << "], backups: [";
        for (auto &name : itx->second.backup_vars) {
          str_os << name << ",";
        }
        str_os << "]";
        if (!itx->second.cast_vars.empty()) {
          str_os << ", casts:[";
          for (auto &obj : itx->second.cast_vars) {
            str_os << obj.second << "->" << obj.first << ",";
          }
          str_os << "]";
        }
      }
      str_os << "\n";
    }
    std::string filename = "./device_";
    filename += std::to_string(thread_id_);
    filename += "_ops.txt";
    WriteToFile(filename.c_str(), str_os.str());
  }
  // debug
  VLOG(0) << "device id=" << thread_id_
          << ", total op count=" << all_desc.size()
          << ", create op count=" << ops_.size()
          << ", skip vars count=" << skip_vars_.size()
          << ", unused vars op count=" << unused_vars_.size()
          << ", offload op count=" << offload_vars_.size()
          << ", offload input count=" << offload_cnt
          << ", cast count=" << cast_cnt;
}
inline void PrintTensor(const std::string &name,
                        const std::string &info,
                        Scope *scope) {
  std::stringstream ss;
  platform::PrintVar(scope, name, info, &ss);
  std::cout << ss.str() << std::endl;
}
bool HogwildWorker::IsNeedOffload(const std::string &name) {
  if (!is_offload_param_) {
    return false;
  }
  if (offload_exts_.empty()) {
    return false;
  }
  for (auto &ext : offload_exts_) {
    if (name.find(ext) == std::string::npos) {
      continue;
    }
    return true;
  }
  return false;
}
void HogwildWorker::CreateThreadScope(const ProgramDesc &program) {
  auto &block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_,
      common::errors::NotFound(
          "Root scope should be set before creating thread scope."));

  thread_scope_ = &root_scope_->NewScope();

  int persist_total = 0;
  int persist_param = 0;
  int persist_share = 0;
  int persist_reset = 0;
  int pinned_param = 0;
  int resize_var_cnt = 0;
  int fp16_param = 0;
  std::vector<std::string> del_var_names;
  for (auto &var : block.AllVars()) {
    auto name = var->Name();
    if (remove_vars_.find(name) != remove_vars_.end()) {
      if (free_param_vars_.find(name) != free_param_vars_.end()) {
        del_var_names.push_back(name);
        VLOG(1) << "remove need delete var name=" << name;
      }
      continue;
    }
    all_param_.push_back(name);
    if (var->Persistable()) {
      ++persist_total;
      if (stat_var_name_map_.find(name) != stat_var_name_map_.end()) {
        Variable *root_var = root_scope_->FindVar(name);
        PADDLE_ENFORCE_NOT_NULL(
            root_var,
            common::errors::NotFound("Root scope should contain variable."));

        auto root_tensor = root_var->Get<phi::DenseTensor>();
        if (root_tensor.place() == place_) {
          continue;
        }
        auto *ptr1 = thread_scope_->Var(name);
        InitializeVariable(ptr1, var->GetType());
        phi::DenseTensor *thread_tensor = ptr1->GetMutable<phi::DenseTensor>();
#define MemsetCallback(cpp_type, proto_type)                                 \
  do {                                                                       \
    if (framework::TransToProtoVarType(root_tensor.dtype()) == proto_type) { \
      SetZero<cpp_type>(thread_tensor, root_tensor);                         \
    }                                                                        \
  } while (0)
        _ForEachDataType_(MemsetCallback);
      }
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
      else if (unpersist_vars_.find(name) == unpersist_vars_.end()) {  // NOLINT
        if (use_gpu_graph_ && use_ps_gpu_) {
          Variable *root_var = root_scope_->FindVar(name);
          if (!root_var) {
            VLOG(0) << "not found var name=" << name;
            continue;
          }
          if (root_var->IsType<phi::SelectedRows>()) {
            continue;
          }
          ++persist_param;
          phi::DenseTensor *root_tensor =
              root_var->GetMutable<phi::DenseTensor>();
          auto var_dtype =
              phi::TransToPhiDataType(static_cast<int>(var->GetDataType()));
          if (root_tensor->dtype() != var_dtype) {
            phi::DenseTensor tmp_tensor;
            tmp_tensor.Resize(root_tensor->dims());
            tmp_tensor.set_layout(root_tensor->layout());
            tmp_tensor.mutable_data(root_tensor->place(), var_dtype);
            framework::TransDataType(
                *root_tensor, var->GetDataType(), &tmp_tensor);
            auto holder = tmp_tensor.MoveMemoryHolder();
            root_tensor->ResetHolderWithType(holder, var_dtype);
            ++fp16_param;
          }
          if (place_ == root_tensor->place()) {
            ++persist_share;
            continue;
          }
          // reset tensor holder
          if (persist_param_vars_.find(name) != persist_param_vars_.end()) {
            // need offload param
            auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
            if (IsNeedOffload(name)) {
              // add offload names
              offload_names_.insert(name);
              // offload moment
              Tensor2Pinned(root_tensor, stream);
              ++pinned_param;
            } else {
              // copy one device to other device
              auto src_place = root_tensor->place();
              auto holder = root_tensor->MoveMemoryHolder();
              auto dst_ptr = root_tensor->mutable_data(
                  place_, root_tensor->dtype(), holder->size());
              memory::Copy(place_,
                           dst_ptr,
                           src_place,
                           holder->ptr(),
                           holder->size(),
                           stream);
              PADDLE_ENFORCE_EQ(phi::is_gpu_place(root_tensor->place()),
                                true,
                                common::errors::InvalidArgument(
                                    "The place of root tensor should be GPU."));
              ++persist_reset;
            }
          } else {
            auto *ptr = thread_scope_->Var(name);
            PADDLE_ENFORCE_EQ(proto::VarType::DENSE_TENSOR,
                              var->GetType(),
                              common::errors::InvalidArgument(
                                  "The type of var should be DENSE_TENSOR."));
            InitializeVariable(ptr, var->GetType());
            phi::DenseTensor *thread_tensor =
                ptr->GetMutable<phi::DenseTensor>();
            TensorCopy(*root_tensor, place_, thread_tensor);
            need_copy_vars_.push_back(name);
            //          VLOG(0) << "need copy var name=" << name;
          }
        }
      } else {
        if (use_gpu_graph_ && use_ps_gpu_) {
          if (free_param_vars_.find(name) != free_param_vars_.end()) {
            del_var_names.push_back(name);
            //          VLOG(0) << "unpersist need delete var name=" << name;
          }
          auto it = param_cast_vars_.find(name);
          if (it == param_cast_vars_.end()) {
            // sharding vars
            auto *ptr = thread_scope_->Var(name);
            InitializeVariable(ptr, var->GetType());
            // set dims
            auto dims = phi::make_ddim(var->GetShape());
            auto var_dtype =
                phi::TransToPhiDataType(static_cast<int>(var->GetDataType()));
            ptr->GetMutable<phi::DenseTensor>()->Resize(dims).set_type(
                var_dtype);
          }
        }
      }
#endif
    } else {
      auto *ptr = thread_scope_->Var(name);
      InitializeVariable(ptr, var->GetType());
      // amp
      auto it = cast_fp16_vars_.find(name);
      if (it != cast_fp16_vars_.end()) {
        auto desc_var = block.FindVar(it->second);
        if (desc_var != nullptr && desc_var->IsParameter()) {
          auto dims = phi::make_ddim(desc_var->GetShape());
          auto var_dtype =
              phi::TransToPhiDataType(static_cast<int>(var->GetDataType()));
          ptr->GetMutable<phi::DenseTensor>()->Resize(dims).set_type(var_dtype);
          ++resize_var_cnt;
        }
      }
    }
  }
  // multi node delete unused vars
  if (!del_var_names.empty()) {
    root_scope_->EraseVars(del_var_names);
  }
  VLOG(0) << "device id=" << thread_id_
          << ", total param count=" << all_param_.size()
          << ", persist count=" << persist_total << ", param=" << persist_param
          << ", fp16=" << fp16_param << ", share=" << persist_share
          << ", reset=" << persist_reset << ", pinned=" << pinned_param
          << ", resize_var=" << resize_var_cnt
          << ", need copy param count=" << need_copy_vars_.size()
          << ", delete vars count=" << del_var_names.size();
}
void HogwildWorker::Finalize() {
#ifdef PADDLE_WITH_HETERPS
  if (!sharding_mode_ && thread_id_ != 0) {
    return;
  }
  for (auto &name : need_copy_vars_) {
    Variable *root_var = root_scope_->FindVar(name);
    if (root_var == nullptr) {
      continue;
    }
    auto root_tensor = root_var->GetMutable<phi::DenseTensor>();
    Variable *var = thread_scope_->FindVar(name);
    auto tensor = var->Get<phi::DenseTensor>();
    TensorCopy(tensor, root_tensor->place(), root_tensor);
  }
  dev_ctx_->Wait();
#endif
}
template <typename T>
void HogwildWorker::SetZero(phi::DenseTensor *tensor,
                            const phi::DenseTensor &root_tensor) {
  tensor->mutable_data<T>(root_tensor.dims(), place_);
  phi::funcs::set_constant(*dev_ctx_, tensor, 0.0);
}

void HogwildWorker::BindingDataFeedMemory() {
  const std::vector<std::string> &input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto const &name : input_feed) {
    device_reader_->AddFeedVar(thread_scope_->FindVar(name), name);
  }
}

void HogwildWorker::CreateDeviceResource(const ProgramDesc &main_prog) {
  BuildShardingDepends(main_prog);
  CreateThreadScope(main_prog);
  CreateThreadOperators(main_prog);

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  if (use_gpu_graph_ && use_ps_gpu_) {
    float *stat_ptr = sync_stat_.mutable_data<float>(place_, sizeof(float) * 3);
    float flags[] = {0.0, 1.0, 1.0};
    auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(stat_ptr,  // output
                                               &flags,
                                               sizeof(float) * 3,
                                               cudaMemcpyHostToDevice,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  }
#endif
}
// check batch num
bool HogwildWorker::CheckBatchNum(int flag) {
  float ret = 0.0;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  if (use_gpu_graph_ && use_ps_gpu_) {
    if (flag > 1) {
      flag = 1;
    } else if (flag < 0) {
      flag = 0;
    }
    //  g_barrier.wait();
    float *stat_ptr = sync_stat_.data<float>();
    int ring_id = 0;
    const auto &comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    phi::distributed::NCCLCommContext *comm_ctx = nullptr;
    PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                      true,
                      common::errors::InvalidArgument(
                          "You choose to use new communication library. "
                          "But ring_id(%d) is "
                          "not found in comm_context_manager.",
                          std::to_string(ring_id)));
    comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
        comm_context_manager.Get(std::to_string(ring_id)));
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
    // comm_ctx->AllReduce only support allreduce on the whole tensor,
    // single element is not supported now.
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::ncclAllReduce(&stat_ptr[flag],
                                    &stat_ptr[2],
                                    1,
                                    ncclFloat32,
                                    ncclProd,
                                    comm_ctx->GetNcclComm(),
                                    stream));

    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&ret,  // output
                                               &stat_ptr[2],
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    //  g_barrier.wait();
  }
#endif
  return (ret > 0.0);
}

bool HogwildWorker::GetPassEnd(int flag) {
  float ret = 0.0;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  if (use_gpu_graph_ && use_ps_gpu_) {
    if (flag > 1) {
      flag = 1;
    } else if (flag < 0) {
      flag = 0;
    }
    //  g_barrier.wait();
    float *stat_ptr = sync_stat_.data<float>();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id_,
                                                          place_.GetDeviceId());
    //  auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
    //  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    auto stream = comm->stream();
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(&stat_ptr[flag],
                                                           &stat_ptr[2],
                                                           1,
                                                           ncclFloat32,
                                                           ncclProd,
                                                           comm->comm(),
                                                           stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&ret,  // output
                                               &stat_ptr[2],
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost,
                                               stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    // g_barrier.wait();
  }
#endif
  return (ret > 0.0);
}

void HogwildWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
#if defined(PADDLE_WITH_HETERPS) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL))
  platform::SetDeviceId(thread_id_);
#elif defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_XPU_BKCL)
  platform::SetXPUDeviceId(thread_id_);
#endif
  device_reader_->Start();
  std::vector<double> op_total_time;
  op_total_time.resize(ops_.size());
  for (double &op_time : op_total_time) {
    op_time = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int cur_batch = 0;
  int batch_cnt = 0;
  if (thread_id_ == 0) {
    quit_flag_.store(false);
  }
  g_barrier.wait();

#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  bool train_mode = false;
  bool is_multi_node = false;
  if (use_gpu_graph_ && use_ps_gpu_) {
    train_mode = device_reader_->IsTrainMode();
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    if (gloo->Size() > 1) {
      is_multi_node = true;
    }
  }
#endif

  timeline.Start();
  uint64_t total_inst = 0;
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  if (use_gpu_graph_ && use_ps_gpu_) {
    device_reader_->InitGraphTrainResource();
  }
#endif

  std::unique_ptr<GarbageCollector> gc = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (max_memory_size >= 0) {
    gc = CreateGarbageCollector(place_, max_memory_size);
  }
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  std::unique_ptr<GPUParallelCopyer> copyer = nullptr;
  if (use_gpu_graph_ && use_ps_gpu_) {
    auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
    if (!offload_vars_.empty()) {
      copyer.reset(new GPUParallelCopyer(
          stream, thread_id_, FLAGS_gpugraph_parallel_stream_num));
    }
  }
#endif
  bool infer_out_of_ins = false;
  while (true) {
    cur_batch = device_reader_->Next();
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
    if (use_gpu_graph_ && use_ps_gpu_) {
      if (FLAGS_gpugraph_force_device_batch_num_equal) {
        if (!CheckBatchNum(cur_batch)) {
          break;
        }
      } else if (train_mode && is_multi_node) {
        int pass_end = device_reader_->get_pass_end();
        bool res = GetPassEnd(pass_end);
        VLOG(2) << "reader pass end: " << pass_end
                << ", hogwild worker pass end: " << res;
        if (res) {
          device_reader_->reset_pass_end();
          VLOG(1) << "get all pass end, train pass will exit";
          break;
        }
      } else if (!train_mode && sharding_mode_) {
        auto pass_end = cur_batch <= 0;
        bool all_pass_end = GetPassEnd(pass_end);
        if (all_pass_end) {
          break;
        }
        if (pass_end) {
          infer_out_of_ins = true;
          VLOG(0) << " card " << thread_id_ << " infer_out_of_ins ";
        }
      } else {
        if (FLAGS_enable_exit_when_partial_worker && train_mode) {
          if (cur_batch <= 0) {
            quit_flag_.store(true, std::memory_order_relaxed);
          }
          g_barrier.wait();
          if (quit_flag_.load(std::memory_order_relaxed) == true) {
            break;
          }
        }
      }
    }
#endif
    if (cur_batch <= 0 && !infer_out_of_ins) {
      break;
    }
    VLOG(3) << "read a batch in thread " << thread_id_;
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    if (infer_out_of_ins) {
      for (size_t i = 0; i < ops_.size(); ++i) {
        timeline.Start();
        auto &op = ops_[i];
        VLOG(3) << "Going to run op " << op_names_[i];
        if (op->Type() == "c_broadcast") {
          op->Run(*thread_scope_, place_);
        }
#ifdef PADDLE_WITH_HETERPS
        dev_ctx_->Wait();
#endif
        VLOG(3) << "Op " << op_names_[i] << " Finished";
        timeline.Pause();
        op_total_time[i] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
        if (gc) {
          DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
        }
      }
    } else {
      for (size_t i = 0; i < ops_.size(); ++i) {
        timeline.Start();
        auto &op = ops_[i];
        VLOG(3) << "Going to run op " << op_names_[i];
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
        auto it = offload_vars_.find(op.get());
        if (use_gpu_graph_ && use_ps_gpu_) {
          // offload
          if (it != offload_vars_.end()) {
            it->second.CopyInputs(
                root_scope_, place_, thread_scope_, copyer.get());
          }
        }
#endif
        op->Run(*thread_scope_, place_);
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
        if (use_gpu_graph_ && use_ps_gpu_) {
          if (it != offload_vars_.end()) {
            it->second.BackUpInputs(root_scope_, thread_scope_, copyer.get());
          }
        }
#endif
#ifdef PADDLE_WITH_HETERPS
        dev_ctx_->Wait();
#endif
        VLOG(3) << "Op " << op_names_[i] << " Finished";
        timeline.Pause();
        op_total_time[i] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
        if (gc) {
          DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
        }
      }
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && (sharding_mode_ || thread_id_ == 0)) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    total_inst += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
#ifdef PADDLE_WITH_HETERPS
    dev_ctx_->Wait();
    for (size_t i = 0; i < op_names_.size(); ++i) {
      VLOG(1) << "card:" << thread_id_ << ", op: " << op_names_[i]
              << ", mean time: " << op_total_time[i] / total_inst
              << "s, total time:" << op_total_time[i] << "sec";
    }
#else
    if (thread_id_ == 0) {
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < ops_.size(); ++i) {
          fprintf(stderr,
                  "op_name:[%zu][%s], op_mean_time:[%fs]\n",
                  i,
                  op_names_[i].c_str(),
                  op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(
            stderr, "%6.2f instances/s\n", total_inst / total_time);  // NOLINT
      }
    }
#endif
    if (gc) {
      gc->DirectClearCallback([this]() { thread_scope_->DropKids(); });
    } else {
      thread_scope_->DropKids();
    }
    timeline.Start();
  }
  VLOG(0) << "GpuPs worker " << thread_id_ << " train cost " << total_time
          << " seconds, ins_num: " << total_inst << " read time: " << read_time
          << "seconds ";

  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }

#if defined PADDLE_WITH_PSCORE
  if (thread_barrier_) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerDecrement();
  }
#endif
}
void HogwildWorker::TrainFiles() {
  platform::SetNumThreads(1);
  platform::Timer timeline;
  timeline.Start();
#if defined(PADDLE_WITH_HETERPS) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL))
  platform::SetDeviceId(thread_id_);
#elif defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_XPU_BKCL)
  platform::SetXPUDeviceId(thread_id_);
#endif

  int total_batch_num = 0;
  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch = 0;
  int batch_cnt = 0;
  if (thread_id_ == 0) {
    quit_flag_.store(false);
    // quit_flag_2 = false;
  }
  g_barrier.wait();

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_CUDA)
  platform::SetDeviceId(thread_id_);
#endif
  // while ((cur_batch = device_reader_->Next()) > 0) {
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  bool is_multi_node = false;
  bool train_mode = false;
  if (use_gpu_graph_ && use_ps_gpu_) {
    train_mode = device_reader_->IsTrainMode();
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    if (gloo->Size() > 1) {
      is_multi_node = true;
    }
  }
#endif
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
  if (use_gpu_graph_ && use_ps_gpu_) {
    device_reader_->InitGraphTrainResource();
  }
#endif

  std::unique_ptr<GarbageCollector> gc = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (max_memory_size >= 0) {
    gc = CreateGarbageCollector(place_, max_memory_size);
  }
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
  std::unique_ptr<GPUParallelCopyer> copyer = nullptr;
  if (use_gpu_graph_ && use_ps_gpu_) {
    auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
    if (!offload_vars_.empty()) {
      copyer.reset(new GPUParallelCopyer(
          stream, thread_id_, FLAGS_gpugraph_parallel_stream_num));
    }
  }
#endif
  bool infer_out_of_ins = false;
  while (true) {
    cur_batch = device_reader_->Next();
#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_PSCORE)
    if (use_gpu_graph_ && use_ps_gpu_) {
      if (FLAGS_gpugraph_force_device_batch_num_equal) {
        if (!CheckBatchNum(cur_batch)) {
          break;
        }
      } else if (train_mode && is_multi_node) {
        bool sage_mode = device_reader_->GetSageMode();
        if (!sage_mode) {
          int pass_end = device_reader_->get_pass_end();
          bool res = GetPassEnd(pass_end);
          VLOG(2) << "reader pass end: " << pass_end
                  << ", hogwild worker pass end: " << res;
          if (res) {
            device_reader_->reset_pass_end();
            VLOG(1) << "get all pass end, train pass will exit";
            break;
          }
        }
      } else if (!train_mode && sharding_mode_) {
        auto pass_end = cur_batch <= 0;
        bool res = GetPassEnd(pass_end);
        if (res) {
          break;
        }
        if (pass_end) {
          infer_out_of_ins = true;
          VLOG(0) << " card " << thread_id_ << " infer_out_of_ins ";
        }
      } else {
        if (FLAGS_enable_exit_when_partial_worker && train_mode) {
          if (cur_batch <= 0) {
            quit_flag_.store(true, std::memory_order_relaxed);
          }
          g_barrier.wait();
          if (quit_flag_.load(std::memory_order_relaxed) == true) {
            break;
          }
        }
      }
    }
#endif
    if (cur_batch <= 0 && !infer_out_of_ins) {
      break;
    }
    if (infer_out_of_ins) {
      for (auto &op : ops_) {
        if (op->Type() == "c_broadcast") {
          op->Run(*thread_scope_, place_);
        }
        if (gc) {
          DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
        }
      }
    } else {
      for (auto &op : ops_) {
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
        auto it = offload_vars_.find(op.get());
        if (use_gpu_graph_ && use_ps_gpu_) {
          // offload
          if (it != offload_vars_.end()) {
            it->second.CopyInputs(
                root_scope_, place_, thread_scope_, copyer.get());
          }
        }
#endif
        if (FLAGS_gpugraph_enable_print_op_debug) {
          VLOG(0) << "thread id=" << thread_id_ << ", "
                  << op->DebugStringEx(thread_scope_);
        }
        op->Run(*thread_scope_, place_);
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
        if (use_gpu_graph_ && use_ps_gpu_) {
          // offload
          if (it != offload_vars_.end()) {
            it->second.BackUpInputs(root_scope_, thread_scope_, copyer.get());
          }
        }
#endif
        if (gc) {
          DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
        }
      }
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && (sharding_mode_ || thread_id_ == 0)) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    // for (auto var_name: thread_scope_->LocalVarNames()) {
    // // for (std::string& var_name : check_nan_var_names_) {
    //   Variable* var = thread_scope_->FindVar(var_name);
    //   if (var == nullptr) {
    //     continue;
    //   }
    //   phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    //   if (tensor == nullptr || !tensor->IsInitialized()) {
    //     continue;
    //   }
    //   if (framework::TensorContainsInf(*tensor) ||
    //       framework::TensorContainsNAN(*tensor)) {
    //     static std::mutex mutex;
    //     {
    //       std::lock_guard<std::mutex> lock(mutex);
    //       VLOG(0) << "worker " << thread_id_ << ": " << var_name
    //               << " contains inf or nan";
    //       // auto all_vars = thread_scope_->LocalVarNames();
    //       std::stringstream ss;
    //       ss << "====== worker " << thread_id_ << "======\n";
    //       for (auto& local_var : thread_scope_->LocalVarNames()) {
    //         platform::PrintVar(thread_scope_, local_var, local_var, &ss);
    //         ss << "\n";
    //       }
    //       std::cout << ss.str() << std::endl;
    //       VLOG(0) << "worker " << thread_id_ << "print nan var done....";
    //     }
    //     sleep(600);
    //     exit(-1);
    //   }
    // }

    total_batch_num += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
    if (gc) {
      gc->DirectClearCallback([this]() { thread_scope_->DropKids(); });
    } else {
      thread_scope_->DropKids();
    }
  }
#ifdef PADDLE_WITH_HETERPS
  dev_ctx_->Wait();
#endif
  timeline.Pause();
  VLOG(1) << "worker " << thread_id_ << " train cost " << timeline.ElapsedSec()
          << " seconds, batch_num: " << total_batch_num;

  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }

#if defined PADDLE_WITH_PSCORE
  if (thread_barrier_) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerDecrement();
  }
#endif
}

void HogwildWorker::PrintFetchVars() {
  // call count
  batch_num_++;
  int batch_per_print = fetch_config_.print_period();
  int fetch_var_num = fetch_config_.fetch_var_names_size();

  if (fetch_var_num == 0) {
    return;
  }

  if (thread_id_ == 0 && batch_num_ % batch_per_print == 0) {
    time_t curtime = 0;
    time(&curtime);
    std::array<char, 80> mbstr;
    std::strftime(mbstr.data(),
                  sizeof(mbstr),
                  "%Y-%m-%d %H:%M:%S",
                  std::localtime(&curtime));

    std::stringstream ss;
    ss << "time: [" << mbstr.data() << "], ";
    ss << "batch: [" << batch_num_ << "], ";

    for (int i = 0; i < fetch_var_num; ++i) {
      platform::PrintVar(thread_scope_,
                         fetch_config_.fetch_var_names(i),
                         fetch_config_.fetch_var_str_format(i),
                         &ss);
      if (i < fetch_var_num - 1) {
        ss << ", ";
      }
    }

    std::cout << ss.str() << std::endl;
  }
}

}  // namespace paddle::framework
