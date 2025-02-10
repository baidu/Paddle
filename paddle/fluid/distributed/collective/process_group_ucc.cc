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

#include "paddle/fluid/distributed/collective/process_group_ucc.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/comm_task_manager.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/utils/data_type.h"

namespace paddle::distributed {

using phi::distributed::ToUCCRedType;
using phi::distributed::UCCDTypeToString;
using phi::distributed::UCCRedTypeToString;

ProcessGroupUCC::UCCTask::UCCTask(const Place& place,
                                  int rank,
                                  CommType comm_type,
                                  bool sync_op,
                                  bool use_calc_stream)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream),
      task_place_(place) {
  // if (!use_calc_stream) {
  //   comm_event_ = std::make_shared<platform::DeviceEvent>(
  //       place, platform::GenerateDeviceEventFlag());
  // }
}

ProcessGroupUCC::UCCTask::~UCCTask() = default;

bool ProcessGroupUCC::UCCTask::IsCompleted() { return task_->IsCompleted(); }

bool ProcessGroupUCC::UCCTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (is_gpu_place(task_place_)) {
    // if (UseCalcStream()) {
    //   VLOG(5) << "Warning: The communication is on calc stream, wait here is
    //   "
    //              "useless.";
    //   return true;
    // }

    // const auto* calc_ctx =
    //     platform::DeviceContextPool::Instance().Get(task_place_);
    // if (comm_event_) {
    //   comm_event_->Wait(platform::Place2DeviceType(task_place_), calc_ctx);
    // }
  } else {
    task_->Wait(timeout);
  }
  return true;
}

void ProcessGroupUCC::UCCTask::Synchronize() { Wait(kWaitTimeout); }

void ProcessGroupUCC::UCCTask::SetTask(std::shared_ptr<UCCCommTask> task) {
  task_ = task;
}

ProcessGroupUCC::ProcessGroupUCC(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid)
    : ProcessGroupWithStream(rank, size, gid), store_(store) {
  VLOG(3) << "ProcessGroupUCC::CreateProcessGroupUCC rank: " << rank;
}

std::shared_ptr<ProcessGroupUCC> ProcessGroupUCC::CreateProcessGroupUCC(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid) {
  auto process_group =
      std::make_shared<ProcessGroupUCC>(store, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

phi::DeviceContext* ProcessGroupUCC::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

phi::DeviceContext* ProcessGroupUCC::GetDeviceContext(
    const Place& place, bool use_calc_stream) const {
  const std::string& key = GetKeyFromPlace(place);
  if (use_calc_stream) {
    const auto& iter = place_to_calc_ctx_.find(key);
    return iter->second;
  } else {
    const auto& iter = place_to_comm_ctx_.find(key);
    PADDLE_ENFORCE_NE(
        iter,
        place_to_comm_ctx_.end(),
        common::errors::NotFound(
            "Cannot find the device context in this process group."));
    return iter->second.get();
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupUCC::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::UCCCommContext* comm_context,
          gpuStream_t stream) -> std::shared_ptr<UCCCommTask> {
        VLOG(3) << "ucc_reduce "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << UCCDTypeToString(phi::ToUCCDataType(in_tensor.dtype()))
                << ", redop: "
                << UCCRedTypeToString(ToUCCRedType(opts.reduce_op))
                << ", root: "
                << opts.root_rank
                // << ", ucc_comm: " << comm_context->GetUCCComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        return comm_context->Reduce(out_tensor,
                                    in_tensor,
                                    ToUCCRedType(opts.reduce_op),
                                    opts.root_rank,
                                    stream);
      },
      in_tensor,
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroupUCC::UCCTask> ProcessGroupUCC::CreateTask(
    const Place& place,
    int rank,
    CommType op_type,
    bool sync_op,
    bool use_calc_stream) {
  return std::make_shared<ProcessGroupUCC::UCCTask>(
      place, rank, op_type, sync_op, use_calc_stream);
}

void ProcessGroupUCC::CreateUCCEnvCache(const Place& place,
                                        const std::string& place_key) {
  VLOG(3) << "init ucc rank_in_group: " << rank_ << ", nranks: " << size_
          << ", place key: " << place_key;

  phi::distributed::CommContextManager::CreateUCCCommContext(
      store_, place_key, rank_, size_);
  if (is_gpu_place(place)) {
    auto comm_ctx = std::make_unique<phi::GPUContext>(place);
    auto* calc_ctx = static_cast<phi::GPUContext*>(
        phi::DeviceContextPool::Instance().Get(place));
    place_to_calc_event_.emplace(
        place_key,
        platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
    place_to_calc_ctx_.emplace(place_key, calc_ctx);
    place_to_comm_ctx_.emplace(place_key, std::move(comm_ctx));
  }
}

phi::distributed::UCCCommContext* ProcessGroupUCC::GetCommContext(
    const std::string* key) {
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::UCCCommContext*>(
      comm_context_manager.Get(*key));
  PADDLE_ENFORCE_NE(comm_context,
                    nullptr,
                    common::errors::Unavailable("UCCCommContext is nullptr"));
  return comm_context;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupUCC::Collective(
    std::function<std::shared_ptr<UCCCommTask>(
        phi::distributed::UCCCommContext*, gpuStream_t)> fn,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);

  const auto& place = tensor.place();
  const auto& key = GetKeyFromPlace(place);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateUCCEnvCache(place, key);
  }

  auto task = CreateTask(place, rank_, comm_type, sync_op, use_calc_stream);

  if (is_gpu_place(place)) {
    platform::CUDADeviceGuard cuda_guard(place);

    // if (!use_calc_stream) {
    //   SyncCalcStream(place, key);
    // }

    const auto* calc_ctx = place_to_calc_ctx_.at(key);
    const auto& comm_ctx = place_to_comm_ctx_.at(key);
    auto ucc_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

    auto ucc_comm_ctx = this->GetCommContext(&key);

    auto ucc_task = fn(ucc_comm_ctx, ucc_stream);
    task->SetTask(ucc_task);
    // if (!use_calc_stream) {
    //   if (!is_coalescing_) {
    //     if (FLAGS_use_stream_safe_cuda_allocator ||
    //         FLAGS_use_cuda_malloc_async_allocator) {
    //       memory::RecordStream(tensor.Holder(), ucc_stream);
    //     }
    //     task->UpdateWaitChain(*comm_ctx);
    //     allocation_stream_pairs_.emplace_back(tensor.Holder(), ucc_stream);
    //   } else {
    //     colaescing_tensors_.emplace_back(
    //         std::make_shared<phi::DenseTensor>(tensor));
    //     colaescing_place_keys_.push_back(key);
    //   }
    // }
  } else {
    auto ucc_task = fn(GetCommContext(&key), nullptr);
    task->SetTask(ucc_task);
    VLOG(3) << "task->SetTask finished!";
  }

  if (sync_op) {
    VLOG(3) << "task wait begin!";
    task->Wait();
    VLOG(3) << "task wait finished!";
  }

  return task;
}

}  // namespace paddle::distributed
