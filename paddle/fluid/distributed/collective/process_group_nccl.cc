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

#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/comm_task_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_task.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "paddle/phi/core/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/utils/data_type.h"

COMMON_DECLARE_bool(benchmark);
COMMON_DECLARE_bool(benchmark_nccl);
COMMON_DECLARE_bool(nccl_blocking_wait);
COMMON_DECLARE_bool(use_stream_safe_cuda_allocator);
COMMON_DECLARE_bool(use_cuda_malloc_async_allocator);
COMMON_DECLARE_bool(enable_async_trace);
COMMON_DECLARE_bool(eager_communication_connection);

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;
constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle::distributed {

using phi::distributed::CheckSizeOnEachRank;
using phi::distributed::IsP2POP;
using phi::distributed::NCCLDTypeToString;
using phi::distributed::NCCLRedTypeToString;
using phi::distributed::SerializeNCCLUniqueId;
using phi::distributed::ToNCCLRedType;

uint64_t ProcessGroupNCCL::s_group_call_counter = 0;

ProcessGroupNCCL::NCCLTask::NCCLTask(const Place& place,
                                     int rank,
                                     CommType comm_type,
                                     bool sync_op,
                                     bool use_calc_stream,
                                     int gid)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream),
      task_place_(place),
      gid_(gid) {
  if (!use_calc_stream) {
    comm_event_ = std::make_shared<platform::DeviceEvent>(
        place, platform::GenerateDeviceEventFlag());
  }
}

ProcessGroupNCCL::NCCLTask::~NCCLTask() = default;

bool ProcessGroupNCCL::NCCLTask::IsCompleted() {
  if (comm_event_) {
    return comm_event_->Query();
  } else {
    return true;
  }
}

void ProcessGroupNCCL::NCCLTask::UpdateWaitChain(
    const phi::DeviceContext& ctx) {
  if (comm_event_) {
    comm_event_->Record(&ctx);
  }
}

void ProcessGroupNCCL::NCCLTask::RemoveHolderStreamInGroup() {
  auto map = distributed::ProcessGroupMapFromGid::getInstance();
  distributed::ProcessGroup* pg = map->get(gid_);
  if (!pg) return;
  auto* pg_nccl = dynamic_cast<ProcessGroupNCCL*>(pg);
  if (!pg_nccl) return;
  pg_nccl->EraseTensorHolders();
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupNCCL::NCCLTask::Wait(std::chrono::milliseconds timeout) {
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(5) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

  const auto* calc_ctx =
      platform::DeviceContextPool::Instance().Get(task_place_);
  if (comm_event_) {
    comm_event_->Wait(platform::Place2DeviceType(task_place_), calc_ctx);
  }

  if (FLAGS_nccl_blocking_wait) {
    // NOTE(shenliang03): It will block host for sync
    while (!IsCompleted()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
    }
  }

  if (IsBlockCPUInWait()) {
    // If we use the work to do barrier, we should block cpu
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
  }
  RemoveHolderStreamInGroup();
  return true;
}

// Same as Wait
void ProcessGroupNCCL::NCCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid,
    int64_t timeout,
    int nccl_comm_init_option)
    : ProcessGroupWithStream(rank, size, gid),
      store_(store),
      place_to_calc_event_(),
      place_to_calc_ctx_(),
      place_to_comm_ctx_(),
      p2p_comm_seq_(),
      place_to_group_key_(),
      pg_timeout_(timeout),
      nccl_comm_init_option_(nccl_comm_init_option),
      allocation_stream_pairs_() {
  LOG(INFO) << "ProcessGroupNCCL pg_timeout_ " << pg_timeout_;
  LOG(INFO) << "ProcessGroupNCCL nccl_comm_init_option_ "
            << nccl_comm_init_option_;
  if (FLAGS_eager_communication_connection) {
    EagerConnect();
  }
}
ProcessGroupNCCL::~ProcessGroupNCCL() {
  LOG(INFO) << "ProcessGroupNCCL destruct ";
  if (FLAGS_enable_async_trace) {
    auto& comm_task_manager = phi::distributed::CommTaskManager::GetInstance();
    comm_task_manager.Stop();
  }
}

void ProcessGroupNCCL::GroupStart() {
  NCCL_CHECK(phi::dynload::ncclGroupStart());
  ++s_group_call_counter;
}

void ProcessGroupNCCL::GroupEnd() {
  NCCL_CHECK(phi::dynload::ncclGroupEnd());
  --s_group_call_counter;
  // NOTE: This is to sync the calc stream and comm stream for debug using
  // batch_isend_irecv
  if (FLAGS_benchmark || FLAGS_benchmark_nccl) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
  }
}

phi::DeviceContext* ProcessGroupNCCL::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

// NOTE(shenliang03): GetDeviceContext is only used for collective, it can't
// be used for p2p op.
phi::DeviceContext* ProcessGroupNCCL::GetDeviceContext(
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

ncclComm_t ProcessGroupNCCL::NCCLComm(const Place& place) const {
  const std::string& key = GetKeyFromPlace(place);
  const auto& iter = place_to_comm_ctx_.find(key);
  PADDLE_ENFORCE_NE(
      iter,
      place_to_comm_ctx_.end(),
      common::errors::NotFound(
          "Cannot find the NCCL communicator in this process group."));
  return iter->second->nccl_comm();
}

phi::distributed::NCCLCommContext* ProcessGroupNCCL::GetOrCreateCommContext(
    const Place& place, CommType comm_type) {
  const auto& key = GetKeyFromPlace(place);
  std::string store_key;
  GetStoreKey(key, comm_type, &store_key);
  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key, store_key, comm_type);
  }
  return GetCommContext(&store_key);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(in_tensor, offset, numel) : in_tensor;
  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        VLOG(3) << "[ncclAllGather] "
                << "sendbuff: " << in_tensor_maybe_partial.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor_maybe_partial.numel()
                << ", datatype: "
                << NCCLDTypeToString(
                       phi::ToNCCLDataType(in_tensor_maybe_partial.dtype()))
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", offset: " << offset
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->AllGather(out_tensor, in_tensor_maybe_partial, stream);
      },
      in_tensor_maybe_partial,
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        VLOG(3) << "[ncclAllReduce] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", redop: "
                << NCCLRedTypeToString(ToNCCLRedType(opts.reduce_op))
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        comm_context->AllReduce(
            out_tensor, in_tensor, ToNCCLRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

for (size_t i = 0; i < in_size_each_rank.size(); ++i) {
  std::cout << in_size_each_rank[i] << std::endl;
}

std::cout << "heihei" << std::endl;
for (size_t i = 0; i < out_size_each_rank.size(); ++i) {
  std::cout << out_size_each_rank[i] << std::endl;
}

  const phi::DDim& out_dim = out_tensor->dims();
  const phi::DDim& in_dim = in_tensor.dims();
  CheckSizeOnEachRank(out_dim, out_size_each_rank, size_);
  CheckSizeOnEachRank(in_dim, in_size_each_rank, size_);

  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor,
              in_tensor,
              in_size_each_rank,
              rank_,
              size_,
              comm_context->GetNcclComm());
        }

        int64_t in_row_size =
            in_dim[0] == 0 ? 0 : in_tensor.numel() / in_dim[0];
        int64_t out_row_size =
            out_dim[0] == 0 ? 0 : out_tensor->numel() / out_dim[0];
        int64_t in_offset = 0, in_numel = 0, out_offset = 0, out_numel = 0;
        phi::DenseTensor input_partial, output_partial;

        VLOG(3) << "[AllToAll] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", out_size_each_rank: "
                << string::join_strings(out_size_each_rank, ',')
                << ", in_size_each_rank: "
                << string::join_strings(in_size_each_rank, ',')
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        GroupStart();
        for (auto i = 0; i < size_; i++) {
          in_numel = in_size_each_rank[i] * in_row_size;

          if (in_numel > 0) {
            input_partial = GetPartialTensor(in_tensor, in_offset, in_numel);
            comm_context->Send(input_partial, in_numel, i, stream);
          }
          in_offset += in_numel;
          out_numel = out_size_each_rank[i] * out_row_size;
          if (out_numel > 0) {
            output_partial =
                GetPartialTensor(*out_tensor, out_offset, out_numel);
            comm_context->Recv(&output_partial, out_numel, i, stream);
          }
          out_offset += out_numel;

          std::cout << "i" << i << std::endl;
          std::cout << "in_numel: " << in_numel << std::endl;
          std::cout << "out_numel: " << out_numel << std::endl;
        }
        GroupEnd();
      },
      in_tensor,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    common::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  phi::GPUPlace place(opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  VLOG(3) << "[Barrier] "
          << "barrier opt: " << opts.device_id;

  auto task = AllReduce(&barrier_tensor,
                        barrier_tensor,
                        {},
                        /*sync_op*/ true,
                        /*use_calc_stream*/ false);
  auto nccl_task = dynamic_cast<NCCLTask*>(task.get());
  nccl_task->SetBlockCPUInWait();
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        int root = opts.source_rank + opts.source_root;

        VLOG(3) << "[ncclBroadcast] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", root: " << root
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->Broadcast(out_tensor, in_tensor, root, stream);
      },
      in_tensor,
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        VLOG(3) << "[ncclReduce] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", redop: "
                << NCCLRedTypeToString(ToNCCLRedType(opts.reduce_op))
                << ", root: " << opts.root_rank
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->Reduce(out_tensor,
                             in_tensor,
                             ToNCCLRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
      },
      in_tensor,
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        VLOG(3) << "[ncclReduceScatter] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", redop: "
                << NCCLRedTypeToString(ToNCCLRedType(opts.reduce_op))
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();
        comm_context->ReduceScatter(
            out_tensor, in_tensor, ToNCCLRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Scatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ opts.root_rank,
      /*cur_rank*/ rank_,
      size_);
  return Collective(
      [&](phi::distributed::NCCLCommContext* comm_context, gpuStream_t stream) {
        if (FLAGS_enable_nccl_dynamic_check) {
          phi::distributed::NCCLDynamicCheck::CheckShape(
              *out_tensor,
              /*root_rank*/ opts.root_rank,
              rank_,
              comm_context->GetNcclComm());
        }

        VLOG(3) << "[Scatter] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
                << ", root: " << opts.root_rank
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        int64_t numel = in_tensor.numel() / size_;
        if (rank_ == opts.root_rank) {
          int64_t offset = 0;
          phi::DenseTensor partial_tensor;
          GroupStart();
          for (auto i = 0; i < size_; i++) {
            partial_tensor = GetPartialTensor(in_tensor, offset, numel);
            comm_context->Send(partial_tensor, numel, i, stream);
            offset += numel;
          }
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
          GroupEnd();
        } else {
          comm_context->Recv(out_tensor, numel, opts.root_rank, stream);
        }
      },
      in_tensor,
      CommType::SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Gather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  std::vector<phi::DenseTensor> partial_tensors;
  if (rank_ == opts.root_rank) {
    partial_tensors.reserve(size_);
    size_t offset = 0;
    size_t numel = out_tensor->numel() / size_;
    for (auto i = 0; i < size_; i++) {
      partial_tensors.push_back(GetPartialTensor(*out_tensor,
                                                 static_cast<int64_t>(offset),
                                                 static_cast<int64_t>(numel)));
      offset += numel;
    }
  }
  return Gather(&partial_tensors, in_tensor, opts, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Gather(
    std::vector<phi::DenseTensor>* gather_tensors_ptr,
    const phi::DenseTensor& in_tensor,
    const GatherOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*gather_tensors_ptr);

  auto& gather_tensors = *gather_tensors_ptr;
  PADDLE_ENFORCE_GT(size_,
                    opts.root_rank,
                    common::errors::InvalidArgument(
                        "root world size [%d]  is less than root rank [%d]",
                        size_,
                        opts.root_rank));
  auto gather_func = [&](phi::distributed::NCCLCommContext* comm_context,
                         gpuStream_t stream) {
    // shape check
    if (FLAGS_enable_nccl_dynamic_check) {
      phi::distributed::NCCLDynamicCheck::CheckGatherShape(
          in_tensor,
          gather_tensors,
          opts.root_rank,
          rank_,
          size_,
          comm_context->GetNcclComm());
    }

    VLOG(3) << "[Gather] "
            << "sendbuff: " << in_tensor.data()
            << ", count: " << in_tensor.numel() << ", datatype: "
            << NCCLDTypeToString(phi::ToNCCLDataType(in_tensor.dtype()))
            << ", root: " << opts.root_rank
            << ", ncclcomm: " << comm_context->GetNcclComm()
            << ", stream: " << stream << ", rank_in_group: " << rank_
            << ", nranks: " << size_ << ", sync_op: " << sync_op
            << ", use_calc_stream: " << use_calc_stream << ", "
            << ", " << GetGroupMessage();

    GroupStart();
    // root receive from all devices
    if (rank_ == opts.root_rank) {
      for (auto i = 0; i < size_; i++) {
        auto& gather_tensor = gather_tensors[i];
        comm_context->Recv(&gather_tensor, gather_tensor.numel(), i, stream);
      }
    }
    // send to root
    comm_context->Send(in_tensor, in_tensor.numel(), opts.root_rank, stream);
    GroupEnd();
  };
  return Collective(
      gather_func, in_tensor, CommType::GATHER, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(*tensor);
  // numel > 0 indicates the tensor need to be sliced
  phi::DenseTensor partial_tensor;
  if (numel > 0) {
    partial_tensor = GetPartialTensor(*tensor, offset, numel);
    tensor = &partial_tensor;
  }

  return Point2Point(
      [&](phi::distributed::NCCLCommContext* comm_context,
          gpuStream_t stream,
          int rank_in_group) {
        VLOG(3) << "[ncclRecv] "
                << "recvbuff: " << tensor->data()
                << ", count: " << tensor->numel() << ", datatype: "
                << NCCLDTypeToString(phi::ToNCCLDataType(tensor->dtype()))
                << ", src_in_group: " << src_rank
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream
                << ", rank_in_group: " << rank_in_group << ", nranks: " << size_
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        comm_context->Recv(tensor, tensor->numel(), rank_in_group, stream);
      },
      src_rank,
      *tensor,
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor, offset, numel) : tensor;

  return Point2Point(
      [&](phi::distributed::NCCLCommContext* comm_context,
          gpuStream_t stream,
          int rank_in_group) {
        VLOG(3) << "[ncclSend] "
                << "sendbuff: " << tensor_maybe_partial.data()
                << ", count: " << tensor_maybe_partial.numel() << ", datatype: "
                << NCCLDTypeToString(
                       phi::ToNCCLDataType(tensor_maybe_partial.dtype()))
                << ", dst_in_group: " << dst_rank
                << ", ncclcomm: " << comm_context->GetNcclComm()
                << ", stream: " << stream
                << ", rank_in_group: " << rank_in_group << ", nranks: " << size_
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream << ", "
                << GetGroupMessage();

        comm_context->Send(tensor_maybe_partial,
                           tensor_maybe_partial.numel(),
                           rank_in_group,
                           stream);
      },
      dst_rank,
      tensor_maybe_partial,
      CommType::SEND,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroupNCCL::NCCLTask> ProcessGroupNCCL::CreateTask(
    const Place& place,
    int rank,
    CommType comm_type,
    bool is_sync,
    bool use_calc_stream,
    int gid) {
  return std::make_shared<ProcessGroupNCCL::NCCLTask>(
      place, rank, comm_type, is_sync, use_calc_stream, gid);
}

void ProcessGroupNCCL::GetStoreKey(const std::string& place_key,
                                   CommType comm_type,
                                   std::string* store_key) {
  bool is_batch_p2p = s_group_call_counter > 0;
  bool is_p2p_op = IsP2POP(comm_type, is_batch_p2p);

  if (!is_p2p_op) {
    *store_key = "nccl_ids/" + std::to_string(gid_) + "/0";
  } else {
    *store_key = "nccl_ids/" + std::to_string(gid_) + "/" + place_key;
  }
  place_to_group_key_[place_key] = *store_key;
}

void ProcessGroupNCCL::CreateNCCLEnvCache(const Place& place,
                                          const std::string& place_key,
                                          const std::string& store_key,
                                          CommType comm_type,
                                          int p2p_rank) {
  VLOG(3) << "init nccl rank_in_group: " << rank_ << ", nranks: " << size_
          << ", gid: " << gid_ << ", place key: " << place_key
          << ", store_key: " << store_key;

  for (size_t i = 0; i < s_group_call_counter; ++i) {
    NCCL_CHECK(phi::dynload::ncclGroupEnd());
  }

  bool is_batch_p2p = s_group_call_counter > 0;
  bool is_p2p_op = IsP2POP(comm_type, is_batch_p2p);

  int num_ranks = is_p2p_op ? 2 : GetSize();
  int rank = is_p2p_op ? p2p_rank : GetRank();

  NCCL_CHECK(phi::dynload::ncclGroupStart());

  phi::distributed::P2POption p2p_opts({is_p2p_op, p2p_rank, num_ranks, rank});
  phi::distributed::CommContextManager::CreateNCCLCommContext(
      store_, store_key, rank_, size_, "", &p2p_opts, nccl_comm_init_option_);

  NCCL_CHECK(phi::dynload::ncclGroupEnd());

  auto nccl_comm_ctx = this->GetCommContext(&store_key);
  VLOG(3) << "Get nccl comm: " << nccl_comm_ctx->GetNcclComm()
          << " for place_key: " << place_key << " on rank_in_group: " << rank
          << " nranks: " << num_ranks << " gid: " << gid_;

  auto comm_ctx = std::make_unique<phi::GPUContext>(place);
  comm_ctx->set_nccl_comm(nccl_comm_ctx->GetNcclComm());

  if (FLAGS_enable_async_trace) {
    // gather global ranks in current group
    size_t gpu_global_rank_size = sizeof(int);
    auto gpu_global_rank = phi::memory_utils::Alloc(
        phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId()),
        gpu_global_rank_size);

    phi::memory_utils::Copy(phi::GPUPlace(),
                            gpu_global_rank->ptr(),
                            phi::CPUPlace(),
                            &global_rank_,
                            gpu_global_rank_size);

    size_t gpu_global_ranks_size = num_ranks * sizeof(int);
    auto gpu_global_ranks = phi::memory_utils::Alloc(
        phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId()),
        gpu_global_ranks_size);

    NCCL_CHECK(phi::dynload::ncclAllGather(gpu_global_rank->ptr(),
                                           gpu_global_ranks->ptr(),
                                           1,
                                           ncclInt,
                                           nccl_comm_ctx->GetNcclComm(),
                                           comm_ctx->stream()));

    std::vector<int> global_ranks(num_ranks);
    phi::memory_utils::Copy(phi::CPUPlace(),
                            global_ranks.data(),
                            phi::GPUPlace(),
                            gpu_global_ranks->ptr(),
                            gpu_global_ranks_size);

    // store global_ranks in current group_key
    std::once_flag flag;
    std::call_once(flag, [this]() {
      phi::distributed::CommContextManager::GetInstance().SetStore(store_);
      phi::distributed::CommTaskManager::GetInstance().SetTimeout(pg_timeout_);
    });

    std::string group_key = place_to_group_key_.at(place_key);
    phi::distributed::CommContextManager::GetInstance().AddGroupRanks(
        group_key, global_ranks);
  }

  auto* calc_ctx = static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));

  place_to_calc_event_.emplace(
      place_key,
      platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
  place_to_calc_ctx_.emplace(place_key, calc_ctx);
  place_to_comm_ctx_.emplace(place_key, std::move(comm_ctx));

  for (size_t i = 0; i < s_group_call_counter; ++i) {
    NCCL_CHECK(phi::dynload::ncclGroupStart());
  }
}

void ProcessGroupNCCL::SyncCalcStream(const Place& place,
                                      const std::string& place_key) {
  auto& calc_event = place_to_calc_event_.at(place_key);
  const auto* calc_ctx = place_to_calc_ctx_.at(place_key);
  const auto* comm_ctx = place_to_comm_ctx_.at(place_key).get();
  calc_event.Record(calc_ctx);
  calc_event.Wait(platform::Place2DeviceType(place), comm_ctx);
}

void ProcessGroupNCCL::EagerConnect() {
  const auto deviceId = phi::backends::gpu::GetCurrentDeviceId();
  const auto& place = phi::GPUPlace(deviceId);
  const auto key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);
  std::string store_key;
  GetStoreKey(key, CommType::ALLREDUCE, &store_key);

  auto it = place_to_comm_ctx_.find(key);
  if (it == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key, store_key, CommType::ALLREDUCE);
  }
}

void ProcessGroupNCCL::EagerConnectRingExchange() {
  std::vector<std::pair<int, int>> peers;
  const auto& place = phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId());

  for (int rank = 0; rank < size_; rank++) {
    auto peer_rank = rank + 1 >= size_ ? 0 : rank + 1;
    peers.push_back(std::make_pair(rank, peer_rank));
  }

  for (auto& peer : peers) {
    int f_rank = peer.first;
    int s_rank = peer.second;

    int peer_rank = 0;
    int cur_rank = rank_;
    if (rank_ == f_rank) {
      peer_rank = s_rank;
    } else if (rank_ == s_rank) {
      peer_rank = f_rank;
    } else {
      continue;
    }

    int low_rank = cur_rank < peer_rank ? cur_rank : peer_rank;
    int high_rank = cur_rank < peer_rank ? peer_rank : cur_rank;
    std::string key =
        std::to_string(low_rank) + "->" + std::to_string(high_rank);

    auto p2p_rank = rank_ < peer_rank ? 0 : 1;
    platform::CUDADeviceGuard cuda_guard(place);
    std::string store_key;
    GetStoreKey(key, CommType::SEND, &store_key);
    if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
      CreateNCCLEnvCache(place, key, store_key, CommType::SEND, p2p_rank);
    }
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Collective(
    std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t)> fn,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);

  comm_seq_++;
  const auto& place = tensor.place();
  const auto& key = GetKeyFromPlace(place);

  platform::CUDADeviceGuard cuda_guard(place);

  std::string store_key;
  GetStoreKey(key, comm_type, &store_key);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key, store_key, comm_type);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place, key);
  }

  auto task =
      CreateTask(place, rank_, comm_type, sync_op, use_calc_stream, gid_);

  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

  auto nccl_comm_ctx = this->GetCommContext(&store_key);

  if (!FLAGS_enable_async_trace) {
    fn(nccl_comm_ctx, nccl_stream);
  } else {
    std::string group_key = place_to_group_key_.at(key);
    auto comm_task =
        std::make_shared<phi::distributed::NCCLCommTask>(place,
                                                         group_key,
                                                         rank_,
                                                         size_,
                                                         gid_,
                                                         comm_seq_,
                                                         tensor.numel(),
                                                         sync_op,
                                                         use_calc_stream,
                                                         nccl_comm,
                                                         nccl_stream,
                                                         comm_type,
                                                         pg_timeout_);
    comm_task->StartRecord();
    fn(nccl_comm_ctx, nccl_stream);
    comm_task->EndRecord();
    comm_task->SetStore(store_);

    auto& comm_task_manager = phi::distributed::CommTaskManager::GetInstance();
    comm_task_manager.CommTaskEnqueue(std::move(comm_task));
  }

  if (!use_calc_stream) {
    if (!is_coalescing_) {
      if (FLAGS_use_stream_safe_cuda_allocator ||
          FLAGS_use_cuda_malloc_async_allocator) {
        memory::RecordStream(tensor.Holder(), nccl_stream);
      }
      task->UpdateWaitChain(*comm_ctx);
      allocation_stream_pairs_.emplace_back(tensor.Holder(), nccl_stream);
    } else {
      colaescing_tensors_.emplace_back(
          std::make_shared<phi::DenseTensor>(tensor));
      colaescing_place_keys_.push_back(key);
    }
  }

  if (FLAGS_enable_nccl_dynamic_check) {
    task->SetBlockCPUInWait();
    task->Wait();
  }

  if (sync_op) {
    task->Wait();
  }

  if (FLAGS_benchmark || FLAGS_benchmark_nccl) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
  }

  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupNCCL::Point2Point(
    std::function<void(phi::distributed::NCCLCommContext*, gpuStream_t, int)>
        fn,
    int peer,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);

  const auto& place = tensor.place();

  int p2p_rank = 0;
  int p2p_target_rank = 0;
  bool is_batch_p2p = s_group_call_counter > 0;
  std::string key = "";

  int p2p_nrank = 0;
  if (is_batch_p2p) {
    key = GetKeyFromPlace(place);
    p2p_rank = rank_;
    p2p_target_rank = peer;
    p2p_nrank = GetSize();
  } else {
    int low_rank = rank_ < peer ? rank_ : peer;
    int high_rank = rank_ < peer ? peer : rank_;
    key = std::to_string(low_rank) + "->" + std::to_string(high_rank);
    p2p_rank = rank_ < peer ? 0 : 1;
    p2p_target_rank = 1 - p2p_rank;
    p2p_nrank = 2;
  }

  platform::CUDADeviceGuard cuda_guard(place);

  std::string store_key;
  GetStoreKey(key, comm_type, &store_key);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateNCCLEnvCache(place, key, store_key, comm_type, p2p_rank);
  }
  if (p2p_comm_seq_.find(key) == p2p_comm_seq_.end()) {
    p2p_comm_seq_[key] = 0;
  }
  p2p_comm_seq_[key]++;

  if (!use_calc_stream) {
    SyncCalcStream(place, key);
  }

  auto task =
      CreateTask(place, rank_, comm_type, sync_op, use_calc_stream, gid_);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);

  auto nccl_comm = comm_ctx->nccl_comm();
  auto nccl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

  std::string group_key = place_to_group_key_.at(key);
  auto comm_task =
      std::make_shared<phi::distributed::NCCLCommTask>(place,
                                                       group_key,
                                                       p2p_rank,
                                                       p2p_nrank,
                                                       gid_,
                                                       p2p_comm_seq_[key],
                                                       tensor.numel(),
                                                       sync_op,
                                                       use_calc_stream,
                                                       nccl_comm,
                                                       nccl_stream,
                                                       comm_type,
                                                       pg_timeout_);

  auto nccl_comm_ctx = this->GetCommContext(&store_key);

  if (!FLAGS_enable_async_trace) {
    fn(nccl_comm_ctx, nccl_stream, p2p_target_rank);
  } else {
    comm_task->StartRecord();
    fn(nccl_comm_ctx, nccl_stream, p2p_target_rank);
    comm_task->EndRecord();
    comm_task->SetStore(store_);

    auto& comm_task_manager = phi::distributed::CommTaskManager::GetInstance();
    comm_task_manager.CommTaskEnqueue(std::move(comm_task));
  }

  if (!use_calc_stream) {
    if (!is_coalescing_) {
      if (FLAGS_use_stream_safe_cuda_allocator ||
          FLAGS_use_cuda_malloc_async_allocator) {
        memory::RecordStream(tensor.Holder(), nccl_stream);
      }
      task->UpdateWaitChain(*comm_ctx);
      allocation_stream_pairs_.emplace_back(tensor.Holder(), nccl_stream);
    } else {
      colaescing_tensors_.emplace_back(
          std::make_shared<phi::DenseTensor>(tensor));
      colaescing_place_keys_.push_back(key);
    }
  }

  if (FLAGS_enable_nccl_dynamic_check) {
    task->SetBlockCPUInWait();
    task->Wait();
  }

  if (sync_op) {
    task->Wait();
  }

  if (!is_batch_p2p && (FLAGS_benchmark || FLAGS_benchmark_nccl)) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
  }

  return task;
}

std::shared_ptr<ProcessGroupNCCL> ProcessGroupNCCL::CreateProcessGroupNCCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid,
    int64_t timeout,
    int nccl_comm_init_option) {
  auto process_group = std::make_shared<ProcessGroupNCCL>(
      store, rank, size, gid, timeout, nccl_comm_init_option);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

phi::distributed::NCCLCommContext* ProcessGroupNCCL::GetCommContext(
    const std::string* key) {
  std::string store_key = std::to_string(this->gid_);
  if (key && !key->empty()) {
    store_key = *key;
  }
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::NCCLCommContext*>(
      comm_context_manager.Get(store_key));
  PADDLE_ENFORCE_NE(comm_context,
                    nullptr,
                    common::errors::Unavailable("NCCLCommContext is nullptr"));
  return comm_context;
}

void ProcessGroupNCCL::StartCoalescing() {
  PADDLE_ENFORCE_EQ(is_coalescing_,
                    false,
                    common::errors::PreconditionNotMet(
                        "Coalescing is on, please call EndCoalesce."));
  is_coalescing_ = true;
  GroupStart();
}

void ProcessGroupNCCL::EndCoalescing(
    std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>> tasks_opt) {
  GroupEnd();

  // NOTE(shenliang03): If using calculate stream, no need to record stream and
  // update task.
  if (!tasks_opt.has_value() || colaescing_tensors_.empty()) {
    is_coalescing_ = false;
    return;
  }

  auto& tasks = tasks_opt.value();

  PADDLE_ENFORCE_EQ(
      tasks.size(),
      colaescing_tensors_.size(),
      common::errors::PreconditionNotMet(
          "Number of tasks[%d] do not match number of collectives[%d].",
          tasks.size(),
          colaescing_tensors_.size()));

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* nccl_task = static_cast<ProcessGroupNCCL::NCCLTask*>(tasks[i].get());
    const auto& tensor = colaescing_tensors_[i];
    const auto& key = colaescing_place_keys_[i];
    const auto& comm_ctx = place_to_comm_ctx_.at(key);
    auto nccl_stream = comm_ctx->stream();

    if (FLAGS_use_stream_safe_cuda_allocator ||
        FLAGS_use_cuda_malloc_async_allocator) {
      memory::RecordStream(tensor->Holder(), nccl_stream);
    }

    nccl_task->UpdateWaitChain(*comm_ctx);
    allocation_stream_pairs_.emplace_back(tensor->Holder(), nccl_stream);
  }

  is_coalescing_ = false;
  colaescing_tensors_.clear();
  colaescing_place_keys_.clear();
}

}  // namespace paddle::distributed
